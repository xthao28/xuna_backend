"""UDC-9 cross-device composite workflow templates and board-button workflow trigger.

Implements:
- `S9.1`: `link_to_phone` (1. `browser.current_url` on PC -> 2. `url.open` on phone, with
  one approval showing both hosts and the domain only; full URL is never logged).
- `S9.2`: `continue_media` (1. `media.now_playing` on A -> 2. `media.play_pause` on A ->
  3. `app.launch` player on B -> 4. `media.play_pause` on B) and `pause_everywhere`
  (`media.play_pause` on all online hosts with `media` granted + board HID if connected).
- `S9.3`: Board-button workflow slots (1 & 2) and `trigger_board_workflow` enforcing
  `UDC-T15` (`R1` runs immediately, `R2` requires phone approval, `R3` cannot be assigned
  or triggered from board buttons).
"""

from __future__ import annotations

import ipaddress
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from urllib.parse import urlparse

from xuna_backend.modules.actions.application.catalog import (
    CAP_BOARD_HID_V1,
    is_tool_granted,
)
from xuna_backend.modules.actions.application.ports import (
    ApprovalBroker,
    ApprovalRepository,
    LocalResultBroker,
)
from xuna_backend.modules.actions.domain.entities import (
    APPROVAL_TTL_SECONDS,
    MAX_TOOL_OUTPUT_BYTES,
    ActionPendingEvent,
    ActionResultEvent,
    ActionStreamEvent,
    ApprovalRequest,
    ApprovalState,
    BoardHidRequestEvent,
    LocalActionRequestEvent,
    gesture_for_action_status,
)
from xuna_backend.modules.actions.domain.schema_validator import (
    ArgumentValidationError,
    SchemaDefinitionError,
    SchemaValidator,
)
from xuna_backend.modules.actions.domain.tool_names import (
    BOARD_MEDIA_KEY_TO_USAGE_INDEX,
    WORKFLOW_TEMPLATE_SCHEMA,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.cognition.domain.types import Locale, Message, Role, ToolCall
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.crypto import args_sha256
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
)

DEADLINE_MS = 10000
DEADLINE_SECONDS = 10.0
BOARD_TARGET_LABEL_VI = "Máy đang ghép"
BOARD_TARGET_LABEL_EN = "Paired host"

POSITION_SYNC_NOTE_VI = (
    "Đồng bộ vị trí phát phụ thuộc vào ứng dụng (ví dụ tài khoản Spotify); "
    "XUNA không chuyển trạng thái phát."
)
POSITION_SYNC_NOTE_EN = (
    "Playback position sync depends on the player (e.g. Spotify account sync); "
    "XUNA does not transfer playback state."
)


class WorkflowTemplate(StrEnum):
    LINK_TO_PHONE = "link_to_phone"
    CONTINUE_MEDIA = "continue_media"
    PAUSE_EVERYWHERE = "pause_everywhere"


def validate_https_public_url(raw_url: str) -> str:
    """Validates that `raw_url` is a public `https://` URL (`<= 2048` chars, no userinfo,
    no IP literals, no `localhost` / `.local` / `.localhost`) and returns its ASCII domain.
    Never includes the URL in exception messages so logs stay URL-free."""
    if not raw_url or len(raw_url) > 2048:
        raise ValueError("url_invalid")
    try:
        parsed = urlparse(raw_url)
    except ValueError as error:
        raise ValueError("url_invalid") from error
    if parsed.scheme.lower() != "https":
        raise ValueError("url_invalid")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url_invalid")
    hostname = parsed.hostname
    if not hostname or hostname.endswith("."):
        raise ValueError("url_invalid")
    try:
        ascii_host = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise ValueError("url_invalid") from error
    if (
        ascii_host == "localhost"
        or ascii_host.endswith(".localhost")
        or ascii_host.endswith(".local")
    ):
        raise ValueError("url_invalid")
    try:
        ipaddress.ip_address(ascii_host.strip("[]"))
        raise ValueError("url_invalid")
    except ValueError as ip_err:
        if str(ip_err) == "url_invalid":
            raise
    parts = ascii_host.split(".")
    if len(parts) < 2 or parts[-1].isdigit():
        raise ValueError("url_invalid")
    return ascii_host


@dataclass(frozen=True, slots=True)
class WorkflowSlotAssignment:
    """Owner-configured board button workflow slot (`1` = GPIO0 short, `2` = GPIO14 short)."""

    owner_id: uuid.UUID
    slot: int
    host_alias: str
    workflow_id: str
    workflow_name: str
    risk: RiskTier

    def __post_init__(self) -> None:
        if self.slot not in (1, 2):
            raise ValueError("invalid_slot")
        if not 1 <= len(self.host_alias.strip()) <= 24:
            raise ValueError("invalid_host_alias")
        if not 1 <= len(self.workflow_id.strip()) <= 64:
            raise ValueError("invalid_workflow_id")
        if not 1 <= len(self.workflow_name.strip()) <= 80:
            raise ValueError("invalid_workflow_name")
        if self.risk.level >= RiskTier.R3.level:
            raise ValueError("workflow_risk_forbidden")


class InMemoryBoardWorkflowSlotStore:
    """Stores owner-assigned board button workflows for slots 1 and 2 (`UDC-9` S9.3)."""

    def __init__(self) -> None:
        self._by_owner_slot: dict[tuple[uuid.UUID, int], WorkflowSlotAssignment] = {}

    def set_slot(self, assignment: WorkflowSlotAssignment) -> WorkflowSlotAssignment:
        if assignment.risk.level >= RiskTier.R3.level:
            raise ValueError("workflow_risk_forbidden")
        self._by_owner_slot[(assignment.owner_id, assignment.slot)] = assignment
        return assignment

    def get_slot(self, owner_id: uuid.UUID, slot: int) -> WorkflowSlotAssignment | None:
        return self._by_owner_slot.get((owner_id, slot))

    def list_slots(self, owner_id: uuid.UUID) -> list[WorkflowSlotAssignment]:
        out: list[WorkflowSlotAssignment] = []
        for slot_num in (1, 2):
            item = self._by_owner_slot.get((owner_id, slot_num))
            if item is not None:
                out.append(item)
        return out


@dataclass(frozen=True, slots=True)
class BoardWorkflowTriggerResult:
    """Result of `POST /v1/actions/workflow-trigger` (`UDC-9` S9.3)."""

    status: str
    slot: int
    host: str
    workflow_id: str
    workflow_name: str
    risk: str
    bubble: str
    gesture: str | None
    approval_id: str | None = None
    summary: str | None = None
    expires_at: str | None = None
    args_sha256: str | None = None


WorkflowEventSink = Callable[[ActionStreamEvent], Awaitable[None]]


async def _noop_emit(_event: ActionStreamEvent) -> None:
    return None


def _encode_workflow_payload(
    status: str,
    output: dict[str, object] | None = None,
    **extra: object,
) -> str:
    body: dict[str, object] = {"status": status}
    if output is not None:
        body["output"] = output
    for k, v in extra.items():
        body[k] = v
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES:
        return encoded
    return json.dumps({"status": status, "output": {"truncated": True}})


class CompositeWorkflowRunner:
    """Executes deterministic cross-device workflow templates (`S9.1`, `S9.2`) and
    board-button workflow triggers (`S9.3`)."""

    def __init__(
        self,
        *,
        host_repo: HostRepository,
        hub: HostSessionHub,
        approval_repo: ApprovalRepository,
        approval_broker: ApprovalBroker,
        local_result_broker: LocalResultBroker,
        audit: AuditLog,
        slot_store: InMemoryBoardWorkflowSlotStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._host_repo = host_repo
        self._hub = hub
        self._approval_repo = approval_repo
        self._approval_broker = approval_broker
        self._local_result_broker = local_result_broker
        self._audit = audit
        self._slot_store = slot_store or InMemoryBoardWorkflowSlotStore()
        self._clock = clock or (lambda: datetime.now(UTC))

    async def handle_template_call(
        self,
        call: ToolCall,
        *,
        owner_id: uuid.UUID,
        locale: Locale,
        client_caps: object,
        board_available: bool,
        emit_event: WorkflowEventSink,
        request_id: str = "",
        self_host_id: str | None = None,
    ) -> Message:
        try:
            clean_args, _ = SchemaValidator.validate_arguments(
                WORKFLOW_TEMPLATE_SCHEMA,
                call.arguments,
                allow_target=False,
            )
        except (ArgumentValidationError, SchemaDefinitionError):
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload("invalid_args"),
                tool_call_id=call.id,
            )

        template_raw = str(clean_args["template"])
        try:
            template = WorkflowTemplate(template_raw)
        except ValueError:
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload("invalid_args"),
                tool_call_id=call.id,
            )

        if template == WorkflowTemplate.LINK_TO_PHONE:
            return await self._run_link_to_phone(
                call=call,
                clean_args=clean_args,
                owner_id=owner_id,
                locale=locale,
                emit_event=emit_event,
                request_id=request_id,
                self_host_id=self_host_id,
            )
        if template == WorkflowTemplate.CONTINUE_MEDIA:
            return await self._run_continue_media(
                call=call,
                clean_args=clean_args,
                owner_id=owner_id,
                locale=locale,
                emit_event=emit_event,
                request_id=request_id,
                self_host_id=self_host_id,
            )
        return await self._run_pause_everywhere(
            call=call,
            owner_id=owner_id,
            locale=locale,
            client_caps=client_caps,
            board_available=board_available,
            emit_event=emit_event,
            request_id=request_id,
            self_host_id=self_host_id,
        )

    async def _load_online_hosts(
        self,
        owner_id: uuid.UUID,
        self_host_id: str | None,
    ) -> tuple[list[Host], dict[HostId, Manifest], dict[HostId, list[Grant]]]:
        hosts = await self._host_repo.list_hosts(owner_id)
        online: list[Host] = []
        manifests: dict[HostId, Manifest] = {}
        grants: dict[HostId, list[Grant]] = {}
        for host in hosts:
            if host.status != HostStatus.ACTIVE:
                continue
            is_self = self_host_id is not None and str(host.host_id) == self_host_id
            if not self._hub.is_online(str(host.host_id)) and not is_self:
                continue
            manifest = await self._host_repo.get_manifest(host.host_id)
            if manifest is None:
                continue
            online.append(host)
            manifests[host.host_id] = manifest
            grants[host.host_id] = await self._host_repo.get_grants(host.host_id)
        online.sort(key=lambda h: str(h.alias).lower())
        return online, manifests, grants

    async def _invoke_host_step(
        self,
        *,
        owner_id: uuid.UUID,
        host: Host,
        tool_name: str,
        arguments: dict[str, object],
        risk: RiskTier,
        request_id: str,
        self_host_id: str | None,
        emit_event: WorkflowEventSink,
        approval_id: uuid.UUID | None = None,
        expires_at: datetime | None = None,
    ) -> tuple[str, dict[str, object]]:
        """Executes a single step on `host` (either via `LocalActionRequestEvent` for
        same-device phone or `hub.send_request` for desktop/background host) and records
        a content-free audit event (never logging URLs or arguments)."""
        now = self._clock()
        invocation_id = str(uuid.uuid4())
        step_approval_id = approval_id or uuid.uuid4()
        step_expires = expires_at or (now + timedelta(seconds=APPROVAL_TTL_SECONDS))
        status = "failed"
        duration_ms = 0
        structured: dict[str, object] = {}

        if self_host_id is not None and str(host.host_id) == self_host_id:
            self._local_result_broker.register(invocation_id, owner_id)
            await emit_event(
                LocalActionRequestEvent(
                    invocation_id=invocation_id,
                    approval_id=str(step_approval_id),
                    target=str(host.alias),
                    tool=tool_name,
                    arguments=arguments,
                )
            )
            report = await self._local_result_broker.wait_report(
                invocation_id,
                timeout_s=DEADLINE_SECONDS,
            )
            raw_status = report.status
            if raw_status in ("denied_by_local_policy", "denied"):
                status = "denied"
            elif raw_status in ("permission_denied", "needs_permission"):
                status = "needs_permission"
            elif raw_status in ("ok", "timeout", "failed", "unsupported", "invalid_args"):
                status = raw_status
            else:
                status = "failed"
            if isinstance(report.output, dict):
                structured = dict(report.output)
        else:
            meta: dict[str, object] = {
                "xuna/invocationId": invocation_id,
                "xuna/deadlineMs": DEADLINE_MS,
            }
            if risk.level >= RiskTier.R2.level:
                meta["xuna/approval"] = {
                    "approvalId": str(step_approval_id),
                    "argsSha256": args_sha256(arguments),
                    "expiresAt": step_expires.astimezone(UTC).isoformat(),
                    "scope": "once",
                }
            call_params: dict[str, object] = {
                "name": tool_name,
                "arguments": arguments,
                "_meta": meta,
            }
            try:
                raw_result = await self._hub.send_request(
                    str(host.host_id),
                    "tools/call",
                    call_params,
                    timeout_s=DEADLINE_SECONDS,
                )
                result_meta = raw_result.get("_meta")
                if isinstance(result_meta, dict):
                    if isinstance(result_meta.get("xuna/status"), str):
                        status = str(result_meta["xuna/status"])
                    if type(result_meta.get("xuna/durationMs")) is int:
                        duration_ms = int(result_meta["xuna/durationMs"])
                sc_raw = raw_result.get("structuredContent")
                if isinstance(sc_raw, dict):
                    structured = dict(sc_raw)
            except TimeoutError:
                status = "timeout"
                duration_ms = DEADLINE_MS
            except Exception:
                status = "failed"

        audit_action = (
            AuditAction.HOST_TOOL_DENIED
            if status == "denied"
            else AuditAction.HOST_TOOL_INVOKED
        )
        await self._audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=audit_action,
                occurred_at=self._clock(),
                subject=str(host.host_id),
                request_id=request_id,
                metadata={
                    "tool": tool_name,
                    "risk": risk.value,
                    "status": status,
                    "durationMs": max(duration_ms, 0),
                    "platform": host.platform.value,
                },
            )
        )
        return status, structured

    async def _request_single_plan_approval(
        self,
        *,
        owner_id: uuid.UUID,
        host_id: HostId | None,
        target_label: str,
        tool_label: str,
        summary: str,
        approval_args: dict[str, object],
        emit_event: WorkflowEventSink,
        request_id: str,
    ) -> tuple[ApprovalState, ApprovalRequest]:
        now = self._clock()
        expires_at = now + timedelta(seconds=APPROVAL_TTL_SECONDS)
        req = ApprovalRequest(
            approval_id=uuid.uuid4(),
            owner_id=owner_id,
            host_id=host_id,
            target_alias=target_label[:32],
            tool=tool_label,
            args_sha256=args_sha256(approval_args),
            risk=RiskTier.R2,
            summary=summary[:240],
            created_at=now,
            expires_at=expires_at,
        )
        await self._approval_repo.save(req)
        await self._approval_broker.create(req)

        await emit_event(
            ActionPendingEvent(
                approval_id=str(req.approval_id),
                target=req.target_alias,
                summary=req.summary,
                risk=req.risk.value,
                expires_in=APPROVAL_TTL_SECONDS,
                host_id=str(host_id) if host_id is not None else None,
                args_sha256=req.args_sha256,
                expires_at=expires_at.astimezone(UTC).isoformat(),
            )
        )

        decision = await self._approval_broker.wait(
            req.approval_id,
            timeout_s=float(APPROVAL_TTL_SECONDS),
        )
        stored = await self._approval_repo.get(req.approval_id)
        if stored is None or stored.state == ApprovalState.PENDING:
            decided_now = self._clock()
            updated = ApprovalRequest(
                approval_id=req.approval_id,
                owner_id=req.owner_id,
                host_id=req.host_id,
                target_alias=req.target_alias,
                tool=req.tool,
                args_sha256=req.args_sha256,
                risk=req.risk,
                summary=req.summary,
                created_at=req.created_at,
                expires_at=req.expires_at,
                state=decision,
                decided_at=decided_now,
            )
            await self._approval_repo.save(updated)
            if decision in (ApprovalState.APPROVED, ApprovalState.REJECTED):
                await self._audit.record(
                    AuditEvent(
                        owner_id=owner_id,
                        action=(
                            AuditAction.HOST_APPROVAL_GRANTED
                            if decision == ApprovalState.APPROVED
                            else AuditAction.HOST_APPROVAL_REJECTED
                        ),
                        occurred_at=decided_now,
                        subject=str(host_id) if host_id is not None else "multi_host",
                        request_id=request_id,
                        metadata={
                            "approvalId": str(req.approval_id),
                            "tool": tool_label,
                            "risk": RiskTier.R2.value,
                            "scope": "once",
                        },
                    )
                )
            req = updated
        else:
            req = stored
        return decision, req

    async def _run_link_to_phone(
        self,
        *,
        call: ToolCall,
        clean_args: dict[str, object],
        owner_id: uuid.UUID,
        locale: Locale,
        emit_event: WorkflowEventSink,
        request_id: str,
        self_host_id: str | None,
    ) -> Message:
        now = self._clock()
        online, manifests, grants = await self._load_online_hosts(owner_id, self_host_id)

        desktop_candidates: list[Host] = []
        phone_candidates: list[Host] = []
        for h in online:
            m = manifests[h.host_id]
            g = grants[h.host_id]
            if h.platform in (HostPlatform.WINDOWS, HostPlatform.MACOS, HostPlatform.LINUX):
                b_desc = m.find_tool("browser.current_url")
                if b_desc is not None and is_tool_granted(b_desc, g, now):
                    desktop_candidates.append(h)
            if h.platform in (HostPlatform.ANDROID, HostPlatform.IOS):
                u_desc = m.find_tool("url.open")
                if u_desc is not None and is_tool_granted(u_desc, g, now):
                    phone_candidates.append(h)

        if not desktop_candidates or not phone_candidates:
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload("no_host"),
                tool_call_id=call.id,
            )

        req_src = str(clean_args.get("source_target", "")).strip()
        if req_src:
            matched_src = [
                h for h in desktop_candidates if str(h.alias).lower() == req_src.lower()
            ]
            if not matched_src:
                return Message(
                    role=Role.TOOL,
                    content=_encode_workflow_payload(
                        "ask_target",
                        choices=[str(h.alias) for h in desktop_candidates],
                    ),
                    tool_call_id=call.id,
                )
            pc_host = matched_src[0]
        elif len(desktop_candidates) == 1:
            pc_host = desktop_candidates[0]
        else:
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload(
                    "ask_target",
                    choices=[str(h.alias) for h in desktop_candidates],
                ),
                tool_call_id=call.id,
            )

        req_dst = str(clean_args.get("destination_target", "")).strip()
        if req_dst:
            matched_dst = [
                h for h in phone_candidates if str(h.alias).lower() == req_dst.lower()
            ]
            if not matched_dst:
                return Message(
                    role=Role.TOOL,
                    content=_encode_workflow_payload(
                        "ask_target",
                        choices=[str(h.alias) for h in phone_candidates],
                    ),
                    tool_call_id=call.id,
                )
            phone_host = matched_dst[0]
        elif len(phone_candidates) == 1:
            phone_host = phone_candidates[0]
        else:
            self_matches = [
                h
                for h in phone_candidates
                if self_host_id is not None and str(h.host_id) == self_host_id
            ]
            if len(self_matches) == 1:
                phone_host = self_matches[0]
            else:
                return Message(
                    role=Role.TOOL,
                    content=_encode_workflow_payload(
                        "ask_target",
                        choices=[str(h.alias) for h in phone_candidates],
                    ),
                    tool_call_id=call.id,
                )

        browser_arg = str(clean_args.get("browser", "auto"))
        step1_args: dict[str, object] = {"browser": browser_arg}
        step1_status, step1_sc = await self._invoke_host_step(
            owner_id=owner_id,
            host=pc_host,
            tool_name="browser.current_url",
            arguments=step1_args,
            risk=RiskTier.R2,
            request_id=request_id,
            self_host_id=self_host_id,
            emit_event=emit_event,
        )
        if step1_status != "ok":
            fail_summary = (
                f"✗ Failed browser.current_url on {pc_host.alias} ({step1_status})"
                if locale == Locale.EN
                else f"✗ Không đọc được trang hiện tại trên {pc_host.alias} ({step1_status})"
            )
            await emit_event(
                ActionResultEvent(
                    invocation_id=str(uuid.uuid4()),
                    target=str(pc_host.alias),
                    summary=fail_summary[:240],
                    status=step1_status,
                    gesture=gesture_for_action_status(step1_status),
                )
            )
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload(
                    step1_status,
                    {
                        "template": WorkflowTemplate.LINK_TO_PHONE.value,
                        "failedStep": 1,
                        "failedTool": "browser.current_url",
                        "failedHost": str(pc_host.alias),
                    },
                ),
                tool_call_id=call.id,
            )

        raw_url = str(step1_sc.get("url", ""))
        resolved_browser = str(step1_sc.get("browser", browser_arg))
        try:
            domain = validate_https_public_url(raw_url)
        except ValueError:
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload(
                    "invalid_args",
                    {
                        "template": WorkflowTemplate.LINK_TO_PHONE.value,
                        "failedStep": 1,
                        "failedTool": "browser.current_url",
                        "failedHost": str(pc_host.alias),
                        "reason": "url_invalid",
                    },
                ),
                tool_call_id=call.id,
            )

        target_label = f"{pc_host.alias} → {phone_host.alias}"[:32]
        pending_summary = (
            f"Send {domain} from {pc_host.alias} to {phone_host.alias}"
            if locale == Locale.EN
            else f"Gửi trang {domain} từ {pc_host.alias} sang {phone_host.alias}"
        )
        # Never store the full URL in approval_args or ApprovalRequest
        approval_args: dict[str, object] = {
            "template": WorkflowTemplate.LINK_TO_PHONE.value,
            "source": str(pc_host.alias),
            "destination": str(phone_host.alias),
            "domain": domain,
        }
        decision, approval_req = await self._request_single_plan_approval(
            owner_id=owner_id,
            host_id=phone_host.host_id,
            target_label=target_label,
            tool_label="workflow.link_to_phone",
            summary=pending_summary,
            approval_args=approval_args,
            emit_event=emit_event,
            request_id=request_id,
        )
        if decision != ApprovalState.APPROVED:
            final_status = "rejected" if decision == ApprovalState.REJECTED else "expired"
            await emit_event(
                ActionResultEvent(
                    invocation_id=str(uuid.uuid4()),
                    target=target_label,
                    summary=f"✗ {pending_summary} ({final_status})"[:240],
                    status=final_status,
                    gesture=gesture_for_action_status(final_status),
                )
            )
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload(final_status),
                tool_call_id=call.id,
            )

        step2_status, _ = await self._invoke_host_step(
            owner_id=owner_id,
            host=phone_host,
            tool_name="url.open",
            arguments={"url": raw_url},
            risk=RiskTier.R2,
            request_id=request_id,
            self_host_id=self_host_id,
            emit_event=emit_event,
            approval_id=approval_req.approval_id,
            expires_at=approval_req.expires_at,
        )
        if step2_status != "ok":
            fail2_summary = (
                f"✗ Failed url.open ({domain}) on {phone_host.alias} ({step2_status})"
                if locale == Locale.EN
                else f"✗ Không mở được trang {domain} trên {phone_host.alias} ({step2_status})"
            )
            await emit_event(
                ActionResultEvent(
                    invocation_id=str(uuid.uuid4()),
                    target=str(phone_host.alias),
                    summary=fail2_summary[:240],
                    status=step2_status,
                    gesture=gesture_for_action_status(step2_status),
                )
            )
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload(
                    step2_status,
                    {
                        "template": WorkflowTemplate.LINK_TO_PHONE.value,
                        "failedStep": 2,
                        "failedTool": "url.open",
                        "failedHost": str(phone_host.alias),
                        "domain": domain,
                    },
                ),
                tool_call_id=call.id,
            )

        ok_summary = (
            f"✓ Opened {domain} on {phone_host.alias} (from {pc_host.alias})"
            if locale == Locale.EN
            else f"✓ Đã mở trang {domain} trên {phone_host.alias} (từ {pc_host.alias})"
        )
        await emit_event(
            ActionResultEvent(
                invocation_id=str(uuid.uuid4()),
                target=target_label,
                summary=ok_summary[:240],
                status="ok",
                gesture=gesture_for_action_status("ok"),
            )
        )

        return Message(
            role=Role.TOOL,
            content=_encode_workflow_payload(
                "ok",
                {
                    "template": WorkflowTemplate.LINK_TO_PHONE.value,
                    "sourceHost": str(pc_host.alias),
                    "destinationHost": str(phone_host.alias),
                    "domain": domain,
                    "browser": resolved_browser,
                },
            ),
            tool_call_id=call.id,
        )

    async def _run_continue_media(
        self,
        *,
        call: ToolCall,
        clean_args: dict[str, object],
        owner_id: uuid.UUID,
        locale: Locale,
        emit_event: WorkflowEventSink,
        request_id: str,
        self_host_id: str | None,
    ) -> Message:
        now = self._clock()
        online, manifests, grants = await self._load_online_hosts(owner_id, self_host_id)

        req_src = str(clean_args.get("source_target", "")).strip()
        req_dst = str(clean_args.get("destination_target", "")).strip()
        if not req_src or not req_dst:
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload(
                    "ask_target",
                    choices=[str(h.alias) for h in online],
                ),
                tool_call_id=call.id,
            )

        src_matches = [h for h in online if str(h.alias).lower() == req_src.lower()]
        dst_matches = [h for h in online if str(h.alias).lower() == req_dst.lower()]
        if not src_matches or not dst_matches or src_matches[0].host_id == dst_matches[0].host_id:
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload(
                    "ask_target",
                    choices=[str(h.alias) for h in online],
                ),
                tool_call_id=call.id,
            )

        host_a = src_matches[0]
        host_b = dst_matches[0]
        manifest_a = manifests[host_a.host_id]
        grants_a = grants[host_a.host_id]
        manifest_b = manifests[host_b.host_id]
        grants_b = grants[host_b.host_id]

        np_desc_a = manifest_a.find_tool("media.now_playing")
        pp_desc_a = manifest_a.find_tool("media.play_pause")
        app_desc_b = manifest_b.find_tool("app.launch")
        pp_desc_b = manifest_b.find_tool("media.play_pause")

        if (
            np_desc_a is None
            or pp_desc_a is None
            or app_desc_b is None
            or pp_desc_b is None
            or not is_tool_granted(np_desc_a, grants_a, now)
            or not is_tool_granted(pp_desc_a, grants_a, now)
            or not is_tool_granted(app_desc_b, grants_b, now)
            or not is_tool_granted(pp_desc_b, grants_b, now)
        ):
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload("denied"),
                tool_call_id=call.id,
            )

        requested_player = str(clean_args.get("player_id", "")).strip() or "spotify"
        target_label = f"{host_a.alias} → {host_b.alias}"[:32]
        pending_summary = (
            f"Continue media from {host_a.alias} to {host_b.alias}"
            if locale == Locale.EN
            else f"Chuyển phát nhạc từ {host_a.alias} sang {host_b.alias}"
        )
        approval_args: dict[str, object] = {
            "template": WorkflowTemplate.CONTINUE_MEDIA.value,
            "source": str(host_a.alias),
            "destination": str(host_b.alias),
            "playerId": requested_player,
        }
        decision, approval_req = await self._request_single_plan_approval(
            owner_id=owner_id,
            host_id=host_b.host_id,
            target_label=target_label,
            tool_label="workflow.continue_media",
            summary=pending_summary,
            approval_args=approval_args,
            emit_event=emit_event,
            request_id=request_id,
        )
        if decision != ApprovalState.APPROVED:
            final_status = "rejected" if decision == ApprovalState.REJECTED else "expired"
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload(final_status),
                tool_call_id=call.id,
            )

        # Step 1: media.now_playing on A
        s1_status, s1_sc = await self._invoke_host_step(
            owner_id=owner_id,
            host=host_a,
            tool_name="media.now_playing",
            arguments={},
            risk=RiskTier.R0,
            request_id=request_id,
            self_host_id=self_host_id,
            emit_event=emit_event,
        )
        if s1_status != "ok":
            return await self._fail_continue_media_step(
                call=call,
                step=1,
                tool_name="media.now_playing",
                failed_host=host_a,
                status=s1_status,
                locale=locale,
                emit_event=emit_event,
            )

        detected_app = str(s1_sc.get("app", "")).strip().lower()
        player_id = (
            str(clean_args["player_id"]).strip()
            if "player_id" in clean_args and str(clean_args["player_id"]).strip()
            else (detected_app or requested_player)
        )

        # Step 2: media.play_pause (pause) on A
        s2_status, _ = await self._invoke_host_step(
            owner_id=owner_id,
            host=host_a,
            tool_name="media.play_pause",
            arguments={},
            risk=RiskTier.R1,
            request_id=request_id,
            self_host_id=self_host_id,
            emit_event=emit_event,
        )
        if s2_status != "ok":
            return await self._fail_continue_media_step(
                call=call,
                step=2,
                tool_name="media.play_pause",
                failed_host=host_a,
                status=s2_status,
                locale=locale,
                emit_event=emit_event,
            )

        # Step 3: app.launch player on B
        s3_status, _ = await self._invoke_host_step(
            owner_id=owner_id,
            host=host_b,
            tool_name="app.launch",
            arguments={"id": player_id},
            risk=RiskTier.R2,
            request_id=request_id,
            self_host_id=self_host_id,
            emit_event=emit_event,
            approval_id=approval_req.approval_id,
            expires_at=approval_req.expires_at,
        )
        if s3_status != "ok":
            return await self._fail_continue_media_step(
                call=call,
                step=3,
                tool_name="app.launch",
                failed_host=host_b,
                status=s3_status,
                locale=locale,
                emit_event=emit_event,
            )

        # Step 4: media.play_pause on B
        s4_status, _ = await self._invoke_host_step(
            owner_id=owner_id,
            host=host_b,
            tool_name="media.play_pause",
            arguments={},
            risk=RiskTier.R1,
            request_id=request_id,
            self_host_id=self_host_id,
            emit_event=emit_event,
        )
        if s4_status != "ok":
            return await self._fail_continue_media_step(
                call=call,
                step=4,
                tool_name="media.play_pause",
                failed_host=host_b,
                status=s4_status,
                locale=locale,
                emit_event=emit_event,
            )

        note = POSITION_SYNC_NOTE_EN if locale == Locale.EN else POSITION_SYNC_NOTE_VI
        ok_summary = (
            f"✓ Continued media on {host_b.alias} from {host_a.alias} ({player_id}). {note}"
            if locale == Locale.EN
            else f"✓ Đã chuyển nhạc từ {host_a.alias} sang {host_b.alias} ({player_id}). {note}"
        )
        await emit_event(
            ActionResultEvent(
                invocation_id=str(uuid.uuid4()),
                target=target_label,
                summary=ok_summary[:240],
                status="ok",
                gesture=gesture_for_action_status("ok"),
            )
        )

        return Message(
            role=Role.TOOL,
            content=_encode_workflow_payload(
                "ok",
                {
                    "template": WorkflowTemplate.CONTINUE_MEDIA.value,
                    "sourceHost": str(host_a.alias),
                    "destinationHost": str(host_b.alias),
                    "playerId": player_id,
                    "stepsCompleted": 4,
                    "note": note,
                },
            ),
            tool_call_id=call.id,
        )

    async def _fail_continue_media_step(
        self,
        *,
        call: ToolCall,
        step: int,
        tool_name: str,
        failed_host: Host,
        status: str,
        locale: Locale,
        emit_event: WorkflowEventSink,
    ) -> Message:
        summary = (
            f"✗ Step {step} ({tool_name}) failed on {failed_host.alias} ({status})"
            if locale == Locale.EN
            else f"✗ Bước {step} ({tool_name}) thất bại trên {failed_host.alias} ({status})"
        )
        await emit_event(
            ActionResultEvent(
                invocation_id=str(uuid.uuid4()),
                target=str(failed_host.alias),
                summary=summary[:240],
                status=status,
                gesture=gesture_for_action_status(status),
            )
        )
        return Message(
            role=Role.TOOL,
            content=_encode_workflow_payload(
                status,
                {
                    "template": WorkflowTemplate.CONTINUE_MEDIA.value,
                    "failedStep": step,
                    "failedTool": tool_name,
                    "failedHost": str(failed_host.alias),
                },
            ),
            tool_call_id=call.id,
        )

    async def _run_pause_everywhere(
        self,
        *,
        call: ToolCall,
        owner_id: uuid.UUID,
        locale: Locale,
        client_caps: object,
        board_available: bool,
        emit_event: WorkflowEventSink,
        request_id: str,
        self_host_id: str | None,
    ) -> Message:
        now = self._clock()
        online, manifests, grants = await self._load_online_hosts(owner_id, self_host_id)

        media_hosts: list[Host] = []
        for h in online:
            m = manifests[h.host_id]
            g = grants[h.host_id]
            pp_desc = m.find_tool("media.play_pause")
            if pp_desc is not None and is_tool_granted(pp_desc, g, now):
                media_hosts.append(h)

        caps_set = (
            set(client_caps)
            if isinstance(client_caps, list | tuple | set | frozenset)
            else set()
        )
        include_board = board_available and CAP_BOARD_HID_V1 in caps_set
        board_label = BOARD_TARGET_LABEL_EN if locale == Locale.EN else BOARD_TARGET_LABEL_VI

        if not media_hosts and not include_board:
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload("no_host"),
                tool_call_id=call.id,
            )

        all_aliases = [str(h.alias) for h in media_hosts]
        if include_board:
            all_aliases.append(board_label)

        aliases_str = ", ".join(all_aliases)
        target_label = aliases_str[:32]
        pending_summary = (
            f"Pause media everywhere ({aliases_str})"
            if locale == Locale.EN
            else f"Dừng nhạc ở mọi nơi ({aliases_str})"
        )
        approval_args: dict[str, object] = {
            "template": WorkflowTemplate.PAUSE_EVERYWHERE.value,
            "targets": all_aliases,
        }
        decision, approval_req = await self._request_single_plan_approval(
            owner_id=owner_id,
            host_id=media_hosts[0].host_id if media_hosts else None,
            target_label=target_label,
            tool_label="workflow.pause_everywhere",
            summary=pending_summary,
            approval_args=approval_args,
            emit_event=emit_event,
            request_id=request_id,
        )
        if decision != ApprovalState.APPROVED:
            final_status = "rejected" if decision == ApprovalState.REJECTED else "expired"
            return Message(
                role=Role.TOOL,
                content=_encode_workflow_payload(final_status),
                tool_call_id=call.id,
            )

        per_host_results: list[dict[str, object]] = []
        step_index = 0

        for h in media_hosts:
            step_index += 1
            m = manifests[h.host_id]
            g = grants[h.host_id]
            np_desc = m.find_tool("media.now_playing")
            should_toggle = True
            if np_desc is not None and is_tool_granted(np_desc, g, now):
                np_status, np_sc = await self._invoke_host_step(
                    owner_id=owner_id,
                    host=h,
                    tool_name="media.now_playing",
                    arguments={},
                    risk=RiskTier.R0,
                    request_id=request_id,
                    self_host_id=self_host_id,
                    emit_event=emit_event,
                )
                if np_status == "ok":
                    playback_state = str(np_sc.get("state", "unknown")).lower()
                    if playback_state in ("paused", "stopped"):
                        should_toggle = False
                        per_host_results.append(
                            {
                                "host": str(h.alias),
                                "status": "ok",
                                "action": "already_paused",
                            }
                        )

            if should_toggle:
                pp_status, _ = await self._invoke_host_step(
                    owner_id=owner_id,
                    host=h,
                    tool_name="media.play_pause",
                    arguments={},
                    risk=RiskTier.R1,
                    request_id=request_id,
                    self_host_id=self_host_id,
                    emit_event=emit_event,
                )
                per_host_results.append(
                    {
                        "host": str(h.alias),
                        "status": pp_status,
                        "action": "paused" if pp_status == "ok" else "failed",
                    }
                )
                if pp_status != "ok":
                    fail_summary = (
                        f"✗ Failed media.play_pause on {h.alias} ({pp_status})"
                        if locale == Locale.EN
                        else f"✗ Dừng nhạc thất bại trên {h.alias} ({pp_status})"
                    )
                    await emit_event(
                        ActionResultEvent(
                            invocation_id=str(uuid.uuid4()),
                            target=str(h.alias),
                            summary=fail_summary[:240],
                            status=pp_status,
                            gesture=gesture_for_action_status(pp_status),
                        )
                    )
                    return Message(
                        role=Role.TOOL,
                        content=_encode_workflow_payload(
                            pp_status,
                            {
                                "template": WorkflowTemplate.PAUSE_EVERYWHERE.value,
                                "failedStep": step_index,
                                "failedTool": "media.play_pause",
                                "failedHost": str(h.alias),
                                "results": per_host_results,
                            },
                        ),
                        tool_call_id=call.id,
                    )

        if include_board:
            step_index += 1
            board_inv_id = str(uuid.uuid4())
            self._local_result_broker.register(board_inv_id, owner_id)
            await emit_event(
                BoardHidRequestEvent(
                    invocation_id=board_inv_id,
                    approval_id=str(approval_req.approval_id),
                    key="stop",
                    usage_index=BOARD_MEDIA_KEY_TO_USAGE_INDEX["stop"],
                )
            )
            board_status = await self._local_result_broker.wait(
                board_inv_id,
                timeout_s=DEADLINE_SECONDS,
            )
            norm_board = "ok" if board_status == "ok" else "failed"
            per_host_results.append(
                {
                    "host": board_label,
                    "status": norm_board,
                    "action": "paused" if norm_board == "ok" else "failed",
                }
            )
            if norm_board != "ok":
                return Message(
                    role=Role.TOOL,
                    content=_encode_workflow_payload(
                        norm_board,
                        {
                            "template": WorkflowTemplate.PAUSE_EVERYWHERE.value,
                            "failedStep": step_index,
                            "failedTool": "board.media_key",
                            "failedHost": board_label,
                            "results": per_host_results,
                        },
                    ),
                    tool_call_id=call.id,
                )

        ok_summary = (
            f"✓ Paused media everywhere ({aliases_str})"
            if locale == Locale.EN
            else f"✓ Đã dừng nhạc ở mọi nơi ({aliases_str})"
        )
        await emit_event(
            ActionResultEvent(
                invocation_id=str(uuid.uuid4()),
                target=target_label,
                summary=ok_summary[:240],
                status="ok",
                gesture=gesture_for_action_status("ok"),
            )
        )

        return Message(
            role=Role.TOOL,
            content=_encode_workflow_payload(
                "ok",
                {
                    "template": WorkflowTemplate.PAUSE_EVERYWHERE.value,
                    "targets": all_aliases,
                    "results": per_host_results,
                },
            ),
            tool_call_id=call.id,
        )

    async def trigger_board_workflow(
        self,
        *,
        owner_id: uuid.UUID,
        slot: int,
        host_alias: str | None = None,
        workflow_id: str | None = None,
        workflow_name: str | None = None,
        risk: RiskTier | None = None,
        approval_id: uuid.UUID | None = None,
        wait_for_approval: bool = False,
        request_id: str = "",
        self_host_id: str | None = None,
    ) -> BoardWorkflowTriggerResult:
        """Executes a board-button workflow trigger (`POST /v1/actions/workflow-trigger`,
        `UDC-9` S9.3):
        - Enforces `UDC-T15`: `R3` workflows can never be assigned to or triggered from
          board buttons (`ValueError("workflow_risk_forbidden")`).
        - `R1` workflows execute immediately (owner is physically present on the board).
        - `R2` workflows require phone approval."""
        if slot not in (1, 2):
            raise ValueError("invalid_slot")

        stored = self._slot_store.get_slot(owner_id, slot)
        resolved_alias = (host_alias or (stored.host_alias if stored else "")).strip()
        resolved_wf_id = (workflow_id or (stored.workflow_id if stored else "")).strip()
        resolved_wf_name = (
            workflow_name
            or (stored.workflow_name if stored else "")
            or resolved_wf_id
        ).strip()
        resolved_risk = risk or (stored.risk if stored else RiskTier.R1)

        if not resolved_alias or not resolved_wf_id:
            raise ValueError("slot_not_configured")

        if resolved_risk.level >= RiskTier.R3.level:
            await self._audit.record(
                AuditEvent(
                    owner_id=owner_id,
                    action=AuditAction.HOST_TOOL_DENIED,
                    occurred_at=self._clock(),
                    subject="board_workflow",
                    request_id=request_id,
                    metadata={
                        "slot": slot,
                        "workflowId": resolved_wf_id,
                        "risk": resolved_risk.value,
                        "reason": "workflow_risk_forbidden",
                    },
                )
            )
            raise ValueError("workflow_risk_forbidden")

        now = self._clock()
        online, manifests, grants = await self._load_online_hosts(owner_id, self_host_id)
        matched_hosts = [
            h for h in online if str(h.alias).lower() == resolved_alias.lower()
        ]
        if not matched_hosts:
            raise ValueError("host_offline")

        host = matched_hosts[0]
        manifest = manifests[host.host_id]
        host_grants = grants[host.host_id]
        wf_desc = manifest.find_tool("workflow.run")
        if wf_desc is None:
            raise ValueError("tool_unsupported")

        effective_risk = (
            resolved_risk
            if resolved_risk.level >= wf_desc.risk.level
            else wf_desc.risk
        )
        if effective_risk.level >= RiskTier.R3.level:
            await self._audit.record(
                AuditEvent(
                    owner_id=owner_id,
                    action=AuditAction.HOST_TOOL_DENIED,
                    occurred_at=now,
                    subject=str(host.host_id),
                    request_id=request_id,
                    metadata={
                        "slot": slot,
                        "workflowId": resolved_wf_id,
                        "risk": effective_risk.value,
                        "reason": "workflow_risk_forbidden",
                    },
                )
            )
            raise ValueError("workflow_risk_forbidden")

        has_grant = any(
            g.family == wf_desc.family
            and g.is_active(now)
            and g.max_risk.allows(effective_risk)
            for g in host_grants
        )
        if not has_grant:
            raise ValueError("grant_denied")

        bubble = f"Đang chạy: {resolved_wf_name}"[:80]
        wf_args: dict[str, object] = {"id": resolved_wf_id}

        if effective_risk == RiskTier.R1:
            status, _ = await self._invoke_host_step(
                owner_id=owner_id,
                host=host,
                tool_name="workflow.run",
                arguments=wf_args,
                risk=RiskTier.R1,
                request_id=request_id,
                self_host_id=self_host_id,
                emit_event=_noop_emit,
            )
            return BoardWorkflowTriggerResult(
                status=status,
                slot=slot,
                host=str(host.alias),
                workflow_id=resolved_wf_id,
                workflow_name=resolved_wf_name,
                risk=effective_risk.value,
                bubble=bubble,
                gesture=gesture_for_action_status(status),
            )

        # Effective risk is R2: requires phone approval
        if approval_id is not None:
            existing = await self._approval_repo.get(approval_id)
            if (
                existing is None
                or existing.owner_id != owner_id
                or existing.state != ApprovalState.APPROVED
                or existing.is_expired(now)
                or existing.args_sha256 != args_sha256(wf_args)
            ):
                raise ValueError("approval_invalid")
            status, _ = await self._invoke_host_step(
                owner_id=owner_id,
                host=host,
                tool_name="workflow.run",
                arguments=wf_args,
                risk=RiskTier.R2,
                request_id=request_id,
                self_host_id=self_host_id,
                emit_event=_noop_emit,
                approval_id=existing.approval_id,
                expires_at=existing.expires_at,
            )
            return BoardWorkflowTriggerResult(
                status=status,
                slot=slot,
                host=str(host.alias),
                workflow_id=resolved_wf_id,
                workflow_name=resolved_wf_name,
                risk=effective_risk.value,
                bubble=bubble,
                gesture=gesture_for_action_status(status),
                approval_id=str(existing.approval_id),
            )

        expires_at = now + timedelta(seconds=APPROVAL_TTL_SECONDS)
        summary = f"Chạy quy trình {resolved_wf_name} trên {host.alias}"[:240]
        req = ApprovalRequest(
            approval_id=uuid.uuid4(),
            owner_id=owner_id,
            host_id=host.host_id,
            target_alias=str(host.alias),
            tool="workflow.run",
            args_sha256=args_sha256(wf_args),
            risk=RiskTier.R2,
            summary=summary,
            created_at=now,
            expires_at=expires_at,
        )
        await self._approval_repo.save(req)
        await self._approval_broker.create(req)

        if not wait_for_approval:
            return BoardWorkflowTriggerResult(
                status="needs_approval",
                slot=slot,
                host=str(host.alias),
                workflow_id=resolved_wf_id,
                workflow_name=resolved_wf_name,
                risk=effective_risk.value,
                bubble=bubble,
                gesture=None,
                approval_id=str(req.approval_id),
                summary=req.summary,
                expires_at=expires_at.astimezone(UTC).isoformat(),
                args_sha256=req.args_sha256,
            )

        decision = await self._approval_broker.wait(
            req.approval_id,
            timeout_s=float(APPROVAL_TTL_SECONDS),
        )
        if decision != ApprovalState.APPROVED:
            final_state = "rejected" if decision == ApprovalState.REJECTED else "expired"
            return BoardWorkflowTriggerResult(
                status=final_state,
                slot=slot,
                host=str(host.alias),
                workflow_id=resolved_wf_id,
                workflow_name=resolved_wf_name,
                risk=effective_risk.value,
                bubble=bubble,
                gesture=gesture_for_action_status(final_state),
                approval_id=str(req.approval_id),
            )

        status, _ = await self._invoke_host_step(
            owner_id=owner_id,
            host=host,
            tool_name="workflow.run",
            arguments=wf_args,
            risk=RiskTier.R2,
            request_id=request_id,
            self_host_id=self_host_id,
            emit_event=_noop_emit,
            approval_id=req.approval_id,
            expires_at=req.expires_at,
        )
        return BoardWorkflowTriggerResult(
            status=status,
            slot=slot,
            host=str(host.alias),
            workflow_id=resolved_wf_id,
            workflow_name=resolved_wf_name,
            risk=effective_risk.value,
            bubble=bubble,
            gesture=gesture_for_action_status(status),
            approval_id=str(req.approval_id),
        )
