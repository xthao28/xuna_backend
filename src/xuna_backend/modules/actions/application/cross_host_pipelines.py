"""Phase 3 cross-machine superpower pipelines (Mesh §5.6.7, §7, §9).

Implements:
- `mesh__clipboard_transfer`: dual-end secret redaction and 1,200-char selection cap.
- `mesh__teleport`: public HTTPS validation and workspace capsule teleportation.
- `mesh__smart_pause_media`: presence-aware fan-out respecting `idempotent=false` (H5).
- `mesh__fleet_vitals`: concurrent `system.vitals` fan-out + Battery Advisor.
- `mesh__fleet_git_status`: concurrent `workspace.git_summary` fan-out across indexed repos.
- Wake-on-LAN reconnect promotion and heavy-task battery advisory helpers.

Zero hardcoded app names, host names, OS paths, or keyword-intent tables (Rule H5).
"""

import asyncio
import base64
import contextlib
import inspect
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from xuna_backend.modules.actions.application.approval_and_board_service import (
    ApprovalAndBoardService,
)
from xuna_backend.modules.actions.application.catalog import is_tool_granted
from xuna_backend.modules.actions.application.mesh_prober import (
    MeshProber,
    ProbeTarget,
    build_probe_targets,
)
from xuna_backend.modules.actions.application.mesh_router import (
    PendingIntent,
    PendingIntentStore,
    wire_candidate_dict,
)
from xuna_backend.modules.actions.application.ports import (
    ApprovalBroker,
    ApprovalRepository,
    ApprovalScopeStore,
    LastTargetStore,
    LocalResultBroker,
    StandingApprovalRepository,
)
from xuna_backend.modules.actions.application.prompts import (
    DEADLINE_MS,
    DEADLINE_SECONDS,
    encode_tool_payload,
)
from xuna_backend.modules.actions.application.tool_discovery import (
    MAX_MESH_HOST_IDS,
    MESH_CLIPBOARD_TRANSFER_FUNCTION,
    MESH_FLEET_GIT_STATUS_FUNCTION,
    MESH_FLEET_VITALS_FUNCTION,
    MESH_SMART_PAUSE_MEDIA_FUNCTION,
    MESH_TELEPORT_FUNCTION,
    encode_mesh_meta_payload,
)
from xuna_backend.modules.actions.application.workflows import validate_https_public_url
from xuna_backend.modules.actions.domain.battery_advisor import (
    LOW_BATTERY_THRESHOLD_PCT,
    evaluate_battery_advisory,
)
from xuna_backend.modules.actions.domain.entities import (
    APPROVAL_TTL_SECONDS,
    ActionPendingEvent,
    ActionResultEvent,
    ActionTargetRequiredEvent,
    ApprovalRequest,
    ApprovalScope,
    ApprovalState,
    HostWakeRequestEvent,
    LocalActionRequestEvent,
    gesture_for_action_status,
)
from xuna_backend.modules.actions.domain.mesh_scoring import (
    ActionTargetRequiredReason,
    CandidateUnavailableReason,
    ScoredHostCandidate,
    match_hosts_by_query,
)
from xuna_backend.modules.actions.domain.mission_envelope import (
    MissionEnvelope,
    MissionStep,
    MissionStepStatus,
)
from xuna_backend.modules.actions.domain.redaction import (
    MAX_CLIPBOARD_WRITE_CHARS,
    MAX_SELECTION_CHARS,
    prepare_selection_for_clipboard,
    redact_sensitive_text,
    validate_clipboard_payload,
)
from xuna_backend.modules.actions.domain.wake_queue import (
    WAKE_QUEUE_TTL,
    DeferredWakeStep,
    InMemoryWakeQueueStore,
    WakeQueueFullError,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.cognition.domain.types import Message, Role, ToolCall
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.crypto import args_sha256
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostId,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
)
from xuna_backend.modules.hosts.domain.session import structured_content_exceeds_limit
from xuna_backend.modules.hosts.domain.twin import HostTwin
from xuna_backend.modules.identity.application.ports import SignatureVerifier

if TYPE_CHECKING:
    from xuna_backend.modules.actions.application.orchestrator import ActionContext

_SAN_PROJECT_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
_CLIPBOARD_WRITE_TOOL = "clipboard.write"
_CLOSED_XTP_STATUSES: frozenset[str] = frozenset(
    {
        "ok",
        "failed",
        "denied",
        "timeout",
        "unsupported",
        "invalid_args",
        "needs_permission",
    }
)


def _normalize_xtp_status(raw: object) -> str:
    """Normalizes untrusted host/local status strings to the closed XTP status enum."""
    if not isinstance(raw, str):
        return "failed"
    cleaned = raw.strip()
    if cleaned in ("denied_by_local_policy", "denied"):
        return "denied"
    if cleaned in ("permission_denied", "needs_permission"):
        return "needs_permission"
    if cleaned in _CLOSED_XTP_STATUSES:
        return cleaned
    return "failed"


def validate_teleport_https_url(raw_url: str) -> str:
    """Rejects ASCII control characters (0x00..0x1F, 0x7F) before public HTTPS URL validation."""
    if not isinstance(raw_url, str) or any(ord(ch) < 32 or ord(ch) == 127 for ch in raw_url):
        raise ValueError("url_invalid")
    return validate_https_public_url(raw_url)


def _aggregate_fanout_failure(
    statuses: Sequence[str],
    error_codes: Sequence[str],
    *,
    empty_err: str = "no_host",
) -> tuple[str, str]:
    """Derives an honest closed status + errorCode when 0 fan-out hosts succeeded."""
    if not statuses:
        return "failed", empty_err
    unique = set(statuses)
    if unique == {"denied"}:
        return "denied", (error_codes[0] if error_codes else "grant_denied")
    if unique == {"timeout"}:
        return "timeout", (error_codes[0] if error_codes else "timeout")
    if unique == {"unsupported"}:
        return "unsupported", (error_codes[0] if error_codes else "unsupported")
    if unique == {"needs_permission"}:
        return "needs_permission", (error_codes[0] if error_codes else "needs_permission")
    return "failed", (error_codes[0] if error_codes else "failed")


def _derive_r2_step_approval(
    base_req: ApprovalRequest,
    *,
    host: Host,
    tool: str,
    arguments: dict[str, object],
) -> ApprovalRequest:
    """Binds a step-specific `args_sha256` to an owner-approved `R2` mission approval."""
    return ApprovalRequest(
        approval_id=base_req.approval_id,
        owner_id=base_req.owner_id,
        host_id=host.host_id,
        target_alias=host.alias.value[:32],
        tool=tool,
        args_sha256=args_sha256(arguments),
        risk=RiskTier.R2,
        summary=base_req.summary,
        created_at=base_req.created_at,
        expires_at=base_req.expires_at,
        state=ApprovalState.APPROVED,
        decided_at=base_req.decided_at,
        scope=base_req.scope,
    )


def _safe_get_twin(hub: HostSessionHub, host_id: str) -> HostTwin | None:
    fn = getattr(hub, "get_twin", None)
    if callable(fn):
        twin = fn(host_id)
        if isinstance(twin, HostTwin):
            return twin
    return None


def _is_host_online(hub: HostSessionHub, host: Host, self_host_id: str | None) -> bool:
    hid = str(host.host_id)
    return hub.is_online(hid) or (self_host_id is not None and hid == self_host_id)


def _sanitize_alias_token(raw: object, *, max_len: int = 64) -> str | None:
    if not isinstance(raw, str):
        return None
    clean = raw.strip()
    if not clean or len(clean) > max_len:
        return None
    if clean in (".", "..") or "/" in clean or "\\" in clean:
        return None
    if not all(ch in _SAN_PROJECT_CHARS for ch in clean):
        return None
    return clean


def _parse_host_id_filter(raw: object) -> set[uuid.UUID] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) > MAX_MESH_HOST_IDS:
        raise ValueError("invalid_args")
    out: set[uuid.UUID] = set()
    for item in raw:
        if not isinstance(item, str) or len(item) > 64:
            raise ValueError("invalid_args")
        try:
            out.add(uuid.UUID(item.strip()))
        except ValueError as exc:
            raise ValueError("invalid_args") from exc
    return out


def build_pending_battery_advisories(
    hosts: Sequence[Host],
    hub: HostSessionHub,
    target_host_id: HostId,
    tool_name: str,
    *,
    self_host_id: str | None = None,
) -> tuple[str, ...]:
    """Evaluates Battery Advisor warnings when starting heavy CU or media work."""
    if tool_name not in ("cu.session_start", "cu.session_extend"):
        return ()
    online_ids: set[str] = set()
    twins_by_host: dict[str, HostTwin] = {}
    for host in hosts:
        if host.status != HostStatus.ACTIVE:
            continue
        hid = str(host.host_id)
        if _is_host_online(hub, host, self_host_id):
            online_ids.add(hid)
            twin = _safe_get_twin(hub, hid)
            if twin is not None:
                twins_by_host[hid] = twin
    advisory = evaluate_battery_advisory(
        hosts,
        twins_by_host,
        online_host_ids=online_ids,
        target_host_id=target_host_id,
        threshold_pct=LOW_BATTERY_THRESHOLD_PCT,
    )
    return advisory.messages if advisory is not None else ()


async def handle_host_ready_wake_queue(
    *,
    owner_id: uuid.UUID,
    host_id: HostId,
    wake_queue: InMemoryWakeQueueStore | None,
    pending_intent_store: PendingIntentStore | None,
    job_runner: Any | None = None,
    now: datetime | None = None,
) -> int:
    """Promotes unexpired WoL steps into `PendingIntentStore` and resumes waiting jobs.

    Never executes `R2+` steps silently on reconnect; owner confirmation still applies.
    """
    check_now = now or datetime.now(UTC)
    handled = 0
    ready_job_ids: set[uuid.UUID] | None = None
    if wake_queue is not None:
        ready_steps = wake_queue.pop_ready_for_host(owner_id, host_id, check_now)
        ready_job_ids = {s.job_id for s in ready_steps if s.job_id is not None}
        interactive = [s for s in ready_steps if s.job_id is None and (s.tool or s.summary)]
        if interactive and pending_intent_store is not None:
            latest = interactive[-1]
            pending_intent_store.save(
                PendingIntent(
                    owner_id=latest.owner_id,
                    pet_id=latest.pet_id,
                    original_tool=latest.tool,
                    original_arguments=dict(latest.arguments),
                    original_text=latest.summary,
                    candidate_host_ids=(latest.host_id,),
                    created_at=check_now,
                    suggested_host_id=latest.host_id,
                    resource_id=latest.resource_id,
                    family=latest.family,
                    client_caps=latest.client_caps,
                )
            )
        handled += len(ready_steps)

    if job_runner is not None:
        resume_fn = getattr(job_runner, "resume_waiting_wake_jobs", None)
        if callable(resume_fn):
            resumed_jobs = await resume_fn(
                owner_id,
                host_id,
                allowed_job_ids=(ready_job_ids or None),
                now=check_now,
            )
            if isinstance(resumed_jobs, int):
                handled += resumed_jobs
    return handled


class CrossHostPipelineRunner:
    """Executes Phase 3 cross-host superpowers with MissionEnvelope grouping and H1-H5 safety."""

    def __init__(
        self,
        *,
        host_repo: HostRepository,
        hub: HostSessionHub,
        approval_repo: ApprovalRepository,
        approval_broker: ApprovalBroker,
        local_result_broker: LocalResultBroker,
        last_target_store: LastTargetStore,
        audit: AuditLog,
        prober: MeshProber,
        verifier: SignatureVerifier | None = None,
        scope_store: ApprovalScopeStore | None = None,
        standing_repo: StandingApprovalRepository | None = None,
        pending_intent_store: PendingIntentStore | None = None,
        wake_queue: InMemoryWakeQueueStore | None = None,
        clock: Callable[[], datetime] | None = None,
        approval_service: ApprovalAndBoardService | None = None,
    ) -> None:
        self._host_repo = host_repo
        self._hub = hub
        self._approval_repo = approval_repo
        self._approval_broker = approval_broker
        self._local_result_broker = local_result_broker
        self._audit = audit
        self._prober = prober
        self._scope_store = scope_store
        self._standing_repo = standing_repo
        self._pending_intent_store = pending_intent_store
        self._wake_queue = wake_queue
        self._clock = clock or (lambda: datetime.now(UTC))
        self._approval_service = approval_service or ApprovalAndBoardService(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_result_broker,
            last_target_store=last_target_store,
            audit=audit,
            scope_store=scope_store,
            verifier=verifier,
            clock=self._clock,
        )

    async def execute(
        self,
        call: ToolCall,
        ctx: "ActionContext",
        *,
        hosts: Sequence[Host],
        manifests: Mapping[HostId, Manifest],
        grants: Mapping[HostId, Sequence[Grant]],
        online: set[str] | frozenset[str],
    ) -> Message:
        """Routes a Phase 3 `mesh__*` pipeline call to its typed handler."""
        try:
            if call.name == MESH_CLIPBOARD_TRANSFER_FUNCTION:
                return await self._run_clipboard_transfer(
                    call, ctx, hosts=hosts, manifests=manifests, grants=grants
                )
            if call.name == MESH_TELEPORT_FUNCTION:
                return await self._run_teleport(
                    call, ctx, hosts=hosts, manifests=manifests, grants=grants
                )
            if call.name == MESH_SMART_PAUSE_MEDIA_FUNCTION:
                return await self._run_smart_pause_media(
                    call, ctx, hosts=hosts, manifests=manifests, grants=grants, online=online
                )
            if call.name == MESH_FLEET_VITALS_FUNCTION:
                return await self._run_fleet_vitals(
                    call, ctx, hosts=hosts, manifests=manifests, grants=grants, online=online
                )
            if call.name == MESH_FLEET_GIT_STATUS_FUNCTION:
                return await self._run_fleet_git_status(
                    call, ctx, hosts=hosts, manifests=manifests, grants=grants, online=online
                )
        except ValueError as err:
            code = str(err)
            if code not in ("invalid_args", "url_invalid"):
                raise
            return Message(
                role=Role.TOOL,
                content=encode_tool_payload("invalid_args", {"errorCode": code}),
                tool_call_id=call.id,
            )
        return Message(
            role=Role.TOOL,
            content=encode_tool_payload("unknown_tool"),
            tool_call_id=call.id,
        )

    async def _load_host_details(
        self,
        host: Host,
        manifests: Mapping[HostId, Manifest],
        grants: Mapping[HostId, Sequence[Grant]],
    ) -> tuple[Manifest | None, Sequence[Grant]]:
        manifest = manifests.get(host.host_id)
        if manifest is None:
            manifest = await self._host_repo.get_manifest(host.host_id)
        host_grants = grants.get(host.host_id)
        if host_grants is None:
            host_grants = await self._host_repo.get_grants(host.host_id)
        return manifest, host_grants

    async def _emit_offline_or_wake(
        self,
        *,
        host: Host,
        tool_name: str,
        arguments: dict[str, object],
        risk: RiskTier,
        ctx: "ActionContext",
        call_id: str,
        now: datetime,
    ) -> Message:
        cand = ScoredHostCandidate(
            host=host,
            score=0.0,
            eligible=False,
            online=False,
            unavailable_reason=CandidateUnavailableReason.HOST_OFFLINE,
            reason=f"{host.alias.value} is offline",
            wol_mac=host.wol_mac,
            wol_available=bool(host.wol_mac),
        )
        cand_wire = wire_candidate_dict(cand)
        wake_enqueued = False
        wake_queue_full = False
        if host.wol_mac and self._wake_queue is not None:
            try:
                self._wake_queue.enqueue(
                    DeferredWakeStep(
                        step_id=uuid.uuid4(),
                        owner_id=ctx.owner_id,
                        pet_id=ctx.pet_id,
                        host_id=host.host_id,
                        host_alias=host.alias.value,
                        tool=tool_name,
                        arguments=dict(arguments),
                        risk=risk,
                        created_at=now,
                        expires_at=now + WAKE_QUEUE_TTL,
                        family=tool_name.split(".", 1)[0] if "." in tool_name else "app",
                        summary=f"Resume {tool_name} on {host.alias.value}",
                        client_caps=tuple(ctx.client_caps),
                    ),
                    now,
                )
                wake_enqueued = True
            except WakeQueueFullError:
                wake_queue_full = True

        summary = (
            f"{host.alias.value} is offline; deferred wake queue is full (wake_queue_full)"
            if wake_queue_full
            else f"{host.alias.value} is offline"
        )
        await ctx.emit_event(
            ActionTargetRequiredEvent(
                invocation_id=str(uuid.uuid4()),
                tool=tool_name,
                reason_code=ActionTargetRequiredReason.HOST_OFFLINE.value,
                target_reason=summary,
                candidates=(cand_wire,),
                error_code=("wake_queue_full" if wake_queue_full else None),
            )
        )
        if wake_enqueued and host.wol_mac:
            await ctx.emit_event(
                HostWakeRequestEvent(
                    invocation_id=str(uuid.uuid4()),
                    host_id=str(host.host_id),
                    target=host.alias.value,
                    mac_address=host.wol_mac,
                )
            )
        extra: dict[str, object] = {
            "reasonCode": ActionTargetRequiredReason.HOST_OFFLINE.value,
            "targetReason": summary,
            "candidates": [cand_wire],
        }
        if wake_queue_full:
            extra["errorCode"] = "wake_queue_full"
        return Message(
            role=Role.TOOL,
            content=encode_tool_payload(
                ActionTargetRequiredReason.HOST_OFFLINE.value,
                {"errorCode": "wake_queue_full"} if wake_queue_full else None,
                **extra,
            ),
            tool_call_id=call_id,
        )

    async def _resolve_named_host(
        self,
        *,
        query: object,
        active_hosts: Sequence[Host],
        tool_name: str,
        arguments: dict[str, object],
        risk: RiskTier,
        ctx: "ActionContext",
        call_id: str,
        now: datetime,
    ) -> tuple[Host | None, Message | None]:
        if not isinstance(query, str) or not query.strip():
            return None, await self._emit_ask_target(
                active_hosts=active_hosts,
                tool_name=tool_name,
                summary="Please specify source and target hosts",
                ctx=ctx,
                call_id=call_id,
            )
        match = match_hosts_by_query(query.strip(), active_hosts)
        if match.status == "unique" and match.host is not None:
            host = match.host
            if not _is_host_online(self._hub, host, ctx.self_host_id):
                return None, await self._emit_offline_or_wake(
                    host=host,
                    tool_name=tool_name,
                    arguments=arguments,
                    risk=risk,
                    ctx=ctx,
                    call_id=call_id,
                    now=now,
                )
            return host, None
        candidates = match.candidates or active_hosts
        return None, await self._emit_ask_target(
            active_hosts=candidates,
            tool_name=tool_name,
            summary=f"Target host '{query.strip()[:32]}' is ambiguous or unmatched",
            ctx=ctx,
            call_id=call_id,
        )

    async def _emit_ask_target(
        self,
        *,
        active_hosts: Sequence[Host],
        tool_name: str,
        summary: str,
        ctx: "ActionContext",
        call_id: str,
    ) -> Message:
        cands_wire = tuple(
            wire_candidate_dict(
                ScoredHostCandidate(
                    host=h,
                    score=0.5,
                    eligible=_is_host_online(self._hub, h, ctx.self_host_id),
                    online=_is_host_online(self._hub, h, ctx.self_host_id),
                    unavailable_reason=(
                        None
                        if _is_host_online(self._hub, h, ctx.self_host_id)
                        else CandidateUnavailableReason.HOST_OFFLINE
                    ),
                    reason=f"{h.alias.value}",
                    wol_mac=h.wol_mac,
                    wol_available=bool(h.wol_mac),
                )
            )
            for h in active_hosts[:5]
        )
        await ctx.emit_event(
            ActionTargetRequiredEvent(
                invocation_id=str(uuid.uuid4()),
                tool=tool_name,
                reason_code=ActionTargetRequiredReason.ASK_TARGET.value,
                target_reason=summary,
                candidates=cands_wire,
            )
        )
        return Message(
            role=Role.TOOL,
            content=encode_tool_payload(
                ActionTargetRequiredReason.ASK_TARGET.value,
                None,
                choices=[h.alias.value for h in active_hosts[:5]],
                reasonCode=ActionTargetRequiredReason.ASK_TARGET.value,
                targetReason=summary,
                candidates=list(cands_wire),
            ),
            tool_call_id=call_id,
        )

    def _check_descriptor_access(
        self,
        descriptor: ToolDescriptor | None,
        grants: Sequence[Grant],
        now: datetime,
        *,
        host_screen_consent: bool,
    ) -> str | None:
        if descriptor is None or descriptor.permission == ToolPermissionState.UNSUPPORTED:
            return "unsupported"
        if is_tool_granted(descriptor, grants, now, host_screen_consent=host_screen_consent):
            return None
        needs_u5 = descriptor.observes or descriptor.family in (
            ToolFamily.CU,
            ToolFamily.SCREEN,
            ToolFamily.UI,
            ToolFamily.INPUT,
        )
        return "u5_consent_required" if needs_u5 and not host_screen_consent else "grant_denied"

    async def _invoke_host_tool(
        self,
        *,
        host: Host,
        tool_name: str,
        arguments: dict[str, object],
        ctx: "ActionContext",
        approval_req: ApprovalRequest | None = None,
    ) -> tuple[str, dict[str, object], str | None]:
        invocation_id = str(uuid.uuid4())
        if ctx.self_host_id is not None and str(host.host_id) == ctx.self_host_id:
            self._local_result_broker.register(invocation_id, ctx.owner_id)
            await ctx.emit_event(
                LocalActionRequestEvent(
                    invocation_id=invocation_id,
                    approval_id=(
                        str(approval_req.approval_id) if approval_req is not None else None
                    ),
                    target=host.alias.value,
                    tool=tool_name,
                    arguments=dict(arguments),
                )
            )
            report = await self._local_result_broker.wait_report(
                invocation_id,
                timeout_s=DEADLINE_SECONDS,
            )
            sc = dict(report.output) if isinstance(report.output, dict) else {}
            err = report.error_code
            if err and "errorCode" not in sc:
                sc["errorCode"] = err
            return _normalize_xtp_status(report.status), sc, err

        meta: dict[str, object] = {
            "xuna/invocationId": invocation_id,
            "xuna/deadlineMs": DEADLINE_MS,
        }
        if approval_req is not None:
            app_meta: dict[str, object] = {
                "approvalId": str(approval_req.approval_id),
                "argsSha256": approval_req.args_sha256,
                "expiresAt": approval_req.expires_at.astimezone(UTC).isoformat(),
                "scope": approval_req.scope.value,
            }
            if approval_req.signature is not None:
                app_meta["signature"] = base64.b64encode(approval_req.signature).decode("ascii")
            if approval_req.signer_public_key is not None:
                app_meta["publicKey"] = base64.b64encode(approval_req.signer_public_key).decode(
                    "ascii"
                )
            meta["xuna/approval"] = app_meta

        try:
            raw_result = await self._hub.send_request(
                str(host.host_id),
                "tools/call",
                {"name": tool_name, "arguments": dict(arguments), "_meta": meta},
                timeout_s=DEADLINE_SECONDS,
            )
        except TimeoutError:
            return "timeout", {"errorCode": "timeout"}, "timeout"
        except RuntimeError:
            return "failed", {"errorCode": "rpc_error"}, "rpc_error"

        status = "failed"
        err_code: str | None = None
        sc_out: dict[str, object] = {}
        if isinstance(raw_result, Mapping):
            result_meta = raw_result.get("_meta")
            if isinstance(result_meta, Mapping):
                status = _normalize_xtp_status(result_meta.get("xuna/status"))
                if isinstance(result_meta.get("xuna/errorCode"), str):
                    err_code = str(result_meta["xuna/errorCode"])[:64]
            sc_raw = raw_result.get("structuredContent")
            if isinstance(sc_raw, Mapping):
                sc_out = dict(sc_raw)
        if structured_content_exceeds_limit(sc_out, allow_media=False):
            return (
                "failed",
                {"errorCode": "structured_content_too_large"},
                "structured_content_too_large",
            )
        if err_code is not None and "errorCode" not in sc_out:
            sc_out["errorCode"] = err_code
        return status, sc_out, err_code

    async def _try_authorize_r2_without_prompt(
        self,
        *,
        ctx: "ActionContext",
        host: Host,
        tool: str,
        arguments: dict[str, object],
        now: datetime,
    ) -> ApprovalRequest | None:
        digest = args_sha256(arguments)
        expires_at = now + timedelta(seconds=APPROVAL_TTL_SECONDS)
        if self._standing_repo is not None:
            standing_hit = await self._standing_repo.consume_match(
                ctx.owner_id, host.host_id, tool, digest, now
            )
            if standing_hit is not None:
                approved_req = ApprovalRequest(
                    approval_id=uuid.uuid4(),
                    owner_id=ctx.owner_id,
                    host_id=host.host_id,
                    target_alias=host.alias.value[:32],
                    tool=tool,
                    args_sha256=digest,
                    risk=RiskTier.R2,
                    summary=f"Standing approval for {tool} on {host.alias.value}"[:240],
                    created_at=now,
                    expires_at=expires_at,
                    state=ApprovalState.APPROVED,
                    decided_at=now,
                    scope=ApprovalScope.ONCE,
                )
                await self._approval_repo.save(approved_req)
                await self._record_pipeline_audit(
                    owner_id=ctx.owner_id,
                    action=AuditAction.HOST_PREAPPROVAL_USED,
                    now=now,
                    subject=str(host.host_id),
                    request_id=ctx.request_id,
                    metadata={
                        "approvalId": str(standing_hit.approval_id),
                        "tool": tool,
                        "risk": RiskTier.R2.value,
                        "uses": standing_hit.uses,
                    },
                )
                return approved_req
        if self._scope_store is not None:
            scope_res = self._scope_store.has_valid_scope(
                ctx.owner_id, ctx.pet_id, host.host_id, tool, now
            )
            has_scope = await scope_res if inspect.isawaitable(scope_res) else bool(scope_res)
            if has_scope:
                approved_req = ApprovalRequest(
                    approval_id=uuid.uuid4(),
                    owner_id=ctx.owner_id,
                    host_id=host.host_id,
                    target_alias=host.alias.value[:32],
                    tool=tool,
                    args_sha256=digest,
                    risk=RiskTier.R2,
                    summary=f"Scoped approval for {tool} on {host.alias.value}"[:240],
                    created_at=now,
                    expires_at=expires_at,
                    state=ApprovalState.APPROVED,
                    decided_at=now,
                    scope=ApprovalScope.CONVERSATION_15M,
                )
                await self._approval_repo.save(approved_req)
                return approved_req
        return None

    async def _request_mission_approval(
        self,
        *,
        mission: MissionEnvelope,
        primary_host: Host,
        approval_tool: str,
        approval_args: dict[str, object],
        target_reason: str,
        ctx: "ActionContext",
        now: datetime,
        battery_advisories: tuple[str, ...] = (),
        risk_override: RiskTier | None = None,
        grant_r2_scopes: Sequence[tuple[HostId, str]] = (),
    ) -> tuple[bool, ApprovalRequest | None, str]:
        risk = risk_override if risk_override is not None else mission.highest_risk()
        if risk.level < RiskTier.R2.level:
            return True, None, "ok"

        digest = args_sha256(approval_args)
        expires_at = now + timedelta(seconds=APPROVAL_TTL_SECONDS)
        if risk == RiskTier.R2:
            pre_auth = await self._try_authorize_r2_without_prompt(
                ctx=ctx,
                host=primary_host,
                tool=approval_tool,
                arguments=approval_args,
                now=now,
            )
            if pre_auth is not None:
                return True, pre_auth, "ok"

        approval_req = ApprovalRequest(
            approval_id=uuid.uuid4(),
            owner_id=ctx.owner_id,
            host_id=primary_host.host_id,
            target_alias=primary_host.alias.value[:32],
            tool=approval_tool,
            args_sha256=digest,
            risk=risk,
            summary=mission.title[:240],
            created_at=now,
            expires_at=expires_at,
        )
        await self._approval_repo.save(approval_req)
        await self._approval_broker.create(approval_req)
        await ctx.emit_event(
            ActionPendingEvent(
                approval_id=str(approval_req.approval_id),
                target=primary_host.alias.value,
                summary=mission.title[:240],
                risk=risk.value,
                expires_in=APPROVAL_TTL_SECONDS,
                host_id=str(primary_host.host_id),
                args_sha256=digest,
                expires_at=expires_at.astimezone(UTC).isoformat(),
                mission_envelope=mission.to_wire(),
                target_reason=target_reason,
                target_platform=primary_host.platform.value,
                battery_advisories=battery_advisories,
            )
        )
        decision_state = await self._approval_broker.wait(
            approval_req.approval_id,
            timeout_s=float(APPROVAL_TTL_SECONDS),
        )
        await self._approval_service.sync_approval_after_wait(
            approval_req, decision_state, ctx.request_id
        )
        if decision_state != ApprovalState.APPROVED:
            final_status = "rejected" if decision_state == ApprovalState.REJECTED else "expired"
            return False, None, final_status

        slot_req = self._approval_broker.get_slot_request(approval_req.approval_id)
        if slot_req is not None:
            approval_req = slot_req
        else:
            stored_req = await self._approval_repo.get(approval_req.approval_id)
            if stored_req is not None:
                approval_req = stored_req
        if approval_req.args_sha256 != digest:
            return False, None, "denied"
        if (
            approval_req.scope == ApprovalScope.CONVERSATION_15M
            and risk == RiskTier.R2
            and self._scope_store is not None
        ):
            scope_targets = [(primary_host.host_id, approval_tool), *grant_r2_scopes]
            seen_scopes: set[tuple[HostId, str]] = set()
            grant_now = self._clock()
            for s_host_id, s_tool in scope_targets:
                if (s_host_id, s_tool) in seen_scopes:
                    continue
                seen_scopes.add((s_host_id, s_tool))
                grant_res = self._scope_store.grant_scope(
                    ctx.owner_id, ctx.pet_id, s_host_id, s_tool, grant_now
                )
                if inspect.isawaitable(grant_res):
                    await grant_res
        return True, approval_req, "ok"

    async def _record_pipeline_audit(
        self,
        *,
        owner_id: uuid.UUID,
        action: AuditAction,
        now: datetime,
        subject: str,
        request_id: str,
        metadata: dict[str, str | int | bool],
    ) -> None:
        await self._audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=action,
                occurred_at=now,
                subject=subject[:64],
                request_id=request_id[:64],
                metadata=metadata,
            )
        )

    async def _run_clipboard_transfer(
        self,
        call: ToolCall,
        ctx: "ActionContext",
        *,
        hosts: Sequence[Host],
        manifests: Mapping[HostId, Manifest],
        grants: Mapping[HostId, Sequence[Grant]],
    ) -> Message:
        now = self._clock()
        args = dict(call.arguments or {})
        allowed_keys = frozenset({"sourceHost", "targetHost", "mode", "text"})
        if not set(args).issubset(allowed_keys):
            raise ValueError("invalid_args")

        mode = args.get("mode", "selection")
        if mode not in ("selection", "clipboard"):
            raise ValueError("invalid_args")

        inline_text = args.get("text")
        prepared_inline = (
            validate_clipboard_payload(inline_text, max_chars=MAX_CLIPBOARD_WRITE_CHARS)
            if inline_text is not None
            else None
        )
        wake_tool = (
            _CLIPBOARD_WRITE_TOOL
            if prepared_inline is not None
            else MESH_CLIPBOARD_TRANSFER_FUNCTION
        )
        wake_args: dict[str, object] = (
            {"text": prepared_inline.text} if prepared_inline is not None else dict(args)
        )

        active_hosts = [h for h in hosts if h.status == HostStatus.ACTIVE]
        target_host, target_err = await self._resolve_named_host(
            query=args.get("targetHost"),
            active_hosts=active_hosts,
            tool_name=wake_tool,
            arguments=wake_args,
            risk=RiskTier.R2,
            ctx=ctx,
            call_id=call.id,
            now=now,
        )
        if target_err is not None or target_host is None:
            return target_err or Message(
                role=Role.TOOL, content=encode_tool_payload("no_host"), tool_call_id=call.id
            )

        target_manifest, target_grants = await self._load_host_details(
            target_host, manifests, grants
        )
        is_target_self = (
            ctx.self_host_id is not None and str(target_host.host_id) == ctx.self_host_id
        )
        write_desc = target_manifest.find_tool("clipboard.write") if target_manifest else None
        if write_desc is None and is_target_self:
            write_desc = ToolDescriptor(
                name="clipboard.write",
                description="Write text to local device clipboard.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string", "maxLength": 4000}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                family=ToolFamily.APP,
                risk=RiskTier.R2,
                observes=False,
                reversible=False,
            )
        target_deny = self._check_descriptor_access(
            write_desc, target_grants, now, host_screen_consent=ctx.host_screen_consent
        )
        if target_deny is not None:
            await self._record_pipeline_audit(
                owner_id=ctx.owner_id,
                action=AuditAction.HOST_TOOL_DENIED,
                now=now,
                subject=str(target_host.host_id),
                request_id=ctx.request_id,
                metadata={"tool": _CLIPBOARD_WRITE_TOOL, "status": "denied", "reason": target_deny},
            )
            return Message(
                role=Role.TOOL,
                content=encode_tool_payload("denied", {"errorCode": target_deny}),
                tool_call_id=call.id,
            )

        write_risk = write_desc.risk if write_desc is not None else RiskTier.R2
        source_host: Host | None = None
        read_tool_name = "selection.read" if mode == "selection" else "clipboard.read"
        read_risk = RiskTier.R1

        if prepared_inline is not None:
            prepared = prepared_inline
            redacted_total = prepared.redacted_count
            mission_steps = [
                MissionStep(
                    host_id=str(target_host.host_id),
                    host_alias=target_host.alias.value,
                    platform=target_host.platform,
                    tool="clipboard.write",
                    summary=f"Write redacted text to {target_host.alias.value}",
                    risk=write_risk,
                    status=MissionStepStatus.PENDING,
                )
            ]
            target_reason = f"Write redacted clipboard text on {target_host.alias.value}"
            write_step_idx = 0
        else:
            source_host, source_err = await self._resolve_named_host(
                query=args.get("sourceHost"),
                active_hosts=active_hosts,
                tool_name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments=dict(args),
                risk=RiskTier.R2,
                ctx=ctx,
                call_id=call.id,
                now=now,
            )
            if source_err is not None or source_host is None:
                return source_err or Message(
                    role=Role.TOOL, content=encode_tool_payload("no_host"), tool_call_id=call.id
                )
            src_manifest, src_grants = await self._load_host_details(source_host, manifests, grants)
            is_src_self = (
                ctx.self_host_id is not None and str(source_host.host_id) == ctx.self_host_id
            )
            if mode == "selection" and src_manifest and src_manifest.find_tool("selection.read"):
                read_tool_name = "selection.read"
            elif (src_manifest and src_manifest.find_tool("clipboard.read")) or is_src_self:
                read_tool_name = "clipboard.read"

            read_desc = src_manifest.find_tool(read_tool_name) if src_manifest else None
            if read_desc is None and is_src_self:
                read_desc = ToolDescriptor(
                    name="clipboard.read",
                    description="Read text from local device clipboard.",
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    family=ToolFamily.APP,
                    risk=RiskTier.R2,
                    observes=True,
                    reversible=True,
                )
            src_deny = self._check_descriptor_access(
                read_desc, src_grants, now, host_screen_consent=ctx.host_screen_consent
            )
            if src_deny is not None or read_desc is None:
                await self._record_pipeline_audit(
                    owner_id=ctx.owner_id,
                    action=AuditAction.HOST_TOOL_DENIED,
                    now=now,
                    subject=str(source_host.host_id),
                    request_id=ctx.request_id,
                    metadata={
                        "tool": read_tool_name,
                        "status": "denied",
                        "reason": src_deny or "unsupported",
                    },
                )
                return Message(
                    role=Role.TOOL,
                    content=encode_tool_payload("denied", {"errorCode": src_deny or "unsupported"}),
                    tool_call_id=call.id,
                )
            read_risk = read_desc.risk
            read_args: dict[str, object] = (
                {"maxChars": MAX_SELECTION_CHARS} if read_tool_name == "selection.read" else {}
            )
            target_reason = (
                f"Transfer clipboard from {source_host.alias.value} to {target_host.alias.value}"
            )
            read_approval_req: ApprovalRequest | None = None
            if read_risk.level >= RiskTier.R2.level:
                planned_read_mission = MissionEnvelope(
                    mission_id=str(uuid.uuid4()),
                    title=target_reason[:120],
                    goal=target_reason,
                    steps=(
                        MissionStep(
                            host_id=str(source_host.host_id),
                            host_alias=source_host.alias.value,
                            platform=source_host.platform,
                            tool=read_tool_name,
                            summary=f"Read clipboard on {source_host.alias.value}",
                            risk=read_risk,
                            status=MissionStepStatus.PENDING,
                        ),
                        MissionStep(
                            host_id=str(target_host.host_id),
                            host_alias=target_host.alias.value,
                            platform=target_host.platform,
                            tool="clipboard.write",
                            summary=f"Write redacted text to {target_host.alias.value}",
                            risk=write_risk,
                            status=MissionStepStatus.PENDING,
                        ),
                    ),
                )
                planned_highest_risk = planned_read_mission.highest_risk()
                eff_read_risk = (
                    planned_highest_risk
                    if planned_highest_risk.level > read_risk.level
                    else read_risk
                )
                read_ok, read_approval_req, read_app_status = await self._request_mission_approval(
                    mission=planned_read_mission,
                    primary_host=source_host,
                    approval_tool=read_tool_name,
                    approval_args=read_args,
                    target_reason=target_reason,
                    ctx=ctx,
                    now=now,
                    risk_override=eff_read_risk,
                )
                if not read_ok:
                    denied_read_mission = planned_read_mission.with_step_status(
                        0, MissionStepStatus.DENIED
                    )
                    await ctx.emit_event(
                        ActionResultEvent(
                            invocation_id=str(uuid.uuid4()),
                            target=source_host.alias.value,
                            summary=f"Clipboard read {read_app_status}",
                            status=read_app_status,
                            gesture=gesture_for_action_status(read_app_status),
                            host_id=str(source_host.host_id),
                            target_reason=target_reason,
                            target_platform=source_host.platform.value,
                            mission_envelope=denied_read_mission.to_wire(),
                        )
                    )
                    return Message(
                        role=Role.TOOL,
                        content=encode_tool_payload(read_app_status),
                        tool_call_id=call.id,
                    )

            read_status, read_sc, read_err = await self._invoke_host_tool(
                host=source_host,
                tool_name=read_tool_name,
                arguments=read_args,
                ctx=ctx,
                approval_req=read_approval_req,
            )
            if read_status != "ok":
                raw_err_code = read_err or (
                    read_sc.get("errorCode") if isinstance(read_sc.get("errorCode"), str) else None
                )
                safe_read_err = (
                    redact_sensitive_text(str(raw_err_code)).text[:64] if raw_err_code else None
                )
                safe_read_err_out: dict[str, object] = (
                    {"errorCode": safe_read_err} if safe_read_err else {}
                )
                await ctx.emit_event(
                    ActionResultEvent(
                        invocation_id=str(uuid.uuid4()),
                        target=source_host.alias.value,
                        summary=f"Failed reading text on {source_host.alias.value}",
                        status=read_status,
                        gesture=gesture_for_action_status(read_status),
                        error_code=safe_read_err,
                        host_id=str(source_host.host_id),
                        target_reason=f"Source read on {source_host.alias.value}",
                        target_platform=source_host.platform.value,
                    )
                )
                return Message(
                    role=Role.TOOL,
                    content=encode_tool_payload(read_status, safe_read_err_out or None),
                    tool_call_id=call.id,
                )

            try:
                prepared = prepare_selection_for_clipboard(
                    read_sc.get("text"), max_chars=MAX_SELECTION_CHARS
                )
            except ValueError as err:
                err_code = (
                    str(err) if str(err) in ("invalid_args", "url_invalid") else "invalid_args"
                )
                failed_read_mission = MissionEnvelope(
                    mission_id=str(uuid.uuid4()),
                    title=target_reason[:120],
                    goal=target_reason,
                    steps=(
                        MissionStep(
                            host_id=str(source_host.host_id),
                            host_alias=source_host.alias.value,
                            platform=source_host.platform,
                            tool=read_tool_name,
                            summary=f"Invalid clipboard text on {source_host.alias.value}",
                            risk=read_risk,
                            status=MissionStepStatus.FAILED,
                        ),
                        MissionStep(
                            host_id=str(target_host.host_id),
                            host_alias=target_host.alias.value,
                            platform=target_host.platform,
                            tool="clipboard.write",
                            summary=f"Write redacted text to {target_host.alias.value}",
                            risk=write_risk,
                            status=MissionStepStatus.PENDING,
                        ),
                    ),
                )
                await ctx.emit_event(
                    ActionResultEvent(
                        invocation_id=str(uuid.uuid4()),
                        target=source_host.alias.value,
                        summary=f"Invalid text read on {source_host.alias.value}",
                        status="failed",
                        gesture=gesture_for_action_status("failed"),
                        error_code=err_code,
                        host_id=str(source_host.host_id),
                        target_reason=target_reason,
                        target_platform=source_host.platform.value,
                        mission_envelope=failed_read_mission.to_wire(),
                    )
                )
                return Message(
                    role=Role.TOOL,
                    content=encode_tool_payload("invalid_args", {"errorCode": err_code}),
                    tool_call_id=call.id,
                )
            redacted_total = prepared.redacted_count
            mission_steps = [
                MissionStep(
                    host_id=str(source_host.host_id),
                    host_alias=source_host.alias.value,
                    platform=source_host.platform,
                    tool=read_tool_name,
                    summary=f"Read selection on {source_host.alias.value}",
                    risk=read_risk,
                    status=MissionStepStatus.OK,
                ),
                MissionStep(
                    host_id=str(target_host.host_id),
                    host_alias=target_host.alias.value,
                    platform=target_host.platform,
                    tool="clipboard.write",
                    summary=f"Write redacted text to {target_host.alias.value}",
                    risk=write_risk,
                    status=MissionStepStatus.PENDING,
                ),
            ]
            write_step_idx = 1

        mission = MissionEnvelope(
            mission_id=str(uuid.uuid4()),
            title=target_reason[:120],
            goal=target_reason,
            steps=tuple(mission_steps),
        )
        # Re-validate and redact a second time before approval binding & write dispatch.
        try:
            final_payload = validate_clipboard_payload(
                prepared.text, max_chars=MAX_CLIPBOARD_WRITE_CHARS
            )
        except ValueError as err:
            err_code = str(err) if str(err) in ("invalid_args", "url_invalid") else "invalid_args"
            failed_mission = mission.with_step_status(write_step_idx, MissionStepStatus.FAILED)
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=str(uuid.uuid4()),
                    target=target_host.alias.value,
                    summary=f"Invalid clipboard payload for {target_host.alias.value}",
                    status="failed",
                    gesture=gesture_for_action_status("failed"),
                    error_code=err_code,
                    host_id=str(target_host.host_id),
                    target_reason=target_reason,
                    target_platform=target_host.platform.value,
                    mission_envelope=failed_mission.to_wire(),
                )
            )
            return Message(
                role=Role.TOOL,
                content=encode_tool_payload("invalid_args", {"errorCode": err_code}),
                tool_call_id=call.id,
            )
        redacted_total += final_payload.redacted_count
        write_args: dict[str, object] = {"text": final_payload.text}

        mission_highest_risk = mission.highest_risk()
        eff_write_risk = (
            mission_highest_risk
            if write_risk.level >= RiskTier.R2.level
            and mission_highest_risk.level > write_risk.level
            else write_risk
        )
        approved, approval_req, app_status = await self._request_mission_approval(
            mission=mission,
            primary_host=target_host,
            approval_tool="clipboard.write",
            approval_args=write_args,
            target_reason=target_reason,
            ctx=ctx,
            now=now,
            risk_override=eff_write_risk,
        )
        if not approved:
            failed_mission = mission.with_step_status(write_step_idx, MissionStepStatus.DENIED)
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=str(uuid.uuid4()),
                    target=target_host.alias.value,
                    summary=f"Clipboard transfer {app_status}",
                    status=app_status,
                    gesture=gesture_for_action_status(app_status),
                    host_id=str(target_host.host_id),
                    target_reason=target_reason,
                    target_platform=target_host.platform.value,
                    mission_envelope=failed_mission.to_wire(),
                )
            )
            return Message(
                role=Role.TOOL,
                content=encode_tool_payload(app_status),
                tool_call_id=call.id,
            )

        write_status, write_sc, write_err = await self._invoke_host_tool(
            host=target_host,
            tool_name="clipboard.write",
            arguments=write_args,
            ctx=ctx,
            approval_req=approval_req,
        )
        step_status = MissionStepStatus.OK if write_status == "ok" else MissionStepStatus.FAILED
        done_mission = mission.with_step_status(write_step_idx, step_status)
        await self._record_pipeline_audit(
            owner_id=ctx.owner_id,
            action=AuditAction.HOST_TOOL_INVOKED,
            now=now,
            subject=str(target_host.host_id),
            request_id=ctx.request_id,
            metadata={
                "tool": _CLIPBOARD_WRITE_TOOL,
                "status": write_status,
                "charCount": final_payload.char_count,
                "redactedCount": redacted_total,
            },
        )
        await ctx.emit_event(
            ActionResultEvent(
                invocation_id=str(uuid.uuid4()),
                target=target_host.alias.value,
                summary=f"Clipboard synced to {target_host.alias.value}",
                status=write_status,
                gesture=gesture_for_action_status(write_status),
                error_code=write_err,
                host_id=str(target_host.host_id),
                target_reason=target_reason,
                target_platform=target_host.platform.value,
                mission_envelope=done_mission.to_wire(),
            )
        )
        return Message(
            role=Role.TOOL,
            content=encode_tool_payload(
                write_status,
                {
                    "charCount": final_payload.char_count,
                    "redactedCount": redacted_total,
                    "truncated": prepared.truncated,
                    "targetHost": target_host.alias.value,
                    **({"errorCode": write_err} if write_err else {}),
                    **({"sourceHost": source_host.alias.value} if source_host is not None else {}),
                },
            ),
            tool_call_id=call.id,
        )

    async def _run_teleport(
        self,
        call: ToolCall,
        ctx: "ActionContext",
        *,
        hosts: Sequence[Host],
        manifests: Mapping[HostId, Manifest],
        grants: Mapping[HostId, Sequence[Grant]],
    ) -> Message:
        now = self._clock()
        args = dict(call.arguments or {})
        allowed_keys = frozenset({"sourceHost", "targetHost", "kind", "project", "openInAppId"})
        if not set(args).issubset(allowed_keys):
            raise ValueError("invalid_args")

        kind = args.get("kind", "auto")
        if kind not in ("auto", "url", "workspace"):
            raise ValueError("invalid_args")

        project_arg = _sanitize_alias_token(args.get("project")) if args.get("project") else None
        if args.get("project") is not None and project_arg is None:
            raise ValueError("invalid_args")

        open_app_arg = (
            _sanitize_alias_token(args.get("openInAppId")) if args.get("openInAppId") else None
        )
        if args.get("openInAppId") is not None and open_app_arg is None:
            raise ValueError("invalid_args")

        active_hosts = [h for h in hosts if h.status == HostStatus.ACTIVE]
        source_host, src_err = await self._resolve_named_host(
            query=args.get("sourceHost"),
            active_hosts=active_hosts,
            tool_name=MESH_TELEPORT_FUNCTION,
            arguments=dict(args),
            risk=RiskTier.R2,
            ctx=ctx,
            call_id=call.id,
            now=now,
        )
        if src_err is not None or source_host is None:
            return src_err or Message(
                role=Role.TOOL, content=encode_tool_payload("no_host"), tool_call_id=call.id
            )

        target_query = args.get("targetHost")
        if target_query is None and ctx.pinned_host_id is not None:
            pinned = next(
                (h for h in active_hosts if h.host_id == ctx.pinned_host_id),
                None,
            )
            if pinned is not None and pinned.host_id != source_host.host_id:
                target_query = pinned.alias.value

        target_host, dst_err = await self._resolve_named_host(
            query=target_query,
            active_hosts=active_hosts,
            tool_name=MESH_TELEPORT_FUNCTION,
            arguments=dict(args),
            risk=RiskTier.R2,
            ctx=ctx,
            call_id=call.id,
            now=now,
        )
        if dst_err is not None or target_host is None:
            return dst_err or Message(
                role=Role.TOOL, content=encode_tool_payload("no_host"), tool_call_id=call.id
            )
        if source_host.host_id == target_host.host_id:
            raise ValueError("invalid_args")

        src_manifest, src_grants = await self._load_host_details(source_host, manifests, grants)
        dst_manifest, dst_grants = await self._load_host_details(target_host, manifests, grants)

        url_src_desc = src_manifest.find_tool("browser.current_url") if src_manifest else None
        url_dst_desc = dst_manifest.find_tool("url.open") if dst_manifest else None
        app_dst_desc = dst_manifest.find_tool("app.launch") if dst_manifest else None
        ws_src_desc = src_manifest.find_tool("workspace.capture_capsule") if src_manifest else None
        ws_open_desc = dst_manifest.find_tool("workspace.open") if dst_manifest else None
        ws_restore_desc = (
            dst_manifest.find_tool("workspace.restore_capsule") if dst_manifest else None
        )
        host_notify_desc = dst_manifest.find_tool("host.notify") if dst_manifest else None

        url_src_ok = (
            self._check_descriptor_access(
                url_src_desc, src_grants, now, host_screen_consent=ctx.host_screen_consent
            )
            is None
        )
        url_dst_ok = (
            self._check_descriptor_access(
                url_dst_desc, dst_grants, now, host_screen_consent=ctx.host_screen_consent
            )
            is None
        )
        app_dst_ok = (
            self._check_descriptor_access(
                app_dst_desc, dst_grants, now, host_screen_consent=ctx.host_screen_consent
            )
            is None
        )
        ws_src_ok = (
            self._check_descriptor_access(
                ws_src_desc, src_grants, now, host_screen_consent=ctx.host_screen_consent
            )
            is None
        )
        ws_open_ok = (
            self._check_descriptor_access(
                ws_open_desc, dst_grants, now, host_screen_consent=ctx.host_screen_consent
            )
            is None
        )
        ws_restore_ok = (
            self._check_descriptor_access(
                ws_restore_desc, dst_grants, now, host_screen_consent=ctx.host_screen_consent
            )
            is None
        )
        host_notify_ok = (
            self._check_descriptor_access(
                host_notify_desc, dst_grants, now, host_screen_consent=ctx.host_screen_consent
            )
            is None
        )

        want_url = (
            kind in ("auto", "url")
            and url_src_ok
            and (url_dst_ok or (open_app_arg is not None and app_dst_ok))
        )
        src_twin = _safe_get_twin(self._hub, str(source_host.host_id))
        has_indexed_ws = bool(
            src_twin is not None and src_twin.resources and src_twin.resources.workspaces
        )
        can_exec_ws_target = (
            (open_app_arg is not None and app_dst_ok and not want_url)
            or ws_open_ok
            or ws_restore_ok
        )
        want_ws = (
            ws_src_ok
            and can_exec_ws_target
            and (
                kind == "workspace"
                or project_arg is not None
                or (kind == "auto" and (has_indexed_ws or not want_url))
            )
        )

        if not want_url and not want_ws:
            return Message(
                role=Role.TOOL,
                content=encode_tool_payload("denied", {"errorCode": "grant_denied"}),
                tool_call_id=call.id,
            )

        target_reason = (
            f"Teleport session from {source_host.alias.value} to {target_host.alias.value}"
        )
        url_src_risk = url_src_desc.risk if url_src_desc else RiskTier.R2
        ws_src_risk = ws_src_desc.risk if ws_src_desc else RiskTier.R1
        planned_url_tool = "app.launch" if (open_app_arg and app_dst_ok) else "url.open"
        planned_url_risk = (
            (app_dst_desc.risk if app_dst_desc else RiskTier.R2)
            if planned_url_tool == "app.launch"
            else (url_dst_desc.risk if url_dst_desc else RiskTier.R2)
        )
        planned_ws_tool: str | None = None
        planned_ws_risk = RiskTier.R2
        if want_ws:
            if open_app_arg and app_dst_ok and not want_url:
                planned_ws_tool = "app.launch"
                planned_ws_risk = app_dst_desc.risk if app_dst_desc else RiskTier.R2
            elif ws_open_ok:
                planned_ws_tool = "workspace.open"
                planned_ws_risk = ws_open_desc.risk if ws_open_desc else RiskTier.R2
            elif ws_restore_ok:
                planned_ws_tool = "workspace.restore_capsule"
                planned_ws_risk = ws_restore_desc.risk if ws_restore_desc else RiskTier.R2

        planned_steps: list[MissionStep] = []
        planned_apps = [open_app_arg] if (open_app_arg and app_dst_ok) else []
        if want_url:
            planned_steps.append(
                MissionStep(
                    host_id=str(source_host.host_id),
                    host_alias=source_host.alias.value,
                    platform=source_host.platform,
                    tool="browser.current_url",
                    summary=f"Read active URL on {source_host.alias.value}",
                    risk=url_src_risk,
                    status=MissionStepStatus.PENDING,
                )
            )
            planned_steps.append(
                MissionStep(
                    host_id=str(target_host.host_id),
                    host_alias=target_host.alias.value,
                    platform=target_host.platform,
                    tool=planned_url_tool,
                    summary=f"Open URL on {target_host.alias.value}",
                    risk=planned_url_risk,
                    status=MissionStepStatus.PENDING,
                )
            )
        if want_ws and planned_ws_tool is not None:
            planned_steps.append(
                MissionStep(
                    host_id=str(source_host.host_id),
                    host_alias=source_host.alias.value,
                    platform=source_host.platform,
                    tool="workspace.capture_capsule",
                    summary=f"Capture workspace on {source_host.alias.value}",
                    risk=ws_src_risk,
                    status=MissionStepStatus.PENDING,
                )
            )
            planned_steps.append(
                MissionStep(
                    host_id=str(target_host.host_id),
                    host_alias=target_host.alias.value,
                    platform=target_host.platform,
                    tool=planned_ws_tool,
                    summary=f"Restore workspace on {target_host.alias.value}",
                    risk=planned_ws_risk,
                    status=MissionStepStatus.PENDING,
                )
            )
        planned_mission = MissionEnvelope(
            mission_id=str(uuid.uuid4()),
            title=target_reason[:120],
            goal=target_reason,
            steps=tuple(planned_steps),
            apps=tuple(planned_apps),
        )
        mission_highest_risk = planned_mission.highest_risk()
        eff_url_src_risk = (
            mission_highest_risk
            if url_src_risk.level >= RiskTier.R2.level
            and mission_highest_risk.level > url_src_risk.level
            else url_src_risk
        )
        eff_ws_src_risk = (
            mission_highest_risk
            if ws_src_risk.level >= RiskTier.R2.level
            and mission_highest_risk.level > ws_src_risk.level
            else ws_src_risk
        )

        steps: list[MissionStep] = []
        target_actions: list[tuple[str, dict[str, object], RiskTier, int]] = []
        domains: list[str] = []
        apps: list[str] = []
        upfront_r2_approval: ApprovalRequest | None = None

        if want_url:
            url_read_approval: ApprovalRequest | None = None
            if eff_url_src_risk.level >= RiskTier.R2.level:
                if eff_url_src_risk == RiskTier.R2:
                    url_read_approval = await self._try_authorize_r2_without_prompt(
                        ctx=ctx,
                        host=source_host,
                        tool="browser.current_url",
                        arguments={},
                        now=now,
                    )
                    if url_read_approval is None:
                        r2_scope_targets: list[tuple[HostId, str]] = [
                            (target_host.host_id, planned_url_tool)
                        ]
                        if want_ws and planned_ws_tool is not None:
                            r2_scope_targets.append((target_host.host_id, planned_ws_tool))
                        (
                            url_ok,
                            upfront_r2_approval,
                            url_app_status,
                        ) = await self._request_mission_approval(
                            mission=planned_mission,
                            primary_host=source_host,
                            approval_tool="browser.current_url",
                            approval_args={},
                            target_reason=target_reason,
                            ctx=ctx,
                            now=now,
                            risk_override=RiskTier.R2,
                            grant_r2_scopes=r2_scope_targets,
                        )
                        if not url_ok:
                            denied_m = planned_mission.with_step_status(0, MissionStepStatus.DENIED)
                            await ctx.emit_event(
                                ActionResultEvent(
                                    invocation_id=str(uuid.uuid4()),
                                    target=source_host.alias.value,
                                    summary=f"Teleport {url_app_status}",
                                    status=url_app_status,
                                    gesture=gesture_for_action_status(url_app_status),
                                    host_id=str(source_host.host_id),
                                    target_reason=target_reason,
                                    target_platform=source_host.platform.value,
                                    mission_envelope=denied_m.to_wire(),
                                )
                            )
                            return Message(
                                role=Role.TOOL,
                                content=encode_tool_payload(url_app_status),
                                tool_call_id=call.id,
                            )
                        url_read_approval = upfront_r2_approval
                else:
                    (
                        url_ok,
                        url_read_approval,
                        url_app_status,
                    ) = await self._request_mission_approval(
                        mission=planned_mission,
                        primary_host=source_host,
                        approval_tool="browser.current_url",
                        approval_args={},
                        target_reason=target_reason,
                        ctx=ctx,
                        now=now,
                        risk_override=eff_url_src_risk,
                    )
                    if not url_ok:
                        denied_m = planned_mission.with_step_status(0, MissionStepStatus.DENIED)
                        await ctx.emit_event(
                            ActionResultEvent(
                                invocation_id=str(uuid.uuid4()),
                                target=source_host.alias.value,
                                summary=f"Teleport {url_app_status}",
                                status=url_app_status,
                                gesture=gesture_for_action_status(url_app_status),
                                host_id=str(source_host.host_id),
                                target_reason=target_reason,
                                target_platform=source_host.platform.value,
                                mission_envelope=denied_m.to_wire(),
                            )
                        )
                        return Message(
                            role=Role.TOOL,
                            content=encode_tool_payload(url_app_status),
                            tool_call_id=call.id,
                        )

            url_status, url_sc, url_err = await self._invoke_host_tool(
                host=source_host,
                tool_name="browser.current_url",
                arguments={},
                ctx=ctx,
                approval_req=url_read_approval,
            )
            if url_status != "ok":
                raw_url_err = url_err or (
                    url_sc.get("errorCode") if isinstance(url_sc.get("errorCode"), str) else None
                )
                safe_url_err = (
                    redact_sensitive_text(str(raw_url_err)).text[:64] if raw_url_err else None
                )
                safe_url_out: dict[str, object] = (
                    {"errorCode": safe_url_err} if safe_url_err else {}
                )
                await ctx.emit_event(
                    ActionResultEvent(
                        invocation_id=str(uuid.uuid4()),
                        target=source_host.alias.value,
                        summary=f"Failed reading active URL on {source_host.alias.value}",
                        status=url_status,
                        gesture=gesture_for_action_status(url_status),
                        error_code=safe_url_err,
                        host_id=str(source_host.host_id),
                        target_reason=target_reason,
                        target_platform=source_host.platform.value,
                    )
                )
                return Message(
                    role=Role.TOOL,
                    content=encode_tool_payload(url_status, safe_url_out or None),
                    tool_call_id=call.id,
                )

            raw_url = str(url_sc.get("url") or "")
            try:
                domain = validate_teleport_https_url(raw_url)
            except ValueError:
                await ctx.emit_event(
                    ActionResultEvent(
                        invocation_id=str(uuid.uuid4()),
                        target=source_host.alias.value,
                        summary="Blocked non-HTTPS or private teleport URL",
                        status="failed",
                        gesture=gesture_for_action_status("failed"),
                        error_code="url_invalid",
                        host_id=str(source_host.host_id),
                        target_reason=target_reason,
                        target_platform=source_host.platform.value,
                    )
                )
                return Message(
                    role=Role.TOOL,
                    content=encode_tool_payload("invalid_args", {"errorCode": "url_invalid"}),
                    tool_call_id=call.id,
                )

            domains.append(domain)
            steps.append(
                MissionStep(
                    host_id=str(source_host.host_id),
                    host_alias=source_host.alias.value,
                    platform=source_host.platform,
                    tool="browser.current_url",
                    summary=f"Read active URL ({domain}) on {source_host.alias.value}",
                    risk=url_src_risk,
                    status=MissionStepStatus.OK,
                )
            )
            if open_app_arg and app_dst_ok:
                apps.append(open_app_arg)
                open_tool = "app.launch"
                open_args: dict[str, object] = {"id": open_app_arg, "openTarget": raw_url}
                open_risk = app_dst_desc.risk if app_dst_desc else RiskTier.R2
            else:
                open_tool = "url.open"
                open_args = {"url": raw_url}
                open_risk = url_dst_desc.risk if url_dst_desc else RiskTier.R2
            idx = len(steps)
            steps.append(
                MissionStep(
                    host_id=str(target_host.host_id),
                    host_alias=target_host.alias.value,
                    platform=target_host.platform,
                    tool=open_tool,
                    summary=f"Open {domain} on {target_host.alias.value}",
                    risk=open_risk,
                    status=MissionStepStatus.PENDING,
                )
            )
            target_actions.append((open_tool, open_args, open_risk, idx))

        captured_capsule: dict[str, object] = {}
        cap_proj = project_arg or "workspace"
        if want_ws:
            cap_args: dict[str, object] = {}
            if project_arg is not None:
                cap_args["project"] = project_arg
            ws_read_approval: ApprovalRequest | None = None

            async def _emit_ws_capture_denied(app_status: str) -> Message:
                denied_m = planned_mission
                if want_url and len(denied_m.steps) > 0:
                    denied_m = denied_m.with_step_status(0, MissionStepStatus.OK)
                start_rem = 1 if want_url else 0
                for rem_idx in range(start_rem, len(denied_m.steps)):
                    denied_m = denied_m.with_step_status(rem_idx, MissionStepStatus.DENIED)
                await ctx.emit_event(
                    ActionResultEvent(
                        invocation_id=str(uuid.uuid4()),
                        target=source_host.alias.value,
                        summary=f"Teleport {app_status}",
                        status=app_status,
                        gesture=gesture_for_action_status(app_status),
                        error_code=app_status,
                        host_id=str(source_host.host_id),
                        target_reason=target_reason,
                        target_platform=source_host.platform.value,
                        mission_envelope=denied_m.to_wire(),
                    )
                )
                return Message(
                    role=Role.TOOL,
                    content=encode_tool_payload(app_status),
                    tool_call_id=call.id,
                )

            if eff_ws_src_risk.level >= RiskTier.R2.level:
                if eff_ws_src_risk == RiskTier.R2:
                    if upfront_r2_approval is not None:
                        ws_read_approval = _derive_r2_step_approval(
                            upfront_r2_approval,
                            host=source_host,
                            tool="workspace.capture_capsule",
                            arguments=cap_args,
                        )
                    else:
                        ws_read_approval = await self._try_authorize_r2_without_prompt(
                            ctx=ctx,
                            host=source_host,
                            tool="workspace.capture_capsule",
                            arguments=cap_args,
                            now=now,
                        )
                    if ws_read_approval is None:
                        ws_ok, ws_auth_req, ws_app_status = await self._request_mission_approval(
                            mission=planned_mission,
                            primary_host=source_host,
                            approval_tool="workspace.capture_capsule",
                            approval_args=cap_args,
                            target_reason=target_reason,
                            ctx=ctx,
                            now=now,
                            risk_override=RiskTier.R2,
                        )
                        if not ws_ok:
                            return await _emit_ws_capture_denied(ws_app_status)
                        upfront_r2_approval = ws_auth_req
                        ws_read_approval = ws_auth_req
                else:
                    ws_ok, ws_read_approval, ws_app_status = await self._request_mission_approval(
                        mission=planned_mission,
                        primary_host=source_host,
                        approval_tool="workspace.capture_capsule",
                        approval_args=cap_args,
                        target_reason=target_reason,
                        ctx=ctx,
                        now=now,
                        risk_override=eff_ws_src_risk,
                    )
                    if not ws_ok:
                        return await _emit_ws_capture_denied(ws_app_status)

            ws_status, ws_sc, ws_err = await self._invoke_host_tool(
                host=source_host,
                tool_name="workspace.capture_capsule",
                arguments=cap_args,
                ctx=ctx,
                approval_req=ws_read_approval,
            )
            if ws_status != "ok" and not want_url:
                raw_ws_err = ws_err or (
                    ws_sc.get("errorCode") if isinstance(ws_sc.get("errorCode"), str) else None
                )
                safe_ws_err = (
                    redact_sensitive_text(str(raw_ws_err)).text[:64] if raw_ws_err else None
                )
                safe_ws_out: dict[str, object] = {"errorCode": safe_ws_err} if safe_ws_err else {}
                await ctx.emit_event(
                    ActionResultEvent(
                        invocation_id=str(uuid.uuid4()),
                        target=source_host.alias.value,
                        summary=f"Failed capturing workspace on {source_host.alias.value}",
                        status=ws_status,
                        gesture=gesture_for_action_status(ws_status),
                        error_code=safe_ws_err,
                        host_id=str(source_host.host_id),
                        target_reason=target_reason,
                        target_platform=source_host.platform.value,
                    )
                )
                return Message(
                    role=Role.TOOL,
                    content=encode_tool_payload(ws_status, safe_ws_out or None),
                    tool_call_id=call.id,
                )
            if ws_status == "ok":
                captured_capsule = ws_sc
                cap_proj = _sanitize_alias_token(ws_sc.get("project")) or project_arg or "workspace"
                cap_id = _sanitize_alias_token(ws_sc.get("capsuleId"))
                steps.append(
                    MissionStep(
                        host_id=str(source_host.host_id),
                        host_alias=source_host.alias.value,
                        platform=source_host.platform,
                        tool="workspace.capture_capsule",
                        summary=f"Captured workspace '{cap_proj}' on {source_host.alias.value}",
                        risk=ws_src_risk,
                        status=MissionStepStatus.OK,
                    )
                )
                if open_app_arg and app_dst_ok and not want_url:
                    apps.append(open_app_arg)
                    a_idx = len(steps)
                    a_risk = app_dst_desc.risk if app_dst_desc else RiskTier.R2
                    steps.append(
                        MissionStep(
                            host_id=str(target_host.host_id),
                            host_alias=target_host.alias.value,
                            platform=target_host.platform,
                            tool="app.launch",
                            summary=f"Launch workspace app on {target_host.alias.value}",
                            risk=a_risk,
                            status=MissionStepStatus.PENDING,
                        )
                    )
                    target_actions.append(
                        ("app.launch", {"id": open_app_arg, "openTarget": cap_proj}, a_risk, a_idx)
                    )
                elif ws_open_ok:
                    o_idx = len(steps)
                    o_risk = ws_open_desc.risk if ws_open_desc else RiskTier.R2
                    steps.append(
                        MissionStep(
                            host_id=str(target_host.host_id),
                            host_alias=target_host.alias.value,
                            platform=target_host.platform,
                            tool="workspace.open",
                            summary=f"Open workspace '{cap_proj}' on {target_host.alias.value}",
                            risk=o_risk,
                            status=MissionStepStatus.PENDING,
                        )
                    )
                    target_actions.append(("workspace.open", {"project": cap_proj}, o_risk, o_idx))
                elif ws_restore_ok:
                    restore_args: dict[str, object] = {"project": cap_proj}
                    if cap_id is not None:
                        restore_args["capsuleId"] = cap_id
                    r_idx = len(steps)
                    r_risk = ws_restore_desc.risk if ws_restore_desc else RiskTier.R2
                    steps.append(
                        MissionStep(
                            host_id=str(target_host.host_id),
                            host_alias=target_host.alias.value,
                            platform=target_host.platform,
                            tool="workspace.restore_capsule",
                            summary=f"Restore workspace '{cap_proj}' on {target_host.alias.value}",
                            risk=r_risk,
                            status=MissionStepStatus.PENDING,
                        )
                    )
                    target_actions.append(
                        ("workspace.restore_capsule", restore_args, r_risk, r_idx)
                    )

        if not target_actions:
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=str(uuid.uuid4()),
                    target=target_host.alias.value,
                    summary=f"No executable target action on {target_host.alias.value}",
                    status="failed",
                    gesture=gesture_for_action_status("failed"),
                    error_code="unsupported",
                    host_id=str(target_host.host_id),
                    target_reason=target_reason,
                    target_platform=target_host.platform.value,
                )
            )
            return Message(
                role=Role.TOOL,
                content=encode_tool_payload("failed", {"errorCode": "unsupported"}),
                tool_call_id=call.id,
            )

        updated_mission = MissionEnvelope(
            mission_id=planned_mission.mission_id,
            title=target_reason[:120],
            goal=target_reason,
            steps=tuple(steps),
            apps=tuple(apps),
            domains=tuple(domains),
        )
        overall_status = "ok"
        first_err: str | None = None
        restore_sc: dict[str, object] = {}
        r2_target_scopes = [
            (target_host.host_id, x[0]) for x in target_actions if x[2] == RiskTier.R2
        ]

        updated_highest_risk = updated_mission.highest_risk()
        for t_tool, t_args, t_risk, s_idx in target_actions:
            eff_t_risk = (
                updated_highest_risk
                if (t_risk.level >= RiskTier.R2.level and updated_highest_risk.level > t_risk.level)
                else t_risk
            )
            step_approval: ApprovalRequest | None = None
            if eff_t_risk == RiskTier.R2:
                if upfront_r2_approval is not None:
                    step_approval = _derive_r2_step_approval(
                        upfront_r2_approval,
                        host=target_host,
                        tool=t_tool,
                        arguments=t_args,
                    )
                else:
                    step_approval = await self._try_authorize_r2_without_prompt(
                        ctx=ctx,
                        host=target_host,
                        tool=t_tool,
                        arguments=t_args,
                        now=now,
                    )
                    if step_approval is None:
                        ok, step_approval, app_status = await self._request_mission_approval(
                            mission=updated_mission,
                            primary_host=target_host,
                            approval_tool=t_tool,
                            approval_args=t_args,
                            target_reason=target_reason,
                            ctx=ctx,
                            now=now,
                            risk_override=RiskTier.R2,
                            grant_r2_scopes=r2_target_scopes,
                        )
                        if not ok:
                            denied_mission = updated_mission
                            for _, _, _, rem_idx in target_actions:
                                if rem_idx >= s_idx:
                                    denied_mission = denied_mission.with_step_status(
                                        rem_idx, MissionStepStatus.DENIED
                                    )
                            await ctx.emit_event(
                                ActionResultEvent(
                                    invocation_id=str(uuid.uuid4()),
                                    target=target_host.alias.value,
                                    summary=f"Teleport {app_status}",
                                    status=app_status,
                                    gesture=gesture_for_action_status(app_status),
                                    host_id=str(target_host.host_id),
                                    target_reason=target_reason,
                                    target_platform=target_host.platform.value,
                                    mission_envelope=denied_mission.to_wire(),
                                )
                            )
                            return Message(
                                role=Role.TOOL,
                                content=encode_tool_payload(app_status),
                                tool_call_id=call.id,
                            )
                        upfront_r2_approval = step_approval
            elif eff_t_risk.level >= RiskTier.R3.level:
                ok, step_approval, app_status = await self._request_mission_approval(
                    mission=updated_mission,
                    primary_host=target_host,
                    approval_tool=t_tool,
                    approval_args=t_args,
                    target_reason=target_reason,
                    ctx=ctx,
                    now=now,
                    risk_override=eff_t_risk,
                )
                if not ok:
                    denied_mission = updated_mission
                    for _, _, _, rem_idx in target_actions:
                        if rem_idx >= s_idx:
                            denied_mission = denied_mission.with_step_status(
                                rem_idx, MissionStepStatus.DENIED
                            )
                    await ctx.emit_event(
                        ActionResultEvent(
                            invocation_id=str(uuid.uuid4()),
                            target=target_host.alias.value,
                            summary=f"Teleport {app_status}",
                            status=app_status,
                            gesture=gesture_for_action_status(app_status),
                            host_id=str(target_host.host_id),
                            target_reason=target_reason,
                            target_platform=target_host.platform.value,
                            mission_envelope=denied_mission.to_wire(),
                        )
                    )
                    return Message(
                        role=Role.TOOL,
                        content=encode_tool_payload(app_status),
                        tool_call_id=call.id,
                    )

            t_status, t_sc, t_err = await self._invoke_host_tool(
                host=target_host,
                tool_name=t_tool,
                arguments=t_args,
                ctx=ctx,
                approval_req=step_approval,
            )
            step_st = (
                MissionStepStatus.OK
                if t_status == "ok"
                else (
                    MissionStepStatus.DENIED if t_status == "denied" else MissionStepStatus.FAILED
                )
            )
            updated_mission = updated_mission.with_step_status(s_idx, step_st)
            if t_tool == "workspace.restore_capsule":
                restore_sc = t_sc
            if (
                want_ws
                and t_status == "ok"
                and t_tool in ("workspace.open", "app.launch")
                and host_notify_ok
            ):
                hint_parts: list[str] = []
                if captured_capsule.get("branch"):
                    hint_parts.append(f"branch={str(captured_capsule['branch'])[:32]}")
                if isinstance(captured_capsule.get("dirtyCount"), int):
                    hint_parts.append(f"dirty={captured_capsule['dirtyCount']}")
                raw_hint = captured_capsule.get("nextStepHint")
                if isinstance(raw_hint, str) and raw_hint.strip():
                    hint_parts.append(redact_sensitive_text(raw_hint.strip()).text[:72])
                hint_body = "; ".join(hint_parts) if hint_parts else f"Workspace {cap_proj} ready"
                with contextlib.suppress(Exception):
                    await self._invoke_host_tool(
                        host=target_host,
                        tool_name="host.notify",
                        arguments={
                            "title": f"Teleport: {cap_proj}"[:40],
                            "body": hint_body[:120],
                        },
                        ctx=ctx,
                    )
            if t_status != "ok" and overall_status == "ok":
                overall_status = t_status
                first_err = t_err

        audit_tool = target_actions[-1][0]
        await self._record_pipeline_audit(
            owner_id=ctx.owner_id,
            action=(
                AuditAction.HOST_TOOL_DENIED
                if overall_status == "denied"
                else AuditAction.HOST_TOOL_INVOKED
            ),
            now=now,
            subject=str(target_host.host_id),
            request_id=ctx.request_id,
            metadata={"tool": audit_tool, "status": overall_status, "steps": len(steps)},
        )
        result_summary = (
            f"Teleported to {target_host.alias.value}"
            if overall_status == "ok"
            else f"Teleport to {target_host.alias.value} failed ({overall_status})"
        )
        await ctx.emit_event(
            ActionResultEvent(
                invocation_id=str(uuid.uuid4()),
                target=target_host.alias.value,
                summary=result_summary,
                status=overall_status,
                gesture=gesture_for_action_status(overall_status),
                error_code=first_err,
                host_id=str(target_host.host_id),
                target_reason=target_reason,
                target_platform=target_host.platform.value,
                mission_envelope=updated_mission.to_wire(),
            )
        )
        out_payload: dict[str, object] = {
            "sourceHost": source_host.alias.value,
            "targetHost": target_host.alias.value,
            "domains": domains,
        }
        if captured_capsule.get("project"):
            out_payload["project"] = str(captured_capsule["project"])[:64]
        if captured_capsule.get("branch"):
            out_payload["branch"] = str(captured_capsule["branch"])[:64]
        if isinstance(captured_capsule.get("dirtyCount"), int):
            out_payload["dirtyCount"] = captured_capsule["dirtyCount"]
        if captured_capsule.get("capsuleId"):
            out_payload["capsuleId"] = str(captured_capsule["capsuleId"])[:64]
        raw_cap_hint = captured_capsule.get("nextStepHint")
        if isinstance(raw_cap_hint, str):
            out_payload["nextStepHint"] = redact_sensitive_text(raw_cap_hint).text[:240]
        if restore_sc:
            if isinstance(restore_sc.get("restored"), bool):
                out_payload["restored"] = restore_sc["restored"]
            if restore_sc.get("capsuleId"):
                out_payload["capsuleId"] = str(restore_sc["capsuleId"])[:64]
            raw_res_hint = restore_sc.get("nextStepHint")
            if isinstance(raw_res_hint, str):
                out_payload["nextStepHint"] = redact_sensitive_text(raw_res_hint).text[:240]
            raw_briefing = restore_sc.get("briefingVi")
            if isinstance(raw_briefing, str):
                out_payload["briefingVi"] = redact_sensitive_text(raw_briefing).text[:240]
        if first_err:
            out_payload["errorCode"] = first_err
        return Message(
            role=Role.TOOL,
            content=encode_tool_payload(overall_status, out_payload),
            tool_call_id=call.id,
        )

    async def _run_smart_pause_media(
        self,
        call: ToolCall,
        ctx: "ActionContext",
        *,
        hosts: Sequence[Host],
        manifests: Mapping[HostId, Manifest],
        grants: Mapping[HostId, Sequence[Grant]],
        online: set[str] | frozenset[str],
    ) -> Message:
        now = self._clock()
        args = dict(call.arguments or {})
        if not set(args).issubset({"hostIds"}):
            raise ValueError("invalid_args")
        filter_ids = _parse_host_id_filter(args.get("hostIds"))

        active_matching = [
            h
            for h in sorted(hosts, key=lambda item: (item.alias.value.lower(), str(item.host_id)))
            if h.status == HostStatus.ACTIVE
            and (filter_ids is None or h.host_id.value in filter_ids)
        ]
        active_online: list[Host] = []
        offline_targets: list[Host] = []
        for h in active_matching:
            if str(h.host_id) in online or (
                ctx.self_host_id and str(h.host_id) == ctx.self_host_id
            ):
                active_online.append(h)
            elif filter_ids is not None:
                offline_targets.append(h)

        playing_hosts: list[Host] = []
        idle_hosts: list[Host] = []
        play_pause_unsupported_hosts: list[Host] = []
        probe_failure_hosts: list[tuple[Host, str]] = []
        skipped_hosts: list[dict[str, object]] = [
            {"hostId": str(h.host_id), "alias": h.alias.value, "status": "offline"}
            for h in offline_targets
        ]
        denied_hosts: list[dict[str, object]] = []
        probe_candidates: list[tuple[ProbeTarget, str, dict[str, object]]] = []

        for host in active_online:
            manifest, host_grants = await self._load_host_details(host, manifests, grants)
            play_desc = manifest.find_tool("media.play_pause") if manifest else None
            deny_reason = self._check_descriptor_access(
                play_desc, host_grants, now, host_screen_consent=ctx.host_screen_consent
            )
            if deny_reason == "unsupported":
                play_pause_unsupported_hosts.append(host)
                skipped_hosts.append(
                    {
                        "hostId": str(host.host_id),
                        "alias": host.alias.value,
                        "status": "unsupported",
                    }
                )
                continue
            if deny_reason is not None:
                denied_hosts.append(
                    {
                        "hostId": str(host.host_id),
                        "alias": host.alias.value,
                        "status": "denied",
                        "errorCode": deny_reason,
                    }
                )
                continue

            twin = _safe_get_twin(self._hub, str(host.host_id))
            if twin is not None and twin.presence is not None:
                if twin.presence.media_playing:
                    playing_hosts.append(host)
                else:
                    idle_hosts.append(host)
                    skipped_hosts.append(
                        {"hostId": str(host.host_id), "alias": host.alias.value, "status": "idle"}
                    )
                continue

            now_playing_desc = manifest.find_tool("media.now_playing") if manifest else None
            if (
                now_playing_desc is not None
                and self._check_descriptor_access(
                    now_playing_desc,
                    host_grants,
                    now,
                    host_screen_consent=ctx.host_screen_consent,
                )
                is None
            ):
                probe_candidates.append(
                    (
                        ProbeTarget(
                            host=host,
                            manifest=manifest,
                            grants=host_grants,
                            online=True,
                        ),
                        "media.now_playing",
                        {},
                    )
                )
            else:
                probe_failure_hosts.append((host, "unknown"))
                skipped_hosts.append(
                    {"hostId": str(host.host_id), "alias": host.alias.value, "status": "unknown"}
                )

        if probe_candidates:
            probe_outcomes = await self._prober.probe_many(
                probe_candidates,
                host_screen_consent=ctx.host_screen_consent,
                now=now,
            )
            for (target, _, _), outcome in zip(probe_candidates, probe_outcomes, strict=True):
                if (
                    outcome.status == "ok"
                    and isinstance(outcome.structured_content, dict)
                    and outcome.structured_content.get("playing") is True
                ):
                    playing_hosts.append(target.host)
                elif outcome.status == "ok":
                    idle_hosts.append(target.host)
                    skipped_hosts.append(
                        {
                            "hostId": str(target.host.host_id),
                            "alias": target.host.alias.value,
                            "status": "idle",
                        }
                    )
                else:
                    probe_failure_hosts.append((target.host, outcome.status))
                    skipped_hosts.append(
                        {
                            "hostId": str(target.host.host_id),
                            "alias": target.host.alias.value,
                            "status": outcome.status,
                        }
                    )

        steps: list[MissionStep] = [
            MissionStep(
                host_id=str(d["hostId"]),
                host_alias=str(d["alias"]),
                platform=next(
                    h.platform for h in active_online if str(h.host_id) == str(d["hostId"])
                ),
                tool="media.play_pause",
                summary=f"Denied media pause on {d['alias']}",
                risk=RiskTier.R1,
                status=MissionStepStatus.DENIED,
            )
            for d in denied_hosts
        ]

        paused_hosts: list[dict[str, object]] = []
        if playing_hosts:
            results = await asyncio.gather(
                *(
                    self._invoke_host_tool(
                        host=h,
                        tool_name="media.play_pause",
                        arguments={},
                        ctx=ctx,
                    )
                    for h in playing_hosts
                ),
                return_exceptions=True,
            )
            for host, res in zip(playing_hosts, results, strict=True):
                if isinstance(res, BaseException):
                    paused_hosts.append(
                        {
                            "hostId": str(host.host_id),
                            "alias": host.alias.value,
                            "status": "failed",
                            "errorCode": "rpc_error",
                        }
                    )
                    steps.append(
                        MissionStep(
                            host_id=str(host.host_id),
                            host_alias=host.alias.value,
                            platform=host.platform,
                            tool="media.play_pause",
                            summary=f"Failed pausing media on {host.alias.value}",
                            risk=RiskTier.R1,
                            status=MissionStepStatus.FAILED,
                        )
                    )
                    continue
                st, sc, err = res
                paused_hosts.append(
                    {
                        "hostId": str(host.host_id),
                        "alias": host.alias.value,
                        "status": st,
                        **({"errorCode": err} if err else {}),
                    }
                )
                step_status = (
                    MissionStepStatus.OK
                    if st == "ok"
                    else (MissionStepStatus.DENIED if st == "denied" else MissionStepStatus.FAILED)
                )
                step_summary = (
                    f"Paused media on {host.alias.value}"
                    if st == "ok"
                    else (
                        f"Denied pausing media on {host.alias.value}"
                        if st == "denied"
                        else f"Failed pausing media ({st}) on {host.alias.value}"
                    )
                )
                steps.append(
                    MissionStep(
                        host_id=str(host.host_id),
                        host_alias=host.alias.value,
                        platform=host.platform,
                        tool="media.play_pause",
                        summary=step_summary,
                        risk=RiskTier.R1,
                        status=step_status,
                    )
                )

        has_idle = bool(idle_hosts)
        unsupported_hosts: list[Host] = [
            *play_pause_unsupported_hosts,
            *(h for h, st in probe_failure_hosts if st == "unsupported"),
        ]
        non_unsupported_probe_failures = [
            (h, st) for h, st in probe_failure_hosts if st != "unsupported"
        ]

        if not playing_hosts:
            if filter_ids is not None and offline_targets:
                for off_h in offline_targets:
                    steps.append(
                        MissionStep(
                            host_id=str(off_h.host_id),
                            host_alias=off_h.alias.value,
                            platform=off_h.platform,
                            tool="media.play_pause",
                            summary=f"{off_h.alias.value} is offline",
                            risk=RiskTier.R1,
                            status=MissionStepStatus.FAILED,
                        )
                    )
            for idle_h in idle_hosts:
                steps.append(
                    MissionStep(
                        host_id=str(idle_h.host_id),
                        host_alias=idle_h.alias.value,
                        platform=idle_h.platform,
                        tool="media.play_pause",
                        summary=f"No active playback detected on {idle_h.alias.value}",
                        risk=RiskTier.R1,
                        status=MissionStepStatus.OK,
                    )
                )
            for pf_h, pf_st in probe_failure_hosts:
                steps.append(
                    MissionStep(
                        host_id=str(pf_h.host_id),
                        host_alias=pf_h.alias.value,
                        platform=pf_h.platform,
                        tool="media.play_pause",
                        summary=f"Failed probing media playback ({pf_st}) on {pf_h.alias.value}",
                        risk=RiskTier.R1,
                        status=MissionStepStatus.FAILED,
                    )
                )
            if not has_idle:
                for u_h in play_pause_unsupported_hosts:
                    steps.append(
                        MissionStep(
                            host_id=str(u_h.host_id),
                            host_alias=u_h.alias.value,
                            platform=u_h.platform,
                            tool="media.play_pause",
                            summary=f"Media playback control unsupported on {u_h.alias.value}",
                            risk=RiskTier.R1,
                            status=MissionStepStatus.FAILED,
                        )
                    )

        mission = MissionEnvelope(
            mission_id=str(uuid.uuid4()),
            title="Smart pause media across active devices",
            goal="Pause media only on devices actively playing",
            steps=tuple(steps),
        )
        primary = (
            playing_hosts[0]
            if playing_hosts
            else (
                idle_hosts[0]
                if idle_hosts and not probe_failure_hosts
                else (
                    active_online[0]
                    if active_online
                    else (offline_targets[0] if offline_targets else None)
                )
            )
        )
        ok_count = sum(1 for p in paused_hosts if p.get("status") == "ok")
        overall_err: str | None = None
        if playing_hosts:
            if ok_count > 0:
                overall = "ok"
                target_reason = f"Paused {ok_count} active playing host(s)"
            else:
                p_statuses = [str(p.get("status", "failed")) for p in paused_hosts]
                p_errors = [str(p["errorCode"]) for p in paused_hosts if p.get("errorCode")]
                overall, overall_err = _aggregate_fanout_failure(
                    p_statuses, p_errors, empty_err="media_error"
                )
                target_reason = f"Failed pausing media on {len(playing_hosts)} playing host(s)"
            payload_status = overall
        elif denied_hosts and not has_idle:
            overall = "denied"
            overall_err = next(
                (str(d["errorCode"]) for d in denied_hosts if d.get("errorCode")),
                "grant_denied",
            )
            target_reason = "Denied pausing media on active hosts"
            payload_status = "denied"
        elif has_idle and probe_failure_hosts:
            overall = "failed"
            overall_err = "probe_failed"
            target_reason = "Failed probing media playback on target hosts"
            payload_status = "failed"
        elif not has_idle:
            if (
                unsupported_hosts
                and not offline_targets
                and not non_unsupported_probe_failures
                and len(unsupported_hosts) == len(active_online)
            ):
                overall = "unsupported"
                overall_err = "unsupported"
                target_reason = "Media playback control is unsupported on target hosts"
                payload_status = "unsupported"
            elif offline_targets and not non_unsupported_probe_failures:
                overall = "failed"
                overall_err = "host_offline"
                target_reason = "All target hosts are offline"
                payload_status = "failed"
            elif probe_failure_hosts or unsupported_hosts:
                overall = "failed"
                overall_err = "probe_failed"
                target_reason = "Failed probing media playback on target hosts"
                payload_status = "failed"
            elif active_matching:
                overall = "failed"
                overall_err = "host_offline"
                target_reason = "All matching hosts are offline"
                payload_status = "failed"
            else:
                overall = "failed"
                overall_err = "no_host"
                target_reason = "No matching hosts for smart pause media"
                payload_status = "failed"
        else:
            overall = "ok"
            target_reason = "No devices reporting active media playback"
            payload_status = "no_active_media"
        await ctx.emit_event(
            ActionResultEvent(
                invocation_id=str(uuid.uuid4()),
                target=primary.alias.value if primary is not None else "Mesh",
                summary=target_reason,
                status=overall,
                gesture=gesture_for_action_status(overall),
                error_code=overall_err,
                host_id=str(primary.host_id) if primary is not None else None,
                target_reason=target_reason,
                target_platform=primary.platform.value if primary is not None else None,
                mission_envelope=mission.to_wire(),
            )
        )
        payload: dict[str, object] = {
            "status": payload_status,
            "pausedHosts": paused_hosts,
            "skippedHosts": skipped_hosts,
            "deniedHosts": denied_hosts,
        }
        if overall_err is not None:
            payload["errorCode"] = overall_err
        return Message(
            role=Role.TOOL,
            content=encode_mesh_meta_payload(payload),
            tool_call_id=call.id,
        )

    async def _run_fleet_vitals(
        self,
        call: ToolCall,
        ctx: "ActionContext",
        *,
        hosts: Sequence[Host],
        manifests: Mapping[HostId, Manifest],
        grants: Mapping[HostId, Sequence[Grant]],
        online: set[str] | frozenset[str],
    ) -> Message:
        args = dict(call.arguments or {})
        if not set(args).issubset({"hostIds"}):
            raise ValueError("invalid_args")
        filter_ids = _parse_host_id_filter(args.get("hostIds"))

        targets = build_probe_targets(
            hosts, manifests, grants, online, self_host_id=ctx.self_host_id
        )
        host_ids_tuple = tuple(sorted(filter_ids, key=str)) if filter_ids is not None else None
        payload = await self._prober.query_status(
            owner_id=ctx.owner_id,
            targets=targets,
            host_ids=host_ids_tuple,
            include_vitals=True,
            host_screen_consent=ctx.host_screen_consent,
        )
        raw_rows = payload.get("hosts")
        rows = raw_rows if isinstance(raw_rows, list) else []
        steps: list[MissionStep] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            vst = str(row.get("vitalsStatus", "failed"))
            step_status = (
                MissionStepStatus.OK
                if vst == "ok"
                else (MissionStepStatus.DENIED if vst == "denied" else MissionStepStatus.FAILED)
            )
            host_obj = next((h for h in hosts if str(h.host_id) == str(row.get("hostId"))), None)
            if host_obj is None:
                continue
            steps.append(
                MissionStep(
                    host_id=str(host_obj.host_id),
                    host_alias=host_obj.alias.value,
                    platform=host_obj.platform,
                    tool="system.vitals",
                    summary=f"Vitals ({vst}) on {host_obj.alias.value}",
                    risk=RiskTier.R1,
                    status=step_status,
                )
            )
        ok_count = sum(
            1 for row in rows if isinstance(row, dict) and row.get("vitalsStatus") == "ok"
        )
        overall_err: str | None = None
        if ok_count >= 1:
            overall_status = "ok"
            event_summary = "Collected fleet vitals"
        elif not rows:
            overall_status = "failed"
            overall_err = "no_host"
            event_summary = "No matching hosts for fleet vitals"
        else:
            online_statuses = [
                str(r.get("vitalsStatus", "failed"))
                for r in rows
                if isinstance(r, dict) and r.get("vitalsStatus") != "offline"
            ]
            online_errors = [
                str(r["vitalsErrorCode"])
                for r in rows
                if isinstance(r, dict)
                and r.get("vitalsStatus") != "offline"
                and r.get("vitalsErrorCode")
            ]
            if not online_statuses:
                overall_status = "failed"
                overall_err = "host_offline"
                event_summary = "All matching hosts are offline"
            else:
                mapped = ["failed" if s == "error" else s for s in online_statuses]
                overall_status, overall_err = _aggregate_fanout_failure(
                    mapped, online_errors, empty_err="probe_failed"
                )
                event_summary = f"Failed collecting fleet vitals ({overall_status})"

        if steps:
            mission = MissionEnvelope(
                mission_id=str(uuid.uuid4()),
                title="Fleet vitals check",
                goal="Query system vitals across online devices",
                steps=tuple(steps),
            )
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=str(uuid.uuid4()),
                    target=steps[0].host_alias,
                    summary=event_summary,
                    status=overall_status,
                    gesture=gesture_for_action_status(overall_status),
                    error_code=overall_err,
                    host_id=steps[0].host_id,
                    target_reason="Parallel fleet vitals check",
                    target_platform=steps[0].platform.value,
                    mission_envelope=mission.to_wire(),
                )
            )
        else:
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=str(uuid.uuid4()),
                    target="Mesh",
                    summary=event_summary,
                    status=overall_status,
                    gesture=gesture_for_action_status(overall_status),
                    error_code=overall_err,
                    target_reason="Parallel fleet vitals check",
                )
            )
        out_payload = dict(payload)
        out_payload["status"] = overall_status
        if overall_err is not None:
            out_payload["errorCode"] = overall_err
        return Message(
            role=Role.TOOL,
            content=encode_mesh_meta_payload(out_payload),
            tool_call_id=call.id,
        )

    async def _run_fleet_git_status(
        self,
        call: ToolCall,
        ctx: "ActionContext",
        *,
        hosts: Sequence[Host],
        manifests: Mapping[HostId, Manifest],
        grants: Mapping[HostId, Sequence[Grant]],
        online: set[str] | frozenset[str],
    ) -> Message:
        now = self._clock()
        args = dict(call.arguments or {})
        if not set(args).issubset({"hostIds", "project"}):
            raise ValueError("invalid_args")
        filter_ids = _parse_host_id_filter(args.get("hostIds"))
        project_filter = (
            _sanitize_alias_token(args.get("project")) if args.get("project") is not None else None
        )
        if args.get("project") is not None and project_filter is None:
            raise ValueError("invalid_args")

        targets = build_probe_targets(
            hosts, manifests, grants, online, self_host_id=ctx.self_host_id
        )
        requests: list[tuple[ProbeTarget, str, dict[str, object]]] = []
        skipped_rows: list[dict[str, object]] = []
        for target in targets:
            if filter_ids is not None and target.host.host_id.value not in filter_ids:
                continue
            if not target.online:
                skipped_rows.append(
                    {
                        "hostId": str(target.host.host_id),
                        "alias": target.host.alias.value,
                        "status": "offline",
                    }
                )
                continue
            twin = _safe_get_twin(self._hub, str(target.host.host_id))
            if project_filter is not None:
                projects = [project_filter]
            elif twin is not None and twin.resources and twin.resources.workspaces:
                projects = list(twin.resources.workspaces[:4])
            else:
                projects = ["recent"]
            for proj in projects:
                requests.append((target, "workspace.git_summary", {"project": proj}))

        outcomes = await self._prober.probe_many(
            requests,
            host_screen_consent=ctx.host_screen_consent,
            now=now,
        )
        results: list[dict[str, object]] = []
        steps: list[MissionStep] = []
        for (target, _, req_args), outcome in zip(requests, outcomes, strict=True):
            row: dict[str, object] = {
                "hostId": outcome.host_id,
                "alias": outcome.alias,
                "project": req_args["project"],
                "status": outcome.status,
            }
            if outcome.error_code is not None:
                row["errorCode"] = outcome.error_code
            if outcome.status == "ok" and outcome.structured_content is not None:
                row["git"] = outcome.structured_content
            results.append(row)
            st = (
                MissionStepStatus.OK
                if outcome.status == "ok"
                else (
                    MissionStepStatus.DENIED
                    if outcome.status == "denied"
                    else MissionStepStatus.FAILED
                )
            )
            steps.append(
                MissionStep(
                    host_id=outcome.host_id,
                    host_alias=outcome.alias,
                    platform=target.host.platform,
                    tool="workspace.git_summary",
                    summary=f"Git status ({req_args['project']}) on {outcome.alias}",
                    risk=RiskTier.R1,
                    status=st,
                )
            )

        ok_count = sum(1 for o in outcomes if o.status == "ok")
        overall_err: str | None = None
        if ok_count >= 1:
            overall_status = "ok"
            event_summary = "Collected fleet git status"
        elif not outcomes:
            overall_status = "failed"
            if skipped_rows:
                overall_err = "host_offline"
                event_summary = "All matching hosts are offline"
            else:
                overall_err = "no_host"
                event_summary = "No online hosts for fleet git status"
        else:
            mapped = ["failed" if o.status == "error" else o.status for o in outcomes]
            o_errors = [o.error_code for o in outcomes if o.error_code]
            overall_status, overall_err = _aggregate_fanout_failure(
                mapped, o_errors, empty_err="probe_failed"
            )
            event_summary = f"Failed collecting fleet git status ({overall_status})"

        if steps:
            mission = MissionEnvelope(
                mission_id=str(uuid.uuid4()),
                title="Fleet git status",
                goal="Query repository summary across online devices",
                steps=tuple(steps),
            )
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=str(uuid.uuid4()),
                    target=steps[0].host_alias,
                    summary=event_summary,
                    status=overall_status,
                    gesture=gesture_for_action_status(overall_status),
                    error_code=overall_err,
                    host_id=steps[0].host_id,
                    target_reason="Parallel fleet repository check",
                    target_platform=steps[0].platform.value,
                    mission_envelope=mission.to_wire(),
                )
            )
        else:
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=str(uuid.uuid4()),
                    target="Mesh",
                    summary=event_summary,
                    status=overall_status,
                    gesture=gesture_for_action_status(overall_status),
                    error_code=overall_err,
                    target_reason="Parallel fleet repository check",
                )
            )

        out_payload: dict[str, object] = {
            "status": overall_status,
            "results": results,
            "skipped": skipped_rows,
        }
        if overall_err is not None:
            out_payload["errorCode"] = overall_err
        return Message(
            role=Role.TOOL,
            content=encode_mesh_meta_payload(out_payload),
            tool_call_id=call.id,
        )
