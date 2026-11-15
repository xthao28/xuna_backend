"""Deterministic tool-call orchestrator for XUNA Brain (UDC-4 §5.3, UDC-7).

Executes the 8-step pipeline per tool call:
1. Function name mapping & forbidden-tool rejection.
2. Strict JSON Schema subset argument validation.
3. Deterministic target resolution without cross-host fallback (including pronouns).
4. Per-host capability grant verification (including expiry & suspension).
5. Owner approval gate (60 s TTL, `once` or R2 `conversation_15m`, R3 phone-signed)
   bound to `args_sha256`.
6. Dispatch via `HostSessionHub` (`tools/call`) or `LocalResultBroker` (`board_hid_request`).
7. Content-free audit logging.
8. Compact `Role.TOOL` JSON response (<= 2 KiB) and `ActionResultEvent` with deterministic gesture.
"""

import base64
import inspect
import re
import uuid
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from xuna_backend.modules.actions.application.approval_and_board_service import (
    ApprovalAndBoardService,
)
from xuna_backend.modules.actions.application.catalog import (
    CAP_BOARD_HID_V1,
    CAP_HOST_ACTIONS_V1,
    collect_eligible_host_tools,
    is_tool_granted,
    load_owner_host_state,
)
from xuna_backend.modules.actions.application.code_synthesizer import (
    CodeSpecialistSynthesizer,
)
from xuna_backend.modules.actions.application.computer_use import (
    MAX_SESSION_MINUTES,
    MAX_SESSION_STEPS,
    ComputerUseSessionState,
    InMemoryComputerUseSessionStore,
    build_session_event,
    hit_test_node_at_coordinates,
    is_activation_key_combo,
    is_consequential_element_name,
    is_editable_field_role,
    is_password_ui_node,
    redact_ui_tree_nodes,
    session_expiry_reason,
)
from xuna_backend.modules.actions.application.cu_template import CuTemplatePin
from xuna_backend.modules.actions.application.ports import (
    ApprovalBroker,
    ApprovalRepository,
    ApprovalScopeStore,
    ComputerUseSessionStore,
    LastTargetStore,
    LocalResultBroker,
    StandingApprovalRepository,
)
from xuna_backend.modules.actions.application.prompts import (
    BOARD_TARGET_LABEL_EN,
    BOARD_TARGET_LABEL_VI,
    DEADLINE_MS,
    DEADLINE_SECONDS,
    SYSTEM_PROMPT_EN,
    SYSTEM_PROMPT_VI,
    encode_tool_payload,
    tool_system_prompt,
    truncate_tool_value,
)
from xuna_backend.modules.actions.application.tool_discovery import (
    META_FUNCTION_NAMES,
    TOOL_DESCRIBE_FUNCTION,
    TOOL_SEARCH_FUNCTION,
    MetaToolArgumentError,
    describe_tool,
    encode_meta_payload,
    parse_invoke,
    search_tools,
)
from xuna_backend.modules.actions.application.workflows import (
    BoardWorkflowTriggerResult,
    CompositeWorkflowRunner,
    InMemoryBoardWorkflowSlotStore,
    WorkflowSlotAssignment,
    validate_https_public_url,
)
from xuna_backend.modules.actions.domain.entities import (
    APPROVAL_TTL_SECONDS,
    ActionPendingEvent,
    ActionResultEvent,
    ActionStreamEvent,
    ActionSummary,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScope,
    ApprovalState,
    ComputerUseSessionEvent,
    EmergencyStopSummary,
    LocalActionRequestEvent,
    gesture_for_action_status,
)
from xuna_backend.modules.actions.domain.personalization import (
    HostDefaultConvention,
    InMemoryHostDefaultStore,
)
from xuna_backend.modules.actions.domain.schema_validator import (
    ArgumentValidationError,
    SchemaDefinitionError,
    SchemaValidator,
)
from xuna_backend.modules.actions.domain.standing_approval import (
    STANDING_SUGGEST_MIN_APPROVALS,
    STANDING_SUGGEST_WINDOW,
    is_standing_eligible,
)
from xuna_backend.modules.actions.domain.target_resolver import (
    TargetResolutionStatus,
    TargetResolver,
)
from xuna_backend.modules.actions.domain.tool_names import (
    BOARD_TOOL,
    WORKFLOW_TEMPLATE_TOOL,
    from_function_name,
    is_forbidden_tool_name,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.cognition.application.ports import TextGenerator
from xuna_backend.modules.cognition.domain.types import Locale, Message, Role, ToolCall
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.application.xtp_frames import (
    HostPreflight,
    parse_preflight_result,
)
from xuna_backend.modules.hosts.domain.crypto import args_sha256
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
)
from xuna_backend.modules.identity.application.ports import SignatureVerifier

__all__ = [
    "BOARD_TARGET_LABEL_EN",
    "BOARD_TARGET_LABEL_VI",
    "CAP_BOARD_HID_V1",
    "CAP_HOST_ACTIONS_V1",
    "DEADLINE_MS",
    "DEADLINE_SECONDS",
    "SYSTEM_PROMPT_EN",
    "SYSTEM_PROMPT_VI",
    "ActionContext",
    "EventSink",
    "Orchestrator",
    "tool_system_prompt",
]

EventSink = Callable[[ActionStreamEvent], Awaitable[None]]

_truncate_value = truncate_tool_value
_encode_tool_payload = encode_tool_payload

# Host error codes that mean the computer-use session no longer exists on the host.
_CU_SESSION_ENDING_ERROR_CODES: frozenset[str] = frozenset(
    {
        "cu_session_required",
        "step_budget_exceeded",
        "time_budget_exceeded",
        "duplicate_action",
        "no_progress",
        "stopped",
    }
)

# UDC-17 S17.1: tools whose risk the host decides from the live screen via `xuna/preflight`.
_PREFLIGHT_TOOLS: frozenset[str] = frozenset(
    {"input.click", "input.type_text", "input.key", "input.drag"}
)
# Desktop agents only; other host kinds keep the legacy estimate until they implement it.
_PREFLIGHT_PLATFORMS: frozenset[HostPlatform] = frozenset(
    {HostPlatform.WINDOWS, HostPlatform.MACOS}
)
# Preflight reads the live window; a slow answer is treated as no answer (R3).
PREFLIGHT_TIMEOUT_SECONDS = 2.0
_METHOD_NOT_FOUND_PREFIX = "rpc_error:-32601:"
# ADR-0015 D1: text typed since the last commit, kept per session for the next commit card.
_MAX_REMEMBERED_DRAFTS = 5
# Input steps that can commit typed text (send, submit), so their cards show it.
_COMMIT_INPUT_TOOLS: frozenset[str] = frozenset({"input.click", "input.key", "input.drag"})

# Host agents built before `xuna/policy.idempotent` existed do not declare it. Only for
# those undeclared descriptors, these known toggles keep repeat protection; a host's own
# declaration always wins. Remove once every shipped host agent declares the flag.
_LEGACY_UNDECLARED_TOGGLES: frozenset[str] = frozenset({"media.play_pause"})


def is_repeat_safe(descriptor: ToolDescriptor) -> bool:
    """Whether re-running a successful call with identical arguments is harmless."""
    if descriptor.idempotent is not None:
        return descriptor.idempotent
    return descriptor.name not in _LEGACY_UNDECLARED_TOGGLES


def _cache_ui_nodes(
    state: ComputerUseSessionState,
    originals: list[dict[str, Any]],
    redacted: list[dict[str, Any]],
) -> None:
    """Indexes UI nodes the host returned by raw id and snapshot ref (with and without `@`)."""
    for orig, red in zip(originals, redacted, strict=False):
        raw_id = str(red.get("id") or "").strip()
        qual_ref = str(red.get("ref") or raw_id).strip()
        if raw_id:
            state.cached_nodes[raw_id] = orig
        if qual_ref:
            state.cached_nodes[qual_ref] = orig
            if qual_ref.startswith("@"):
                state.cached_nodes[qual_ref[1:]] = orig


def _apply_ui_changes(
    structured_content: dict[str, object], state: ComputerUseSessionState | None
) -> None:
    """Redacts the elements an `input.*` result reports as new (`changes.added`, UDC-17
    S17.4) and keeps the session's node cache in step: new refs are added, dropped refs
    removed. The host already redacts; this is the backend's own check."""
    changes = structured_content.get("changes")
    if not isinstance(changes, dict):
        return
    raw_added = changes.get("added")
    if isinstance(raw_added, list):
        dict_added = [n for n in raw_added if isinstance(n, dict)]
        redacted = redact_ui_tree_nodes(dict_added)
        changes["added"] = redacted
        if state is not None:
            _cache_ui_nodes(state, dict_added, redacted)
    raw_removed = changes.get("removed")
    if state is not None and isinstance(raw_removed, list):
        for ref in raw_removed:
            if isinstance(ref, str) and ref.strip():
                key = ref.strip()
                state.cached_nodes.pop(key, None)
                if key.startswith("@"):
                    state.cached_nodes.pop(key[1:], None)


@dataclass(frozen=True, slots=True)
class CuSessionStopResult:
    """Outcome of stopping one control session from the phone (UDC-17 S17.3)."""

    stopped: bool
    host_confirmed: bool
    already_ended: bool


# Best-effort `cu.session_end` sent to the host when the owner stops a session.
_CU_STOP_DEADLINE_MS = 5000


@dataclass(frozen=True, slots=True)
class ActionContext:
    """Per-turn context passed into `Orchestrator.handle_call`."""

    owner_id: uuid.UUID
    pet_id: str
    locale: Locale
    client_caps: Collection[str]
    board_available: bool
    emit_event: EventSink
    request_id: str = ""
    self_host_id: str | None = None
    host_defaults: dict[str, str] | None = None
    host_screen_consent: bool = True
    # UDC-17 S17.5: the owner tapped "Chạy lại" on a template saved on their phone; this
    # turn's `cu.session_start` must match its scope. `None` for every other turn.
    cu_template: CuTemplatePin | None = None


class Orchestrator:
    """Deterministic tool-call orchestrator (UDC-4 §5.3, UDC-7)."""

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
        scope_store: ApprovalScopeStore | None = None,
        verifier: SignatureVerifier | None = None,
        workflow_slot_store: InMemoryBoardWorkflowSlotStore | None = None,
        host_default_store: InMemoryHostDefaultStore | None = None,
        clock: Callable[[], datetime] | None = None,
        code_generator: TextGenerator | None = None,
        standing_repo: StandingApprovalRepository | None = None,
        cu_session_store: ComputerUseSessionStore | None = None,
    ) -> None:
        self._standing_repo = standing_repo
        self._cu_session_store: ComputerUseSessionStore = (
            cu_session_store if cu_session_store is not None else InMemoryComputerUseSessionStore()
        )
        self._host_repo = host_repo
        self._hub = hub
        self._approval_repo = approval_repo
        self._approval_broker = approval_broker
        self._local_result_broker = local_result_broker
        self._last_target_store = last_target_store
        self._audit = audit
        self._scope_store = scope_store
        self._verifier = verifier
        self._clock = clock or (lambda: datetime.now(UTC))
        self._code_synthesizer = (
            CodeSpecialistSynthesizer(code_generator) if code_generator is not None else None
        )
        self._workflow_slot_store = workflow_slot_store or InMemoryBoardWorkflowSlotStore()
        self._host_default_store = host_default_store or InMemoryHostDefaultStore()
        self._approval_and_board = ApprovalAndBoardService(
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
        self._workflow_runner = CompositeWorkflowRunner(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_result_broker,
            audit=audit,
            slot_store=self._workflow_slot_store,
            clock=self._clock,
        )

    async def emergency_stop(
        self,
        *,
        owner_id: uuid.UUID,
        reason: str = "owner_app",
        request_id: str = "",
    ) -> EmergencyStopSummary:
        """Cancels all active invocations, rejects pending approvals, sends `xuna/stop`
        to every live host, suspends R2+ grants, and audits `host.stopped` (`UDC-7` S7.3)."""
        now = self._clock()
        running = await self._cu_session_store.get_active_for_owner(owner_id)
        if running is not None:
            await self._end_cu_state(running, reason, now, request_id)
        await self._cu_session_store.stop_all_for_owner(owner_id, reason, now)
        return await self._approval_and_board.emergency_stop(
            owner_id=owner_id,
            reason=reason,
            request_id=request_id,
        )

    async def active_cu_session_event(
        self, owner_id: uuid.UUID, request_id: str = ""
    ) -> ComputerUseSessionEvent | None:
        """The owner's running session for the phone's session card, after the heartbeat,
        time and step budgets are applied (UDC-17 S17.3, S17.6), or `None`."""
        now = self._clock()
        state = await self._live_cu_session(owner_id, None, now, request_id)
        return build_session_event(state, now, "") if state is not None else None

    async def _live_cu_session(
        self,
        owner_id: uuid.UUID,
        host_id: HostId | None,
        now: datetime,
        request_id: str,
    ) -> ComputerUseSessionState | None:
        """The owner's running session (on `host_id` when given), or `None`.

        Applies the stop rules lazily: a session whose phone heartbeat is older than
        `HEARTBEAT_TIMEOUT_S` (UDC-8 dead-man switch) or whose time or step budget ran out
        ends here. On a lost heartbeat the host is also told to end its session, since the
        host does not see the phone.
        """
        state = await self._cu_session_store.get_active_for_owner(owner_id)
        if state is None:
            return None
        reason = session_expiry_reason(state, now)
        if reason is not None:
            await self._end_cu_state(state, reason, now)
            if reason == "heartbeat_timeout":
                confirmed = await self._end_cu_session_on_host(state, "stopped")
                await self._audit_cu_session_stop(
                    owner_id, state.host_id, reason, confirmed, request_id
                )
            return None
        if host_id is not None and state.host_id != host_id:
            return None
        return state

    async def _end_cu_state(
        self,
        state: ComputerUseSessionState,
        reason: str,
        now: datetime,
        request_id: str = "",
    ) -> None:
        """Ends `state` in the store and on the object this turn holds.

        The first end of a running session is audited with counts only (UDC-17 S17.6
        measurement): why it ended, steps used, owner approvals asked and minutes run.
        """
        was_active = state.active
        await self._cu_session_store.mark_ended(state.session_id, reason, now)
        if state.active:
            state.active = False
            state.stop_reason = reason
        state.cached_nodes.clear()
        state.typed_since_commit.clear()
        state.typed_count = 0
        if was_active:
            minutes = max(int((now - state.started_at_utc).total_seconds() // 60), 0)
            await self._audit.record(
                AuditEvent(
                    owner_id=state.owner_id,
                    action=AuditAction.HOST_CU_SESSION_ENDED,
                    occurred_at=now,
                    subject=str(state.host_id),
                    request_id=request_id,
                    metadata={
                        "reason": reason[:64],
                        "steps": state.steps_used,
                        "prompts": state.owner_prompts,
                        "minutes": minutes,
                    },
                )
            )

    async def _end_cu_session_on_host(self, state: ComputerUseSessionState, outcome: str) -> bool:
        """Best-effort `cu.session_end` to the host; whether the host confirmed it."""
        try:
            raw = await self._hub.send_request(
                str(state.host_id),
                "tools/call",
                {
                    "name": "cu.session_end",
                    "arguments": {"sessionId": state.session_id, "outcome": outcome},
                    "_meta": {
                        "xuna/invocationId": str(uuid.uuid4()),
                        "xuna/deadlineMs": _CU_STOP_DEADLINE_MS,
                    },
                },
                timeout_s=_CU_STOP_DEADLINE_MS / 1000,
            )
        except Exception:
            return False
        meta = raw.get("_meta")
        return isinstance(meta, dict) and meta.get("xuna/status") == "ok"

    async def _audit_cu_session_stop(
        self,
        owner_id: uuid.UUID,
        host_id: HostId,
        reason: str,
        host_confirmed: bool,
        request_id: str,
    ) -> None:
        await self._audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=AuditAction.HOST_CU_SESSION_STOPPED,
                occurred_at=self._clock(),
                subject=str(host_id),
                request_id=request_id,
                metadata={"hostConfirmed": host_confirmed, "reason": reason},
            )
        )

    async def stop_cu_session(
        self,
        *,
        owner_id: uuid.UUID,
        session_id: str,
        request_id: str = "",
    ) -> CuSessionStopResult | None:
        """Ends one control session from the phone's session card (UDC-17 S17.3).

        Owner-scoped and idempotent. The backend copy ends first, so the R2 exemption and
        the typed text are gone at once; then the host is asked, best effort, to end its
        session too. E-Stop stays the stronger control. Returns `None` when the owner has
        no such session.
        """
        state = await self._cu_session_store.get(session_id)
        if state is None or state.owner_id != owner_id:
            return None
        if not state.active:
            return CuSessionStopResult(stopped=True, host_confirmed=False, already_ended=True)
        await self._end_cu_state(state, "owner_stopped", self._clock())
        host_confirmed = await self._end_cu_session_on_host(state, "stopped")
        await self._audit_cu_session_stop(
            owner_id, state.host_id, "owner_stopped", host_confirmed, request_id
        )
        return CuSessionStopResult(stopped=True, host_confirmed=host_confirmed, already_ended=False)

    async def record_cu_heartbeat(
        self, *, owner_id: uuid.UUID, session_id: str, request_id: str = ""
    ) -> ComputerUseSessionState | None:
        """Phone heartbeat for the owner's running session (UDC-8 dead-man switch).

        The stop rules run first, so a heartbeat that arrives after the session already
        timed out does not revive it. Returns the running session, or `None`.
        """
        now = self._clock()
        state = await self._live_cu_session(owner_id, None, now, request_id)
        if state is None or state.session_id != session_id:
            return None
        if not await self._cu_session_store.touch_heartbeat(session_id, owner_id, now):
            return None
        state.last_heartbeat_at = now
        return state

    @staticmethod
    def _lookup_cached_cu_node(
        active_cu: ComputerUseSessionState | None,
        clean_args: dict[str, object],
    ) -> dict[str, object] | None:
        if active_cu is None:
            return None
        raw_id = clean_args.get("elementId")
        if not isinstance(raw_id, str) or not raw_id.strip():
            return None
        key = raw_id.strip()
        node = active_cu.cached_nodes.get(key)
        if node is None and key.startswith("@"):
            node = active_cu.cached_nodes.get(key[1:])
        return node

    @staticmethod
    def _is_consequential_web_action(args: dict[str, object]) -> bool:
        """Checks whether a browser action targets a consequential or sensitive element."""
        for key in ("ref", "selector", "targetText", "elementName", "text", "value"):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                # Split camelCase identifiers (e.g. payNow -> pay Now, deleteItem -> delete Item)
                decomposed = re.sub(r"([a-z])([A-Z])", r"\1 \2", val)
                if is_consequential_element_name(val) or is_consequential_element_name(decomposed):
                    return True
                normalized = decomposed.replace("_", " ").replace("-", " ").lower()
                if is_consequential_element_name(normalized):
                    return True
                words = set(re.findall(r"[\w]+", normalized))
                # Word-boundary sensitive matches (prevent false positives on
                # spin, opinion, wildcard, cardigan)
                sensitive_tokens = ("pin", "card", "credit", "token", "secret", "wire", "thẻ")
                if any(w in words for w in sensitive_tokens):
                    return True
                if (
                    "checkout" in normalized
                    or "check out" in normalized
                    or any(
                        tok in normalized
                        for tok in (
                            "password",
                            "passwd",
                            "cvv",
                            "cvc",
                            "payment",
                            "checkout",
                            "check out",
                            "transfer",
                            "chuyen khoan",
                            "thanh toan",
                            "dat hang",
                            "xoa",
                            "mua ngay",
                            "xac nhan",
                            "mật khẩu",
                            "mat khau",
                            "the tin dung",
                            "the visa",
                            "the mastercard",
                            "the atm",
                        )
                    )
                ):
                    return True
        return False

    @classmethod
    def _effective_tool_risk(
        cls,
        descriptor: ToolDescriptor,
        clean_args: dict[str, object],
        active_cu: ComputerUseSessionState | None,
    ) -> RiskTier:
        if descriptor.name == "mail.send":
            return RiskTier.R3
        if descriptor.name == "mail.create_draft":
            return RiskTier.R2
        if descriptor.name in ("web.click", "web.fill"):
            if cls._is_consequential_web_action(clean_args):
                return RiskTier.R3
            return descriptor.risk
        if descriptor.name == "input.click":
            x_norm = clean_args.get("xNorm")
            y_norm = clean_args.get("yNorm")
            is_coord_click = isinstance(x_norm, int | float) and isinstance(y_norm, int | float)

            explicit_name = clean_args.get("elementName")
            if isinstance(explicit_name, str) and is_consequential_element_name(explicit_name):
                return RiskTier.R3

            cached_node = None
            if isinstance(x_norm, int | float) and isinstance(y_norm, int | float):
                if (
                    float(x_norm) < 0.0
                    or float(x_norm) > 1.0
                    or float(y_norm) < 0.0
                    or float(y_norm) > 1.0
                ):
                    return RiskTier.R3
                if active_cu is not None:
                    cached_node = hit_test_node_at_coordinates(
                        list(active_cu.cached_nodes.values()), float(x_norm), float(y_norm)
                    )
            else:
                cached_node = cls._lookup_cached_cu_node(active_cu, clean_args)

            if cached_node is not None:
                if is_password_ui_node(cached_node):
                    return RiskTier.R3
                cached_name = cached_node.get("name")
                if isinstance(cached_name, str) and is_consequential_element_name(cached_name):
                    return RiskTier.R3
                if active_cu is not None and active_cu.typed_count > 0:
                    if is_coord_click:
                        role = str(cached_node.get("role") or "")
                        if (
                            not is_editable_field_role(role)
                            or cached_node.get("editable") is False
                            or cached_node.get("readOnly") is True
                        ):
                            return RiskTier.R3
                    else:
                        if not cached_name or not str(cached_name).strip():
                            return RiskTier.R3
            elif active_cu is not None:
                # No tree read in this process (not inspected yet, or the backend restarted),
                # or unclassified coordinate hit: target is unknown, so escalate to R3.
                return RiskTier.R3
        elif descriptor.name == "input.drag":
            from_x = clean_args.get("fromXNorm") or clean_args.get("startXNorm")
            from_y = clean_args.get("fromYNorm") or clean_args.get("startYNorm")
            to_x = clean_args.get("toXNorm") or clean_args.get("endXNorm")
            to_y = clean_args.get("toYNorm") or clean_args.get("endYNorm")

            for c in (from_x, from_y, to_x, to_y):
                if isinstance(c, int | float) and (c < 0.0 or c > 1.0):
                    return RiskTier.R3

            start_node = None
            end_node = None
            if active_cu is not None:
                nodes = list(active_cu.cached_nodes.values())
                if isinstance(from_x, int | float) and isinstance(from_y, int | float):
                    start_node = hit_test_node_at_coordinates(nodes, float(from_x), float(from_y))
                if isinstance(to_x, int | float) and isinstance(to_y, int | float):
                    end_node = hit_test_node_at_coordinates(nodes, float(to_x), float(to_y))

                if (start_node and is_password_ui_node(start_node)) or (
                    end_node and is_password_ui_node(end_node)
                ):
                    return RiskTier.R3

                # Uncommitted draft text protection:
                if active_cu.typed_count > 0:
                    return RiskTier.R3

                for node in (start_node, end_node):
                    if node:
                        name = node.get("name")
                        if isinstance(name, str) and is_consequential_element_name(name):
                            return RiskTier.R3

                if start_node is None or end_node is None:
                    return RiskTier.R3
            elif active_cu is not None:
                return RiskTier.R3
        elif descriptor.name == "input.key":
            combo = clean_args.get("combo")
            if isinstance(combo, str):
                norm_combo = "+".join(p.strip().lower() for p in combo.split("+") if p.strip())
                is_safe_editing = norm_combo in {
                    "cmd+n", "cmd+a", "cmd+c", "cmd+v",
                    "command+n", "command+a", "command+c", "command+v",
                    "ctrl+n", "ctrl+a", "ctrl+c", "ctrl+v",
                }
                is_draft_break = norm_combo in {"enter", "return", "shift+enter", "shift+return"}
                focused_role = str(
                    clean_args.get("focusedRole") or clean_args.get("focused_role") or ""
                ).lower()
                cached_node = cls._lookup_cached_cu_node(active_cu, clean_args)
                if cached_node is not None:
                    node_role = str(cached_node.get("role") or "").lower()
                    if node_role:
                        focused_role = node_role
                is_draft_area = focused_role in {"textarea", "textfield", "searchfield", "document"}
                in_form = clean_args.get("inForm") is True
                allow_draft = (
                    is_draft_break and not in_form and (active_cu is not None or is_draft_area)
                )
                if is_safe_editing or allow_draft:
                    return descriptor.risk
                if is_activation_key_combo(combo):
                    return RiskTier.R3
        elif descriptor.name == "app.launch":
            if active_cu is not None:
                raw_app = (
                    clean_args.get("id")
                    or clean_args.get("app_id")
                    or clean_args.get("name")
                    or ""
                )
                app_name = str(raw_app).strip().lower()
                if app_name and app_name not in active_cu.allowed_apps:
                    return RiskTier.R3
        return descriptor.risk

    @staticmethod
    def _declared_risk_floor(descriptor: ToolDescriptor, clean_args: dict[str, object]) -> RiskTier:
        """Risk implied by the call's own arguments.

        Used together with a host preflight answer: it can raise that answer (a model that
        names its target "Send" gets R3), never lower it. Activation keys are not raised
        here: whether Enter/Space is safe depends on the live focus, which only the host
        sees (ADR-0015 D2). The legacy estimate for hosts without preflight still treats
        them as R3.
        """
        if descriptor.name == "mail.send":
            return RiskTier.R3
        if descriptor.name == "mail.create_draft":
            return RiskTier.R2
        if descriptor.name in ("web.click", "web.fill"):
            if Orchestrator._is_consequential_web_action(clean_args):
                return RiskTier.R3
            return descriptor.risk
        if descriptor.name == "input.click":
            name = clean_args.get("elementName")
            if isinstance(name, str) and is_consequential_element_name(name):
                return RiskTier.R3
        elif descriptor.name == "input.drag":
            for k in ("elementName", "fromElementName", "toElementName"):
                name = clean_args.get(k)
                if isinstance(name, str) and is_consequential_element_name(name):
                    return RiskTier.R3
        return descriptor.risk

    async def _host_preflight(
        self, host_id: HostId, tool_name: str, clean_args: dict[str, object]
    ) -> HostPreflight | None:
        """Asks the host which risk it would enforce (`xuna/preflight`, UDC-17 S17.1).

        Returns `None` only when the host answers "method not found", i.e. it predates
        preflight, so the legacy estimate applies. Any other failure (timeout, offline,
        malformed answer) is fail-closed: R3 with no element label.
        """
        try:
            raw = await self._hub.send_request(
                str(host_id),
                "xuna/preflight",
                {"name": tool_name, "arguments": dict(clean_args)},
                timeout_s=PREFLIGHT_TIMEOUT_SECONDS,
            )
            return parse_preflight_result(raw)
        except Exception as error:
            if isinstance(error, RuntimeError) and str(error).startswith(_METHOD_NOT_FOUND_PREFIX):
                return None
            return HostPreflight(
                status="ok",
                effective_risk=RiskTier.R3,
                reason_code="preflight_unavailable",
                error_code=None,
                element_role=None,
                element_name=None,
            )

    async def _update_cu_session_after_call(
        self,
        *,
        owner_id: uuid.UUID,
        host_id: HostId,
        descriptor: ToolDescriptor,
        clean_args: dict[str, object],
        status: str,
        structured_content: dict[str, object],
        active_cu: ComputerUseSessionState | None,
        now: datetime,
        target_alias: str = "",
    ) -> None:
        if descriptor.name == "cu.session_extend" and status == "ok" and active_cu is not None:
            # UDC-17 S17.3: take the full lists the host now enforces, not the call's arguments.
            host_apps = structured_content.get("apps")
            host_domains = structured_content.get("domains")
            if isinstance(host_apps, list) and isinstance(host_domains, list):
                active_cu.allowed_apps = tuple(
                    str(a).strip().lower() for a in host_apps if str(a).strip()
                )
                active_cu.allowed_domains = tuple(
                    str(d).strip().lower() for d in host_domains if str(d).strip()
                )
                await self._cu_session_store.set_scope(
                    active_cu.session_id, active_cu.allowed_apps, active_cu.allowed_domains
                )
            return

        if descriptor.name == "cu.session_start" and status == "ok":
            previous = await self._cu_session_store.get_active_for_owner(owner_id)
            if previous is not None:
                await self._end_cu_state(previous, "replaced", now)
            raw_sid = structured_content.get("sessionId")
            session_id = (
                raw_sid.strip()
                if isinstance(raw_sid, str) and raw_sid.strip()
                else str(uuid.uuid4())
            )
            raw_apps = clean_args.get("apps")
            apps = (
                tuple(str(a).strip().lower() for a in raw_apps if str(a).strip())
                if isinstance(raw_apps, list)
                else ()
            )
            raw_domains = clean_args.get("domains")
            domains = (
                tuple(str(d).strip().lower() for d in raw_domains if str(d).strip())
                if isinstance(raw_domains, list)
                else ()
            )
            # The host validated and accepted these arguments; the clamp only keeps the
            # backend's own budget inside the same bounds if a host ever accepts more.
            raw_steps = clean_args.get("maxSteps")
            max_steps = min(max(raw_steps if type(raw_steps) is int else 12, 1), MAX_SESSION_STEPS)
            raw_mins = clean_args.get("maxMinutes")
            max_mins = min(max(raw_mins if type(raw_mins) is int else 5, 1), MAX_SESSION_MINUTES)
            # `start` ends any other running session of the owner as `replaced`.
            await self._cu_session_store.start(
                ComputerUseSessionState(
                    session_id=session_id,
                    owner_id=owner_id,
                    host_id=host_id,
                    goal=str(clean_args.get("goal", "")),
                    allowed_apps=apps,
                    allowed_domains=domains,
                    max_steps=max_steps,
                    max_minutes=max_mins,
                    started_at_utc=now,
                    last_heartbeat_at=now,
                    target_alias=target_alias,
                ),
                now,
            )
            return

        if descriptor.name == "cu.session_end":
            outcome = str(clean_args.get("outcome") or "completed")
            raw_sid = clean_args.get("sessionId")
            if isinstance(raw_sid, str) and raw_sid.strip():
                named = await self._cu_session_store.get(raw_sid.strip())
                if named is not None and named.owner_id == owner_id:
                    await self._end_cu_state(named, outcome, now)
            if active_cu is not None:
                await self._end_cu_state(active_cu, outcome, now)
            return

        if (
            active_cu is not None
            and status != "ok"
            and descriptor.family in (ToolFamily.SCREEN, ToolFamily.UI, ToolFamily.INPUT)
        ):
            # The host is the authority on the session: once it reports the session gone or
            # out of budget, stop exempting R2 steps from approval here too.
            err_code = str(structured_content.get("errorCode") or "")
            if err_code in _CU_SESSION_ENDING_ERROR_CODES:
                await self._end_cu_state(active_cu, err_code, now)
            return

        if descriptor.name in ("ui.inspect", "ui.observe_hybrid") and status == "ok":
            raw_nodes = structured_content.get("nodes")
            if isinstance(raw_nodes, list):
                dict_nodes = [n for n in raw_nodes if isinstance(n, dict)]
                redacted = redact_ui_tree_nodes(dict_nodes)
                structured_content["nodes"] = redacted
                if active_cu is not None:
                    active_cu.cached_nodes.clear()
                    _cache_ui_nodes(active_cu, dict_nodes, redacted)
            return

        if descriptor.name == "ui.find" and status == "ok":
            # UDC-17 S17.4: candidates join the host's current snapshot, so the cache grows
            # instead of being replaced.
            raw_matches = structured_content.get("matches")
            if isinstance(raw_matches, list):
                dict_matches = [n for n in raw_matches if isinstance(n, dict)]
                redacted = redact_ui_tree_nodes(dict_matches)
                structured_content["matches"] = redacted
                if active_cu is not None:
                    _cache_ui_nodes(active_cu, dict_matches, redacted)
            return

        if descriptor.family == ToolFamily.INPUT and status == "ok":
            _apply_ui_changes(structured_content, active_cu)

        if descriptor.family == ToolFamily.INPUT and active_cu is not None and status == "ok":
            steps = await self._cu_session_store.record_step(active_cu.session_id)
            if steps is None:
                return
            active_cu.steps_used = steps
            if steps >= active_cu.max_steps:
                await self._end_cu_state(active_cu, "step_budget_exceeded", now)

    async def decide_approval(
        self,
        *,
        approval_id: uuid.UUID,
        owner_id: uuid.UUID,
        decision: ApprovalDecision,
        scope: ApprovalScope = ApprovalScope.ONCE,
        signature: bytes | None = None,
        signer_public_key: bytes | None = None,
        request_id: str = "",
    ) -> ApprovalRequest | None:
        """Owner-scoped, idempotent approval decision entry point
        (`POST /v1/actions/approvals/{id}`)."""
        return await self._approval_and_board.decide_approval(
            approval_id=approval_id,
            owner_id=owner_id,
            decision=decision,
            scope=scope,
            signature=signature,
            signer_public_key=signer_public_key,
            request_id=request_id,
        )

    def report_board_result(
        self,
        *,
        invocation_id: str,
        owner_id: uuid.UUID,
        status: str,
        output: dict[str, object] | None = None,
        error_code: str | None = None,
    ) -> bool:
        """Owner-scoped result callback (`POST /v1/actions/{invocationId}/result`)."""
        return self._local_result_broker.report(
            invocation_id=invocation_id,
            owner_id=owner_id,
            status=status,
            output=output,
            error_code=error_code,
        )

    async def handle_call(self, call: ToolCall, ctx: ActionContext) -> Message:
        """Executes the 8-step orchestrator pipeline for a single `ToolCall`."""
        if call.name in META_FUNCTION_NAMES:
            if CAP_HOST_ACTIONS_V1 not in ctx.client_caps:
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload("unknown_tool"),
                    tool_call_id=call.id,
                )
            return await self._handle_meta_call(call, ctx)

        tool_name = from_function_name(call.name)
        if tool_name is None or is_forbidden_tool_name(tool_name):
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("unknown_tool"),
                tool_call_id=call.id,
            )

        if tool_name == BOARD_TOOL:
            return await self._approval_and_board.handle_board_call(
                call,
                owner_id=ctx.owner_id,
                locale=ctx.locale,
                client_caps=ctx.client_caps,
                board_available=ctx.board_available,
                emit_event=ctx.emit_event,
                request_id=ctx.request_id,
            )

        if CAP_HOST_ACTIONS_V1 not in ctx.client_caps:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("unknown_tool"),
                tool_call_id=call.id,
            )

        if tool_name == WORKFLOW_TEMPLATE_TOOL:
            return await self._workflow_runner.handle_template_call(
                call,
                owner_id=ctx.owner_id,
                locale=ctx.locale,
                client_caps=ctx.client_caps,
                board_available=ctx.board_available,
                emit_event=ctx.emit_event,
                request_id=ctx.request_id,
                self_host_id=ctx.self_host_id,
            )

        return await self._handle_host_call(tool_name, call, ctx)

    def assign_board_workflow_slot(
        self,
        assignment: WorkflowSlotAssignment,
    ) -> WorkflowSlotAssignment:
        return self._workflow_slot_store.set_slot(assignment)

    def set_host_default(
        self,
        *,
        owner_id: uuid.UUID,
        pet_id: str,
        family: str,
        host_alias: str,
        candidate_id: uuid.UUID | None = None,
    ) -> HostDefaultConvention:
        """Registers an owner-confirmed `host_default` convention (UDC-10 S10.1)."""
        return self._host_default_store.add_confirmed(
            owner_id=owner_id,
            pet_id=pet_id,
            family=family,
            host_alias=host_alias,
            candidate_id=candidate_id,
        )

    def delete_host_default(
        self,
        candidate_id: uuid.UUID,
        *,
        owner_id: uuid.UUID | None = None,
    ) -> bool:
        """Deletes a `host_default` convention, taking effect immediately (UDC-10 S10.1)."""
        return self._host_default_store.delete(candidate_id, owner_id=owner_id)

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
        return await self._workflow_runner.trigger_board_workflow(
            owner_id=owner_id,
            slot=slot,
            host_alias=host_alias,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            risk=risk,
            approval_id=approval_id,
            wait_for_approval=wait_for_approval,
            request_id=request_id,
            self_host_id=self_host_id,
        )

    async def _handle_meta_call(self, call: ToolCall, ctx: ActionContext) -> Message:
        """`tool_search` / `tool_describe` / `tool_invoke` over the same eligible host tools
        the catalog uses; `tool_invoke` re-enters `_handle_host_call` with every gate."""
        hosts, manifests, grants, online = await load_owner_host_state(
            owner_id=ctx.owner_id,
            host_repo=self._host_repo,
            hub=self._hub,
            self_host_id=ctx.self_host_id,
        )
        eligible = collect_eligible_host_tools(
            hosts=hosts,
            manifests_by_host=manifests,
            grants_by_host=grants,
            online_host_ids=online,
            host_screen_consent=ctx.host_screen_consent,
            now=self._clock(),
        )
        try:
            if call.name == TOOL_SEARCH_FUNCTION:
                output = search_tools(
                    eligible, call.arguments.get("query"), call.arguments.get("limit")
                )
                content = encode_meta_payload("ok", output)
            elif call.name == TOOL_DESCRIBE_FUNCTION:
                content = describe_tool(eligible, call.arguments.get("name"))
            else:
                invoke = parse_invoke(eligible, call.arguments)
                inner_call = ToolCall(
                    id=call.id,
                    name=invoke.function_name,
                    arguments=invoke.arguments,
                )
                return await self._handle_host_call(invoke.tool_name, inner_call, ctx)
        except MetaToolArgumentError as error:
            hint = (
                "Use tool_search to find a valid tool name."
                if error.code == "unknown_tool"
                else "Check the arguments against tool_describe and retry."
            )
            content = _encode_tool_payload(error.code, hint=hint)
        return Message(role=Role.TOOL, content=content, tool_call_id=call.id)

    async def _handle_host_call(
        self,
        tool_name: str,
        call: ToolCall,
        ctx: ActionContext,
    ) -> Message:
        now = self._clock()
        hosts = await self._host_repo.list_hosts(ctx.owner_id)
        active_hosts = [h for h in hosts if h.status == HostStatus.ACTIVE]

        manifests_by_host: dict[HostId, Manifest] = {}
        grants_by_host: dict[HostId, list[Grant]] = {}
        online_hosts_with_tool: list[Host] = []
        eligible_hosts: list[Host] = []
        reference_descriptor: ToolDescriptor | None = None

        for host in active_hosts:
            manifest = await self._host_repo.get_manifest(host.host_id)
            if manifest is None:
                continue
            descriptor = manifest.find_tool(tool_name)
            if descriptor is None:
                continue
            if reference_descriptor is None:
                reference_descriptor = descriptor
            is_self = ctx.self_host_id is not None and str(host.host_id) == ctx.self_host_id
            if not self._hub.is_online(str(host.host_id)) and not is_self:
                continue
            manifests_by_host[host.host_id] = manifest
            grants = await self._host_repo.get_grants(host.host_id)
            grants_by_host[host.host_id] = grants
            online_hosts_with_tool.append(host)
            if is_tool_granted(
                descriptor,
                grants,
                now,
                host_screen_consent=ctx.host_screen_consent,
            ):
                eligible_hosts.append(host)

        if reference_descriptor is None:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("unknown_tool"),
                tool_call_id=call.id,
            )

        try:
            SchemaValidator.validate_schema_definition(reference_descriptor.input_schema)
        except SchemaDefinitionError:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("unknown_tool"),
                tool_call_id=call.id,
            )

        try:
            clean_args, requested_target = SchemaValidator.validate_arguments(
                reference_descriptor.input_schema,
                call.arguments,
                allow_target=True,
            )
        except ArgumentValidationError:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("invalid_args"),
                tool_call_id=call.id,
            )

        if tool_name == "web.navigate":
            raw_url = str(clean_args.get("url", ""))
            try:
                validate_https_public_url(raw_url)
            except ValueError:
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload("invalid_args", {"errorCode": "url_invalid"}),
                    tool_call_id=call.id,
                )

        if tool_name in ("input.click", "input.drag"):
            for k in (
                "xNorm",
                "yNorm",
                "fromXNorm",
                "fromYNorm",
                "toXNorm",
                "toYNorm",
                "startXNorm",
                "startYNorm",
                "endXNorm",
                "endYNorm",
            ):
                val = clean_args.get(k)
                if isinstance(val, int | float) and (val < 0.0 or val > 1.0):
                    return Message(
                        role=Role.TOOL,
                        content=_encode_tool_payload(
                            "invalid_args", {"errorCode": "out_of_bounds"}
                        ),
                        tool_call_id=call.id,
                    )

        last_target = self._last_target_store.get_last_target(ctx.owner_id, ctx.pet_id, now)
        last_alias = last_target[0] if last_target is not None else None
        last_at = last_target[1] if last_target is not None else None
        family_key = reference_descriptor.family.value
        default_alias = (
            ctx.host_defaults.get(family_key)
            if ctx.host_defaults is not None and family_key in ctx.host_defaults
            else self._host_default_store.get_default_alias(
                owner_id=ctx.owner_id,
                pet_id=ctx.pet_id,
                family=reference_descriptor.family,
            )
        )

        resolution = TargetResolver.resolve(
            requested_target=requested_target,
            eligible_hosts=(
                eligible_hosts
                if eligible_hosts or requested_target is not None
                else online_hosts_with_tool
            ),
            all_tool_hosts=online_hosts_with_tool,
            last_target_alias=last_alias,
            last_target_at=last_at,
            now=now,
            host_default_alias=default_alias,
        )
        if resolution.status == TargetResolutionStatus.NO_HOST:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("no_host"),
                tool_call_id=call.id,
            )
        if resolution.status == TargetResolutionStatus.ASK_TARGET or resolution.host is None:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("ask_target", choices=list(resolution.choices)),
                tool_call_id=call.id,
            )

        host = resolution.host
        target_alias = str(host.alias)
        host_manifest = manifests_by_host[host.host_id]
        descriptor = host_manifest.find_tool(tool_name)
        if descriptor is None:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("unknown_tool"),
                tool_call_id=call.id,
            )
        host_grants = grants_by_host.get(host.host_id, [])
        invocation_id = str(uuid.uuid4())

        if descriptor.input_schema != reference_descriptor.input_schema:
            try:
                SchemaValidator.validate_schema_definition(descriptor.input_schema)
                clean_args, _ = SchemaValidator.validate_arguments(
                    descriptor.input_schema,
                    call.arguments,
                    allow_target=True,
                )
            except (SchemaDefinitionError, ArgumentValidationError):
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload("invalid_args"),
                    tool_call_id=call.id,
                )

        if descriptor.permission == ToolPermissionState.UNSUPPORTED:
            await self._audit.record(
                AuditEvent(
                    owner_id=ctx.owner_id,
                    action=AuditAction.HOST_TOOL_DENIED,
                    occurred_at=now,
                    subject=str(host.host_id),
                    request_id=ctx.request_id,
                    metadata={
                        "tool": descriptor.name,
                        "risk": descriptor.risk.value,
                        "status": "unsupported",
                        "reason": "unsupported",
                        "platform": host.platform.value,
                    },
                )
            )
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=invocation_id,
                    target=target_alias,
                    summary=ActionSummary.format_result(
                        descriptor.name, clean_args, target_alias, "unsupported", ctx.locale
                    ),
                    status="unsupported",
                    gesture=gesture_for_action_status("unsupported"),
                )
            )
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("unsupported"),
                tool_call_id=call.id,
            )

        if not is_tool_granted(
            descriptor,
            host_grants,
            now,
            host_screen_consent=ctx.host_screen_consent,
        ):
            needs_u5 = descriptor.observes or descriptor.family in (
                ToolFamily.CU,
                ToolFamily.SCREEN,
                ToolFamily.UI,
                ToolFamily.INPUT,
            )
            deny_reason = (
                "u5_consent_required"
                if needs_u5 and not ctx.host_screen_consent
                else "grant_denied"
            )
            await self._audit.record(
                AuditEvent(
                    owner_id=ctx.owner_id,
                    action=AuditAction.HOST_TOOL_DENIED,
                    occurred_at=now,
                    subject=str(host.host_id),
                    request_id=ctx.request_id,
                    metadata={
                        "tool": descriptor.name,
                        "risk": descriptor.risk.value,
                        "status": "denied",
                        "reason": deny_reason,
                        "platform": host.platform.value,
                    },
                )
            )
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=invocation_id,
                    target=target_alias,
                    summary=ActionSummary.format_result(
                        descriptor.name, clean_args, target_alias, "denied", ctx.locale
                    ),
                    status="denied",
                    gesture=gesture_for_action_status("denied"),
                    error_code=deny_reason,
                    host_id=str(host.host_id),
                )
            )
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("denied", {"errorCode": deny_reason}),
                tool_call_id=call.id,
            )

        active_cu = await self._live_cu_session(ctx.owner_id, host.host_id, now, ctx.request_id)
        # UDC-17 S17.1: the host decides the risk of an input step from the live screen. The
        # backend only ever raises that answer. Hosts that predate preflight keep the legacy
        # estimate; the local self-host path has no preflight round trip.
        is_self_host = ctx.self_host_id is not None and str(host.host_id) == ctx.self_host_id
        preflight: HostPreflight | None = None
        if (
            descriptor.name in _PREFLIGHT_TOOLS
            and not is_self_host
            and host.platform in _PREFLIGHT_PLATFORMS
        ):
            preflight = await self._host_preflight(host.host_id, descriptor.name, clean_args)
        live_label: str | None = None
        if preflight is None:
            effective_risk = self._effective_tool_risk(descriptor, clean_args, active_cu)
        elif preflight.status != "ok":
            # The host would refuse this call: tell the model why, do not ask the owner.
            refusal_code = preflight.error_code or "policy_blocked"
            if active_cu is not None and refusal_code in _CU_SESSION_ENDING_ERROR_CODES:
                await self._end_cu_state(active_cu, refusal_code, now)
            await self._audit.record(
                AuditEvent(
                    owner_id=ctx.owner_id,
                    action=AuditAction.HOST_TOOL_DENIED,
                    occurred_at=now,
                    subject=str(host.host_id),
                    request_id=ctx.request_id,
                    metadata={
                        "tool": descriptor.name,
                        "risk": descriptor.risk.value,
                        "status": preflight.status,
                        "reason": refusal_code,
                        "platform": host.platform.value,
                    },
                )
            )
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=invocation_id,
                    target=target_alias,
                    summary=ActionSummary.format_result(
                        descriptor.name, clean_args, target_alias, preflight.status, ctx.locale
                    ),
                    status=preflight.status,
                    gesture=gesture_for_action_status(preflight.status),
                    error_code=refusal_code,
                    host_id=str(host.host_id),
                )
            )
            if active_cu is not None and not active_cu.active:
                await ctx.emit_event(build_session_event(active_cu, now, ""))
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload(preflight.status, {"errorCode": refusal_code}),
                tool_call_id=call.id,
            )
        else:
            effective_risk = max(
                (
                    descriptor.risk,
                    preflight.effective_risk or RiskTier.R3,
                    self._declared_risk_floor(descriptor, clean_args),
                ),
                key=lambda tier: tier.level,
            )
            live_label = preflight.element_name
        # The approval card names the element the host found on screen, not the model's label.
        summary_args: dict[str, object] = (
            clean_args if live_label is None else {**clean_args, "elementName": live_label}
        )

        if effective_risk != descriptor.risk and not any(
            g.family == descriptor.family and g.is_active(now) and g.max_risk.allows(effective_risk)
            for g in host_grants
        ):
            await self._audit.record(
                AuditEvent(
                    owner_id=ctx.owner_id,
                    action=AuditAction.HOST_TOOL_DENIED,
                    occurred_at=now,
                    subject=str(host.host_id),
                    request_id=ctx.request_id,
                    metadata={
                        "tool": descriptor.name,
                        "risk": effective_risk.value,
                        "status": "denied",
                        "reason": "risk_above_grant",
                        "platform": host.platform.value,
                    },
                )
            )
            await ctx.emit_event(
                ActionResultEvent(
                    invocation_id=invocation_id,
                    target=target_alias,
                    summary=ActionSummary.format_result(
                        descriptor.name, summary_args, target_alias, "denied", ctx.locale
                    ),
                    status="denied",
                    gesture=gesture_for_action_status("denied"),
                    error_code="risk_above_grant",
                    host_id=str(host.host_id),
                )
            )
            # The step needs a higher risk than any active grant allows (for example the
            # host raised it to R3 and the family grant stops at R2): say so, with a code
            # the app can turn into a button to that host's grants.
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("denied", {"errorCode": "risk_above_grant"}),
                tool_call_id=call.id,
            )

        if descriptor.name in ("input.click", "input.type_text", "input.key", "input.drag"):
            cached_node = self._lookup_cached_cu_node(active_cu, clean_args)
            if cached_node is None and descriptor.name == "input.click" and active_cu is not None:
                x_norm = clean_args.get("xNorm")
                y_norm = clean_args.get("yNorm")
                if isinstance(x_norm, int | float) and isinstance(y_norm, int | float):
                    cached_node = hit_test_node_at_coordinates(
                        list(active_cu.cached_nodes.values()), float(x_norm), float(y_norm)
                    )
            if cached_node is None and descriptor.name == "input.drag" and active_cu is not None:
                from_x = clean_args.get("fromXNorm") or clean_args.get("startXNorm")
                from_y = clean_args.get("fromYNorm") or clean_args.get("startYNorm")
                to_x = clean_args.get("toXNorm") or clean_args.get("endXNorm")
                to_y = clean_args.get("toYNorm") or clean_args.get("endYNorm")
                sn = (
                    hit_test_node_at_coordinates(
                        list(active_cu.cached_nodes.values()), float(from_x), float(from_y)
                    )
                    if isinstance(from_x, int | float) and isinstance(from_y, int | float)
                    else None
                )
                en = (
                    hit_test_node_at_coordinates(
                        list(active_cu.cached_nodes.values()), float(to_x), float(to_y)
                    )
                    if isinstance(to_x, int | float) and isinstance(to_y, int | float)
                    else None
                )
                if sn and is_password_ui_node(sn):
                    cached_node = sn
                elif en and is_password_ui_node(en):
                    cached_node = en

            # The model's declared name counts only when it types: a click on "Quên mật
            # khẩu" is not typing into a password field (UDC-17 S17.6).
            explicit_elem_name = clean_args.get("elementName")
            synthetic_node: dict[str, object] = (
                {"name": explicit_elem_name, "role": "textfield"}
                if isinstance(explicit_elem_name, str)
                and descriptor.name not in ("input.click", "input.drag")
                else {}
            )
            if (cached_node is not None and is_password_ui_node(cached_node)) or (
                bool(synthetic_node) and is_password_ui_node(synthetic_node)
            ):
                await self._audit.record(
                    AuditEvent(
                        owner_id=ctx.owner_id,
                        action=AuditAction.HOST_TOOL_DENIED,
                        occurred_at=now,
                        subject=str(host.host_id),
                        request_id=ctx.request_id,
                        metadata={
                            "tool": descriptor.name,
                            "risk": effective_risk.value,
                            "status": "denied",
                            "reason": "password_field_forbidden",
                            "platform": host.platform.value,
                        },
                    )
                )
                await ctx.emit_event(
                    ActionResultEvent(
                        invocation_id=invocation_id,
                        target=target_alias,
                        summary=ActionSummary.format_result(
                            descriptor.name, summary_args, target_alias, "denied", ctx.locale
                        ),
                        status="denied",
                        gesture=gesture_for_action_status("denied"),
                        error_code="password_field_forbidden",
                        host_id=str(host.host_id),
                    )
                )
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload(
                        "denied", {"errorCode": "password_field_forbidden"}
                    ),
                    tool_call_id=call.id,
                )

        template = ctx.cu_template
        if template is not None and descriptor.name == "cu.session_start":
            mismatch = template.mismatch(str(host.host_id), clean_args)
            if mismatch is not None:
                # Refused before anyone is asked: the owner chose to rerun exactly this scope.
                await self._audit.record(
                    AuditEvent(
                        owner_id=ctx.owner_id,
                        action=AuditAction.HOST_TOOL_DENIED,
                        occurred_at=now,
                        subject=str(host.host_id),
                        request_id=ctx.request_id,
                        metadata={
                            "tool": descriptor.name,
                            "risk": effective_risk.value,
                            "status": "denied",
                            "reason": "template_mismatch",
                            "platform": host.platform.value,
                        },
                    )
                )
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload(
                        "denied",
                        {
                            "errorCode": "template_mismatch",
                            "mismatch": mismatch,
                            "expected": template.expected(),
                        },
                    ),
                    tool_call_id=call.id,
                )

        # UDC-17 S17.3 (extend) and the S17.5 follow-up (start): the owner signs exactly the
        # scope on the card, so a scope the card cannot show in full is refused before anyone
        # is asked. A rerun's scope is pinned by its template, so the model is told it cannot
        # trim it (`templatePinned`) instead of being sent into `template_mismatch`.
        scope_too_long: dict[str, object] | None = None
        if descriptor.name == "cu.session_extend" and not ActionSummary.extend_scope_fits_card(
            summary_args, target_alias, ctx.locale
        ):
            scope_too_long = {"errorCode": "extend_scope_too_long"}
        elif descriptor.name == "cu.session_start" and not ActionSummary.start_scope_fits_card(
            summary_args, target_alias, ctx.locale
        ):
            scope_too_long = {"errorCode": "start_scope_too_long"}
            if template is not None:
                scope_too_long["templatePinned"] = True
        if scope_too_long is not None:
            await self._audit.record(
                AuditEvent(
                    owner_id=ctx.owner_id,
                    action=AuditAction.HOST_TOOL_DENIED,
                    occurred_at=now,
                    subject=str(host.host_id),
                    request_id=ctx.request_id,
                    metadata={
                        "tool": descriptor.name,
                        "risk": effective_risk.value,
                        "status": "denied",
                        "reason": str(scope_too_long["errorCode"]),
                        "platform": host.platform.value,
                    },
                )
            )
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("denied", scope_too_long),
                tool_call_id=call.id,
            )

        self._last_target_store.set_last_target(ctx.owner_id, ctx.pet_id, target_alias, now)

        raw_launch_id = (
            clean_args.get("id") or clean_args.get("app_id") or clean_args.get("name") or ""
        )
        is_in_scope_app_launch = (
            active_cu is not None
            and descriptor.name == "app.launch"
            and str(raw_launch_id).strip().lower() in active_cu.allowed_apps
        )
        if active_cu is not None and descriptor.name == "app.launch" and not is_in_scope_app_launch:
            effective_risk = RiskTier.R3

        exempt_by_cu_session = (
            active_cu is not None
            and (
                (
                    descriptor.family in (ToolFamily.SCREEN, ToolFamily.UI, ToolFamily.INPUT)
                    and effective_risk == RiskTier.R2
                )
                or is_in_scope_app_launch
            )
        )

        approval_req: ApprovalRequest | None = None
        if effective_risk.level >= RiskTier.R2.level and not exempt_by_cu_session:
            # ADR-0014 D1: an owner approval given in advance for exactly this call. The grant
            # check above already ran, so suspended grants (and E-Stop, which suspends them)
            # make a standing approval inert. Eligibility is re-checked against the manifest as
            # it is now, so a shrunken allowlist stops the approval from matching.
            standing_hit = None
            if (
                self._standing_repo is not None
                and effective_risk == RiskTier.R2
                and is_standing_eligible(descriptor, clean_args)
            ):
                standing_hit = await self._standing_repo.consume_match(
                    ctx.owner_id,
                    host.host_id,
                    descriptor.name,
                    args_sha256(clean_args),
                    now,
                )
            has_15m_scope = False
            if effective_risk == RiskTier.R2 and self._scope_store is not None:
                scope_res = self._scope_store.has_valid_scope(
                    ctx.owner_id, ctx.pet_id, host.host_id, descriptor.name, now
                )
                if inspect.isawaitable(scope_res):
                    has_15m_scope = await scope_res
                else:
                    has_15m_scope = bool(scope_res)
            expires_at = now + timedelta(seconds=APPROVAL_TTL_SECONDS)
            if standing_hit is not None:
                approval_req = ApprovalRequest(
                    approval_id=uuid.uuid4(),
                    owner_id=ctx.owner_id,
                    host_id=host.host_id,
                    target_alias=target_alias,
                    tool=descriptor.name,
                    args_sha256=args_sha256(clean_args),
                    risk=effective_risk,
                    summary=ActionSummary.format_pending(
                        descriptor.name, summary_args, target_alias, ctx.locale
                    ),
                    created_at=now,
                    expires_at=expires_at,
                    state=ApprovalState.APPROVED,
                    decided_at=now,
                    scope=ApprovalScope.ONCE,
                )
                await self._approval_repo.save(approval_req)
                await self._audit.record(
                    AuditEvent(
                        owner_id=ctx.owner_id,
                        action=AuditAction.HOST_PREAPPROVAL_USED,
                        occurred_at=now,
                        subject=str(host.host_id),
                        request_id=ctx.request_id,
                        metadata={
                            "approvalId": str(standing_hit.approval_id),
                            "tool": descriptor.name,
                            "risk": effective_risk.value,
                            "uses": standing_hit.uses,
                        },
                    )
                )
            elif has_15m_scope:
                approval_req = ApprovalRequest(
                    approval_id=uuid.uuid4(),
                    owner_id=ctx.owner_id,
                    host_id=host.host_id,
                    target_alias=target_alias,
                    tool=descriptor.name,
                    args_sha256=args_sha256(clean_args),
                    risk=effective_risk,
                    summary=ActionSummary.format_pending(
                        descriptor.name, summary_args, target_alias, ctx.locale
                    ),
                    created_at=now,
                    expires_at=expires_at,
                    state=ApprovalState.APPROVED,
                    decided_at=now,
                    scope=ApprovalScope.CONVERSATION_15M,
                )
                await self._approval_repo.save(approval_req)
                await self._audit.record(
                    AuditEvent(
                        owner_id=ctx.owner_id,
                        action=AuditAction.HOST_APPROVAL_GRANTED,
                        occurred_at=now,
                        subject=str(host.host_id),
                        request_id=ctx.request_id,
                        metadata={
                            "approvalId": str(approval_req.approval_id),
                            "tool": descriptor.name,
                            "risk": effective_risk.value,
                            "scope": ApprovalScope.CONVERSATION_15M.value,
                        },
                    )
                )
            else:
                # Offer "always allow this" only when the backend could accept it (ADR-0014).
                offer = (
                    self._standing_repo is not None
                    and effective_risk == RiskTier.R2
                    and is_standing_eligible(descriptor, clean_args)
                )
                standing_offer_tool = descriptor.name if offer else None
                standing_offer_args = dict(clean_args) if offer else None
                # UDC-17 S17.5: only a count travels; nothing is switched on for the owner.
                standing_approved_count: int | None = None
                if offer:
                    approved = await self._approval_repo.count_approved(
                        ctx.owner_id,
                        host.host_id,
                        descriptor.name,
                        args_sha256(clean_args),
                        now - STANDING_SUGGEST_WINDOW,
                    )
                    if approved >= STANDING_SUGGEST_MIN_APPROVALS:
                        standing_approved_count = approved
                pending_summary = ActionSummary.format_pending(
                    descriptor.name, summary_args, target_alias, ctx.locale
                )
                # The card may carry draft text; the stored approval row never does.
                card_summary = pending_summary
                if template is not None and descriptor.name == "cu.session_start":
                    card_summary = ActionSummary.with_template_name(
                        pending_summary, template.name, ctx.locale
                    )
                if (
                    effective_risk == RiskTier.R3
                    and descriptor.name in _COMMIT_INPUT_TOOLS
                    and active_cu is not None
                    and (active_cu.typed_since_commit or active_cu.typed_count > 0)
                ):
                    card_summary = ActionSummary.with_typed_text(
                        pending_summary,
                        active_cu.typed_since_commit,
                        ctx.locale,
                        typed_count=active_cu.typed_count,
                    )
                approval_req = ApprovalRequest(
                    approval_id=uuid.uuid4(),
                    owner_id=ctx.owner_id,
                    host_id=host.host_id,
                    target_alias=target_alias,
                    tool=descriptor.name,
                    args_sha256=args_sha256(clean_args),
                    risk=effective_risk,
                    summary=pending_summary,
                    created_at=now,
                    expires_at=expires_at,
                )
                await self._approval_repo.save(approval_req)
                await self._approval_broker.create(approval_req)
                if active_cu is not None:
                    prompts = await self._cu_session_store.record_prompt(active_cu.session_id)
                    if prompts is not None:
                        active_cu.owner_prompts = prompts

                await ctx.emit_event(
                    ActionPendingEvent(
                        approval_id=str(approval_req.approval_id),
                        target=target_alias,
                        summary=card_summary,
                        risk=approval_req.risk.value,
                        expires_in=APPROVAL_TTL_SECONDS,
                        host_id=str(host.host_id),
                        args_sha256=approval_req.args_sha256,
                        expires_at=expires_at.astimezone(UTC).isoformat(),
                        standing_tool=standing_offer_tool,
                        standing_arguments=standing_offer_args,
                        standing_approved_count=standing_approved_count,
                    )
                )

                decision_state = await self._approval_broker.wait(
                    approval_req.approval_id,
                    timeout_s=float(APPROVAL_TTL_SECONDS),
                )
                await self._approval_and_board.sync_approval_after_wait(
                    approval_req, decision_state, ctx.request_id
                )

                if decision_state != ApprovalState.APPROVED:
                    final_status = (
                        "rejected" if decision_state == ApprovalState.REJECTED else "expired"
                    )
                    await ctx.emit_event(
                        ActionResultEvent(
                            invocation_id=invocation_id,
                            target=target_alias,
                            summary=ActionSummary.format_result(
                                descriptor.name,
                                summary_args,
                                target_alias,
                                final_status,
                                ctx.locale,
                            ),
                            status=final_status,
                            gesture=gesture_for_action_status(final_status),
                        )
                    )
                    return Message(
                        role=Role.TOOL,
                        content=_encode_tool_payload(final_status),
                        tool_call_id=call.id,
                    )

                slot_req = self._approval_broker.get_slot_request(approval_req.approval_id)
                if slot_req is not None:
                    approval_req = slot_req
                else:
                    stored_req = await self._approval_repo.get(approval_req.approval_id)
                    if stored_req is not None:
                        approval_req = stored_req

                if (
                    approval_req.scope == ApprovalScope.CONVERSATION_15M
                    and effective_risk == RiskTier.R2
                    and self._scope_store is not None
                ):
                    grant_res = self._scope_store.grant_scope(
                        ctx.owner_id,
                        ctx.pet_id,
                        host.host_id,
                        descriptor.name,
                        self._clock(),
                    )
                    if inspect.isawaitable(grant_res):
                        await grant_res

            if args_sha256(clean_args) != approval_req.args_sha256:
                await ctx.emit_event(
                    ActionResultEvent(
                        invocation_id=invocation_id,
                        target=target_alias,
                        summary=ActionSummary.format_result(
                            descriptor.name,
                            summary_args,
                            target_alias,
                            "denied",
                            ctx.locale,
                        ),
                        status="denied",
                        gesture=gesture_for_action_status("denied"),
                    )
                )
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload("denied"),
                    tool_call_id=call.id,
                )

        status = "failed"
        duration_ms = 0
        structured_content: dict[str, object] = {}

        if ctx.self_host_id is not None and str(host.host_id) == ctx.self_host_id:
            self._local_result_broker.register(invocation_id, ctx.owner_id)
            await ctx.emit_event(
                LocalActionRequestEvent(
                    invocation_id=invocation_id,
                    approval_id=(
                        str(approval_req.approval_id) if approval_req is not None else None
                    ),
                    target=target_alias,
                    tool=descriptor.name,
                    arguments=clean_args,
                )
            )
            local_report = await self._local_result_broker.wait_report(
                invocation_id,
                timeout_s=DEADLINE_SECONDS,
            )
            raw_status = local_report.status
            if raw_status in ("denied_by_local_policy", "denied"):
                status = "denied"
            elif raw_status in ("permission_denied", "needs_permission"):
                status = "needs_permission"
            elif raw_status in ("ok", "timeout", "failed", "unsupported", "invalid_args"):
                status = raw_status
            else:
                status = "failed"
            if isinstance(local_report.output, dict):
                structured_content = dict(local_report.output)
            if local_report.error_code and "errorCode" not in structured_content:
                structured_content["errorCode"] = local_report.error_code
        else:
            schema_props = (
                descriptor.input_schema.get("properties")
                if isinstance(descriptor.input_schema, dict)
                else None
            )
            supports_latm_script = isinstance(schema_props, dict) and "scriptCode" in schema_props
            code_specialist_used = False

            if (
                descriptor.name == "forge.synthesize_skill"
                and supports_latm_script
                and self._code_synthesizer is not None
                and not str(clean_args.get("scriptCode", "")).strip()
            ):
                draft = await self._code_synthesizer.synthesize_or_refine(
                    goal=str(clean_args.get("goal", "")),
                    runtime=str(clean_args.get("runtime", "python")),
                    tool_name=str(clean_args.get("toolName", "custom_tool")),
                    test_args_json=str(clean_args.get("testArgsJson", "{}")),
                    locale=ctx.locale,
                )
                if draft is not None:
                    clean_args = dict(clean_args)
                    clean_args["scriptCode"] = draft.script_code
                    clean_args["runtime"] = draft.runtime
                    if not str(clean_args.get("toolName", "")).strip():
                        clean_args["toolName"] = draft.tool_name
                    if not str(clean_args.get("testArgsJson", "")).strip():
                        clean_args["testArgsJson"] = draft.test_args_json
                    code_specialist_used = True

            meta: dict[str, object] = {
                "xuna/invocationId": invocation_id,
                "xuna/deadlineMs": DEADLINE_MS,
            }
            if approval_req is not None:
                approval_meta: dict[str, object] = {
                    "approvalId": str(approval_req.approval_id),
                    "argsSha256": approval_req.args_sha256,
                    "expiresAt": approval_req.expires_at.astimezone(UTC).isoformat(),
                    "scope": approval_req.scope.value,
                }
                if approval_req.signature is not None:
                    approval_meta["signature"] = base64.b64encode(approval_req.signature).decode(
                        "ascii"
                    )
                if approval_req.signer_public_key is not None:
                    approval_meta["publicKey"] = base64.b64encode(
                        approval_req.signer_public_key
                    ).decode("ascii")
                meta["xuna/approval"] = approval_meta

            call_params: dict[str, object] = {
                "name": descriptor.name,
                "arguments": clean_args,
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
                    structured_content = dict(sc_raw)
                if isinstance(result_meta, dict):
                    remote_error = result_meta.get("xuna/errorCode")
                    if isinstance(remote_error, str) and "errorCode" not in structured_content:
                        structured_content["errorCode"] = remote_error[:64]

                # Automatic Code-Specialist Self-Debug Loop when dryRunOk is False
                if (
                    descriptor.name == "forge.synthesize_skill"
                    and supports_latm_script
                    and self._code_synthesizer is not None
                    and status == "ok"
                    and structured_content.get("dryRunOk") is False
                    and str(structured_content.get("stderrPreview", "")).strip()
                ):
                    for _ in range(2):
                        if structured_content.get("dryRunOk") is not False:
                            break
                        repaired = await self._code_synthesizer.synthesize_or_refine(
                            goal=str(clean_args.get("goal", "")),
                            runtime=str(clean_args.get("runtime", "python")),
                            tool_name=str(
                                structured_content.get("toolName")
                                or clean_args.get("toolName", "custom_tool")
                            ),
                            test_args_json=str(clean_args.get("testArgsJson", "{}")),
                            broken_script=str(clean_args.get("scriptCode", "")),
                            stderr_preview=str(structured_content.get("stderrPreview", "")),
                            locale=ctx.locale,
                        )
                        if repaired is None:
                            break
                        clean_args = dict(clean_args)
                        clean_args["scriptCode"] = repaired.script_code
                        clean_args["runtime"] = repaired.runtime
                        skill_id_val = str(structured_content.get("skillId", "")).strip()
                        if skill_id_val:
                            clean_args["skillIdToUpdate"] = skill_id_val
                        retry_params: dict[str, object] = {
                            "name": descriptor.name,
                            "arguments": clean_args,
                            "_meta": {
                                "xuna/invocationId": str(uuid.uuid4()),
                                "xuna/deadlineMs": DEADLINE_MS,
                            },
                        }
                        retry_res = await self._hub.send_request(
                            str(host.host_id),
                            "tools/call",
                            retry_params,
                            timeout_s=DEADLINE_SECONDS,
                        )
                        retry_sc = retry_res.get("structuredContent")
                        if isinstance(retry_sc, dict):
                            structured_content = dict(retry_sc)
                            code_specialist_used = True
                if code_specialist_used and structured_content:
                    structured_content["codeSpecialistDelegated"] = True
            except TimeoutError:
                status = "timeout"
                duration_ms = DEADLINE_MS
            except Exception:
                status = "failed"

        await self._update_cu_session_after_call(
            owner_id=ctx.owner_id,
            host_id=host.host_id,
            descriptor=descriptor,
            clean_args=clean_args,
            status=status,
            structured_content=structured_content,
            active_cu=active_cu,
            now=self._clock(),
            target_alias=target_alias,
        )
        # A timed-out call may still have typed on the host, so its text is kept too.
        if active_cu is not None and active_cu.active and status in ("ok", "timeout"):
            if descriptor.name == "input.type_text":
                # Remembered for the next commit card (ADR-0015 D1), signed or not.
                text = clean_args.get("text")
                if isinstance(text, str) and text.strip():
                    active_cu.typed_since_commit.append(text)
                    del active_cu.typed_since_commit[:-_MAX_REMEMBERED_DRAFTS]
                    active_cu.typed_count += 1
                    await self._cu_session_store.set_typed_count(
                        active_cu.session_id, active_cu.typed_count
                    )
            elif (
                status == "ok"
                and effective_risk == RiskTier.R3
                and descriptor.name in _COMMIT_INPUT_TOOLS
            ):
                # The owner signed a commit whose card showed the typed text.
                active_cu.typed_since_commit.clear()
                active_cu.typed_count = 0
                await self._cu_session_store.set_typed_count(active_cu.session_id, 0)

        audit_action = (
            AuditAction.HOST_TOOL_DENIED if status == "denied" else AuditAction.HOST_TOOL_INVOKED
        )
        await self._audit.record(
            AuditEvent(
                owner_id=ctx.owner_id,
                action=audit_action,
                occurred_at=self._clock(),
                subject=str(host.host_id),
                request_id=ctx.request_id,
                metadata={
                    "tool": descriptor.name,
                    "risk": effective_risk.value,
                    "status": status,
                    "durationMs": max(duration_ms, 0),
                    "platform": host.platform.value,
                },
            )
        )

        result_summary = ActionSummary.format_result(
            descriptor.name,
            summary_args if live_label is not None else clean_args,
            target_alias,
            status,
            ctx.locale,
        )
        raw_error = structured_content.get("errorCode")
        await ctx.emit_event(
            ActionResultEvent(
                invocation_id=invocation_id,
                target=target_alias,
                summary=result_summary,
                status=status,
                gesture=gesture_for_action_status(status),
                error_code=raw_error[:64]
                if status != "ok" and isinstance(raw_error, str)
                else None,
                host_id=str(host.host_id) if status != "ok" else None,
            )
        )
        if descriptor.family in (
            ToolFamily.CU,
            ToolFamily.SCREEN,
            ToolFamily.UI,
            ToolFamily.INPUT,
        ):
            # UDC-17 S17.3: the phone's session card follows every step of the session.
            card_now = self._clock()
            card_state = await self._live_cu_session(
                ctx.owner_id, host.host_id, card_now, ctx.request_id
            )
            if card_state is None and active_cu is not None:
                card_state = active_cu
            if card_state is not None:
                await ctx.emit_event(build_session_event(card_state, card_now, result_summary))
        detached_image_b64: str | None = None
        detached_image_mime: str = "image/jpeg"
        if descriptor.name == "ui.observe_hybrid" and isinstance(structured_content, dict):
            sc_obj = structured_content.get("screenshot")
            if isinstance(sc_obj, dict):
                b64_val = sc_obj.pop("dataBase64", None) or sc_obj.pop("bytesBase64", None)
                if isinstance(b64_val, str):
                    detached_image_b64 = b64_val
                if isinstance(sc_obj.get("mimeType"), str):
                    detached_image_mime = str(sc_obj["mimeType"])
            elif isinstance(sc_obj, str):
                sc_str = structured_content.pop("screenshot", None)
                if isinstance(sc_str, str):
                    detached_image_b64 = sc_str
            if not detached_image_b64 and "jpegBase64" in structured_content:
                raw_b64 = structured_content.pop("jpegBase64", None)
                if isinstance(raw_b64, str):
                    detached_image_b64 = raw_b64
            if not detached_image_b64 and "dataBase64" in structured_content:
                raw_b64 = structured_content.pop("dataBase64", None)
                if isinstance(raw_b64, str):
                    detached_image_b64 = raw_b64

        if status == "ok" and not is_repeat_safe(descriptor):
            content = _encode_tool_payload(status, structured_content, repeatSafe=False)
        else:
            content = _encode_tool_payload(status, structured_content)
        return Message(
            role=Role.TOOL,
            content=content,
            tool_call_id=call.id,
            image_base64=detached_image_b64,
            image_mime_type=detached_image_mime,
        )
