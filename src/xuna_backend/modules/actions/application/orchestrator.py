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
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from xuna_backend.modules.actions.application.approval_and_board_service import (
    ApprovalAndBoardService,
)
from xuna_backend.modules.actions.application.autopilot_policy import (
    autopilot_input_risk,
    cu_session_risk_for_autopilot,
    declared_risk_floor,
    is_consequential_web_action,
    resolve_host_trust_preset,
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
    contains_prompt_injection_payload,
    hit_test_node_at_coordinates,
    is_activation_key_combo,
    is_consequential_element_name,
    is_editable_field_role,
    is_hard_denied_target,
    is_password_ui_node,
    redact_ui_tree_nodes,
    session_expiry_reason,
)
from xuna_backend.modules.actions.application.cu_template import CuTemplatePin
from xuna_backend.modules.actions.application.exec_fs_support import (
    missing_preflight_refusal,
    requires_mandatory_preflight,
)
from xuna_backend.modules.actions.application.mesh_call_helper import (
    build_cu_mission_envelope,
    build_pipeline_runner,
    dispatch_mesh_tool,
    emit_host_action_result,
    emit_no_host_required,
    emit_pending_with_battery,
    emit_unresolved_ctx,
    extract_tool_remediation,
    handle_pre_route_ctx,
    resolve_effective_target_reason,
    validate_cu_session_goal,
)
from xuna_backend.modules.actions.application.mesh_prober import MeshProber
from xuna_backend.modules.actions.application.mesh_router import (
    PendingIntentStore,
    PreRouteOutcome,
    record_execution_habit,
    resolve_host_target_for_call,
)
from xuna_backend.modules.actions.application.ports import (
    ApprovalBroker,
    ApprovalRepository,
    ApprovalScopeStore,
    ComputerUseSessionStore,
    HabitRepository,
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
    state_driven_hint,
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
    ActionStreamEvent,
    ActionSummary,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScope,
    ApprovalState,
    ComputerUseSessionEvent,
    EmergencyStopSummary,
    LocalActionRequestEvent,
)
from xuna_backend.modules.actions.domain.job import AutonomousJobState
from xuna_backend.modules.actions.domain.personalization import (
    HostDefaultConvention,
    InMemoryHostDefaultStore,
)
from xuna_backend.modules.actions.domain.preflight_preview import (
    sanitize_preflight_preview,
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
from xuna_backend.modules.actions.domain.tool_names import (
    AUTONOMOUS_JOB_SCHEMA,
    AUTONOMOUS_JOB_TOOL,
    BOARD_TOOL,
    WORKFLOW_TEMPLATE_TOOL,
    from_function_name,
    is_forbidden_tool_name,
)
from xuna_backend.modules.actions.domain.wake_queue import InMemoryWakeQueueStore
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

if TYPE_CHECKING:
    from xuna_backend.modules.actions.application.job_runner import AutonomousJobRunner

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

_logger = logging.getLogger(__name__)
_AUDIT_CODE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_AUDIT_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{0,64}$")

_truncate_value = truncate_tool_value
_encode_tool_payload = encode_tool_payload


def _sanitize_prompt_note_field(raw: str | None, max_len: int) -> str | None:
    if not raw:
        return None
    collapsed = " ".join(
        "".join(ch if (ord(ch) >= 32 and ord(ch) != 127) else " " for ch in raw).split()
    ).strip()
    if not collapsed:
        return None
    if contains_prompt_injection_payload(collapsed):
        return "[redacted]"
    return collapsed[:max_len]


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
PREFLIGHT_TIMEOUT_SECONDS = 3.0
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
    # Wave E (UDC-18): Optional envelope representing a multi-step mission or AutonomousJob plan.
    mission_envelope: dict[str, object] | None = None
    pinned_host_id: HostId | None = None
    follow_me: bool = False
    client_subnet_hash: str | None = None
    client_tz_offset_minutes: int | None = None
    habit_recorded_in_preroute: bool = False
    pre_route_target_required: PreRouteOutcome | None = None
    pinned_target_reason: str | None = None


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
        job_runner: "AutonomousJobRunner | None" = None,
        habit_repo: HabitRepository | None = None,
        pending_intent_store: PendingIntentStore | None = None,
        mesh_prober: MeshProber | None = None,
        wake_queue: InMemoryWakeQueueStore | None = None,
    ) -> None:
        self._standing_repo = standing_repo
        self._job_runner = job_runner
        self._habit_repo = habit_repo
        self._pending_intent_store = pending_intent_store
        self._wake_queue = wake_queue
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
        self._mesh_prober = mesh_prober or MeshProber(hub=hub, clock=self._clock)
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
        self._cross_host_pipelines = build_pipeline_runner(
            self._approval_and_board,
            self._mesh_prober,
            standing_repo,
            pending_intent_store,
            wake_queue,
        )

    async def emergency_stop(
        self,
        *,
        owner_id: uuid.UUID,
        reason: str = "owner_app",
        request_id: str = "",
        host_id: HostId | None = None,
    ) -> EmergencyStopSummary:
        """Cancels active invocations, rejects approvals, sends `xuna/stop` (`UDC-7` S7.3)."""
        now = self._clock()
        for running in await self._cu_session_store.list_active_for_owner(owner_id):
            if host_id is None or running.host_id == host_id:
                await self._end_cu_state(running, reason, now, request_id)
        await self._cu_session_store.stop_all_for_owner(owner_id, reason, now, host_id=host_id)
        summary = await self._approval_and_board.emergency_stop(
            owner_id=owner_id, reason=reason, request_id=request_id, host_id=host_id
        )
        if self._job_runner is not None:
            await self._job_runner.cancel_all_for_owner(owner_id, reason=reason, host_id=host_id)
        if self._wake_queue is not None:
            self._wake_queue.clear_owner(owner_id, host_id=host_id)
        return summary

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
        state = await self._cu_session_store.get_active_for_owner(owner_id, host_id=host_id)
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
        state.last_window.clear()
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

    async def cleanup_dangling_cu_session(
        self,
        *,
        owner_id: uuid.UUID,
        host_id: HostId | None = None,
        reason: str = "turn_aborted",
        request_id: str = "",
    ) -> ComputerUseSessionEvent | None:
        """Closes a dangling CU session on abnormal turn exit, preserving detached jobs.

        Idempotent and non-raising: ends backend state first, notifies the host best-effort
        with a valid `cu.session_end` outcome, audits safely, and returns the ended event.
        """
        raw_reason = (reason or "").strip()[:64]
        clean_reason = raw_reason if _AUDIT_CODE_RE.match(raw_reason) else "turn_aborted"
        raw_rid = (request_id or "").strip()[:64]
        safe_request_id = raw_rid if _AUDIT_REQUEST_ID_RE.match(raw_rid) else ""

        now = self._clock()
        try:
            state = await self._cu_session_store.get_active_for_owner(owner_id, host_id=host_id)
        except Exception:
            _logger.warning("cleanup_dangling_cu_session: store lookup failed", exc_info=True)
            return None
        if state is None or not state.active:
            return None

        if self._job_runner is not None:
            try:
                active_job = await self._job_runner.get_active_for_owner(owner_id)
            except Exception:
                active_job = None
            if (
                active_job is not None
                and not active_job.is_terminal
                and active_job.session_id == state.session_id
            ):
                return None

        if clean_reason in ("turn_max_rounds", "turn_timeout"):
            host_outcome = "budget_exceeded"
        elif clean_reason in ("turn_provider_error", "turn_failed"):
            host_outcome = "failed"
        else:
            host_outcome = "stopped"

        try:
            await self._end_cu_state(state, clean_reason, now, safe_request_id)
        except Exception:
            _logger.warning("cleanup_dangling_cu_session: _end_cu_state failed", exc_info=True)
            state.active = False
            state.stop_reason = clean_reason
            state.cached_nodes.clear()
            state.typed_since_commit.clear()
            state.last_window.clear()
            state.typed_count = 0

        host_confirmed = await self._end_cu_session_on_host(state, host_outcome)
        try:
            await self._audit_cu_session_stop(
                owner_id,
                state.host_id,
                clean_reason,
                host_confirmed,
                safe_request_id,
            )
        except Exception:
            _logger.warning("cleanup_dangling_cu_session: audit stop failed", exc_info=True)
        return build_session_event(state, now, "")

    @staticmethod
    def _record_cu_window_context(
        state: ComputerUseSessionState | None,
        *,
        app_id: object = None,
        window_title: object = None,
        pid: object = None,
        window_id: object = None,
        clear_title_if_app_changed: bool = False,
    ) -> None:
        if state is None or not state.active:
            return
        prev_app = state.last_app_id
        next_app: str | None = None
        if isinstance(app_id, str):
            cleaned_app = " ".join(app_id.split()).strip()[:64]
            if cleaned_app:
                next_app = cleaned_app
                state.last_app_id = cleaned_app
        app_changed = (
            clear_title_if_app_changed
            and next_app is not None
            and (prev_app or "").lower() != next_app.lower()
        )
        if isinstance(window_title, str) and window_title.strip():
            state.last_window_title = window_title
        elif app_changed:
            state.last_window_title = None
        if isinstance(pid, int) and not isinstance(pid, bool) and 1 <= pid <= 2_147_483_647:
            state.last_window["pid"] = str(pid)
        elif app_changed:
            state.last_window.pop("pid", None)
        if (
            isinstance(window_id, int)
            and not isinstance(window_id, bool)
            and 0 <= window_id <= 2_147_483_647
        ):
            state.last_window["windowId"] = str(window_id)
        elif app_changed:
            state.last_window.pop("windowId", None)

    async def active_cu_session_prompt_note(
        self,
        owner_id: uuid.UUID,
        locale: Locale = Locale.VI,
        *,
        host_id: HostId | None = None,
        request_id: str = "",
    ) -> str | None:
        """Returns a bounded, Rule H5-compliant prompt note when a live CU session exists."""
        now = self._clock()
        state = await self._live_cu_session(owner_id, host_id, now, request_id)
        if state is None:
            return None
        remaining_steps = max(state.max_steps - state.steps_used, 0)
        seconds_left = max(
            int(state.max_minutes * 60 - (now - state.started_at_utc).total_seconds()),
            0,
        )
        remaining_minutes = max((seconds_left + 59) // 60, 0) if seconds_left > 0 else 0
        apps_str = ", ".join(state.allowed_apps[:8])
        domains_str = ", ".join(state.allowed_domains[:12])
        target_label = state.target_alias or str(state.host_id)
        app_label = _sanitize_prompt_note_field(state.last_app_id, 64)
        safe_title = _sanitize_prompt_note_field(state.last_window_title, 100)
        if locale == Locale.EN:
            loc_parts: list[str] = []
            if app_label:
                loc_parts.append(f"app={app_label}")
            if safe_title:
                loc_parts.append(f"window='{safe_title}'")
            window_clause = f"; last observed {', '.join(loc_parts)}" if loc_parts else ""
            return (
                f"\n- Active computer-use session '{state.session_id}' on '{target_label}' "
                f"is still running (apps=[{apps_str or 'none'}], "
                f"domains=[{domains_str or 'none'}], "
                f"remainingSteps={remaining_steps}, remainingMinutes={remaining_minutes}"
                f"{window_clause}). Continue inside this session with ui__inspect / "
                "ui__observe_hybrid / input__* instead of restarting or calling app__launch "
                "unless a different app or domain must be added via cu__session_extend."
            )
        loc_parts_vi: list[str] = []
        if app_label:
            loc_parts_vi.append(f"app={app_label}")
        if safe_title:
            loc_parts_vi.append(f"cửa sổ='{safe_title}'")
        window_clause_vi = f"; cửa sổ gần nhất: {', '.join(loc_parts_vi)}" if loc_parts_vi else ""
        return (
            f"\n- Phiên điều khiển '{state.session_id}' trên '{target_label}' vẫn đang mở "
            f"(apps=[{apps_str or 'không có'}], domains=[{domains_str or 'không có'}], "
            f"còn {remaining_steps} bước / {remaining_minutes} phút{window_clause_vi}). "
            "Hãy thao tác tiếp ngay trong phiên này bằng ui__inspect / ui__observe_hybrid / "
            "input__* thay vì mở lại từ đầu hoặc gọi app__launch, trừ khi cần thêm app/tên "
            "miền mới qua cu__session_extend."
        )

    async def record_cu_heartbeat(
        self, *, owner_id: uuid.UUID, session_id: str, request_id: str = ""
    ) -> ComputerUseSessionState | None:
        """Phone heartbeat for the owner's running session (UDC-8 dead-man switch).

        The stop rules run first, so a heartbeat that arrives after the session already
        timed out does not revive it. Returns the running session, or `None`.
        """
        now = self._clock()
        raw_state = await self._cu_session_store.get(session_id)
        if raw_state is None or raw_state.owner_id != owner_id or not raw_state.active:
            return None
        state = await self._live_cu_session(owner_id, raw_state.host_id, now, request_id)
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
    def _is_consequential_web_action(args: dict[str, object], *, autopilot: bool = False) -> bool:
        """Checks whether a browser action targets a consequential or sensitive element."""
        return is_consequential_web_action(args, autopilot=autopilot)

    @classmethod
    def _effective_tool_risk(
        cls,
        descriptor: ToolDescriptor,
        clean_args: dict[str, object],
        active_cu: ComputerUseSessionState | None,
        *,
        autopilot: bool = False,
    ) -> RiskTier:
        if descriptor.name == "mail.send":
            return RiskTier.R3
        if descriptor.name == "mail.create_draft":
            return RiskTier.R2
        if descriptor.name in ("cu.session_start", "cu.session_extend"):
            return cu_session_risk_for_autopilot(clean_args) if autopilot else descriptor.risk
        if descriptor.name in ("web.click", "web.fill"):
            if cls._is_consequential_web_action(clean_args, autopilot=autopilot):
                return RiskTier.R3
            return descriptor.risk
        if autopilot and descriptor.name in (
            "input.click",
            "input.drag",
            "input.key",
            "input.type_text",
            "input.scroll",
            "app.launch",
        ):
            cached = (
                cls._lookup_cached_cu_node(active_cu, clean_args)
                if descriptor.name == "input.click"
                else None
            )
            return autopilot_input_risk(descriptor, clean_args, active_cu, cached_node=cached)
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
                    "cmd+n",
                    "cmd+a",
                    "cmd+c",
                    "cmd+v",
                    "command+n",
                    "command+a",
                    "command+c",
                    "command+v",
                    "ctrl+n",
                    "ctrl+a",
                    "ctrl+c",
                    "ctrl+v",
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
                    clean_args.get("id") or clean_args.get("app_id") or clean_args.get("name") or ""
                )
                app_name = str(raw_app).strip().lower()
                if app_name and app_name not in active_cu.allowed_apps:
                    return RiskTier.R3
        return descriptor.risk

    @staticmethod
    def _declared_risk_floor(
        descriptor: ToolDescriptor,
        clean_args: dict[str, object],
        *,
        autopilot: bool = False,
    ) -> RiskTier:
        """Risk implied by the call's own arguments, combined with host preflight answers."""
        return declared_risk_floor(descriptor, clean_args, autopilot=autopilot)

    async def _resolve_host_trust_preset(self, host: Host, *, is_self_host: bool) -> str:
        return await resolve_host_trust_preset(
            self._hub, str(host.host_id), host.platform, is_self_host=is_self_host
        )

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

        if descriptor.name == "cu.session_start":
            raw_nodes = structured_content.get("nodes")
            dict_nodes: list[dict[str, Any]] = []
            redacted_nodes: list[dict[str, Any]] = []
            if isinstance(raw_nodes, list) and raw_nodes:
                dict_nodes = [n for n in raw_nodes if isinstance(n, dict)]
                redacted_nodes = redact_ui_tree_nodes(dict_nodes)
                structured_content["nodes"] = redacted_nodes
            if status != "ok":
                return
            previous = await self._cu_session_store.get_active_for_owner(owner_id, host_id=host_id)
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
            new_state = ComputerUseSessionState(
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
            )
            if dict_nodes:
                _cache_ui_nodes(new_state, dict_nodes, redacted_nodes)
            self._record_cu_window_context(
                new_state,
                app_id=structured_content.get("foregroundApp"),
                window_title=structured_content.get("windowTitle"),
                pid=structured_content.get("pid"),
                window_id=structured_content.get("windowId"),
            )
            await self._cu_session_store.start(new_state, now)
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

        if descriptor.name == "window.active_context" and status == "ok":
            self._record_cu_window_context(
                active_cu,
                app_id=structured_content.get("appName"),
                window_title=structured_content.get("windowTitle"),
                pid=structured_content.get("pid"),
                window_id=structured_content.get("windowId"),
                clear_title_if_app_changed=True,
            )
            return

        if descriptor.name == "app.launch" and status == "ok":
            self._record_cu_window_context(
                active_cu,
                app_id=structured_content.get("id") or clean_args.get("id"),
                window_title=structured_content.get("windowTitle"),
                pid=structured_content.get("pid"),
                window_id=structured_content.get("windowId"),
                clear_title_if_app_changed=True,
            )
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
            self._record_cu_window_context(
                active_cu,
                app_id=structured_content.get("appId"),
                window_title=structured_content.get("windowTitle"),
                pid=structured_content.get("pid"),
                window_id=structured_content.get("windowId"),
                clear_title_if_app_changed=True,
            )
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
            self._record_cu_window_context(
                active_cu,
                app_id=structured_content.get("appId"),
                window_title=structured_content.get("windowTitle"),
                pid=structured_content.get("pid"),
                window_id=structured_content.get("windowId"),
                clear_title_if_app_changed=True,
            )
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

        if tool_name == AUTONOMOUS_JOB_TOOL:
            return await self._handle_autonomous_job_call(call, ctx)

        return await self._handle_host_call(tool_name, call, ctx)

    async def _handle_autonomous_job_call(
        self,
        call: ToolCall,
        ctx: ActionContext,
    ) -> Message:
        """Handles `autonomous.job` (`start`, `status`, `pause`, `resume`, `steer`, `cancel`)."""
        runner = self._job_runner
        if runner is None:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload(
                    "failed",
                    {"errorCode": "executor_not_configured"},
                ),
                tool_call_id=call.id,
            )
        try:
            SchemaValidator.validate_arguments(AUTONOMOUS_JOB_SCHEMA, call.arguments)
        except ArgumentValidationError:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload(
                    "invalid_args",
                    {"errorCode": "schema_mismatch"},
                ),
                tool_call_id=call.id,
            )

        action = str(call.arguments.get("action") or "").strip().lower()
        if action == "start":
            goal = str(call.arguments.get("goal") or "").strip()
            if not goal:
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload(
                        "invalid_args",
                        {"errorCode": "missing_goal"},
                    ),
                    tool_call_id=call.id,
                )
            hosts, _, _, online_ids = await load_owner_host_state(
                owner_id=ctx.owner_id,
                host_repo=self._host_repo,
                hub=self._hub,
                self_host_id=ctx.self_host_id,
            )
            online_hosts = [
                h
                for h in hosts
                if h.status == HostStatus.ACTIVE
                and str(h.host_id) in online_ids
                and (ctx.pinned_host_id is None or h.host_id == ctx.pinned_host_id)
            ]
            if not online_hosts:
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload(
                        "not_ready",
                        {"errorCode": "no_host"},
                    ),
                    tool_call_id=call.id,
                )
            raw_target = call.arguments.get("target")
            target_host: Host | None = None
            if isinstance(raw_target, str) and raw_target.strip():
                needle = raw_target.strip().lower()
                matched = [h for h in online_hosts if str(h.alias).lower() == needle]
                if len(matched) == 1:
                    target_host = matched[0]
                else:
                    return Message(
                        role=Role.TOOL,
                        content=_encode_tool_payload(
                            "ask_target",
                            {
                                "errorCode": "host_offline",
                                "candidates": [str(h.alias) for h in online_hosts],
                            },
                        ),
                        tool_call_id=call.id,
                    )
            elif len(online_hosts) == 1:
                target_host = online_hosts[0]
            else:
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload(
                        "ask_target",
                        {
                            "errorCode": "ask_target",
                            "candidates": [str(h.alias) for h in online_hosts],
                        },
                    ),
                    tool_call_id=call.id,
                )

            raw_title = call.arguments.get("title")
            title = (
                raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else goal[:60]
            )
            raw_max_steps = call.arguments.get("maxSteps")
            max_steps = (
                int(raw_max_steps)
                if isinstance(raw_max_steps, int) and not isinstance(raw_max_steps, bool)
                else 30
            )
            raw_max_mins = call.arguments.get("maxMinutes")
            max_minutes = (
                int(raw_max_mins)
                if isinstance(raw_max_mins, int) and not isinstance(raw_max_mins, bool)
                else 10
            )
            apps_raw = call.arguments.get("apps")
            apps = [str(a) for a in apps_raw] if isinstance(apps_raw, list) else []
            domains_raw = call.arguments.get("domains")
            domains = [str(d) for d in domains_raw] if isinstance(domains_raw, list) else []

            planned_sub_goals = runner.planner.generate_plan(goal, title=title)
            active_cu = await self._cu_session_store.get_active_for_owner(
                ctx.owner_id, host_id=target_host.host_id
            )
            if (apps or domains) and (
                active_cu is None or active_cu.host_id != target_host.host_id
            ):
                cu_args: dict[str, object] = {
                    "goal": goal[:200],
                    "apps": apps if apps else [],
                    "domains": domains,
                    "maxSteps": min(max_steps, MAX_SESSION_STEPS),
                    "maxMinutes": min(max_minutes, MAX_SESSION_MINUTES),
                    "target": str(target_host.alias),
                }
                mission_ctx = ActionContext(
                    owner_id=ctx.owner_id,
                    pet_id=ctx.pet_id,
                    locale=ctx.locale,
                    client_caps=ctx.client_caps,
                    board_available=ctx.board_available,
                    emit_event=ctx.emit_event,
                    request_id=ctx.request_id,
                    self_host_id=ctx.self_host_id,
                    host_defaults=ctx.host_defaults,
                    host_screen_consent=ctx.host_screen_consent,
                    cu_template=ctx.cu_template,
                    pinned_host_id=ctx.pinned_host_id,
                    mission_envelope={
                        "goal": goal[:200],
                        "apps": apps,
                        "domains": domains,
                        "maxSteps": min(max_steps, MAX_SESSION_STEPS),
                        "maxMinutes": min(max_minutes, MAX_SESSION_MINUTES),
                        "planSteps": [sg.title for sg in planned_sub_goals],
                    },
                )
                cu_call = ToolCall(
                    id=f"{call.id}-cu-start",
                    name="cu__session_start",
                    arguments=cu_args,
                )
                cu_msg = await self._handle_host_call("cu.session_start", cu_call, mission_ctx)
                try:
                    cu_payload = json.loads(cu_msg.content)
                except ValueError:
                    cu_payload = {"status": "failed"}
                if not isinstance(cu_payload, dict) or cu_payload.get("status") != "ok":
                    return Message(
                        role=Role.TOOL,
                        content=cu_msg.content,
                        tool_call_id=call.id,
                    )
                active_cu = await self._cu_session_store.get_active_for_owner(
                    ctx.owner_id, host_id=target_host.host_id
                )

            job = await runner.start_job(
                job_id=uuid.uuid4(),
                owner_id=ctx.owner_id,
                pet_id=ctx.pet_id or "xuna_pet",
                host_id=target_host.host_id,
                session_id=active_cu.session_id if active_cu is not None else None,
                title=title,
                goal=goal,
                max_steps=max_steps,
                sub_goals=planned_sub_goals,
            )
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload(
                    "ok",
                    {
                        "jobId": str(job.job_id),
                        "state": job.state.value,
                        "title": job.title,
                        "goal": job.goal,
                        "maxSteps": job.max_steps,
                        "subGoals": [sg.title for sg in job.sub_goals],
                    },
                    repeatSafe=False,
                ),
                tool_call_id=call.id,
            )

        raw_job_id = call.arguments.get("jobId")
        target_job = None
        if isinstance(raw_job_id, str) and raw_job_id.strip():
            try:
                parsed_uuid = uuid.UUID(raw_job_id.strip())
                candidate_job = await runner.get_job(parsed_uuid)
                if candidate_job is not None and candidate_job.owner_id == ctx.owner_id:
                    target_job = candidate_job
            except ValueError:
                target_job = None
        else:
            target_job = await runner.get_active_for_owner(ctx.owner_id)
            if target_job is None and action == "status":
                recent = await runner.list_for_owner(ctx.owner_id, limit=1)
                if recent:
                    target_job = recent[0]

        if target_job is None:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload(
                    "failed",
                    {"errorCode": "job_not_found"},
                ),
                tool_call_id=call.id,
            )

        try:
            if action == "status":
                current_job = target_job
            elif action == "pause":
                current_job = await runner.pause_job(target_job.job_id)
            elif action == "resume":
                if (
                    target_job.state == AutonomousJobState.AWAITING_R3_CHECKPOINT
                    and target_job.pending_approval_id is not None
                ):
                    prev_req = await self._approval_repo.get(target_job.pending_approval_id)
                    if prev_req is None:
                        return Message(
                            role=Role.TOOL,
                            content=_encode_tool_payload(
                                "failed",
                                {"errorCode": "checkpoint_approval_missing"},
                            ),
                            tool_call_id=call.id,
                        )
                    now = self._clock()
                    expires_at = now + timedelta(seconds=APPROVAL_TTL_SECONDS)
                    target_label = prev_req.target_alias
                    host_obj = (
                        await self._host_repo.get_host(prev_req.host_id)
                        if prev_req.host_id is not None
                        else None
                    )
                    if host_obj is not None:
                        target_label = str(host_obj.alias)
                    fresh_req = ApprovalRequest(
                        approval_id=uuid.uuid4(),
                        owner_id=ctx.owner_id,
                        host_id=prev_req.host_id,
                        target_alias=target_label,
                        tool=prev_req.tool,
                        args_sha256=prev_req.args_sha256,
                        risk=prev_req.risk,
                        summary=prev_req.summary,
                        created_at=now,
                        expires_at=expires_at,
                    )
                    await self._approval_repo.save(fresh_req)
                    await self._approval_broker.create(fresh_req)
                    await ctx.emit_event(
                        ActionPendingEvent(
                            approval_id=str(fresh_req.approval_id),
                            target=target_label,
                            summary=prev_req.summary,
                            risk=fresh_req.risk.value,
                            expires_in=APPROVAL_TTL_SECONDS,
                            host_id=(
                                str(prev_req.host_id) if prev_req.host_id is not None else None
                            ),
                            args_sha256=fresh_req.args_sha256,
                            expires_at=expires_at.astimezone(UTC).isoformat(),
                            target_platform=(
                                host_obj.platform.value if host_obj is not None else None
                            ),
                        )
                    )
                    decision = await self._approval_broker.wait(
                        fresh_req.approval_id,
                        timeout_s=float(APPROVAL_TTL_SECONDS),
                    )
                    await self._approval_and_board.sync_approval_after_wait(
                        fresh_req, decision, ctx.request_id
                    )
                    if decision != ApprovalState.APPROVED:
                        rej_code = "rejected" if decision == ApprovalState.REJECTED else "expired"
                        return Message(
                            role=Role.TOOL,
                            content=_encode_tool_payload(
                                rej_code,
                                {"errorCode": f"approval_{rej_code}"},
                            ),
                            tool_call_id=call.id,
                        )
                    current_job = await runner.resume_job(
                        target_job.job_id,
                        approval_id=fresh_req.approval_id,
                    )
                else:
                    current_job = await runner.resume_job(target_job.job_id)
            elif action == "steer":
                instruction = str(call.arguments.get("instruction") or "").strip()
                if not instruction:
                    return Message(
                        role=Role.TOOL,
                        content=_encode_tool_payload(
                            "invalid_args",
                            {"errorCode": "missing_instruction"},
                        ),
                        tool_call_id=call.id,
                    )
                current_job = await runner.steer_job(target_job.job_id, instruction)
            elif action == "cancel":
                reason = str(call.arguments.get("reason") or "owner_takeover").strip()
                current_job = await runner.cancel_job(
                    target_job.job_id,
                    reason=reason or "owner_takeover",
                )
            else:
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload(
                        "invalid_args",
                        {"errorCode": "invalid_action"},
                    ),
                    tool_call_id=call.id,
                )
        except ValueError as err:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload(
                    "failed",
                    {"errorCode": "invalid_job_transition", "detail": str(err)[:120]},
                ),
                tool_call_id=call.id,
            )

        return Message(
            role=Role.TOOL,
            content=_encode_tool_payload(
                "ok",
                {
                    "jobId": str(current_job.job_id),
                    "state": current_job.state.value,
                    "title": current_job.title,
                    "stepsUsed": current_job.steps_used,
                    "maxSteps": current_job.max_steps,
                    "activeSubGoalIndex": current_job.active_sub_goal_index,
                    "subGoals": [
                        {"title": sg.title, "status": sg.status.value}
                        for sg in current_job.sub_goals
                    ],
                    **({"stopReason": current_job.stop_reason} if current_job.stop_reason else {}),
                },
                repeatSafe=(action == "status"),
            ),
            tool_call_id=call.id,
        )

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
            pinned_host_id=ctx.pinned_host_id,
        )
        try:
            mesh_msg = await dispatch_mesh_tool(
                call,
                ctx,
                self._mesh_prober,
                self._cross_host_pipelines,
                hosts,
                manifests,
                grants,
                online,
            )
            if mesh_msg is not None:
                return mesh_msg
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
        normalized_call_args: dict[str, object] = (
            dict(call.arguments) if isinstance(call.arguments, dict) else call.arguments
        )
        if isinstance(normalized_call_args, dict):
            if tool_name == "app.launch" and isinstance(normalized_call_args.get("id"), str):
                raw_id = str(normalized_call_args["id"]).strip().lower()
                normalized_call_args["id"] = "_".join(raw_id.split())
            elif tool_name in ("cu.session_start", "cu.session_extend"):
                raw_apps = normalized_call_args.get("apps")
                if isinstance(raw_apps, list):
                    normalized_call_args["apps"] = [
                        "_".join(a.strip().lower().split()) if isinstance(a, str) else a
                        for a in raw_apps
                    ]
        pinned = ctx.pinned_host_id if isinstance(ctx.pinned_host_id, HostId) else None
        hosts = await self._host_repo.list_hosts(ctx.owner_id)
        active_hosts = [
            h
            for h in hosts
            if h.status == HostStatus.ACTIVE and (pinned is None or h.host_id == pinned)
        ]

        manifests_by_host: dict[HostId, Manifest] = {}
        grants_by_host: dict[HostId, list[Grant]] = {}
        online_hosts_with_tool: list[Host] = []
        eligible_descriptors: list[ToolDescriptor] = []
        online_descriptors: list[ToolDescriptor] = []
        offline_descriptors: list[ToolDescriptor] = []

        for host in active_hosts:
            manifest = await self._host_repo.get_manifest(host.host_id)
            if manifest is None:
                continue
            descriptor = manifest.find_tool(tool_name)
            if descriptor is None:
                continue
            manifests_by_host[host.host_id] = manifest
            is_self = ctx.self_host_id is not None and str(host.host_id) == ctx.self_host_id
            if not self._hub.is_online(str(host.host_id)) and not is_self:
                offline_descriptors.append(descriptor)
                continue
            grants = await self._host_repo.get_grants(host.host_id)
            grants_by_host[host.host_id] = grants
            online_hosts_with_tool.append(host)
            online_descriptors.append(descriptor)
            if is_tool_granted(
                descriptor,
                grants,
                now,
                host_screen_consent=ctx.host_screen_consent,
            ):
                eligible_descriptors.append(descriptor)

        candidate_descriptors = eligible_descriptors or online_descriptors or offline_descriptors
        if not candidate_descriptors:
            return await emit_no_host_required(
                ctx, call.id, tool_name, self._pending_intent_store, now, self._wake_queue
            )

        valid_schema_descriptors: list[ToolDescriptor] = []
        for cand in candidate_descriptors:
            try:
                SchemaValidator.validate_schema_definition(cand.input_schema)
                valid_schema_descriptors.append(cand)
            except SchemaDefinitionError:
                continue
        if not valid_schema_descriptors:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("unknown_tool"),
                tool_call_id=call.id,
            )

        reference_descriptor: ToolDescriptor | None = None
        clean_args: dict[str, object] = {}
        requested_target: str | None = None
        for cand in valid_schema_descriptors:
            try:
                clean_args, requested_target = SchemaValidator.validate_arguments(
                    cand.input_schema,
                    normalized_call_args,
                    allow_target=True,
                )
                reference_descriptor = cand
                break
            except ArgumentValidationError:
                continue
        if reference_descriptor is None:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("invalid_args"),
                tool_call_id=call.id,
            )

        goal_err = validate_cu_session_goal(tool_name, clean_args)
        if goal_err is not None:
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("invalid_args", {"errorCode": goal_err}),
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

        if ctx.pre_route_target_required is not None:
            return await handle_pre_route_ctx(
                ctx,
                call.id,
                tool_name,
                clean_args,
                reference_descriptor,
                self._pending_intent_store,
                now,
                self._wake_queue,
            )

        routing = await resolve_host_target_for_call(
            tool_name=tool_name,
            clean_args=clean_args,
            requested_target=requested_target,
            ctx=ctx,
            active_hosts=active_hosts,
            manifests_by_host=manifests_by_host,
            grants_by_host=grants_by_host,
            online_hosts_with_tool=online_hosts_with_tool,
            reference_descriptor=reference_descriptor,
            hub=self._hub,
            last_target_store=self._last_target_store,
            host_default_store=self._host_default_store,
            habit_repo=self._habit_repo,
            pending_store=self._pending_intent_store,
            cu_session_store=self._cu_session_store,
            now=now,
        )
        if not routing.resolved or routing.host is None:
            return await emit_unresolved_ctx(
                ctx,
                call.id,
                tool_name,
                routing,
                clean_args,
                reference_descriptor,
                now,
                self._wake_queue,
            )

        host = routing.host
        eff_target_reason = resolve_effective_target_reason(ctx.pinned_target_reason, routing)
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
                    normalized_call_args,
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
            await emit_host_action_result(
                ctx,
                invocation_id,
                host,
                descriptor.name,
                clean_args,
                "unsupported",
                eff_target_reason,
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
            await emit_host_action_result(
                ctx,
                invocation_id,
                host,
                descriptor.name,
                clean_args,
                "denied",
                eff_target_reason,
                deny_reason,
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
        is_autopilot = False
        if descriptor.risk.level >= RiskTier.R2.level or descriptor.name in _PREFLIGHT_TOOLS:
            is_autopilot = (
                await self._resolve_host_trust_preset(host, is_self_host=is_self_host)
                == "autopilot"
            )
        preflight: HostPreflight | None = None
        mandatory_preflight = requires_mandatory_preflight(descriptor.name)
        if (
            (descriptor.name in _PREFLIGHT_TOOLS or mandatory_preflight)
            and not is_self_host
            and host.platform in _PREFLIGHT_PLATFORMS
        ):
            preflight = await self._host_preflight(host.host_id, descriptor.name, clean_args)
        if mandatory_preflight and preflight is None:
            # ADR-0018: exec/fs writes never run on a backend estimate.
            preflight = missing_preflight_refusal()
        live_label: str | None = None
        if preflight is None:
            effective_risk = self._effective_tool_risk(
                descriptor, clean_args, active_cu, autopilot=is_autopilot
            )
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
            await emit_host_action_result(
                ctx,
                invocation_id,
                host,
                descriptor.name,
                clean_args,
                preflight.status,
                eff_target_reason,
                refusal_code,
            )
            if active_cu is not None and not active_cu.active:
                await ctx.emit_event(build_session_event(active_cu, now, ""))
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload(preflight.status, {"errorCode": refusal_code}),
                tool_call_id=call.id,
            )
        else:
            effective_base_risk = (
                RiskTier.R2
                if is_autopilot and descriptor.name in ("cu.session_start", "cu.session_extend")
                else descriptor.risk
            )
            effective_risk = max(
                (
                    effective_base_risk,
                    preflight.effective_risk or RiskTier.R3,
                    self._declared_risk_floor(descriptor, clean_args, autopilot=is_autopilot),
                ),
                key=lambda tier: tier.level,
            )
            live_label = preflight.element_name
        if descriptor.name == "draft.apply":
            req_effect = clean_args.get("effectSha256")
            if (
                preflight is None
                or preflight.effect_sha256 is None
                or not isinstance(req_effect, str)
                or preflight.effect_sha256 != req_effect
            ):
                await emit_host_action_result(
                    ctx,
                    invocation_id,
                    host,
                    descriptor.name,
                    clean_args,
                    "denied",
                    eff_target_reason,
                    "effect_sha256_mismatch",
                )
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload("denied", {"errorCode": "effect_sha256_mismatch"}),
                    tool_call_id=call.id,
                )
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
            await emit_host_action_result(
                ctx,
                invocation_id,
                host,
                descriptor.name,
                summary_args,
                "denied",
                eff_target_reason,
                "risk_above_grant",
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
                await emit_host_action_result(
                    ctx,
                    invocation_id,
                    host,
                    descriptor.name,
                    summary_args,
                    "denied",
                    eff_target_reason,
                    "password_field_forbidden",
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

        if (
            descriptor.name == "cu.session_start"
            and active_cu is not None
            and ctx.cu_template is None
        ):
            raw_req_apps = clean_args.get("apps")
            req_apps = (
                {str(a).strip().lower() for a in raw_req_apps if isinstance(a, str) and a.strip()}
                if isinstance(raw_req_apps, list)
                else set()
            )
            raw_req_domains = clean_args.get("domains")
            req_domains = (
                {
                    str(d).strip().lower()
                    for d in raw_req_domains
                    if isinstance(d, str) and d.strip()
                }
                if isinstance(raw_req_domains, list)
                else set()
            )
            active_apps = {a.strip().lower() for a in active_cu.allowed_apps if a.strip()}
            active_domains = {d.strip().lower() for d in active_cu.allowed_domains if d.strip()}
            if (
                req_apps
                and req_apps <= active_apps
                and req_domains <= active_domains
                and not any(is_hard_denied_target(app) for app in req_apps)
            ):
                reused_output: dict[str, object] = {
                    "sessionId": active_cu.session_id,
                    "reused": True,
                    "apps": list(active_cu.allowed_apps),
                    "domains": list(active_cu.allowed_domains),
                }
                if active_cu.last_app_id is not None:
                    reused_output["foregroundApp"] = active_cu.last_app_id
                    norm_last_app = "_".join(active_cu.last_app_id.strip().lower().split())
                    if (
                        active_cu.last_app_id.strip().lower() in req_apps
                        or norm_last_app in req_apps
                    ):
                        reused_output["targetAlreadyFocused"] = True
                if active_cu.last_window_title is not None:
                    reused_output["windowTitle"] = active_cu.last_window_title
                await self._audit.record(
                    AuditEvent(
                        owner_id=ctx.owner_id,
                        action=AuditAction.HOST_TOOL_INVOKED,
                        occurred_at=now,
                        subject=str(host.host_id),
                        request_id=ctx.request_id,
                        metadata={
                            "tool": descriptor.name,
                            "risk": effective_risk.value,
                            "status": "ok",
                            "durationMs": 0,
                            "platform": host.platform.value,
                            "reused": True,
                        },
                    )
                )
                reuse_summary = await emit_host_action_result(
                    ctx, invocation_id, host, descriptor.name, clean_args, "ok", eff_target_reason
                )
                await ctx.emit_event(build_session_event(active_cu, now, reuse_summary))
                reuse_hint = (
                    "Reused active computer-use session (existing scope and budgets preserved). "
                    "Do NOT call app__launch if targetAlreadyFocused is true; continue with "
                    "ui__inspect, ui__observe_hybrid, or ui__find on the current window."
                )
                return Message(
                    role=Role.TOOL,
                    content=_encode_tool_payload("ok", reused_output, hint=reuse_hint),
                    tool_call_id=call.id,
                )

        raw_launch_id = (
            clean_args.get("id") or clean_args.get("app_id") or clean_args.get("name") or ""
        )
        is_in_scope_app_launch = (
            active_cu is not None
            and descriptor.name == "app.launch"
            and str(raw_launch_id).strip().lower() in active_cu.allowed_apps
        )
        if (
            active_cu is not None
            and descriptor.name == "app.launch"
            and not is_in_scope_app_launch
            and not is_autopilot
        ):
            effective_risk = RiskTier.R3

        exempt_by_cu_session = active_cu is not None and (
            (
                descriptor.family in (ToolFamily.SCREEN, ToolFamily.UI, ToolFamily.INPUT)
                and effective_risk == RiskTier.R2
            )
            or is_in_scope_app_launch
        )
        exempt_by_autopilot = is_autopilot and effective_risk == RiskTier.R2

        approval_req: ApprovalRequest | None = None
        if (
            effective_risk.level >= RiskTier.R2.level
            and not exempt_by_cu_session
            and not exempt_by_autopilot
        ):
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
                effect_hash_v2 = (
                    preflight.effect_sha256
                    if descriptor.name == "draft.apply" and preflight is not None
                    else None
                )
                is_effect_v2 = effect_hash_v2 is not None
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
                    approval_version=2 if is_effect_v2 else 1,
                    effect_sha256=effect_hash_v2,
                )
                await self._approval_repo.save(approval_req)
                await self._approval_broker.create(approval_req)
                if active_cu is not None:
                    prompts = await self._cu_session_store.record_prompt(active_cu.session_id)
                    if prompts is not None:
                        active_cu.owner_prompts = prompts

                mission_env = ctx.mission_envelope or build_cu_mission_envelope(
                    descriptor.name, clean_args, active_cu
                )

                await emit_pending_with_battery(
                    ctx=ctx,
                    hosts=hosts,
                    hub=self._hub,
                    host=host,
                    approval_req=approval_req,
                    card_summary=card_summary,
                    expires_at_iso=expires_at.astimezone(UTC).isoformat(),
                    standing_tool=standing_offer_tool,
                    standing_args=standing_offer_args,
                    standing_count=standing_approved_count,
                    mission_env=mission_env,
                    eff_target_reason=eff_target_reason,
                    host_repo=self._host_repo,
                    preview=(
                        sanitize_preflight_preview(preflight.preview)
                        if preflight is not None and preflight.preview is not None
                        else None
                    ),
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
                    await emit_host_action_result(
                        ctx,
                        invocation_id,
                        host,
                        descriptor.name,
                        summary_args,
                        final_status,
                        eff_target_reason,
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
                await emit_host_action_result(
                    ctx,
                    invocation_id,
                    host,
                    descriptor.name,
                    summary_args,
                    "denied",
                    eff_target_reason,
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
                if approval_req.approval_version != 1:
                    approval_meta["version"] = approval_req.approval_version
                if approval_req.effect_sha256 is not None:
                    approval_meta["effectSha256"] = approval_req.effect_sha256
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

        if status == "ok" and self._habit_repo is not None:
            await record_execution_habit(
                self._habit_repo,
                ctx=ctx,
                descriptor=descriptor,
                clean_args=clean_args,
                host_id=host.host_id,
                now=self._clock(),
            )
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

        is_failure = status != "ok" or structured_content.get("status") == "failed"
        raw_error = structured_content.get("errorCode")
        err_code = raw_error[:64] if is_failure and isinstance(raw_error, str) else None
        remediation = extract_tool_remediation(status, structured_content)
        result_summary = await emit_host_action_result(
            ctx,
            invocation_id,
            host,
            descriptor.name,
            summary_args if live_label is not None else clean_args,
            status,
            eff_target_reason,
            err_code,
            remediation=remediation,
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

        extra_payload: dict[str, object] = {}
        if status == "ok" and not is_repeat_safe(descriptor):
            extra_payload["repeatSafe"] = False
        if descriptor.name == "cu.session_start" and status == "ok":
            if structured_content.get("targetAlreadyFocused") is True:
                raw_nodes_hint = structured_content.get("nodes")
                has_nodes = isinstance(raw_nodes_hint, list) and len(raw_nodes_hint) > 0
                if has_nodes:
                    extra_payload["hint"] = (
                        "Session opened and target app is already focused in the foreground "
                        "with an initial UI snapshot (`snapshotId`, `nodes`). Do NOT call "
                        "app__launch; act directly on the returned element refs or call "
                        "ui__observe_hybrid / ui__find if you need visual confirmation."
                    )
                else:
                    extra_payload["hint"] = (
                        "Session opened and target app is already focused in the foreground. "
                        "Do NOT call app__launch; call ui__inspect or ui__observe_hybrid "
                        "directly on the current window."
                    )
            else:
                extra_payload["hint"] = (
                    "Session opened. If the target app is not yet in the foreground, call "
                    "app__launch(id=...) from `apps` (or web__navigate for web pages) before "
                    "calling ui__inspect or ui__observe_hybrid."
                )
        elif status == "ok" and state_driven_hint(structured_content) is None:
            if descriptor.name == "input.key" and is_activation_key_combo(
                str(clean_args.get("combo", ""))
            ):
                extra_payload["hint"] = (
                    "Activation key dispatched (status='ok'), which does NOT prove the goal "
                    "succeeded (Rule H1: e.g., Enter in a search field only shows results). "
                    "Check `changes`, `windowTitle`, `media__now_playing`, or call ui__find / "
                    "ui__inspect / ui__observe_hybrid and activate the target result before "
                    "reporting completion."
                )
            elif descriptor.name == "input.click" and clean_args.get("hover") is True:
                extra_payload["hint"] = (
                    "Hover completed (hover=true); no click was sent yet. Read `changes.added` "
                    "or call ui__inspect / ui__observe_hybrid to click any newly revealed "
                    "control before reporting completion."
                )
            elif descriptor.name == "input.click" and not structured_content.get("changes"):
                extra_payload["hint"] = (
                    "Click dispatched (status='ok') without a UI `changes` diff. Rule H1: "
                    "verify real state via `windowTitle`, `media__now_playing`, or ui__inspect / "
                    "ui__observe_hybrid before reporting completion."
                )
            elif (
                descriptor.name == "app.launch"
                and not clean_args.get("openTarget")
                and not structured_content.get("nodes")
            ):
                extra_payload["hint"] = (
                    "App launch/activation dispatched without `openTarget` or UI `nodes`. "
                    "Rule H1: focusing an app does not complete an in-app task; verify "
                    "`windowTitle` via window__active_context or use a CU session to inspect "
                    "and complete the task."
                )
        content = _encode_tool_payload(status, structured_content, **extra_payload)
        return Message(
            role=Role.TOOL,
            content=content,
            tool_call_id=call.id,
            image_base64=detached_image_b64,
            image_mime_type=detached_image_mime,
        )
