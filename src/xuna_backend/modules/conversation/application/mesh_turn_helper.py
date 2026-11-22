"""Conversation turn helpers for Spatial Device Mesh routing, repair & WS validation.

Bridges `MeshPreRouter`, scoped catalog loading, live `repair_dispatch` streaming, and
strict WebSocket `hello`/`turn.commit` schema validation without importing `core.*`,
frameworks, or `infrastructure` (enforced by `test_architecture.py`), and without
hardcoded app names, host names, or OS paths (Rule H5).
"""

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)

from xuna_backend.modules.actions.application.catalog import load_tool_catalog_for_owner
from xuna_backend.modules.actions.application.mesh_router import (
    MeshPreRouter,
    PreRouteOutcome,
    format_host_roster,
    wire_candidate_dict,
)
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    EventSink,
    Orchestrator,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionStreamEvent,
    ActionTargetRequiredEvent,
    HostWakeRequestEvent,
)
from xuna_backend.modules.actions.domain.redaction import redact_sensitive_text
from xuna_backend.modules.actions.domain.tool_names import to_function_name
from xuna_backend.modules.actions.domain.wake_queue import (
    WAKE_QUEUE_TTL,
    DeferredWakeStep,
    WakeQueueFullError,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.cognition.application.ports import TextGenerator
from xuna_backend.modules.cognition.domain.types import (
    ChatFinished,
    ChatRequest,
    Locale,
    Message,
    Role,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from xuna_backend.modules.consent.application.ports import ConsentRepository
from xuna_backend.modules.conversation.application.chat_turn import (
    MAX_EXPERT_INPUT_CHARS,
    CuCleanupCallback,
    TurnEvent,
    chat_turn,
)
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.entities import HostId, RiskTier, validate_non_zero_uuid

_SUBNET_HASH_PATTERN = r"^[0-9a-f]{16,64}$"
NonZeroHostUuid = Annotated[uuid.UUID, AfterValidator(validate_non_zero_uuid)]


def _require_int_schema_version(v: object) -> object:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError("schemaVersion must be an integer")
    return v


class WsHelloPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["hello"]
    schemaVersion: Annotated[Literal[1], BeforeValidator(_require_int_schema_version)]
    auth: dict[str, Any] = Field(default_factory=dict)
    petId: Any = None
    speak: bool = True
    locale: str | None = None
    voice: str | None = None
    mode: str | None = None
    audio: dict[str, Any] | None = None
    clientCapabilities: list[str] | None = Field(default=None, max_length=16)
    boardHid: dict[str, Any] | None = None
    selfHostId: str | None = Field(default=None, max_length=64)
    targetHostId: NonZeroHostUuid | None = None
    followMe: bool | None = None
    clientSubnetHash: str | None = Field(default=None, max_length=64, pattern=_SUBNET_HASH_PATTERN)


class WsTurnCommitPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["turn.commit"]
    text: str = Field(default="", max_length=MAX_EXPERT_INPUT_CHARS)
    context: dict[str, Any] = Field(default_factory=dict)
    mode: str | None = None
    voice: str | None = None
    clientCapabilities: list[str] | None = Field(default=None, max_length=16)
    boardHid: dict[str, Any] | None = None
    selfHostId: str | None = Field(default=None, max_length=64)
    targetHostId: NonZeroHostUuid | None = None
    followMe: bool | None = None
    clientSubnetHash: str | None = Field(default=None, max_length=64, pattern=_SUBNET_HASH_PATTERN)


class _WsTurnContextMeshFields(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    targetHostId: NonZeroHostUuid | None = None
    followMe: bool | None = None
    clientSubnetHash: str | None = Field(default=None, max_length=64, pattern=_SUBNET_HASH_PATTERN)


def parse_ws_hello(raw: dict[str, Any]) -> WsHelloPayload:
    return WsHelloPayload.model_validate(raw)


def parse_ws_turn_commit(
    raw: dict[str, Any],
) -> tuple[WsTurnCommitPayload, _WsTurnContextMeshFields]:
    parsed = WsTurnCommitPayload.model_validate(raw)
    ctx_fields = _WsTurnContextMeshFields.model_validate(parsed.context)
    return parsed, ctx_fields


def parse_ws_optional_host_id(raw: object) -> HostId | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("invalid_host_id")
    return HostId.from_str(raw.strip())


def tz_offset_minutes_from_dt(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    offset = dt.utcoffset()
    if offset is None:
        return None
    return int(offset.total_seconds() // 60)


def is_validation_error(err: BaseException) -> bool:
    return isinstance(err, ValidationError | ValueError)


@dataclass(frozen=True, slots=True)
class MeshTurnPrepResult:
    outcome: PreRouteOutcome
    tools: tuple[ToolSpec, ...]
    host_roster: str | None
    host_screen_consent: bool
    pinned_host_id: HostId | None
    habit_recorded_in_preroute: bool
    effective_text: str
    target_required_event: ActionTargetRequiredEvent | None
    short_circuit: bool
    pre_route_target_required: PreRouteOutcome | None
    pinned_target_reason: str | None
    effective_client_caps: tuple[str, ...] = ()
    wake_request_event: HostWakeRequestEvent | None = None
    error_code: str | None = None


def _build_target_required_event(outcome: PreRouteOutcome) -> ActionTargetRequiredEvent:
    reason_code = outcome.reason_code or "ask_target"
    target_reason = (
        outcome.target_reason.summary
        if outcome.target_reason is not None
        else "Please specify which host to target"
    )
    candidates = tuple(wire_candidate_dict(c) for c in outcome.candidates)
    return ActionTargetRequiredEvent(
        invocation_id=str(uuid.uuid4()),
        tool="",
        reason_code=reason_code,
        target_reason=target_reason,
        candidates=candidates,
    )


async def prepare_mesh_turn(
    *,
    owner_id: uuid.UUID,
    pet_id: str,
    text: str,
    client_caps: tuple[str, ...],
    board_available: bool,
    self_host_id: str | None,
    target_host_id: HostId | None,
    follow_me: bool | None,
    tz_offset_minutes: int | None,
    include_autonomous_job: bool,
    consent_repo: ConsentRepository | None,
    host_repo: HostRepository | None,
    hub: HostSessionHub | None,
    audit: AuditLog | None,
    pre_router: MeshPreRouter | None,
) -> MeshTurnPrepResult:
    host_screen_consent = False
    if client_caps and consent_repo is not None:
        try:
            consent_state = await consent_repo.current(owner_id)
            host_screen_consent = consent_state.host_screen
        except Exception:
            host_screen_consent = False

    lowered = text.lstrip().lower()
    is_slash = lowered.startswith(("/may", "/follow"))
    has_pending_fn = (
        getattr(pre_router, "has_pending_intent", None) if pre_router is not None else None
    )
    has_pending = bool(has_pending_fn(owner_id, pet_id)) if callable(has_pending_fn) else False
    should_pre_route = (
        bool(client_caps)
        or target_host_id is not None
        or follow_me is not None
        or is_slash
        or has_pending
    )

    if not should_pre_route or host_repo is None or hub is None:
        return MeshTurnPrepResult(
            outcome=PreRouteOutcome(kind="pass"),
            tools=(),
            host_roster=None,
            host_screen_consent=host_screen_consent,
            pinned_host_id=None,
            habit_recorded_in_preroute=False,
            effective_text=text,
            target_required_event=None,
            short_circuit=False,
            pre_route_target_required=None,
            pinned_target_reason=None,
            effective_client_caps=client_caps,
        )

    hosts = await host_repo.list_hosts(owner_id)
    outcome = (
        await pre_router.route(
            owner_id=owner_id,
            pet_id=pet_id,
            text=text,
            hosts=hosts,
            hub=hub,
            target_host_id=target_host_id,
            follow_me=follow_me,
            self_host_id=self_host_id,
            client_tz_offset_minutes=tz_offset_minutes,
        )
        if pre_router is not None
        else PreRouteOutcome(kind="pass")
    )

    if outcome.kind == "not_found":
        return MeshTurnPrepResult(
            outcome=outcome,
            tools=(),
            host_roster=None,
            host_screen_consent=host_screen_consent,
            pinned_host_id=None,
            habit_recorded_in_preroute=False,
            effective_text=text,
            target_required_event=None,
            short_circuit=True,
            pre_route_target_required=None,
            pinned_target_reason=None,
            effective_client_caps=client_caps,
        )

    if outcome.kind == "slash_handled":
        return MeshTurnPrepResult(
            outcome=outcome,
            tools=(),
            host_roster=None,
            host_screen_consent=host_screen_consent,
            pinned_host_id=outcome.pinned_host_id,
            habit_recorded_in_preroute=False,
            effective_text=text,
            target_required_event=None,
            short_circuit=True,
            pre_route_target_required=None,
            pinned_target_reason=None,
            effective_client_caps=client_caps,
        )

    if outcome.kind == "repair_dispatch":
        effective_caps = client_caps or outcome.replay_client_caps
        if not client_caps and consent_repo is not None:
            try:
                consent_state = await consent_repo.current(owner_id)
                host_screen_consent = consent_state.host_screen
            except Exception:
                host_screen_consent = False
        repair_tools: tuple[ToolSpec, ...] = ()
        repair_roster: str | None = None
        if not outcome.replay_tool and effective_caps:
            repair_tools = await load_tool_catalog_for_owner(
                owner_id=owner_id,
                client_caps=effective_caps,
                board_available=board_available,
                host_repo=host_repo,
                hub=hub,
                self_host_id=self_host_id,
                host_screen_consent=host_screen_consent,
                include_autonomous_job=include_autonomous_job,
                audit=audit,
                pinned_host_id=outcome.pinned_host_id,
                include_mesh_tools=True,
            )
            if repair_tools:
                repair_roster = (
                    format_host_roster(
                        hosts=hosts,
                        hub=hub,
                        owner_id=owner_id,
                        pinned_host_id=outcome.pinned_host_id,
                        self_host_id=self_host_id,
                    )
                    or None
                )
        return MeshTurnPrepResult(
            outcome=outcome,
            tools=repair_tools,
            host_roster=repair_roster,
            host_screen_consent=host_screen_consent,
            pinned_host_id=outcome.pinned_host_id,
            habit_recorded_in_preroute=outcome.habit_recorded_in_preroute,
            effective_text=(outcome.cleaned_text or text if not outcome.replay_tool else text),
            target_required_event=None,
            short_circuit=False,
            pre_route_target_required=None,
            pinned_target_reason=(
                outcome.target_reason.summary if outcome.target_reason is not None else None
            ),
            effective_client_caps=effective_caps,
        )

    pinned_host_id: HostId | None = outcome.pinned_host_id
    if pinned_host_id is None and outcome.kind in ("pass", "pin") and not outcome.skip_focus_pin:
        focus_fn = getattr(hub, "get_focus_state", None)
        focus_state = focus_fn(owner_id) if callable(focus_fn) else None
        pinned_host_id = focus_state.focused_host_id if focus_state is not None else None

    tools: tuple[ToolSpec, ...] = ()
    host_roster: str | None = None
    if client_caps:
        tools = await load_tool_catalog_for_owner(
            owner_id=owner_id,
            client_caps=client_caps,
            board_available=board_available,
            host_repo=host_repo,
            hub=hub,
            self_host_id=self_host_id,
            host_screen_consent=host_screen_consent,
            include_autonomous_job=include_autonomous_job,
            audit=audit,
            pinned_host_id=pinned_host_id if outcome.kind in ("pass", "pin") else None,
            include_mesh_tools=True,
        )
        if tools:
            host_roster = (
                format_host_roster(
                    hosts=hosts,
                    hub=hub,
                    owner_id=owner_id,
                    pinned_host_id=pinned_host_id if outcome.kind in ("pass", "pin") else None,
                    self_host_id=self_host_id,
                )
                or None
            )

    if outcome.kind == "target_required":
        tr_event = _build_target_required_event(outcome)
        if is_slash or not tools or outcome.reason_code == "host_offline":
            wake_req_event: HostWakeRequestEvent | None = None
            wake_err_code: str | None = None
            if (
                outcome.reason_code == "host_offline"
                and outcome.candidates
                and outcome.candidates[0].wol_mac
            ):
                cand = outcome.candidates[0]
                wake_queue = getattr(pre_router, "wake_queue", None)
                now_fn = getattr(pre_router, "clock_now", None)
                check_now = now_fn() if callable(now_fn) else datetime.now(UTC)
                enqueued = True
                if wake_queue is not None:
                    try:
                        wake_queue.enqueue(
                            DeferredWakeStep(
                                step_id=uuid.uuid4(),
                                owner_id=owner_id,
                                pet_id=pet_id,
                                host_id=cand.host.host_id,
                                host_alias=cand.host.alias.value,
                                tool="",
                                arguments={},
                                risk=RiskTier.R2,
                                created_at=check_now,
                                expires_at=check_now + WAKE_QUEUE_TTL,
                                summary=redact_sensitive_text(text).text[:120],
                                client_caps=tuple(client_caps),
                            ),
                            check_now,
                        )
                    except WakeQueueFullError:
                        enqueued = False
                        wake_err_code = "wake_queue_full"
                        tr_event = ActionTargetRequiredEvent(
                            invocation_id=tr_event.invocation_id,
                            tool=tr_event.tool,
                            reason_code=tr_event.reason_code,
                            target_reason=(
                                f"{tr_event.target_reason}; "
                                "deferred wake queue is full (wake_queue_full)"
                            ),
                            candidates=tr_event.candidates,
                            error_code="wake_queue_full",
                        )
                if enqueued:
                    wake_req_event = HostWakeRequestEvent(
                        invocation_id=str(uuid.uuid4()),
                        host_id=str(cand.host.host_id),
                        target=cand.host.alias.value,
                        mac_address=cand.wol_mac,
                    )
            return MeshTurnPrepResult(
                outcome=outcome,
                tools=(),
                host_roster=None,
                host_screen_consent=host_screen_consent,
                pinned_host_id=None,
                habit_recorded_in_preroute=False,
                effective_text=outcome.cleaned_text or text,
                target_required_event=tr_event,
                short_circuit=True,
                pre_route_target_required=None,
                pinned_target_reason=None,
                effective_client_caps=client_caps,
                wake_request_event=wake_req_event,
                error_code=wake_err_code,
            )
        return MeshTurnPrepResult(
            outcome=outcome,
            tools=tools,
            host_roster=host_roster,
            host_screen_consent=host_screen_consent,
            pinned_host_id=None,
            habit_recorded_in_preroute=False,
            effective_text=outcome.cleaned_text or text,
            target_required_event=tr_event,
            short_circuit=False,
            pre_route_target_required=outcome,
            pinned_target_reason=None,
            effective_client_caps=client_caps,
        )

    return MeshTurnPrepResult(
        outcome=outcome,
        tools=tools,
        host_roster=host_roster,
        host_screen_consent=host_screen_consent,
        pinned_host_id=pinned_host_id,
        habit_recorded_in_preroute=outcome.habit_recorded_in_preroute,
        effective_text=outcome.cleaned_text or text,
        target_required_event=None,
        short_circuit=False,
        pre_route_target_required=None,
        pinned_target_reason=(
            outcome.target_reason.summary if outcome.target_reason is not None else None
        ),
        effective_client_caps=client_caps,
    )


async def stream_short_circuit_turn(
    prep: MeshTurnPrepResult,
    locale: Locale,
) -> AsyncIterator[TurnEvent]:
    if prep.outcome.kind == "target_required":
        if prep.target_required_event is not None:
            yield prep.target_required_event
            if prep.wake_request_event is not None:
                yield prep.wake_request_event
            yield TextDelta(prep.target_required_event.target_reason)
    elif prep.outcome.kind == "slash_handled":
        notice = prep.outcome.focus_notice
        if notice is not None and notice.alias:
            msg = (
                f"Đã chuyển sang {notice.alias}."
                if locale == Locale.VI
                else f"Focused on {notice.alias}."
            )
        elif notice is not None and notice.follow_me:
            msg = "Đã bật chế độ Follow Me." if locale == Locale.VI else "Follow Me enabled."
        else:
            msg = (
                "Đã chuyển về chế độ tự động."
                if locale == Locale.VI
                else "Switched to auto host selection."
            )
        yield TextDelta(msg)
    yield ChatFinished(
        provider="mesh_pre_router",
        model="deterministic",
        latency_ms=0,
        input_tokens=0,
        output_tokens=0,
    )


async def stream_with_target_fallback(
    events: AsyncIterator[TurnEvent],
    fallback_event: ActionTargetRequiredEvent | None,
) -> AsyncIterator[TurnEvent]:
    saw_target_required = False
    async for ev in events:
        if isinstance(ev, ActionTargetRequiredEvent):
            saw_target_required = True
        elif (
            isinstance(ev, ChatFinished) and fallback_event is not None and not saw_target_required
        ):
            yield fallback_event
            saw_target_required = True
        yield ev


async def execute_repair_dispatch(
    *,
    outcome: PreRouteOutcome,
    generator: TextGenerator,
    base_request: ChatRequest,
    orchestrator: Orchestrator,
    action_ctx_factory: Callable[[EventSink], ActionContext],
    sections: bool,
    locale_vi: bool,
    code_generator: TextGenerator | None = None,
    code_timeout_s: float | None = None,
    cu_cleanup: CuCleanupCallback | None = None,
) -> AsyncIterator[TurnEvent]:
    if not outcome.replay_tool:

        async def _run_repair_tool(tc: ToolCall, emit: EventSink) -> Message:
            return await orchestrator.handle_call(tc, action_ctx_factory(emit))

        async for ev in chat_turn(
            generator,
            base_request,
            sections=sections,
            locale_vi=locale_vi,
            tool_runner=_run_repair_tool if base_request.tools else None,
            code_generator=code_generator,
            code_timeout_s=code_timeout_s,
            cu_cleanup=cu_cleanup,
        ):
            yield ev
        return

    fn_name = (
        outcome.replay_tool
        if "__" in outcome.replay_tool
        else to_function_name(outcome.replay_tool)
    )
    call = ToolCall(
        id=f"repair_{uuid.uuid4().hex[:8]}",
        name=fn_name,
        arguments=dict(outcome.replay_arguments or {}),
    )
    event_queue: asyncio.Queue[ActionStreamEvent | None] = asyncio.Queue()

    async def _emit(ev: ActionStreamEvent) -> None:
        await event_queue.put(ev)

    async def _run_call() -> Message:
        try:
            return await orchestrator.handle_call(call, action_ctx_factory(_emit))
        finally:
            await event_queue.put(None)

    call_task = asyncio.create_task(_run_call())
    try:
        while True:
            item = await event_queue.get()
            if item is None:
                break
            yield item
        tool_msg = await call_task
    finally:
        if not call_task.done():
            call_task.cancel()
            with contextlib.suppress(BaseException):
                await call_task

    summary_req = ChatRequest(
        messages=(
            *base_request.messages,
            Message(Role.ASSISTANT, "", tool_calls=(call,)),
            tool_msg,
        ),
        locale=base_request.locale,
        max_output_tokens=base_request.max_output_tokens,
        timeout_s=base_request.timeout_s,
        tools=(),
    )
    async for ev in chat_turn(
        generator,
        summary_req,
        sections=sections,
        locale_vi=locale_vi,
        code_generator=code_generator,
        code_timeout_s=code_timeout_s,
        cu_cleanup=cu_cleanup,
    ):
        yield ev
