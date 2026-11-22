"""Spatial call event helpers extracted from `orchestrator.py` (Mesh §5.6.7, §9.1).

Keeps `orchestrator.py` concise while consistently populating `hostId`, `targetReason`,
`targetPlatform`, `missionEnvelope`, and `ActionTargetRequiredEvent` across all outcomes.
Zero hardcoded app names, host names, OS paths, or keyword tables (H5).
"""

import uuid
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from xuna_backend.modules.actions.application.approval_and_board_service import (
    ApprovalAndBoardService,
)
from xuna_backend.modules.actions.application.catalog import is_tool_granted
from xuna_backend.modules.actions.application.cross_host_pipelines import CrossHostPipelineRunner
from xuna_backend.modules.actions.application.mesh_prober import (
    MeshProber,
    execute_mesh_meta_call,
)
from xuna_backend.modules.actions.application.mesh_router import (
    HostCallRoutingResult,
    PendingIntent,
    PendingIntentStore,
    PreRouteOutcome,
    build_action_target_required_event,
    wire_candidate_dict,
)
from xuna_backend.modules.actions.application.ports import (
    StandingApprovalRepository,
)
from xuna_backend.modules.actions.application.prompts import encode_tool_payload
from xuna_backend.modules.actions.application.tool_discovery import (
    MESH_FIND_RESOURCE_FUNCTION,
    MESH_PIPELINE_FUNCTION_NAMES,
    MESH_STATUS_FUNCTION,
)
from xuna_backend.modules.actions.domain.action_summary import (
    ActionSummary,
    gesture_for_action_status,
)
from xuna_backend.modules.actions.domain.battery_advisor import evaluate_battery_advisory
from xuna_backend.modules.actions.domain.entities import (
    APPROVAL_TTL_SECONDS,
    ActionPendingEvent,
    ActionResultEvent,
    ActionStreamEvent,
    ActionTargetRequiredEvent,
    ApprovalRequest,
    HostWakeRequestEvent,
)
from xuna_backend.modules.actions.domain.mesh_scoring import (
    ActionTargetRequiredReason,
    ScoredHostCandidate,
    extract_resource_query,
)
from xuna_backend.modules.actions.domain.mission_envelope import build_cu_session_mission_dict
from xuna_backend.modules.actions.domain.wake_queue import (
    WAKE_QUEUE_TTL,
    DeferredWakeStep,
    InMemoryWakeQueueStore,
    WakeQueueFullError,
)
from xuna_backend.modules.cognition.domain.types import Message, Role, ToolCall
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostId,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolPermissionState,
)
from xuna_backend.modules.hosts.domain.twin import HostTwin

if TYPE_CHECKING:
    from xuna_backend.modules.actions.application.orchestrator import ActionContext

EventSink = Callable[[ActionStreamEvent], Awaitable[None]]


def _prepare_wol_wake(
    *,
    reason_code: str,
    candidates: Sequence[ScoredHostCandidate],
    owner_id: uuid.UUID,
    pet_id: str,
    tool_name: str,
    clean_args: Mapping[str, object],
    risk: RiskTier,
    family: str,
    client_caps: tuple[str, ...],
    now: datetime,
    wake_queue: InMemoryWakeQueueStore | None,
) -> tuple[HostWakeRequestEvent | None, bool]:
    if reason_code != ActionTargetRequiredReason.HOST_OFFLINE.value or not candidates:
        return None, False
    cand = candidates[0]
    if not cand.wol_mac:
        return None, False
    if wake_queue is not None:
        try:
            wake_queue.enqueue(
                DeferredWakeStep(
                    step_id=uuid.uuid4(),
                    owner_id=owner_id,
                    pet_id=pet_id,
                    host_id=cand.host.host_id,
                    host_alias=cand.host.alias.value,
                    tool=tool_name,
                    arguments=dict(clean_args),
                    risk=risk,
                    created_at=now,
                    expires_at=now + WAKE_QUEUE_TTL,
                    resource_id=extract_resource_query(tool_name, clean_args),
                    family=family,
                    summary=f"Resume {tool_name} on {cand.host.alias.value}",
                    client_caps=client_caps,
                ),
                now,
            )
        except WakeQueueFullError:
            return None, True
    return (
        HostWakeRequestEvent(
            invocation_id=str(uuid.uuid4()),
            host_id=str(cand.host.host_id),
            target=cand.host.alias.value,
            mac_address=cand.wol_mac,
        ),
        False,
    )


async def emit_no_host_required(
    ctx: "ActionContext",
    call_id: str,
    tool_name: str,
    pending_store: PendingIntentStore | None,
    now: datetime,
    wake_queue: InMemoryWakeQueueStore | None = None,
) -> Message:
    """Emits `ActionTargetRequiredEvent` (`no_host` or pre-route fallback) if unadvertised."""
    if ctx.pre_route_target_required is not None:
        pre = ctx.pre_route_target_required
        reason_code = pre.reason_code or ActionTargetRequiredReason.ASK_TARGET.value
        target_reason = (
            pre.target_reason.summary
            if pre.target_reason is not None
            else "Please choose which host to target"
        )
        cands_wire = tuple(wire_candidate_dict(c) for c in pre.candidates)
        if (
            reason_code == ActionTargetRequiredReason.ASK_TARGET.value
            and pending_store is not None
            and pre.candidates
        ):
            pending_store.save(
                PendingIntent(
                    owner_id=ctx.owner_id,
                    pet_id=ctx.pet_id,
                    original_tool=tool_name,
                    original_arguments={},
                    original_text="",
                    candidate_host_ids=tuple(c.host.host_id for c in pre.candidates),
                    created_at=now,
                    suggested_host_id=pre.candidates[0].host.host_id,
                    resource_id=None,
                    family=tool_name.split(".", 1)[0],
                    client_caps=tuple(ctx.client_caps),
                )
            )
        wake_event, wake_queue_full = _prepare_wol_wake(
            reason_code=reason_code,
            candidates=pre.candidates,
            owner_id=ctx.owner_id,
            pet_id=ctx.pet_id,
            tool_name=tool_name,
            clean_args={},
            risk=RiskTier.R2,
            family=tool_name.split(".", 1)[0],
            client_caps=tuple(ctx.client_caps),
            now=now,
            wake_queue=wake_queue,
        )
        if wake_queue_full:
            target_reason = f"{target_reason}; deferred wake queue is full (wake_queue_full)"
        await ctx.emit_event(
            ActionTargetRequiredEvent(
                invocation_id=str(uuid.uuid4()),
                tool=tool_name,
                reason_code=reason_code,
                target_reason=target_reason,
                candidates=cands_wire,
                error_code=("wake_queue_full" if wake_queue_full else None),
            )
        )
        if wake_event is not None:
            await ctx.emit_event(wake_event)
        extra: dict[str, object] = {
            "choices": [c.host.alias.value for c in pre.candidates],
            "reasonCode": reason_code,
            "targetReason": target_reason,
            "candidates": list(cands_wire),
        }
        if wake_queue_full:
            extra["errorCode"] = "wake_queue_full"
        return Message(
            role=Role.TOOL,
            content=encode_tool_payload(
                reason_code,
                {"errorCode": "wake_queue_full"} if wake_queue_full else None,
                **extra,
            ),
            tool_call_id=call_id,
        )

    summary = f"No active enrolled host supports {tool_name}"
    await ctx.emit_event(
        ActionTargetRequiredEvent(
            invocation_id=str(uuid.uuid4()),
            tool=tool_name,
            reason_code=ActionTargetRequiredReason.NO_HOST.value,
            target_reason=summary,
            candidates=(),
        )
    )
    return Message(
        role=Role.TOOL,
        content=encode_tool_payload(
            ActionTargetRequiredReason.NO_HOST.value,
            None,
            choices=[],
            reasonCode=ActionTargetRequiredReason.NO_HOST.value,
            targetReason=summary,
            candidates=[],
        ),
        tool_call_id=call_id,
    )


async def handle_pre_route_target_required(
    *,
    outcome: PreRouteOutcome,
    call_id: str,
    tool_name: str,
    clean_args: Mapping[str, object],
    reference_descriptor: ToolDescriptor,
    owner_id: uuid.UUID,
    pet_id: str,
    emit_event: EventSink,
    pending_store: PendingIntentStore | None,
    now: datetime,
    client_caps: tuple[str, ...] = (),
    wake_queue: InMemoryWakeQueueStore | None = None,
) -> Message:
    """Handles a pre-routed `target_required` outcome inside tool-call execution."""
    reason_code = outcome.reason_code or ActionTargetRequiredReason.ASK_TARGET.value
    target_reason = (
        outcome.target_reason.summary
        if outcome.target_reason is not None
        else "Please choose which host to target"
    )
    candidates_wire = tuple(wire_candidate_dict(c) for c in outcome.candidates)
    if (
        reason_code == ActionTargetRequiredReason.ASK_TARGET.value
        and pending_store is not None
        and outcome.candidates
    ):
        pending_store.save(
            PendingIntent(
                owner_id=owner_id,
                pet_id=pet_id,
                original_tool=tool_name,
                original_arguments=dict(clean_args),
                original_text="",
                candidate_host_ids=tuple(c.host.host_id for c in outcome.candidates),
                created_at=now,
                suggested_host_id=outcome.candidates[0].host.host_id,
                resource_id=extract_resource_query(tool_name, clean_args),
                family=reference_descriptor.family.value,
                client_caps=tuple(client_caps),
            )
        )
    wake_event, wake_queue_full = _prepare_wol_wake(
        reason_code=reason_code,
        candidates=outcome.candidates,
        owner_id=owner_id,
        pet_id=pet_id,
        tool_name=tool_name,
        clean_args=clean_args,
        risk=reference_descriptor.risk,
        family=reference_descriptor.family.value,
        client_caps=tuple(client_caps),
        now=now,
        wake_queue=wake_queue,
    )
    if wake_queue_full:
        target_reason = f"{target_reason}; deferred wake queue is full (wake_queue_full)"
    await emit_event(
        ActionTargetRequiredEvent(
            invocation_id=str(uuid.uuid4()),
            tool=tool_name,
            reason_code=reason_code,
            target_reason=target_reason,
            candidates=candidates_wire,
            error_code=("wake_queue_full" if wake_queue_full else None),
        )
    )
    if wake_event is not None:
        await emit_event(wake_event)
    extra: dict[str, object] = {
        "choices": [c.host.alias.value for c in outcome.candidates],
        "reasonCode": reason_code,
        "targetReason": target_reason,
        "candidates": list(candidates_wire),
    }
    if wake_queue_full:
        extra["errorCode"] = "wake_queue_full"
    return Message(
        role=Role.TOOL,
        content=encode_tool_payload(
            reason_code,
            {"errorCode": "wake_queue_full"} if wake_queue_full else None,
            **extra,
        ),
        tool_call_id=call_id,
    )


async def emit_unresolved_target_required(
    *,
    emit_event: EventSink,
    call_id: str,
    tool_name: str,
    routing: HostCallRoutingResult,
    ctx: "ActionContext | None" = None,
    clean_args: Mapping[str, object] | None = None,
    risk: RiskTier = RiskTier.R2,
    family: str | None = None,
    now: datetime | None = None,
    wake_queue: InMemoryWakeQueueStore | None = None,
) -> Message:
    """Emits `ActionTargetRequiredEvent` and returns the compact `Role.TOOL` payload."""
    err_status = routing.error_status or ActionTargetRequiredReason.ASK_TARGET.value
    err_extra = dict(
        routing.error_extra
        or ({"choices": list(routing.choices)} if err_status == "ask_target" else {})
    )
    wake_event: HostWakeRequestEvent | None = None
    wake_queue_full = False
    if ctx is not None and now is not None and routing.scoring_decision is not None:
        wake_event, wake_queue_full = _prepare_wol_wake(
            reason_code=err_status,
            candidates=routing.scoring_decision.candidates,
            owner_id=ctx.owner_id,
            pet_id=ctx.pet_id,
            tool_name=tool_name,
            clean_args=clean_args or {},
            risk=risk,
            family=family or tool_name.split(".", 1)[0],
            client_caps=tuple(ctx.client_caps),
            now=now,
            wake_queue=wake_queue,
        )
    tr_event = build_action_target_required_event(str(uuid.uuid4()), tool_name, routing)
    if wake_queue_full:
        full_reason = f"{tr_event.target_reason}; deferred wake queue is full (wake_queue_full)"
        tr_event = ActionTargetRequiredEvent(
            invocation_id=tr_event.invocation_id,
            tool=tr_event.tool,
            reason_code=tr_event.reason_code,
            target_reason=full_reason,
            candidates=tr_event.candidates,
            error_code="wake_queue_full",
        )
        err_extra["targetReason"] = full_reason
        err_extra["errorCode"] = "wake_queue_full"
    await emit_event(tr_event)
    if wake_event is not None:
        await emit_event(wake_event)
    return Message(
        role=Role.TOOL,
        content=encode_tool_payload(
            err_status,
            {"errorCode": "wake_queue_full"} if wake_queue_full else None,
            **err_extra,
        ),
        tool_call_id=call_id,
    )


async def handle_pre_route_ctx(
    ctx: "ActionContext",
    call_id: str,
    tool_name: str,
    clean_args: Mapping[str, object],
    reference_descriptor: ToolDescriptor,
    pending_store: PendingIntentStore | None,
    now: datetime,
    wake_queue: InMemoryWakeQueueStore | None = None,
) -> Message:
    """Thin wrapper forwarding `ActionContext` to `handle_pre_route_target_required`."""
    if ctx.pre_route_target_required is None:
        raise ValueError("pre_route_target_required is required")
    return await handle_pre_route_target_required(
        outcome=ctx.pre_route_target_required,
        call_id=call_id,
        tool_name=tool_name,
        clean_args=clean_args,
        reference_descriptor=reference_descriptor,
        owner_id=ctx.owner_id,
        pet_id=ctx.pet_id,
        emit_event=ctx.emit_event,
        pending_store=pending_store,
        now=now,
        client_caps=tuple(ctx.client_caps),
        wake_queue=wake_queue,
    )


async def emit_unresolved_ctx(
    ctx: "ActionContext",
    call_id: str,
    tool_name: str,
    routing: HostCallRoutingResult,
    clean_args: Mapping[str, object],
    reference_descriptor: ToolDescriptor,
    now: datetime,
    wake_queue: InMemoryWakeQueueStore | None = None,
) -> Message:
    """Thin wrapper forwarding `ActionContext` to `emit_unresolved_target_required`."""
    return await emit_unresolved_target_required(
        emit_event=ctx.emit_event,
        call_id=call_id,
        tool_name=tool_name,
        routing=routing,
        ctx=ctx,
        clean_args=clean_args,
        risk=reference_descriptor.risk,
        family=reference_descriptor.family.value,
        now=now,
        wake_queue=wake_queue,
    )


def enrich_pending_battery_advisories(
    hosts: Sequence[Host],
    hub: HostSessionHub,
    target_host_id: HostId,
    tool_name: str,
    target_reason: str | None,
    self_host_id: str | None = None,
    *,
    capable_host_ids: Collection[HostId] | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    """Returns `(target_reason_with_advisory, advisories)` for heavy CU sessions."""
    if tool_name not in ("cu.session_start", "cu.session_extend"):
        return target_reason, ()
    online_ids: set[str] = set()
    twins_by_host: dict[str, HostTwin] = {}
    get_twin = getattr(hub, "get_twin", None)
    for h in hosts:
        if h.status != HostStatus.ACTIVE:
            continue
        hid = str(h.host_id)
        if hub.is_online(hid) or (self_host_id is not None and hid == self_host_id):
            online_ids.add(hid)
            if callable(get_twin):
                twin = get_twin(hid)
                if isinstance(twin, HostTwin):
                    twins_by_host[hid] = twin
    advisory = evaluate_battery_advisory(
        hosts,
        twins_by_host,
        online_host_ids=online_ids,
        target_host_id=target_host_id,
        capable_host_ids=capable_host_ids,
    )
    if advisory is None:
        return target_reason, ()
    msgs = advisory.messages
    updated_reason = f"{target_reason} · {msgs[0]}" if target_reason else msgs[0]
    return updated_reason, msgs


async def emit_pending_with_battery(
    *,
    ctx: "ActionContext",
    hosts: Sequence[Host],
    hub: HostSessionHub,
    host: Host,
    approval_req: ApprovalRequest,
    card_summary: str,
    expires_at_iso: str,
    standing_tool: str | None,
    standing_args: dict[str, object] | None,
    standing_count: int | None,
    mission_env: dict[str, object] | None,
    eff_target_reason: str | None,
    host_repo: HostRepository | None = None,
    capable_host_ids: Collection[HostId] | None = None,
    now: datetime | None = None,
) -> None:
    """Emits `ActionPendingEvent` enriched with Battery Advisor warnings when applicable."""
    eff_capable = capable_host_ids
    if (
        eff_capable is None
        and host_repo is not None
        and approval_req.tool in ("cu.session_start", "cu.session_extend")
    ):
        check_now = now or approval_req.created_at
        capable_set: set[HostId] = set()
        for h in hosts:
            if h.status != HostStatus.ACTIVE:
                continue
            hid = str(h.host_id)
            is_self = ctx.self_host_id is not None and hid == ctx.self_host_id
            if not (hub.is_online(hid) or is_self):
                continue
            manifest = await host_repo.get_manifest(h.host_id)
            desc = manifest.find_tool(approval_req.tool) if manifest is not None else None
            if desc is not None and desc.permission != ToolPermissionState.UNSUPPORTED:
                host_grants = await host_repo.get_grants(h.host_id)
                if is_tool_granted(
                    desc,
                    host_grants,
                    check_now,
                    host_screen_consent=ctx.host_screen_consent,
                ):
                    capable_set.add(h.host_id)
        eff_capable = capable_set
    eff_reason, batt_adv = enrich_pending_battery_advisories(
        hosts,
        hub,
        host.host_id,
        approval_req.tool,
        eff_target_reason,
        ctx.self_host_id,
        capable_host_ids=eff_capable,
    )
    await ctx.emit_event(
        ActionPendingEvent(
            approval_id=str(approval_req.approval_id),
            target=str(host.alias),
            summary=card_summary,
            risk=approval_req.risk.value,
            expires_in=APPROVAL_TTL_SECONDS,
            host_id=str(host.host_id),
            args_sha256=approval_req.args_sha256,
            expires_at=expires_at_iso,
            standing_tool=standing_tool,
            standing_arguments=standing_args,
            standing_approved_count=standing_count,
            mission_envelope=mission_env,
            target_reason=eff_reason,
            target_platform=host.platform.value,
            battery_advisories=batt_adv,
        )
    )


def validate_cu_session_goal(tool_name: str, clean_args: dict[str, object]) -> str | None:
    """Validates non-empty `goal` on `cu.session_start` when `goal` is supplied."""
    if tool_name == "cu.session_start" and "goal" in clean_args:
        raw_goal = clean_args["goal"]
        clean_goal = raw_goal.strip() if isinstance(raw_goal, str) else ""
        if not clean_goal:
            return "missing_goal"
        clean_args["goal"] = clean_goal
    return None


def _parse_positive_int(raw: object, default: int) -> int:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return default
    return default


def build_cu_mission_envelope(
    descriptor_name: str,
    clean_args: Mapping[str, object],
    active_cu: object | None,
) -> dict[str, object] | None:
    """Builds `MissionEnvelope` wire payload for `cu.session_start` / `cu.session_extend`."""
    if descriptor_name not in ("cu.session_start", "cu.session_extend"):
        return None
    mission_goal = str(clean_args.get("goal") or "")
    cu_goal = getattr(active_cu, "goal", None) if active_cu is not None else None
    if not mission_goal.strip() and cu_goal:
        mission_goal = str(cu_goal)
    apps_raw = clean_args.get("apps")
    apps_list = [str(a) for a in apps_raw] if isinstance(apps_raw, list | tuple) else []
    cu_apps = getattr(active_cu, "allowed_apps", None) if active_cu is not None else None
    if not apps_list and isinstance(cu_apps, list | tuple):
        apps_list = [str(a) for a in cu_apps]
    domains_raw = clean_args.get("domains")
    domains_list = [str(d) for d in domains_raw] if isinstance(domains_raw, list | tuple) else []
    cu_domains = getattr(active_cu, "allowed_domains", None) if active_cu is not None else None
    if not domains_list and isinstance(cu_domains, list | tuple):
        domains_list = [str(d) for d in cu_domains]
    plan_raw = clean_args.get("planSteps")
    plan_steps = [str(s) for s in plan_raw] if isinstance(plan_raw, list | tuple) else None
    return build_cu_session_mission_dict(
        goal=mission_goal,
        apps=apps_list,
        domains=domains_list,
        max_steps=_parse_positive_int(clean_args.get("maxSteps"), 12),
        max_minutes=_parse_positive_int(clean_args.get("maxMinutes"), 5),
        plan_steps=plan_steps,
    )


def build_pipeline_runner(
    approval_service: ApprovalAndBoardService,
    prober: MeshProber,
    standing_repo: StandingApprovalRepository | None = None,
    pending_intent_store: PendingIntentStore | None = None,
    wake_queue: InMemoryWakeQueueStore | None = None,
) -> CrossHostPipelineRunner:
    """Constructs `CrossHostPipelineRunner` from `ApprovalAndBoardService` dependencies."""
    return CrossHostPipelineRunner(
        host_repo=approval_service._host_repo,
        hub=approval_service._hub,
        approval_repo=approval_service._approval_repo,
        approval_broker=approval_service._approval_broker,
        local_result_broker=approval_service._local_result_broker,
        last_target_store=approval_service._last_target_store,
        audit=approval_service._audit,
        prober=prober,
        scope_store=approval_service._scope_store,
        standing_repo=standing_repo,
        pending_intent_store=pending_intent_store,
        wake_queue=wake_queue,
        clock=approval_service._clock,
        approval_service=approval_service,
    )


async def dispatch_mesh_tool(
    call: ToolCall,
    ctx: "ActionContext",
    prober: MeshProber,
    pipelines: CrossHostPipelineRunner,
    hosts: Sequence[Host],
    manifests: Mapping[HostId, Manifest],
    grants: Mapping[HostId, Sequence[Grant]],
    online: set[str] | frozenset[str],
) -> Message | None:
    """Dispatches `mesh__*` discovery or Phase 3 pipeline tools, returning `None` otherwise."""
    if call.name in MESH_PIPELINE_FUNCTION_NAMES:
        return await pipelines.execute(
            call, ctx, hosts=hosts, manifests=manifests, grants=grants, online=online
        )
    if call.name in (MESH_STATUS_FUNCTION, MESH_FIND_RESOURCE_FUNCTION):
        content = await execute_mesh_meta_call(
            prober=prober,
            tool_name=call.name,
            arguments=call.arguments,
            owner_id=ctx.owner_id,
            hosts=hosts,
            manifests=manifests,
            grants=grants,
            online=online,
            self_host_id=ctx.self_host_id,
            host_screen_consent=ctx.host_screen_consent,
        )
        return Message(role=Role.TOOL, content=content, tool_call_id=call.id)
    return None


def resolve_effective_target_reason(
    pinned_target_reason: str | None,
    routing: HostCallRoutingResult,
) -> str | None:
    """Returns the human-readable target selection explanation for a resolved host."""
    if pinned_target_reason:
        return pinned_target_reason
    if routing.target_reason is not None:
        return routing.target_reason.summary
    return None


async def emit_host_action_result(
    ctx: "ActionContext",
    invocation_id: str,
    host: Host,
    tool_name: str,
    summary_args: Mapping[str, object],
    status: str,
    target_reason: str | None = None,
    error_code: str | None = None,
) -> str:
    """Formats summary and emits `ActionResultEvent` with consistent spatial metadata."""
    target_alias = str(host.alias)
    summary = ActionSummary.format_result(
        tool_name, dict(summary_args), target_alias, status, ctx.locale
    )
    await ctx.emit_event(
        ActionResultEvent(
            invocation_id=invocation_id,
            target=target_alias,
            summary=summary,
            status=status,
            gesture=gesture_for_action_status(status),
            error_code=error_code,
            host_id=str(host.host_id),
            target_reason=target_reason,
            target_platform=host.platform.value,
            mission_envelope=ctx.mission_envelope,
        )
    )
    return summary
