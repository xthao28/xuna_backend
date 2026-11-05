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
import uuid
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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
from xuna_backend.modules.actions.application.ports import (
    ApprovalBroker,
    ApprovalRepository,
    ApprovalScopeStore,
    LastTargetStore,
    LocalResultBroker,
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
from xuna_backend.modules.hosts.domain.crypto import args_sha256
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

# Host agents built before `xuna/policy.idempotent` existed do not declare it. Only for
# those undeclared descriptors, these known toggles keep repeat protection; a host's own
# declaration always wins. Remove once every shipped host agent declares the flag.
_LEGACY_UNDECLARED_TOGGLES: frozenset[str] = frozenset({"media.play_pause"})


def is_repeat_safe(descriptor: ToolDescriptor) -> bool:
    """Whether re-running a successful call with identical arguments is harmless."""
    if descriptor.idempotent is not None:
        return descriptor.idempotent
    return descriptor.name not in _LEGACY_UNDECLARED_TOGGLES


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
    ) -> None:
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
        return await self._approval_and_board.emergency_stop(
            owner_id=owner_id,
            reason=reason,
            request_id=request_id,
        )

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
            if is_tool_granted(descriptor, grants, now):
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

        if not is_tool_granted(descriptor, host_grants, now):
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
                        "reason": "grant_denied",
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
                )
            )
            return Message(
                role=Role.TOOL,
                content=_encode_tool_payload("denied"),
                tool_call_id=call.id,
            )

        self._last_target_store.set_last_target(ctx.owner_id, ctx.pet_id, target_alias, now)

        approval_req: ApprovalRequest | None = None
        if descriptor.risk.level >= RiskTier.R2.level:
            has_15m_scope = (
                descriptor.risk == RiskTier.R2
                and self._scope_store is not None
                and self._scope_store.has_valid_scope(
                    ctx.owner_id, ctx.pet_id, host.host_id, descriptor.name, now
                )
            )
            expires_at = now + timedelta(seconds=APPROVAL_TTL_SECONDS)
            if has_15m_scope:
                approval_req = ApprovalRequest(
                    approval_id=uuid.uuid4(),
                    owner_id=ctx.owner_id,
                    host_id=host.host_id,
                    target_alias=target_alias,
                    tool=descriptor.name,
                    args_sha256=args_sha256(clean_args),
                    risk=descriptor.risk,
                    summary=ActionSummary.format_pending(
                        descriptor.name, clean_args, target_alias, ctx.locale
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
                            "risk": descriptor.risk.value,
                            "scope": ApprovalScope.CONVERSATION_15M.value,
                        },
                    )
                )
            else:
                approval_req = ApprovalRequest(
                    approval_id=uuid.uuid4(),
                    owner_id=ctx.owner_id,
                    host_id=host.host_id,
                    target_alias=target_alias,
                    tool=descriptor.name,
                    args_sha256=args_sha256(clean_args),
                    risk=descriptor.risk,
                    summary=ActionSummary.format_pending(
                        descriptor.name, clean_args, target_alias, ctx.locale
                    ),
                    created_at=now,
                    expires_at=expires_at,
                )
                await self._approval_repo.save(approval_req)
                await self._approval_broker.create(approval_req)

                await ctx.emit_event(
                    ActionPendingEvent(
                        approval_id=str(approval_req.approval_id),
                        target=target_alias,
                        summary=approval_req.summary,
                        risk=approval_req.risk.value,
                        expires_in=APPROVAL_TTL_SECONDS,
                        host_id=str(host.host_id),
                        args_sha256=approval_req.args_sha256,
                        expires_at=expires_at.astimezone(UTC).isoformat(),
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
                                clean_args,
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
                    and descriptor.risk == RiskTier.R2
                    and self._scope_store is not None
                ):
                    self._scope_store.grant_scope(
                        ctx.owner_id,
                        ctx.pet_id,
                        host.host_id,
                        descriptor.name,
                        self._clock(),
                    )

            if args_sha256(clean_args) != approval_req.args_sha256:
                await ctx.emit_event(
                    ActionResultEvent(
                        invocation_id=invocation_id,
                        target=target_alias,
                        summary=ActionSummary.format_result(
                            descriptor.name,
                            clean_args,
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
                    "risk": descriptor.risk.value,
                    "status": status,
                    "durationMs": max(duration_ms, 0),
                    "platform": host.platform.value,
                },
            )
        )

        await ctx.emit_event(
            ActionResultEvent(
                invocation_id=invocation_id,
                target=target_alias,
                summary=ActionSummary.format_result(
                    descriptor.name, clean_args, target_alias, status, ctx.locale
                ),
                status=status,
                gesture=gesture_for_action_status(status),
            )
        )
        if status == "ok" and not is_repeat_safe(descriptor):
            content = _encode_tool_payload(status, structured_content, repeatSafe=False)
        else:
            content = _encode_tool_payload(status, structured_content)
        return Message(role=Role.TOOL, content=content, tool_call_id=call.id)
