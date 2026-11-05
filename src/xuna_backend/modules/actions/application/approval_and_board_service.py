"""Owner approval verification, emergency stop, and Board HID execution (UDC-4, UDC-7).

Extracted from `orchestrator.py` to separate lifecycle/approval/board responsibilities
from host tool routing.
"""

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from xuna_backend.modules.actions.application.catalog import CAP_BOARD_HID_V1
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
    DEADLINE_SECONDS,
    encode_tool_payload,
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
    BoardHidRequestEvent,
    EmergencyStopSummary,
    gesture_for_action_status,
)
from xuna_backend.modules.actions.domain.schema_validator import (
    ArgumentValidationError,
    SchemaDefinitionError,
    SchemaValidator,
)
from xuna_backend.modules.actions.domain.tool_names import (
    BOARD_MEDIA_KEY_SCHEMA,
    BOARD_MEDIA_KEY_TO_USAGE_INDEX,
    BOARD_TOOL,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.cognition.domain.types import Locale, Message, Role, ToolCall
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.crypto import (
    approval_proof_message,
    args_sha256,
    validate_public_key,
)
from xuna_backend.modules.hosts.domain.entities import (
    HostPlatform,
    HostStatus,
    RiskTier,
)
from xuna_backend.modules.identity.application.ports import SignatureVerifier

EventSink = Callable[[ActionStreamEvent], Awaitable[None]]


class ApprovalAndBoardService:
    """Handles emergency stop, R2/R3 approval decisions, and `board.media_key` invocations."""

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
        scope_store: ApprovalScopeStore | None,
        verifier: SignatureVerifier | None,
        clock: Callable[[], datetime],
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
        self._clock = clock

    async def emergency_stop(
        self,
        *,
        owner_id: uuid.UUID,
        reason: str = "owner_app",
        request_id: str = "",
    ) -> EmergencyStopSummary:
        now = self._clock()
        hosts = await self._host_repo.list_hosts(owner_id)
        hosts_notified = 0
        for host in hosts:
            if host.status == HostStatus.ACTIVE and await self._hub.stop_host(
                str(host.host_id), reason=reason
            ):
                hosts_notified += 1

        broker_rejected = self._approval_broker.reject_all_for_owner(owner_id, now)
        db_rejected = await self._approval_repo.reject_pending_for_owner(owner_id, now)
        approvals_cancelled = max(len(broker_rejected), db_rejected)

        self._local_result_broker.cancel_for_owner(owner_id)
        self._last_target_store.clear_owner(owner_id)
        if self._scope_store is not None:
            self._scope_store.clear_owner(owner_id)

        grants_suspended = await self._host_repo.suspend_r2_plus_grants_for_owner(owner_id, now)

        safe_reason = (
            reason
            if reason in ("owner_app", "owner_host", "owner_board", "revoked")
            else "owner_app"
        )
        await self._audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=AuditAction.HOST_STOPPED,
                occurred_at=now,
                subject="all_hosts",
                request_id=request_id,
                metadata={
                    "reason": safe_reason,
                    "hostsNotified": hosts_notified,
                    "approvalsCancelled": approvals_cancelled,
                    "grantsSuspended": grants_suspended,
                },
            )
        )
        return EmergencyStopSummary(
            stopped=True,
            hosts_notified=hosts_notified,
            approvals_cancelled=approvals_cancelled,
            grants_suspended=grants_suspended,
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
        now = self._clock()
        existing = await self._approval_repo.get(approval_id)
        if existing is None or existing.owner_id != owner_id:
            return None

        if scope == ApprovalScope.CONVERSATION_15M and (
            existing.risk.level >= RiskTier.R3.level or existing.host_id is None
        ):
            raise ValueError("r3_requires_once_scope")

        resolved_pub_key = signer_public_key
        if (
            decision == ApprovalDecision.APPROVE
            and existing.risk == RiskTier.R3
            and existing.host_id is not None
            and signature is not None
        ):
            if len(signature) != 64:
                raise ValueError("approval_signature_invalid")
            expires_iso = existing.expires_at.astimezone(UTC).isoformat()
            msg = approval_proof_message(
                str(existing.approval_id),
                existing.args_sha256,
                str(existing.host_id),
                expires_iso,
            )
            owned_hosts = await self._host_repo.list_hosts(owner_id)
            phone_keys = [
                h.public_key
                for h in owned_hosts
                if h.status == HostStatus.ACTIVE
                and h.platform in (HostPlatform.ANDROID, HostPlatform.IOS)
            ]
            if resolved_pub_key is not None:
                validate_public_key(resolved_pub_key)
                if phone_keys and resolved_pub_key not in phone_keys:
                    raise ValueError("approval_signature_invalid")
                candidate_keys = [resolved_pub_key]
            else:
                candidate_keys = phone_keys

            if self._verifier is not None and candidate_keys:
                matched_key: bytes | None = None
                for cand in candidate_keys:
                    if self._verifier.verify(cand, msg, signature):
                        matched_key = cand
                        break
                if matched_key is None:
                    raise ValueError("approval_signature_invalid")
                resolved_pub_key = matched_key

        was_pending = existing.state == ApprovalState.PENDING
        updated = await self._approval_broker.decide(
            approval_id=approval_id,
            owner_id=owner_id,
            decision=decision,
            now=now,
            scope=scope,
            signature=signature,
            signer_public_key=resolved_pub_key,
        )
        if updated is None:
            if existing.state != ApprovalState.PENDING:
                return existing
            if existing.is_expired(now):
                expired_req = ApprovalRequest(
                    approval_id=existing.approval_id,
                    owner_id=existing.owner_id,
                    host_id=existing.host_id,
                    target_alias=existing.target_alias,
                    tool=existing.tool,
                    args_sha256=existing.args_sha256,
                    risk=existing.risk,
                    summary=existing.summary,
                    created_at=existing.created_at,
                    expires_at=existing.expires_at,
                    state=ApprovalState.EXPIRED,
                    decided_at=now,
                    scope=existing.scope,
                )
                await self._approval_repo.save(expired_req)
                return expired_req
            new_state = (
                ApprovalState.APPROVED
                if decision == ApprovalDecision.APPROVE
                else ApprovalState.REJECTED
            )
            updated = ApprovalRequest(
                approval_id=existing.approval_id,
                owner_id=existing.owner_id,
                host_id=existing.host_id,
                target_alias=existing.target_alias,
                tool=existing.tool,
                args_sha256=existing.args_sha256,
                risk=existing.risk,
                summary=existing.summary,
                created_at=existing.created_at,
                expires_at=existing.expires_at,
                state=new_state,
                decided_at=now,
                scope=scope,
                signature=signature,
                signer_public_key=resolved_pub_key,
            )

        await self._approval_repo.save(updated)
        if was_pending and updated.state in (
            ApprovalState.APPROVED,
            ApprovalState.REJECTED,
        ):
            audit_action = (
                AuditAction.HOST_APPROVAL_GRANTED
                if updated.state == ApprovalState.APPROVED
                else AuditAction.HOST_APPROVAL_REJECTED
            )
            await self._audit.record(
                AuditEvent(
                    owner_id=owner_id,
                    action=audit_action,
                    occurred_at=now,
                    subject=str(updated.host_id) if updated.host_id else "board",
                    request_id=request_id,
                    metadata={
                        "approvalId": str(updated.approval_id),
                        "tool": updated.tool,
                        "risk": updated.risk.value,
                        "scope": updated.scope.value,
                    },
                )
            )
        return updated

    async def handle_board_call(
        self,
        call: ToolCall,
        *,
        owner_id: uuid.UUID,
        locale: Locale,
        client_caps: object,
        board_available: bool,
        emit_event: EventSink,
        request_id: str,
    ) -> Message:
        if CAP_BOARD_HID_V1 not in client_caps or not board_available:  # type: ignore[operator]
            return Message(
                role=Role.TOOL,
                content=encode_tool_payload("no_host"),
                tool_call_id=call.id,
            )

        try:
            clean_args, _ = SchemaValidator.validate_arguments(
                BOARD_MEDIA_KEY_SCHEMA,
                call.arguments,
                allow_target=False,
            )
        except (ArgumentValidationError, SchemaDefinitionError):
            return Message(
                role=Role.TOOL,
                content=encode_tool_payload("invalid_args"),
                tool_call_id=call.id,
            )

        key = str(clean_args["key"])
        usage_index = BOARD_MEDIA_KEY_TO_USAGE_INDEX[key]
        target_label = BOARD_TARGET_LABEL_EN if locale == Locale.EN else BOARD_TARGET_LABEL_VI
        invocation_id = str(uuid.uuid4())
        now = self._clock()
        expires_at = now + timedelta(seconds=APPROVAL_TTL_SECONDS)

        approval_req = ApprovalRequest(
            approval_id=uuid.uuid4(),
            owner_id=owner_id,
            host_id=None,
            target_alias=target_label,
            tool=BOARD_TOOL,
            args_sha256=args_sha256(clean_args),
            risk=RiskTier.R2,
            summary=ActionSummary.format_pending(BOARD_TOOL, clean_args, target_label, locale),
            created_at=now,
            expires_at=expires_at,
        )
        await self._approval_repo.save(approval_req)
        await self._approval_broker.create(approval_req)

        await emit_event(
            ActionPendingEvent(
                approval_id=str(approval_req.approval_id),
                target=target_label,
                summary=approval_req.summary,
                risk=approval_req.risk.value,
                expires_in=APPROVAL_TTL_SECONDS,
                host_id=None,
                args_sha256=approval_req.args_sha256,
                expires_at=expires_at.astimezone(UTC).isoformat(),
            )
        )

        decision_state = await self._approval_broker.wait(
            approval_req.approval_id,
            timeout_s=float(APPROVAL_TTL_SECONDS),
        )
        await self.sync_approval_after_wait(approval_req, decision_state, request_id)

        if decision_state != ApprovalState.APPROVED:
            final_status = "rejected" if decision_state == ApprovalState.REJECTED else "expired"
            await emit_event(
                ActionResultEvent(
                    invocation_id=invocation_id,
                    target=target_label,
                    summary=ActionSummary.format_result(
                        BOARD_TOOL, clean_args, target_label, final_status, locale
                    ),
                    status=final_status,
                    gesture=gesture_for_action_status(final_status),
                )
            )
            return Message(
                role=Role.TOOL,
                content=encode_tool_payload(final_status),
                tool_call_id=call.id,
            )

        await self._audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=AuditAction.BOARD_HID_REQUESTED,
                occurred_at=self._clock(),
                subject="board",
                request_id=request_id,
                metadata={
                    "tool": BOARD_TOOL,
                    "usageIndex": usage_index,
                    "approvalId": str(approval_req.approval_id),
                },
            )
        )

        self._local_result_broker.register(invocation_id, owner_id)
        await emit_event(
            BoardHidRequestEvent(
                invocation_id=invocation_id,
                approval_id=str(approval_req.approval_id),
                key=key,
                usage_index=usage_index,
            )
        )

        board_status = await self._local_result_broker.wait(
            invocation_id,
            timeout_s=DEADLINE_SECONDS,
        )
        normalized_status = (
            board_status
            if board_status in ("ok", "not_ready", "busy", "failed", "timeout", "denied")
            else "failed"
        )
        event_status = (
            "ok"
            if normalized_status == "ok"
            else ("timeout" if normalized_status == "timeout" else "failed")
        )

        await emit_event(
            ActionResultEvent(
                invocation_id=invocation_id,
                target=target_label,
                summary=ActionSummary.format_result(
                    BOARD_TOOL, clean_args, target_label, event_status, locale
                ),
                status=event_status,
                gesture=gesture_for_action_status(event_status),
            )
        )
        output: dict[str, object] = (
            {"key": key, "usageIndex": usage_index}
            if normalized_status == "ok"
            else {"reason": normalized_status}
        )
        return Message(
            role=Role.TOOL,
            content=encode_tool_payload(normalized_status, output),
            tool_call_id=call.id,
        )

    async def sync_approval_after_wait(
        self,
        req: ApprovalRequest,
        state: ApprovalState,
        request_id: str,
    ) -> None:
        stored = await self._approval_repo.get(req.approval_id)
        if stored is not None and stored.state != ApprovalState.PENDING:
            return

        slot_req = self._approval_broker.get_slot_request(req.approval_id)
        now = self._clock()
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
            state=state,
            decided_at=now,
            scope=slot_req.scope if slot_req is not None else req.scope,
            signature=slot_req.signature if slot_req is not None else req.signature,
            signer_public_key=(
                slot_req.signer_public_key if slot_req is not None else req.signer_public_key
            ),
        )
        await self._approval_repo.save(updated)
        if state in (ApprovalState.APPROVED, ApprovalState.REJECTED):
            audit_action = (
                AuditAction.HOST_APPROVAL_GRANTED
                if state == ApprovalState.APPROVED
                else AuditAction.HOST_APPROVAL_REJECTED
            )
            await self._audit.record(
                AuditEvent(
                    owner_id=req.owner_id,
                    action=audit_action,
                    occurred_at=now,
                    subject=str(req.host_id) if req.host_id else "board",
                    request_id=request_id,
                    metadata={
                        "approvalId": str(req.approval_id),
                        "tool": req.tool,
                        "risk": req.risk.value,
                        "scope": updated.scope.value,
                    },
                )
            )
