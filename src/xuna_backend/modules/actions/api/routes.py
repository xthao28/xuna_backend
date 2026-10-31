"""HTTP routes for `/v1/actions` approvals, emergency stop, and board HID results (UDC-4, UDC-7)."""

import base64
import binascii
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from xuna_backend.core.errors import AppError
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.actions.application.computer_use import (
    InMemoryComputerUseSessionStore,
)
from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.application.ports import (
    ApprovalBroker,
    ApprovalRepository,
    ApprovalScopeStore,
    LastTargetStore,
    LocalResultBroker,
)
from xuna_backend.modules.actions.domain.entities import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScope,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditEvent
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.entities import Grant, Host, HostId, Manifest
from xuna_backend.modules.identity.application.ports import SignatureVerifier

router = APIRouter(prefix="/v1/actions", tags=["actions"])


class DecideApprovalBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ApprovalDecision
    scope: ApprovalScope = ApprovalScope.ONCE
    signature: str | None = Field(default=None, min_length=80, max_length=120)
    publicKey: str | None = Field(default=None, min_length=80, max_length=120)


class EmergencyStopBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: Literal["owner_app", "owner_host", "owner_board", "revoked"] = "owner_app"


class ReportActionResultBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "ok",
        "not_ready",
        "busy",
        "failed",
        "denied",
        "denied_by_local_policy",
        "permission_denied",
        "needs_permission",
        "invalid_args",
        "unsupported",
        "timeout",
    ]
    output: dict[str, Any] | None = None
    errorCode: str | None = Field(default=None, max_length=64)


def _decode_b64_flex(raw: str, expected_len: int) -> bytes:
    clean = raw.strip()
    rem = len(clean) % 4
    if rem == 2:
        clean += "=="
    elif rem == 3:
        clean += "="
    try:
        if "-" in clean or "_" in clean:
            decoded = base64.urlsafe_b64decode(clean.encode("ascii"))
        else:
            decoded = base64.b64decode(clean.encode("ascii"), validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("invalid base64 encoding") from error
    if len(decoded) != expected_len:
        raise ValueError(f"expected {expected_len} bytes")
    return decoded


@asynccontextmanager
async def _open_short_session(app_state: Any) -> AsyncIterator[Any]:
    db = getattr(app_state, "database", None)
    if db is None:
        yield None
        return
    async with db.sessions() as session:
        yield session


class _SessionPerCallHostRepository:
    """Adapter that opens a short-lived DB session per read so no connection is held
    during 60 s approval waits or 10 s host WebSocket calls."""

    def __init__(self, app_state: Any) -> None:
        self._state = app_state

    def _repo(self, session: Any) -> HostRepository:
        factory: Callable[[Any], HostRepository] = self._state.host_repository
        return factory(session)

    async def get_host(self, host_id: HostId) -> Host | None:
        async with _open_short_session(self._state) as s:
            return await self._repo(s).get_host(host_id)

    async def save_host(self, host: Host) -> None:
        async with _open_short_session(self._state) as s:
            await self._repo(s).save_host(host)
            if s is not None:
                await s.commit()

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        async with _open_short_session(self._state) as s:
            return await self._repo(s).list_hosts(owner_id)

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        async with _open_short_session(self._state) as s:
            return await self._repo(s).get_grants(host_id)

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        async with _open_short_session(self._state) as s:
            await self._repo(s).set_grants(host_id, grants)
            if s is not None:
                await s.commit()

    async def suspend_r2_plus_grants_for_owner(
        self, owner_id: uuid.UUID, now: datetime
    ) -> int:
        async with _open_short_session(self._state) as s:
            count = await self._repo(s).suspend_r2_plus_grants_for_owner(owner_id, now)
            if s is not None:
                await s.commit()
            return count

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        async with _open_short_session(self._state) as s:
            return await self._repo(s).get_manifest(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        async with _open_short_session(self._state) as s:
            await self._repo(s).save_manifest(manifest)
            if s is not None:
                await s.commit()


class _SessionPerCallApprovalRepository:
    def __init__(self, app_state: Any) -> None:
        self._state = app_state

    def _repo(self, session: Any) -> ApprovalRepository:
        factory: Callable[[Any], ApprovalRepository] = self._state.approval_repository
        return factory(session)

    async def save(self, req: ApprovalRequest) -> None:
        async with _open_short_session(self._state) as s:
            await self._repo(s).save(req)
            if s is not None:
                await s.commit()

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        async with _open_short_session(self._state) as s:
            return await self._repo(s).get(approval_id)

    async def reject_pending_for_owner(
        self, owner_id: uuid.UUID, now: datetime
    ) -> int:
        async with _open_short_session(self._state) as s:
            count = await self._repo(s).reject_pending_for_owner(owner_id, now)
            if s is not None:
                await s.commit()
            return count


class _SessionPerCallAuditLog:
    def __init__(self, app_state: Any) -> None:
        self._state = app_state

    def _log(self, session: Any) -> AuditLog:
        factory: Callable[[Any], AuditLog] = self._state.audit_log
        return factory(session)

    async def record(self, event: AuditEvent) -> None:
        async with _open_short_session(self._state) as s:
            await self._log(s).record(event)
            if s is not None:
                await s.commit()

    async def recent(self, owner_id: uuid.UUID, limit: int) -> list[AuditEvent]:
        async with _open_short_session(self._state) as s:
            return await self._log(s).recent(owner_id, limit)


def build_stream_orchestrator(app_state: Any) -> Orchestrator:
    """Builds an `Orchestrator` safe for use inside long-lived SSE / WebSocket streams."""
    hub: HostSessionHub = app_state.host_session_hub
    approval_broker: ApprovalBroker = app_state.approval_broker
    local_result_broker: LocalResultBroker = app_state.local_result_broker
    last_target_store: LastTargetStore = app_state.last_target_store
    scope_store: ApprovalScopeStore | None = getattr(
        app_state, "approval_scope_store", None
    )
    verifier: SignatureVerifier | None = getattr(
        app_state, "signature_verifier", None
    )
    return Orchestrator(
        host_repo=_SessionPerCallHostRepository(app_state),
        hub=hub,
        approval_repo=_SessionPerCallApprovalRepository(app_state),
        approval_broker=approval_broker,
        local_result_broker=local_result_broker,
        last_target_store=last_target_store,
        audit=_SessionPerCallAuditLog(app_state),
        scope_store=scope_store,
        verifier=verifier,
    )


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", "")
    return str(rid) if isinstance(rid, str) else ""


@router.post("/approvals/{approval_id}")
async def decide_action_approval(
    approval_id: uuid.UUID,
    body: DecideApprovalBody,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> dict[str, Any]:
    """Owner-scoped, idempotent approval decision (`approve` or `reject`)."""
    host_factory: Callable[[object], HostRepository] = state(request, "host_repository")
    approval_factory: Callable[[object], ApprovalRepository] = state(request, "approval_repository")
    audit_factory: Callable[[object], AuditLog] = state(request, "audit_log")
    hub: HostSessionHub = state(request, "host_session_hub")
    approval_broker: ApprovalBroker = state(request, "approval_broker")
    local_result_broker: LocalResultBroker = state(request, "local_result_broker")
    last_target_store: LastTargetStore = state(request, "last_target_store")
    scope_store: ApprovalScopeStore | None = getattr(
        request.app.state, "approval_scope_store", None
    )
    verifier: SignatureVerifier | None = getattr(
        request.app.state, "signature_verifier", None
    )

    sig_bytes: bytes | None = None
    pub_bytes: bytes | None = None
    try:
        if body.signature is not None:
            sig_bytes = _decode_b64_flex(body.signature, 64)
        if body.publicKey is not None:
            pub_bytes = _decode_b64_flex(body.publicKey, 65)
    except ValueError as error:
        raise AppError("approval_signature_invalid", HTTPStatus.BAD_REQUEST, str(error)) from error

    orch = Orchestrator(
        host_repo=host_factory(session),
        hub=hub,
        approval_repo=approval_factory(session),
        approval_broker=approval_broker,
        local_result_broker=local_result_broker,
        last_target_store=last_target_store,
        audit=audit_factory(session),
        scope_store=scope_store,
        verifier=verifier,
    )
    try:
        updated = await orch.decide_approval(
            approval_id=approval_id,
            owner_id=owner.owner_id,
            decision=body.decision,
            scope=body.scope,
            signature=sig_bytes,
            signer_public_key=pub_bytes,
            request_id=_request_id(request),
        )
    except ValueError as error:
        code = str(error)
        if code in ("r3_requires_once_scope", "approval_signature_invalid"):
            raise AppError(code, HTTPStatus.BAD_REQUEST, code) from error
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, code) from error

    if updated is None:
        raise AppError("not_found", HTTPStatus.NOT_FOUND)
    await session.commit()
    return {
        "approvalId": str(updated.approval_id),
        "state": updated.state.value,
        "scope": updated.scope.value,
        "decidedAt": (
            updated.decided_at.astimezone(UTC).isoformat()
            if updated.decided_at is not None
            else None
        ),
    }


@router.post("/stop")
async def emergency_stop_actions(
    request: Request,
    owner: CurrentOwner,
    session: Session,
    body: EmergencyStopBody | None = None,
) -> dict[str, Any]:
    """Emergency stop end-to-end (`UDC-7` S7.3): cancels active invocations, rejects
    pending approvals, sends `xuna/stop` to live hosts, suspends R2+ grants, and
    audits `host.stopped`."""
    host_factory: Callable[[object], HostRepository] = state(request, "host_repository")
    approval_factory: Callable[[object], ApprovalRepository] = state(request, "approval_repository")
    audit_factory: Callable[[object], AuditLog] = state(request, "audit_log")
    hub: HostSessionHub = state(request, "host_session_hub")
    approval_broker: ApprovalBroker = state(request, "approval_broker")
    local_result_broker: LocalResultBroker = state(request, "local_result_broker")
    last_target_store: LastTargetStore = state(request, "last_target_store")
    scope_store: ApprovalScopeStore | None = getattr(
        request.app.state, "approval_scope_store", None
    )

    orch = Orchestrator(
        host_repo=host_factory(session),
        hub=hub,
        approval_repo=approval_factory(session),
        approval_broker=approval_broker,
        local_result_broker=local_result_broker,
        last_target_store=last_target_store,
        audit=audit_factory(session),
        scope_store=scope_store,
    )
    reason = body.reason if body is not None else "owner_app"
    cu_store: InMemoryComputerUseSessionStore | None = getattr(
        request.app.state, "cu_session_store", None
    )
    if cu_store is not None:
        cu_store.stop_all_for_owner(owner.owner_id, reason=reason)
    summary = await orch.emergency_stop(
        owner_id=owner.owner_id,
        reason=reason,
        request_id=_request_id(request),
    )
    await session.commit()
    return {
        "stopped": summary.stopped,
        "hostsNotified": summary.hosts_notified,
        "approvalsCancelled": summary.approvals_cancelled,
        "grantsSuspended": summary.grants_suspended,
    }


@router.post("/sessions/{session_id}/heartbeat")
async def heartbeat_computer_use_session(
    session_id: str,
    request: Request,
    owner: CurrentOwner,
) -> dict[str, Any]:
    """UDC-8 S8.3 / S8.4: 5 s phone heartbeat for an active computer-use session.
    Missing 3 heartbeats (15 s) trips the dead-man switch and terminates the session."""
    cu_store: InMemoryComputerUseSessionStore | None = getattr(
        request.app.state, "cu_session_store", None
    )
    if cu_store is None:
        raise AppError("not_found", HTTPStatus.NOT_FOUND)
    state_obj = cu_store.get(session_id)
    if state_obj is None or state_obj.owner_id != owner.owner_id or not state_obj.active:
        raise AppError("not_found", HTTPStatus.NOT_FOUND)
    state_obj.last_heartbeat_monotonic = time.monotonic()
    return {
        "sessionId": session_id,
        "active": True,
        "step": state_obj.steps_used,
        "maxSteps": state_obj.max_steps,
    }


@router.post("/{invocation_id}/result")
async def report_action_result(
    invocation_id: str,
    body: ReportActionResultBody,
    request: Request,
    owner: CurrentOwner,
) -> dict[str, Any]:
    """Owner-scoped callback reporting the result of a client-executed action
    (e.g. BLE Opcode 12 `source = 2` for `board.media_key`)."""
    local_result_broker: LocalResultBroker = state(request, "local_result_broker")
    accepted = local_result_broker.report(
        invocation_id=invocation_id,
        owner_id=owner.owner_id,
        status=body.status,
        output=body.output,
        error_code=body.errorCode,
    )
    if not accepted:
        raise AppError("not_found", HTTPStatus.NOT_FOUND)
    return {
        "invocationId": invocation_id,
        "status": body.status,
    }
