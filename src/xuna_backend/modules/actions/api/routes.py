"""HTTP routes for `/v1/actions` approvals and board HID results (UDC-4 §5.2-§5.4)."""

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC
from http import HTTPStatus
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from xuna_backend.core.errors import AppError
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.application.ports import (
    ApprovalBroker,
    ApprovalRepository,
    LastTargetStore,
    LocalResultBroker,
)
from xuna_backend.modules.actions.domain.entities import (
    ApprovalDecision,
    ApprovalRequest,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditEvent
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.entities import Grant, Host, HostId, Manifest

router = APIRouter(prefix="/v1/actions", tags=["actions"])


class DecideApprovalBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ApprovalDecision


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
    return Orchestrator(
        host_repo=_SessionPerCallHostRepository(app_state),
        hub=hub,
        approval_repo=_SessionPerCallApprovalRepository(app_state),
        approval_broker=approval_broker,
        local_result_broker=local_result_broker,
        last_target_store=last_target_store,
        audit=_SessionPerCallAuditLog(app_state),
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

    orch = Orchestrator(
        host_repo=host_factory(session),
        hub=hub,
        approval_repo=approval_factory(session),
        approval_broker=approval_broker,
        local_result_broker=local_result_broker,
        last_target_store=last_target_store,
        audit=audit_factory(session),
    )
    updated = await orch.decide_approval(
        approval_id=approval_id,
        owner_id=owner.owner_id,
        decision=body.decision,
        request_id=_request_id(request),
    )
    if updated is None:
        raise AppError("not_found", HTTPStatus.NOT_FOUND)
    await session.commit()
    return {
        "approvalId": str(updated.approval_id),
        "state": updated.state.value,
        "decidedAt": (
            updated.decided_at.astimezone(UTC).isoformat()
            if updated.decided_at is not None
            else None
        ),
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
