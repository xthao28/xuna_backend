"""HTTP routes for `/v1/hosts` enrollment, management and dev invocation (UDC-3 §5.4)."""

import time
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import JSONResponse

from xuna_backend.core.config import Settings
from xuna_backend.core.db import commit_or_conflict
from xuna_backend.core.errors import AppError
from xuna_backend.core.idempotency import StoredResponse, parse_key, request_hash, run_once
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.hosts.api.host_socket import router as socket_router
from xuna_backend.modules.hosts.api.schemas import (
    ClaimHostRequest,
    ConfirmHostRequest,
    DevInvokeToolRequest,
    SetHostGrantsRequest,
)
from xuna_backend.modules.hosts.application.ports import (
    EnrollmentRepository,
    HostRepository,
    HostSessionHub,
)
from xuna_backend.modules.hosts.application.use_cases import (
    ClaimEnrollmentCommand,
    HostError,
    HostSummaryView,
    claim_enrollment,
    confirm_enrollment,
    invoke_tool,
    list_hosts,
    revoke_host,
    set_grants,
)
from xuna_backend.modules.hosts.domain.entities import Grant, Host, HostId, Manifest
from xuna_backend.modules.hosts.domain.session import HostRateLimiter

router = APIRouter(prefix="/v1", tags=["hosts"])
router.include_router(socket_router)

IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]

_ERROR_STATUS = {
    "not_found": HTTPStatus.NOT_FOUND,
    "conflict": HTTPStatus.CONFLICT,
    "rate_limited": HTTPStatus.TOO_MANY_REQUESTS,
    "unauthorized": HTTPStatus.FORBIDDEN,
    "validation_failed": HTTPStatus.UNPROCESSABLE_ENTITY,
    "invalid_argument": HTTPStatus.UNPROCESSABLE_ENTITY,
    "host_offline": HTTPStatus.CONFLICT,
    "tool_not_found": HTTPStatus.NOT_FOUND,
    "grant_denied": HTTPStatus.FORBIDDEN,
}


def _host_repo(request: Request, session: object) -> HostRepository:
    factory: Callable[[object], HostRepository] = state(request, "host_repository")
    return factory(session)


def _enrollment_repo(request: Request, session: object) -> EnrollmentRepository:
    factory: Callable[[object], EnrollmentRepository] = state(request, "host_enrollment_repository")
    return factory(session)


def _audit(request: Request, session: object) -> AuditLog:
    factory: Callable[[object], AuditLog] = state(request, "audit_log")
    return factory(session)


def _hub(request: Request) -> HostSessionHub:
    hub: HostSessionHub = state(request, "host_session_hub")
    return hub


def _problem(error: HostError) -> AppError:
    return AppError(
        error.code,
        _ERROR_STATUS.get(error.code, HTTPStatus.BAD_REQUEST),
        error.detail,
    )


def _grant_body(grant: Grant) -> dict[str, Any]:
    return {
        "family": grant.family.value,
        "maxRisk": grant.max_risk.value,
        "enabled": grant.enabled,
        "updatedAt": grant.updated_at.astimezone(UTC).isoformat(),
    }


def _host_body(
    host: Host,
    grants: list[Grant],
    *,
    online: bool = False,
    manifest: Manifest | None = None,
) -> dict[str, Any]:
    return {
        "hostId": str(host.host_id),
        "alias": host.alias.value,
        "platform": host.platform.value,
        "hostname": host.hostname,
        "osVersion": host.os_version,
        "agentVersion": host.agent_version,
        "status": host.status.value,
        "online": online,
        "createdAt": host.created_at.astimezone(UTC).isoformat(),
        "lastSeenAt": (
            host.last_seen_at.astimezone(UTC).isoformat() if host.last_seen_at is not None else None
        ),
        "grants": [_grant_body(g) for g in grants],
        "tools": [t.to_wire() for t in manifest.tools] if manifest is not None else [],
    }


def _summary_body(view: HostSummaryView) -> dict[str, Any]:
    return _host_body(
        view.host,
        view.grants,
        online=view.online,
        manifest=view.manifest,
    )


def _json(response: StoredResponse) -> JSONResponse:
    return JSONResponse(response.body, status_code=response.status.value)


@router.post("/hosts", status_code=HTTPStatus.ACCEPTED)
async def claim_host_endpoint(
    body: ClaimHostRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    now = datetime.now(UTC)
    claim_limiter: HostRateLimiter = state(request, "host_claim_limiter")
    command = ClaimEnrollmentCommand(
        enrollment_code=body.enrollmentCode,
        alias=body.alias,
        key_fingerprint=body.keyFingerprint,
    )

    async def action() -> StoredResponse:
        pending = await claim_enrollment(
            _enrollment_repo(request, session),
            _audit(request, session),
            claim_limiter,
            owner.owner_id,
            command,
            now,
            time.monotonic(),
            getattr(request.state, "request_id", ""),
        )
        pid = str(pending.pending_id)
        return StoredResponse(
            HTTPStatus.ACCEPTED,
            {
                "hostId": pid,
                "pendingId": pid,
                "sas": pending.sas,
                "platform": pending.platform.value,
                "hostname": pending.hostname,
            },
        )

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash("claim_host", body.model_dump(mode="json")),
            now=now,
            action=action,
        )
    except HostError as error:
        await commit_or_conflict(session)
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)


@router.post("/hosts/{host_id}/confirm", status_code=HTTPStatus.OK)
async def confirm_host_endpoint(
    host_id: str,
    body: ConfirmHostRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)
    hub = _hub(request)

    async def action() -> StoredResponse:
        host, grants = await confirm_enrollment(
            _enrollment_repo(request, session),
            _host_repo(request, session),
            hub,
            _audit(request, session),
            owner.owner_id,
            parsed_id,
            now,
            getattr(request.state, "request_id", ""),
        )
        return StoredResponse(
            HTTPStatus.OK,
            _host_body(host, grants, online=hub.is_online(str(host.host_id))),
        )

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash(
                "confirm_host",
                {"hostId": str(parsed_id), **body.model_dump(mode="json")},
            ),
            now=now,
            action=action,
        )
    except HostError as error:
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)


@router.get("/hosts")
async def list_hosts_endpoint(
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> dict[str, Any]:
    views = await list_hosts(_host_repo(request, session), _hub(request), owner.owner_id)
    return {"items": [_summary_body(v) for v in views]}


@router.delete("/hosts/{host_id}", status_code=HTTPStatus.NO_CONTENT)
async def revoke_host_endpoint(
    host_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> Response:
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)

    async def action() -> StoredResponse:
        await revoke_host(
            _host_repo(request, session),
            _hub(request),
            _audit(request, session),
            owner.owner_id,
            parsed_id,
            now,
            getattr(request.state, "request_id", ""),
        )
        return StoredResponse(HTTPStatus.NO_CONTENT, {})

    try:
        await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash("revoke_host", {"hostId": str(parsed_id)}),
            now=now,
            action=action,
        )
    except HostError as error:
        raise _problem(error) from error
    await commit_or_conflict(session)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.put("/hosts/{host_id}/grants")
async def set_host_grants_endpoint(
    host_id: str,
    body: SetHostGrantsRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)

    async def action() -> StoredResponse:
        grants = await set_grants(
            _host_repo(request, session),
            _audit(request, session),
            owner.owner_id,
            parsed_id,
            [(g.parsed_family(), g.parsed_max_risk(), g.enabled) for g in body.grants],
            now,
            getattr(request.state, "request_id", ""),
        )
        return StoredResponse(
            HTTPStatus.OK,
            {"hostId": str(parsed_id), "grants": [_grant_body(g) for g in grants]},
        )

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash(
                "set_host_grants",
                {"hostId": str(parsed_id), **body.model_dump(mode="json")},
            ),
            now=now,
            action=action,
        )
    except HostError as error:
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)


@router.post("/hosts/{host_id}/tools:invoke")
async def dev_invoke_host_tool_endpoint(
    host_id: str,
    body: DevInvokeToolRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    """Dev/test tool invocation endpoint enabled only when `XUNA_DEV_HOST_TOOLS=1`."""
    settings: Settings = state(request, "settings")
    if not settings.dev_host_tools:
        raise AppError("not_found", HTTPStatus.NOT_FOUND)
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)

    async def action() -> StoredResponse:
        result = await invoke_tool(
            _host_repo(request, session),
            _hub(request),
            _audit(request, session),
            owner.owner_id,
            parsed_id,
            body.tool,
            body.arguments,
            now,
            deadline_ms=body.deadlineMs,
            include_dev_approval=body.approve,
            request_id=getattr(request.state, "request_id", ""),
        )
        return StoredResponse(
            HTTPStatus.OK,
            {
                "invocationId": result.invocation_id,
                "tool": result.tool_name,
                "risk": result.risk.value,
                "status": result.status,
                "isError": result.is_error,
                "structuredContent": result.structured_content,
                "text": result.text,
                "durationMs": result.duration_ms,
                "errorCode": result.error_code,
            },
        )

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash(
                "dev_invoke_host_tool",
                {"hostId": str(parsed_id), **body.model_dump(mode="json")},
            ),
            now=now,
            action=action,
        )
    except HostError as error:
        await commit_or_conflict(session)
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)
