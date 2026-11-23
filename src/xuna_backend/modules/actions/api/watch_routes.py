"""HTTP API routes for Wake Inbox and job watches (`/v1/actions/wake-inbox`)."""

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from xuna_backend.core.db import commit_or_conflict
from xuna_backend.core.errors import AppError
from xuna_backend.core.idempotency import StoredResponse, parse_key, request_hash, run_once
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session
from xuna_backend.modules.actions.application.wake_inbox import WakeInboxService
from xuna_backend.modules.actions.application.watch_coordinator import (
    WatchConflictError,
    WatchCoordinator,
    WatchNotFoundError,
)


def _require_int_schema_version(v: object) -> object:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError("schemaVersion must be an integer")
    return v


StrictSchemaVersion = Annotated[Literal[1], BeforeValidator(_require_int_schema_version)]
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]

router = APIRouter(prefix="/v1/actions", tags=["watch"])


class AckWakeInboxBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: StrictSchemaVersion = 1
    dismiss: bool = False


class ExtendWatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: StrictSchemaVersion = 1
    additionalSeconds: int = Field(default=300, ge=10, le=900)


class WakeNowBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: StrictSchemaVersion = 1


def _wake_inbox(request: Request) -> WakeInboxService:
    svc: WakeInboxService | None = getattr(request.app.state, "wake_inbox_service", None)
    if svc is None:
        raise AppError(
            "service_unavailable",
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Wake inbox service not initialized",
        )
    return svc


def _watch_coordinator(request: Request) -> WatchCoordinator:
    coord: WatchCoordinator | None = getattr(request.app.state, "watch_coordinator", None)
    if coord is None:
        raise AppError(
            "service_unavailable",
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Watch coordinator not initialized",
        )
    return coord


def _parse_uuid_or_404(raw: str, label: str) -> uuid.UUID:
    try:
        parsed = uuid.UUID(raw)
    except ValueError as err:
        raise AppError("not_found", HTTPStatus.NOT_FOUND, f"{label} not found") from err
    if parsed.int == 0:
        raise AppError("not_found", HTTPStatus.NOT_FOUND, f"{label} not found")
    return parsed


async def _run_idempotent(
    request: Request,
    session: Any,
    *,
    owner_id: uuid.UUID,
    idempotency_key: str | None,
    route_name: str,
    fingerprint_payload: dict[str, Any],
    action: Callable[[], Awaitable[StoredResponse]],
) -> dict[str, Any]:
    key = parse_key(idempotency_key)
    fingerprint = request_hash(route_name, fingerprint_payload)
    now = datetime.now(UTC)
    if hasattr(session, "scalar") and hasattr(session, "execute"):
        stored = await run_once(
            session,
            owner_id=owner_id,
            key=key,
            fingerprint=fingerprint,
            now=now,
            action=action,
        )
        await commit_or_conflict(session)
        return stored.body

    fallback: dict[tuple[uuid.UUID, uuid.UUID], tuple[str, StoredResponse]] = getattr(
        request.app.state, "_watch_idempotency_fallback", {}
    )
    if not hasattr(request.app.state, "_watch_idempotency_fallback"):
        request.app.state._watch_idempotency_fallback = fallback
    existing = fallback.get((owner_id, key))
    if existing is not None:
        prev_fp, prev_resp = existing
        if prev_fp != fingerprint:
            raise AppError("idempotency_key_reused", HTTPStatus.CONFLICT)
        return prev_resp.body
    stored = await action()
    fallback[(owner_id, key)] = (fingerprint, stored)
    return stored.body


@router.get("/wake-inbox")
async def list_wake_inbox(
    request: Request,
    owner: CurrentOwner,
    unreadOnly: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    """Lists active/unread Wake Inbox notifications for the authenticated owner."""
    svc = _wake_inbox(request)
    items = await svc.list_items(
        owner.owner_id,
        unread_only=unreadOnly,
        include_dismissed=False,
        limit=limit,
    )
    unread_count = sum(1 for item in items if item.read_at is None)
    return {
        "items": [item.to_dict() for item in items],
        "unreadCount": unread_count,
    }


@router.post("/wake-inbox/{item_id}/ack")
async def ack_wake_inbox_item(
    item_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
    body: AckWakeInboxBody | None = None,
) -> dict[str, Any]:
    """Marks a Wake Inbox item as read (and optionally dismissed)."""
    parsed_id = _parse_uuid_or_404(item_id, "Wake inbox item")
    payload = body or AckWakeInboxBody()
    svc = _wake_inbox(request)

    async def _act() -> StoredResponse:
        updated = await svc.ack_item(
            owner.owner_id,
            parsed_id,
            dismiss=payload.dismiss,
        )
        if updated is None:
            raise AppError("not_found", HTTPStatus.NOT_FOUND, "Wake inbox item not found")
        return StoredResponse(HTTPStatus.OK, updated.to_dict())

    return await _run_idempotent(
        request,
        session,
        owner_id=owner.owner_id,
        idempotency_key=idempotency_key,
        route_name="POST /v1/actions/wake-inbox/{item_id}/ack",
        fingerprint_payload={"itemId": str(parsed_id), "dismiss": payload.dismiss},
        action=_act,
    )


@router.post("/jobs/{job_id}/extend-watch")
async def extend_job_watch(
    job_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
    body: ExtendWatchBody | None = None,
) -> dict[str, Any]:
    """Extends a paused or active job watch window up to `900s`."""
    parsed_job_id = _parse_uuid_or_404(job_id, "Job")
    payload = body or ExtendWatchBody()
    coord = _watch_coordinator(request)

    async def _act() -> StoredResponse:
        try:
            watch = await coord.extend_watch(
                job_id=parsed_job_id,
                owner_id=owner.owner_id,
                additional_seconds=payload.additionalSeconds,
            )
        except WatchNotFoundError as err:
            raise AppError("not_found", HTTPStatus.NOT_FOUND, str(err)) from err
        except WatchConflictError as err:
            if err.code in {
                "no_watch",
                "job_terminal",
                "max_wake_checks_reached",
                "invalid_state",
            }:
                raise AppError("conflict", HTTPStatus.CONFLICT, err.code) from err
            raise AppError(err.code, HTTPStatus.CONFLICT, err.code) from err
        return StoredResponse(HTTPStatus.OK, watch.to_summary_dict())

    return await _run_idempotent(
        request,
        session,
        owner_id=owner.owner_id,
        idempotency_key=idempotency_key,
        route_name="POST /v1/actions/jobs/{job_id}/extend-watch",
        fingerprint_payload={
            "jobId": str(parsed_job_id),
            "additionalSeconds": payload.additionalSeconds,
        },
        action=_act,
    )


@router.post("/jobs/{job_id}/wake-now")
async def wake_job_now(
    job_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
    body: WakeNowBody | None = None,
) -> dict[str, Any]:
    """Triggers an immediate verification wake check on a waiting/paused job."""
    parsed_job_id = _parse_uuid_or_404(job_id, "Job")
    _ = body or WakeNowBody()
    coord = _watch_coordinator(request)

    async def _act() -> StoredResponse:
        try:
            job = await coord.wake_now(job_id=parsed_job_id, owner_id=owner.owner_id)
        except WatchNotFoundError as err:
            raise AppError("not_found", HTTPStatus.NOT_FOUND, str(err)) from err
        except WatchConflictError as err:
            raise AppError("conflict", HTTPStatus.CONFLICT, err.code) from err
        return StoredResponse(HTTPStatus.OK, job.to_dict())

    return await _run_idempotent(
        request,
        session,
        owner_id=owner.owner_id,
        idempotency_key=idempotency_key,
        route_name="POST /v1/actions/jobs/{job_id}/wake-now",
        fingerprint_payload={"jobId": str(parsed_job_id)},
        action=_act,
    )


@router.get("/jobs/{job_id}/watch")
async def get_job_watch(
    job_id: str,
    request: Request,
    owner: CurrentOwner,
) -> dict[str, Any]:
    """Returns the latest `JobWatchSummary` for an owner-scoped autonomous job."""
    parsed_job_id = _parse_uuid_or_404(job_id, "Job")
    coord = _watch_coordinator(request)
    try:
        watch = await coord.get_job_watch_summary(
            job_id=parsed_job_id,
            owner_id=owner.owner_id,
        )
    except WatchNotFoundError as err:
        raise AppError("not_found", HTTPStatus.NOT_FOUND, str(err)) from err
    return watch.to_summary_dict()
