"""CLD-12 consent endpoints and the `RequireConsent` route guard."""

from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from xuna_backend.core.db import commit_or_conflict
from xuna_backend.core.errors import AppError
from xuna_backend.core.idempotency import StoredResponse, parse_key, request_hash, run_once
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.consent.application.ports import ConsentRepository
from xuna_backend.modules.consent.application.use_cases import (
    ConsentError,
    UpdateConsent,
    require_consent,
    update_consent,
)
from xuna_backend.modules.consent.domain.consent import ConsentClass, ConsentState

router = APIRouter(prefix="/v1", tags=["consent"])
_STATUS = {
    "consent_required": HTTPStatus.FORBIDDEN,
    "consent_version_conflict": HTTPStatus.CONFLICT,
}


class ConsentBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: Literal[1]
    aggregateMetrics: bool
    experienceEvents: bool
    conversationText: bool
    expectedVersion: int = Field(ge=0)
    # Optional for clients that predate newer classes; omitted keeps the current value.
    conversationVoice: bool | None = None
    ownerDocuments: bool | None = None
    hostScreen: bool | None = None
    hostNotifications: bool | None = None


def _repo(request: Request, session: object) -> ConsentRepository:
    factory: Callable[[object], ConsentRepository] = state(request, "consent_repository")
    return factory(session)


def _body(consent: ConsentState) -> dict[str, Any]:
    return {
        "aggregateMetrics": consent.aggregate_metrics,
        "experienceEvents": consent.experience_events,
        "conversationText": consent.conversation_text,
        "conversationVoice": consent.conversation_voice,
        "ownerDocuments": consent.owner_documents,
        "hostScreen": consent.host_screen,
        "hostNotifications": consent.host_notifications,
        "version": consent.version,
        "updatedAt": consent.updated_at.astimezone(UTC).isoformat() if consent.updated_at else None,
    }


def _problem(error: ConsentError) -> AppError:
    return AppError(error.code, _STATUS.get(error.code, HTTPStatus.BAD_REQUEST))


@router.get("/consents")
async def get_consents(request: Request, owner: CurrentOwner, session: Session) -> dict[str, Any]:
    return _body(await _repo(request, session).current(owner.owner_id))


@router.put("/consents")
async def put_consents(
    body: ConsentBody,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JSONResponse:
    now = datetime.now(UTC)
    audit_for: Callable[[object], AuditLog] = state(request, "audit_log")
    command = UpdateConsent(
        aggregate_metrics=body.aggregateMetrics,
        experience_events=body.experienceEvents,
        conversation_text=body.conversationText,
        expected_version=body.expectedVersion,
        conversation_voice=body.conversationVoice,
        owner_documents=body.ownerDocuments,
        host_screen=body.hostScreen,
        host_notifications=body.hostNotifications,
    )

    async def action() -> StoredResponse:
        updated = await update_consent(
            _repo(request, session),
            audit_for(session),
            owner.owner_id,
            command,
            now,
            request.state.request_id,
        )
        return StoredResponse(HTTPStatus.OK, _body(updated))

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash("consents", body.model_dump(mode="json")),
            now=now,
            action=action,
        )
    except ConsentError as error:
        raise _problem(error) from error
    await commit_or_conflict(session)
    return JSONResponse(response.body, status_code=response.status.value)


def RequireConsent(consent: ConsentClass) -> Any:
    """Route guard, used as
    `owner: Annotated[Owner, RequireConsent(ConsentClass.EXPERIENCE_EVENTS)]`.
    Every route that stores or forwards owner data of a class must use it."""

    async def guard(request: Request, owner: CurrentOwner, session: Session) -> Owner:
        try:
            await require_consent(_repo(request, session), owner.owner_id, consent)
        except ConsentError as error:
            raise _problem(error) from error
        return owner

    return Depends(guard)
