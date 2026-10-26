"""CLD-30 / CLD-31 Suggestions and sleep learning consolidation API routes."""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import JSONResponse

from xuna_backend.core.db import commit_or_conflict
from xuna_backend.core.errors import AppError
from xuna_backend.core.idempotency import StoredResponse, parse_key, request_hash, run_once
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.consent.application.ports import ConsentRepository
from xuna_backend.modules.consent.domain.consent import ConsentState
from xuna_backend.modules.identity.application.ports import IdentityRepository
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.application.ports import MemoryRepository
from xuna_backend.modules.suggestions.application.ports import SuggestionRepository
from xuna_backend.modules.suggestions.application.schemas import (
    CloudSuggestionResponse,
    ConsolidateSuggestionsResponse,
    ListSuggestionsResponse,
    PatchSuggestionRequest,
)
from xuna_backend.modules.suggestions.application.use_cases import (
    SuggestionError,
    consolidate_suggestions,
    delete_suggestion,
    get_suggestion,
    list_suggestions,
    update_suggestion_status,
)
from xuna_backend.modules.suggestions.domain.entities import (
    CloudSuggestion,
    SuggestionKind,
    SuggestionStatus,
)

router = APIRouter(prefix="/v1", tags=["suggestions"])
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]

_STATUS: dict[str, HTTPStatus] = {
    "consent_required": HTTPStatus.FORBIDDEN,
    "not_found": HTTPStatus.NOT_FOUND,
    "invalid_transition": HTTPStatus.CONFLICT,
    "suggestion_expired": HTTPStatus.CONFLICT,
    "invalid_action": HTTPStatus.UNPROCESSABLE_ENTITY,
}


def _suggestion_repo(request: Request, session: object) -> SuggestionRepository:
    factory: Callable[[object], SuggestionRepository] = state(request, "suggestion_repository")
    return factory(session)


def _identity_repo(request: Request, session: object) -> IdentityRepository:
    factory: Callable[[object], IdentityRepository] = state(request, "identity_repository")
    return factory(session)


def _memory_repo(request: Request, session: object) -> MemoryRepository:
    factory: Callable[[object], MemoryRepository] = state(request, "memory_repository")
    return factory(session)


async def _current_consent(request: Request, session: object, owner_id: uuid.UUID) -> ConsentState:
    factory: Callable[[object], ConsentRepository] = state(request, "consent_repository")
    return await factory(session).current(owner_id)


def _problem(error: SuggestionError) -> AppError:
    return AppError(error.code, _STATUS.get(error.code, HTTPStatus.BAD_REQUEST))


def _to_suggestion_response(s: CloudSuggestion) -> CloudSuggestionResponse:
    return CloudSuggestionResponse(
        suggestionId=str(s.suggestion_id),
        petId=s.pet_id.value,
        brainId=str(s.brain_id),
        kind=s.kind.value,
        target=s.target,
        delta=s.delta,
        bounds=s.bounds,
        provenance=s.provenance.value,
        evidenceRef=s.evidence_ref,
        status=s.status.value,
        expiresAt=s.expires_at.astimezone(UTC).isoformat(),
        createdAt=s.created_at.astimezone(UTC).isoformat(),
        appliedAt=s.applied_at.astimezone(UTC).isoformat() if s.applied_at else None,
    )


def _json(response: StoredResponse) -> JSONResponse:
    return JSONResponse(response.body, status_code=response.status.value)


@router.post("/pets/{pet_id}/suggestions/consolidate")
async def post_consolidate(
    pet_id: str,
    owner: CurrentOwner,
    session: Session,
    request: Request,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    now = datetime.now(UTC)
    try:
        parsed_pet_id = PetId(pet_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    repo = _suggestion_repo(request, session)
    id_repo = _identity_repo(request, session)
    mem_repo = _memory_repo(request, session)
    audit: AuditLog = state(request, "audit_log")(session)

    async def action() -> StoredResponse:
        consent = await _current_consent(request, session, owner.owner_id)
        try:
            suggestions = await consolidate_suggestions(
                repo=repo,
                identity_repo=id_repo,
                memory_repo=mem_repo,
                audit=audit,
                owner_id=owner.owner_id,
                pet_id=parsed_pet_id,
                consent=consent,
                now=now,
                request_id=getattr(request.state, "request_id", ""),
            )
            items = [_to_suggestion_response(s) for s in suggestions]
            resp = ConsolidateSuggestionsResponse(items=items, count=len(items))
            return StoredResponse(HTTPStatus.OK, resp.model_dump(mode="json"))
        except SuggestionError as error:
            raise _problem(error) from error

    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash("consolidate", {"pet_id": pet_id}),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)


@router.get("/pets/{pet_id}/suggestions")
async def get_suggestions(
    pet_id: str,
    owner: CurrentOwner,
    session: Session,
    request: Request,
    status: str | None = Query(default=None),
    kind: str | None = Query(default=None),
) -> ListSuggestionsResponse:
    try:
        parsed_pet_id = PetId(pet_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    repo = _suggestion_repo(request, session)
    id_repo = _identity_repo(request, session)

    status_filter = SuggestionStatus(status) if status else None
    kind_filter = SuggestionKind(kind) if kind else None

    try:
        suggestions = await list_suggestions(
            repo=repo,
            identity_repo=id_repo,
            owner_id=owner.owner_id,
            pet_id=parsed_pet_id,
            status=status_filter,
            kind=kind_filter,
        )
        return ListSuggestionsResponse(
            items=[_to_suggestion_response(s) for s in suggestions],
        )
    except SuggestionError as error:
        raise _problem(error) from error


@router.get("/pets/{pet_id}/suggestions/{suggestion_id}")
async def get_suggestion_by_id(
    pet_id: str,
    suggestion_id: str,
    owner: CurrentOwner,
    session: Session,
    request: Request,
) -> CloudSuggestionResponse:
    try:
        parsed_pet_id = PetId(pet_id)
        parsed_suggestion_id = uuid.UUID(suggestion_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    repo = _suggestion_repo(request, session)
    id_repo = _identity_repo(request, session)

    try:
        suggestion = await get_suggestion(
            repo=repo,
            identity_repo=id_repo,
            owner_id=owner.owner_id,
            pet_id=parsed_pet_id,
            suggestion_id=parsed_suggestion_id,
        )
        return _to_suggestion_response(suggestion)
    except SuggestionError as error:
        raise _problem(error) from error


@router.patch("/pets/{pet_id}/suggestions/{suggestion_id}")
async def patch_suggestion(
    pet_id: str,
    suggestion_id: str,
    body: PatchSuggestionRequest,
    owner: CurrentOwner,
    session: Session,
    request: Request,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    now = datetime.now(UTC)
    try:
        parsed_pet_id = PetId(pet_id)
        parsed_suggestion_id = uuid.UUID(suggestion_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    repo = _suggestion_repo(request, session)
    id_repo = _identity_repo(request, session)
    audit: AuditLog = state(request, "audit_log")(session)

    async def action() -> StoredResponse:
        try:
            updated = await update_suggestion_status(
                repo=repo,
                identity_repo=id_repo,
                audit=audit,
                owner_id=owner.owner_id,
                pet_id=parsed_pet_id,
                suggestion_id=parsed_suggestion_id,
                action=body.action,
                now=now,
                request_id=getattr(request.state, "request_id", ""),
            )
            resp = _to_suggestion_response(updated)
            return StoredResponse(HTTPStatus.OK, resp.model_dump(mode="json"))
        except SuggestionError as error:
            raise _problem(error) from error

    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash("patch_suggestion", {"id": suggestion_id, "action": body.action}),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)


@router.delete("/pets/{pet_id}/suggestions/{suggestion_id}", status_code=HTTPStatus.NO_CONTENT)
async def delete_suggestion_by_id(
    pet_id: str,
    suggestion_id: str,
    owner: CurrentOwner,
    session: Session,
    request: Request,
) -> Response:
    try:
        parsed_pet_id = PetId(pet_id)
        parsed_suggestion_id = uuid.UUID(suggestion_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    repo = _suggestion_repo(request, session)
    id_repo = _identity_repo(request, session)

    try:
        await delete_suggestion(
            repo=repo,
            identity_repo=id_repo,
            owner_id=owner.owner_id,
            pet_id=parsed_pet_id,
            suggestion_id=parsed_suggestion_id,
        )
        await commit_or_conflict(session)
        return Response(status_code=HTTPStatus.NO_CONTENT)
    except SuggestionError as error:
        raise _problem(error) from error
