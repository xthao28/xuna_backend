"""CLD-14 projections and memory candidate endpoints."""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import JSONResponse

from xuna_backend.core.db import commit_or_conflict
from xuna_backend.core.errors import AppError
from xuna_backend.core.idempotency import StoredResponse, parse_key, request_hash, run_once
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.consent.application.ports import ConsentRepository
from xuna_backend.modules.consent.domain.consent import ConsentState
from xuna_backend.modules.identity.application.ports import (
    IdentityRepository,
    SignatureVerifier,
)
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.application.ports import MemoryRepository
from xuna_backend.modules.memory.application.schemas import (
    CreateCandidateRequest,
    IngestProjectionRequest,
    PatchCandidateRequest,
)
from xuna_backend.modules.memory.application.use_cases import (
    MemoryError,
    create_candidate,
    get_latest_projection,
    ingest_projection,
    list_candidates,
    list_projections,
    patch_candidate,
)
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    MemoryCandidate,
    Projection,
)

router = APIRouter(prefix="/v1", tags=["memory"])
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]

_STATUS: dict[str, HTTPStatus] = {
    "consent_required": HTTPStatus.FORBIDDEN,
    "invalid_envelope": HTTPStatus.UNPROCESSABLE_ENTITY,
    "pet_id_mismatch": HTTPStatus.UNPROCESSABLE_ENTITY,
    "generation_mismatch": HTTPStatus.UNPROCESSABLE_ENTITY,
    "not_found": HTTPStatus.NOT_FOUND,
    "projection_conflict": HTTPStatus.CONFLICT,
    "conflict_stale_generation": HTTPStatus.CONFLICT,
    "challenge_invalid": HTTPStatus.CONFLICT,
    "device_key_mismatch": HTTPStatus.CONFLICT,
    "invalid_signature": HTTPStatus.FORBIDDEN,
    "invalid_transition": HTTPStatus.CONFLICT,
    "generation_required": HTTPStatus.UNPROCESSABLE_ENTITY,
}


def _repo(request: Request, session: object) -> MemoryRepository:
    factory: Callable[[object], MemoryRepository] = state(request, "memory_repository")
    return factory(session)


def _identity_repo(request: Request, session: object) -> IdentityRepository:
    factory: Callable[[object], IdentityRepository] = state(request, "identity_repository")
    return factory(session)


async def _current_consent(request: Request, session: object, owner_id: uuid.UUID) -> ConsentState:
    factory: Callable[[object], ConsentRepository] = state(request, "consent_repository")
    return await factory(session).current(owner_id)


def _problem(error: MemoryError) -> AppError:
    return AppError(error.code, _STATUS.get(error.code, HTTPStatus.BAD_REQUEST))


def _projection_body(projection: Projection) -> dict[str, Any]:
    return {
        "projectionId": str(projection.projection_id),
        "petId": projection.pet_id.value,
        "snapshotGeneration": projection.snapshot_generation,
        "schemaVersion": projection.schema_version,
        "payloadSha256": projection.payload_sha256,
        "payload": projection.payload,
        "deviceAttested": projection.device_attested,
        "createdAt": projection.created_at.astimezone(UTC).isoformat(),
    }


def _candidate_body(c: MemoryCandidate) -> dict[str, Any]:
    return {
        "candidateId": str(c.candidate_id),
        "petId": c.pet_id.value,
        "source": c.source,
        "candidateType": c.candidate_type.value,
        "content": c.content,
        "status": c.status.value,
        "createdAt": c.created_at.astimezone(UTC).isoformat(),
        "updatedAt": c.updated_at.astimezone(UTC).isoformat(),
        "confirmedAt": c.confirmed_at.astimezone(UTC).isoformat() if c.confirmed_at else None,
        "committedGeneration": c.committed_generation,
    }


def _json(response: StoredResponse) -> JSONResponse:
    return JSONResponse(response.body, status_code=response.status.value)


@router.put("/pets/{pet_id}/projections/{snapshot_generation}")
async def put_projection(
    pet_id: str,
    snapshot_generation: int,
    body: IngestProjectionRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    now = datetime.now(UTC)
    try:
        parsed_pet_id = PetId(pet_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    verifier: SignatureVerifier = state(request, "signature_verifier")
    audit_for: Callable[[object], AuditLog] = state(request, "audit_log")

    async def action() -> StoredResponse:
        consent = await _current_consent(request, session, owner.owner_id)
        try:
            projection, created = await ingest_projection(
                repo=_repo(request, session),
                identity_repo=_identity_repo(request, session),
                verifier=verifier,
                audit=audit_for(session),
                owner_id=owner.owner_id,
                pet_id=parsed_pet_id,
                generation=snapshot_generation,
                envelope=body.export,
                device_proof=body.deviceProof,
                consent=consent,
                now=now,
                request_id=request.state.request_id,
            )
        except MemoryError as error:
            raise _problem(error) from error

        status = HTTPStatus.CREATED if created else HTTPStatus.OK
        return StoredResponse(status, _projection_body(projection))

    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash("projections", body.model_dump(mode="json")),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)


@router.get("/pets/{pet_id}/projections/latest")
async def get_latest(
    pet_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> dict[str, Any]:
    try:
        parsed_pet_id = PetId(pet_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    try:
        projection = await get_latest_projection(
            _repo(request, session),
            _identity_repo(request, session),
            owner.owner_id,
            parsed_pet_id,
        )
    except MemoryError as error:
        raise _problem(error) from error
    return _projection_body(projection)


@router.get("/pets/{pet_id}/projections")
async def get_projections(
    pet_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> dict[str, Any]:
    try:
        parsed_pet_id = PetId(pet_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    try:
        projections = await list_projections(
            _repo(request, session),
            _identity_repo(request, session),
            owner.owner_id,
            parsed_pet_id,
            limit=limit,
        )
    except MemoryError as error:
        raise _problem(error) from error
    return {"items": [_projection_body(p) for p in projections]}


@router.post("/memory-candidates", status_code=HTTPStatus.CREATED)
async def post_candidate(
    body: CreateCandidateRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    now = datetime.now(UTC)

    async def action() -> StoredResponse:
        try:
            candidate = await create_candidate(
                _repo(request, session),
                _identity_repo(request, session),
                owner.owner_id,
                body,
                now,
            )
        except MemoryError as error:
            raise _problem(error) from error
        return StoredResponse(HTTPStatus.CREATED, _candidate_body(candidate))

    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash("memory_candidates", body.model_dump(mode="json")),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)


@router.patch("/memory-candidates/{candidate_id}")
async def patch_candidate_endpoint(
    candidate_id: str,
    body: PatchCandidateRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    now = datetime.now(UTC)
    try:
        parsed_id = uuid.UUID(candidate_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    audit_for: Callable[[object], AuditLog] = state(request, "audit_log")

    async def action() -> StoredResponse:
        try:
            updated = await patch_candidate(
                _repo(request, session),
                audit_for(session),
                owner.owner_id,
                parsed_id,
                body,
                now,
                request_id=request.state.request_id,
            )
        except MemoryError as error:
            raise _problem(error) from error
        return StoredResponse(HTTPStatus.OK, _candidate_body(updated))

    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash(f"candidate:{candidate_id}", body.model_dump(mode="json")),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)


@router.get("/memory-candidates")
async def get_candidates(
    petId: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    status: Annotated[CandidateStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> dict[str, Any]:
    try:
        parsed_pet_id = PetId(petId)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    try:
        candidates = await list_candidates(
            _repo(request, session),
            _identity_repo(request, session),
            owner.owner_id,
            parsed_pet_id,
            status=status,
            limit=limit,
        )
    except MemoryError as error:
        raise _problem(error) from error
    return {"items": [_candidate_body(c) for c in candidates]}
