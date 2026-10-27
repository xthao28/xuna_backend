"""CLD-11 identity endpoints (contract draft §2, §4)."""

import base64
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import JSONResponse

from xuna_backend.core.db import commit_or_conflict
from xuna_backend.core.errors import AppError
from xuna_backend.core.idempotency import StoredResponse, parse_key, request_hash, run_once
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.identity.api.schemas import BindPetRequest, ChallengeRequest
from xuna_backend.modules.identity.application.ports import (
    IdentityRepository,
    SignatureVerifier,
)
from xuna_backend.modules.identity.application.use_cases import (
    BindPet,
    IdentityError,
    bind_pet,
    get_owned_pet,
    issue_challenge,
    list_pets,
    unbind_pet,
)
from xuna_backend.modules.identity.domain.entities import Pet
from xuna_backend.modules.identity.domain.pet_id import PetId

router = APIRouter(prefix="/v1", tags=["identity"])
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]

_STATUS = {
    "invalid_public_key": HTTPStatus.UNPROCESSABLE_ENTITY,
    "invalid_signature": HTTPStatus.FORBIDDEN,
    "challenge_invalid": HTTPStatus.CONFLICT,
    "pet_already_bound": HTTPStatus.CONFLICT,
    "device_key_mismatch": HTTPStatus.CONFLICT,
    "not_found": HTTPStatus.NOT_FOUND,
}


def _repo(request: Request, session: object) -> IdentityRepository:
    factory: Callable[[object], IdentityRepository] = state(request, "identity_repository")
    return factory(session)


def _problem(error: IdentityError) -> AppError:
    return AppError(error.code, _STATUS.get(error.code, HTTPStatus.BAD_REQUEST))


def _pet_body(pet: Pet) -> dict[str, Any]:
    return {
        "petId": pet.pet_id.value,
        "brainId": str(pet.active_brain_id),
        "deviceId": str(pet.device_id),
        "origin": pet.origin.value,
        "lastGeneration": pet.last_generation,
        "boundAt": pet.bound_at.astimezone(UTC).isoformat(),
        "deviceBound": pet.device_bound,
    }


def _json(response: StoredResponse) -> JSONResponse:
    return JSONResponse(response.body, status_code=response.status.value)


@router.post("/devices/challenges", status_code=HTTPStatus.CREATED)
async def create_challenge(
    body: ChallengeRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    now = datetime.now(UTC)

    async def action() -> StoredResponse:
        challenge = await issue_challenge(
            _repo(request, session), owner.owner_id, PetId(body.petId), now
        )
        return StoredResponse(
            HTTPStatus.CREATED,
            {
                "challengeId": str(challenge.challenge_id),
                "nonce": base64.b64encode(challenge.nonce).decode(),
                "expiresAt": challenge.expires_at.astimezone(UTC).isoformat(),
            },
        )

    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash("challenges", body.model_dump(mode="json")),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)


@router.post("/pets", status_code=HTTPStatus.CREATED)
async def create_pet(
    body: BindPetRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    now = datetime.now(UTC)
    verifier: SignatureVerifier = state(request, "signature_verifier")
    try:
        challenge_id = uuid.UUID(body.deviceProof.challengeId)
    except ValueError as error:
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY) from error
    command = BindPet(
        pet_id=PetId(body.petId),
        origin=body.origin,
        snapshot_generation=body.snapshotGeneration,
        challenge_id=challenge_id,
        public_key=body.deviceProof.key_bytes(),
        signature=body.deviceProof.signature_bytes(),
    )

    async def action() -> StoredResponse:
        audit_for: Callable[[object], AuditLog] = state(request, "audit_log")
        pet, created = await bind_pet(
            _repo(request, session),
            verifier,
            audit_for(session),
            owner.owner_id,
            command,
            now,
            request.state.request_id,
        )
        return StoredResponse(HTTPStatus.CREATED if created else HTTPStatus.OK, _pet_body(pet))

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash("pets", body.model_dump(mode="json")),
            now=now,
            action=action,
        )
    except IdentityError as error:
        await commit_or_conflict(session)  # keep the challenge consumed: single use
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)


@router.get("/pets")
async def get_pets(request: Request, owner: CurrentOwner, session: Session) -> dict[str, Any]:
    pets = await list_pets(_repo(request, session), owner.owner_id)
    return {"items": [_pet_body(pet) for pet in pets]}


@router.get("/pets/{pet_id}")
async def get_pet(
    pet_id: str, request: Request, owner: CurrentOwner, session: Session
) -> dict[str, Any]:
    try:
        parsed = PetId(pet_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    try:
        pet = await get_owned_pet(_repo(request, session), owner.owner_id, parsed)
    except IdentityError as error:
        raise _problem(error) from error
    return _pet_body(pet)


@router.delete("/pets/{pet_id}/device", status_code=HTTPStatus.NO_CONTENT)
async def unbind_device(
    pet_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> Response:
    """Unbinds the board: its key is revoked, the pet's cloud data stays."""
    try:
        parsed = PetId(pet_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)

    async def action() -> StoredResponse:
        audit_for: Callable[[object], AuditLog] = state(request, "audit_log")
        await unbind_pet(
            _repo(request, session),
            audit_for(session),
            owner.owner_id,
            parsed,
            now,
            request.state.request_id,
        )
        return StoredResponse(HTTPStatus.NO_CONTENT, {})

    try:
        await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash("unbind", {"petId": parsed.value}),
            now=now,
            action=action,
        )
    except IdentityError as error:
        raise _problem(error) from error
    await commit_or_conflict(session)
    return Response(status_code=HTTPStatus.NO_CONTENT)
