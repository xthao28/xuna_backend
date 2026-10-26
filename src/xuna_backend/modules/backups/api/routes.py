"""CLD-20 backup, restore, and deletion tombstone endpoints."""

import base64
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import JSONResponse

from xuna_backend.core.config import Settings
from xuna_backend.core.db import commit_or_conflict
from xuna_backend.core.errors import AppError
from xuna_backend.core.idempotency import StoredResponse, parse_key, request_hash, run_once
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.backups.application.ports import BackupRepository
from xuna_backend.modules.backups.application.schemas import (
    BackupManifestResponse,
    CreateBackupRequest,
    CreateTombstoneRequest,
    EncryptedDataSchema,
    ForgetAllResponse,
    ListBackupsResponse,
    ListTombstonesResponse,
    RestoreBackupRequest,
    RestoreBackupResponse,
    TombstoneResponse,
)
from xuna_backend.modules.backups.application.use_cases import (
    BackupError,
    CreateBackupInput,
    RestoreDeviceProofInput,
    create_backup,
    create_tombstone,
    delete_backup,
    forget_all,
    get_backup,
    list_backups,
    list_tombstones,
    restore_backup,
)
from xuna_backend.modules.backups.domain.entities import (
    BackupManifest,
    DeletionTombstone,
    EncryptedData,
)
from xuna_backend.modules.identity.application.ports import (
    IdentityRepository,
    SignatureVerifier,
)
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.application.ports import MemoryRepository

router = APIRouter(prefix="/v1", tags=["backups"])
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]

_STATUS: dict[str, HTTPStatus] = {
    "not_found": HTTPStatus.NOT_FOUND,
    "invalid_input": HTTPStatus.UNPROCESSABLE_ENTITY,
    "validation_failed": HTTPStatus.UNPROCESSABLE_ENTITY,
    "challenge_invalid": HTTPStatus.CONFLICT,
    "device_key_mismatch": HTTPStatus.CONFLICT,
    "invalid_public_key": HTTPStatus.UNPROCESSABLE_ENTITY,
    "invalid_signature": HTTPStatus.FORBIDDEN,
    "corrupt_backup": HTTPStatus.UNPROCESSABLE_ENTITY,
    "tombstone_conflict": HTTPStatus.CONFLICT,
}


def _backup_repo(request: Request, session: object) -> BackupRepository:
    factory: Callable[[object], BackupRepository] = state(request, "backup_repository")
    return factory(session)


def _identity_repo(request: Request, session: object) -> IdentityRepository:
    factory: Callable[[object], IdentityRepository] = state(request, "identity_repository")
    return factory(session)


def _memory_repo(request: Request, session: object) -> MemoryRepository:
    factory: Callable[[object], MemoryRepository] = state(request, "memory_repository")
    return factory(session)


def _secret_key(request: Request) -> str:
    settings: Settings = request.app.state.settings
    return settings.backup_encryption_key or settings.database_url


def _problem(error: BackupError) -> AppError:
    return AppError(error.code, _STATUS.get(error.code, HTTPStatus.BAD_REQUEST))


def _json(response: StoredResponse) -> JSONResponse:
    return JSONResponse(response.body, status_code=response.status.value)


def _to_manifest_response(manifest: BackupManifest) -> BackupManifestResponse:
    return BackupManifestResponse(
        backupId=str(manifest.backup_id),
        petId=manifest.pet_id.value,
        brainId=str(manifest.brain_id),
        snapshotGeneration=manifest.snapshot_generation,
        schemaVersion=manifest.schema_version,
        payloadSha256=manifest.payload_sha256,
        encryptedData=EncryptedDataSchema(
            algorithm=manifest.encrypted_data.algorithm,
            keyId=manifest.encrypted_data.key_id,
            nonce=manifest.encrypted_data.nonce,
            ciphertext=manifest.encrypted_data.ciphertext,
            tag=manifest.encrypted_data.tag,
        ),
        createdAt=manifest.created_at.isoformat(),
    )


def _to_tombstone_response(tombstone: DeletionTombstone) -> TombstoneResponse:
    return TombstoneResponse(
        tombstoneId=str(tombstone.tombstone_id),
        petId=tombstone.pet_id.value,
        itemType=tombstone.item_type,
        itemId=tombstone.item_id,
        deletedAt=tombstone.deleted_at.isoformat(),
        reason=tombstone.reason,
    )


@router.post("/pets/{pet_id}/backups", status_code=HTTPStatus.CREATED)
async def post_backup(
    pet_id: str,
    body: CreateBackupRequest,
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

    repo = _backup_repo(request, session)
    id_repo = _identity_repo(request, session)
    audit: AuditLog = state(request, "audit_log")(session)
    secret = _secret_key(request)

    encrypted_domain: EncryptedData | None = None
    if body.encryptedData is not None:
        encrypted_domain = EncryptedData(
            algorithm=body.encryptedData.algorithm,
            key_id=body.encryptedData.keyId,
            nonce=body.encryptedData.nonce,
            ciphertext=body.encryptedData.ciphertext,
            tag=body.encryptedData.tag,
        )

    command = CreateBackupInput(
        snapshot_generation=body.snapshotGeneration,
        export_payload=body.export,
        encrypted_data=encrypted_domain,
        payload_sha256=body.payloadSha256,
    )

    async def action() -> StoredResponse:
        try:
            manifest = await create_backup(
                repo=repo,
                identity_repo=id_repo,
                audit=audit,
                owner_id=owner.owner_id,
                pet_id=parsed_pet_id,
                command=command,
                secret_key=secret,
                now=now,
                request_id=getattr(request.state, "request_id", ""),
            )
            resp = _to_manifest_response(manifest)
            return StoredResponse(HTTPStatus.CREATED, resp.model_dump(mode="json"))
        except BackupError as error:
            raise _problem(error) from error

    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash("backups", body.model_dump(mode="json")),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)


@router.get("/pets/{pet_id}/backups")
async def get_backups(
    pet_id: str,
    owner: CurrentOwner,
    session: Session,
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ListBackupsResponse:
    repo = _backup_repo(request, session)
    id_repo = _identity_repo(request, session)
    try:
        manifests = await list_backups(
            repo=repo,
            identity_repo=id_repo,
            owner_id=owner.owner_id,
            pet_id=PetId(pet_id),
            limit=limit,
        )
        return ListBackupsResponse(items=[_to_manifest_response(m) for m in manifests])
    except BackupError as error:
        raise _problem(error) from error


@router.get("/pets/{pet_id}/backups/{backup_id}")
async def get_backup_by_id(
    pet_id: str,
    backup_id: uuid.UUID,
    owner: CurrentOwner,
    session: Session,
    request: Request,
) -> BackupManifestResponse:
    repo = _backup_repo(request, session)
    id_repo = _identity_repo(request, session)
    try:
        manifest = await get_backup(
            repo=repo,
            identity_repo=id_repo,
            owner_id=owner.owner_id,
            pet_id=PetId(pet_id),
            backup_id=backup_id,
        )
        return _to_manifest_response(manifest)
    except BackupError as error:
        raise _problem(error) from error


@router.delete("/pets/{pet_id}/backups/{backup_id}", status_code=HTTPStatus.NO_CONTENT)
async def delete_backup_by_id(
    pet_id: str,
    backup_id: uuid.UUID,
    owner: CurrentOwner,
    session: Session,
    request: Request,
) -> Response:
    repo = _backup_repo(request, session)
    id_repo = _identity_repo(request, session)
    try:
        await delete_backup(
            repo=repo,
            identity_repo=id_repo,
            owner_id=owner.owner_id,
            pet_id=PetId(pet_id),
            backup_id=backup_id,
        )
        await commit_or_conflict(session)
        return Response(status_code=HTTPStatus.NO_CONTENT)
    except BackupError as error:
        raise _problem(error) from error


@router.post("/pets/{pet_id}/backups/{backup_id}/restore")
async def post_restore(
    pet_id: str,
    backup_id: uuid.UUID,
    body: RestoreBackupRequest,
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

    repo = _backup_repo(request, session)
    id_repo = _identity_repo(request, session)
    verifier: SignatureVerifier = state(request, "signature_verifier")
    audit: AuditLog = state(request, "audit_log")(session)
    secret = _secret_key(request)

    device_proof: RestoreDeviceProofInput | None = None
    if body.deviceProof is not None:
        try:
            pubkey_bytes = base64.b64decode(body.deviceProof.publicKey)
            sig_bytes = base64.b64decode(body.deviceProof.signature)
            challenge_uuid = uuid.UUID(body.deviceProof.challengeId)
            device_proof = RestoreDeviceProofInput(
                challenge_id=challenge_uuid,
                public_key=pubkey_bytes,
                signature=sig_bytes,
            )
        except Exception as error:
            raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY) from error

    async def action() -> StoredResponse:
        try:
            result = await restore_backup(
                repo=repo,
                identity_repo=id_repo,
                verifier=verifier,
                audit=audit,
                owner_id=owner.owner_id,
                pet_id=parsed_pet_id,
                backup_id=backup_id,
                device_proof=device_proof,
                secret_key=secret,
                now=now,
                request_id=getattr(request.state, "request_id", ""),
            )
            resp = RestoreBackupResponse(
                petId=result.pet_id.value,
                brainId=str(result.brain_id),
                deviceId=str(result.device_id),
                snapshotGeneration=result.snapshot_generation,
                restoredPayload=result.restored_payload,
                tombstonesApplied=result.tombstones_applied,
            )
            return StoredResponse(HTTPStatus.OK, resp.model_dump(mode="json"))
        except BackupError as error:
            raise _problem(error) from error

    fingerprint_data: dict[str, Any] = {
        "backupId": str(backup_id),
        "body": body.model_dump(mode="json"),
    }
    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash("restore", fingerprint_data),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)


@router.post("/pets/{pet_id}/tombstones", status_code=HTTPStatus.CREATED)
async def post_tombstone(
    pet_id: str,
    body: CreateTombstoneRequest,
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

    repo = _backup_repo(request, session)
    id_repo = _identity_repo(request, session)
    mem_repo = _memory_repo(request, session)
    audit: AuditLog = state(request, "audit_log")(session)

    async def action() -> StoredResponse:
        try:
            tombstone = await create_tombstone(
                repo=repo,
                identity_repo=id_repo,
                memory_repo=mem_repo,
                audit=audit,
                owner_id=owner.owner_id,
                pet_id=parsed_pet_id,
                item_type=body.itemType,
                item_id=body.itemId,
                reason=body.reason,
                now=now,
                request_id=getattr(request.state, "request_id", ""),
            )
            resp = _to_tombstone_response(tombstone)
            return StoredResponse(HTTPStatus.CREATED, resp.model_dump(mode="json"))
        except BackupError as error:
            raise _problem(error) from error

    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash("tombstones", body.model_dump(mode="json")),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)


@router.get("/pets/{pet_id}/tombstones")
async def get_tombstones(
    pet_id: str,
    owner: CurrentOwner,
    session: Session,
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> ListTombstonesResponse:
    repo = _backup_repo(request, session)
    id_repo = _identity_repo(request, session)
    try:
        tombstones = await list_tombstones(
            repo=repo,
            identity_repo=id_repo,
            owner_id=owner.owner_id,
            pet_id=PetId(pet_id),
            limit=limit,
        )
        return ListTombstonesResponse(items=[_to_tombstone_response(t) for t in tombstones])
    except BackupError as error:
        raise _problem(error) from error


@router.delete("/pets/{pet_id}/memory-candidates/{candidate_id}")
async def delete_candidate_with_tombstone(
    pet_id: str,
    candidate_id: uuid.UUID,
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

    repo = _backup_repo(request, session)
    id_repo = _identity_repo(request, session)
    mem_repo = _memory_repo(request, session)
    audit: AuditLog = state(request, "audit_log")(session)

    async def action() -> StoredResponse:
        try:
            tombstone = await create_tombstone(
                repo=repo,
                identity_repo=id_repo,
                memory_repo=mem_repo,
                audit=audit,
                owner_id=owner.owner_id,
                pet_id=parsed_pet_id,
                item_type="candidate",
                item_id=str(candidate_id),
                reason="deleted_by_owner",
                now=now,
                request_id=getattr(request.state, "request_id", ""),
            )
            resp = _to_tombstone_response(tombstone)
            return StoredResponse(HTTPStatus.OK, resp.model_dump(mode="json"))
        except BackupError as error:
            raise _problem(error) from error

    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash("delete_candidate", {"candidateId": str(candidate_id)}),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)


@router.post("/pets/{pet_id}/forget")
async def post_forget(
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

    repo = _backup_repo(request, session)
    id_repo = _identity_repo(request, session)
    mem_repo = _memory_repo(request, session)
    audit: AuditLog = state(request, "audit_log")(session)

    async def action() -> StoredResponse:
        try:
            p_id, new_brain_id, tombstone_id = await forget_all(
                repo=repo,
                identity_repo=id_repo,
                memory_repo=mem_repo,
                audit=audit,
                owner_id=owner.owner_id,
                pet_id=parsed_pet_id,
                now=now,
                request_id=getattr(request.state, "request_id", ""),
            )
            resp = ForgetAllResponse(
                petId=p_id.value,
                newBrainId=str(new_brain_id),
                tombstoneId=str(tombstone_id),
            )
            return StoredResponse(HTTPStatus.OK, resp.model_dump(mode="json"))
        except BackupError as error:
            raise _problem(error) from error

    response = await run_once(
        session,
        owner_id=owner.owner_id,
        key=parse_key(idempotency_key),
        fingerprint=request_hash("forget_all", {"petId": parsed_pet_id.value}),
        now=now,
        action=action,
    )
    await commit_or_conflict(session)
    return _json(response)
