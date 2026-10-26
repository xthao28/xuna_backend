"""Use cases for backups, restore, and deletion tombstones (CLD-20, ADR-0008).

Enforces:
1. Ownership verification on every operation.
2. Hardware replacement (rebind device) keeps PetId and BrainId.
3. ForgetAll keeps PetId, archives old Brain, and starts a new BrainId.
4. Tombstones prevent resurrection of deleted memories on restore.
5. Corrupt or tampered backups fail safely.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.backups.application.crypto import (
    decrypt_payload,
    derive_key,
    encrypt_payload,
)
from xuna_backend.modules.backups.application.ports import BackupRepository
from xuna_backend.modules.backups.domain.entities import (
    BackupManifest,
    DeletionTombstone,
    EncryptedData,
    RestoreResult,
)
from xuna_backend.modules.identity.application.ports import (
    IdentityRepository,
    SignatureVerifier,
)
from xuna_backend.modules.identity.domain.device_proof import (
    SIGNATURE_SIZE,
    ProofPurpose,
    proof_message,
    validate_public_key,
)
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.application.ports import MemoryRepository
from xuna_backend.modules.memory.domain.entities import CandidateStatus, MemoryCandidate


class BackupError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class CreateBackupInput:
    snapshot_generation: int
    export_payload: dict[str, Any] | None = None
    encrypted_data: EncryptedData | None = None
    payload_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RestoreDeviceProofInput:
    challenge_id: uuid.UUID
    public_key: bytes
    signature: bytes


async def create_backup(
    repo: BackupRepository,
    identity_repo: IdentityRepository,
    audit: AuditLog,
    owner_id: uuid.UUID,
    pet_id: PetId,
    command: CreateBackupInput,
    secret_key: str | bytes,
    now: datetime,
    request_id: str = "",
) -> BackupManifest:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise BackupError("not_found")

    if command.encrypted_data is not None:
        if not command.payload_sha256:
            raise BackupError("invalid_input")
        encrypted_entity = command.encrypted_data
        sha256_hex = command.payload_sha256
    elif command.export_payload is not None:
        # Sanitize export payload against active tombstones before backup
        payload_copy = dict(command.export_payload)
        tombstones = await repo.list_tombstones(pet_id)
        tombstone_ids = {t.item_id for t in tombstones}
        if "payload" in payload_copy and isinstance(payload_copy["payload"], dict):
            inner = dict(payload_copy["payload"])
            if "memories" in inner and isinstance(inner["memories"], list):
                inner["memories"] = [
                    m for m in inner["memories"]
                    if str(m.get("context", "")) not in tombstone_ids
                ]
            payload_copy["payload"] = inner

        raw_bytes = json.dumps(payload_copy, sort_keys=True).encode("utf-8")
        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
        key = derive_key(secret_key)
        encrypted_entity = encrypt_payload(raw_bytes, key, key_id="server-aes256")
    else:
        raise BackupError("invalid_input")

    manifest = BackupManifest(
        backup_id=uuid.uuid4(),
        pet_id=pet_id,
        owner_id=owner_id,
        brain_id=pet.active_brain_id,
        snapshot_generation=command.snapshot_generation,
        schema_version=1,
        payload_sha256=sha256_hex,
        encrypted_data=encrypted_entity,
        created_at=now,
    )
    await repo.add_backup(manifest)

    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.BACKUP_CREATED,
            occurred_at=now,
            subject=pet_id.value,
            request_id=request_id,
            metadata={
                "backupId": str(manifest.backup_id),
                "generation": manifest.snapshot_generation,
            },
        )
    )
    return manifest


async def list_backups(
    repo: BackupRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    pet_id: PetId,
    limit: int = 50,
) -> list[BackupManifest]:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise BackupError("not_found")
    return await repo.list_backups(pet_id, limit=limit)


async def get_backup(
    repo: BackupRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    pet_id: PetId,
    backup_id: uuid.UUID,
) -> BackupManifest:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise BackupError("not_found")
    manifest = await repo.get_backup(backup_id)
    if manifest is None or manifest.pet_id != pet_id or manifest.owner_id != owner_id:
        raise BackupError("not_found")
    return manifest


async def delete_backup(
    repo: BackupRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    pet_id: PetId,
    backup_id: uuid.UUID,
) -> None:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise BackupError("not_found")
    manifest = await repo.get_backup(backup_id)
    if manifest is None or manifest.pet_id != pet_id or manifest.owner_id != owner_id:
        raise BackupError("not_found")
    await repo.delete_backup(backup_id)


async def restore_backup(
    repo: BackupRepository,
    identity_repo: IdentityRepository,
    verifier: SignatureVerifier,
    audit: AuditLog,
    owner_id: uuid.UUID,
    pet_id: PetId,
    backup_id: uuid.UUID,
    device_proof: RestoreDeviceProofInput | None,
    secret_key: str | bytes,
    now: datetime,
    request_id: str = "",
) -> RestoreResult:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise BackupError("not_found")

    manifest = await repo.get_backup(backup_id)
    if manifest is None or manifest.pet_id != pet_id or manifest.owner_id != owner_id:
        raise BackupError("not_found")

    active_device_id = pet.device_id
    if device_proof is not None:
        # Replacement hardware: rebind to new device key while preserving PetId and BrainId
        try:
            validate_public_key(device_proof.public_key)
        except ValueError as error:
            raise BackupError("invalid_public_key") from error
        if len(device_proof.signature) != SIGNATURE_SIZE:
            raise BackupError("invalid_signature")

        challenge = await identity_repo.consume_challenge(device_proof.challenge_id)
        if (
            challenge is None
            or challenge.consumed
            or challenge.owner_id != owner_id
            or challenge.pet_id != pet_id
            or challenge.expires_at <= now
        ):
            raise BackupError("challenge_invalid")

        message = proof_message(
            challenge.nonce,
            pet_id,
            ProofPurpose.BIND,
            manifest.snapshot_generation,
            device_proof.public_key,
        )
        if not verifier.verify(device_proof.public_key, message, device_proof.signature):
            raise BackupError("invalid_signature")

        # Rebind to new device key
        pet = await identity_repo.rebind_device(pet_id, device_proof.public_key, now)
        active_device_id = pet.device_id

    # Decrypt and verify payload integrity
    key = derive_key(secret_key)
    try:
        raw_bytes = decrypt_payload(manifest.encrypted_data, key)
    except ValueError as error:
        raise BackupError("corrupt_backup") from error

    computed_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if computed_sha256 != manifest.payload_sha256:
        raise BackupError("corrupt_backup")

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except Exception as error:
        raise BackupError("corrupt_backup") from error

    # Apply deletion tombstones (T10: prevent resurrection of deleted memories)
    tombstones = await repo.list_tombstones(pet_id)
    tombstone_ids = {t.item_id for t in tombstones}
    tombstones_applied = 0

    if isinstance(payload, dict) and "payload" in payload and isinstance(payload["payload"], dict):
        inner = payload["payload"]
        if "memories" in inner and isinstance(inner["memories"], list):
            original_count = len(inner["memories"])
            filtered = [
                m for m in inner["memories"]
                if str(m.get("context", "")) not in tombstone_ids
                and str(m.get("action", "")) not in tombstone_ids
            ]
            tombstones_applied += (original_count - len(filtered))
            inner["memories"] = filtered

    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.BACKUP_RESTORED,
            occurred_at=now,
            subject=pet_id.value,
            request_id=request_id,
            metadata={
                "backupId": str(manifest.backup_id),
                "generation": manifest.snapshot_generation,
                "tombstones": tombstones_applied,
            },
        )
    )

    return RestoreResult(
        pet_id=pet_id,
        brain_id=pet.active_brain_id,
        device_id=active_device_id,
        snapshot_generation=manifest.snapshot_generation,
        restored_payload=payload,
        tombstones_applied=tombstones_applied,
    )


async def create_tombstone(
    repo: BackupRepository,
    identity_repo: IdentityRepository,
    memory_repo: MemoryRepository | None,
    audit: AuditLog,
    owner_id: uuid.UUID,
    pet_id: PetId,
    item_type: str,
    item_id: str,
    reason: str | None,
    now: datetime,
    request_id: str = "",
) -> DeletionTombstone:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise BackupError("not_found")

    tombstone = DeletionTombstone(
        tombstone_id=uuid.uuid4(),
        pet_id=pet_id,
        owner_id=owner_id,
        item_type=item_type,
        item_id=item_id,
        deleted_at=now,
        reason=reason,
    )
    await repo.add_tombstone(tombstone)

    # If item_type is candidate, mark candidate in memory_candidates table as DELETED
    if item_type == "candidate" and memory_repo is not None:
        try:
            candidate_uuid = uuid.UUID(item_id)
            candidate = await memory_repo.get_candidate(candidate_uuid)
            if candidate is not None and candidate.pet_id == pet_id:
                updated = MemoryCandidate(
                    candidate_id=candidate.candidate_id,
                    owner_id=candidate.owner_id,
                    pet_id=candidate.pet_id,
                    source=candidate.source,
                    candidate_type=candidate.candidate_type,
                    content=candidate.content,
                    status=CandidateStatus.DELETED,
                    created_at=candidate.created_at,
                    updated_at=now,
                    confirmed_at=candidate.confirmed_at,
                    committed_generation=candidate.committed_generation,
                )
                await memory_repo.update_candidate(updated)
        except ValueError:
            pass

    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.TOMBSTONE_CREATED,
            occurred_at=now,
            subject=pet_id.value,
            request_id=request_id,
            metadata={"itemType": item_type[:32]},
        )
    )
    return tombstone


async def list_tombstones(
    repo: BackupRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    pet_id: PetId,
    limit: int = 100,
) -> list[DeletionTombstone]:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise BackupError("not_found")
    return await repo.list_tombstones(pet_id, limit=limit)


async def forget_all(
    repo: BackupRepository,
    identity_repo: IdentityRepository,
    memory_repo: MemoryRepository | None,
    audit: AuditLog,
    owner_id: uuid.UUID,
    pet_id: PetId,
    now: datetime,
    request_id: str = "",
) -> tuple[PetId, uuid.UUID, uuid.UUID]:
    """ForgetAll keeps PetId, archives old BrainId, starts a new BrainId, and leaves a tombstone."""
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise BackupError("not_found")

    # Rotate brain: archives active brain, starts a new active BrainId
    updated_pet = await identity_repo.rotate_brain(pet_id, now)

    # Leave a master tombstone
    tombstone = DeletionTombstone(
        tombstone_id=uuid.uuid4(),
        pet_id=pet_id,
        owner_id=owner_id,
        item_type="memory",
        item_id="*",
        deleted_at=now,
        reason="forget_all",
    )
    await repo.add_tombstone(tombstone)

    # Mark all candidates deleted
    if memory_repo is not None:
        candidates = await memory_repo.list_candidates(pet_id, limit=500)
        for c in candidates:
            if c.status not in (CandidateStatus.REJECTED, CandidateStatus.DELETED):
                updated_cand = MemoryCandidate(
                    candidate_id=c.candidate_id,
                    owner_id=c.owner_id,
                    pet_id=c.pet_id,
                    source=c.source,
                    candidate_type=c.candidate_type,
                    content=c.content,
                    status=CandidateStatus.DELETED,
                    created_at=c.created_at,
                    updated_at=now,
                    confirmed_at=c.confirmed_at,
                    committed_generation=c.committed_generation,
                )
                await memory_repo.update_candidate(updated_cand)

    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.PET_FORGOTTEN,
            occurred_at=now,
            subject=pet_id.value,
            request_id=request_id,
            metadata={"newBrain": str(updated_pet.active_brain_id)},
        )
    )
    return pet_id, updated_pet.active_brain_id, tombstone.tombstone_id
