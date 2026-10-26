"""Domain entities for encrypted backups, restore, and deletion tombstones (CLD-20, ADR-0008).

Domain rules: stdlib and own/other domain only. No pydantic or external frameworks.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from xuna_backend.modules.identity.domain.pet_id import PetId


@dataclass(frozen=True, slots=True)
class EncryptedData:
    algorithm: str  # e.g. "aes-256-gcm"
    key_id: str  # identifier for the key / owner-derived key
    nonce: str  # base64 encoded
    ciphertext: str  # base64 encoded
    tag: str  # base64 encoded

    def __post_init__(self) -> None:
        if (
            not self.algorithm
            or not self.key_id
            or not self.nonce
            or not self.ciphertext
            or not self.tag
        ):
            raise ValueError("all fields of EncryptedData must be non-empty")


@dataclass(frozen=True, slots=True)
class BackupManifest:
    backup_id: uuid.UUID
    pet_id: PetId
    owner_id: uuid.UUID
    brain_id: uuid.UUID
    snapshot_generation: int
    schema_version: int
    payload_sha256: str
    encrypted_data: EncryptedData
    created_at: datetime

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if not 0 <= self.snapshot_generation <= 0xFFFFFFFF:
            raise ValueError("snapshot_generation must be a u32")
        if len(self.payload_sha256) != 64:
            raise ValueError("payload_sha256 must be a 64-char hex string")


@dataclass(frozen=True, slots=True)
class DeletionTombstone:
    tombstone_id: uuid.UUID
    pet_id: PetId
    owner_id: uuid.UUID
    item_type: str  # "candidate", "fact", "stated_preference", "memory"
    item_id: str  # candidate UUID string, fact key name, or "*"
    deleted_at: datetime
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.deleted_at.tzinfo is None:
            raise ValueError("deleted_at must be timezone-aware")
        if not self.item_type or not self.item_id:
            raise ValueError("item_type and item_id must be non-empty")


@dataclass(frozen=True, slots=True)
class RestoreResult:
    pet_id: PetId
    brain_id: uuid.UUID
    device_id: uuid.UUID
    snapshot_generation: int
    restored_payload: dict[str, Any]
    tombstones_applied: int
