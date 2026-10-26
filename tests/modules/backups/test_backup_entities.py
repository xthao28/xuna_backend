"""Unit tests for backup and tombstone domain entities (CLD-20)."""

import uuid
from datetime import UTC, datetime

import pytest

from xuna_backend.modules.backups.domain.entities import (
    BackupManifest,
    DeletionTombstone,
    EncryptedData,
)
from xuna_backend.modules.identity.domain.pet_id import PetId

PET_ID = PetId("d945c77026fd4f4a8a899bbff4aeed86")


def test_encrypted_data_validates_fields() -> None:
    data = EncryptedData(
        algorithm="aes-256-gcm",
        key_id="k1",
        nonce="bm9uY2U=",
        ciphertext="Y2lwaGVy",
        tag="dGFn",
    )
    assert data.algorithm == "aes-256-gcm"

    with pytest.raises(ValueError, match="non-empty"):
        EncryptedData(algorithm="", key_id="k1", nonce="n", ciphertext="c", tag="t")


def test_backup_manifest_validates_invariants() -> None:
    enc = EncryptedData(
        algorithm="aes-256-gcm",
        key_id="k1",
        nonce="bm9uY2U=",
        ciphertext="Y2lwaGVy",
        tag="dGFn",
    )
    now = datetime.now(UTC)
    manifest = BackupManifest(
        backup_id=uuid.uuid4(),
        pet_id=PET_ID,
        owner_id=uuid.uuid4(),
        brain_id=uuid.uuid4(),
        snapshot_generation=42,
        schema_version=1,
        payload_sha256="a" * 64,
        encrypted_data=enc,
        created_at=now,
    )
    assert manifest.snapshot_generation == 42

    with pytest.raises(ValueError, match="timezone-aware"):
        BackupManifest(
            backup_id=uuid.uuid4(),
            pet_id=PET_ID,
            owner_id=uuid.uuid4(),
            brain_id=uuid.uuid4(),
            snapshot_generation=42,
            schema_version=1,
            payload_sha256="a" * 64,
            encrypted_data=enc,
            created_at=datetime.now(),  # naive
        )

    with pytest.raises(ValueError, match="64-char hex string"):
        BackupManifest(
            backup_id=uuid.uuid4(),
            pet_id=PET_ID,
            owner_id=uuid.uuid4(),
            brain_id=uuid.uuid4(),
            snapshot_generation=42,
            schema_version=1,
            payload_sha256="too-short",
            encrypted_data=enc,
            created_at=now,
        )


def test_deletion_tombstone_validates_invariants() -> None:
    now = datetime.now(UTC)
    tombstone = DeletionTombstone(
        tombstone_id=uuid.uuid4(),
        pet_id=PET_ID,
        owner_id=uuid.uuid4(),
        item_type="candidate",
        item_id="cand-123",
        deleted_at=now,
        reason="deleted_by_owner",
    )
    assert tombstone.item_type == "candidate"

    with pytest.raises(ValueError, match="non-empty"):
        DeletionTombstone(
            tombstone_id=uuid.uuid4(),
            pet_id=PET_ID,
            owner_id=uuid.uuid4(),
            item_type="",
            item_id="cand-123",
            deleted_at=now,
        )
