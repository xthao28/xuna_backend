"""Tests for memory domain entities."""

import uuid
from datetime import UTC, datetime

import pytest

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
    Projection,
)

PET = PetId("d945c77026fd4f4a8a899bbff4aeed86")
OWNER = uuid.uuid4()
NOW = datetime.now(UTC)


def test_projection_entity_validation() -> None:
    p = Projection(
        projection_id=uuid.uuid4(),
        pet_id=PET,
        owner_id=OWNER,
        snapshot_generation=42,
        schema_version=1,
        payload_sha256="a" * 64,
        payload={"dummy": True},
        device_attested=True,
        created_at=NOW,
    )
    assert p.snapshot_generation == 42
    assert p.device_attested is True

    # Bad sha256 length
    with pytest.raises(ValueError, match="64-char hex"):
        Projection(
            projection_id=uuid.uuid4(),
            pet_id=PET,
            owner_id=OWNER,
            snapshot_generation=42,
            schema_version=1,
            payload_sha256="short",
            payload={},
            device_attested=False,
            created_at=NOW,
        )

    # Negative generation
    with pytest.raises(ValueError, match="u32"):
        Projection(
            projection_id=uuid.uuid4(),
            pet_id=PET,
            owner_id=OWNER,
            snapshot_generation=-1,
            schema_version=1,
            payload_sha256="a" * 64,
            payload={},
            device_attested=False,
            created_at=NOW,
        )


def test_memory_candidate_validation() -> None:
    c = MemoryCandidate(
        candidate_id=uuid.uuid4(),
        owner_id=OWNER,
        pet_id=PET,
        source="conversation",
        candidate_type=CandidateType.STATED_PREFERENCE,
        content={"action": "happy", "polarity": "like"},
        status=CandidateStatus.AWAITING_OWNER_CONFIRMATION,
        created_at=NOW,
        updated_at=NOW,
    )
    assert c.status == CandidateStatus.AWAITING_OWNER_CONFIRMATION

    # Non-timezone aware
    with pytest.raises(ValueError, match="timezone-aware"):
        MemoryCandidate(
            candidate_id=uuid.uuid4(),
            owner_id=OWNER,
            pet_id=PET,
            source="conversation",
            candidate_type=CandidateType.FACT,
            content={},
            status=CandidateStatus.CONFIRMED,
            created_at=datetime.now(),  # naive
            updated_at=NOW,
        )
