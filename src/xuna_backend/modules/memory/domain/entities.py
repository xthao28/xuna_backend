"""Domain entities for projections and memory candidates (CLD-14, D23).

Domain rules: stdlib and own/other domain only. No pydantic or frameworks.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from xuna_backend.modules.identity.domain.pet_id import PetId


class CandidateStatus(StrEnum):
    AWAITING_OWNER_CONFIRMATION = "awaiting_owner_confirmation"
    AWAITING_DEVICE_COMMIT = "awaiting_device_commit"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    DELETED = "deleted"


class CandidateType(StrEnum):
    STATED_PREFERENCE = "stated_preference"
    FACT = "fact"
    KNOWLEDGE = "knowledge"


@dataclass(frozen=True, slots=True)
class Projection:
    projection_id: uuid.UUID
    pet_id: PetId
    owner_id: uuid.UUID
    snapshot_generation: int
    schema_version: int
    payload_sha256: str
    payload: dict[str, Any]
    device_attested: bool
    created_at: datetime

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if not 0 <= self.snapshot_generation <= 0xFFFFFFFF:
            raise ValueError("snapshot_generation must be a u32")
        if len(self.payload_sha256) != 64:
            raise ValueError("payload_sha256 must be a 64-char hex string")


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    candidate_id: uuid.UUID
    owner_id: uuid.UUID
    pet_id: PetId
    source: str
    candidate_type: CandidateType
    content: dict[str, Any]
    status: CandidateStatus
    created_at: datetime
    updated_at: datetime
    confirmed_at: datetime | None = None
    committed_generation: int | None = None
    domain_id: uuid.UUID | None = None
    relation: str | None = None
    related_candidate_id: uuid.UUID | None = None
    fact_key: int | None = None
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        if self.confirmed_at is not None and self.confirmed_at.tzinfo is None:
            raise ValueError("confirmed_at must be timezone-aware")
        if (
            self.committed_generation is not None
            and not 0 <= self.committed_generation <= 0xFFFFFFFF
        ):
            raise ValueError("committed_generation must be a u32")
        if self.fact_key is not None and not 0 <= self.fact_key <= 0xFFFFFFFF:
            raise ValueError("fact_key must be a u32")
