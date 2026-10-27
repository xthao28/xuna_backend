"""Domain entities for Knowledge Domains (M06 KD-3, K-D1..K-D7).

Domain rules: stdlib and own/other domain only. No pydantic or frameworks.
"""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from xuna_backend.modules.identity.domain.pet_id import PetId


class DomainStatus(StrEnum):
    PENDING_DEVICE = "pending_device"
    ACTIVE = "active"
    RESTING = "resting"
    DELETED = "deleted"


class DomainOrigin(StrEnum):
    OWNER = "owner"
    SUGGESTION = "suggestion"


class SuggestionKind(StrEnum):
    NEW_DOMAIN = "new_domain"
    ADD_TO_DOMAIN = "add_to_domain"


class SuggestionStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


class RelationKind(StrEnum):
    NEW = "new"
    REFINES = "refines"
    CONTRADICTS = "contradicts"


def compute_fact_key(text: str) -> int:
    """Computes a deterministic 32-bit unsigned non-zero fact key from text.

    Uses SHA-256 of NFC-normalized text, returning a uint32 in range [1, 0xFFFFFFFF].
    """
    normalized = unicodedata.normalize("NFC", text.strip())
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    key = int.from_bytes(digest[:4], byteorder="little")
    return key if key != 0 else 1


@dataclass(frozen=True, slots=True)
class KnowledgeDomain:
    domain_id: uuid.UUID
    owner_id: uuid.UUID
    pet_id: PetId
    name: str
    description: str
    status: DomainStatus
    origin: DomainOrigin
    created_at: datetime
    updated_at: datetime
    device_slot: int | None = None
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.name or len(self.name.strip()) == 0 or len(self.name) > 40:
            raise ValueError("name must be 1 to 40 characters")
        if len(self.description) > 160:
            raise ValueError("description must be at most 160 characters")
        if self.device_slot is not None and not 0 <= self.device_slot <= 4:
            raise ValueError("device_slot must be in 0..4")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")


@dataclass(frozen=True, slots=True)
class KnowledgeSuggestion:
    suggestion_id: uuid.UUID
    owner_id: uuid.UUID
    pet_id: PetId
    kind: SuggestionKind
    proposed_name: str
    proposed_description: str
    member_candidate_ids: list[uuid.UUID]
    status: SuggestionStatus
    created_at: datetime
    target_domain_id: uuid.UUID | None = None
    decided_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.proposed_name or len(self.proposed_name) > 40:
            raise ValueError("proposed_name must be 1 to 40 characters")
        if len(self.proposed_description) > 160:
            raise ValueError("proposed_description must be at most 160 characters")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.decided_at is not None and self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class KnowledgeSummary:
    domain_id: uuid.UUID
    text: str
    source_candidate_ids: list[uuid.UUID]
    generated_at: datetime
    stale: bool = False

    def __post_init__(self) -> None:
        if len(self.text) > 400:
            raise ValueError("text must be at most 400 characters")
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
