"""Domain entities for cloud suggestions and sleep learning (CLD-30, CLD-31, H1-H8).

Domain rules: stdlib and own/other domain only. No pydantic or frameworks.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from xuna_backend.modules.identity.domain.pet_id import PetId

_TARGET_PATTERN = re.compile(r"^[a-z0-9_.]+$")


class SuggestionKind(StrEnum):
    LEARNER_PARAMETER = "learner_parameter"
    HABIT_PRIOR = "habit_prior"
    CONTENT_PACK = "content_pack"


class SuggestionProvenance(StrEnum):
    CLOUD_INFERRED = "cloud_inferred"
    POPULATION = "population"


class SuggestionStatus(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVERTED = "reverted"


@dataclass(frozen=True, slots=True)
class CloudSuggestion:
    suggestion_id: uuid.UUID
    pet_id: PetId
    brain_id: uuid.UUID
    kind: SuggestionKind
    target: str
    delta: float
    bounds: dict[str, float]
    provenance: SuggestionProvenance
    expires_at: datetime
    created_at: datetime
    evidence_ref: str | None = None
    status: SuggestionStatus = SuggestionStatus.PENDING
    applied_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.applied_at is not None and self.applied_at.tzinfo is None:
            raise ValueError("applied_at must be timezone-aware")
        if not self.target or not _TARGET_PATTERN.match(self.target):
            raise ValueError(f"invalid suggestion target: {self.target!r}")
        if "min" not in self.bounds or "max" not in self.bounds:
            raise ValueError("bounds must contain 'min' and 'max'")
        if self.bounds["min"] > self.bounds["max"]:
            raise ValueError(
                f"bounds min ({self.bounds['min']}) cannot exceed max ({self.bounds['max']})"
            )

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def can_transition_to(self, new_status: SuggestionStatus) -> bool:
        match (self.status, new_status):
            case (SuggestionStatus.PENDING, SuggestionStatus.APPLIED):
                return True
            case (SuggestionStatus.PENDING, SuggestionStatus.REJECTED):
                return True
            case (SuggestionStatus.PENDING, SuggestionStatus.EXPIRED):
                return True
            case (SuggestionStatus.APPLIED, SuggestionStatus.REVERTED):
                return True
            case _:
                return False
