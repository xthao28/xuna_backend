"""Domain entities for persistent host routing habits and defaults (Spatial Mesh §5.1.3-§5.1.4).

Stdlib only. Stores per-(owner, pet, context, host) decayed evidence and explicit
server-side scope defaults (separate from mobile UDC-10 phone-local defaults).
"""

import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from xuna_backend.modules.hosts.domain.entities import HostId

DEFAULT_HABIT_GAMMA = 0.85


class HostDefaultSource(StrEnum):
    CONFIRMED = "confirmed"
    SLASH_CMD = "slash_cmd"
    MIGRATED = "migrated"


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


@dataclass(frozen=True, slots=True)
class HostRoutingHabit:
    """Persistent exponential-decay routing habit record (`host_routing_habits`)."""

    habit_id: uuid.UUID
    owner_id: uuid.UUID
    pet_id: str
    context_key: str
    host_id: HostId
    created_at: datetime
    updated_at: datetime
    positive_score: float = 0.0
    negative_score: float = 0.0
    sample_count: int = 0
    decay_gamma: float = DEFAULT_HABIT_GAMMA
    last_reason_code: str | None = None

    def __post_init__(self) -> None:
        if _has_control_chars(self.pet_id):
            raise ValueError("pet_id must be 1..64 non-control characters")
        trimmed_pet = self.pet_id.strip(" ")
        if not 1 <= len(trimmed_pet) <= 64:
            raise ValueError("pet_id must be 1..64 non-control characters")
        object.__setattr__(self, "pet_id", trimmed_pet)

        if _has_control_chars(self.context_key):
            raise ValueError("context_key must be 1..128 non-control characters")
        trimmed_key = self.context_key.strip(" ")
        if not 1 <= len(trimmed_key) <= 128:
            raise ValueError("context_key must be 1..128 non-control characters")
        object.__setattr__(self, "context_key", trimmed_key)

        if (
            isinstance(self.positive_score, bool)
            or not isinstance(self.positive_score, int | float)
            or not math.isfinite(self.positive_score)
            or self.positive_score < 0.0
        ):
            raise ValueError("positive_score must be a finite float >= 0.0")
        object.__setattr__(self, "positive_score", float(self.positive_score))

        if (
            isinstance(self.negative_score, bool)
            or not isinstance(self.negative_score, int | float)
            or not math.isfinite(self.negative_score)
            or self.negative_score < 0.0
        ):
            raise ValueError("negative_score must be a finite float >= 0.0")
        object.__setattr__(self, "negative_score", float(self.negative_score))

        if type(self.sample_count) is not int or self.sample_count < 0:
            raise ValueError("sample_count must be an integer >= 0")

        if (
            isinstance(self.decay_gamma, bool)
            or not isinstance(self.decay_gamma, int | float)
            or not math.isfinite(self.decay_gamma)
            or not (0.0 < self.decay_gamma < 1.0)
        ):
            raise ValueError("decay_gamma must be finite in (0.0, 1.0)")
        object.__setattr__(self, "decay_gamma", float(self.decay_gamma))

        if self.last_reason_code is not None:
            if _has_control_chars(self.last_reason_code):
                raise ValueError("last_reason_code must be 1..64 non-control characters")
            trimmed_reason = self.last_reason_code.strip(" ")
            if not 1 <= len(trimmed_reason) <= 64:
                raise ValueError("last_reason_code must be 1..64 non-control characters")
            object.__setattr__(self, "last_reason_code", trimmed_reason)

        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("created_at and updated_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class HostDefaultRecord:
    """Durable server-side default host per `(owner_id, pet_id, scope_key)` (`host_defaults`)."""

    default_id: uuid.UUID
    owner_id: uuid.UUID
    pet_id: str
    scope_key: str
    host_id: HostId
    created_at: datetime
    updated_at: datetime
    source: HostDefaultSource = HostDefaultSource.CONFIRMED

    def __post_init__(self) -> None:
        if _has_control_chars(self.pet_id):
            raise ValueError("pet_id must be 1..64 non-control characters")
        trimmed_pet = self.pet_id.strip(" ")
        if not 1 <= len(trimmed_pet) <= 64:
            raise ValueError("pet_id must be 1..64 non-control characters")
        object.__setattr__(self, "pet_id", trimmed_pet)

        if _has_control_chars(self.scope_key):
            raise ValueError("scope_key must be 1..64 non-control characters")
        trimmed_scope = self.scope_key.strip(" ")
        if not 1 <= len(trimmed_scope) <= 64:
            raise ValueError("scope_key must be 1..64 non-control characters")
        object.__setattr__(self, "scope_key", trimmed_scope)

        object.__setattr__(self, "source", HostDefaultSource(self.source))

        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("created_at and updated_at must be timezone-aware")
