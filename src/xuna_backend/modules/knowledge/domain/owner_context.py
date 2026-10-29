"""Owner context domain value objects for knowledge-aware turns (KX-1, plan §6.6).

Domain rules: stdlib only.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

_VALID_SLOT_STATES = frozenset({"empty", "active", "resting"})


@dataclass(frozen=True, slots=True)
class DeviceSlotView:
    """From the app's BLE read; untrusted but bounded."""

    slot: int
    level: int
    state: str

    def __post_init__(self) -> None:
        if not 0 <= self.slot <= 4:
            raise ValueError("slot must be in 0..4")
        if not 0 <= self.level <= 3:
            raise ValueError("level must be in 0..3")
        if self.state not in _VALID_SLOT_STATES:
            raise ValueError("state must be empty, active, or resting")


@dataclass(frozen=True, slots=True)
class DomainBrief:
    domain_id: uuid.UUID
    name: str
    brief: str
    level: int | None
    safety_tier: str = "general"


@dataclass(frozen=True, slots=True)
class DocSnippet:
    """Document snippet (KX-5)."""

    title: str
    text: str


@dataclass(frozen=True, slots=True)
class OwnerContext:
    domain: DomainBrief | None = None
    conventions: tuple[str, ...] = ()
    cases: tuple[str, ...] = ()
    snippets: tuple[DocSnippet, ...] = ()
    domain_score: float = 0.0
    domain_explicit: bool = False

