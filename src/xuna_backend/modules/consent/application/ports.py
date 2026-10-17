import uuid
from typing import Protocol

from xuna_backend.modules.consent.domain.consent import ConsentState


class ConsentRepository(Protocol):
    async def current(self, owner_id: uuid.UUID) -> ConsentState:
        """Latest recorded state, or U0 when none exists."""
        ...

    async def append(self, owner_id: uuid.UUID, state: ConsentState) -> bool:
        """Insert-only; False when `state.version` already exists (race)."""
        ...
