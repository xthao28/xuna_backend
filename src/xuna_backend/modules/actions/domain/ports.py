"""Domain ports for `AutonomousJob` (Wave D).

Pure domain layer: stdlib-only, framework-free.
"""

import uuid
from typing import Protocol

from xuna_backend.modules.actions.domain.job import AutonomousJob, AutonomousJobState


class AutonomousJobRepository(Protocol):
    """Persistence protocol for `AutonomousJob` entities."""

    async def save(self, job: AutonomousJob) -> None:
        """Saves or updates an autonomous job."""
        ...

    async def get(self, job_id: uuid.UUID) -> AutonomousJob | None:
        """Retrieves a job by its unique identifier."""
        ...

    async def get_active_for_owner(self, owner_id: uuid.UUID) -> AutonomousJob | None:
        """Retrieves the active job for the owner, if any."""
        ...

    async def list_for_owner(
        self,
        owner_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        states: tuple[AutonomousJobState, ...] | None = None,
    ) -> list[AutonomousJob]:
        """Lists jobs belonging to an owner, optionally filtered by states."""
        ...

    async def count_for_owner(
        self,
        owner_id: uuid.UUID,
        states: tuple[AutonomousJobState, ...] | None = None,
    ) -> int:
        """Counts jobs belonging to an owner, optionally filtered by states."""
        ...
