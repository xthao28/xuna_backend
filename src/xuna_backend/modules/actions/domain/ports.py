"""Domain ports for `AutonomousJob` (Wave D).

Pure domain layer: stdlib-only, framework-free.
"""

import uuid
from datetime import datetime
from typing import Protocol

from xuna_backend.modules.actions.domain.job import AutonomousJob, AutonomousJobState
from xuna_backend.modules.actions.domain.watch import JobWatch, WakeInboxItem
from xuna_backend.modules.hosts.domain.entities import HostId


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


class JobWatchRepository(Protocol):
    """Persistence protocol for durable `JobWatch` records (`job_watches`)."""

    async def save(self, watch: JobWatch) -> None: ...

    async def get(self, watch_id: uuid.UUID) -> JobWatch | None: ...

    async def get_active_for_job(self, job_id: uuid.UUID) -> JobWatch | None: ...

    async def get_latest_for_job(self, job_id: uuid.UUID) -> JobWatch | None: ...

    async def list_active_for_host(
        self, host_id: HostId, *, owner_id: uuid.UUID | None = None
    ) -> list[JobWatch]: ...

    async def list_active_watches(self, *, limit: int = 200) -> list[JobWatch]: ...

    async def list_expired_active(self, now: datetime, *, limit: int = 200) -> list[JobWatch]: ...

    async def cancel_active_for_job(
        self, job_id: uuid.UUID, *, reason: str, now: datetime
    ) -> list[JobWatch]: ...

    async def cancel_active_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        host_id: HostId | None = None,
        reason: str,
        now: datetime,
    ) -> list[JobWatch]: ...


class WakeInboxRepository(Protocol):
    """Persistence protocol for `WakeInboxItem` records (`wake_inbox_items`)."""

    async def save(self, item: WakeInboxItem) -> None: ...

    async def get(self, item_id: uuid.UUID) -> WakeInboxItem | None: ...

    async def list_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        unread_only: bool = False,
        include_dismissed: bool = False,
        limit: int = 50,
    ) -> list[WakeInboxItem]: ...

    async def mark_read_or_dismiss(
        self,
        item_id: uuid.UUID,
        owner_id: uuid.UUID,
        *,
        now: datetime,
        dismiss: bool = False,
    ) -> WakeInboxItem | None: ...
