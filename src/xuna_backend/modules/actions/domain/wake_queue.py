"""Pure-stdlib Wake-and-Run (WoL) deferred resume queue (`TTL = 180s`, max 3/owner).

Tracks deferred steps when an offline target has a configured `wol_mac` and bounds
active entries per owner (`wake_queue_full`). Never executes `R2+` steps silently.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from xuna_backend.modules.hosts.domain.entities import HostId, RiskTier

WAKE_QUEUE_TTL = timedelta(seconds=180)
MAX_DEFERRED_WAKE_STEPS_PER_OWNER = 3


class WakeQueueFullError(ValueError):
    """Raised when an owner already holds `MAX_DEFERRED_WAKE_STEPS_PER_OWNER` active steps."""

    def __init__(self) -> None:
        super().__init__("wake_queue_full")


@dataclass(frozen=True, slots=True)
class DeferredWakeStep:
    """An action or autonomous job step deferred until its target host sends `xuna/ready`."""

    step_id: uuid.UUID
    owner_id: uuid.UUID
    pet_id: str
    host_id: HostId
    host_alias: str
    tool: str
    arguments: dict[str, object]
    risk: RiskTier
    created_at: datetime
    expires_at: datetime
    job_id: uuid.UUID | None = None
    approval_id: uuid.UUID | None = None
    mission_id: str | None = None
    resource_id: str | None = None
    family: str = "app"
    summary: str = ""
    client_caps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("DeferredWakeStep timestamps must be timezone-aware")
        if self.expires_at <= self.created_at:
            raise ValueError("DeferredWakeStep expires_at must be after created_at")

    def is_expired(self, now: datetime) -> bool:
        """Returns `True` once `now` reaches or passes `expires_at` (`180s` TTL)."""
        return now >= self.expires_at


class InMemoryWakeQueueStore:
    """Process-local bounded queue of deferred WoL steps keyed by `owner_id`."""

    def __init__(
        self,
        *,
        max_per_owner: int = MAX_DEFERRED_WAKE_STEPS_PER_OWNER,
        ttl: timedelta = WAKE_QUEUE_TTL,
    ) -> None:
        self._max_per_owner = max_per_owner
        self._ttl = ttl
        self._by_owner: dict[uuid.UUID, list[DeferredWakeStep]] = {}

    @property
    def ttl(self) -> timedelta:
        return self._ttl

    def purge_expired(self, now: datetime, *, owner_id: uuid.UUID | None = None) -> int:
        """Removes expired steps and returns the count purged."""
        removed = 0
        owners = [owner_id] if owner_id is not None else list(self._by_owner.keys())
        for oid in owners:
            bucket = self._by_owner.get(oid)
            if not bucket:
                continue
            kept = [step for step in bucket if not step.is_expired(now)]
            removed += len(bucket) - len(kept)
            if kept:
                self._by_owner[oid] = kept
            else:
                self._by_owner.pop(oid, None)
        return removed

    def enqueue(self, step: DeferredWakeStep, now: datetime) -> DeferredWakeStep:
        """Enqueues `step` after purging expired entries; fails closed on full queue."""
        if step.is_expired(now):
            raise ValueError("wake_step_already_expired")
        self.purge_expired(now, owner_id=step.owner_id)
        bucket = self._by_owner.setdefault(step.owner_id, [])
        if len(bucket) >= self._max_per_owner:
            raise WakeQueueFullError()
        bucket.append(step)
        return step

    def list_active(
        self,
        owner_id: uuid.UUID,
        now: datetime,
        *,
        host_id: HostId | None = None,
    ) -> list[DeferredWakeStep]:
        """Returns active unexpired steps for `owner_id` (optionally filtered by `host_id`)."""
        self.purge_expired(now, owner_id=owner_id)
        bucket = self._by_owner.get(owner_id, ())
        if host_id is None:
            return list(bucket)
        return [step for step in bucket if step.host_id == host_id]

    def pop_ready_for_host(
        self,
        owner_id: uuid.UUID,
        host_id: HostId,
        now: datetime,
    ) -> list[DeferredWakeStep]:
        """Purges expired steps and pops all unexpired steps targeting `(owner_id, host_id)`."""
        self.purge_expired(now, owner_id=owner_id)
        bucket = self._by_owner.get(owner_id)
        if not bucket:
            return []
        ready: list[DeferredWakeStep] = []
        remaining: list[DeferredWakeStep] = []
        for step in bucket:
            if step.host_id == host_id:
                ready.append(step)
            else:
                remaining.append(step)
        if remaining:
            self._by_owner[owner_id] = remaining
        else:
            self._by_owner.pop(owner_id, None)
        return ready

    def clear_owner(self, owner_id: uuid.UUID, *, host_id: HostId | None = None) -> int:
        """Clears all deferred wake steps for `owner_id` (or only `host_id` on Tier-1 stop)."""
        bucket = self._by_owner.get(owner_id)
        if not bucket:
            return 0
        if host_id is None:
            self._by_owner.pop(owner_id, None)
            return len(bucket)
        kept = [step for step in bucket if step.host_id != host_id]
        removed = len(bucket) - len(kept)
        if kept:
            self._by_owner[owner_id] = kept
        else:
            self._by_owner.pop(owner_id, None)
        return removed
