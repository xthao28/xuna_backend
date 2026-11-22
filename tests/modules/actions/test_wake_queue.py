"""Unit tests for `actions.domain.wake_queue` (180s TTL, max 3/owner, Tier-1 clear)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from xuna_backend.modules.actions.domain.wake_queue import (
    MAX_DEFERRED_WAKE_STEPS_PER_OWNER,
    WAKE_QUEUE_TTL,
    DeferredWakeStep,
    InMemoryWakeQueueStore,
    WakeQueueFullError,
)
from xuna_backend.modules.hosts.domain.entities import HostId, RiskTier

T0 = datetime(2026, 11, 20, 14, 0, 0, tzinfo=UTC)


def _step(owner_id: uuid.UUID, host_id: HostId, idx: int, now: datetime = T0) -> DeferredWakeStep:
    return DeferredWakeStep(
        step_id=uuid.uuid4(),
        owner_id=owner_id,
        pet_id="pet-1",
        host_id=host_id,
        host_alias="WinSleep",
        tool="app.launch",
        arguments={"id": f"app_{idx}"},
        risk=RiskTier.R2,
        created_at=now,
        expires_at=now + WAKE_QUEUE_TTL,
        summary=f"Launch app_{idx}",
    )


def test_wake_queue_enforces_max_3_per_owner_and_purges_180s_ttl() -> None:
    owner_a = uuid.uuid4()
    owner_b = uuid.uuid4()
    host_a = HostId.generate()
    host_b = HostId.generate()
    store = InMemoryWakeQueueStore()

    for i in range(MAX_DEFERRED_WAKE_STEPS_PER_OWNER):
        store.enqueue(_step(owner_a, host_a, i, T0), T0)

    # 4th step fails closed for owner_a while owner_b has an independent quota
    with pytest.raises(WakeQueueFullError, match="wake_queue_full"):
        store.enqueue(_step(owner_a, host_a, 99, T0), T0)
    store.enqueue(_step(owner_b, host_b, 0, T0), T0)

    # At T0 + 179s steps are still active and pop on host ready
    assert len(store.list_active(owner_a, T0 + timedelta(seconds=179))) == 3

    # At T0 + 180s all steps expire and are purged cleanly
    assert store.pop_ready_for_host(owner_a, host_a, T0 + WAKE_QUEUE_TTL) == []
    assert store.list_active(owner_a, T0 + WAKE_QUEUE_TTL) == []

    # After expiry, owner_a can enqueue again
    store.enqueue(_step(owner_a, host_a, 4, T0 + WAKE_QUEUE_TTL), T0 + WAKE_QUEUE_TTL)
    popped = store.pop_ready_for_host(owner_a, host_a, T0 + timedelta(seconds=181))
    assert len(popped) == 1
    assert popped[0].arguments == {"id": "app_4"}


def test_wake_queue_clear_owner_supports_tier1_host_isolation() -> None:
    owner = uuid.uuid4()
    host_1 = HostId.generate()
    host_2 = HostId.generate()
    store = InMemoryWakeQueueStore()
    store.enqueue(_step(owner, host_1, 1, T0), T0)
    store.enqueue(_step(owner, host_2, 2, T0), T0)

    removed_one = store.clear_owner(owner, host_id=host_1)
    assert removed_one == 1
    remaining = store.list_active(owner, T0)
    assert len(remaining) == 1
    assert remaining[0].host_id == host_2

    assert store.clear_owner(owner) == 1
    assert store.list_active(owner, T0) == []
