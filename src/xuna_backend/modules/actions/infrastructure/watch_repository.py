"""Durable and in-memory repositories for `JobWatch` and `WakeInboxItem` (AW-M2)."""

import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.actions.domain.ports import JobWatchRepository, WakeInboxRepository
from xuna_backend.modules.actions.domain.watch import (
    ContinuationMode,
    JobWatch,
    WakeInboxItem,
    WakeInboxKind,
    WakeSuggestedAction,
    WatchCompositeSignal,
    WatchGestureCue,
    WatchMode,
    WatchStatus,
    WatchTarget,
)
from xuna_backend.modules.actions.infrastructure.models import (
    JobWatchModel,
    WakeInboxItemModel,
)
from xuna_backend.modules.hosts.domain.entities import HostId


def _watch_row_to_entity(row: JobWatchModel) -> JobWatch:
    return JobWatch(
        watch_id=row.watch_id,
        job_id=row.job_id,
        owner_id=row.owner_id,
        host_id=HostId(row.host_id),
        mode=WatchMode(row.mode),
        signals=tuple(WatchCompositeSignal(s) for s in (row.signals or [])),
        target=WatchTarget.from_dict(dict(row.target_spec or {})),
        done_criteria=row.done_criteria,
        continuation_mode=ContinuationMode(row.continuation_mode),
        quiescence_seconds=row.quiescence_seconds,
        max_wait_seconds=row.max_wait_seconds,
        max_wake_checks=row.max_wake_checks,
        wake_checks_used=row.wake_checks_used,
        status=WatchStatus(row.status),
        last_event_kind=row.last_event_kind,
        expires_at=row.expires_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _inbox_row_to_entity(row: WakeInboxItemModel) -> WakeInboxItem:
    return WakeInboxItem(
        item_id=row.item_id,
        owner_id=row.owner_id,
        pet_id=row.pet_id,
        job_id=row.job_id,
        host_id=HostId(row.host_id) if row.host_id is not None else None,
        kind=WakeInboxKind(row.kind),
        title=row.title,
        summary=row.summary,
        suggested_actions=tuple(WakeSuggestedAction(a) for a in (row.suggested_actions or [])),
        gesture_cue=WatchGestureCue(row.gesture_cue) if row.gesture_cue else None,
        read_at=row.read_at,
        dismissed_at=row.dismissed_at,
        created_at=row.created_at,
    )


class InMemoryJobWatchRepository(JobWatchRepository):
    """In-memory implementation of `JobWatchRepository` for unit tests."""

    def __init__(self) -> None:
        self._watches: dict[uuid.UUID, JobWatch] = {}

    async def save(self, watch: JobWatch) -> None:
        self._watches[watch.watch_id] = watch

    async def get(self, watch_id: uuid.UUID) -> JobWatch | None:
        return self._watches.get(watch_id)

    async def get_active_for_job(self, job_id: uuid.UUID) -> JobWatch | None:
        matches = [
            w
            for w in self._watches.values()
            if w.job_id == job_id and w.status == WatchStatus.ACTIVE
        ]
        if not matches:
            return None
        matches.sort(key=lambda w: w.created_at, reverse=True)
        return matches[0]

    async def get_latest_for_job(self, job_id: uuid.UUID) -> JobWatch | None:
        matches = [w for w in self._watches.values() if w.job_id == job_id]
        if not matches:
            return None
        matches.sort(key=lambda w: w.created_at, reverse=True)
        return matches[0]

    async def list_active_for_host(
        self, host_id: HostId, *, owner_id: uuid.UUID | None = None
    ) -> list[JobWatch]:
        matches = [
            w
            for w in self._watches.values()
            if w.host_id == host_id
            and w.status == WatchStatus.ACTIVE
            and (owner_id is None or w.owner_id == owner_id)
        ]
        matches.sort(key=lambda w: w.created_at)
        return matches

    async def list_active_watches(self, *, limit: int = 200) -> list[JobWatch]:
        matches = [w for w in self._watches.values() if w.status == WatchStatus.ACTIVE]
        matches.sort(key=lambda w: w.created_at)
        return matches[:limit]

    async def list_expired_active(self, now: datetime, *, limit: int = 200) -> list[JobWatch]:
        matches = [
            w
            for w in self._watches.values()
            if w.status == WatchStatus.ACTIVE and w.expires_at <= now
        ]
        matches.sort(key=lambda w: w.expires_at)
        return matches[:limit]

    async def cancel_active_for_job(
        self, job_id: uuid.UUID, *, reason: str, now: datetime
    ) -> list[JobWatch]:
        cancelled: list[JobWatch] = []
        for watch_id, w in list(self._watches.items()):
            if w.job_id == job_id and w.status == WatchStatus.ACTIVE:
                updated = replace(
                    w, status=WatchStatus.CANCELLED, last_event_kind=reason[:24], updated_at=now
                )
                self._watches[watch_id] = updated
                cancelled.append(updated)
        return cancelled

    async def cancel_active_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        host_id: HostId | None = None,
        reason: str,
        now: datetime,
    ) -> list[JobWatch]:
        cancelled: list[JobWatch] = []
        for watch_id, w in list(self._watches.items()):
            if (
                w.owner_id == owner_id
                and w.status == WatchStatus.ACTIVE
                and (host_id is None or w.host_id == host_id)
            ):
                updated = replace(
                    w, status=WatchStatus.CANCELLED, last_event_kind=reason[:24], updated_at=now
                )
                self._watches[watch_id] = updated
                cancelled.append(updated)
        return cancelled


class SqlJobWatchRepository(JobWatchRepository):
    """SQLAlchemy / PostgreSQL implementation of `JobWatchRepository`."""

    def __init__(self, sessions: Callable[[], AsyncSession]) -> None:
        self._sessions = sessions

    async def save(self, watch: JobWatch) -> None:
        async with self._sessions() as s:
            existing = await s.get(JobWatchModel, watch.watch_id)
            signals_data = [sig.value for sig in watch.signals]
            target_data = dict(watch.target.to_wire_dict())
            if existing is not None:
                existing.mode = watch.mode.value
                existing.signals = signals_data
                existing.target_spec = target_data
                existing.done_criteria = watch.done_criteria[:240]
                existing.continuation_mode = watch.continuation_mode.value
                existing.quiescence_seconds = watch.quiescence_seconds
                existing.max_wait_seconds = watch.max_wait_seconds
                existing.max_wake_checks = watch.max_wake_checks
                existing.wake_checks_used = watch.wake_checks_used
                existing.status = watch.status.value
                existing.last_event_kind = watch.last_event_kind
                existing.expires_at = watch.expires_at
                existing.updated_at = watch.updated_at
            else:
                s.add(
                    JobWatchModel(
                        watch_id=watch.watch_id,
                        job_id=watch.job_id,
                        owner_id=watch.owner_id,
                        host_id=watch.host_id.value,
                        mode=watch.mode.value,
                        signals=signals_data,
                        target_spec=target_data,
                        done_criteria=watch.done_criteria[:240],
                        continuation_mode=watch.continuation_mode.value,
                        quiescence_seconds=watch.quiescence_seconds,
                        max_wait_seconds=watch.max_wait_seconds,
                        max_wake_checks=watch.max_wake_checks,
                        wake_checks_used=watch.wake_checks_used,
                        status=watch.status.value,
                        last_event_kind=watch.last_event_kind,
                        expires_at=watch.expires_at,
                        created_at=watch.created_at,
                        updated_at=watch.updated_at,
                    )
                )
            await s.commit()

    async def get(self, watch_id: uuid.UUID) -> JobWatch | None:
        async with self._sessions() as s:
            if not hasattr(s, "get"):
                return None
            row = await s.get(JobWatchModel, watch_id)
            return _watch_row_to_entity(row) if row is not None else None

    async def get_active_for_job(self, job_id: uuid.UUID) -> JobWatch | None:
        async with self._sessions() as s:
            if not hasattr(s, "execute"):
                return None
            stmt = (
                select(JobWatchModel)
                .where(
                    JobWatchModel.job_id == job_id,
                    JobWatchModel.status == WatchStatus.ACTIVE.value,
                )
                .order_by(JobWatchModel.created_at.desc())
                .limit(1)
            )
            res = await s.execute(stmt)
            if res is None:
                return None
            row = res.scalars().first()
            return _watch_row_to_entity(row) if row is not None else None

    async def get_latest_for_job(self, job_id: uuid.UUID) -> JobWatch | None:
        async with self._sessions() as s:
            if not hasattr(s, "execute"):
                return None
            stmt = (
                select(JobWatchModel)
                .where(JobWatchModel.job_id == job_id)
                .order_by(JobWatchModel.created_at.desc())
                .limit(1)
            )
            res = await s.execute(stmt)
            if res is None:
                return None
            row = res.scalars().first()
            return _watch_row_to_entity(row) if row is not None else None

    async def list_active_for_host(
        self, host_id: HostId, *, owner_id: uuid.UUID | None = None
    ) -> list[JobWatch]:
        async with self._sessions() as s:
            if not hasattr(s, "execute"):
                return []
            stmt = select(JobWatchModel).where(
                JobWatchModel.host_id == host_id.value,
                JobWatchModel.status == WatchStatus.ACTIVE.value,
            )
            if owner_id is not None:
                stmt = stmt.where(JobWatchModel.owner_id == owner_id)
            stmt = stmt.order_by(JobWatchModel.created_at.asc())
            res = await s.execute(stmt)
            if res is None:
                return []
            return [_watch_row_to_entity(r) for r in res.scalars().all()]

    async def list_active_watches(self, *, limit: int = 200) -> list[JobWatch]:
        async with self._sessions() as s:
            if not hasattr(s, "execute"):
                return []
            stmt = (
                select(JobWatchModel)
                .where(JobWatchModel.status == WatchStatus.ACTIVE.value)
                .order_by(JobWatchModel.created_at.asc())
                .limit(limit)
            )
            res = await s.execute(stmt)
            if res is None:
                return []
            return [_watch_row_to_entity(r) for r in res.scalars().all()]

    async def list_expired_active(self, now: datetime, *, limit: int = 200) -> list[JobWatch]:
        async with self._sessions() as s:
            if not hasattr(s, "execute"):
                return []
            stmt = (
                select(JobWatchModel)
                .where(
                    JobWatchModel.status == WatchStatus.ACTIVE.value,
                    JobWatchModel.expires_at <= now,
                )
                .order_by(JobWatchModel.expires_at.asc())
                .limit(limit)
            )
            res = await s.execute(stmt)
            if res is None:
                return []
            return [_watch_row_to_entity(r) for r in res.scalars().all()]

    async def cancel_active_for_job(
        self, job_id: uuid.UUID, *, reason: str, now: datetime
    ) -> list[JobWatch]:
        async with self._sessions() as s:
            if not hasattr(s, "execute"):
                return []
            stmt = select(JobWatchModel).where(
                JobWatchModel.job_id == job_id,
                JobWatchModel.status == WatchStatus.ACTIVE.value,
            )
            res = await s.execute(stmt)
            if res is None:
                return []
            rows = list(res.scalars().all())
            for row in rows:
                row.status = WatchStatus.CANCELLED.value
                row.last_event_kind = reason[:24]
                row.updated_at = now
            if rows:
                await s.commit()
            return [_watch_row_to_entity(r) for r in rows]

    async def cancel_active_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        host_id: HostId | None = None,
        reason: str,
        now: datetime,
    ) -> list[JobWatch]:
        async with self._sessions() as s:
            if not hasattr(s, "execute"):
                return []
            stmt = select(JobWatchModel).where(
                JobWatchModel.owner_id == owner_id,
                JobWatchModel.status == WatchStatus.ACTIVE.value,
            )
            if host_id is not None:
                stmt = stmt.where(JobWatchModel.host_id == host_id.value)
            res = await s.execute(stmt)
            if res is None:
                return []
            rows = list(res.scalars().all())
            for row in rows:
                row.status = WatchStatus.CANCELLED.value
                row.last_event_kind = reason[:24]
                row.updated_at = now
            if rows:
                await s.commit()
            return [_watch_row_to_entity(r) for r in rows]


class InMemoryWakeInboxRepository(WakeInboxRepository):
    """In-memory implementation of `WakeInboxRepository` for unit tests."""

    def __init__(self) -> None:
        self._items: dict[uuid.UUID, WakeInboxItem] = {}

    async def save(self, item: WakeInboxItem) -> None:
        self._items[item.item_id] = item

    async def get(self, item_id: uuid.UUID) -> WakeInboxItem | None:
        return self._items.get(item_id)

    async def list_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        unread_only: bool = False,
        include_dismissed: bool = False,
        limit: int = 50,
    ) -> list[WakeInboxItem]:
        matches = [
            item
            for item in self._items.values()
            if item.owner_id == owner_id
            and (include_dismissed or item.dismissed_at is None)
            and (not unread_only or item.read_at is None)
        ]
        matches.sort(key=lambda item: item.created_at, reverse=True)
        return matches[:limit]

    async def mark_read_or_dismiss(
        self,
        item_id: uuid.UUID,
        owner_id: uuid.UUID,
        *,
        now: datetime,
        dismiss: bool = False,
    ) -> WakeInboxItem | None:
        item = self._items.get(item_id)
        if item is None or item.owner_id != owner_id:
            return None
        updated = replace(
            item,
            read_at=item.read_at or now,
            dismissed_at=now if dismiss else item.dismissed_at,
        )
        self._items[item_id] = updated
        return updated


class SqlWakeInboxRepository(WakeInboxRepository):
    """SQLAlchemy / PostgreSQL implementation of `WakeInboxRepository`."""

    def __init__(self, sessions: Callable[[], AsyncSession]) -> None:
        self._sessions = sessions

    async def save(self, item: WakeInboxItem) -> None:
        async with self._sessions() as s:
            existing = await s.get(WakeInboxItemModel, item.item_id)
            actions_data = [a.value for a in item.suggested_actions]
            host_uuid = item.host_id.value if item.host_id is not None else None
            cue_val = item.gesture_cue.value if item.gesture_cue is not None else None
            if existing is not None:
                existing.title = item.title[:120]
                existing.summary = item.summary
                existing.suggested_actions = actions_data
                existing.gesture_cue = cue_val
                existing.read_at = item.read_at
                existing.dismissed_at = item.dismissed_at
            else:
                s.add(
                    WakeInboxItemModel(
                        item_id=item.item_id,
                        owner_id=item.owner_id,
                        pet_id=item.pet_id[:64],
                        job_id=item.job_id,
                        host_id=host_uuid,
                        kind=item.kind.value,
                        title=item.title[:120],
                        summary=item.summary,
                        suggested_actions=actions_data,
                        gesture_cue=cue_val,
                        read_at=item.read_at,
                        dismissed_at=item.dismissed_at,
                        created_at=item.created_at,
                    )
                )
            await s.commit()

    async def get(self, item_id: uuid.UUID) -> WakeInboxItem | None:
        async with self._sessions() as s:
            if not hasattr(s, "get"):
                return None
            row = await s.get(WakeInboxItemModel, item_id)
            return _inbox_row_to_entity(row) if row is not None else None

    async def list_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        unread_only: bool = False,
        include_dismissed: bool = False,
        limit: int = 50,
    ) -> list[WakeInboxItem]:
        async with self._sessions() as s:
            if not hasattr(s, "execute"):
                return []
            stmt = select(WakeInboxItemModel).where(WakeInboxItemModel.owner_id == owner_id)
            if not include_dismissed:
                stmt = stmt.where(WakeInboxItemModel.dismissed_at.is_(None))
            if unread_only:
                stmt = stmt.where(WakeInboxItemModel.read_at.is_(None))
            stmt = stmt.order_by(WakeInboxItemModel.created_at.desc()).limit(limit)
            res = await s.execute(stmt)
            if res is None:
                return []
            return [_inbox_row_to_entity(r) for r in res.scalars().all()]

    async def mark_read_or_dismiss(
        self,
        item_id: uuid.UUID,
        owner_id: uuid.UUID,
        *,
        now: datetime,
        dismiss: bool = False,
    ) -> WakeInboxItem | None:
        async with self._sessions() as s:
            if not hasattr(s, "execute"):
                return None
            stmt = select(WakeInboxItemModel).where(
                WakeInboxItemModel.item_id == item_id,
                WakeInboxItemModel.owner_id == owner_id,
            )
            res = await s.execute(stmt)
            if res is None:
                return None
            row = res.scalars().first()
            if row is None:
                return None
            if row.read_at is None:
                row.read_at = now
            if dismiss:
                row.dismissed_at = now
            await s.commit()
            return _inbox_row_to_entity(row)
