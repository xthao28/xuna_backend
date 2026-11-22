"""PostgreSQL and in-memory repositories for `HostRoutingHabit` & `HostDefaultRecord`."""

import uuid

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.actions.domain.habits import (
    HostDefaultRecord,
    HostDefaultSource,
    HostRoutingHabit,
)
from xuna_backend.modules.actions.infrastructure.models import (
    HostDefaultModel,
    HostRoutingHabitModel,
)
from xuna_backend.modules.hosts.domain.entities import HostId


def _to_habit(row: HostRoutingHabitModel) -> HostRoutingHabit:
    return HostRoutingHabit(
        habit_id=row.habit_id,
        owner_id=row.owner_id,
        pet_id=row.pet_id,
        context_key=row.context_key,
        host_id=HostId(row.host_id),
        positive_score=row.positive_score,
        negative_score=row.negative_score,
        sample_count=row.sample_count,
        decay_gamma=row.decay_gamma,
        last_reason_code=row.last_reason_code,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_default(row: HostDefaultModel) -> HostDefaultRecord:
    return HostDefaultRecord(
        default_id=row.default_id,
        owner_id=row.owner_id,
        pet_id=row.pet_id,
        scope_key=row.scope_key,
        host_id=HostId(row.host_id),
        source=HostDefaultSource(row.source),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SqlHabitRepository:
    """AsyncSession-scoped repository so habit mutations & deletes commit atomically with audit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_habit(
        self,
        owner_id: uuid.UUID,
        pet_id: str,
        context_key: str,
        host_id: HostId,
    ) -> HostRoutingHabit | None:
        stmt = select(HostRoutingHabitModel).where(
            HostRoutingHabitModel.owner_id == owner_id,
            HostRoutingHabitModel.pet_id == pet_id,
            HostRoutingHabitModel.context_key == context_key,
            HostRoutingHabitModel.host_id == host_id.value,
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return _to_habit(row) if row is not None else None

    async def get_habit_by_id(
        self, owner_id: uuid.UUID, habit_id: uuid.UUID
    ) -> HostRoutingHabit | None:
        stmt = select(HostRoutingHabitModel).where(
            HostRoutingHabitModel.habit_id == habit_id,
            HostRoutingHabitModel.owner_id == owner_id,
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return _to_habit(row) if row is not None else None

    async def list_habits(
        self,
        owner_id: uuid.UUID,
        *,
        pet_id: str | None = None,
        context_key: str | None = None,
    ) -> list[HostRoutingHabit]:
        conditions = [HostRoutingHabitModel.owner_id == owner_id]
        if pet_id is not None:
            conditions.append(HostRoutingHabitModel.pet_id == pet_id)
        if context_key is not None:
            conditions.append(HostRoutingHabitModel.context_key == context_key)
        stmt = (
            select(HostRoutingHabitModel)
            .where(*conditions)
            .order_by(HostRoutingHabitModel.updated_at.desc(), HostRoutingHabitModel.habit_id.asc())
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [_to_habit(row) for row in rows]

    async def upsert_habit(self, habit: HostRoutingHabit) -> HostRoutingHabit:
        bind = self._session.bind
        dialect_name = bind.dialect.name if bind is not None else "postgresql"
        values = {
            "habit_id": habit.habit_id,
            "owner_id": habit.owner_id,
            "pet_id": habit.pet_id,
            "context_key": habit.context_key,
            "host_id": habit.host_id.value,
            "positive_score": habit.positive_score,
            "negative_score": habit.negative_score,
            "sample_count": habit.sample_count,
            "decay_gamma": habit.decay_gamma,
            "last_reason_code": habit.last_reason_code,
            "created_at": habit.created_at,
            "updated_at": habit.updated_at,
        }
        update_set = {
            "positive_score": habit.positive_score,
            "negative_score": habit.negative_score,
            "sample_count": habit.sample_count,
            "decay_gamma": habit.decay_gamma,
            "last_reason_code": habit.last_reason_code,
            "updated_at": habit.updated_at,
        }
        returning_cols = (
            HostRoutingHabitModel.habit_id,
            HostRoutingHabitModel.owner_id,
            HostRoutingHabitModel.pet_id,
            HostRoutingHabitModel.context_key,
            HostRoutingHabitModel.host_id,
            HostRoutingHabitModel.positive_score,
            HostRoutingHabitModel.negative_score,
            HostRoutingHabitModel.sample_count,
            HostRoutingHabitModel.decay_gamma,
            HostRoutingHabitModel.last_reason_code,
            HostRoutingHabitModel.created_at,
            HostRoutingHabitModel.updated_at,
        )
        if dialect_name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            stmt = (
                sqlite_insert(HostRoutingHabitModel)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=["owner_id", "pet_id", "context_key", "host_id"],
                    set_=update_set,
                )
                .returning(*returning_cols)
            )
            row = (await self._session.execute(stmt)).one()
        else:
            pg_stmt = (
                pg_insert(HostRoutingHabitModel)
                .values(**values)
                .on_conflict_do_update(
                    constraint="uq_host_routing_habits_key",
                    set_=update_set,
                )
                .returning(*returning_cols)
            )
            row = (await self._session.execute(pg_stmt)).one()
        return HostRoutingHabit(
            habit_id=row.habit_id,
            owner_id=row.owner_id,
            pet_id=row.pet_id,
            context_key=row.context_key,
            host_id=HostId(row.host_id),
            positive_score=row.positive_score,
            negative_score=row.negative_score,
            sample_count=row.sample_count,
            decay_gamma=row.decay_gamma,
            last_reason_code=row.last_reason_code,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def delete_habit(self, owner_id: uuid.UUID, habit_id: uuid.UUID) -> bool:
        res = await self._session.execute(
            delete(HostRoutingHabitModel).where(
                HostRoutingHabitModel.habit_id == habit_id,
                HostRoutingHabitModel.owner_id == owner_id,
            )
        )
        await self._session.flush()
        return (res.rowcount or 0) > 0

    async def delete_all_habits(self, owner_id: uuid.UUID, *, pet_id: str | None = None) -> int:
        conditions = [HostRoutingHabitModel.owner_id == owner_id]
        if pet_id is not None:
            conditions.append(HostRoutingHabitModel.pet_id == pet_id)
        res = await self._session.execute(delete(HostRoutingHabitModel).where(*conditions))
        await self._session.flush()
        return res.rowcount or 0

    async def get_default(
        self, owner_id: uuid.UUID, pet_id: str, scope_key: str
    ) -> HostDefaultRecord | None:
        stmt = select(HostDefaultModel).where(
            HostDefaultModel.owner_id == owner_id,
            HostDefaultModel.pet_id == pet_id,
            HostDefaultModel.scope_key == scope_key,
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return _to_default(row) if row is not None else None

    async def list_defaults(
        self, owner_id: uuid.UUID, *, pet_id: str | None = None
    ) -> list[HostDefaultRecord]:
        conditions = [HostDefaultModel.owner_id == owner_id]
        if pet_id is not None:
            conditions.append(HostDefaultModel.pet_id == pet_id)
        stmt = (
            select(HostDefaultModel)
            .where(*conditions)
            .order_by(HostDefaultModel.updated_at.desc(), HostDefaultModel.default_id.asc())
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [_to_default(row) for row in rows]

    async def upsert_default(self, default: HostDefaultRecord) -> HostDefaultRecord:
        bind = self._session.bind
        dialect_name = bind.dialect.name if bind is not None else "postgresql"
        values = {
            "default_id": default.default_id,
            "owner_id": default.owner_id,
            "pet_id": default.pet_id,
            "scope_key": default.scope_key,
            "host_id": default.host_id.value,
            "source": default.source.value,
            "created_at": default.created_at,
            "updated_at": default.updated_at,
        }
        update_set = {
            "host_id": default.host_id.value,
            "source": default.source.value,
            "updated_at": default.updated_at,
        }
        returning_cols = (
            HostDefaultModel.default_id,
            HostDefaultModel.owner_id,
            HostDefaultModel.pet_id,
            HostDefaultModel.scope_key,
            HostDefaultModel.host_id,
            HostDefaultModel.source,
            HostDefaultModel.created_at,
            HostDefaultModel.updated_at,
        )
        if dialect_name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            stmt = (
                sqlite_insert(HostDefaultModel)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=["owner_id", "pet_id", "scope_key"],
                    set_=update_set,
                )
                .returning(*returning_cols)
            )
            row = (await self._session.execute(stmt)).one()
        else:
            pg_stmt = (
                pg_insert(HostDefaultModel)
                .values(**values)
                .on_conflict_do_update(
                    constraint="uq_host_defaults_owner_pet_scope",
                    set_=update_set,
                )
                .returning(*returning_cols)
            )
            row = (await self._session.execute(pg_stmt)).one()
        return HostDefaultRecord(
            default_id=row.default_id,
            owner_id=row.owner_id,
            pet_id=row.pet_id,
            scope_key=row.scope_key,
            host_id=HostId(row.host_id),
            source=HostDefaultSource(row.source),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def delete_default(self, owner_id: uuid.UUID, pet_id: str, scope_key: str) -> bool:
        res = await self._session.execute(
            delete(HostDefaultModel).where(
                HostDefaultModel.owner_id == owner_id,
                HostDefaultModel.pet_id == pet_id,
                HostDefaultModel.scope_key == scope_key,
            )
        )
        await self._session.flush()
        return (res.rowcount or 0) > 0


class InMemoryHabitRepository:
    """In-memory `HabitRepository` for unit tests and DB-less harnesses."""

    def __init__(self) -> None:
        self._habits: dict[tuple[uuid.UUID, str, str, uuid.UUID], HostRoutingHabit] = {}
        self._defaults: dict[tuple[uuid.UUID, str, str], HostDefaultRecord] = {}

    async def get_habit(
        self,
        owner_id: uuid.UUID,
        pet_id: str,
        context_key: str,
        host_id: HostId,
    ) -> HostRoutingHabit | None:
        return self._habits.get((owner_id, pet_id, context_key, host_id.value))

    async def get_habit_by_id(
        self, owner_id: uuid.UUID, habit_id: uuid.UUID
    ) -> HostRoutingHabit | None:
        for item in self._habits.values():
            if item.owner_id == owner_id and item.habit_id == habit_id:
                return item
        return None

    async def list_habits(
        self,
        owner_id: uuid.UUID,
        *,
        pet_id: str | None = None,
        context_key: str | None = None,
    ) -> list[HostRoutingHabit]:
        rows = [
            item
            for item in self._habits.values()
            if item.owner_id == owner_id
            and (pet_id is None or item.pet_id == pet_id)
            and (context_key is None or item.context_key == context_key)
        ]
        rows.sort(key=lambda item: (item.updated_at, str(item.habit_id)), reverse=True)
        return rows

    async def upsert_habit(self, habit: HostRoutingHabit) -> HostRoutingHabit:
        key = (habit.owner_id, habit.pet_id, habit.context_key, habit.host_id.value)
        existing = self._habits.get(key)
        if existing is not None:
            stored = HostRoutingHabit(
                habit_id=existing.habit_id,
                owner_id=habit.owner_id,
                pet_id=habit.pet_id,
                context_key=habit.context_key,
                host_id=habit.host_id,
                positive_score=habit.positive_score,
                negative_score=habit.negative_score,
                sample_count=habit.sample_count,
                decay_gamma=habit.decay_gamma,
                last_reason_code=habit.last_reason_code,
                created_at=existing.created_at,
                updated_at=habit.updated_at,
            )
        else:
            stored = habit
        self._habits[key] = stored
        return stored

    async def delete_habit(self, owner_id: uuid.UUID, habit_id: uuid.UUID) -> bool:
        for key, item in list(self._habits.items()):
            if item.owner_id == owner_id and item.habit_id == habit_id:
                del self._habits[key]
                return True
        return False

    async def delete_all_habits(self, owner_id: uuid.UUID, *, pet_id: str | None = None) -> int:
        deleted = 0
        for key, item in list(self._habits.items()):
            if item.owner_id == owner_id and (pet_id is None or item.pet_id == pet_id):
                del self._habits[key]
                deleted += 1
        return deleted

    async def get_default(
        self, owner_id: uuid.UUID, pet_id: str, scope_key: str
    ) -> HostDefaultRecord | None:
        return self._defaults.get((owner_id, pet_id, scope_key))

    async def list_defaults(
        self, owner_id: uuid.UUID, *, pet_id: str | None = None
    ) -> list[HostDefaultRecord]:
        rows = [
            item
            for item in self._defaults.values()
            if item.owner_id == owner_id and (pet_id is None or item.pet_id == pet_id)
        ]
        rows.sort(key=lambda item: (item.updated_at, str(item.default_id)), reverse=True)
        return rows

    async def upsert_default(self, default: HostDefaultRecord) -> HostDefaultRecord:
        key = (default.owner_id, default.pet_id, default.scope_key)
        existing = self._defaults.get(key)
        if existing is not None:
            stored = HostDefaultRecord(
                default_id=existing.default_id,
                owner_id=default.owner_id,
                pet_id=default.pet_id,
                scope_key=default.scope_key,
                host_id=default.host_id,
                source=default.source,
                created_at=existing.created_at,
                updated_at=default.updated_at,
            )
        else:
            stored = default
        self._defaults[key] = stored
        return stored

    async def delete_default(self, owner_id: uuid.UUID, pet_id: str, scope_key: str) -> bool:
        key = (owner_id, pet_id, scope_key)
        if key in self._defaults:
            del self._defaults[key]
            return True
        return False
