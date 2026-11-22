"""Domain, in-memory, and PostgreSQL tests for `HostRoutingHabit` & `HostDefaultRecord`."""

import asyncio
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
from sqlalchemy.exc import IntegrityError

from xuna_backend.core.db import Database
from xuna_backend.modules.accounts.infrastructure.models import OwnerRow
from xuna_backend.modules.actions.domain.habits import (
    DEFAULT_HABIT_GAMMA,
    HostDefaultRecord,
    HostDefaultSource,
    HostRoutingHabit,
)
from xuna_backend.modules.actions.infrastructure.habit_repository import (
    InMemoryHabitRepository,
    SqlHabitRepository,
)
from xuna_backend.modules.actions.infrastructure.models import (
    HostDefaultModel,
    HostRoutingHabitModel,
)
from xuna_backend.modules.hosts.domain.entities import HostId
from xuna_backend.modules.hosts.infrastructure.models import HostModel

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 11, 20, 16, 0, 0, tzinfo=UTC)
KEY = b"\x04" + bytes(range(1, 65))


@pytest.fixture
def db_url(database_url: str) -> str:
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
    )
    with psycopg.connect(database_url.replace("+psycopg", "")) as connection:
        connection.execute("TRUNCATE host_routing_habits, host_defaults, hosts, owners CASCADE")
    return database_url


def test_habit_and_default_domain_validation() -> None:
    owner_id = uuid.uuid4()
    host_id = HostId.generate()

    habit = HostRoutingHabit(
        habit_id=uuid.uuid4(),
        owner_id=owner_id,
        pet_id="pet-1",
        context_key="media:music:evening",
        host_id=host_id,
        positive_score=1.5,
        negative_score=0.2,
        sample_count=3,
        decay_gamma=DEFAULT_HABIT_GAMMA,
        last_reason_code="confirmed_pick",
        created_at=NOW,
        updated_at=NOW,
    )
    assert habit.decay_gamma == 0.85

    base_kw: dict[str, object] = {
        "habit_id": uuid.uuid4(),
        "owner_id": owner_id,
        "pet_id": "pet-1",
        "context_key": "media:-:evening",
        "host_id": host_id,
        "created_at": NOW,
        "updated_at": NOW,
    }
    for bad in (
        {"pet_id": ""},
        {"pet_id": "x" * 65},
        {"pet_id": "pet-1\n"},
        {"pet_id": "\tpet-1"},
        {"context_key": ""},
        {"context_key": "x" * 129},
        {"context_key": "bad\nkey"},
        {"context_key": "media:-:evening\r"},
        {"positive_score": -0.1},
        {"positive_score": float("nan")},
        {"positive_score": True},
        {"negative_score": -1.0},
        {"negative_score": float("inf")},
        {"sample_count": -1},
        {"sample_count": True},
        {"decay_gamma": 0.0},
        {"decay_gamma": 1.0},
        {"decay_gamma": float("nan")},
        {"last_reason_code": ""},
        {"last_reason_code": "r" * 65},
        {"last_reason_code": "confirmed_pick\n"},
        {"created_at": datetime(2026, 11, 20, 16, 0)},
        {"updated_at": datetime(2026, 11, 20, 16, 0)},
    ):
        with pytest.raises(ValueError):
            HostRoutingHabit(**{**base_kw, **bad})  # type: ignore[arg-type]

    default = HostDefaultRecord(
        default_id=uuid.uuid4(),
        owner_id=owner_id,
        pet_id="pet-1",
        scope_key="media",
        host_id=host_id,
        source=HostDefaultSource.SLASH_CMD,
        created_at=NOW,
        updated_at=NOW,
    )
    assert default.source == HostDefaultSource.SLASH_CMD

    for bad_def in (
        {"scope_key": "x" * 65},
        {"pet_id": "pet-1\n"},
        {"scope_key": "media\t"},
    ):
        with pytest.raises(ValueError):
            HostDefaultRecord(
                default_id=uuid.uuid4(),
                owner_id=owner_id,
                pet_id=str(bad_def.get("pet_id", "pet-1")),
                scope_key=str(bad_def.get("scope_key", "media")),
                host_id=host_id,
                created_at=NOW,
                updated_at=NOW,
            )


def test_in_memory_habit_repository_crud_and_scoping() -> None:
    async def _run() -> None:
        repo = InMemoryHabitRepository()
        owner_1, owner_2 = uuid.uuid4(), uuid.uuid4()
        host_1, host_2 = HostId.generate(), HostId.generate()

        h_id = uuid.uuid4()
        h1 = HostRoutingHabit(
            habit_id=h_id,
            owner_id=owner_1,
            pet_id="pet-1",
            context_key="media:-:evening",
            host_id=host_1,
            positive_score=1.0,
            negative_score=0.0,
            sample_count=1,
            last_reason_code="confirmed_pick",
            created_at=NOW,
            updated_at=NOW,
        )
        saved = await repo.upsert_habit(h1)
        assert saved.habit_id == h_id

        # Upsert conflict preserves habit_id and created_at
        h1_updated = HostRoutingHabit(
            habit_id=uuid.uuid4(),
            owner_id=owner_1,
            pet_id="pet-1",
            context_key="media:-:evening",
            host_id=host_1,
            positive_score=1.85,
            negative_score=0.0,
            sample_count=2,
            last_reason_code="execution_ok",
            created_at=NOW + timedelta(minutes=5),
            updated_at=NOW + timedelta(minutes=5),
        )
        upserted = await repo.upsert_habit(h1_updated)
        assert upserted.habit_id == h_id
        assert upserted.created_at == NOW
        assert upserted.sample_count == 2

        await repo.upsert_habit(
            HostRoutingHabit(
                habit_id=uuid.uuid4(),
                owner_id=owner_1,
                pet_id="pet-2",
                context_key="cu:-:morning",
                host_id=host_2,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        await repo.upsert_habit(
            HostRoutingHabit(
                habit_id=uuid.uuid4(),
                owner_id=owner_2,
                pet_id="pet-1",
                context_key="media:-:evening",
                host_id=host_1,
                created_at=NOW,
                updated_at=NOW,
            )
        )

        assert len(await repo.list_habits(owner_1)) == 2
        assert len(await repo.list_habits(owner_1, pet_id="pet-1")) == 1
        assert len(await repo.list_habits(owner_1, context_key="cu:-:morning")) == 1
        assert await repo.get_habit_by_id(owner_2, h_id) is None

        assert not await repo.delete_habit(owner_2, h_id)
        assert await repo.delete_habit(owner_1, h_id)
        assert not await repo.delete_habit(owner_1, h_id)
        assert await repo.delete_all_habits(owner_1, pet_id="pet-2") == 1
        assert len(await repo.list_habits(owner_2)) == 1

        d_id = uuid.uuid4()
        await repo.upsert_default(
            HostDefaultRecord(
                default_id=d_id,
                owner_id=owner_1,
                pet_id="pet-1",
                scope_key="media",
                host_id=host_1,
                source=HostDefaultSource.CONFIRMED,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        d_up = await repo.upsert_default(
            HostDefaultRecord(
                default_id=uuid.uuid4(),
                owner_id=owner_1,
                pet_id="pet-1",
                scope_key="media",
                host_id=host_2,
                source=HostDefaultSource.SLASH_CMD,
                created_at=NOW + timedelta(minutes=1),
                updated_at=NOW + timedelta(minutes=1),
            )
        )
        assert d_up.default_id == d_id
        assert d_up.created_at == NOW
        assert d_up.host_id == host_2
        assert d_up.source == HostDefaultSource.SLASH_CMD
        assert await repo.delete_default(owner_1, "pet-1", "media")
        assert not await repo.delete_default(owner_1, "pet-1", "media")

    asyncio.run(_run())


@pytest.mark.postgres
def test_postgres_habit_repository_crud_upserts_and_cascades(db_url: str) -> None:
    async def _run() -> None:
        db = Database(db_url)
        try:
            owner_1, owner_2 = uuid.uuid4(), uuid.uuid4()
            host_1, host_2 = uuid.uuid4(), uuid.uuid4()
            async with db.sessions() as session:
                for oid, sub in ((owner_1, "hab-owner-1"), (owner_2, "hab-owner-2")):
                    session.add(
                        OwnerRow(
                            owner_id=oid,
                            issuer="https://auth.test.xuna",
                            subject=sub,
                            created_at=NOW,
                        )
                    )
                await session.flush()
                for hid, oid, alias in (
                    (host_1, owner_1, "Mac"),
                    (host_2, owner_1, "WinPC"),
                ):
                    session.add(
                        HostModel(
                            host_id=hid,
                            owner_id=oid,
                            alias=alias,
                            platform="macos" if alias == "Mac" else "windows",
                            hostname=alias.lower(),
                            os_version="1",
                            agent_version="0.1.0",
                            public_key=KEY,
                            status="active",
                            created_at=NOW,
                            alias_folded=alias.lower(),
                        )
                    )
                await session.commit()

            # 1. Habit CRUD & atomic ON CONFLICT upsert
            habit_id_1 = uuid.uuid4()
            async with db.sessions() as session:
                repo = SqlHabitRepository(session)
                inserted = await repo.upsert_habit(
                    HostRoutingHabit(
                        habit_id=habit_id_1,
                        owner_id=owner_1,
                        pet_id="pet-1",
                        context_key="media:music:evening",
                        host_id=HostId(host_1),
                        positive_score=1.0,
                        negative_score=0.0,
                        sample_count=1,
                        decay_gamma=0.85,
                        last_reason_code="confirmed_pick",
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )
                assert inserted.habit_id == habit_id_1
                await repo.upsert_habit(
                    HostRoutingHabit(
                        habit_id=uuid.uuid4(),
                        owner_id=owner_1,
                        pet_id="pet-2",
                        context_key="cu:-:morning",
                        host_id=HostId(host_2),
                        positive_score=0.0,
                        negative_score=1.0,
                        sample_count=1,
                        last_reason_code="redirect_penalty",
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )
                await session.commit()

            # Conflict upsert preserves original habit_id & created_at
            later = NOW + timedelta(minutes=10)
            async with db.sessions() as session:
                repo = SqlHabitRepository(session)
                updated = await repo.upsert_habit(
                    HostRoutingHabit(
                        habit_id=uuid.uuid4(),
                        owner_id=owner_1,
                        pet_id="pet-1",
                        context_key="media:music:evening",
                        host_id=HostId(host_1),
                        positive_score=1.85,
                        negative_score=0.0,
                        sample_count=2,
                        decay_gamma=0.85,
                        last_reason_code="execution_ok",
                        created_at=later,
                        updated_at=later,
                    )
                )
                assert updated.habit_id == habit_id_1
                assert updated.created_at == NOW
                assert updated.sample_count == 2
                assert updated.positive_score == pytest.approx(1.85)
                await session.commit()

            async with db.sessions() as session:
                repo = SqlHabitRepository(session)
                fetched = await repo.get_habit(
                    owner_1, "pet-1", "media:music:evening", HostId(host_1)
                )
                assert fetched is not None
                assert fetched.habit_id == habit_id_1
                assert await repo.get_habit_by_id(owner_2, habit_id_1) is None
                assert len(await repo.list_habits(owner_1)) == 2
                assert len(await repo.list_habits(owner_1, pet_id="pet-1")) == 1
                assert len(await repo.list_habits(owner_1, context_key="cu:-:morning")) == 1
                assert await repo.list_habits(owner_2) == []

                assert not await repo.delete_habit(owner_2, habit_id_1)
                assert await repo.delete_habit(owner_1, habit_id_1)
                assert not await repo.delete_habit(owner_1, habit_id_1)
                await repo.upsert_habit(
                    HostRoutingHabit(
                        habit_id=uuid.uuid4(),
                        owner_id=owner_1,
                        pet_id="pet-temp",
                        context_key="app:-:night",
                        host_id=HostId(host_1),
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )
                assert await repo.delete_all_habits(owner_1, pet_id="pet-temp") == 1
                assert len(await repo.list_habits(owner_1)) == 1
                await session.commit()

            # 2. HostDefaultRecord CRUD & ON CONFLICT upsert
            def_id_1 = uuid.uuid4()
            async with db.sessions() as session:
                repo = SqlHabitRepository(session)
                d1 = await repo.upsert_default(
                    HostDefaultRecord(
                        default_id=def_id_1,
                        owner_id=owner_1,
                        pet_id="pet-1",
                        scope_key="media",
                        host_id=HostId(host_1),
                        source=HostDefaultSource.CONFIRMED,
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )
                assert d1.default_id == def_id_1
                d2 = await repo.upsert_default(
                    HostDefaultRecord(
                        default_id=uuid.uuid4(),
                        owner_id=owner_1,
                        pet_id="pet-1",
                        scope_key="media",
                        host_id=HostId(host_2),
                        source=HostDefaultSource.MIGRATED,
                        created_at=later,
                        updated_at=later,
                    )
                )
                assert d2.default_id == def_id_1
                assert d2.created_at == NOW
                assert d2.host_id == HostId(host_2)
                assert d2.source == HostDefaultSource.MIGRATED
                assert len(await repo.list_defaults(owner_1, pet_id="pet-1")) == 1
                assert await repo.get_default(owner_2, "pet-1", "media") is None
                await session.commit()

            # 3. Direct duplicate ORM insert raises IntegrityError
            async with db.sessions() as session:
                session.add(
                    HostDefaultModel(
                        default_id=uuid.uuid4(),
                        owner_id=owner_1,
                        pet_id="pet-1",
                        scope_key="media",
                        host_id=host_1,
                        source="confirmed",
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )
                with pytest.raises(IntegrityError):
                    await session.commit()

            async with db.sessions() as session:
                session.add(
                    HostRoutingHabitModel(
                        habit_id=uuid.uuid4(),
                        owner_id=owner_1,
                        pet_id="pet-2",
                        context_key="cu:-:morning",
                        host_id=host_2,
                        positive_score=0.0,
                        negative_score=0.0,
                        sample_count=0,
                        decay_gamma=0.85,
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )
                with pytest.raises(IntegrityError):
                    await session.commit()

            # 4. Deleting HostModel cascades to both host_routing_habits and host_defaults
            async with db.sessions() as session:
                host_row = await session.get(HostModel, host_2)
                assert host_row is not None
                await session.delete(host_row)
                await session.commit()

            async with db.sessions() as session:
                repo = SqlHabitRepository(session)
                assert await repo.list_habits(owner_1) == []
                assert await repo.list_defaults(owner_1) == []
        finally:
            await db.dispose()

    asyncio.run(_run())
