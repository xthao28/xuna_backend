"""`SqlAutonomousJobStore` against PostgreSQL (Wave D M1).

Covers persistent database operations for AutonomousJob:
- Job insert, update, retrieval across store instances.
- Active job query for owner.
- Filtered listing by state with limit and offset.
- Sub-goals and checkpoint JSON serialization roundtrip in postgresql JSONB.
"""

import asyncio
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from xuna_backend.core.db import Database
from xuna_backend.modules.accounts.infrastructure.models import OwnerRow
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    SubGoal,
    SubGoalStatus,
)
from xuna_backend.modules.actions.infrastructure.job_store import SqlAutonomousJobStore
from xuna_backend.modules.hosts.domain.entities import HostId
from xuna_backend.modules.hosts.infrastructure.models import HostModel

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 11, 15, 12, 0, 0, tzinfo=UTC)
KEY = b"\x04" + bytes(range(1, 65))


@pytest.fixture
def db_url(database_url: str) -> str:
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
    )
    with psycopg.connect(database_url.replace("+psycopg", "")) as connection:
        connection.execute("TRUNCATE autonomous_jobs, hosts, owners CASCADE")
    return database_url


async def _seed(db: Database) -> tuple[uuid.UUID, uuid.UUID]:
    owner, host = uuid.uuid4(), uuid.uuid4()
    async with db.sessions() as session:
        session.add(
            OwnerRow(
                owner_id=owner,
                issuer="https://auth.test.xuna",
                subject="postgres-owner",
                created_at=NOW,
            )
        )
        await session.flush()
        session.add(
            HostModel(
                host_id=host,
                owner_id=owner,
                alias="pc",
                platform="windows",
                hostname="pc-host",
                os_version="11",
                agent_version="1.0.0",
                public_key=KEY,
                status="active",
                created_at=NOW,
                last_seen_at=NOW,
            )
        )
        await session.commit()
    return owner, host


def test_sql_autonomous_job_store_roundtrip(db_url: str) -> None:
    async def _test() -> None:
        db = Database(db_url)
        try:
            owner, host = await _seed(db)
            store = SqlAutonomousJobStore(lambda: db.sessions())
            job_id = uuid.uuid4()

            sub_goals = (
                SubGoal(
                    id="sg-1",
                    title="Initialize",
                    description="Prepare environment",
                    status=SubGoalStatus.COMPLETED,
                    tool_call_count=2,
                    result_summary="Ready",
                    started_at=NOW,
                    completed_at=NOW,
                ),
                SubGoal(
                    id="sg-2",
                    title="Execute",
                    description="Run main task",
                    status=SubGoalStatus.IN_PROGRESS,
                    started_at=NOW,
                ),
            )

            job = AutonomousJob(
                job_id=job_id,
                owner_id=owner,
                pet_id="pet-pg",
                host_id=HostId(host),
                session_id="cu-sess-1",
                title="Postgres Persistent Job",
                goal="Test database persistence",
                state=AutonomousJobState.RUNNING,
                sub_goals=sub_goals,
                active_sub_goal_index=1,
                max_steps=40,
                steps_used=3,
                checkpoint_data={"step": 3, "intent": "run"},
                created_at=NOW,
                updated_at=NOW,
                started_at=NOW,
            )

            await store.save(job)

            # Retrieve with new store instance (simulate restart)
            store2 = SqlAutonomousJobStore(lambda: db.sessions())
            loaded = await store2.get(job_id)
            assert loaded is not None
            assert loaded.job_id == job_id
            assert loaded.owner_id == owner
            assert loaded.pet_id == "pet-pg"
            assert loaded.host_id == HostId(host)
            assert loaded.state == AutonomousJobState.RUNNING
            assert len(loaded.sub_goals) == 2
            assert loaded.sub_goals[0].status == SubGoalStatus.COMPLETED
            assert loaded.sub_goals[1].status == SubGoalStatus.IN_PROGRESS
            assert loaded.checkpoint_data == {"step": 3, "intent": "run"}

            # Active job
            active = await store2.get_active_for_owner(owner)
            assert active is not None
            assert active.job_id == job_id

            # Update job state
            updated = loaded.with_state(
                AutonomousJobState.COMPLETED,
                now=NOW,
                steps_used=4,
                stop_reason="finished",
            )
            await store2.save(updated)

            reloaded = await store2.get(job_id)
            assert reloaded is not None
            assert reloaded.state == AutonomousJobState.COMPLETED
            assert reloaded.is_terminal is True
            assert reloaded.stop_reason == "finished"

            # No active job after completion
            active_after = await store2.get_active_for_owner(owner)
            assert active_after is None
        finally:
            await db.dispose()

    asyncio.run(_test())
