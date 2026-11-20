"""`SqlComputerUseSessionStore` against PostgreSQL (UDC-17 S17.6).

Covers what the in-memory store cannot: a session that survives a new store instance (a
restart), atomic step counting across concurrent writers, one running session per owner
enforced by the database, and that draft text and UI trees never reach a column.
"""

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
from xuna_backend.modules.actions.application.computer_use import ComputerUseSessionState
from xuna_backend.modules.actions.infrastructure.cu_session_store import (
    SqlComputerUseSessionStore,
)
from xuna_backend.modules.actions.infrastructure.models import CuSessionModel
from xuna_backend.modules.hosts.domain.entities import HostId
from xuna_backend.modules.hosts.infrastructure.models import HostModel

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 11, 13, 9, 0, 0, tzinfo=UTC)
KEY = b"\x04" + bytes(range(1, 65))


@pytest.fixture
def db_url(database_url: str) -> str:
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
    )
    with psycopg.connect(database_url.replace("+psycopg", "")) as connection:
        connection.execute("TRUNCATE cu_sessions, hosts, owners CASCADE")
    return database_url


async def _seed(db: Database) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    owner, other, host = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with db.sessions() as session:
        for owner_id, subject in ((owner, "owner-a"), (other, "owner-b")):
            session.add(
                OwnerRow(
                    owner_id=owner_id,
                    issuer="https://auth.test.xuna",
                    subject=subject,
                    created_at=NOW,
                )
            )
        await session.flush()
        session.add(
            HostModel(
                host_id=host,
                owner_id=owner,
                alias="Mac",
                platform="macos",
                hostname="mac-host",
                os_version="1",
                agent_version="1.0.0",
                public_key=KEY,
                status="active",
                created_at=NOW,
            )
        )
        await session.commit()
    return owner, other, host


def _session(session_id: str, owner: uuid.UUID, host: uuid.UUID) -> ComputerUseSessionState:
    return ComputerUseSessionState(
        session_id=session_id,
        owner_id=owner,
        host_id=HostId(host),
        goal="Gửi báo cáo cho Lan",
        allowed_apps=("notes", "mail"),
        allowed_domains=("lich.example.com",),
        max_steps=40,
        max_minutes=20,
        started_at_utc=NOW,
        last_heartbeat_at=NOW,
        target_alias="Mac",
    )


def test_sessions_survive_a_restart_without_their_drafts_or_ui_tree(db_url: str) -> None:
    async def run() -> None:
        db = Database(db_url)
        try:
            owner, other, host = await _seed(db)
            store = SqlComputerUseSessionStore(db.sessions)
            assert await store.get_active_for_owner(owner) is None

            await store.start(_session("cu-1", owner, host), NOW)
            live = await store.get_active_for_owner(owner)
            assert live is not None and live.session_id == "cu-1"
            assert live.goal == "Gửi báo cáo cho Lan"
            assert live.allowed_apps == ("notes", "mail")
            assert live.started_at_utc == NOW
            assert await store.get_active_for_owner(other) is None

            live.typed_since_commit.append("mã mở cửa 4821")
            live.cached_nodes["@s1:e1"] = {"name": "Gửi", "role": "button"}
            live.last_app_id = "notes"
            live.last_window_title = "Ghi chú tuần"
            await store.set_typed_count("cu-1", 1)
            assert await store.record_step("cu-1") == 1
            await store.set_scope("cu-1", ("notes", "mail", "calendar"), ("lich.example.com",))
            later = NOW + timedelta(seconds=5)
            assert await store.touch_heartbeat("cu-1", owner, later)
            assert not await store.touch_heartbeat("cu-1", other, later)

            # Same process: the draft, tree, and last_window are still attached to the session.
            again = await store.get("cu-1")
            assert again is not None
            assert again.typed_since_commit == ["mã mở cửa 4821"]
            assert "@s1:e1" in again.cached_nodes
            assert again.last_app_id == "notes"
            assert again.last_window_title == "Ghi chú tuần"

            # Nothing of them is in the table.
            with psycopg.connect(db_url.replace("+psycopg", "")) as connection:
                row = connection.execute("SELECT * FROM cu_sessions").fetchone()
            assert row is not None
            assert "4821" not in repr(row)
            assert "Gửi'" not in repr(row)
            assert "Ghi chú tuần" not in repr(row)

            # A restart: a new store instance reads the same session back.
            restarted = SqlComputerUseSessionStore(db.sessions)
            back = await restarted.get_active_for_owner(owner)
            assert back is not None
            assert back.steps_used == 1
            assert back.typed_count == 1
            assert back.typed_since_commit == []
            assert back.cached_nodes == {}
            assert back.last_app_id is None
            assert back.last_window_title is None
            assert back.allowed_apps == ("notes", "mail", "calendar")
            assert back.last_heartbeat_at == later
        finally:
            await db.dispose()

    asyncio.run(run())


def test_steps_count_atomically_and_one_session_runs_per_owner(db_url: str) -> None:
    async def run() -> None:
        db = Database(db_url)
        try:
            owner, _, host = await _seed(db)
            store = SqlComputerUseSessionStore(db.sessions)
            await store.start(_session("cu-1", owner, host), NOW)
            counts = await asyncio.gather(*(store.record_step("cu-1") for _ in range(10)))
            assert sorted(c for c in counts if c is not None) == list(range(1, 11))

            # Starting another session replaces the running one.
            await store.start(_session("cu-2", owner, host), NOW)
            first = await store.get("cu-1")
            assert first is not None and not first.active
            assert first.stop_reason == "replaced"
            assert first.goal == ""
            live = await store.get_active_for_owner(owner)
            assert live is not None and live.session_id == "cu-2"

            # The database itself refuses a second running session.
            async with db.sessions() as session:
                row = CuSessionModel(
                    session_id="cu-3",
                    owner_id=owner,
                    host_id=host,
                    target_alias="Mac",
                    goal="x",
                    apps=["notes"],
                    domains=[],
                    max_steps=1,
                    max_minutes=1,
                    steps_used=0,
                    typed_count=0,
                    started_at=NOW,
                    last_heartbeat_at=NOW,
                    active=True,
                )
                session.add(row)
                with pytest.raises(IntegrityError):
                    await session.commit()
        finally:
            await db.dispose()

    asyncio.run(run())


def test_ending_keeps_the_first_reason_and_stops_counting(db_url: str) -> None:
    async def run() -> None:
        db = Database(db_url)
        try:
            owner, other, host = await _seed(db)
            store = SqlComputerUseSessionStore(db.sessions)
            await store.start(_session("cu-1", owner, host), NOW)
            await store.set_typed_count("cu-1", 2)
            prompts = await asyncio.gather(*(store.record_prompt("cu-1") for _ in range(4)))
            assert sorted(p for p in prompts if p is not None) == [2, 3, 4, 5]
            counted = await SqlComputerUseSessionStore(db.sessions).get("cu-1")
            assert counted is not None and counted.owner_prompts == 5
            live_s = await store.get_active_for_owner(owner)
            assert live_s is not None
            live_s.last_app_id = "notes"
            live_s.last_window_title = "Draft"
            ended = await store.mark_ended("cu-1", "heartbeat_timeout", NOW)
            assert ended is not None and not ended.active
            assert ended.stop_reason == "heartbeat_timeout"
            assert ended.goal == "" and ended.typed_count == 0
            assert ended.last_app_id is None and ended.last_window_title is None
            again = await store.mark_ended("cu-1", "owner_stopped", NOW)
            assert again is not None and again.stop_reason == "heartbeat_timeout"
            assert await store.record_step("cu-1") is None
            assert await store.record_prompt("cu-1") is None
            assert not await store.touch_heartbeat("cu-1", owner, NOW)
            assert await store.get_active_for_owner(owner) is None

            await store.start(_session("cu-2", owner, host), NOW)
            assert await store.stop_all_for_owner(other, "owner_app", NOW) == 0
            assert await store.stop_all_for_owner(owner, "owner_app", NOW) == 1
            assert await store.get_active_for_owner(owner) is None

            # Ended rows older than a day are dropped when the owner starts a new session.
            await store.start(_session("cu-3", owner, host), NOW + timedelta(days=2))
            assert await store.get("cu-1") is None
            assert await store.get("cu-3") is not None
        finally:
            await db.dispose()

    asyncio.run(run())


def test_restarted_host_reusing_session_id_and_cross_owner_collision_do_not_fail(
    db_url: str,
) -> None:
    async def run() -> None:
        db = Database(db_url)
        try:
            owner, other, host = await _seed(db)
            other_host = uuid.uuid4()
            async with db.sessions() as session:
                session.add(
                    HostModel(
                        host_id=other_host,
                        owner_id=other,
                        alias="OtherMac",
                        platform="macos",
                        hostname="other-mac",
                        os_version="1",
                        agent_version="1.0.0",
                        public_key=KEY,
                        status="active",
                        created_at=NOW,
                    )
                )
                await session.commit()

            store = SqlComputerUseSessionStore(db.sessions)
            await store.start(_session("cu-session-1", owner, host), NOW)
            await store.mark_ended("cu-session-1", "completed", NOW + timedelta(seconds=10))

            # Same owner's host restarts within 24h and sends "cu-session-1" again -> succeeds!
            later = NOW + timedelta(minutes=5)
            await store.start(_session("cu-session-1", owner, host), later)
            live = await store.get_active_for_owner(owner)
            assert live is not None and live.session_id == "cu-session-1" and live.active

            # Another owner's host also sends "cu-session-1" -> disambiguated without PK crash!
            other_state = _session("cu-session-1", other, other_host)
            await store.start(other_state, later)
            other_live = await store.get_active_for_owner(other)
            assert other_live is not None and other_live.active
            assert other_live.session_id != "cu-session-1"
            # First owner's active session remains untouched
            still_live = await store.get_active_for_owner(owner)
            assert still_live is not None and still_live.session_id == "cu-session-1"
        finally:
            await db.dispose()

    asyncio.run(run())
