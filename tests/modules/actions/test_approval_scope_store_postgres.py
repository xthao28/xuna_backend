"""`SqlApprovalScopeStore` tests against PostgreSQL (Wave A R2, table `approval_scopes`).

Covers:
- Scopes survive a new store instance (server restart).
- Valid scopes return True within 15-minute TTL; expired scopes return False.
- Upsert on re-grant updates expiration and created_at timestamps.
- Isolation across owner_id, pet_id, host_id, and tool.
- Clear owner removes all scopes for that owner without affecting other owners.
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

from xuna_backend.core.db import Database
from xuna_backend.modules.accounts.infrastructure.models import OwnerRow
from xuna_backend.modules.actions.domain.entities import CONVERSATION_SCOPE_TTL
from xuna_backend.modules.actions.infrastructure.approval_scope_store import (
    SqlApprovalScopeStore,
)
from xuna_backend.modules.hosts.domain.entities import HostId
from xuna_backend.modules.hosts.infrastructure.models import HostModel

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)
KEY = b"\x04" + bytes(range(1, 65))


@pytest.fixture
def db_url(database_url: str) -> str:
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
    )
    with psycopg.connect(database_url.replace("+psycopg", "")) as connection:
        connection.execute("TRUNCATE approval_scopes, hosts, owners CASCADE")
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


def test_approval_scopes_survive_store_restart(db_url: str) -> None:
    async def run() -> None:
        db = Database(db_url)
        try:
            owner, _, host = await _seed(db)
            pet_id = "pet-xuna-1"
            host_id = HostId(host)
            tool = "app.launch"

            store1 = SqlApprovalScopeStore(db.sessions)

            # 1. Initially no valid scope
            assert not await store1.has_valid_scope(owner, pet_id, host_id, tool, NOW)

            # 2. Grant 15-minute scope
            expires_at = await store1.grant_scope(owner, pet_id, host_id, tool, NOW)
            assert expires_at == NOW + CONVERSATION_SCOPE_TTL

            # Verified directly on store1
            assert await store1.has_valid_scope(owner, pet_id, host_id, tool, NOW)

            # 3. Simulate server restart: new store instance reads from database
            store2 = SqlApprovalScopeStore(db.sessions)
            assert await store2.has_valid_scope(owner, pet_id, host_id, tool, NOW)

            # 4. Verified raw row exists in PostgreSQL table
            with psycopg.connect(db_url.replace("+psycopg", "")) as conn:
                row = conn.execute(
                    "SELECT owner_id, pet_id, host_id, tool, expires_at FROM approval_scopes"
                ).fetchone()
            assert row is not None
            assert str(row[0]) == str(owner)
            assert row[1] == pet_id
            assert str(row[2]) == str(host)
            assert row[3] == tool
        finally:
            await db.dispose()

    asyncio.run(run())


def test_approval_scope_ttl_expiration(db_url: str) -> None:
    async def run() -> None:
        db = Database(db_url)
        try:
            owner, _, host = await _seed(db)
            pet_id = "pet-xuna-ttl"
            host_id = HostId(host)
            tool = "ui.inspect"

            store = SqlApprovalScopeStore(db.sessions)
            await store.grant_scope(owner, pet_id, host_id, tool, NOW)

            # Valid at grant time and within 15 minutes
            assert await store.has_valid_scope(owner, pet_id, host_id, tool, NOW)
            assert await store.has_valid_scope(
                owner, pet_id, host_id, tool, NOW + timedelta(minutes=14, seconds=59)
            )

            # Expired at 15 minutes and beyond
            assert not await store.has_valid_scope(
                owner, pet_id, host_id, tool, NOW + timedelta(minutes=15)
            )
            assert not await store.has_valid_scope(
                owner, pet_id, host_id, tool, NOW + timedelta(minutes=20)
            )
        finally:
            await db.dispose()

    asyncio.run(run())


def test_approval_scope_upsert_on_regrant(db_url: str) -> None:
    async def run() -> None:
        db = Database(db_url)
        try:
            owner, _, host = await _seed(db)
            pet_id = "pet-xuna-upsert"
            host_id = HostId(host)
            tool = "app.launch"

            store = SqlApprovalScopeStore(db.sessions)
            first_expiry = await store.grant_scope(owner, pet_id, host_id, tool, NOW)
            assert first_expiry == NOW + CONVERSATION_SCOPE_TTL

            # Re-grant 10 minutes later extends expiration
            later = NOW + timedelta(minutes=10)
            second_expiry = await store.grant_scope(owner, pet_id, host_id, tool, later)
            assert second_expiry == later + CONVERSATION_SCOPE_TTL
            assert second_expiry > first_expiry

            # At NOW + 16m (past first expiry, but within second expiry): still valid!
            assert await store.has_valid_scope(
                owner, pet_id, host_id, tool, NOW + timedelta(minutes=16)
            )

            # Exactly one row exists in table (no duplicates)
            with psycopg.connect(db_url.replace("+psycopg", "")) as conn:
                count = conn.execute("SELECT count(*) FROM approval_scopes").fetchone()
            assert count is not None and count[0] == 1
        finally:
            await db.dispose()

    asyncio.run(run())


def test_approval_scope_isolation_and_clear_owner(db_url: str) -> None:
    async def run() -> None:
        db = Database(db_url)
        try:
            owner_a, owner_b, host = await _seed(db)
            pet_id = "pet-xuna-iso"
            host_id = HostId(host)
            tool = "app.launch"

            store = SqlApprovalScopeStore(db.sessions)
            await store.grant_scope(owner_a, pet_id, host_id, tool, NOW)
            await store.grant_scope(owner_b, pet_id, host_id, tool, NOW)

            # Isolation checks
            assert not await store.has_valid_scope(owner_a, "other-pet", host_id, tool, NOW)
            assert not await store.has_valid_scope(
                owner_a, pet_id, HostId.generate(), tool, NOW
            )
            assert not await store.has_valid_scope(owner_a, pet_id, host_id, "other.tool", NOW)

            # Clear owner A only
            deleted = await store.clear_owner(owner_a)
            assert deleted == 1

            # Owner A's grant is gone
            assert not await store.has_valid_scope(owner_a, pet_id, host_id, tool, NOW)

            # Owner B's grant is intact
            assert await store.has_valid_scope(owner_b, pet_id, host_id, tool, NOW)
        finally:
            await db.dispose()

    asyncio.run(run())
