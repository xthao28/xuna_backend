"""`SqlStandingApprovalRepository` against PostgreSQL (ADR-0014 D1).

The in-memory repository is covered elsewhere; this exercises the real SQL: the use-count and
expiry filters, `SELECT ... FOR UPDATE` consumption, owner scoping and revocation.
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
from xuna_backend.modules.actions.domain.standing_approval import StandingApproval
from xuna_backend.modules.actions.infrastructure.standing_repository import (
    SqlStandingApprovalRepository,
)
from xuna_backend.modules.hosts.domain.crypto import args_sha256
from xuna_backend.modules.hosts.domain.entities import HostId
from xuna_backend.modules.hosts.infrastructure.models import HostModel

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 11, 6, 12, 0, 0, tzinfo=UTC)
ARGS = {"app_id": "com.spotify.music"}
KEY = b"\x04" + bytes(range(1, 65))


@pytest.fixture
def db_url(database_url: str) -> str:
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
    )
    with psycopg.connect(database_url.replace("+psycopg", "")) as connection:
        connection.execute("TRUNCATE host_standing_approvals, hosts, owners CASCADE")
    return database_url


def _approval(
    owner: uuid.UUID,
    host: uuid.UUID,
    *,
    approval_id: uuid.UUID | None = None,
    expires_in: timedelta = timedelta(days=7),
    max_uses: int | None = None,
    uses: int = 0,
    args: dict[str, object] | None = None,
) -> StandingApproval:
    return StandingApproval(
        approval_id=approval_id or uuid.uuid4(),
        owner_id=owner,
        host_id=HostId(host),
        target_alias="PC",
        tool="app.launch",
        args_sha256=args_sha256(args or ARGS),
        summary="Mở ứng dụng",
        created_at=NOW,
        expires_at=NOW + expires_in,
        max_uses=max_uses,
        signature=bytes(64),
        signer_public_key=KEY,
        uses=uses,
    )


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
                alias="PC",
                platform="windows",
                hostname="pc-host",
                os_version="1",
                agent_version="1.0.0",
                public_key=KEY,
                status="active",
                created_at=NOW,
            )
        )
        await session.commit()
    return owner, other, host


def test_sql_repository_save_list_consume_revoke_and_scoping(db_url: str) -> None:
    async def run() -> None:
        db = Database(db_url)
        try:
            owner, other, host = await _seed(db)
            digest = args_sha256(ARGS)
            host_id = HostId(host)

            async with db.sessions() as session:
                repo = SqlStandingApprovalRepository(session)
                unlimited = _approval(owner, host)
                once = _approval(owner, host, max_uses=1, args={"app_id": "org.mozilla.firefox"})
                await repo.save(unlimited)
                await repo.save(once)
                await session.commit()

            async with db.sessions() as session:
                repo = SqlStandingApprovalRepository(session)
                listed = await repo.list_active(owner, NOW)
                assert {a.approval_id for a in listed} == {
                    unlimited.approval_id,
                    once.approval_id,
                }
                assert await repo.list_active(other, NOW) == [], "another owner sees nothing"
                assert await repo.get(other, unlimited.approval_id) is None

                # Consuming counts a use and returns the updated record.
                first = await repo.consume_match(owner, host_id, "app.launch", digest, NOW)
                assert first is not None and first.uses == 1
                second = await repo.consume_match(owner, host_id, "app.launch", digest, NOW)
                assert second is not None and second.uses == 2
                await session.commit()

            async with db.sessions() as session:
                repo = SqlStandingApprovalRepository(session)
                stored = await repo.get(owner, unlimited.approval_id)
                assert stored is not None and stored.uses == 2, "uses are persisted"

                # A one-use approval matches exactly once, then drops out of both queries.
                once_digest = args_sha256({"app_id": "org.mozilla.firefox"})
                assert await repo.consume_match(owner, host_id, "app.launch", once_digest, NOW)
                assert (
                    await repo.consume_match(owner, host_id, "app.launch", once_digest, NOW) is None
                )
                assert once.approval_id not in {
                    a.approval_id for a in await repo.list_active(owner, NOW)
                }

                # No match across owners, tools, hosts or argument sets.
                assert await repo.consume_match(other, host_id, "app.launch", digest, NOW) is None
                assert await repo.consume_match(owner, host_id, "url.open", digest, NOW) is None
                assert (
                    await repo.consume_match(owner, HostId(uuid.uuid4()), "app.launch", digest, NOW)
                    is None
                )
                assert (
                    await repo.consume_match(
                        owner, host_id, "app.launch", args_sha256({"app_id": "x"}), NOW
                    )
                    is None
                )

                # Expiry: the same approval stops matching once its time has passed.
                later = NOW + timedelta(days=8)
                assert await repo.consume_match(owner, host_id, "app.launch", digest, later) is None
                assert await repo.list_active(owner, later) == []

                # Revocation is owner scoped, one-way and takes effect immediately.
                assert not await repo.revoke(other, unlimited.approval_id, NOW)
                assert await repo.revoke(owner, unlimited.approval_id, NOW)
                assert not await repo.revoke(owner, unlimited.approval_id, NOW)
                assert await repo.consume_match(owner, host_id, "app.launch", digest, NOW) is None
                await session.commit()
        finally:
            await db.dispose()

    asyncio.run(run())


def test_concurrent_consumption_never_exceeds_max_uses(db_url: str) -> None:
    """Two sessions race for the last use; `FOR UPDATE` lets exactly one win."""

    async def run() -> None:
        db = Database(db_url)
        try:
            owner, _, host = await _seed(db)
            approval = _approval(owner, host, max_uses=1)
            async with db.sessions() as session:
                await SqlStandingApprovalRepository(session).save(approval)
                await session.commit()

            async def attempt() -> bool:
                async with db.sessions() as session:
                    found = await SqlStandingApprovalRepository(session).consume_match(
                        owner, HostId(host), "app.launch", args_sha256(ARGS), NOW
                    )
                    await session.commit()
                    return found is not None

            results = await asyncio.gather(*(attempt() for _ in range(6)))
            assert sum(results) == 1

            async with db.sessions() as session:
                stored = await SqlStandingApprovalRepository(session).get(
                    owner, approval.approval_id
                )
                assert stored is not None and stored.uses == 1
        finally:
            await db.dispose()

    asyncio.run(run())
