"""`SqlApprovalRepository.count_approved` against PostgreSQL (UDC-17 S17.5).

Only approved rows for exactly this owner, host, tool and `args_sha256`, decided inside the
window, are counted.
"""

import asyncio
import uuid
from datetime import timedelta

import psycopg
import pytest
from tests.modules.actions.test_cu_session_store_postgres import NOW, _seed
from tests.modules.actions.test_cu_session_store_postgres import db_url as db_url

from xuna_backend.core.db import Database
from xuna_backend.modules.actions.domain.entities import ApprovalRequest, ApprovalState
from xuna_backend.modules.actions.infrastructure.repository import SqlApprovalRepository
from xuna_backend.modules.hosts.domain.entities import HostId, RiskTier

pytestmark = pytest.mark.postgres

SHA_A = "a" * 64
SHA_B = "b" * 64


def _row(
    owner: uuid.UUID,
    host: uuid.UUID | None,
    *,
    sha: str = SHA_A,
    tool: str = "app.launch",
    state: ApprovalState = ApprovalState.APPROVED,
    decided_ago: timedelta = timedelta(days=1),
) -> ApprovalRequest:
    decided = NOW - decided_ago
    return ApprovalRequest(
        approval_id=uuid.uuid4(),
        owner_id=owner,
        host_id=HostId(host) if host is not None else None,
        target_alias="Mac",
        tool=tool,
        args_sha256=sha,
        risk=RiskTier.R2,
        summary="Mở ứng dụng trên Mac",
        created_at=decided,
        expires_at=decided + timedelta(minutes=2),
        state=state,
        decided_at=None if state == ApprovalState.PENDING else decided,
    )


def test_count_approved_counts_only_this_exact_call_in_the_window(db_url: str) -> None:  # noqa: F811
    async def run() -> None:
        with psycopg.connect(db_url.replace("+psycopg", "")) as connection:
            connection.execute("TRUNCATE action_approvals CASCADE")
        db = Database(db_url)
        try:
            owner, other, host = await _seed(db)
            rows = [
                _row(owner, host),
                _row(owner, host),
                _row(owner, host, decided_ago=timedelta(days=29)),
                _row(owner, host, sha=SHA_B),
                _row(owner, host, tool="app.close"),
                _row(owner, host, state=ApprovalState.REJECTED),
                _row(owner, host, state=ApprovalState.PENDING),
                _row(owner, host, decided_ago=timedelta(days=31)),
                _row(owner, None),
            ]
            async with db.sessions() as session:
                repo = SqlApprovalRepository(session)
                for row in rows:
                    await repo.save(row)
                await session.commit()

            since = NOW - timedelta(days=30)
            async with db.sessions() as session:
                repo = SqlApprovalRepository(session)
                assert (
                    await repo.count_approved(owner, HostId(host), "app.launch", SHA_A, since) == 3
                )
                assert (
                    await repo.count_approved(other, HostId(host), "app.launch", SHA_A, since) == 0
                )
                assert (
                    await repo.count_approved(
                        owner, HostId(uuid.uuid4()), "app.launch", SHA_A, since
                    )
                    == 0
                )
        finally:
            await db.dispose()

    asyncio.run(run())
