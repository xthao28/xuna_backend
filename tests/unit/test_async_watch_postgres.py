"""PostgreSQL integration tests for migration 0023 (`job_watches`, `wake_inbox_items`)."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from xuna_backend.core.db import Database
from xuna_backend.modules.actions.domain.job import AutonomousJob, AutonomousJobState
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
from xuna_backend.modules.actions.infrastructure.job_store import SqlAutonomousJobStore
from xuna_backend.modules.actions.infrastructure.watch_repository import (
    SqlJobWatchRepository,
    SqlWakeInboxRepository,
)
from xuna_backend.modules.hosts.domain.entities import HostId


@pytest.mark.postgres
def test_sql_watch_and_wake_inbox_repositories_roundtrip(database_url: str) -> None:
    owner_id = uuid.uuid4()
    host_id = HostId.generate()
    job_id = uuid.uuid4()
    now = datetime.now(UTC)
    pubkey = b"\x04" + bytes(range(1, 65))
    conn_str = database_url.replace("+psycopg", "")

    with psycopg.connect(conn_str) as conn:
        conn.execute(
            "INSERT INTO owners (owner_id, issuer, subject, created_at) VALUES (%s, %s, %s, %s)",
            (owner_id, "https://auth.test.xuna", f"watch-pg-{owner_id}", now),
        )
        conn.execute(
            "INSERT INTO hosts (host_id, owner_id, alias, platform, hostname, os_version, "
            "agent_version, public_key, status, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                host_id.value,
                owner_id,
                "WatchHost",
                "macos",
                "host",
                "14.0",
                "0.2.0",
                pubkey,
                "active",
                now,
            ),
        )
        conn.commit()

    async def _run() -> None:
        db = Database(database_url)
        try:
            job_store = SqlAutonomousJobStore(db.sessions)
            watch_repo = SqlJobWatchRepository(db.sessions)
            inbox_repo = SqlWakeInboxRepository(db.sessions)

            job = AutonomousJob(
                job_id=job_id,
                owner_id=owner_id,
                pet_id="pet_pg",
                host_id=host_id,
                session_id=None,
                title="PG watch job",
                goal="Verify SQL persistence",
                state=AutonomousJobState.WAITING_WATCH,
                created_at=now,
                updated_at=now,
            )
            await job_store.save(job)
            active_job = await job_store.get_active_for_owner(owner_id)
            assert active_job is not None
            assert active_job.job_id == job_id
            assert active_job.state == AutonomousJobState.WAITING_WATCH

            watch = JobWatch(
                watch_id=uuid.uuid4(),
                job_id=job_id,
                owner_id=owner_id,
                host_id=host_id,
                mode=WatchMode.COMPOSITE,
                signals=(WatchCompositeSignal.UI_QUIESCENCE, WatchCompositeSignal.PROCESS_IDLE),
                target=WatchTarget(pid=4242, window_id=7, app_id="editor", workspace_alias="repo"),
                done_criteria="Build completes",
                continuation_mode=ContinuationMode.AUTO_CONTINUE,
                expires_at=now + timedelta(seconds=300),
                created_at=now,
                updated_at=now,
            )
            await watch_repo.save(watch)
            loaded_watch = await watch_repo.get_active_for_job(job_id)
            assert loaded_watch is not None
            assert loaded_watch.target.pid == 4242
            assert loaded_watch.signals == (
                WatchCompositeSignal.UI_QUIESCENCE,
                WatchCompositeSignal.PROCESS_IDLE,
            )
            assert len(await watch_repo.list_active_for_host(host_id, owner_id=owner_id)) == 1

            item = WakeInboxItem(
                item_id=uuid.uuid4(),
                owner_id=owner_id,
                pet_id="pet_pg",
                job_id=job_id,
                host_id=host_id,
                kind=WakeInboxKind.JOB_REPORT,
                title="PG Done",
                summary="Build succeeded.",
                suggested_actions=(WakeSuggestedAction.DISMISS,),
                gesture_cue=WatchGestureCue.BOUNCE_PLAY,
                created_at=now,
            )
            await inbox_repo.save(item)
            listed = await inbox_repo.list_for_owner(owner_id, unread_only=True)
            assert len(listed) == 1
            assert listed[0].item_id == item.item_id

            acked = await inbox_repo.mark_read_or_dismiss(
                item.item_id, owner_id, now=now + timedelta(seconds=5), dismiss=False
            )
            assert acked is not None and acked.read_at is not None
            assert await inbox_repo.list_for_owner(owner_id, unread_only=True) == []

            cancelled = await watch_repo.cancel_active_for_owner(
                owner_id, reason="estop", now=now + timedelta(seconds=10)
            )
            assert len(cancelled) == 1
            assert cancelled[0].status == WatchStatus.CANCELLED
        finally:
            await db.dispose()

    asyncio.run(_run())
