import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _alembic(database_url: str, *args: str) -> None:
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", *args], cwd=ROOT, env=env, check=True
    )


@pytest.mark.postgres
def test_upgrade_downgrade_round_trip(database_url: str) -> None:
    _alembic(database_url, "upgrade", "head")
    _alembic(database_url, "downgrade", "base")
    _alembic(database_url, "upgrade", "head")


def test_single_head() -> None:
    versions = list((ROOT / "migrations" / "versions").glob("*.py"))
    heads = {p.stem.split("_")[0] for p in versions}
    assert len(versions) == len(heads), "duplicate revision prefixes"


@pytest.mark.postgres
def test_0022_hosts_backfill_and_collision_guard(database_url: str) -> None:
    _alembic(database_url, "downgrade", "base")
    _alembic(database_url, "upgrade", "0021")

    owner_1 = uuid.uuid4()
    owner_2 = uuid.uuid4()
    host_1 = uuid.uuid4()
    host_2 = uuid.uuid4()
    host_3 = uuid.uuid4()
    host_revoked = uuid.uuid4()
    host_other_owner = uuid.uuid4()
    base_time = datetime(2026, 11, 20, 10, 0, 0, tzinfo=UTC)
    pubkey = b"\x04" + bytes(range(1, 65))

    conn_str = database_url.replace("+psycopg", "")
    with psycopg.connect(conn_str) as conn:
        for oid, sub in ((owner_1, "mig-owner-1"), (owner_2, "mig-owner-2")):
            conn.execute(
                "INSERT INTO owners (owner_id, issuer, subject, created_at) "
                "VALUES (%s, %s, %s, %s)",
                (oid, "https://auth.test.xuna", sub, base_time),
            )
        rows = [
            (host_1, owner_1, "Máy của Thảo", "macos", "active", base_time),
            (
                host_2,
                owner_1,
                "May cua Thao",
                "windows",
                "active",
                base_time + timedelta(seconds=1),
            ),
            (
                host_3,
                owner_1,
                "MÁY CỦA THẢO",
                "android",
                "active",
                base_time + timedelta(seconds=2),
            ),
            (
                host_revoked,
                owner_1,
                "Máy của Thảo",
                "windows",
                "revoked",
                base_time + timedelta(seconds=3),
            ),
            (host_other_owner, owner_2, "Máy của Thảo", "ios", "active", base_time),
        ]
        for hid, oid, alias, platform, status, created_at in rows:
            conn.execute(
                "INSERT INTO hosts (host_id, owner_id, alias, platform, hostname, os_version, "
                "agent_version, public_key, status, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (hid, oid, alias, platform, "host", "1.0", "0.1.0", pubkey, status, created_at),
            )
        conn.commit()

    _alembic(database_url, "upgrade", "0022")

    with psycopg.connect(conn_str) as conn:
        fetched = conn.execute(
            "SELECT host_id, host_kind, place, nicknames, alias_folded, wol_mac " "FROM hosts"
        ).fetchall()
        by_id = {r[0]: r[1:] for r in fetched}

    assert by_id[host_1] == ("desktop", None, [], "may cua thao", None)
    assert by_id[host_2] == ("desktop", None, [], "may cua thao_2", None)
    assert by_id[host_3] == ("mobile", None, [], "may cua thao_3", None)
    assert by_id[host_revoked] == ("desktop", None, [], "may cua thao", None)
    assert by_id[host_other_owner] == ("mobile", None, [], "may cua thao", None)

    _alembic(database_url, "downgrade", "0021")
    _alembic(database_url, "upgrade", "head")
