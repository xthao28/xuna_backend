import os
import subprocess
import sys
from pathlib import Path

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
