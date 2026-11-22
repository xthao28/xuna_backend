"""Session and per-test fixtures for opaque-box Spatial Device Mesh E2E tests."""

import asyncio
import contextlib
import os
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import pytest
import uvicorn

from tests.helpers import LocalIssuer
from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app

BACKEND_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = BACKEND_ROOT / "tests" / "fixtures" / "xtp"
APPEND_ONLY_TABLES = frozenset({"alembic_version", "audit_events", "consent_records"})


def run_alembic(database_url: str, *args: str) -> None:
    """Runs an Alembic CLI command against the given PostgreSQL database URL."""
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_ROOT,
        env=env,
        check=True,
    )


def pg_dsn(database_url: str) -> str:
    """Converts SQLAlchemy psycopg URL to a plain libpq DSN for psycopg.connect."""
    return database_url.replace("+psycopg", "")


def fetch_scalar(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] = ()) -> Any:
    """Returns the first column of the first row from a psycopg query, or None."""
    row = conn.execute(sql, params).fetchone()
    return row[0] if row is not None else None


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def migrated_db() -> str:
    url = os.environ.get("XUNA_TEST_DATABASE_URL", "")
    if not url:
        pytest.skip("XUNA_TEST_DATABASE_URL not set")
    run_alembic(url, "upgrade", "head")
    return url


@pytest.fixture
def clean_mesh_db(migrated_db: str) -> str:
    """Truncates mutable tables while preserving append-only audit/consent triggers."""
    dsn = pg_dsn(migrated_db)
    for attempt in range(4):
        try:
            with psycopg.connect(dsn) as conn:
                rows = conn.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """
                ).fetchall()
                mutable = [row[0] for row in rows if row[0] not in APPEND_ONLY_TABLES]
                if mutable:
                    quoted = ", ".join(f'"{name}"' for name in mutable)
                    conn.execute(f"TRUNCATE {quoted} CASCADE")
                conn.commit()
            return migrated_db
        except psycopg.errors.DeadlockDetected:
            if attempt == 3:
                raise
            time.sleep(0.05 * (attempt + 1))
    return migrated_db


@dataclass(frozen=True, slots=True)
class MeshServerHandle:
    base_url: str
    ws_url: str
    issuer: LocalIssuer
    database_url: str


@asynccontextmanager
async def start_mesh_server(
    database_url: str,
    *,
    generator: Any = None,
    recognizer: Any = None,
    synthesizer: Any = None,
    dev_host_tools: bool = True,
) -> AsyncIterator[MeshServerHandle]:
    """Starts a real in-process uvicorn HTTP/SSE/WS server on an ephemeral localhost port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = int(sock.getsockname()[1])

    issuer = LocalIssuer()
    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url=database_url,
            dev_host_tools=dev_host_tools,
        ),
        verifier=issuer.verifier(),
        generator=generator,
        recognizer=recognizer,
        synthesizer=synthesizer,
    )
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    serve_task = asyncio.create_task(server.serve(sockets=[sock]))

    deadline = asyncio.get_running_loop().time() + 5.0
    try:
        while not server.started:
            if serve_task.done():
                serve_task.result()
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Timed out waiting for uvicorn mesh server startup")
            await asyncio.sleep(0.01)

        yield MeshServerHandle(
            base_url=f"http://127.0.0.1:{port}",
            ws_url=f"ws://127.0.0.1:{port}",
            issuer=issuer,
            database_url=database_url,
        )
    finally:
        server.should_exit = True
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(serve_task, timeout=5.0)
        if not serve_task.done():
            serve_task.cancel()
            with contextlib.suppress(BaseException):
                await serve_task
        with contextlib.suppress(OSError):
            sock.close()


@pytest.fixture
def mesh_server_factory(
    clean_mesh_db: str,
) -> Callable[..., Any]:
    """Returns `start_mesh_server` bound to the cleaned per-test PostgreSQL database."""

    def _factory(
        *,
        generator: Any = None,
        recognizer: Any = None,
        synthesizer: Any = None,
        dev_host_tools: bool = True,
    ) -> Any:
        return start_mesh_server(
            clean_mesh_db,
            generator=generator,
            recognizer=recognizer,
            synthesizer=synthesizer,
            dev_host_tools=dev_host_tools,
        )

    return _factory
