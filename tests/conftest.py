import asyncio
import os
import sys
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app

if sys.platform == "win32":
    # psycopg's async mode cannot run on Windows' default ProactorEventLoop.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

TEST_DATABASE_URL = os.environ.get("XUNA_TEST_DATABASE_URL", "")
# Unreachable on purpose: app tests must not need a database unless marked.
OFFLINE_URL = "postgresql+psycopg://xuna:unused@127.0.0.1:1/none"


@pytest.fixture
def offline_client() -> Iterator[TestClient]:
    app = create_app(Settings(environment=Environment.TEST, database_url=OFFLINE_URL))
    with TestClient(app) as client:
        yield client


@pytest.fixture
def database_url() -> str:
    if not TEST_DATABASE_URL:
        pytest.skip("XUNA_TEST_DATABASE_URL not set")
    return TEST_DATABASE_URL
