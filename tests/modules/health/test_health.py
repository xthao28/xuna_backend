import pytest
from fastapi.testclient import TestClient

from xuna_backend.core.config import Environment, Settings
from xuna_backend.core.observability import REQUEST_ID_HEADER
from xuna_backend.main import create_app


def test_healthz_needs_no_dependencies(offline_client: TestClient) -> None:
    response = offline_client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_request_id_is_echoed_only_when_well_formed(offline_client: TestClient) -> None:
    good = offline_client.get("/healthz", headers={REQUEST_ID_HEADER: "abcdef12-3456"})
    assert good.headers[REQUEST_ID_HEADER] == "abcdef12-3456"
    bad = offline_client.get("/healthz", headers={REQUEST_ID_HEADER: "x\r\ninjected"})
    assert bad.headers[REQUEST_ID_HEADER] != "x\r\ninjected"


def test_readyz_reports_problem_when_database_down(offline_client: TestClient) -> None:
    response = offline_client.get("/readyz")
    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["code"] == "database_unavailable"
    assert body["status"] == 503


def test_unknown_route_uses_framework_404(offline_client: TestClient) -> None:
    assert offline_client.get("/nope").status_code == 404


@pytest.mark.postgres
def test_readyz_with_postgres(database_url: str) -> None:
    app = create_app(Settings(environment=Environment.TEST, database_url=database_url))
    with TestClient(app) as client:
        assert client.get("/readyz").json() == {"status": "ready"}
