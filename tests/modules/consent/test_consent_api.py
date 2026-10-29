"""CLD-12 against PostgreSQL: U0 default, consent changes, guard, audit trail,
append-only tables."""

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.helpers import LocalIssuer, idem
from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.consent.api.routes import RequireConsent
from xuna_backend.modules.consent.domain.consent import ConsentClass

pytestmark = pytest.mark.postgres
ROOT = Path(__file__).resolve().parents[3]
TABLES = ", ".join(
    (
        "audit_events",
        "consent_records",
        "idempotency_keys",
        "device_challenges",
        "devices",
        "brains",
        "pets",
        "owners",
    )
)


def _raw(url: str) -> str:
    return url.replace("+psycopg", "")


@pytest.fixture
def issuer() -> LocalIssuer:
    return LocalIssuer()


@pytest.fixture
def client(database_url: str, issuer: LocalIssuer) -> Iterator[TestClient]:
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
    )
    with psycopg.connect(_raw(database_url)) as connection:
        connection.execute("ALTER TABLE audit_events DISABLE TRIGGER audit_events_append_only")
        connection.execute(
            "ALTER TABLE consent_records DISABLE TRIGGER consent_records_append_only"
        )
        connection.execute(f"TRUNCATE {TABLES} CASCADE")
        connection.execute("ALTER TABLE audit_events ENABLE TRIGGER audit_events_append_only")
        connection.execute("ALTER TABLE consent_records ENABLE TRIGGER consent_records_append_only")
    app: FastAPI = create_app(
        Settings(environment=Environment.TEST, database_url=database_url),
        verifier=issuer.verifier(),
    )

    @app.post("/test/u2-upload")
    async def guarded(
        owner: Annotated[Owner, RequireConsent(ConsentClass.EXPERIENCE_EVENTS)],
    ) -> dict[str, str]:
        return {"accepted": str(owner.owner_id)}

    with TestClient(app) as test_client:
        yield test_client


def auth(issuer: LocalIssuer, subject: str = "owner-1") -> dict[str, str]:
    return {"Authorization": f"Bearer {issuer.token(subject)}"}


def body(version: int, **flags: bool) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "aggregateMetrics": flags.get("u1", False),
        "experienceEvents": flags.get("u2", False),
        "conversationText": flags.get("u3", False),
        "expectedVersion": version,
    }


def test_u0_default_and_guard(client: TestClient, issuer: LocalIssuer) -> None:
    headers = auth(issuer)
    assert client.get("/v1/consents", headers=headers).json() == {
        "aggregateMetrics": False,
        "experienceEvents": False,
        "conversationText": False,
        "conversationVoice": False,
        "ownerDocuments": False,
        "version": 0,
        "updatedAt": None,
    }
    denied = client.post("/test/u2-upload", headers=headers)
    assert (denied.status_code, denied.json()["code"]) == (403, "consent_required")


def test_opt_in_enables_only_that_class_and_is_audited(
    client: TestClient, issuer: LocalIssuer
) -> None:
    headers = auth(issuer)
    key = idem()
    updated = client.put("/v1/consents", json=body(0, u2=True), headers={**headers, **key})
    assert updated.status_code == 200, updated.text
    assert updated.json()["version"] == 1
    replay = client.put("/v1/consents", json=body(0, u2=True), headers={**headers, **key})
    assert replay.json() == updated.json()
    assert client.post("/test/u2-upload", headers=headers).status_code == 200
    stale = client.put("/v1/consents", json=body(0), headers={**headers, **idem()})
    assert (stale.status_code, stale.json()["code"]) == (409, "consent_version_conflict")
    withdrawn = client.put("/v1/consents", json=body(1), headers={**headers, **idem()})
    assert withdrawn.json()["version"] == 2
    assert client.post("/test/u2-upload", headers=headers).status_code == 403
    trail = client.get("/v1/audit-events", headers=headers).json()["items"]
    assert [(e["action"], e["metadata"]) for e in trail] == [
        ("consent.changed", {"version": 2, "experienceevents": False}),
        ("consent.changed", {"version": 1, "experienceevents": True}),
    ]
    assert client.get("/v1/audit-events", headers=auth(issuer, "other")).json() == {"items": []}


def test_voice_opt_in_survives_a_put_without_the_field(
    client: TestClient, issuer: LocalIssuer
) -> None:
    headers = auth(issuer, "voice-owner")
    on = client.put(
        "/v1/consents",
        json={**body(0, u3=True), "conversationVoice": True},
        headers={**headers, **idem()},
    )
    assert on.status_code == 200, on.text
    assert on.json()["conversationVoice"] is True
    legacy = client.put("/v1/consents", json=body(1), headers={**headers, **idem()})
    assert legacy.json()["conversationVoice"] is True
    assert legacy.json()["conversationText"] is False
    bad = client.put(
        "/v1/consents",
        json={**body(2), "conversationVoice": "yes"},
        headers={**headers, **idem()},
    )
    assert bad.status_code == 422


def test_owner_documents_opt_in_survives_put_without_field(
    client: TestClient, issuer: LocalIssuer
) -> None:
    headers = auth(issuer, "docs-owner")
    on = client.put(
        "/v1/consents",
        json={**body(0, u3=True), "ownerDocuments": True},
        headers={**headers, **idem()},
    )
    assert on.status_code == 200, on.text
    assert on.json()["ownerDocuments"] is True
    legacy = client.put("/v1/consents", json=body(1), headers={**headers, **idem()})
    assert legacy.json()["ownerDocuments"] is True
    assert legacy.json()["conversationText"] is False


def test_consent_is_per_owner(client: TestClient, issuer: LocalIssuer) -> None:
    client.put("/v1/consents", json=body(0, u3=True), headers={**auth(issuer), **idem()})
    assert client.get("/v1/consents", headers=auth(issuer, "other")).json()["version"] == 0


def test_history_tables_are_append_only(
    client: TestClient, issuer: LocalIssuer, database_url: str
) -> None:
    client.put("/v1/consents", json=body(0, u1=True), headers={**auth(issuer), **idem()})
    with psycopg.connect(_raw(database_url)) as connection:
        for statement in (
            "UPDATE consent_records SET aggregate_metrics = false",
            "DELETE FROM consent_records",
            "UPDATE audit_events SET action = 'x'",
            "DELETE FROM audit_events",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                connection.execute(statement)
            connection.rollback()
