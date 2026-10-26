"""Integration API tests for suggestions and sleep learning against PostgreSQL (CLD-30, CLD-31, H1-H8)."""

import base64
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from tests.helpers import LocalIssuer, SimulatedDevice, b64, idem

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.identity.domain.device_proof import ProofPurpose, proof_message
from xuna_backend.modules.identity.domain.pet_id import PetId

pytestmark = pytest.mark.postgres
ROOT = Path(__file__).resolve().parents[3]
PET = "d945c77026fd4f4a8a899bbff4aeed86"
TABLES = (
    "cloud_suggestions, deletion_tombstones, backup_manifests, memory_candidates, memory_projections, "
    "idempotency_keys, device_challenges, devices, brains, pets, owners"
)


@pytest.fixture
def issuer() -> LocalIssuer:
    return LocalIssuer()


@pytest.fixture
def client(database_url: str, issuer: LocalIssuer) -> Iterator[TestClient]:
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    with psycopg.connect(database_url.replace("+psycopg", "")) as connection:
        connection.execute(f"TRUNCATE {TABLES} CASCADE")
    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url=database_url,
        ),
        verifier=issuer.verifier(),
    )
    with TestClient(app) as test_client:
        yield test_client


def auth(issuer: LocalIssuer, subject: str = "owner-1") -> dict[str, str]:
    return {"Authorization": f"Bearer {issuer.token(subject)}"}


def bind_test_pet(
    client: TestClient, issuer: LocalIssuer, device: SimulatedDevice
) -> dict[str, Any]:
    c_res = client.post(
        "/v1/devices/challenges",
        json={"schemaVersion": 1, "petId": PET},
        headers={**auth(issuer), **idem()},
    )
    assert c_res.status_code == 201, c_res.text
    challenge = c_res.json()
    msg = proof_message(
        base64.b64decode(challenge["nonce"]),
        PetId(PET),
        ProofPurpose.BIND,
        1,
        device.public_key,
    )
    sig = device.sign(msg)
    bind_res = client.post(
        "/v1/pets",
        json={
            "schemaVersion": 1,
            "petId": PET,
            "origin": "hatched_on_device",
            "snapshotGeneration": 1,
            "deviceProof": {
                "challengeId": challenge["challengeId"],
                "publicKey": b64(device.public_key),
                "signature": b64(sig),
            },
        },
        headers={**auth(issuer), **idem()},
    )
    assert bind_res.status_code == 201, bind_res.text
    return bind_res.json()


def test_consolidation_requires_u1_or_u2_consent(
    client: TestClient, issuer: LocalIssuer
) -> None:
    device = SimulatedDevice()
    bind_test_pet(client, issuer, device)

    # 1. Under default U0: consolidate fails with 403 Forbidden (consent_required)
    res = client.post(
        f"/v1/pets/{PET}/suggestions/consolidate",
        headers={**auth(issuer), **idem()},
    )
    assert res.status_code == 403
    assert res.json()["code"] == "consent_required"

    # 2. Grant U1 consent (aggregate_metrics)
    consent_res = client.put(
        "/v1/consents",
        json={
            "schemaVersion": 1,
            "aggregateMetrics": True,
            "experienceEvents": False,
            "conversationText": False,
            "expectedVersion": 0,
        },
        headers={**auth(issuer), **idem()},
    )
    assert consent_res.status_code == 200

    # 3. Consolidate succeeds under U1
    key = idem()
    res2 = client.post(
        f"/v1/pets/{PET}/suggestions/consolidate",
        headers={**auth(issuer), **key},
    )
    assert res2.status_code == 200
    body = res2.json()
    assert body["count"] == 3
    assert len(body["items"]) == 3

    # Idempotent retry returns identical payload
    res2_retry = client.post(
        f"/v1/pets/{PET}/suggestions/consolidate",
        headers={**auth(issuer), **key},
    )
    assert res2_retry.status_code == 200
    assert res2_retry.json() == body


def test_suggestions_lifecycle_and_crud(
    client: TestClient, issuer: LocalIssuer
) -> None:
    device = SimulatedDevice()
    bind_test_pet(client, issuer, device)

    # Grant U1 consent
    client.put(
        "/v1/consents",
        json={
            "schemaVersion": 1,
            "aggregateMetrics": True,
            "experienceEvents": False,
            "conversationText": False,
            "expectedVersion": 0,
        },
        headers={**auth(issuer), **idem()},
    )

    # Consolidate
    res = client.post(
        f"/v1/pets/{PET}/suggestions/consolidate",
        headers={**auth(issuer), **idem()},
    )
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 3

    # List all
    list_res = client.get(
        f"/v1/pets/{PET}/suggestions",
        headers=auth(issuer),
    )
    assert list_res.status_code == 200
    assert len(list_res.json()["items"]) == 3

    # Filter by kind
    filtered_res = client.get(
        f"/v1/pets/{PET}/suggestions?kind=learner_parameter",
        headers=auth(issuer),
    )
    assert filtered_res.status_code == 200
    assert len(filtered_res.json()["items"]) == 2

    # Get single suggestion
    first = items[0]
    suggestion_id = first["suggestionId"]
    get_res = client.get(
        f"/v1/pets/{PET}/suggestions/{suggestion_id}",
        headers=auth(issuer),
    )
    assert get_res.status_code == 200
    assert get_res.json()["suggestionId"] == suggestion_id

    # Apply suggestion
    patch_res = client.patch(
        f"/v1/pets/{PET}/suggestions/{suggestion_id}",
        json={"action": "apply"},
        headers={**auth(issuer), **idem()},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["status"] == "applied"
    assert patch_res.json()["appliedAt"] is not None

    # Revert applied suggestion
    revert_res = client.patch(
        f"/v1/pets/{PET}/suggestions/{suggestion_id}",
        json={"action": "revert"},
        headers={**auth(issuer), **idem()},
    )
    assert revert_res.status_code == 200
    assert revert_res.json()["status"] == "reverted"

    # Delete suggestion
    del_res = client.delete(
        f"/v1/pets/{PET}/suggestions/{suggestion_id}",
        headers=auth(issuer),
    )
    assert del_res.status_code == 204

    # Fetch deleted returns 404
    get_after_del = client.get(
        f"/v1/pets/{PET}/suggestions/{suggestion_id}",
        headers=auth(issuer),
    )
    assert get_after_del.status_code == 404
