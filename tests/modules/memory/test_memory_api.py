"""Integration API tests for memory projections and candidates against PostgreSQL."""

import base64
import json
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
GOLDEN = ROOT / "tests" / "fixtures" / "xuna_export_v1_golden.json"
TABLES = (
    "memory_candidates, memory_projections, idempotency_keys, "
    "device_challenges, devices, brains, pets, owners"
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
        Settings(environment=Environment.TEST, database_url=database_url),
        verifier=issuer.verifier(),
    )
    with TestClient(app) as test_client:
        yield test_client


def auth(issuer: LocalIssuer, subject: str = "owner-1") -> dict[str, str]:
    return {"Authorization": f"Bearer {issuer.token(subject)}"}


def golden_envelope() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def bind_test_pet(client: TestClient, headers: dict[str, str], device: SimulatedDevice) -> None:
    challenge = client.post(
        "/v1/devices/challenges",
        json={"schemaVersion": 1, "petId": PET},
        headers={**headers, **idem()},
    )
    data = challenge.json()
    message = proof_message(
        base64.b64decode(data["nonce"]),
        PetId(PET),
        ProofPurpose.BIND,
        37,
        device.public_key,
    )
    client.post(
        "/v1/pets",
        json={
            "schemaVersion": 1,
            "petId": PET,
            "origin": "migrated_from_v1",
            "snapshotGeneration": 37,
            "deviceProof": {
                "challengeId": data["challengeId"],
                "publicKey": b64(device.public_key),
                "signature": b64(device.sign(message)),
            },
        },
        headers={**headers, **idem()},
    )


def test_put_projection_consent_enforcement_and_success(
    client: TestClient, issuer: LocalIssuer
) -> None:
    device = SimulatedDevice()
    headers = auth(issuer, "owner-proj-1")
    bind_test_pet(client, headers, device)

    # 1. Without U1 consent -> 403 consent_required
    resp = client.put(
        f"/v1/pets/{PET}/projections/38",
        json={"schemaVersion": 1, "export": golden_envelope()},
        headers={**headers, **idem()},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "consent_required"

    # 2. Grant U1 only (aggregateMetrics)
    consent_resp = client.put(
        "/v1/consents",
        json={
            "schemaVersion": 1,
            "aggregateMetrics": True,
            "experienceEvents": False,
            "conversationText": False,
            "expectedVersion": 0,
        },
        headers={**headers, **idem()},
    )
    assert consent_resp.status_code == 200

    # 3. PUT projection succeeds (201 Created)
    put_resp = client.put(
        f"/v1/pets/{PET}/projections/38",
        json={"schemaVersion": 1, "export": golden_envelope()},
        headers={**headers, **idem()},
    )
    assert put_resp.status_code == 201
    body = put_resp.json()
    assert body["snapshotGeneration"] == 38
    assert body["deviceAttested"] is False
    # memories must be stripped because experienceEvents=False
    assert body["payload"]["memories"] == []

    # 4. GET latest projection
    latest_resp = client.get(f"/v1/pets/{PET}/projections/latest", headers=headers)
    assert latest_resp.status_code == 200
    assert latest_resp.json()["snapshotGeneration"] == 38

    # 5. GET projections list
    list_resp = client.get(f"/v1/pets/{PET}/projections", headers=headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()["items"]) == 1


def test_put_projection_device_proof(client: TestClient, issuer: LocalIssuer) -> None:
    device = SimulatedDevice()
    headers = auth(issuer, "owner-proof-1")
    bind_test_pet(client, headers, device)

    client.put(
        "/v1/consents",
        json={
            "schemaVersion": 1,
            "aggregateMetrics": True,
            "experienceEvents": True,
            "conversationText": False,
            "expectedVersion": 0,
        },
        headers={**headers, **idem()},
    )

    # Issue challenge for projection
    challenge = client.post(
        "/v1/devices/challenges",
        json={"schemaVersion": 1, "petId": PET},
        headers={**headers, **idem()},
    ).json()

    envelope = golden_envelope()
    compact = json.dumps(envelope["payload"], separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    msg = proof_message(
        base64.b64decode(challenge["nonce"]),
        PetId(PET),
        ProofPurpose.PROJECTION,
        38,
        compact,
    )
    sig = device.sign(msg)

    put_resp = client.put(
        f"/v1/pets/{PET}/projections/38",
        json={
            "schemaVersion": 1,
            "export": envelope,
            "deviceProof": {
                "challengeId": challenge["challengeId"],
                "publicKey": b64(device.public_key),
                "signature": b64(sig),
            },
        },
        headers={**headers, **idem()},
    )
    assert put_resp.status_code == 201
    assert put_resp.json()["deviceAttested"] is True
    # experienceEvents was True so memories must be present
    assert len(put_resp.json()["payload"]["memories"]) == 2


def test_candidates_api_flow(client: TestClient, issuer: LocalIssuer) -> None:
    device = SimulatedDevice()
    headers = auth(issuer, "owner-cand-1")
    bind_test_pet(client, headers, device)

    # 1. Create candidate
    create_resp = client.post(
        "/v1/memory-candidates",
        json={
            "schemaVersion": 1,
            "petId": PET,
            "source": "conversation",
            "candidateType": "stated_preference",
            "content": {"action": "happy", "polarity": "like", "context": "play"},
        },
        headers={**headers, **idem()},
    )
    assert create_resp.status_code == 201
    cand = create_resp.json()
    assert cand["status"] == "awaiting_owner_confirmation"
    cand_id = cand["candidateId"]

    # 2. List candidates
    list_resp = client.get(f"/v1/memory-candidates?petId={PET}", headers=headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()["items"]) == 1

    # 3. Confirm candidate
    patch_resp = client.patch(
        f"/v1/memory-candidates/{cand_id}",
        json={"schemaVersion": 1, "action": "confirm"},
        headers={**headers, **idem()},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "awaiting_device_commit"

    # 4. Commit candidate
    commit_resp = client.patch(
        f"/v1/memory-candidates/{cand_id}",
        json={"schemaVersion": 1, "action": "commit", "committedGeneration": 39},
        headers={**headers, **idem()},
    )
    assert commit_resp.status_code == 200
    assert commit_resp.json()["status"] == "confirmed"
    assert commit_resp.json()["committedGeneration"] == 39
