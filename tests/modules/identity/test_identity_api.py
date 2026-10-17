"""End-to-end CLD-11 flow against PostgreSQL: auth → challenge → bind → read."""

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
TABLES = "idempotency_keys, device_challenges, devices, brains, pets, owners"


@pytest.fixture
def issuer() -> LocalIssuer:
    return LocalIssuer()


@pytest.fixture
def client(database_url: str, issuer: LocalIssuer) -> Iterator[TestClient]:
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
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


def bind_body(
    client: TestClient, headers: dict[str, str], device: SimulatedDevice, generation: int = 38
) -> dict[str, Any]:
    challenge = client.post(
        "/v1/devices/challenges",
        json={"schemaVersion": 1, "petId": PET},
        headers={**headers, **idem()},
    )
    assert challenge.status_code == 201, challenge.text
    data = challenge.json()
    message = proof_message(
        base64.b64decode(data["nonce"]),
        PetId(PET),
        ProofPurpose.BIND,
        generation,
        device.public_key,
    )
    return {
        "schemaVersion": 1,
        "petId": PET,
        "origin": "migrated_from_v1",
        "snapshotGeneration": generation,
        "deviceProof": {
            "challengeId": data["challengeId"],
            "publicKey": b64(device.public_key),
            "signature": b64(device.sign(message)),
        },
    }


def test_auth_required_and_me(client: TestClient, issuer: LocalIssuer) -> None:
    assert client.get("/v1/me").json()["code"] == "unauthorized"
    bad = client.get("/v1/me", headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    first = client.get("/v1/me", headers=auth(issuer)).json()["userId"]
    assert client.get("/v1/me", headers=auth(issuer)).json()["userId"] == first
    assert client.get("/v1/me", headers=auth(issuer, "owner-2")).json()["userId"] != first


def test_bind_list_get_and_idempotent_replay(client: TestClient, issuer: LocalIssuer) -> None:
    headers = auth(issuer)
    device = SimulatedDevice()
    body = bind_body(client, headers, device)
    key = idem()
    created = client.post("/v1/pets", json=body, headers={**headers, **key})
    assert created.status_code == 201, created.text
    pet = created.json()
    assert pet["petId"] == PET
    assert pet["lastGeneration"] == 38
    replay = client.post("/v1/pets", json=body, headers={**headers, **key})
    assert (replay.status_code, replay.json()) == (201, pet)
    reused = client.post(
        "/v1/pets", json={**body, "snapshotGeneration": 39}, headers={**headers, **key}
    )
    assert reused.json()["code"] == "idempotency_key_reused"
    again = client.post(
        "/v1/pets", json=bind_body(client, headers, device), headers={**headers, **idem()}
    )
    assert (again.status_code, again.json()["brainId"]) == (200, pet["brainId"])
    assert client.get("/v1/pets", headers=headers).json()["items"] == [pet]
    assert client.get(f"/v1/pets/{PET}", headers=headers).json() == pet


def test_other_owner_cannot_see_or_take_the_pet(client: TestClient, issuer: LocalIssuer) -> None:
    device = SimulatedDevice()
    owner = auth(issuer)
    assert (
        client.post(
            "/v1/pets", json=bind_body(client, owner, device), headers={**owner, **idem()}
        ).status_code
        == 201
    )
    intruder = auth(issuer, "intruder")
    assert client.get(f"/v1/pets/{PET}", headers=intruder).status_code == 404
    assert client.get("/v1/pets", headers=intruder).json() == {"items": []}
    taken = client.post(
        "/v1/pets", json=bind_body(client, intruder, device), headers={**intruder, **idem()}
    )
    assert (taken.status_code, taken.json()["code"]) == (409, "pet_already_bound")


def test_rejected_proof_still_consumes_the_challenge(
    client: TestClient, issuer: LocalIssuer
) -> None:
    headers = auth(issuer)
    body = bind_body(client, headers, SimulatedDevice())
    forged = {**body, "deviceProof": {**body["deviceProof"], "signature": b64(bytes(64))}}
    first = client.post("/v1/pets", json=forged, headers={**headers, **idem()})
    assert (first.status_code, first.json()["code"]) == (403, "invalid_signature")
    retry = client.post("/v1/pets", json=body, headers={**headers, **idem()})
    assert (retry.status_code, retry.json()["code"]) == (409, "challenge_invalid")


def test_validation_and_missing_idempotency_key(client: TestClient, issuer: LocalIssuer) -> None:
    headers = auth(issuer)
    missing = client.post(
        "/v1/devices/challenges", json={"schemaVersion": 1, "petId": PET}, headers=headers
    )
    assert missing.json()["code"] == "idempotency_key_required"
    bad = client.post(
        "/v1/devices/challenges",
        json={"schemaVersion": 1, "petId": PET.upper()},
        headers={**headers, **idem()},
    )
    assert (bad.status_code, bad.json()["code"]) == (422, "validation_failed")
    assert PET.upper() not in bad.text
