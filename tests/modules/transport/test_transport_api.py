"""Integration tests for Direct Device Transport API (CLD-40, CLD-41, E14)."""

import base64
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

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
    "device_transport_statuses, cloud_suggestions, deletion_tombstones, backup_manifests, "
    "memory_candidates, memory_projections, idempotency_keys, device_challenges, "
    "devices, brains, pets, owners"
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


def _bind_pet(client: TestClient, issuer: LocalIssuer, device: SimulatedDevice) -> str:
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
    resp = client.post(
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
    assert resp.status_code == 201, resp.text
    data = resp.json()
    return str(data["deviceId"])


def test_device_transport_lifecycle(client: TestClient, issuer: LocalIssuer) -> None:
    headers = auth(issuer)
    device = SimulatedDevice()
    device_id = _bind_pet(client, issuer, device)

    # 1. Device reports online status via MQTT uplink webhook
    status_topic = f"xuna/v1/{device_id}/status"
    resp = client.post(
        "/v1/transport/mqtt/uplink",
        json={
            "topic": status_topic,
            "payload": {"status": "online"},
            "clientId": "xuna-esp32-01",
            "ipAddress": "192.168.1.50",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["handled"] is True

    # 2. Query device status from API
    get_resp = client.get(f"/v1/devices/{device_id}/transport/status", headers=headers)
    assert get_resp.status_code == 200
    status_data = get_resp.json()
    assert status_data["deviceId"] == device_id
    assert status_data["status"] == "online"
    assert status_data["ipAddress"] == "192.168.1.50"

    # 3. Send command down to device
    cmd_resp = client.post(
        f"/v1/devices/{device_id}/transport/command",
        headers=headers,
        json={"action": "reboot", "args": {"reason": "test"}},
    )
    assert cmd_resp.status_code == 200
    cmd_data = cmd_resp.json()
    assert cmd_data["action"] == "reboot"
    assert cmd_data["status"] == "published"
    assert cmd_data["topic"] == f"xuna/v1/{device_id}/down/cmd"

    # 4. Device reports offline status (LWT)
    resp = client.post(
        "/v1/transport/mqtt/uplink",
        json={"topic": status_topic, "payload": {"status": "offline"}},
    )
    assert resp.status_code == 200

    get_resp2 = client.get(f"/v1/devices/{device_id}/transport/status", headers=headers)
    assert get_resp2.status_code == 200
    assert get_resp2.json()["status"] == "offline"
