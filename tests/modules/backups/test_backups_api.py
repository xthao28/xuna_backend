"""Integration API tests for backups, restore, and tombstones against PostgreSQL (CLD-20)."""

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
    "deletion_tombstones, backup_manifests, memory_candidates, memory_projections, "
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
            backup_encryption_key="test-key-32-chars-long-for-db!!",
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


def test_backup_lifecycle_and_restore_same_device(client: TestClient, issuer: LocalIssuer) -> None:
    device = SimulatedDevice()
    bound = bind_test_pet(client, issuer, device)
    initial_brain_id = bound["brainId"]
    initial_device_id = bound["deviceId"]

    export_payload = {
        "schemaVersion": 1,
        "payload": {
            "identity": {"petId": PET, "snapshotGeneration": 5},
            "life": {"activeSeconds": 600},
            "memories": [
                {"context": "quiet", "action": "idle", "approved": True},
                {"context": "play", "action": "happy", "approved": True},
            ],
        },
    }

    # 1. Create backup
    create_res = client.post(
        f"/v1/pets/{PET}/backups",
        json={"schemaVersion": 1, "snapshotGeneration": 5, "export": export_payload},
        headers={**auth(issuer), **idem()},
    )
    assert create_res.status_code == 201, create_res.text
    backup = create_res.json()
    backup_id = backup["backupId"]
    assert backup["snapshotGeneration"] == 5
    assert backup["brainId"] == initial_brain_id

    # 2. List backups
    list_res = client.get(f"/v1/pets/{PET}/backups", headers=auth(issuer))
    assert list_res.status_code == 200
    assert len(list_res.json()["items"]) == 1

    # 3. Get backup by ID
    get_res = client.get(f"/v1/pets/{PET}/backups/{backup_id}", headers=auth(issuer))
    assert get_res.status_code == 200
    assert get_res.json()["backupId"] == backup_id

    # 4. Restore backup on same device
    restore_res = client.post(
        f"/v1/pets/{PET}/backups/{backup_id}/restore",
        json={"schemaVersion": 1},
        headers={**auth(issuer), **idem()},
    )
    assert restore_res.status_code == 200, restore_res.text
    restored = restore_res.json()
    assert restored["petId"] == PET
    assert restored["brainId"] == initial_brain_id
    assert restored["deviceId"] == initial_device_id
    assert restored["tombstonesApplied"] == 0
    assert len(restored["restoredPayload"]["payload"]["memories"]) == 2

    # 5. Delete backup
    del_res = client.delete(f"/v1/pets/{PET}/backups/{backup_id}", headers=auth(issuer))
    assert del_res.status_code == 204
    get_after_del = client.get(f"/v1/pets/{PET}/backups/{backup_id}", headers=auth(issuer))
    assert get_after_del.status_code == 404


def test_restore_to_replacement_device(client: TestClient, issuer: LocalIssuer) -> None:
    old_device = SimulatedDevice()
    bound = bind_test_pet(client, issuer, old_device)
    brain_id = bound["brainId"]
    old_device_id = bound["deviceId"]

    # Create backup
    export_payload = {
        "schemaVersion": 1,
        "payload": {
            "identity": {"petId": PET, "snapshotGeneration": 8},
        },
    }
    create_res = client.post(
        f"/v1/pets/{PET}/backups",
        json={"schemaVersion": 1, "snapshotGeneration": 8, "export": export_payload},
        headers={**auth(issuer), **idem()},
    )
    assert create_res.status_code == 201
    backup_id = create_res.json()["backupId"]

    # Replacement device arrives!
    new_device = SimulatedDevice()
    c_res = client.post(
        "/v1/devices/challenges",
        json={"schemaVersion": 1, "petId": PET},
        headers={**auth(issuer), **idem()},
    )
    assert c_res.status_code == 201
    challenge = c_res.json()

    msg = proof_message(
        base64.b64decode(challenge["nonce"]),
        PetId(PET),
        ProofPurpose.BIND,
        8,
        new_device.public_key,
    )
    sig = new_device.sign(msg)

    restore_res = client.post(
        f"/v1/pets/{PET}/backups/{backup_id}/restore",
        json={
            "schemaVersion": 1,
            "deviceProof": {
                "challengeId": challenge["challengeId"],
                "publicKey": b64(new_device.public_key),
                "signature": b64(sig),
            },
        },
        headers={**auth(issuer), **idem()},
    )
    assert restore_res.status_code == 200, restore_res.text
    restored = restore_res.json()

    # Crucial acceptance check: PetId and BrainId preserved; DeviceId rebound!
    assert restored["petId"] == PET
    assert restored["brainId"] == brain_id
    assert restored["deviceId"] != old_device_id


def test_tombstones_prevent_memory_resurrection(client: TestClient, issuer: LocalIssuer) -> None:
    device = SimulatedDevice()
    bind_test_pet(client, issuer, device)

    export_payload = {
        "schemaVersion": 1,
        "payload": {
            "identity": {"petId": PET, "snapshotGeneration": 12},
            "memories": [
                {"context": "quiet", "action": "idle", "approved": True},
                {"context": "play", "action": "happy", "approved": True},
            ],
        },
    }
    create_res = client.post(
        f"/v1/pets/{PET}/backups",
        json={"schemaVersion": 1, "snapshotGeneration": 12, "export": export_payload},
        headers={**auth(issuer), **idem()},
    )
    backup_id = create_res.json()["backupId"]

    # Owner creates tombstone for "play"
    tomb_res = client.post(
        f"/v1/pets/{PET}/tombstones",
        json={
            "schemaVersion": 1,
            "itemType": "memory",
            "itemId": "play",
            "reason": "forgot_play_habit",
        },
        headers={**auth(issuer), **idem()},
    )
    assert tomb_res.status_code == 201

    # List tombstones
    list_tomb = client.get(f"/v1/pets/{PET}/tombstones", headers=auth(issuer))
    assert list_tomb.status_code == 200
    assert len(list_tomb.json()["items"]) == 1

    # Restore: "play" memory must NOT resurrect!
    restore_res = client.post(
        f"/v1/pets/{PET}/backups/{backup_id}/restore",
        json={"schemaVersion": 1},
        headers={**auth(issuer), **idem()},
    )
    assert restore_res.status_code == 200
    restored = restore_res.json()
    assert restored["tombstonesApplied"] == 1
    memories = restored["restoredPayload"]["payload"]["memories"]
    assert len(memories) == 1
    assert memories[0]["context"] == "quiet"


def test_forget_all_rotates_brain(client: TestClient, issuer: LocalIssuer) -> None:
    device = SimulatedDevice()
    bound = bind_test_pet(client, issuer, device)
    old_brain_id = bound["brainId"]

    forget_res = client.post(
        f"/v1/pets/{PET}/forget",
        headers={**auth(issuer), **idem()},
    )
    assert forget_res.status_code == 200, forget_res.text
    forget_data = forget_res.json()
    assert forget_data["petId"] == PET
    assert forget_data["newBrainId"] != old_brain_id


def test_cross_tenant_access_denied(client: TestClient, issuer: LocalIssuer) -> None:
    device = SimulatedDevice()
    bind_test_pet(client, issuer, device)

    # Owner 1 creates backup
    create_res = client.post(
        f"/v1/pets/{PET}/backups",
        json={"schemaVersion": 1, "snapshotGeneration": 1, "export": {"schemaVersion": 1}},
        headers={**auth(issuer, subject="owner-1"), **idem()},
    )
    assert create_res.status_code == 201
    backup_id = create_res.json()["backupId"]

    # Owner 2 tries to access or restore Owner 1's backup
    assert (
        client.get(f"/v1/pets/{PET}/backups", headers=auth(issuer, subject="owner-2")).status_code
        == 404
    )
    assert (
        client.get(
            f"/v1/pets/{PET}/backups/{backup_id}", headers=auth(issuer, subject="owner-2")
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/v1/pets/{PET}/backups/{backup_id}/restore",
            json={"schemaVersion": 1},
            headers={**auth(issuer, subject="owner-2"), **idem()},
        ).status_code
        == 404
    )
