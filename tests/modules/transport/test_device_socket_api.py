"""`/v1/devices/connect` end to end against PostgreSQL (DC-3, contract "Device connection")."""

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect
from tests.helpers import LocalIssuer, SimulatedDevice, b64, idem

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.identity.domain.device_proof import ProofPurpose, proof_message
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.transport.domain.device_session import (
    GatewayLimits,
    SlidingWindowLimiter,
)

pytestmark = pytest.mark.postgres
ROOT = Path(__file__).resolve().parents[3]
PET = "d945c77026fd4f4a8a899bbff4aeed86"
TABLES = (
    "device_transport_statuses, cloud_suggestions, deletion_tombstones, backup_manifests, "
    "memory_candidates, memory_projections, idempotency_keys, device_challenges, "
    "devices, brains, pets, owners"
)
PROTOCOL = ["xuna.device.v1"]


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


class Owner:
    def __init__(self, client: TestClient, issuer: LocalIssuer) -> None:
        self.client, self.issuer = client, issuer
        self.subject = f"owner-{uuid.uuid4()}"
        self.device = SimulatedDevice()
        self.device_id = ""

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.issuer.token(self.subject)}"}

    def bind(self) -> "Owner":
        challenge = self.client.post(
            "/v1/devices/challenges",
            json={"schemaVersion": 1, "petId": PET},
            headers={**self.headers(), **idem()},
        ).json()
        message = proof_message(
            base64.b64decode(challenge["nonce"]),
            PetId(PET),
            ProofPurpose.BIND,
            1,
            self.device.public_key,
        )
        response = self.client.post(
            "/v1/pets",
            json={
                "schemaVersion": 1,
                "petId": PET,
                "origin": "hatched_on_device",
                "snapshotGeneration": 1,
                "deviceProof": {
                    "challengeId": challenge["challengeId"],
                    "publicKey": b64(self.device.public_key),
                    "signature": b64(self.device.sign(message)),
                },
            },
            headers={**self.headers(), **idem()},
        )
        assert response.status_code == 201, response.text
        self.device_id = response.json()["deviceId"]
        return self

    def consent(self, on: bool, expected_version: int = 0) -> "Owner":
        response = self.client.put(
            "/v1/consents",
            json={
                "schemaVersion": 1,
                "aggregateMetrics": on,
                "experienceEvents": False,
                "conversationText": False,
                "expectedVersion": expected_version,
            },
            headers={**self.headers(), **idem()},
        )
        assert response.status_code == 200, response.text
        return self

    def status(self) -> dict[str, Any]:
        response = self.client.get(
            f"/v1/devices/{self.device_id}/transport/status", headers=self.headers()
        )
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        return body

    def audit_actions(self) -> list[str]:
        items = self.client.get("/v1/audit-events?limit=20", headers=self.headers()).json()
        return [item["action"] for item in items["items"]]


def hello(ws: WebSocketTestSession, pet: str = PET) -> bytes:
    ws.send_text(
        json.dumps(
            {
                "type": "hello",
                "schemaVersion": 1,
                "petId": pet,
                "firmware": "0.2.0-mvp",
                "capabilities": ["projection.v1"],
            }
        )
    )
    challenge = ws.receive_json()
    assert challenge["type"] == "challenge" and challenge["expiresIn"] == 60
    return base64.b64decode(challenge["nonce"])


def prove(
    ws: WebSocketTestSession, device: SimulatedDevice, nonce: bytes, generation: int = 40
) -> None:
    message = proof_message(nonce, PetId(PET), ProofPurpose.SESSION, generation, device.public_key)
    ws.send_text(
        json.dumps(
            {
                "type": "proof",
                "schemaVersion": 1,
                "generation": generation,
                "signature": b64(device.sign(message)),
            }
        )
    )


def closed_with(ws: WebSocketTestSession) -> tuple[int, str]:
    with pytest.raises(WebSocketDisconnect) as closed:
        ws.receive_text()
    return closed.value.code, closed.value.reason


def connect_ready(client: TestClient, owner: Owner) -> WebSocketTestSession:
    ws = client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL).__enter__()
    prove(ws, owner.device, hello(ws))
    ready = ws.receive_json()
    assert ready["type"] == "ready" and ready["heartbeatS"] == 30
    return ws


def test_handshake_heartbeat_and_online_status(client: TestClient, issuer: LocalIssuer) -> None:
    owner = Owner(client, issuer).bind().consent(True)
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        assert ws.accepted_subprotocol == "xuna.device.v1"
        prove(ws, owner.device, hello(ws))
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["serverTime"].endswith("+00:00")
        ws.send_text('{"type":"ping","schemaVersion":1,"seq":7}')
        assert ws.receive_json() == {"type": "pong", "schemaVersion": 1, "seq": 7}
        status = owner.status()
        assert status["status"] == "online" and status["firmware"] == "0.2.0-mvp"
        assert status["connectedAt"] is not None and status["ipAddress"] is None
    deadline = time.monotonic() + 5
    while owner.status()["status"] != "offline":
        assert time.monotonic() < deadline, "session not marked offline"
        time.sleep(0.05)
    assert "device.connected" in owner.audit_actions()


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("wrong_key", (1008, "unauthorized")),
        ("consent_off", (1008, "consent_required")),
        ("not_bound", (1008, "not_bound")),
        ("ping_before_proof", (1008, "unauthorized")),
        ("binary", (1002, "unsupported_protocol")),
        ("unknown_type", (1002, "unsupported_protocol")),
        ("oversize", (1009, "frame_too_large")),
    ],
)
def test_handshake_rejections(
    client: TestClient, issuer: LocalIssuer, case: str, expected: tuple[int, str]
) -> None:
    owner = Owner(client, issuer)
    if case != "not_bound":
        owner.bind().consent(case != "consent_off")
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        if case == "binary":
            ws.send_bytes(b"\x00")
        elif case == "unknown_type":
            ws.send_text('{"type":"command","schemaVersion":1}')
        elif case == "oversize":
            ws.send_text(json.dumps({"type": "hello", "pad": "x" * 5000}))
        else:
            nonce = hello(ws)
            if case == "ping_before_proof":
                ws.send_text('{"type":"ping","schemaVersion":1,"seq":1}')
            else:
                prove(ws, SimulatedDevice() if case == "wrong_key" else owner.device, nonce)
        assert closed_with(ws) == expected
    if case in ("wrong_key", "consent_off"):
        assert "device.rejected" in owner.audit_actions()


def test_proof_from_an_earlier_session_is_refused(client: TestClient, issuer: LocalIssuer) -> None:
    owner = Owner(client, issuer).bind().consent(True)
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as first:
        old_nonce = hello(first)
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        hello(ws)
        prove(ws, owner.device, old_nonce)  # signed for another challenge
        assert closed_with(ws) == (1008, "unauthorized")


def test_missing_subprotocol_is_refused(client: TestClient) -> None:
    with client.websocket_connect("/v1/devices/connect") as ws:
        assert closed_with(ws) == (1002, "unsupported_protocol")


def test_hello_timeout_and_expired_challenge(client: TestClient, issuer: LocalIssuer) -> None:
    owner = Owner(client, issuer).bind().consent(True)
    client.app.state.device_gateway_limits = GatewayLimits(hello_timeout_s=0.2, challenge_ttl_s=0.2)
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        assert closed_with(ws) == (1008, "unauthorized")  # no hello in time
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "hello",
                    "schemaVersion": 1,
                    "petId": PET,
                    "firmware": "0.2.0-mvp",
                    "capabilities": [],
                }
            )
        )
        nonce = base64.b64decode(ws.receive_json()["nonce"])
        time.sleep(0.3)
        prove(ws, owner.device, nonce)
        assert closed_with(ws) == (1008, "unauthorized")


def test_challenge_rate_limit(client: TestClient, issuer: LocalIssuer) -> None:
    Owner(client, issuer).bind().consent(True)
    client.app.state.device_challenge_limiter = SlidingWindowLimiter(1, 600)
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        hello(ws)
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "hello",
                    "schemaVersion": 1,
                    "petId": PET,
                    "firmware": "0.2.0-mvp",
                    "capabilities": [],
                }
            )
        )
        assert closed_with(ws) == (1008, "rate_limited")


def test_newer_session_replaces_older(client: TestClient, issuer: LocalIssuer) -> None:
    owner = Owner(client, issuer).bind().consent(True)
    first = connect_ready(client, owner)
    second = connect_ready(client, owner)
    try:
        assert closed_with(first) == (4000, "replaced")
        second.send_text('{"type":"ping","schemaVersion":1,"seq":2}')
        assert second.receive_json()["seq"] == 2
        assert owner.status()["status"] == "online"  # the replaced one did not mark offline
    finally:
        second.__exit__(None, None, None)
        first.__exit__(None, None, None)


def test_consent_withdrawn_ends_live_session(client: TestClient, issuer: LocalIssuer) -> None:
    owner = Owner(client, issuer).bind().consent(True)
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        prove(ws, owner.device, hello(ws))
        assert ws.receive_json()["type"] == "ready"
        owner.consent(False, expected_version=1)
        ws.send_text('{"type":"ping","schemaVersion":1,"seq":3}')
        assert closed_with(ws) == (1008, "consent_required")


GOLDEN = ROOT / "tests" / "fixtures" / "xuna_export_v1_golden.json"


def envelope(generation: int = 38, tamper: bool = False) -> dict[str, Any]:
    """The Dart golden, re-hashed for another generation when asked."""
    document: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))
    payload = document["payload"]
    payload["identity"]["snapshotGeneration"] = generation
    compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    document["sha256"] = hashlib.sha256(compact.encode()).hexdigest()
    if tamper:
        payload["life"]["feedbackCount"] += 1
    return document


def send_projection(ws: WebSocketTestSession, generation: int, **kwargs: Any) -> dict[str, Any]:
    ws.send_text(
        json.dumps(
            {
                "type": "projection",
                "schemaVersion": 1,
                "generation": generation,
                "export": envelope(generation, **kwargs),
            }
        )
    )
    ack: dict[str, Any] = ws.receive_json()
    assert ack["type"] == "ack" and ack["ref"] == "projection"
    assert ack["generation"] == generation
    return ack


def test_projection_over_the_session_is_stored_once(
    client: TestClient, issuer: LocalIssuer
) -> None:
    owner = Owner(client, issuer).bind().consent(True)
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        prove(ws, owner.device, hello(ws))
        assert ws.receive_json()["type"] == "ready"
        assert send_projection(ws, 38)["result"] == "stored"
        assert send_projection(ws, 38)["result"] == "duplicate"
        assert send_projection(ws, 37)["result"] == "stale"
        assert send_projection(ws, 39, tamper=True)["result"] == "invalid"
        assert send_projection(ws, 40)["result"] == "stored"
        ws.send_text('{"type":"ping","schemaVersion":1,"seq":9}')
        assert ws.receive_json()["seq"] == 9  # the session lives on after a refusal
    latest = client.get(f"/v1/pets/{PET}/projections/latest", headers=owner.headers())
    assert latest.status_code == 200, latest.text
    body = latest.json()
    assert body["snapshotGeneration"] == 40 and body["deviceAttested"] is True


def test_projection_without_u1_is_acknowledged_as_consent_required(
    client: TestClient, issuer: LocalIssuer
) -> None:
    owner = Owner(client, issuer).bind()
    response = client.put(
        "/v1/consents",
        json={
            "schemaVersion": 1,
            "aggregateMetrics": False,
            "experienceEvents": True,
            "conversationText": False,
            "expectedVersion": 0,
        },
        headers={**owner.headers(), **idem()},
    )
    assert response.status_code == 200
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        prove(ws, owner.device, hello(ws))
        assert ws.receive_json()["type"] == "ready"  # U2 alone opens the session
        assert send_projection(ws, 38)["result"] == "consent_required"


def test_only_projections_may_exceed_four_kib(client: TestClient, issuer: LocalIssuer) -> None:
    owner = Owner(client, issuer).bind().consent(True)
    with client.websocket_connect("/v1/devices/connect", subprotocols=PROTOCOL) as ws:
        prove(ws, owner.device, hello(ws))
        assert ws.receive_json()["type"] == "ready"
        ws.send_text(json.dumps({"type": "ping", "schemaVersion": 1, "seq": 1, "pad": "x" * 5000}))
        assert closed_with(ws) == (1009, "frame_too_large")
