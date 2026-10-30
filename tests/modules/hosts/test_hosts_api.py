"""API and WebSocket tests for `/v1/hosts` and `/v1/hosts/connect` (UDC-3 S3.2).

Includes:
1. Fake-backed `TestClient` WebSocket handshake through to `xuna/ready` and `tools/list`
   (runs in every environment without requiring PostgreSQL).
2. `@pytest.mark.postgres` end-to-end enrollment, claim, confirm, WebSocket session,
   grant update, dev tool invocation, and revocation against PostgreSQL.
"""

import base64
import json
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from tests.helpers import LocalIssuer, MemoryAuditLog, SimulatedDevice, b64, idem
from tests.modules.hosts.test_hosts_use_cases import (
    FakeEnrollmentRepository,
    FakeHostRepository,
)

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.hosts.domain.crypto import (
    HostProofPurpose,
    host_proof_message,
    public_key_fingerprint_b64url,
)
from xuna_backend.modules.hosts.domain.entities import (
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
)

ROOT = Path(__file__).resolve().parents[3]
TABLES = (
    "host_grants, host_manifests, host_enrollments, hosts, "
    "device_transport_statuses, cloud_suggestions, deletion_tombstones, backup_manifests, "
    "memory_candidates, memory_projections, idempotency_keys, device_challenges, "
    "devices, brains, pets, owners"
)
SUBPROTOCOL = ["xtp/1"]


class _DummyDbSession:
    async def commit(self) -> None:
        pass


class _DummyDatabase:
    @asynccontextmanager
    async def sessions(self) -> AsyncIterator[_DummyDbSession]:
        yield _DummyDbSession()

    async def dispose(self) -> None:
        pass


def _sample_tools_wire() -> list[dict[str, object]]:
    return [
        {
            "name": "media.play_pause",
            "description": "Toggle playback of the active media session.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["playing", "paused", "unknown"],
                    }
                },
                "required": ["state"],
                "additionalProperties": False,
            },
            "_meta": {
                "xuna/policy": {
                    "version": 1,
                    "family": "media",
                    "risk": "R1",
                    "observes": False,
                    "reversible": True,
                    "permission": "granted",
                }
            },
        }
    ]


def test_websocket_handshake_to_ready_and_tools_list_with_fake_agent() -> None:
    """Verifies `/v1/hosts/connect` handshake -> `xuna/ready` -> `tools/list` with a fake agent."""
    issuer = LocalIssuer()
    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
        ),
        verifier=issuer.verifier(),
    )
    host_repo = FakeHostRepository()
    enroll_repo = FakeEnrollmentRepository()
    audit_log = MemoryAuditLog()

    agent_key = SimulatedDevice()
    host_id = HostId.generate()
    owner_id = uuid.uuid4()
    now = datetime(2026, 10, 30, 9, 0, tzinfo=UTC)
    host_repo.hosts[host_id.value] = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("PC phòng làm việc"),
        platform=HostPlatform.WINDOWS,
        hostname="PC-THAO",
        os_version="10.0.22631",
        agent_version="0.1.0",
        public_key=agent_key.public_key,
        status=HostStatus.ACTIVE,
        created_at=now,
        last_seen_at=now,
    )

    with TestClient(app) as client:
        app.state.database = _DummyDatabase()
        app.state.host_repository = lambda _db: host_repo
        app.state.host_enrollment_repository = lambda _db: enroll_repo
        app.state.audit_log = lambda _db: audit_log

        with client.websocket_connect("/v1/hosts/connect", subprotocols=SUBPROTOCOL) as ws:
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "xuna/hello",
                        "params": {
                            "protocols": ["xtp/1"],
                            "hostId": str(host_id),
                            "platform": "windows",
                            "osVersion": "10.0.22631",
                            "agentVersion": "0.1.0",
                        },
                    }
                )
            )
            hello_resp = json.loads(ws.receive_text())
            assert hello_resp == {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocol": "xtp/1"},
            }

            challenge = json.loads(ws.receive_text())
            assert challenge["method"] == "xuna/challenge"
            assert challenge["params"]["purpose"] == 2
            nonce = base64.b64decode(challenge["params"]["nonce"])

            proof_msg = host_proof_message(
                nonce=nonce,
                purpose=HostProofPurpose.SESSION,
                payload=agent_key.public_key,
                host_id=host_id,
            )
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "xuna/proof",
                        "params": {
                            "purpose": 2,
                            "signature": b64(agent_key.sign(proof_msg)),
                        },
                    }
                )
            )
            proof_resp = json.loads(ws.receive_text())
            assert proof_resp == {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"verified": True},
            }

            ready_msg = json.loads(ws.receive_text())
            assert ready_msg["method"] == "xuna/ready"
            assert ready_msg["params"]["heartbeatS"] == 30

            tools_list_req = json.loads(ws.receive_text())
            assert tools_list_req["method"] == "tools/list"
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": tools_list_req["id"],
                        "result": {"tools": _sample_tools_wire()},
                    }
                )
            )

            # Send a ping to flush the manifest recording loop before closing
            ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping", "params": {}}))
            ping_resp = json.loads(ws.receive_text())
            assert ping_resp == {"jsonrpc": "2.0", "id": 99, "result": {}}

    saved_manifest = host_repo.manifests.get(host_id.value)
    assert saved_manifest is not None
    assert len(saved_manifest.tools) == 1
    assert saved_manifest.tools[0].name == "media.play_pause"


@pytest.fixture
def pg_issuer() -> LocalIssuer:
    return LocalIssuer()


@pytest.fixture
def pg_client(database_url: str, pg_issuer: LocalIssuer) -> Iterator[TestClient]:
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
            dev_host_tools=True,
        ),
        verifier=pg_issuer.verifier(),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.postgres
def test_postgres_host_lifecycle_routes_and_repositories(
    pg_client: TestClient,
    pg_issuer: LocalIssuer,
) -> None:
    headers = {"Authorization": f"Bearer {pg_issuer.token('owner-hosts-1')}"}
    other_headers = {"Authorization": f"Bearer {pg_issuer.token('owner-hosts-2')}"}
    agent_key = SimulatedDevice()
    code_bytes = bytes(range(1, 17))
    code_b64url = base64.urlsafe_b64encode(code_bytes).rstrip(b"=").decode("ascii")
    fp_b64url = public_key_fingerprint_b64url(agent_key.public_key)

    # 1. Seed pending enrollment via start_enrollment in DB
    import asyncio

    from xuna_backend.modules.hosts.application.use_cases import (
        StartEnrollmentCommand,
        start_enrollment,
    )

    nonce = bytes(range(32))
    enroll_sig = agent_key.sign(host_proof_message(nonce, HostProofPurpose.ENROLL, code_bytes))

    async def _seed_pending() -> None:
        app_state = pg_client.app.state  # type: ignore[attr-defined]
        async with app_state.database.sessions() as db:
            repo = app_state.host_enrollment_repository(db)
            verifier = app_state.signature_verifier
            await start_enrollment(
                repo,
                verifier,
                StartEnrollmentCommand(
                    code=code_bytes,
                    public_key=agent_key.public_key,
                    nonce=nonce,
                    signature=enroll_sig,
                    platform=HostPlatform.WINDOWS,
                    os_version="10.0.22631",
                    agent_version="0.1.0",
                    hostname="PC-PHONG-LAM-VIEC",
                ),
                datetime.now(UTC),
            )
            await db.commit()

    asyncio.run(_seed_pending())

    # 2. Owner claims enrollment (`POST /v1/hosts` -> 202 Accepted)
    claim_res = pg_client.post(
        "/v1/hosts",
        headers={**headers, **idem()},
        json={
            "schemaVersion": 1,
            "enrollmentCode": code_b64url,
            "keyFingerprint": fp_b64url,
            "alias": "PC phòng làm việc",
        },
    )
    assert claim_res.status_code == 202, claim_res.text
    claim_body: dict[str, Any] = claim_res.json()
    host_id = str(claim_body["hostId"])
    assert claim_body["pendingId"] == host_id
    assert len(claim_body["sas"]) == 6

    # 3. Owner confirms SAS (`POST /v1/hosts/{id}/confirm` -> 200 OK)
    confirm_res = pg_client.post(
        f"/v1/hosts/{host_id}/confirm",
        headers={**headers, **idem()},
        json={"schemaVersion": 1},
    )
    assert confirm_res.status_code == 200, confirm_res.text
    confirm_body = confirm_res.json()
    assert confirm_body["hostId"] == host_id
    assert confirm_body["alias"] == "PC phòng làm việc"
    assert confirm_body["status"] == "active"

    # 4. Enrolled host connects over WebSocket and registers `tools/list`
    with pg_client.websocket_connect("/v1/hosts/connect", subprotocols=SUBPROTOCOL) as ws:
        ws.send_text(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "xuna/hello",
                    "params": {
                        "protocols": ["xtp/1"],
                        "hostId": host_id,
                        "platform": "windows",
                        "osVersion": "10.0.22631",
                        "agentVersion": "0.1.0",
                    },
                }
            )
        )
        assert json.loads(ws.receive_text())["result"] == {"protocol": "xtp/1"}
        ch = json.loads(ws.receive_text())
        s_nonce = base64.b64decode(ch["params"]["nonce"])
        s_msg = host_proof_message(
            s_nonce,
            HostProofPurpose.SESSION,
            agent_key.public_key,
            HostId.from_str(host_id),
        )
        ws.send_text(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "xuna/proof",
                    "params": {
                        "purpose": 2,
                        "signature": b64(agent_key.sign(s_msg)),
                    },
                }
            )
        )
        assert json.loads(ws.receive_text())["result"] == {"verified": True}
        assert json.loads(ws.receive_text())["method"] == "xuna/ready"
        tl_req = json.loads(ws.receive_text())
        ws.send_text(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": tl_req["id"],
                    "result": {"tools": _sample_tools_wire()},
                }
            )
        )
        ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 10, "method": "ping", "params": {}}))
        assert json.loads(ws.receive_text())["id"] == 10

        # While connected, `GET /v1/hosts` reports `online: True` and stored manifest
        list_res = pg_client.get("/v1/hosts", headers=headers)
        assert list_res.status_code == 200
        items = list_res.json()["items"]
        assert len(items) == 1
        assert items[0]["online"] is True
        assert len(items[0]["tools"]) == 1

    # 5. Update grants (`PUT /v1/hosts/{id}/grants`)
    grants_res = pg_client.put(
        f"/v1/hosts/{host_id}/grants",
        headers={**headers, **idem()},
        json={
            "schemaVersion": 1,
            "grants": [
                {"family": "app", "maxRisk": "R2", "enabled": True},
                {"family": "url", "maxRisk": "R2", "enabled": True},
            ],
        },
    )
    assert grants_res.status_code == 200, grants_res.text

    # Another owner gets 404 not_found (never leaks existence)
    other_del = pg_client.delete(
        f"/v1/hosts/{host_id}",
        headers={**other_headers, **idem()},
    )
    assert other_del.status_code == 404

    # 6. Owner revokes host (`DELETE /v1/hosts/{id}` -> 204 No Content)
    del_res = pg_client.delete(
        f"/v1/hosts/{host_id}",
        headers={**headers, **idem()},
    )
    assert del_res.status_code == 204
    assert pg_client.get("/v1/hosts", headers=headers).json()["items"] == []
