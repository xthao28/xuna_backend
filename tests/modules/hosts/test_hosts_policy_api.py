"""Tests for GET & PUT /v1/hosts/{host_id}/policy (UDC-18 Wave E M1)."""

import base64
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from tests.helpers import LocalIssuer, MemoryAuditLog, SimulatedDevice
from tests.modules.hosts.test_hosts_use_cases import (
    FakeEnrollmentRepository,
    FakeHostRepository,
)

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.application.computer_use import (
    InMemoryComputerUseSessionStore,
)
from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.hosts.domain.crypto import (
    HostProofPurpose,
    host_proof_message,
)
from xuna_backend.modules.hosts.domain.entities import (
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
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


class _FakeOwnerRepo:
    def __init__(self, owner: Owner) -> None:
        self._owner = owner

    async def get_or_create(self, claims: object) -> Owner:
        return self._owner


def _setup_app(
    owner_id: uuid.UUID,
    host_repo: FakeHostRepository,
    audit_log: MemoryAuditLog,
    issuer: LocalIssuer,
) -> tuple[TestClient, str]:
    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
        ),
        verifier=issuer.verifier(),
    )
    app.state.database = _DummyDatabase()
    app.state.cu_session_store = InMemoryComputerUseSessionStore()
    app.state.owner_repository = lambda _db: _FakeOwnerRepo(
        Owner(owner_id=owner_id, issuer="test", subject="owner-test")
    )
    app.state.host_repository = lambda _db: host_repo
    app.state.host_enrollment_repository = lambda _db: FakeEnrollmentRepository()
    app.state.audit_log = lambda _db: audit_log

    token = issuer.token("owner-test")
    client = TestClient(app)
    return client, token


@pytest.fixture(autouse=True)
def _patch_run_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import xuna_backend.modules.hosts.api.routes as host_routes

    async def _fake_run_once(
        _session: object,
        *,
        owner_id: object,
        key: object,
        fingerprint: str,
        now: datetime,
        action: object,
    ) -> object:
        return await action()  # type: ignore[operator]

    monkeypatch.setattr(host_routes, "run_once", _fake_run_once)


def test_get_policy_404_when_host_not_found_or_wrong_owner() -> None:
    owner_id = uuid.uuid4()
    other_owner_id = uuid.uuid4()
    host_repo = FakeHostRepository()
    audit_log = MemoryAuditLog()
    issuer = LocalIssuer()

    host_id = HostId.generate()
    now = datetime.now(UTC)
    host_repo.hosts[host_id.value] = Host(
        host_id=host_id,
        owner_id=other_owner_id,  # different owner
        alias=HostAlias("MacBook"),
        platform=HostPlatform.MACOS,
        hostname="macbook",
        os_version="14.0",
        agent_version="0.1.0",
        public_key=SimulatedDevice().public_key,
        status=HostStatus.ACTIVE,
        created_at=now,
        last_seen_at=now,
    )

    client, token = _setup_app(owner_id, host_repo, audit_log, issuer)
    with client:
        # 1. Non-existent host
        res = client.get(
            f"/v1/hosts/{uuid.uuid4()}/policy",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 404
        assert res.json()["code"] == "not_found"

        # 2. Host owned by another user
        res = client.get(
            f"/v1/hosts/{host_id}/policy",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 404
        assert res.json()["code"] == "not_found"


def test_get_policy_409_fail_closed_when_host_offline() -> None:
    owner_id = uuid.uuid4()
    host_repo = FakeHostRepository()
    audit_log = MemoryAuditLog()
    issuer = LocalIssuer()

    host_id = HostId.generate()
    now = datetime.now(UTC)
    host_repo.hosts[host_id.value] = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("Office PC"),
        platform=HostPlatform.WINDOWS,
        hostname="pc",
        os_version="11.0",
        agent_version="0.1.0",
        public_key=SimulatedDevice().public_key,
        status=HostStatus.ACTIVE,
        created_at=now,
        last_seen_at=now,
    )

    client, token = _setup_app(owner_id, host_repo, audit_log, issuer)
    with client:
        # Host is registered but has no active WebSocket session in hub
        res = client.get(
            f"/v1/hosts/{host_id}/policy",
            headers={"Authorization": f"Bearer {token}"},
        )
        # Fail-closed: 409 conflict host_offline
        assert res.status_code == 409
        assert res.json()["code"] == "host_offline"


def test_put_policy_409_fail_closed_when_host_offline() -> None:
    owner_id = uuid.uuid4()
    host_repo = FakeHostRepository()
    audit_log = MemoryAuditLog()
    issuer = LocalIssuer()

    host_id = HostId.generate()
    now = datetime.now(UTC)
    host_repo.hosts[host_id.value] = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("Office PC"),
        platform=HostPlatform.WINDOWS,
        hostname="pc",
        os_version="11.0",
        agent_version="0.1.0",
        public_key=SimulatedDevice().public_key,
        status=HostStatus.ACTIVE,
        created_at=now,
        last_seen_at=now,
    )

    client, token = _setup_app(owner_id, host_repo, audit_log, issuer)
    with client:
        res = client.put(
            f"/v1/hosts/{host_id}/policy",
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": str(uuid.uuid4()),
            },
            json={
                "schemaVersion": 1,
                "trustPreset": "standard",
                "allowApps": ["vscode"],
                "denyApps": [],
                "allowDomains": ["github.com"],
                "denyDomains": [],
            },
        )
        assert res.status_code == 409
        assert res.json()["code"] == "host_offline"


def test_get_and_put_policy_200_ok_when_host_online() -> None:
    """Verifies end-to-end get_policy & set_policy through connected WebSocket session."""
    owner_id = uuid.uuid4()
    host_repo = FakeHostRepository()
    audit_log = MemoryAuditLog()
    issuer = LocalIssuer()
    agent_key = SimulatedDevice()

    host_id = HostId.generate()
    now = datetime.now(UTC)
    host_repo.hosts[host_id.value] = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("Laptop Pro"),
        platform=HostPlatform.MACOS,
        hostname="macbook-pro",
        os_version="14.5",
        agent_version="0.1.0",
        public_key=agent_key.public_key,
        status=HostStatus.ACTIVE,
        created_at=now,
        last_seen_at=now,
    )

    client, token = _setup_app(owner_id, host_repo, audit_log, issuer)
    with (
        client,
        client.websocket_connect("/v1/hosts/connect", subprotocols=SUBPROTOCOL) as ws,
    ):
        ws.send_text(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "xuna/hello",
                    "params": {
                        "protocols": ["xtp/1"],
                        "hostId": str(host_id),
                        "platform": "macos",
                        "osVersion": "14.5",
                        "agentVersion": "0.1.0",
                    },
                }
            )
        )
        assert json.loads(ws.receive_text())["result"]["protocol"] == "xtp/1"

        challenge = json.loads(ws.receive_text())
        nonce = base64.b64decode(challenge["params"]["nonce"])
        proof_msg = host_proof_message(
            nonce=nonce,
            purpose=HostProofPurpose.SESSION,
            payload=agent_key.public_key,
            host_id=host_id,
        )
        sig_b64 = base64.b64encode(agent_key.sign(proof_msg)).decode("ascii")
        ws.send_text(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "xuna/proof",
                    "params": {"purpose": 2, "signature": sig_b64},
                }
            )
        )
        assert json.loads(ws.receive_text())["result"]["verified"] is True
        assert json.loads(ws.receive_text())["method"] == "xuna/ready"

        tools_req = json.loads(ws.receive_text())
        assert tools_req["method"] == "tools/list"
        ws.send_text(
            json.dumps({"jsonrpc": "2.0", "id": tools_req["id"], "result": {"tools": []}})
        )

        hub = client.app.state.host_session_hub
        assert hub.is_online(str(host_id)) is True


def test_put_policy_422_when_policy_blocked() -> None:
    """Verifies that when desktop returns policy_blocked, backend translates it to 422."""
    import asyncio

    from xuna_backend.modules.hosts.application.use_cases import HostError, set_host_policy

    owner_id = uuid.uuid4()
    host_id = HostId.generate()
    host_repo = FakeHostRepository()
    now = datetime.now(UTC)
    host_repo.hosts[host_id.value] = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("Laptop Pro"),
        platform=HostPlatform.MACOS,
        hostname="macbook-pro",
        os_version="14.5",
        agent_version="0.1.0",
        public_key=SimulatedDevice().public_key,
        status=HostStatus.ACTIVE,
        created_at=now,
        last_seen_at=now,
    )

    class _MockHub:
        def is_online(self, hid: str) -> bool:
            return True

        async def send_request(self, hid: str, method: str, params: dict, timeout_s: float):
            err = "rpc_error:-32602:policy_blocked: Application 'Terminal' is restricted"
            raise RuntimeError(err)

    audit_log = MemoryAuditLog()

    with pytest.raises(HostError) as exc_info:
        asyncio.run(
            set_host_policy(
                hosts=host_repo,
                hub=_MockHub(),  # type: ignore[arg-type]
                audit=audit_log,
                owner_id=owner_id,
                host_id=host_id,
                policy_data={"trustPreset": "standard", "allowApps": ["Terminal"]},
                now=now,
            )
        )
    assert exc_info.value.code == "policy_blocked"
    assert "Terminal" in exc_info.value.detail


def test_set_host_policy_success_and_audit() -> None:
    """Verifies set_host_policy success records AuditAction.HOST_POLICY_UPDATED."""
    from xuna_backend.modules.hosts.application.use_cases import set_host_policy

    owner_id = uuid.uuid4()
    host_id = HostId.generate()
    host_repo = FakeHostRepository()
    now = datetime.now(UTC)
    host_repo.hosts[host_id.value] = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("Laptop Pro"),
        platform=HostPlatform.MACOS,
        hostname="macbook-pro",
        os_version="14.5",
        agent_version="0.1.0",
        public_key=SimulatedDevice().public_key,
        status=HostStatus.ACTIVE,
        created_at=now,
        last_seen_at=now,
    )

    class _MockHub:
        def is_online(self, hid: str) -> bool:
            return True

        async def send_request(self, hid: str, method: str, params: dict, timeout_s: float):
            return {"schemaVersion": 1, "status": "ok"}

    audit_log = MemoryAuditLog()
    import asyncio
    result = asyncio.run(
        set_host_policy(
            hosts=host_repo,
            hub=_MockHub(),  # type: ignore[arg-type]
            audit=audit_log,
            owner_id=owner_id,
            host_id=host_id,
            policy_data={
                "trustPreset": "autopilot",
                "allowApps": ["vscode", "notes"],
                "denyApps": [],
                "allowDomains": ["github.com"],
                "denyDomains": [],
            },
            now=now,
            request_id="req-123",
        )
    )
    assert result == {"schemaVersion": 1, "status": "ok"}
    assert len(audit_log.events) == 1
    ev = audit_log.events[0]
    assert ev.action == AuditAction.HOST_POLICY_UPDATED
    assert ev.subject == str(host_id)
    assert ev.request_id == "req-123"
    assert ev.metadata["trustPreset"] == "autopilot"
    assert ev.metadata["allowAppsCount"] == 2
