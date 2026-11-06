"""HTTP routes for standing approvals (ADR-0014 D1): create, list, revoke, error codes.

Runs without PostgreSQL: repositories are in-memory fakes behind `app.state` factories, the
phone signs with a real P-256 key and the production `P256SignatureVerifier` checks it.
"""

import asyncio
import base64
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from fastapi.testclient import TestClient
from tests.helpers import LocalIssuer, MemoryAuditLog

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.infrastructure.standing_repository import (
    InMemoryStandingApprovalRepository,
)
from xuna_backend.modules.hosts.domain.crypto import args_sha256, preapproval_proof_message
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)

NOW = datetime.now(UTC)


class _DummySession:
    async def commit(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def flush(self) -> None:
        pass


class _DummyDatabase:
    @asynccontextmanager
    async def sessions(self) -> AsyncIterator[_DummySession]:
        yield _DummySession()

    async def dispose(self) -> None:
        pass


class _OwnerRepo:
    def __init__(self, owner: Owner) -> None:
        self._owner = owner

    async def get_or_create(self, claims: Any) -> Owner:
        return self._owner


class _HostRepo:
    def __init__(self) -> None:
        self.hosts: dict[HostId, Host] = {}
        self.manifests: dict[HostId, Manifest] = {}
        self.grants: dict[HostId, list[Grant]] = {}

    async def get_host(self, host_id: HostId) -> Host | None:
        return self.hosts.get(host_id)

    async def save_host(self, host: Host) -> None:
        self.hosts[host.host_id] = host

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        return [h for h in self.hosts.values() if h.owner_id == owner_id]

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self.grants.get(host_id, []))

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self.grants[host_id] = list(grants)

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifests.get(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        self.manifests[manifest.host_id] = manifest


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


class _Harness:
    def __init__(self) -> None:
        self.issuer = LocalIssuer()
        self.owner = Owner(
            owner_id=uuid.uuid4(), issuer="https://auth.test.xuna", subject="owner-1"
        )
        app = create_app(
            Settings(
                environment=Environment.TEST,
                database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
            ),
            verifier=self.issuer.verifier(),
        )
        self.hosts = _HostRepo()
        self.standing = InMemoryStandingApprovalRepository()
        self.audit = MemoryAuditLog()
        app.state.database = _DummyDatabase()
        app.state.owner_repository = lambda _s: _OwnerRepo(self.owner)
        app.state.host_repository = lambda _s: self.hosts
        app.state.standing_approval_repository = lambda _s: self.standing
        app.state.audit_log = lambda _s: self.audit
        self.client = TestClient(app)
        self.headers = {"Authorization": f"Bearer {self.issuer.token('owner-1')}"}

        self.private = ec.generate_private_key(ec.SECP256R1())
        self.public = self.private.public_key().public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )
        phone = self._host("Pixel", HostPlatform.ANDROID, self.public)
        self.pc = self._host("PC", HostPlatform.WINDOWS, b"\x04" + bytes(range(1, 65)))
        launch = ToolDescriptor(
            name="app.launch",
            description="Launch an allowlisted app.",
            input_schema={
                "type": "object",
                "properties": {"app_id": {"type": "string", "enum": ["com.spotify.music"]}},
                "required": ["app_id"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.APP,
            risk=RiskTier.R2,
        )
        asyncio.run(self.hosts.save_host(phone))
        asyncio.run(self.hosts.save_host(self.pc))
        asyncio.run(
            self.hosts.save_manifest(
                Manifest(host_id=self.pc.host_id, tools=(launch,), updated_at=NOW)
            )
        )

    def _host(self, alias: str, platform: HostPlatform, key: bytes) -> Host:
        return Host(
            host_id=HostId.generate(),
            owner_id=self.owner.owner_id,
            alias=HostAlias(alias),
            platform=platform,
            hostname=f"{alias.lower()}-host",
            os_version="1",
            agent_version="1.0.0",
            public_key=key,
            status=HostStatus.ACTIVE,
            created_at=NOW,
            last_seen_at=NOW,
        )

    def body(
        self,
        *,
        approval_id: uuid.UUID | None = None,
        args: dict[str, object] | None = None,
        tool: str = "app.launch",
        days: int = 7,
    ) -> dict[str, object]:
        approval_id = approval_id or uuid.uuid4()
        args = {"app_id": "com.spotify.music"} if args is None else args
        expires = int((datetime.now(UTC) + timedelta(days=days)).timestamp())
        message = preapproval_proof_message(
            str(approval_id), tool, args_sha256(args), str(self.pc.host_id), expires, 0
        )
        der = self.private.sign(message, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return {
            "schemaVersion": 1,
            "approvalId": str(approval_id),
            "hostId": str(self.pc.host_id),
            "tool": tool,
            "arguments": args,
            "expiresAtEpoch": expires,
            "signature": _b64(r.to_bytes(32, "big") + s.to_bytes(32, "big")),
            "publicKey": _b64(self.public),
        }


def test_create_list_and_revoke_round_trip() -> None:
    h = _Harness()
    body = h.body()

    created = h.client.post("/v1/actions/standing-approvals", json=body, headers=h.headers)
    assert created.status_code == 201, created.text
    item = created.json()
    assert item["approvalId"] == body["approvalId"]
    assert item["tool"] == "app.launch"
    assert item["target"] == "PC"
    assert "com.spotify.music" not in str(h.audit.events[0].metadata)

    again = h.client.post("/v1/actions/standing-approvals", json=body, headers=h.headers)
    assert again.status_code == 201
    assert again.json()["approvalId"] == item["approvalId"]

    listed = h.client.get("/v1/actions/standing-approvals", headers=h.headers).json()
    assert [i["approvalId"] for i in listed["items"]] == [item["approvalId"]]

    revoked = h.client.delete(
        f"/v1/actions/standing-approvals/{item['approvalId']}", headers=h.headers
    )
    assert revoked.status_code == 200
    assert h.client.get("/v1/actions/standing-approvals", headers=h.headers).json() == {"items": []}
    missing = h.client.delete(f"/v1/actions/standing-approvals/{uuid.uuid4()}", headers=h.headers)
    assert missing.status_code == 404
    assert missing.json()["code"] == "not_found"


def test_stable_error_codes_for_refused_requests() -> None:
    h = _Harness()

    bad_args = h.client.post(
        "/v1/actions/standing-approvals",
        json=h.body(args={"app_id": "com.evil.app"}),
        headers=h.headers,
    )
    assert bad_args.status_code == 422
    assert bad_args.json()["code"] == "invalid_args"

    unknown = h.client.post(
        "/v1/actions/standing-approvals",
        json=h.body(tool="media.play_pause"),
        headers=h.headers,
    )
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "unknown_tool"

    too_long = h.client.post(
        "/v1/actions/standing-approvals", json=h.body(days=45), headers=h.headers
    )
    assert too_long.status_code == 422
    assert too_long.json()["code"] == "invalid_expiry"

    tampered = h.body()
    tampered["tool"] = "app.launch"
    tampered["arguments"] = {"app_id": "com.spotify.music"}
    tampered["signature"] = _b64(bytes(64))
    bad_sig = h.client.post("/v1/actions/standing-approvals", json=tampered, headers=h.headers)
    assert bad_sig.status_code == 400
    assert bad_sig.json()["code"] == "signature_invalid"

    bad_b64 = h.body()
    bad_b64["signature"] = "!" * 90
    garbled = h.client.post("/v1/actions/standing-approvals", json=bad_b64, headers=h.headers)
    assert garbled.status_code == 400
    assert garbled.json()["code"] == "approval_signature_invalid"

    assert h.client.get("/v1/actions/standing-approvals", headers=h.headers).json() == {"items": []}


def test_requests_without_a_token_are_rejected() -> None:
    h = _Harness()
    assert h.client.get("/v1/actions/standing-approvals").status_code in (401, 403)
    assert h.client.post("/v1/actions/standing-approvals", json=h.body()).status_code in (401, 403)


def test_a_normal_approval_decision_cannot_claim_the_standing_scope() -> None:
    h = _Harness()
    response = h.client.post(
        f"/v1/actions/approvals/{uuid.uuid4()}",
        json={"schemaVersion": 1, "decision": "approve", "scope": "standing"},
        headers=h.headers,
    )
    assert response.status_code == 422
