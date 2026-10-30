"""Tests for UDC-5 S5.2: phone self-enrollment (`/v1/hosts/self*`), manifest update,
and same-device execution (`local_action_request` + `/v1/actions/{id}/result` with `output`).
"""

import asyncio
import base64
import time
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from tests.helpers import LocalIssuer, MemoryAuditLog, SimulatedDevice
from tests.modules.hosts.test_hosts_api import _DummyDatabase
from tests.modules.hosts.test_hosts_use_cases import (
    FakeEnrollmentRepository,
    FakeHostRepository,
)

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.actions.application.catalog import (
    CAP_HOST_ACTIONS_V1,
    load_tool_catalog_for_owner,
)
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    Orchestrator,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionResultEvent,
    ActionStreamEvent,
    ApprovalDecision,
    ApprovalRequest,
    LocalActionRequestEvent,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.cognition.domain.types import Locale, Role, ToolCall
from xuna_backend.modules.hosts.application.use_cases import (
    HostError,
    SelfEnrollHostCommand,
    self_enroll_host,
    set_grants,
    update_host_manifest,
)
from xuna_backend.modules.hosts.domain.crypto import (
    HostProofPurpose,
    decode_b64url_16b,
    host_proof_message,
)
from xuna_backend.modules.hosts.domain.entities import (
    HostId,
    HostPlatform,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)
from xuna_backend.modules.hosts.domain.session import (
    HostRateLimiter,
    SelfEnrollmentChallengeStore,
)
from xuna_backend.modules.hosts.infrastructure.session_hub import InMemoryHostSessionHub
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier


class _FakeApprovalRepository:
    def __init__(self) -> None:
        self.items: dict[uuid.UUID, ApprovalRequest] = {}

    async def save(self, req: ApprovalRequest) -> None:
        self.items[req.approval_id] = req

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        return self.items.get(approval_id)


def _android_tools_wire() -> list[dict[str, object]]:
    return [
        {
            "name": "media.play_pause",
            "description": "Phát hoặc tạm dừng nhạc trên điện thoại.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["toggle", "play", "pause"],
                    }
                },
                "additionalProperties": False,
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["playing", "paused", "unknown"],
                    },
                    "package": {"type": "string"},
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
                    "platform": "android",
                }
            },
        },
        {
            "name": "app.launch",
            "description": "Mở ứng dụng đã cho phép trên điện thoại.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "enum": [
                            "com.spotify.music",
                            "com.google.android.apps.youtube.music",
                        ],
                    }
                },
                "required": ["id"],
                "additionalProperties": False,
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "launched": {"type": "boolean"},
                    "id": {"type": "string"},
                },
                "required": ["launched", "id"],
                "additionalProperties": False,
            },
            "_meta": {
                "xuna/policy": {
                    "version": 1,
                    "family": "app",
                    "risk": "R2",
                    "observes": False,
                    "reversible": False,
                    "permission": "granted",
                    "platform": "android",
                }
            },
        },
    ]


def test_android_tool_descriptor_preserves_platform_in_wire_format() -> None:
    raw = _android_tools_wire()[0]
    descriptor = ToolDescriptor.from_wire(raw)
    assert descriptor.platform == "android"
    wire = descriptor.to_wire()
    meta = wire["_meta"]
    assert isinstance(meta, dict)
    policy = meta["xuna/policy"]
    assert isinstance(policy, dict)
    assert policy["platform"] == "android"


def test_self_enroll_host_and_idempotent_update_and_manifest_refresh() -> None:
    async def _run() -> None:
        host_repo = FakeHostRepository()
        verifier = P256SignatureVerifier()
        audit = MemoryAuditLog()
        limiter = HostRateLimiter(5, 60.0)
        challenge_store = SelfEnrollmentChallengeStore(60.0)
        owner_id = uuid.uuid4()
        now = datetime.now(UTC)
        mono = time.monotonic()

        phone_key = SimulatedDevice()
        nonce = b"\x11" * 32
        code = b"\x22" * 16
        challenge_store.issue(str(owner_id), nonce, code, mono)

        proof_msg = host_proof_message(
            nonce=nonce,
            purpose=HostProofPurpose.ENROLL,
            payload=code,
        )
        signature = phone_key.sign(proof_msg)

        host, grants, manifest = await self_enroll_host(
            host_repo,
            verifier,
            audit,
            limiter,
            challenge_store,
            owner_id,
            SelfEnrollHostCommand(
                platform=HostPlatform.ANDROID,
                os_version="Android 14 (SDK 34)",
                agent_version="0.1.0",
                alias="Pixel 8",
                public_key=phone_key.public_key,
                signature=signature,
                hostname="Pixel-8",
                tools=_android_tools_wire(),
            ),
            now,
            mono,
        )

        assert host.platform == HostPlatform.ANDROID
        assert host.alias.value == "Pixel 8"
        assert manifest is not None
        assert len(manifest.tools) == 2
        assert any(g.family == ToolFamily.MEDIA and g.enabled for g in grants)
        assert any(e.action == AuditAction.HOST_ENROLLED for e in audit.events)

        # Re-enrolling with the same public key updates alias & preserves host_id
        nonce2 = b"\x33" * 32
        code2 = b"\x44" * 16
        challenge_store.issue(str(owner_id), nonce2, code2, mono + 1.0)
        sig2 = phone_key.sign(
            host_proof_message(nonce=nonce2, purpose=HostProofPurpose.ENROLL, payload=code2)
        )
        host2, _, _ = await self_enroll_host(
            host_repo,
            verifier,
            audit,
            limiter,
            challenge_store,
            owner_id,
            SelfEnrollHostCommand(
                platform=HostPlatform.ANDROID,
                os_version="Android 14 (SDK 34)",
                agent_version="0.1.1",
                alias="Pixel 8 Pro",
                public_key=phone_key.public_key,
                signature=sig2,
                tools=_android_tools_wire()[:1],
            ),
            now,
            mono + 1.0,
        )
        assert host2.host_id == host.host_id
        assert host2.alias.value == "Pixel 8 Pro"

        # Manifest refresh without re-enrolling
        updated_manifest = await update_host_manifest(
            host_repo,
            owner_id,
            host.host_id,
            _android_tools_wire(),
            now,
        )
        assert len(updated_manifest.tools) == 2

    asyncio.run(_run())


def test_self_enroll_rejects_invalid_signature() -> None:
    async def _run() -> None:
        host_repo = FakeHostRepository()
        verifier = P256SignatureVerifier()
        audit = MemoryAuditLog()
        limiter = HostRateLimiter(5, 60.0)
        challenge_store = SelfEnrollmentChallengeStore(60.0)
        owner_id = uuid.uuid4()
        now = datetime.now(UTC)
        mono = time.monotonic()

        phone_key = SimulatedDevice()
        other_key = SimulatedDevice()
        nonce = b"\x55" * 32
        code = b"\x66" * 16
        challenge_store.issue(str(owner_id), nonce, code, mono)

        bad_sig = other_key.sign(
            host_proof_message(nonce=nonce, purpose=HostProofPurpose.ENROLL, payload=code)
        )

        with pytest.raises(HostError) as exc_info:
            await self_enroll_host(
                host_repo,
                verifier,
                audit,
                limiter,
                challenge_store,
                owner_id,
                SelfEnrollHostCommand(
                    platform=HostPlatform.ANDROID,
                    os_version="Android 14",
                    agent_version="0.1.0",
                    alias="Pixel 8",
                    public_key=phone_key.public_key,
                    signature=bad_sig,
                ),
                now,
                mono,
            )
        assert exc_info.value.code == "unauthorized"
        assert any(e.action == AuditAction.HOST_ENROLL_REJECTED for e in audit.events)

    asyncio.run(_run())


def test_same_device_execution_r1_and_r2_without_background_ws() -> None:
    """When `self_host_id` matches the enrolled Android phone, tools appear in the
    turn catalog and execute via `LocalActionRequestEvent` + `LocalResultBroker`
    with structured `output` even though the phone has no background WebSocket.
    Other clients (without `self_host_id`) see the phone as offline.
    """

    async def _run() -> None:
        host_repo = FakeHostRepository()
        verifier = P256SignatureVerifier()
        audit = MemoryAuditLog()
        limiter = HostRateLimiter(5, 60.0)
        challenge_store = SelfEnrollmentChallengeStore(60.0)
        hub = InMemoryHostSessionHub()
        owner_id = uuid.uuid4()
        now = datetime.now(UTC)
        mono = time.monotonic()

        phone_key = SimulatedDevice()
        nonce = b"\x77" * 32
        code = b"\x88" * 16
        challenge_store.issue(str(owner_id), nonce, code, mono)
        sig = phone_key.sign(
            host_proof_message(nonce=nonce, purpose=HostProofPurpose.ENROLL, payload=code)
        )

        host, _, _ = await self_enroll_host(
            host_repo,
            verifier,
            audit,
            limiter,
            challenge_store,
            owner_id,
            SelfEnrollHostCommand(
                platform=HostPlatform.ANDROID,
                os_version="Android 14",
                agent_version="0.1.0",
                alias="Điện thoại",
                public_key=phone_key.public_key,
                signature=sig,
                tools=_android_tools_wire(),
            ),
            now,
            mono,
        )
        # Enable R2 grant for `app`
        await set_grants(
            host_repo,
            audit,
            owner_id,
            host.host_id,
            [(ToolFamily.APP, RiskTier.R2, True)],
            now,
        )

        # 1. Without `self_host_id` and without WS, catalog is empty
        other_client_catalog = await load_tool_catalog_for_owner(
            owner_id=owner_id,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            host_repo=host_repo,
            hub=hub,
            self_host_id=None,
        )
        assert other_client_catalog == ()

        # 2. With `self_host_id`, catalog includes both `media__play_pause` and `app__launch`
        same_device_catalog = await load_tool_catalog_for_owner(
            owner_id=owner_id,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            host_repo=host_repo,
            hub=hub,
            self_host_id=str(host.host_id),
        )
        tool_names = {t.name for t in same_device_catalog}
        assert tool_names == {"media__play_pause", "app__launch"}

        approval_repo = _FakeApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_result_broker = InMemoryLocalResultBroker()
        last_target_store = InMemoryLastTargetStore()

        orch = Orchestrator(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_result_broker,
            last_target_store=last_target_store,
            audit=audit,
            clock=lambda: now,
        )

        emitted: list[ActionStreamEvent] = []

        async def on_event(ev: ActionStreamEvent) -> None:
            emitted.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner_id,
                    decision=ApprovalDecision.APPROVE,
                )
            elif isinstance(ev, LocalActionRequestEvent):
                if ev.tool == "media.play_pause":
                    local_result_broker.report(
                        invocation_id=ev.invocation_id,
                        owner_id=owner_id,
                        status="ok",
                        output={"state": "paused", "package": "com.spotify.music"},
                    )
                elif ev.tool == "app.launch":
                    local_result_broker.report(
                        invocation_id=ev.invocation_id,
                        owner_id=owner_id,
                        status="ok",
                        output={"launched": True, "id": "com.spotify.music"},
                    )

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=on_event,
            self_host_id=str(host.host_id),
        )

        # Execute R1 `media__play_pause` on same device
        msg1 = await orch.handle_call(
            ToolCall(id="call-1", name="media__play_pause", arguments={"mode": "pause"}),
            ctx,
        )
        assert msg1.role == Role.TOOL
        assert '"status":"ok"' in msg1.content
        assert '"package":"com.spotify.music"' in msg1.content
        assert any(
            isinstance(e, LocalActionRequestEvent) and e.tool == "media.play_pause"
            for e in emitted
        )
        assert any(
            isinstance(e, ActionResultEvent)
            and e.status == "ok"
            and e.gesture == "bounce_play"
            for e in emitted
        )

        # Execute R2 `app__launch` on same device (approval first, then local_action_request)
        emitted.clear()
        msg2 = await asyncio.wait_for(
            orch.handle_call(
                ToolCall(
                    id="call-2",
                    name="app__launch",
                    arguments={"id": "com.spotify.music"},
                ),
                ctx,
            ),
            timeout=2.0,
        )
        assert msg2.role == Role.TOOL
        assert '"status":"ok"' in msg2.content
        assert '"launched":true' in msg2.content
        assert isinstance(emitted[0], ActionPendingEvent)
        assert isinstance(emitted[1], LocalActionRequestEvent)
        assert emitted[1].approval_id == emitted[0].approval_id
        assert isinstance(emitted[2], ActionResultEvent)
        assert emitted[2].status == "ok"

    asyncio.run(_run())


def test_http_self_enroll_and_action_result_with_output() -> None:
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
    owner_id = uuid.uuid4()

    class _FakeOwnerRepo:
        async def get_or_create(self, claims: object) -> object:
            from xuna_backend.modules.accounts.domain.owner import Owner

            return Owner(
                owner_id=owner_id,
                issuer=str(getattr(claims, "issuer", "test")),
                subject=str(getattr(claims, "subject", "owner-sub-1")),
            )

    class _NoopIdempotencySession(_DummyDatabase):
        pass

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

    import xuna_backend.modules.hosts.api.routes as host_routes

    orig_run_once = host_routes.run_once
    host_routes.run_once = _fake_run_once  # type: ignore[assignment]

    app.state.database = _DummyDatabase()
    app.state.owner_repository = lambda _s: _FakeOwnerRepo()
    app.state.host_repository = lambda _s: host_repo
    app.state.host_enrollment_repository = lambda _s: enroll_repo
    app.state.audit_log = lambda _s: audit_log

    token = issuer.token("owner-sub-1")
    headers = {"Authorization": f"Bearer {token}"}
    phone_key = SimulatedDevice()

    with TestClient(app) as client:
        app.state.database = _DummyDatabase()
        ch_res = client.post("/v1/hosts/self/challenge", headers=headers)
        assert ch_res.status_code == 200
        ch_body = ch_res.json()
        raw_nonce = str(ch_body["nonce"])
        rem = len(raw_nonce) % 4
        padded_nonce = raw_nonce + ("=" * (4 - rem) if rem else "")
        nonce_bytes = base64.urlsafe_b64decode(padded_nonce.encode("ascii"))
        code_bytes = decode_b64url_16b(str(ch_body["enrollmentCode"]))

        proof_msg = host_proof_message(
            nonce=nonce_bytes,
            purpose=HostProofPurpose.ENROLL,
            payload=code_bytes,
        )
        sig_b64 = base64.urlsafe_b64encode(phone_key.sign(proof_msg)).rstrip(b"=").decode("ascii")
        pub_b64 = (
            base64.urlsafe_b64encode(phone_key.public_key).rstrip(b"=").decode("ascii")
        )

        enroll_res = client.post(
            "/v1/hosts/self",
            headers=headers,
            json={
                "schemaVersion": 1,
                "platform": "android",
                "osVersion": "Android 14",
                "agentVersion": "0.1.0",
                "alias": "Galaxy S24",
                "publicKey": pub_b64,
                "proof": sig_b64,
                "tools": _android_tools_wire(),
            },
        )
        assert enroll_res.status_code == 200
        enrolled = enroll_res.json()
        host_id_str = str(enrolled["hostId"])
        assert enrolled["platform"] == "android"
        assert enrolled["alias"] == "Galaxy S24"
        assert len(enrolled["tools"]) == 2

        # Update manifest via PUT /v1/hosts/{host_id}/manifest
        put_res = client.put(
            f"/v1/hosts/{host_id_str}/manifest",
            headers=headers,
            json={"schemaVersion": 1, "tools": _android_tools_wire()[:1]},
        )
        assert put_res.status_code == 200
        assert len(put_res.json()["tools"]) == 1

        # Verify POST /v1/actions/{invocation_id}/result accepts `output` and `errorCode`
        inv_id = str(uuid.uuid4())
        app.state.local_result_broker.register(inv_id, owner_id)
        res_post = client.post(
            f"/v1/actions/{inv_id}/result",
            headers=headers,
            json={
                "status": "ok",
                "output": {"state": "paused", "package": "com.spotify.music"},
            },
        )
        assert res_post.status_code == 200
        report = asyncio.run(app.state.local_result_broker.wait_report(inv_id, timeout_s=1.0))
        assert report.status == "ok"
        assert report.output == {"state": "paused", "package": "com.spotify.music"}

        # Verify S5.3 Mode B WebSocket (`/v1/hosts/connect`) for self-enrolled Android phone
        with client.websocket_connect(
            "/v1/hosts/connect",
            subprotocols=["xtp/1"],
        ) as ws:
            ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "xuna/hello",
                    "params": {
                        "protocols": ["xtp/1"],
                        "hostId": host_id_str,
                        "platform": "android",
                        "osVersion": "Android 14",
                        "agentVersion": "0.1.0",
                    },
                }
            )
            hello_ack = ws.receive_json()
            assert hello_ack == {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocol": "xtp/1"},
            }
            challenge_frame = ws.receive_json()
            assert challenge_frame["method"] == "xuna/challenge"
            assert challenge_frame["params"]["purpose"] == 2
            ws_nonce = base64.b64decode(challenge_frame["params"]["nonce"])
            session_proof_msg = host_proof_message(
                nonce=ws_nonce,
                purpose=HostProofPurpose.SESSION,
                payload=phone_key.public_key,
                host_id=HostId.from_str(host_id_str),
            )
            session_sig = base64.b64encode(phone_key.sign(session_proof_msg)).decode("ascii")
            ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "xuna/proof",
                    "params": {"purpose": 2, "signature": session_sig},
                }
            )
            proof_ack = ws.receive_json()
            assert proof_ack["result"] == {"verified": True}
            ready_frame = ws.receive_json()
            assert ready_frame["method"] == "xuna/ready"
            tools_req = ws.receive_json()
            assert tools_req["method"] == "tools/list"
            ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": tools_req["id"],
                    "result": {"tools": _android_tools_wire()},
                }
            )
            assert app.state.host_session_hub.is_online(host_id_str) is True
        assert app.state.host_session_hub.is_online(host_id_str) is False
    host_routes.run_once = orig_run_once


def test_ios_self_enroll_and_unsupported_tools_filtered_and_rejected() -> None:
    """UDC-6 S6.2: iPhone self-enrolls with `platform: 'ios'`, supported tools (`url.open`,
    Apple Music `media.*`, `host.status`) and unsupported tools (`app.launch`, `media.volume_set`)
    marked `permission: 'unsupported'`. Unsupported tools are omitted from the LLM catalog
    and rejected with `status: 'unsupported'` if invoked."""

    async def _run() -> None:
        host_repo = FakeHostRepository()
        verifier = P256SignatureVerifier()
        audit = MemoryAuditLog()
        limiter = HostRateLimiter(5, 60.0)
        challenge_store = SelfEnrollmentChallengeStore(60.0)
        hub = InMemoryHostSessionHub()
        owner_id = uuid.uuid4()
        now = datetime.now(UTC)
        mono = time.monotonic()

        ios_key = SimulatedDevice()
        nonce = b"\x99" * 32
        code = b"\xaa" * 16
        challenge_store.issue(str(owner_id), nonce, code, mono)
        sig = ios_key.sign(
            host_proof_message(nonce=nonce, purpose=HostProofPurpose.ENROLL, payload=code)
        )

        ios_tools: list[dict[str, object]] = [
            {
                "name": "url.open",
                "description": "Open an allowlisted HTTPS URL on the iPhone.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string", "maxLength": 2048}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "additionalProperties": False,
                },
                "_meta": {
                    "xuna/policy": {
                        "version": 1,
                        "family": "url",
                        "risk": "R2",
                        "observes": False,
                        "reversible": False,
                        "permission": "granted",
                        "platform": "ios",
                    }
                },
            },
            {
                "name": "media.play_pause",
                "description": "Toggle Apple Music playback on the iPhone.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
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
                        "platform": "ios",
                    }
                },
            },
            {
                "name": "app.launch",
                "description": "Unsupported on iOS.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"app_id": {"type": "string"}},
                    "required": ["app_id"],
                    "additionalProperties": False,
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "additionalProperties": False,
                },
                "_meta": {
                    "xuna/policy": {
                        "version": 1,
                        "family": "app",
                        "risk": "R2",
                        "observes": False,
                        "reversible": False,
                        "permission": "unsupported",
                        "platform": "ios",
                    }
                },
            },
        ]

        host, _, _ = await self_enroll_host(
            host_repo,
            verifier,
            audit,
            limiter,
            challenge_store,
            owner_id,
            SelfEnrollHostCommand(
                platform=HostPlatform.IOS,
                os_version="iOS 17.5",
                agent_version="1.0.0",
                alias="iPhone 15",
                public_key=ios_key.public_key,
                signature=sig,
                hostname="iPhone-15",
                tools=ios_tools,
            ),
            now,
            mono,
        )
        assert host.platform == HostPlatform.IOS

        await set_grants(
            host_repo,
            audit,
            owner_id,
            host.host_id,
            [
                (ToolFamily.URL, RiskTier.R2, True),
                (ToolFamily.APP, RiskTier.R2, True),
            ],
            now,
        )

        # Catalog includes `url__open` and `media__play_pause`, never `app__launch`
        catalog = await load_tool_catalog_for_owner(
            owner_id=owner_id,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            host_repo=host_repo,
            hub=hub,
            self_host_id=str(host.host_id),
        )
        assert {t.name for t in catalog} == {"url__open", "media__play_pause"}

        # Calling `app__launch` directly returns `status: "unsupported"`
        orch = Orchestrator(
            host_repo=host_repo,
            hub=hub,
            approval_repo=_FakeApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            clock=lambda: now,
        )
        emitted: list[ActionStreamEvent] = []

        async def on_event(ev: ActionStreamEvent) -> None:
            emitted.append(ev)

        msg = await orch.handle_call(
            ToolCall(id="c-unsup", name="app__launch", arguments={"app_id": "com.spotify.music"}),
            ActionContext(
                owner_id=owner_id,
                pet_id="pet-1",
                locale=Locale.VI,
                client_caps=(CAP_HOST_ACTIONS_V1,),
                board_available=False,
                emit_event=on_event,
                self_host_id=str(host.host_id),
            ),
        )
        assert '"status":"unsupported"' in msg.content
        assert any(isinstance(e, ActionResultEvent) and e.status == "unsupported" for e in emitted)

    asyncio.run(_run())
