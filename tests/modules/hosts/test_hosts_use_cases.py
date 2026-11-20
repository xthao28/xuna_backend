"""Application use-case unit tests with fakes for the `hosts` module (UDC-3 S3.2)."""

import asyncio
import base64
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from tests.helpers import MemoryAuditLog, SimulatedDevice

from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.hosts.application.use_cases import (
    ClaimEnrollmentCommand,
    DevApprovalInput,
    HostError,
    OpenHostSessionCommand,
    StartEnrollmentCommand,
    claim_enrollment,
    confirm_enrollment,
    invoke_tool,
    list_hosts,
    open_host_session,
    preflight_tool,
    record_duplicate_session_rejection,
    record_manifest,
    revoke_host,
    set_grants,
    start_enrollment,
)
from xuna_backend.modules.hosts.domain.crypto import (
    HostProofPurpose,
    args_sha256,
    host_proof_message,
    public_key_fingerprint_b64url,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    PendingEnrollment,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)
from xuna_backend.modules.hosts.domain.session import (
    HostRateLimiter,
    HostSessionRejected,
)
from xuna_backend.modules.hosts.infrastructure.session_hub import InMemoryHostSessionHub
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier


class FakeEnrollmentRepository:
    def __init__(self) -> None:
        self.by_id: dict[uuid.UUID, PendingEnrollment] = {}

    async def save_pending(self, enrollment: PendingEnrollment) -> None:
        self.by_id[enrollment.pending_id.value] = enrollment

    async def get_pending_by_code_hash(self, code_sha256: str) -> PendingEnrollment | None:
        for item in self.by_id.values():
            if item.code_sha256 == code_sha256:
                return item
        return None

    async def get_pending_by_id(self, pending_id: HostId) -> PendingEnrollment | None:
        return self.by_id.get(pending_id.value)

    async def delete_pending(self, pending_id: HostId) -> None:
        self.by_id.pop(pending_id.value, None)


class FakeHostRepository:
    def __init__(self) -> None:
        self.hosts: dict[uuid.UUID, Host] = {}
        self.grants: dict[uuid.UUID, list[Grant]] = {}
        self.manifests: dict[uuid.UUID, Manifest] = {}

    async def get_host(self, host_id: HostId) -> Host | None:
        return self.hosts.get(host_id.value)

    async def save_host(self, host: Host) -> None:
        self.hosts[host.host_id.value] = host

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        return [h for h in self.hosts.values() if h.owner_id == owner_id]

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self.grants.get(host_id.value, []))

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self.grants[host_id.value] = list(grants)

    async def suspend_r2_plus_grants_for_owner(self, owner_id: uuid.UUID, now: datetime) -> int:
        from dataclasses import replace

        suspended = 0
        for h in self.hosts.values():
            if h.owner_id != owner_id:
                continue
            current = self.grants.get(h.host_id.value, [])
            updated: list[Grant] = []
            for g in current:
                if g.enabled and g.max_risk.level >= RiskTier.R2.level and g.suspended_at is None:
                    updated.append(replace(g, suspended_at=now, updated_at=now))
                    suspended += 1
                else:
                    updated.append(g)
            self.grants[h.host_id.value] = updated
        return suspended

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifests.get(host_id.value)

    async def save_manifest(self, manifest: Manifest) -> None:
        self.manifests[manifest.host_id.value] = manifest


def _sample_tools() -> list[ToolDescriptor]:
    empty_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return [
        ToolDescriptor(
            name="media.play_pause",
            description="Toggle media playback.",
            input_schema=empty_schema,
            output_schema=empty_schema,
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        ),
        ToolDescriptor(
            name="app.launch",
            description="Launch an allowlisted app.",
            input_schema=empty_schema,
            output_schema=empty_schema,
            family=ToolFamily.APP,
            risk=RiskTier.R2,
        ),
    ]


def _b64url_no_pad(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


async def _enroll_host_fixture(
    enroll_repo: FakeEnrollmentRepository,
    host_repo: FakeHostRepository,
    hub: InMemoryHostSessionHub,
    audit: MemoryAuditLog,
    owner_id: uuid.UUID,
    now: datetime,
) -> tuple[Host, SimulatedDevice]:
    agent_key = SimulatedDevice()
    verifier = P256SignatureVerifier()
    code = bytes(range(1, 17))
    nonce = bytes(range(32))
    msg = host_proof_message(nonce, HostProofPurpose.ENROLL, code)
    sig = agent_key.sign(msg)

    await start_enrollment(
        enroll_repo,
        verifier,
        StartEnrollmentCommand(
            code=code,
            public_key=agent_key.public_key,
            nonce=nonce,
            signature=sig,
            platform=HostPlatform.WINDOWS,
            os_version="10.0.22631",
            agent_version="0.1.0",
            hostname="PC-THAO",
        ),
        now,
    )
    limiter = HostRateLimiter(5, 60.0)
    claimed = await claim_enrollment(
        enroll_repo,
        audit,
        limiter,
        owner_id,
        ClaimEnrollmentCommand(
            enrollment_code=_b64url_no_pad(code),
            alias="PC của Thảo",
            key_fingerprint=public_key_fingerprint_b64url(agent_key.public_key),
        ),
        now,
        100.0,
    )
    assert len(claimed.sas) == 6
    host, grants = await confirm_enrollment(
        enroll_repo,
        host_repo,
        hub,
        audit,
        owner_id,
        claimed.pending_id,
        now,
    )
    assert any(g.family == ToolFamily.MEDIA and g.enabled for g in grants)
    return host, agent_key


def test_enrollment_happy_path_and_session_open() -> None:
    async def _run() -> None:
        enroll_repo = FakeEnrollmentRepository()
        host_repo = FakeHostRepository()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        owner_id = uuid.uuid4()
        now = datetime(2026, 10, 30, 8, 0, tzinfo=UTC)

        host, agent_key = await _enroll_host_fixture(
            enroll_repo, host_repo, hub, audit, owner_id, now
        )
        assert AuditAction.HOST_ENROLLED in [e.action for e in audit.events]

        # Open authenticated session (purpose 2)
        session_nonce = bytes(range(32, 64))
        session_msg = host_proof_message(
            session_nonce,
            HostProofPurpose.SESSION,
            agent_key.public_key,
            host.host_id,
        )
        verifier = P256SignatureVerifier()
        connected_host = await open_host_session(
            host_repo,
            verifier,
            audit,
            OpenHostSessionCommand(
                host_id=host.host_id,
                nonce=session_nonce,
                signature=agent_key.sign(session_msg),
                platform=HostPlatform.WINDOWS,
                os_version="10.0.22631",
                agent_version="0.1.0",
            ),
            now + timedelta(seconds=5),
        )
        assert connected_host.host_id == host.host_id
        assert AuditAction.HOST_CONNECTED in [e.action for e in audit.events]

    asyncio.run(_run())


def test_expired_enrollment_code_and_fingerprint_mismatch() -> None:
    async def _run() -> None:
        enroll_repo = FakeEnrollmentRepository()
        audit = MemoryAuditLog()
        verifier = P256SignatureVerifier()
        limiter = HostRateLimiter(5, 60.0)
        owner_id = uuid.uuid4()
        now = datetime(2026, 10, 30, 8, 0, tzinfo=UTC)

        agent_key = SimulatedDevice()
        code = bytes(range(1, 17))
        nonce = bytes(range(32))
        await start_enrollment(
            enroll_repo,
            verifier,
            StartEnrollmentCommand(
                code=code,
                public_key=agent_key.public_key,
                nonce=nonce,
                signature=agent_key.sign(host_proof_message(nonce, HostProofPurpose.ENROLL, code)),
                platform=HostPlatform.MACOS,
                os_version="15.1",
                agent_version="0.1.0",
                hostname="MacBook-Pro",
            ),
            now,
        )

        # Expired after 5 minutes
        with pytest.raises(HostError) as exc_info:
            await claim_enrollment(
                enroll_repo,
                audit,
                limiter,
                owner_id,
                ClaimEnrollmentCommand(
                    enrollment_code=_b64url_no_pad(code),
                    alias="Mac của Thảo",
                ),
                now + timedelta(seconds=301),
                100.0,
            )
        assert exc_info.value.code == "not_found"

        # Re-create and test fingerprint mismatch -> 409 conflict + host.enroll_rejected
        await start_enrollment(
            enroll_repo,
            verifier,
            StartEnrollmentCommand(
                code=code,
                public_key=agent_key.public_key,
                nonce=nonce,
                signature=agent_key.sign(host_proof_message(nonce, HostProofPurpose.ENROLL, code)),
                platform=HostPlatform.MACOS,
                os_version="15.1",
                agent_version="0.1.0",
                hostname="MacBook-Pro",
            ),
            now,
        )
        with pytest.raises(HostError) as mismatch_exc:
            await claim_enrollment(
                enroll_repo,
                audit,
                limiter,
                owner_id,
                ClaimEnrollmentCommand(
                    enrollment_code=_b64url_no_pad(code),
                    alias="Mac của Thảo",
                    key_fingerprint="AAAAAAAAAAAAAAAAAAAAAA",
                ),
                now + timedelta(seconds=10),
                101.0,
            )
        assert mismatch_exc.value.code == "conflict"
        assert any(
            e.action == AuditAction.HOST_ENROLL_REJECTED
            and e.metadata.get("reason") == "fingerprint_mismatch"
            for e in audit.events
        )

    asyncio.run(_run())


def test_duplicate_session_closes_both_and_audits_rejection() -> None:
    async def _run() -> None:
        enroll_repo = FakeEnrollmentRepository()
        host_repo = FakeHostRepository()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        owner_id = uuid.uuid4()
        now = datetime(2026, 10, 30, 8, 0, tzinfo=UTC)
        host, _ = await _enroll_host_fixture(enroll_repo, host_repo, hub, audit, owner_id, now)

        closed_first = False

        async def _noop_send(_: str) -> None:
            pass

        async def _close_dup_first() -> None:
            nonlocal closed_first
            closed_first = True

        async def _noop_close() -> None:
            pass

        token1 = object()
        token2 = object()
        assert await hub.open_session(
            str(host.host_id), token1, _noop_send, _close_dup_first, _noop_close
        )
        assert hub.is_online(str(host.host_id))

        # Second concurrent session for the same hostId closes the first and returns False
        second_opened = await hub.open_session(
            str(host.host_id), token2, _noop_send, _noop_close, _noop_close
        )
        assert second_opened is False
        assert closed_first is True
        assert not hub.is_online(str(host.host_id))

        await record_duplicate_session_rejection(host_repo, audit, host.host_id, now)
        assert any(
            e.action == AuditAction.HOST_REJECTED
            and e.metadata.get("reason") == "duplicate_session"
            for e in audit.events
        )

    asyncio.run(_run())


def test_revoke_during_live_session_and_grant_enforcement() -> None:
    async def _run() -> None:
        enroll_repo = FakeEnrollmentRepository()
        host_repo = FakeHostRepository()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        owner_id = uuid.uuid4()
        now = datetime(2026, 10, 30, 8, 0, tzinfo=UTC)
        host, agent_key = await _enroll_host_fixture(
            enroll_repo, host_repo, hub, audit, owner_id, now
        )
        await record_manifest(host_repo, host.host_id, _sample_tools(), now)

        sent_frames: list[str] = []
        revoked_closed = False

        async def _send_text(text: str) -> None:
            sent_frames.append(text)
            # Immediately answer tools/call requests in the test fake
            import json

            frame = json.loads(text)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(host.host_id),
                    frame["id"],
                    {
                        "content": [{"type": "text", "text": "opened"}],
                        "structuredContent": {"id": "vscode"},
                        "isError": False,
                        "_meta": {"xuna/status": "ok", "xuna/durationMs": 42},
                    },
                )

        async def _close_revoked() -> None:
            nonlocal revoked_closed
            revoked_closed = True

        async def _noop() -> None:
            pass

        await hub.open_session(str(host.host_id), object(), _send_text, _noop, _close_revoked)

        # 1. R2 tool `app.launch` without grant -> denied (`grant_denied`)
        denied_res = await invoke_tool(
            host_repo,
            hub,
            audit,
            owner_id,
            host.host_id,
            "app.launch",
            {"id": "vscode"},
            now,
            include_dev_approval=True,
        )
        assert denied_res.status == "denied"
        assert denied_res.error_code == "grant_denied"

        # 2. Enable `app` R2 grant, but omit approval -> denied (`approval_required`)
        await set_grants(
            host_repo,
            audit,
            owner_id,
            host.host_id,
            [(ToolFamily.APP, RiskTier.R2, True)],
            now,
        )
        no_approval_res = await invoke_tool(
            host_repo,
            hub,
            audit,
            owner_id,
            host.host_id,
            "app.launch",
            {"id": "vscode"},
            now,
            include_dev_approval=False,
        )
        assert no_approval_res.status == "denied"
        assert no_approval_res.error_code == "approval_required"

        # 3. With grant and dev approval -> ok, and audit metadata never contains arguments
        ok_res = await invoke_tool(
            host_repo,
            hub,
            audit,
            owner_id,
            host.host_id,
            "app.launch",
            {"id": "vscode"},
            now,
            include_dev_approval=True,
        )
        assert ok_res.status == "ok"
        assert ok_res.structured_content == {"id": "vscode"}
        invoked_events = [e for e in audit.events if e.action == AuditAction.HOST_TOOL_INVOKED]
        assert len(invoked_events) == 1
        assert "vscode" not in str(invoked_events[0].metadata)

        # 4. Revoke during live session -> sends xuna/stop {reason: "revoked"}, closes socket,
        #    and refuses subsequent reconnects with UNAUTHORIZED.
        await revoke_host(host_repo, hub, audit, owner_id, host.host_id, now)
        assert revoked_closed is True
        assert not hub.is_online(str(host.host_id))
        assert any('"method":"xuna/stop"' in f for f in sent_frames)
        assert len(await list_hosts(host_repo, hub, owner_id)) == 0

        with pytest.raises(HostSessionRejected):
            await open_host_session(
                host_repo,
                P256SignatureVerifier(),
                audit,
                OpenHostSessionCommand(
                    host_id=host.host_id,
                    nonce=bytes(32),
                    signature=agent_key.sign(
                        host_proof_message(
                            bytes(32),
                            HostProofPurpose.SESSION,
                            agent_key.public_key,
                            host.host_id,
                        )
                    ),
                    platform=HostPlatform.WINDOWS,
                    os_version="10.0.22631",
                    agent_version="0.1.0",
                ),
                now,
            )

    asyncio.run(_run())


def test_preflight_tool_and_dev_approval_forwarding() -> None:
    async def _run() -> None:
        now = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
        owner_id = uuid.uuid4()
        other_owner_id = uuid.uuid4()
        host_repo = FakeHostRepository()
        audit = MemoryAuditLog()
        hub = InMemoryHostSessionHub()

        host = Host(
            host_id=HostId.generate(),
            owner_id=owner_id,
            alias=HostAlias("Mac"),
            platform=HostPlatform.MACOS,
            hostname="mac-host",
            os_version="15.0",
            agent_version="0.1.0",
            public_key=SimulatedDevice().public_key,
            status=HostStatus.ACTIVE,
            created_at=now,
        )
        await host_repo.save_host(host)
        input_click = ToolDescriptor(
            name="input.click",
            description="Click",
            input_schema={
                "type": "object",
                "properties": {"xNorm": {"type": "number"}, "yNorm": {"type": "number"}},
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.INPUT,
            risk=RiskTier.R2,
        )
        cu_start = ToolDescriptor(
            name="cu.session_start",
            description="Start CU",
            input_schema={
                "type": "object",
                "properties": {"goal": {"type": "string"}, "apps": {"type": "array"}},
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.CU,
            risk=RiskTier.R3,
        )
        await record_manifest(host_repo, host.host_id, [input_click, cu_start], now)
        await set_grants(
            host_repo,
            audit,
            owner_id,
            host.host_id,
            [(ToolFamily.CU, RiskTier.R3, True), (ToolFamily.INPUT, RiskTier.R3, True)],
            now,
        )

        # Offline host -> host_offline
        with pytest.raises(HostError) as off_err:
            await preflight_tool(
                host_repo, hub, owner_id, host.host_id, "input.click", {"xNorm": 0.5, "yNorm": 0.5}
            )
        assert off_err.value.code == "host_offline"

        sent_frames: list[dict[str, object]] = []
        reply_mode = ["ok"]

        async def send_frame(raw: str) -> None:
            import json

            frame = json.loads(raw)
            sent_frames.append(frame)
            if frame.get("method") == "xuna/preflight":
                if reply_mode[0] == "timeout":
                    return
                if reply_mode[0] == "denied":
                    hub.on_response(
                        str(host.host_id),
                        frame["id"],
                        {
                            "preflight": {
                                "status": "denied",
                                "errorCode": "password_field_forbidden",
                            }
                        },
                    )
                    return
                if reply_mode[0] == "malformed":
                    hub.on_response(
                        str(host.host_id),
                        frame["id"],
                        {"unexpected": True},
                    )
                    return
                hub.on_response(
                    str(host.host_id),
                    frame["id"],
                    {
                        "preflight": {
                            "status": "ok",
                            "effectiveRisk": "R3",
                            "reasonCode": "sensitive_control",
                            "element": {"role": "button", "name": "SecretSubmitBtn"},
                        }
                    },
                )
            elif frame.get("method") == "tools/call":
                hub.on_response(
                    str(host.host_id),
                    frame["id"],
                    {
                        "structuredContent": {"sessionId": "cu-signed"},
                        "_meta": {"xuna/status": "ok"},
                    },
                )

        async def close_frame() -> None:
            return None

        await hub.open_session(str(host.host_id), object(), send_frame, close_frame, close_frame)

        # 1. preflight ok: sends exact {"name", "arguments"} and never audits element_name
        audit.events.clear()
        res_ok = await preflight_tool(
            host_repo, hub, owner_id, host.host_id, "input.click", {"xNorm": 0.5, "yNorm": 0.5}
        )
        assert res_ok.status == "ok"
        assert res_ok.effective_risk == RiskTier.R3
        assert res_ok.reason_code == "sensitive_control"
        assert res_ok.element_role == "button"
        assert res_ok.element_name == "SecretSubmitBtn"
        assert sent_frames[-1]["params"] == {
            "name": "input.click",
            "arguments": {"xNorm": 0.5, "yNorm": 0.5},
        }
        assert audit.events == []

        # 2. preflight denied
        reply_mode[0] = "denied"
        res_denied = await preflight_tool(
            host_repo, hub, owner_id, host.host_id, "input.click", {"xNorm": 0.5, "yNorm": 0.5}
        )
        assert res_denied.status == "denied"
        assert res_denied.error_code == "password_field_forbidden"

        # 3. preflight timeout
        reply_mode[0] = "timeout"
        res_timeout = await preflight_tool(
            host_repo,
            hub,
            owner_id,
            host.host_id,
            "input.click",
            {"xNorm": 0.5, "yNorm": 0.5},
            deadline_ms=100,
        )
        assert res_timeout.status == "timeout"
        assert res_timeout.error_code == "deadline_exceeded"

        # 4. preflight malformed -> validation_failed
        reply_mode[0] = "malformed"
        with pytest.raises(HostError) as mal_err:
            await preflight_tool(
                host_repo, hub, owner_id, host.host_id, "input.click", {"xNorm": 0.5, "yNorm": 0.5}
            )
        assert mal_err.value.code == "validation_failed"

        # 5. wrong owner & missing tool
        with pytest.raises(HostError) as own_err:
            await preflight_tool(host_repo, hub, other_owner_id, host.host_id, "input.click", {})
        assert own_err.value.code == "not_found"

        with pytest.raises(HostError) as tool_err:
            await preflight_tool(host_repo, hub, owner_id, host.host_id, "input.nonexistent", {})
        assert tool_err.value.code == "tool_not_found"

        # 6. invoke_tool with dev_approval forwards exact approval fields when approved
        approval_id = str(uuid.uuid4())
        expires_at = "2026-11-19T13:00:00+00:00"
        sig_b64 = base64.b64encode(b"\x11" * 64).decode("ascii")
        pub_b64 = base64.b64encode(b"\x04" + b"\x22" * 64).decode("ascii")
        dev_approval = DevApprovalInput(
            approval_id=approval_id,
            expires_at=expires_at,
            scope="once",
            signature=sig_b64,
            public_key=pub_b64,
        )
        cu_args: dict[str, object] = {"goal": "Signed", "apps": ["notes"]}

        # include_dev_approval=False STILL denies even if dev_approval is supplied
        denied_inv = await invoke_tool(
            host_repo,
            hub,
            audit,
            owner_id,
            host.host_id,
            "cu.session_start",
            cu_args,
            now,
            include_dev_approval=False,
            dev_approval=dev_approval,
        )
        assert denied_inv.status == "denied"
        assert denied_inv.error_code == "approval_required"

        ok_inv = await invoke_tool(
            host_repo,
            hub,
            audit,
            owner_id,
            host.host_id,
            "cu.session_start",
            cu_args,
            now,
            include_dev_approval=True,
            dev_approval=dev_approval,
        )
        assert ok_inv.status == "ok"
        call_params = sent_frames[-1]["params"]
        assert isinstance(call_params, dict)
        meta = call_params["_meta"]
        assert isinstance(meta, dict)
        assert meta["xuna/approval"] == {
            "approvalId": approval_id,
            "argsSha256": args_sha256(cu_args),
            "expiresAt": expires_at,
            "scope": "once",
            "signature": sig_b64,
            "publicKey": pub_b64,
        }

    asyncio.run(_run())
