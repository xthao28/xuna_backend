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
    HostError,
    OpenHostSessionCommand,
    StartEnrollmentCommand,
    claim_enrollment,
    confirm_enrollment,
    invoke_tool,
    list_hosts,
    open_host_session,
    record_duplicate_session_rejection,
    record_manifest,
    revoke_host,
    set_grants,
    start_enrollment,
)
from xuna_backend.modules.hosts.domain.crypto import (
    HostProofPurpose,
    host_proof_message,
    public_key_fingerprint_b64url,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostId,
    HostPlatform,
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
