"""Standing approvals (ADR-0014 D1): eligibility, signed creation, matching and revocation.

Signatures are real P-256 ECDSA over `xuna.preapproval.v1`, verified by the production
`P256SignatureVerifier`, so the wire message format is exercised end to end.
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from xuna_backend.modules.actions.application.catalog import CAP_HOST_ACTIONS_V1
from xuna_backend.modules.actions.application.orchestrator import ActionContext, Orchestrator
from xuna_backend.modules.actions.application.standing_approvals import StandingApprovalService
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
)
from xuna_backend.modules.actions.domain.standing_approval import (
    STANDING_MAX_LIFETIME,
    StandingApproval,
    StandingApprovalError,
    check_standing_eligibility,
    is_standing_eligible,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.actions.infrastructure.standing_repository import (
    InMemoryStandingApprovalRepository,
)
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.domain.crypto import (
    PREAPPROVAL_DOMAIN,
    args_sha256,
    preapproval_proof_message,
)
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
    ToolPermissionState,
)
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier

NOW = datetime(2026, 11, 6, 12, 0, 0, tzinfo=UTC)
LAUNCH_ARGS: dict[str, object] = {"app_id": "com.spotify.music"}


def _descriptor(
    name: str = "app.launch",
    *,
    family: ToolFamily = ToolFamily.APP,
    risk: RiskTier = RiskTier.R2,
    observes: bool = False,
    schema: dict[str, object] | None = None,
    permission: ToolPermissionState = ToolPermissionState.GRANTED,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description="Test tool.",
        input_schema=schema
        or {
            "type": "object",
            "properties": {
                "app_id": {
                    "type": "string",
                    "enum": ["com.spotify.music", "org.mozilla.firefox"],
                }
            },
            "required": ["app_id"],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=family,
        risk=risk,
        observes=observes,
        permission=permission,
    )


class _Phone:
    """An enrolled phone with a real P-256 key."""

    def __init__(self) -> None:
        self.private = ec.generate_private_key(ec.SECP256R1())
        self.public = self.private.public_key().public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )

    def sign(self, message: bytes) -> bytes:
        der = self.private.sign(message, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")


class _AuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)

    async def recent(self, owner_id: uuid.UUID, limit: int) -> list[AuditEvent]:
        return [e for e in self.events if e.owner_id == owner_id][:limit]

    def actions(self) -> list[AuditAction]:
        return [e.action for e in self.events]


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


class _Hub:
    def __init__(self) -> None:
        self.online: set[str] = set()
        self.requests: list[dict[str, object]] = []

    def is_online(self, host_id: str) -> bool:
        return host_id in self.online

    async def send_request(
        self, host_id: str, method: str, params: dict[str, object], timeout_s: float
    ) -> dict[str, object]:
        self.requests.append(params)
        return {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {},
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 1},
        }


def _host(owner: uuid.UUID, alias: str, platform: HostPlatform, key: bytes) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner,
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


class _World:
    """Owner with an enrolled Android phone (the signer) and a Windows PC host."""

    def __init__(self, descriptor: ToolDescriptor | None = None) -> None:
        self.owner = uuid.uuid4()
        self.phone_key = _Phone()
        self.repo = _HostRepo()
        self.audit = _AuditLog()
        self.standing = InMemoryStandingApprovalRepository()
        self.now = NOW
        self.phone_host = _host(self.owner, "Pixel", HostPlatform.ANDROID, self.phone_key.public)
        self.pc = _host(self.owner, "PC", HostPlatform.WINDOWS, b"\x04" + bytes(range(1, 65)))
        self.descriptor = descriptor or _descriptor()
        self.service = StandingApprovalService(
            host_repo=self.repo,
            repository=self.standing,
            audit=self.audit,
            verifier=P256SignatureVerifier(),
            clock=lambda: self.now,
        )

    async def setup(self, grant_risk: RiskTier = RiskTier.R2) -> None:
        for host in (self.phone_host, self.pc):
            await self.repo.save_host(host)
        await self.repo.save_manifest(
            Manifest(host_id=self.pc.host_id, tools=(self.descriptor,), updated_at=NOW)
        )
        await self.repo.set_grants(
            self.pc.host_id,
            [
                Grant(
                    host_id=self.pc.host_id,
                    family=self.descriptor.family,
                    max_risk=grant_risk,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )

    async def create(
        self,
        *,
        args: dict[str, object] | None = None,
        approval_id: uuid.UUID | None = None,
        expires_in: timedelta = timedelta(days=7),
        max_uses: int | None = None,
        signer: _Phone | None = None,
        tool: str | None = None,
    ) -> StandingApproval:
        args = LAUNCH_ARGS if args is None else args
        approval_id = approval_id or uuid.uuid4()
        tool_name = tool or self.descriptor.name
        expires = int((self.now + expires_in).timestamp())
        signer = signer or self.phone_key
        message = preapproval_proof_message(
            str(approval_id),
            tool_name,
            args_sha256(args),
            str(self.pc.host_id),
            expires,
            max_uses or 0,
        )
        return await self.service.create(
            owner_id=self.owner,
            approval_id=approval_id,
            host_id=self.pc.host_id,
            tool=tool_name,
            arguments=dict(args),
            expires_at_epoch_s=expires,
            max_uses=max_uses,
            signature=signer.sign(message),
            signer_public_key=signer.public,
        )

    def orchestrator(self, hub: _Hub) -> Orchestrator:
        return Orchestrator(
            host_repo=self.repo,
            hub=hub,
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=self.audit,
            clock=lambda: self.now,
            standing_repo=self.standing,
        )


async def _call(
    world: _World,
    orch: Orchestrator,
    args: dict[str, object],
    *,
    name: str = "app__launch",
) -> tuple[str, list[ActionStreamEvent]]:
    """Runs one tool call. Returns `"ok"`-style status, or `"pending"` if it asked the owner."""
    events: list[ActionStreamEvent] = []

    async def emit(event: ActionStreamEvent) -> None:
        events.append(event)

    ctx = ActionContext(
        owner_id=world.owner,
        pet_id="pet-1",
        locale=Locale.VI,
        client_caps=(CAP_HOST_ACTIONS_V1,),
        board_available=False,
        emit_event=emit,
    )
    task = asyncio.create_task(orch.handle_call(ToolCall(id="c1", name=name, arguments=args), ctx))
    for _ in range(200):
        if task.done() or any(isinstance(e, ActionPendingEvent) for e in events):
            break
        await asyncio.sleep(0)
    if not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return "pending", events
    return str(json.loads(task.result().content)["status"]), events


# --------------------------------------------------------------------------- eligibility


@pytest.mark.parametrize(
    ("descriptor", "args", "code"),
    [
        (_descriptor(), {"app_id": "com.spotify.music"}, None),
        (_descriptor(), {}, None),
        (_descriptor(risk=RiskTier.R1), {"app_id": "org.mozilla.firefox"}, "not_eligible_risk"),
        (_descriptor(risk=RiskTier.R3), {"app_id": "org.mozilla.firefox"}, "not_eligible_risk"),
        (_descriptor(observes=True), {"app_id": "org.mozilla.firefox"}, "not_eligible_observes"),
        (
            _descriptor("ui.scroll", family=ToolFamily.UI),
            {"app_id": "org.mozilla.firefox"},
            "not_eligible_family",
        ),
        (
            _descriptor("cu.step", family=ToolFamily.CU),
            {"app_id": "org.mozilla.firefox"},
            "not_eligible_family",
        ),
        (
            _descriptor(permission=ToolPermissionState.UNSUPPORTED),
            {"app_id": "org.mozilla.firefox"},
            "not_eligible_unsupported",
        ),
        (_descriptor(), {"app_id": "com.evil.app"}, "not_eligible_free_text"),
        (
            _descriptor(
                "url.open",
                family=ToolFamily.URL,
                schema={
                    "type": "object",
                    "properties": {"url": {"type": "string", "maxLength": 256}},
                    "additionalProperties": False,
                },
            ),
            {"url": "https://example.com"},
            "not_eligible_free_text",
        ),
        (_descriptor(), {"app_id": ["a"]}, "not_eligible_argument"),
    ],
)
def test_eligibility_table(
    descriptor: ToolDescriptor, args: dict[str, object], code: str | None
) -> None:
    if code is None:
        check_standing_eligibility(descriptor, args)
        assert is_standing_eligible(descriptor, args) is True
        return
    with pytest.raises(StandingApprovalError) as error:
        check_standing_eligibility(descriptor, args)
    assert error.value.code == code
    assert is_standing_eligible(descriptor, args) is False


def test_standing_approval_entity_rules() -> None:
    def make(**overrides: object) -> StandingApproval:
        base: dict[str, object] = {
            "approval_id": uuid.uuid4(),
            "owner_id": uuid.uuid4(),
            "host_id": HostId.generate(),
            "target_alias": "PC",
            "tool": "app.launch",
            "args_sha256": args_sha256(LAUNCH_ARGS),
            "summary": "Mở ứng dụng",
            "created_at": NOW,
            "expires_at": NOW + timedelta(days=7),
            "max_uses": None,
            "signature": bytes(64),
            "signer_public_key": b"\x04" + bytes(64),
        }
        base.update(overrides)
        return StandingApproval(**base)  # type: ignore[arg-type]

    approval = make()
    assert approval.is_usable(NOW)
    assert not approval.is_usable(NOW + timedelta(days=7))
    assert not make(revoked_at=NOW).is_usable(NOW)
    assert not make(max_uses=2, uses=2).is_usable(NOW)
    with pytest.raises(ValueError, match="30 days"):
        make(expires_at=NOW + STANDING_MAX_LIFETIME + timedelta(seconds=1))
    with pytest.raises(ValueError, match="1..1000"):
        make(max_uses=0)


def test_preapproval_message_is_domain_separated_and_deterministic() -> None:
    message = preapproval_proof_message("id", "app.launch", "a" * 64, "host", 1_800_000_000, 0)
    assert message.startswith(PREAPPROVAL_DOMAIN)
    assert message == preapproval_proof_message(
        "id", "app.launch", "a" * 64, "host", 1_800_000_000, 0
    )
    assert message != preapproval_proof_message(
        "id", "app.launch", "a" * 64, "host", 1_800_000_000, 1
    )
    assert not message.startswith(b"xuna.approval.v1\x00")
    with pytest.raises(ValueError, match="non-empty"):
        preapproval_proof_message("", "app.launch", "a" * 64, "host", 1, 0)
    with pytest.raises(ValueError, match="positive"):
        preapproval_proof_message("id", "app.launch", "a" * 64, "host", 0, 0)


# --------------------------------------------------------------------------- creation


def test_create_stores_a_signed_approval_and_audits_without_content() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        approval = await world.create(max_uses=3)

        assert approval.tool == "app.launch"
        assert approval.args_sha256 == args_sha256(LAUNCH_ARGS)
        assert approval.expires_at == NOW + timedelta(days=7)
        assert approval.max_uses == 3
        assert world.audit.actions() == [AuditAction.HOST_PREAPPROVAL_CREATED]
        metadata = world.audit.events[0].metadata
        assert "com.spotify.music" not in json.dumps(metadata), "audit must be content-free"
        listed = await world.service.list_active(world.owner)
        assert [a.approval_id for a in listed] == [approval.approval_id]

    asyncio.run(run())


def test_create_is_idempotent_and_detects_conflicting_reuse_of_an_id() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        approval_id = uuid.uuid4()
        first = await world.create(approval_id=approval_id)
        again = await world.create(approval_id=approval_id)
        assert again.approval_id == first.approval_id
        assert len(await world.service.list_active(world.owner)) == 1

        with pytest.raises(StandingApprovalError) as error:
            await world.create(approval_id=approval_id, args={"app_id": "org.mozilla.firefox"})
        assert error.value.code == "standing_approval_conflict"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"expires_in": timedelta(days=31)}, "invalid_expiry"),
        ({"expires_in": timedelta(seconds=-5)}, "invalid_expiry"),
        ({"max_uses": 1001}, "invalid_max_uses"),
        ({"args": {"app_id": "com.evil.app"}}, "invalid_args"),
        ({"args": {"unknown": 1}}, "invalid_args"),
        ({"tool": "media.play_pause"}, "unknown_tool"),
    ],
)
def test_create_refuses_invalid_requests(kwargs: dict[str, object], code: str) -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        with pytest.raises(StandingApprovalError) as error:
            await world.create(**kwargs)  # type: ignore[arg-type]
        assert error.value.code == code
        assert await world.service.list_active(world.owner) == []
        assert world.audit.events == []

    asyncio.run(run())


def test_create_refuses_ineligible_tools_even_when_signed() -> None:
    async def run() -> None:
        for descriptor, code in (
            (_descriptor("app.kill", risk=RiskTier.R3), "not_eligible_risk"),
            (_descriptor("ui.scroll", family=ToolFamily.UI), "not_eligible_family"),
            (_descriptor("app.read", observes=True), "not_eligible_observes"),
        ):
            world = _World(descriptor)
            await world.setup(grant_risk=RiskTier.R3)
            with pytest.raises(StandingApprovalError) as error:
                await world.create()
            assert error.value.code == code

    asyncio.run(run())


def test_create_requires_a_valid_signature_from_an_enrolled_phone() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()

        # A key that signs correctly but is not an enrolled phone key.
        with pytest.raises(StandingApprovalError) as stranger:
            await world.create(signer=_Phone())
        assert stranger.value.code == "signature_invalid"

        # The enrolled phone's key, but the signature covers different arguments.
        approval_id = uuid.uuid4()
        expires = int((NOW + timedelta(days=7)).timestamp())
        wrong = world.phone_key.sign(
            preapproval_proof_message(
                str(approval_id),
                "app.launch",
                args_sha256({"app_id": "org.mozilla.firefox"}),
                str(world.pc.host_id),
                expires,
                0,
            )
        )
        with pytest.raises(StandingApprovalError) as tampered:
            await world.service.create(
                owner_id=world.owner,
                approval_id=approval_id,
                host_id=world.pc.host_id,
                tool="app.launch",
                arguments=dict(LAUNCH_ARGS),
                expires_at_epoch_s=expires,
                max_uses=None,
                signature=wrong,
                signer_public_key=world.phone_key.public,
            )
        assert tampered.value.code == "signature_invalid"

        # A normal approval signature (xuna.approval.v1) cannot be replayed as a standing one.
        from xuna_backend.modules.hosts.domain.crypto import approval_proof_message

        replay = world.phone_key.sign(
            approval_proof_message(
                str(approval_id),
                args_sha256(LAUNCH_ARGS),
                str(world.pc.host_id),
                datetime.fromtimestamp(expires, UTC).isoformat(),
            )
        )
        with pytest.raises(StandingApprovalError) as replayed:
            await world.service.create(
                owner_id=world.owner,
                approval_id=approval_id,
                host_id=world.pc.host_id,
                tool="app.launch",
                arguments=dict(LAUNCH_ARGS),
                expires_at_epoch_s=expires,
                max_uses=None,
                signature=replay,
                signer_public_key=world.phone_key.public,
            )
        assert replayed.value.code == "signature_invalid"
        assert await world.service.list_active(world.owner) == []

    asyncio.run(run())


def test_create_fails_closed_without_a_verifier_and_for_foreign_hosts() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        world.service = StandingApprovalService(
            host_repo=world.repo,
            repository=world.standing,
            audit=world.audit,
            verifier=None,
            clock=lambda: NOW,
        )
        with pytest.raises(StandingApprovalError) as no_verifier:
            await world.create()
        assert no_verifier.value.code == "signature_unverifiable"

        other = _World()
        await other.setup()
        other.service = StandingApprovalService(
            host_repo=world.repo,
            repository=world.standing,
            audit=world.audit,
            verifier=P256SignatureVerifier(),
            clock=lambda: NOW,
        )
        other.owner = uuid.uuid4()
        with pytest.raises(StandingApprovalError) as foreign:
            await other.create()
        assert foreign.value.code in {"host_not_found", "signature_invalid"}

    asyncio.run(run())


def test_revoke_is_owner_scoped_and_audited() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        approval = await world.create()

        assert not await world.service.revoke(
            owner_id=uuid.uuid4(), approval_id=approval.approval_id
        )
        assert await world.service.revoke(owner_id=world.owner, approval_id=approval.approval_id)
        assert await world.service.list_active(world.owner) == []
        assert world.audit.actions()[-1] == AuditAction.HOST_PREAPPROVAL_REVOKED

    asyncio.run(run())


# --------------------------------------------------------------------------- orchestrator


def test_a_matching_standing_approval_runs_without_asking_and_is_audited() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        await world.create()
        hub = _Hub()
        hub.online.add(str(world.pc.host_id))

        status, events = await _call(world, world.orchestrator(hub), dict(LAUNCH_ARGS))

        assert status == "ok"
        assert not any(isinstance(e, ActionPendingEvent) for e in events)
        assert len(hub.requests) == 1
        used = [e for e in world.audit.events if e.action == AuditAction.HOST_PREAPPROVAL_USED]
        assert len(used) == 1
        assert used[0].metadata["uses"] == 1
        assert "com.spotify.music" not in json.dumps(used[0].metadata)

    asyncio.run(run())


def test_a_different_argument_still_asks_the_owner() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        await world.create()
        hub = _Hub()
        hub.online.add(str(world.pc.host_id))

        status, events = await _call(
            world, world.orchestrator(hub), {"app_id": "org.mozilla.firefox"}
        )

        assert status == "pending"
        pending = [e for e in events if isinstance(e, ActionPendingEvent)]
        assert len(pending) == 1
        # The phone is offered "always allow this" with exactly the validated arguments.
        assert pending[0].standing_tool == "app.launch"
        assert pending[0].standing_arguments == {"app_id": "org.mozilla.firefox"}
        assert hub.requests == []

    asyncio.run(run())


def test_max_uses_expiry_and_revocation_each_stop_the_approval() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        hub = _Hub()
        hub.online.add(str(world.pc.host_id))
        orch = world.orchestrator(hub)

        once = await world.create(max_uses=1)
        assert (await _call(world, orch, dict(LAUNCH_ARGS)))[0] == "ok"
        assert (await _call(world, orch, dict(LAUNCH_ARGS)))[0] == "pending"

        await world.service.revoke(owner_id=world.owner, approval_id=once.approval_id)
        fresh = await world.create(expires_in=timedelta(days=1))
        assert (await _call(world, orch, dict(LAUNCH_ARGS)))[0] == "ok"
        await world.service.revoke(owner_id=world.owner, approval_id=fresh.approval_id)
        assert (await _call(world, orch, dict(LAUNCH_ARGS)))[0] == "pending"

        await world.create(expires_in=timedelta(days=1))
        world.now = NOW + timedelta(days=2)
        assert (await _call(world, orch, dict(LAUNCH_ARGS)))[0] == "pending"

    asyncio.run(run())


def test_a_suspended_or_missing_grant_makes_the_approval_inert() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        await world.create()
        hub = _Hub()
        hub.online.add(str(world.pc.host_id))
        orch = world.orchestrator(hub)

        await world.repo.set_grants(
            world.pc.host_id,
            [
                Grant(
                    host_id=world.pc.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                    suspended_at=NOW,
                )
            ],
        )
        status, _ = await _call(world, orch, dict(LAUNCH_ARGS))

        assert status == "denied"
        assert hub.requests == []

    asyncio.run(run())


def test_the_approval_stops_matching_when_the_manifest_no_longer_pins_the_value() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        await world.create()
        shrunk = _descriptor(
            schema={
                "type": "object",
                "properties": {"app_id": {"type": "string", "enum": ["org.mozilla.firefox"]}},
                "required": ["app_id"],
                "additionalProperties": False,
            }
        )
        await world.repo.save_manifest(
            Manifest(host_id=world.pc.host_id, tools=(shrunk,), updated_at=NOW)
        )
        hub = _Hub()
        hub.online.add(str(world.pc.host_id))

        status, _ = await _call(world, world.orchestrator(hub), dict(LAUNCH_ARGS))

        assert status == "invalid_args", "the allowlist no longer contains the pinned app"
        assert hub.requests == []

    asyncio.run(run())


def test_an_r3_tool_is_never_matched_even_if_a_record_exists() -> None:
    async def run() -> None:
        r3 = _descriptor("app.kill", risk=RiskTier.R3)
        world = _World(r3)
        await world.setup(grant_risk=RiskTier.R3)
        # Bypass the service to simulate a bad or old record in the store.
        approval_id = uuid.uuid4()
        await world.standing.save(
            StandingApproval(
                approval_id=approval_id,
                owner_id=world.owner,
                host_id=world.pc.host_id,
                target_alias="PC",
                tool="app.kill",
                args_sha256=args_sha256(LAUNCH_ARGS),
                summary="Đóng ứng dụng",
                created_at=NOW,
                expires_at=NOW + timedelta(days=1),
                max_uses=None,
                signature=bytes(64),
                signer_public_key=world.phone_key.public,
            )
        )
        hub = _Hub()
        hub.online.add(str(world.pc.host_id))

        status, events = await _call(
            world, world.orchestrator(hub), dict(LAUNCH_ARGS), name="app__kill"
        )

        assert status == "pending"
        pending = [e for e in events if isinstance(e, ActionPendingEvent)]
        assert (
            pending[0].standing_tool is None
        ), "no offer for a tool that can never be pre-approved"
        assert hub.requests == []

    asyncio.run(run())
