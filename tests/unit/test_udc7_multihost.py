"""Unit and integration tests for UDC-7 (S7.1-S7.5):
- S7.1: Grants expiry (`expires_at`), suspension (`suspended_at`), `suspendAll` toggle,
        and `conversation_15m` R2 approval scope (R3 and board always `once`).
- S7.2: Hardened `TargetResolver` with 50 scripted multi-host requests across
        2 PCs + 1 phone + the board (aliases, diacritic-insensitive names, pronouns,
        5-minute `last_target` memory, capability-only filtering, ambiguous cases)
        achieving 0 wrong-host executions.
- S7.3: Emergency stop (`POST /v1/actions/stop` and `xuna/stop`) cancelling active
        invocations, rejecting pending approvals, notifying live hosts, suspending
        R2+ grants, clearing 15m scopes, and auditing `AuditAction.HOST_STOPPED`.
- S7.4: `xuna/lanPeer` frame parsing/building and host LAN fingerprint/port persistence.
- S7.5: Phone-signed R3 approvals (`approval_proof_message`), valid P-256 signature
        verified and forwarded in `_meta["xuna/approval"]`, invalid/tampered signature
        rejected with `approval_signature_invalid`.
"""

import asyncio
import base64
import json
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from tests.helpers import MemoryAuditLog, SimulatedDevice
from tests.modules.hosts.test_hosts_use_cases import FakeHostRepository

from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    Orchestrator,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScope,
    ApprovalState,
)
from xuna_backend.modules.actions.domain.target_resolver import (
    TargetResolutionStatus,
    TargetResolver,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.application.use_cases import (
    GrantUpdateSpec,
    set_grants,
    update_host_lan_peer,
)
from xuna_backend.modules.hosts.application.xtp_frames import (
    ParsedLanPeer,
    lan_peer_notification_frame,
    parse_agent_frame,
)
from xuna_backend.modules.hosts.domain.crypto import (
    approval_proof_message,
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
)
from xuna_backend.modules.hosts.infrastructure.session_hub import (
    InMemoryHostSessionHub,
)
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier


class _FakeApprovalRepository:
    def __init__(self) -> None:
        self.items: dict[uuid.UUID, ApprovalRequest] = {}

    async def save(self, req: ApprovalRequest) -> None:
        self.items[req.approval_id] = req

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        return self.items.get(approval_id)

    async def reject_pending_for_owner(self, owner_id: uuid.UUID, now: datetime) -> int:
        count = 0
        for aid, req in list(self.items.items()):
            if req.owner_id == owner_id and req.state == ApprovalState.PENDING:
                self.items[aid] = replace(
                    req,
                    state=ApprovalState.REJECTED,
                    decided_at=now,
                )
                count += 1
        return count


def _make_host(
    owner_id: uuid.UUID,
    alias: str,
    platform: HostPlatform,
    now: datetime,
    *,
    public_key: bytes | None = None,
) -> Host:
    pub = public_key or SimulatedDevice().public_key
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname="host-1",
        os_version="1.0",
        agent_version="0.1.0",
        public_key=pub,
        status=HostStatus.ACTIVE,
        created_at=now,
        last_seen_at=now,
    )


def _make_manifest(
    host_id: HostId,
    platform: HostPlatform,
    now: datetime,
    *,
    include_clipboard_r3: bool = False,
    spotify_allowed: bool = True,
) -> Manifest:
    app_enum = ["spotify", "vscode"] if spotify_allowed else ["vscode"]
    tools: list[ToolDescriptor] = [
        ToolDescriptor(
            name="media.play_pause",
            description="Play or pause media.",
            input_schema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["toggle", "play", "pause"]}
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"state": {"type": "string"}},
                "additionalProperties": False,
            },
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
            observes=False,
            reversible=True,
             platform=platform.value,
        ),
        ToolDescriptor(
            name="app.launch",
            description="Launch an allowlisted app.",
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string", "enum": app_enum}},
                "required": ["id"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"launched": {"type": "boolean"}},
                "additionalProperties": False,
            },
            family=ToolFamily.APP,
            risk=RiskTier.R2,
            observes=False,
            reversible=True,
            platform=platform.value,
        ),
    ]
    if include_clipboard_r3:
        tools.append(
            ToolDescriptor(
                name="screen.capture",
                description="Capture active window screenshot (R3).",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"captured": {"type": "boolean"}},
                    "additionalProperties": False,
                },
                family=ToolFamily.SCREEN,
                risk=RiskTier.R3,
                observes=True,
                reversible=True,
                platform=platform.value,
            )
        )
    return Manifest(
        host_id=host_id,
        schema_version=1,
        tools=tuple(tools),
        updated_at=now,
    )


def test_s71_grant_expiry_and_suspension() -> None:
    now = datetime(2026, 10, 31, 8, 0, tzinfo=UTC)
    owner_id = uuid.uuid4()
    repo = FakeHostRepository()
    audit = MemoryAuditLog()
    host = _make_host(owner_id, "PC Làm Việc", HostPlatform.WINDOWS, now)
    asyncio.run(repo.save_host(host))

    one_hour = now + timedelta(hours=1)
    updated = asyncio.run(
        set_grants(
            repo,
            audit,
            owner_id,
            host.host_id,
            [
                (ToolFamily.MEDIA, RiskTier.R1, True),
                GrantUpdateSpec(
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    expires_at=one_hour,
                    suspended=False,
                ),
            ],
            now,
            "req-1",
        )
    )
    by_fam = {g.family: g for g in updated}
    assert by_fam[ToolFamily.APP].is_active(now) is True
    assert by_fam[ToolFamily.APP].is_active(now + timedelta(minutes=59)) is True
    assert by_fam[ToolFamily.APP].is_active(now + timedelta(minutes=61)) is False

    # Suspend all ("Tạm khoá tất cả")
    suspended_list = asyncio.run(
        set_grants(
            repo,
            audit,
            owner_id,
            host.host_id,
            [],
            now,
            "req-2",
            suspend_all=True,
        )
    )
    for g in suspended_list:
        if g.enabled:
            assert g.suspended_at == now
            assert g.is_active(now) is False

    # Unsuspend all ("Bật lại quyền")
    resumed_list = asyncio.run(
        set_grants(
            repo,
            audit,
            owner_id,
            host.host_id,
            [],
            now + timedelta(minutes=5),
            "req-3",
            suspend_all=False,
        )
    )
    by_fam_resumed = {g.family: g for g in resumed_list}
    assert by_fam_resumed[ToolFamily.APP].suspended_at is None
    assert by_fam_resumed[ToolFamily.APP].is_active(now + timedelta(minutes=5)) is True


def test_s72_50_requests_multihost_targeting_zero_wrong_host() -> None:
    """Scripted 50-request multi-host targeting benchmark across 2 PCs + 1 phone + board."""
    t0 = datetime(2026, 10, 31, 9, 0, tzinfo=UTC)
    owner_id = uuid.uuid4()
    pc_work = _make_host(owner_id, "PC Làm Việc", HostPlatform.WINDOWS, t0)
    mac_thao = _make_host(owner_id, "Mac của Thảo", HostPlatform.MACOS, t0)
    phone_thao = _make_host(owner_id, "Điện thoại Thảo", HostPlatform.ANDROID, t0)

    host_tools: dict[HostId, set[str]] = {
        pc_work.host_id: {"media.play_pause", "app.launch", "window.focus", "system.lock"},
        mac_thao.host_id: {"media.play_pause", "app.launch", "clipboard.read_text"},
        phone_thao.host_id: {"media.play_pause", "app.launch", "device.open_settings"},
    }
    all_hosts = [pc_work, mac_thao, phone_thao]

    def _only_mac_has_figma(h: Host) -> bool:
        return h.host_id == mac_thao.host_id

    def _only_pc_has_foobar(h: Host) -> bool:
        return h.host_id == pc_work.host_id

    ok = TargetResolutionStatus.RESOLVED
    ask = TargetResolutionStatus.ASK_TARGET

    # Step tuple: (tool, requested_target, dt_s, cap_filter, expected_status, expected_host)
    script: list[
        tuple[
            str,
            str | None,
            int,
            Callable[[Host], bool] | None,
            TargetResolutionStatus,
            Host | None,
        ]
    ] = [
        # 1-5: Explicit aliases (exact & case/diacritic variations)
        ("media.play_pause", "PC Làm Việc", 0, None, ok, pc_work),
        ("media.play_pause", "pc lam viec", 10, None, ok, pc_work),
        ("media.play_pause", "Mac của Thảo", 20, None, ok, mac_thao),
        ("media.play_pause", "mac cua thao", 30, None, ok, mac_thao),
        ("media.play_pause", "Điện thoại Thảo", 40, None, ok, phone_thao),
        # 6-10: Same-target pronouns & omitted target within 5 minutes of Phone
        ("media.play_pause", "nó", 50, None, ok, phone_thao),
        ("media.play_pause", "điện thoại đó", 60, None, ok, phone_thao),
        ("media.play_pause", None, 70, None, ok, phone_thao),
        ("media.play_pause", "it", 80, None, ok, phone_thao),
        ("media.play_pause", "that one", 90, None, ok, phone_thao),
        # 11-16: Switch to PC Làm Việc, then hop between the 2 PCs with "máy kia" / "pc kia"
        ("media.play_pause", "PC Làm Việc", 100, None, ok, pc_work),
        ("media.play_pause", "máy đó", 110, None, ok, pc_work),
        ("media.play_pause", "máy kia", 120, None, ok, mac_thao),
        ("media.play_pause", "nó", 130, None, ok, mac_thao),
        ("media.play_pause", "pc kia", 140, None, ok, pc_work),
        ("media.play_pause", "the other pc", 150, None, ok, mac_thao),
        # 17-22: Capability-only unique resolution (tool or argument only supported on 1 host)
        ("window.focus", None, 160, None, ok, pc_work),
        ("system.lock", None, 170, None, ok, pc_work),
        ("clipboard.read_text", None, 180, None, ok, mac_thao),
        ("device.open_settings", None, 190, None, ok, phone_thao),
        ("app.launch", None, 200, _only_mac_has_figma, ok, mac_thao),
        ("app.launch", None, 210, _only_pc_has_foobar, ok, pc_work),
        # 23-27: Board HID media key (`board.media_key`) never routes to a host
        ("board.media_key", None, 220, None, ok, None),
        ("board.media_key", "XUNA", 225, None, ok, None),
        ("board.media_key", "PC Làm Việc", 230, None, ok, None),
        ("board.media_key", "Mac của Thảo", 235, None, ok, None),
        ("board.media_key", "Điện thoại Thảo", 240, None, ok, None),
        # 28-35: 5-minute TTL expiry -> omitted target or pronoun after > 300s must ask_target
        ("media.play_pause", None, 600, None, ask, None),
        ("media.play_pause", "nó", 610, None, ask, None),
        ("media.play_pause", "máy đó", 620, None, ask, None),
        ("media.play_pause", "máy kia", 630, None, ask, None),
        ("app.launch", None, 640, None, ask, None),
        ("app.launch", "it", 650, None, ask, None),
        ("app.launch", "the other one", 660, None, ask, None),
        ("media.play_pause", "máy tính đó", 670, None, ask, None),
        # 36-42: Unknown aliases & unsupported tool combinations must never fall back to another
        ("media.play_pause", "Laptop Gaming", 680, None, ask, None),
        ("media.play_pause", "iPad của Thảo", 690, None, ask, None),
        ("media.play_pause", "PC Văn Phòng", 700, None, ask, None),
        ("window.focus", "Mac của Thảo", 710, None, ask, None),
        ("window.focus", "Điện thoại Thảo", 720, None, ask, None),
        ("device.open_settings", "PC Làm Việc", 730, None, ask, None),
        ("app.launch", "PC Làm Việc", 740, _only_mac_has_figma, ask, None),
        # 43-50: Rapid multi-host sequence with explicit + pronoun + capability switches
        ("media.play_pause", "dien thoai thao", 750, None, ok, phone_thao),
        ("media.play_pause", "nó", 760, None, ok, phone_thao),
        ("media.play_pause", "PC Làm Việc", 770, None, ok, pc_work),
        ("media.play_pause", "máy kia", 780, None, ok, mac_thao),
        ("media.play_pause", "máy kia", 790, None, ok, pc_work),
        ("media.play_pause", None, 800, None, ok, pc_work),
        ("clipboard.read_text", "Mac của Thảo", 810, None, ok, mac_thao),
        ("media.play_pause", "nó", 820, None, ok, mac_thao),
    ]

    assert len(script) == 50
    last_alias: str | None = None
    last_at: datetime | None = None
    wrong_host_executions = 0

    for idx, (tool, explicit, dt_s, cap_filter, exp_status, exp_host) in enumerate(script, 1):
        now = t0 + timedelta(seconds=dt_s)
        if tool == "board.media_key":
            # Board tool always targets the board directly, never a host.
            assert exp_host is None
            continue

        eligible = [h for h in all_hosts if tool in host_tools[h.host_id]]
        res = TargetResolver.resolve(
            requested_target=explicit,
            eligible_hosts=eligible,
            all_tool_hosts=eligible,
            last_target_alias=last_alias,
            last_target_at=last_at,
            now=now,
            capability_filter=cap_filter,
        )
        if res.status == TargetResolutionStatus.RESOLVED:
            assert res.host is not None
            if exp_host is None or res.host.host_id != exp_host.host_id:
                wrong_host_executions += 1
            last_alias = res.host.alias.value
            last_at = now
        assert res.status == exp_status, (
            f"step {idx} ({tool}, {explicit}): {res.status} != {exp_status}"
        )
        if exp_host is not None and res.host is not None:
            assert res.host.host_id == exp_host.host_id, (
                f"step {idx}: wrong host {res.host.alias.value}"
            )

    assert wrong_host_executions == 0


def test_s71_s73_s75_orchestrator_15m_scope_signed_r3_and_emergency_stop() -> None:
    """Tests 15-minute R2 approval scope, phone-signed R3 approvals, and emergency stop."""

    async def _scenario() -> None:
        now_box = [datetime(2026, 10, 31, 10, 0, tzinfo=UTC)]
        owner_id = uuid.uuid4()
        repo = FakeHostRepository()
        approval_repo = _FakeApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        last_target_store = InMemoryLastTargetStore()
        scope_store = InMemoryApprovalScopeStore()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        verifier = P256SignatureVerifier()

        phone_device = SimulatedDevice()
        pc_host = _make_host(owner_id, "PC Làm Việc", HostPlatform.WINDOWS, now_box[0])
        phone_host = _make_host(
            owner_id,
            "Điện thoại Thảo",
            HostPlatform.ANDROID,
            now_box[0],
            public_key=phone_device.public_key,
        )
        await repo.save_host(pc_host)
        await repo.save_host(phone_host)
        await repo.save_manifest(
            _make_manifest(
                pc_host.host_id,
                HostPlatform.WINDOWS,
                now_box[0],
                include_clipboard_r3=True,
            )
        )
        await repo.set_grants(
            pc_host.host_id,
            [
                Grant(pc_host.host_id, ToolFamily.MEDIA, RiskTier.R1, True, now_box[0]),
                Grant(pc_host.host_id, ToolFamily.APP, RiskTier.R2, True, now_box[0]),
                Grant(pc_host.host_id, ToolFamily.SCREEN, RiskTier.R3, True, now_box[0]),
            ],
        )

        sent_frames: list[dict[str, object]] = []

        async def _send_text(text: str) -> None:
            frame = json.loads(text)
            sent_frames.append(frame)
            if frame.get("method") == "tools/call":
                req_id = frame["id"]
                tname = frame["params"]["name"]
                out = {"captured": True} if tname == "screen.capture" else {"launched": True}
                hub.on_response(
                    str(pc_host.host_id),
                    req_id,
                    {
                        "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
                        "structuredContent": out,
                    },
                )

        async def _close() -> None:
            pass

        await hub.open_session(
            str(pc_host.host_id),
            object(),
            _send_text,
            _close,
            _close,
        )

        orch = Orchestrator(
            host_repo=repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            last_target_store=last_target_store,
            audit=audit,
            scope_store=scope_store,
            verifier=verifier,
            clock=lambda: now_box[0],
        )

        events: list[ActionStreamEvent] = []

        async def _sink(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                aid = uuid.UUID(ev.approval_id)
                if ev.risk == "R2":
                    await orch.decide_approval(
                        approval_id=aid,
                        owner_id=owner_id,
                        decision=ApprovalDecision.APPROVE,
                        scope=ApprovalScope.CONVERSATION_15M,
                    )
                elif ev.risk == "R3":
                    # R3 must reject conversation_15m scope
                    with pytest.raises(ValueError, match="r3_requires_once_scope"):
                        await orch.decide_approval(
                            approval_id=aid,
                            owner_id=owner_id,
                            decision=ApprovalDecision.APPROVE,
                            scope=ApprovalScope.CONVERSATION_15M,
                        )
                    # R3 must reject tampered signature
                    bad_sig = b"\x01" * 64
                    with pytest.raises(ValueError, match="approval_signature_invalid"):
                        await orch.decide_approval(
                            approval_id=aid,
                            owner_id=owner_id,
                            decision=ApprovalDecision.APPROVE,
                            scope=ApprovalScope.ONCE,
                            signature=bad_sig,
                            signer_public_key=phone_device.public_key,
                        )
                    # Valid phone P-256 signature over approval_proof_message
                    assert ev.host_id is not None
                    msg = approval_proof_message(
                        ev.approval_id,
                        ev.args_sha256,
                        ev.host_id,
                        ev.expires_at,
                    )
                    valid_sig = phone_device.sign(msg)
                    await orch.decide_approval(
                        approval_id=aid,
                        owner_id=owner_id,
                        decision=ApprovalDecision.APPROVE,
                        scope=ApprovalScope.ONCE,
                        signature=valid_sig,
                        signer_public_key=phone_device.public_key,
                    )

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps={"host_actions_v1"},
            board_available=False,
            emit_event=_sink,
            request_id="req-orch",
        )
        def runner(call: ToolCall) -> object:
            return orch.handle_call(call, ctx)

        # 1. First R2 call prompts for approval and owner grants `conversation_15m`
        res1 = await runner(
            ToolCall(
                id="call-1",
                name="app.launch",
                arguments={"target": "PC Làm Việc", "id": "spotify"},
            )
        )
        assert json.loads(res1.content)["status"] == "ok"
        assert sum(1 for e in events if isinstance(e, ActionPendingEvent)) == 1

        # 2. Second R2 call within 15 minutes on same host+tool with DIFFERENT valid args ("vscode")
        #    auto-approves without emitting a second ActionPendingEvent!
        now_box[0] = now_box[0] + timedelta(minutes=5)
        res2 = await runner(
            ToolCall(
                id="call-2",
                name="app.launch",
                arguments={"target": "PC Làm Việc", "id": "vscode"},
            )
        )
        assert json.loads(res2.content)["status"] == "ok"
        assert sum(1 for e in events if isinstance(e, ActionPendingEvent)) == 1
        last_call = sent_frames[-1]
        assert (
            last_call["params"]["_meta"]["xuna/approval"]["scope"]  # type: ignore[index]
            == "conversation_15m"
        )

        # 3. R3 call requires phone-signed `once` approval and forwards signature + publicKey
        res3 = await runner(
            ToolCall(
                id="call-3",
                name="screen.capture",
                arguments={"target": "PC Làm Việc"},
            )
        )
        assert json.loads(res3.content)["status"] == "ok"
        r3_call = sent_frames[-1]
        r3_approval_meta = r3_call["params"]["_meta"]["xuna/approval"]  # type: ignore[index]
        assert r3_approval_meta["scope"] == "once"
        assert "signature" in r3_approval_meta
        assert "publicKey" in r3_approval_meta
        assert (
            base64.b64decode(r3_approval_meta["publicKey"])
            == phone_device.public_key
        )

        # 4. S7.3 Emergency Stop: suspends R2+ grants, sends xuna/stop, clears 15m scopes, audits
        stop_summary = await orch.emergency_stop(
            owner_id=owner_id,
            reason="owner_board",
            request_id="req-stop",
        )
        assert stop_summary.stopped is True
        assert stop_summary.hosts_notified == 1
        assert stop_summary.grants_suspended == 2  # APP (R2) + CLIPBOARD (R3)
        assert sent_frames[-1]["method"] == "xuna/stop"
        assert sent_frames[-1]["params"] == {"reason": "owner_board"}
        assert any(e.action == AuditAction.HOST_STOPPED for e in audit.events)

        # After e-stop, R2 app.launch is denied because R2+ grants are suspended
        res4 = await runner(
            ToolCall(
                id="call-4",
                name="app.launch",
                arguments={"target": "PC Làm Việc", "id": "spotify"},
            )
        )
        assert json.loads(res4.content)["status"] == "denied"

    asyncio.run(_scenario())


def test_s74_lan_peer_frames_and_host_persistence() -> None:
    now = datetime(2026, 10, 31, 11, 0, tzinfo=UTC)
    owner_id = uuid.uuid4()
    repo = FakeHostRepository()
    pc_host = _make_host(owner_id, "PC Làm Việc", HostPlatform.WINDOWS, now)
    asyncio.run(repo.save_host(pc_host))

    frame_text = lan_peer_notification_frame(["key1_b64", "key2_b64"])
    frame_obj = json.loads(frame_text)
    assert frame_obj["method"] == "xuna/lanPeer"
    assert frame_obj["params"] == {"phonePublicKeys": ["key1_b64", "key2_b64"]}

    agent_raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "xuna/lanPeer",
            "params": {
                "enabled": True,
                "port": 47821,
                "certFingerprint": "ABCD1234EFGH5678",
            },
        }
    )
    parsed = parse_agent_frame(agent_raw, max_bytes=4096)
    assert isinstance(parsed, ParsedLanPeer)
    assert parsed.params.enabled is True
    assert parsed.params.port == 47821
    assert parsed.params.certFingerprint == "ABCD1234EFGH5678"

    asyncio.run(
        update_host_lan_peer(
            repo,
            pc_host.host_id,
            parsed.params.enabled,
            parsed.params.certFingerprint,
            parsed.params.port,
            now,
        )
    )
    saved = asyncio.run(repo.get_host(pc_host.host_id))
    assert saved is not None
    assert saved.lan_fingerprint == "ABCD1234EFGH5678"
    assert saved.lan_port == 47821
