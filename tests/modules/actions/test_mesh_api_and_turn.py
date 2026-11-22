"""M2.3 unit, API, SSE, and WebSocket integration tests for Spatial Device Mesh.

Covers:
1. Domain event wire payloads (`ActionPendingEvent`, `ActionResultEvent`,
   `ActionTargetRequiredEvent`).
2. `Orchestrator` spatial fields (`hostId` on ok & error, `targetReason`, `targetPlatform`) and
   `ActionTargetRequiredEvent` emission (`ask_target`, `grant_denied`, `no_host`).
3. 2-Tier E-Stop (`POST /v1/actions/stop`, `Orchestrator.emergency_stop`, WS `action.stop`) with
   per-host isolation, foreign `hostId` 404 fail-closed, and sibling preservation.
4. `PATCH /v1/hosts/{host_id}`, extended `GET /v1/hosts` (`presence: null` offline), and focus
   clearing on host revocation.
5. `GET / DELETE /v1/actions/habits` (content-free audit) and `POST /v1/actions/focus`.
6. Conversation SSE (`/turns`) and Voice WS (`/v1/voice`) mesh routing, strict WS validation,
   `/may` & `/follow` short-circuit, and `PendingIntent` 1-word `repair_dispatch` streaming.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from tests.helpers import LocalIssuer, MemoryAuditLog
from tests.modules.actions.test_actions_api_and_turn import (
    FAKE_PUBKEY,
    NOW,
    PET_HEX,
    _DummyDatabase,
    _FakeConsentRepo,
    _FakeHostRepo,
    _FakeIdentityRepo,
    _FakeKnowledgeCandidatePort,
    _FakeKnowledgeRepo,
    _FakeMemoryRepo,
    _FakeOwnerRepo,
    _parse_sse,
    _ScriptedLLM,
)
from tests.modules.actions.test_actions_voice_websocket import _FakeRecognizer, _FakeSynthesizer

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    InMemoryComputerUseSessionStore,
)
from xuna_backend.modules.actions.application.orchestrator import ActionContext, Orchestrator
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionResultEvent,
    ActionTargetRequiredEvent,
    ApprovalRequest,
    ApprovalState,
)
from xuna_backend.modules.actions.domain.habits import HostRoutingHabit
from xuna_backend.modules.actions.domain.job import AutonomousJob, AutonomousJobState
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.habit_repository import InMemoryHabitRepository
from xuna_backend.modules.actions.infrastructure.job_store import InMemoryAutonomousJobStore
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.cognition.domain.types import (
    ChatFinished,
    Locale,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostKind,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
)
from xuna_backend.modules.hosts.domain.twin import (
    HostFocusChangedNotice,
    HostPresence,
)


class _MeshFakeHostRepo(_FakeHostRepo):
    async def suspend_r2_plus_grants_for_owner(self, owner_id: uuid.UUID, now: datetime) -> int:
        count = 0
        for host in self.hosts.values():
            if host.owner_id != owner_id or host.status != HostStatus.ACTIVE:
                continue
            grants = list(self.grants.get(host.host_id, []))
            next_grants: list[Grant] = []
            changed = False
            for g in grants:
                if g.max_risk.level >= RiskTier.R2.level and g.suspended_at is None:
                    next_grants.append(
                        Grant(
                            host_id=g.host_id,
                            family=g.family,
                            max_risk=g.max_risk,
                            enabled=g.enabled,
                            updated_at=now,
                            expires_at=g.expires_at,
                            suspended_at=now,
                        )
                    )
                    count += 1
                    changed = True
                else:
                    next_grants.append(g)
            if changed:
                self.grants[host.host_id] = next_grants
        return count


def _done(latency_ms: int = 8) -> ChatFinished:
    return ChatFinished(
        provider="fake",
        model="m1",
        input_tokens=10,
        output_tokens=5,
        latency_ms=latency_ms,
    )


def _make_host(
    owner_id: uuid.UUID,
    alias: str,
    *,
    platform: HostPlatform = HostPlatform.MACOS,
    host_kind: HostKind = HostKind.DESKTOP,
) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname=f"{alias.lower()}.local",
        os_version="15.0",
        agent_version="0.2.0",
        public_key=FAKE_PUBKEY,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        host_kind=host_kind,
    )


def _media_tool() -> ToolDescriptor:
    return ToolDescriptor(
        name="media.play_pause",
        description="Toggle media playback",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=ToolFamily.MEDIA,
        risk=RiskTier.R1,
    )


def _app_tool() -> ToolDescriptor:
    return ToolDescriptor(
        name="app.launch",
        description="Launch app",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=ToolFamily.APP,
        risk=RiskTier.R2,
    )


def _build_mesh_app(
    llm: _ScriptedLLM,
    *,
    extra_owners: list[Owner] | None = None,
) -> tuple[
    TestClient,
    LocalIssuer,
    Owner,
    _MeshFakeHostRepo,
    InMemoryApprovalRepository,
    InMemoryHabitRepository,
    MemoryAuditLog,
    Any,
]:
    issuer = LocalIssuer()
    owner = Owner(
        owner_id=uuid.uuid4(),
        issuer="https://auth.test.xuna",
        subject="owner-mesh-1",
    )
    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
        ),
        verifier=issuer.verifier(),
        generator=llm,
        recognizer=_FakeRecognizer("hello"),
        synthesizer=_FakeSynthesizer(),
    )
    app.state.database = _DummyDatabase()
    app.state.cu_session_store = InMemoryComputerUseSessionStore()
    app.state.approval_scope_store = InMemoryApprovalScopeStore()

    class _MultiOwnerRepo:
        def __init__(self, owners: list[Owner]) -> None:
            self._by_sub = {o.subject: o for o in owners}

        async def get_by_subject(self, _issuer: str, subject: str) -> Owner | None:
            return self._by_sub.get(subject)

        async def get_or_create(self, claims: Any) -> Owner:
            return self._by_sub[claims.subject]

        async def upsert_owner(self, o: Owner) -> Owner:
            self._by_sub[o.subject] = o
            return o

    owners = [owner, *(extra_owners or [])]
    owner_repo = _MultiOwnerRepo(owners) if extra_owners else _FakeOwnerRepo(owner)
    identity_repo = _FakeIdentityRepo(owner.owner_id, PET_HEX)
    consent_repo = _FakeConsentRepo()
    mem_repo = _FakeMemoryRepo()
    k_repo = _FakeKnowledgeRepo()
    c_port = _FakeKnowledgeCandidatePort()
    host_repo = _MeshFakeHostRepo()
    approval_repo = InMemoryApprovalRepository()
    habit_repo = InMemoryHabitRepository()
    audit_log = MemoryAuditLog()

    app.state.owner_repository = lambda _s: owner_repo
    app.state.identity_repository = lambda _s: identity_repo
    app.state.consent_repository = lambda _s: consent_repo
    app.state.memory_repository = lambda _s: mem_repo
    app.state.knowledge_repository = lambda _s: k_repo
    app.state.knowledge_candidate_adapter = lambda _s: c_port
    app.state.host_repository = lambda _s: host_repo
    app.state.approval_repository = lambda _s: approval_repo
    app.state.habit_repository = lambda _s: habit_repo
    app.state.audit_log = lambda _s: audit_log

    return (
        TestClient(app),
        issuer,
        owner,
        host_repo,
        approval_repo,
        habit_repo,
        audit_log,
        app,
    )


async def _connect_fake_host(hub: Any, owner_id: uuid.UUID, host: Host) -> None:
    async def _noop_send(_txt: str) -> None:
        return None

    async def _noop_close() -> None:
        return None

    await hub.open_session(
        str(host.host_id),
        object(),
        _noop_send,
        _noop_close,
        _noop_close,
        owner_id=owner_id,
    )


def test_domain_events_wire_payload_preserves_back_compat_and_emits_spatial_fields() -> None:
    # Backwards compatible defaults omit None spatial fields
    pending_min = ActionPendingEvent(
        approval_id="a1",
        target="Mac",
        summary="Open app",
        risk="R2",
        expires_in=60,
    )
    p_min = pending_min.wire_payload()
    assert "targetReason" not in p_min
    assert "targetPlatform" not in p_min

    pending_full = ActionPendingEvent(
        approval_id="a2",
        target="Mac",
        summary="Open app",
        risk="R3",
        expires_in=60,
        host_id="h1",
        target_reason="Focused host",
        target_platform="macos",
    )
    p_full = pending_full.wire_payload()
    assert p_full["targetReason"] == "Focused host"
    assert p_full["targetPlatform"] == "macos"

    res_min = ActionResultEvent(invocation_id="i1", target="Mac", summary="Done", status="ok")
    r_min = res_min.wire_payload()
    assert set(r_min.keys()) == {"invocationId", "target", "summary", "status"}

    res_full = ActionResultEvent(
        invocation_id="i2",
        target="Win",
        summary="Done",
        status="ok",
        gesture="bounce_play",
        host_id="h2",
        target_reason="Best match",
        target_platform="windows",
        mission_envelope={"goal": "Test", "missionId": "m1"},
    )
    r_full = res_full.wire_payload()
    assert r_full["hostId"] == "h2"
    assert r_full["targetReason"] == "Best match"
    assert r_full["targetPlatform"] == "windows"
    assert r_full["missionEnvelope"] == {"goal": "Test", "missionId": "m1"}

    tr = ActionTargetRequiredEvent(
        invocation_id="i3",
        tool="media.play_pause",
        reason_code="ask_target",
        target_reason="Choose host",
        candidates=({"hostId": "h1", "alias": "Mac"},),
    )
    assert tr.wire_payload() == {
        "invocationId": "i3",
        "tool": "media.play_pause",
        "reasonCode": "ask_target",
        "targetReason": "Choose host",
        "candidates": [{"hostId": "h1", "alias": "Mac"}],
    }


def test_orchestrator_spatial_events_on_ok_denied_no_host_and_tier1_estop() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        mac = _make_host(owner, "MacNode", platform=HostPlatform.MACOS)
        win = _make_host(owner, "WinNode", platform=HostPlatform.WINDOWS)
        host_repo = _MeshFakeHostRepo()
        await host_repo.save_host(mac)
        await host_repo.save_host(win)
        await host_repo.save_manifest(Manifest(mac.host_id, (_media_tool(), _app_tool()), NOW))
        await host_repo.save_manifest(Manifest(win.host_id, (_media_tool(), _app_tool()), NOW))
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(mac.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),
                Grant(mac.host_id, ToolFamily.APP, RiskTier.R2, True, NOW),
            ],
        )
        await host_repo.set_grants(
            win.host_id,
            [
                Grant(win.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),
                Grant(win.host_id, ToolFamily.APP, RiskTier.R2, True, NOW),
            ],
        )

        class _Hub:
            def __init__(self) -> None:
                self.online = {str(mac.host_id), str(win.host_id)}
                self.stopped: list[tuple[str, str]] = []

            def is_online(self, hid: str) -> bool:
                return hid in self.online

            async def send_request(
                self, hid: str, method: str, params: dict[str, object], timeout_s: float
            ) -> dict[str, object]:
                return {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}}

            async def stop_host(self, hid: str, reason: str = "owner_app") -> bool:
                self.stopped.append((hid, reason))
                return hid in self.online

        hub = _Hub()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        scope_store = InMemoryApprovalScopeStore()
        cu_store = InMemoryComputerUseSessionStore()
        orch = Orchestrator(
            host_repo=host_repo,
            hub=hub,  # type: ignore[arg-type]
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=MemoryAuditLog(),
            scope_store=scope_store,
            cu_session_store=cu_store,
            clock=lambda: NOW,
        )
        events: list[object] = []

        async def _emit(ev: object) -> None:
            events.append(ev)

        # 1. Ambiguous R1 call emits ActionTargetRequiredEvent
        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=("host_actions_v1",),
            board_available=False,
            emit_event=_emit,
        )
        await orch.handle_call(ToolCall(id="c1", name="media__play_pause", arguments={}), ctx)
        assert any(isinstance(e, ActionTargetRequiredEvent) for e in events)
        tr_ev = next(e for e in events if isinstance(e, ActionTargetRequiredEvent))
        assert tr_ev.reason_code == "ask_target"

        # 2. Pinned R1 call emits ActionResultEvent WITH host_id, target_reason, target_platform
        events.clear()
        ctx_pin = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=("host_actions_v1",),
            board_available=False,
            emit_event=_emit,
            pinned_host_id=mac.host_id,
            pinned_target_reason="Explicit target MacNode",
        )
        await orch.handle_call(ToolCall(id="c2", name="media__play_pause", arguments={}), ctx_pin)
        res_ok = next(e for e in events if isinstance(e, ActionResultEvent))
        assert res_ok.status == "ok"
        assert res_ok.host_id == str(mac.host_id)
        assert res_ok.target_reason == "Explicit target MacNode"
        assert res_ok.target_platform == "macos"

        # 3. Unknown/unadvertised host tool emits ActionTargetRequiredEvent(reason_code="no_host")
        events.clear()
        await orch.handle_call(ToolCall(id="c3", name="workflow__run", arguments={}), ctx)
        no_host_ev = next(e for e in events if isinstance(e, ActionTargetRequiredEvent))
        assert no_host_ev.reason_code == "no_host"

        # 4. Tier-1 E-Stop isolates MacNode from WinNode across CU lanes, scopes, approvals, grants
        await cu_store.start(
            ComputerUseSessionState(
                session_id="cu-mac",
                owner_id=owner,
                host_id=mac.host_id,
                target_alias="MacNode",
                goal="Mac work",
                allowed_apps=(),
                allowed_domains=(),
                max_steps=5,
                max_minutes=5,
                started_at_utc=NOW,
                last_heartbeat_at=NOW,
            ),
            NOW,
        )
        await cu_store.start(
            ComputerUseSessionState(
                session_id="cu-win",
                owner_id=owner,
                host_id=win.host_id,
                target_alias="WinNode",
                goal="Win work",
                allowed_apps=(),
                allowed_domains=(),
                max_steps=5,
                max_minutes=5,
                started_at_utc=NOW,
                last_heartbeat_at=NOW,
            ),
            NOW,
        )
        await scope_store.grant_scope(owner, "pet-1", mac.host_id, "app.launch", NOW)
        await scope_store.grant_scope(owner, "pet-1", win.host_id, "app.launch", NOW)

        ap_mac = ApprovalRequest(
            approval_id=uuid.uuid4(),
            owner_id=owner,
            host_id=mac.host_id,
            target_alias="MacNode",
            tool="app.launch",
            args_sha256="a" * 64,
            risk=RiskTier.R2,
            summary="Mac launch",
            created_at=NOW,
            expires_at=NOW + timedelta(seconds=60),
        )
        ap_win = ApprovalRequest(
            approval_id=uuid.uuid4(),
            owner_id=owner,
            host_id=win.host_id,
            target_alias="WinNode",
            tool="app.launch",
            args_sha256="b" * 64,
            risk=RiskTier.R2,
            summary="Win launch",
            created_at=NOW,
            expires_at=NOW + timedelta(seconds=60),
        )
        await approval_repo.save(ap_mac)
        await approval_repo.save(ap_win)
        await approval_broker.create(ap_mac)
        await approval_broker.create(ap_win)

        stop_mac = await orch.emergency_stop(
            owner_id=owner, reason="owner_app", host_id=mac.host_id
        )
        assert stop_mac.host_id == str(mac.host_id)
        assert stop_mac.hosts_notified == 1
        assert stop_mac.approvals_cancelled == 1
        assert stop_mac.grants_suspended == 1
        assert hub.stopped == [(str(mac.host_id), "owner_app")]
        assert await cu_store.get_active_for_owner(owner, host_id=mac.host_id) is None
        assert await cu_store.get_active_for_owner(owner, host_id=win.host_id) is not None
        assert not await scope_store.has_valid_scope(owner, "pet-1", mac.host_id, "app.launch", NOW)
        assert await scope_store.has_valid_scope(owner, "pet-1", win.host_id, "app.launch", NOW)
        assert (await approval_repo.get(ap_mac.approval_id)).state == ApprovalState.REJECTED  # type: ignore[union-attr]
        assert (await approval_repo.get(ap_win.approval_id)).state == ApprovalState.PENDING  # type: ignore[union-attr]
        win_grants = await host_repo.get_grants(win.host_id)
        assert all(g.suspended_at is None for g in win_grants)

    asyncio.run(_run())


def test_patch_host_get_hosts_habits_focus_and_tier1_stop_http_endpoints(
    monkeypatch: MonkeyPatch,
) -> None:
    from xuna_backend.core.idempotency import parse_key
    from xuna_backend.modules.hosts.api import routes as host_routes

    seen_idem: dict[tuple[uuid.UUID, str], Any] = {}

    async def _fake_run_once(
        _session: Any,
        *,
        owner_id: uuid.UUID,
        key: str,
        fingerprint: str,
        now: Any,
        action: Any,
    ) -> Any:
        if (owner_id, key) in seen_idem:
            return seen_idem[(owner_id, key)]
        res = await action()
        seen_idem[(owner_id, key)] = res
        return res

    monkeypatch.setattr(host_routes, "run_once", _fake_run_once)
    monkeypatch.setattr(host_routes, "parse_key", parse_key)

    llm = _ScriptedLLM([[TextDelta("ok"), _done()]])
    foreign_owner = Owner(
        owner_id=uuid.uuid4(),
        issuer="https://auth.test.xuna",
        subject="owner-foreign-2",
    )
    client, issuer, owner, host_repo, _, habit_repo, audit_log, app = _build_mesh_app(
        llm, extra_owners=[foreign_owner]
    )
    headers = {"Authorization": f"Bearer {issuer.token('owner-mesh-1')}"}
    foreign_headers = {"Authorization": f"Bearer {issuer.token('owner-foreign-2')}"}
    hub = app.state.host_session_hub

    mac = _make_host(owner.owner_id, "Mac Nhà", platform=HostPlatform.MACOS)
    win = _make_host(owner.owner_id, "WinDesk", platform=HostPlatform.WINDOWS)

    async def _seed() -> None:
        await host_repo.save_host(mac)
        await host_repo.save_host(win)
        await _connect_fake_host(hub, owner.owner_id, mac)
        hub.update_presence(
            str(mac.host_id),
            HostPresence(
                idle_seconds=12,
                screen_locked=False,
                media_playing=True,
                battery_percent=80,
                is_charging=True,
                foreground_app_id="editor_x",
                subnet_hash="a1b2c3d4e5f60718293a4b5c6d7e8f90",
                updated_at=NOW,
            ),
        )
        await habit_repo.upsert_habit(
            HostRoutingHabit(
                habit_id=uuid.uuid4(),
                owner_id=owner.owner_id,
                pet_id=PET_HEX,
                context_key="media:music:evening",
                host_id=mac.host_id,
                created_at=NOW,
                updated_at=NOW,
                positive_score=3.2,
                negative_score=0.0,
                sample_count=5,
                last_reason_code="confirmed_pick",
            )
        )

    asyncio.run(_seed())

    # 1. GET /v1/hosts shows online presence for Mac and presence: null for offline Win
    list_r = client.get("/v1/hosts", headers=headers)
    assert list_r.status_code == 200
    items = {it["alias"]: it for it in list_r.json()["items"]}
    assert items["Mac Nhà"]["hostKind"] == "desktop"
    assert items["Mac Nhà"]["focused"] is False
    assert items["Mac Nhà"]["presence"]["mediaPlaying"] is True
    assert items["WinDesk"]["presence"] is None

    # 2. PATCH /v1/hosts/{id}: validation, Idempotency-Key, 409, 404, clear with ""
    no_idem = client.patch(
        f"/v1/hosts/{mac.host_id}", headers=headers, json={"schemaVersion": 1, "place": "office"}
    )
    assert no_idem.status_code == 400

    for bad_payload in (
        {"schemaVersion": 1, "alias": "   "},
        {"schemaVersion": 1, "place": "   "},
        {"schemaVersion": 1, "nicknames": ["valid", "   "]},
        {"schemaVersion": 1, "nicknames": [f"n{i}" for i in range(11)]},
        {"schemaVersion": 1, "wolMac": "ZZ:00:11:22:33:44"},
        {"schemaVersion": 1, "hostKind": "robot"},
        {"schemaVersion": 1, "extraKey": 1},
    ):
        r_bad = client.patch(
            f"/v1/hosts/{mac.host_id}",
            headers={**headers, "Idempotency-Key": str(uuid.uuid4())},
            json=bad_payload,
        )
        assert r_bad.status_code == 422, (bad_payload, r_bad.text)

    r_ok = client.patch(
        f"/v1/hosts/{mac.host_id}",
        headers={**headers, "Idempotency-Key": str(uuid.uuid4())},
        json={
            "schemaVersion": 1,
            "place": "office",
            "nicknames": ["work_mac"],
            "wolMac": "aa:bb:cc:dd:ee:ff",
        },
    )
    assert r_ok.status_code == 200
    assert r_ok.json()["place"] == "office"
    assert r_ok.json()["nicknames"] == ["work_mac"]
    assert r_ok.json()["wolMac"] == "AA:BB:CC:DD:EE:FF"

    r_clr = client.patch(
        f"/v1/hosts/{mac.host_id}",
        headers={**headers, "Idempotency-Key": str(uuid.uuid4())},
        json={"schemaVersion": 1, "place": "", "wolMac": ""},
    )
    assert r_clr.status_code == 200
    assert r_clr.json()["place"] is None
    assert r_clr.json()["wolMac"] is None

    r_409 = client.patch(
        f"/v1/hosts/{win.host_id}",
        headers={**headers, "Idempotency-Key": str(uuid.uuid4())},
        json={"schemaVersion": 1, "alias": "mac nha"},
    )
    assert r_409.status_code == 409
    assert r_409.json()["code"] == "conflict"

    r_404 = client.patch(
        f"/v1/hosts/{mac.host_id}",
        headers={**foreign_headers, "Idempotency-Key": str(uuid.uuid4())},
        json={"schemaVersion": 1, "alias": "Steal"},
    )
    assert r_404.status_code == 404

    # 3. POST /v1/actions/focus pins host, rejects foreign with 404, revoke emits notice
    focus_notices: list[HostFocusChangedNotice] = []
    unsub = hub.subscribe_focus(
        owner.owner_id,
        lambda n: asyncio.sleep(0, result=focus_notices.append(n)),  # type: ignore[arg-type]
    )
    try:
        r_focus = client.post(
            "/v1/actions/focus",
            headers=headers,
            json={"schemaVersion": 1, "hostId": str(mac.host_id), "followMe": True},
        )
        assert r_focus.status_code == 200
        assert r_focus.json()["focusedHostId"] == str(mac.host_id)
        assert r_focus.json()["focusedAlias"] == "Mac Nhà"
        assert r_focus.json()["followMe"] is True
        assert focus_notices[-1].source == "app"

        assert (
            client.post(
                "/v1/actions/focus",
                headers=foreign_headers,
                json={"schemaVersion": 1, "hostId": str(mac.host_id)},
            ).status_code
            == 404
        )

        # Revoking focused Mac emits revoked notice & clears focus
        del_r = client.delete(
            f"/v1/hosts/{mac.host_id}",
            headers={**headers, "Idempotency-Key": str(uuid.uuid4())},
        )
        assert del_r.status_code == 204
        assert any(n.source == "revoked" and n.host_id is None for n in focus_notices)
    finally:
        unsub()

    # 4. Habits GET + DELETE (single & bulk) with content-free audit metadata
    habits_r = client.get(f"/v1/actions/habits?petId={PET_HEX}", headers=headers)
    assert habits_r.status_code == 200
    h_items = habits_r.json()["items"]
    assert len(h_items) == 1
    assert h_items[0]["why"].startswith("Picked 5 times")
    hid_str = h_items[0]["habitId"]

    assert (
        client.delete(f"/v1/actions/habits/{hid_str}", headers=foreign_headers).status_code == 404
    )
    del_one = client.delete(f"/v1/actions/habits/{hid_str}", headers=headers)
    assert del_one.status_code == 200
    assert del_one.json() == {"habitId": hid_str, "deleted": True}

    del_all = client.delete(f"/v1/actions/habits?petId={PET_HEX}", headers=headers)
    assert del_all.status_code == 200
    assert del_all.json() == {"deletedCount": 0}

    habit_audits = [e for e in audit_log.events if e.action == AuditAction.HOST_HABIT_DELETED]
    assert len(habit_audits) == 2
    for ev in habit_audits:
        assert "contextKey" not in ev.metadata

    # 5. POST /v1/actions/stop foreign host -> 404; valid host -> 200 with hostId
    assert (
        client.post(
            "/v1/actions/stop",
            headers=foreign_headers,
            json={"schemaVersion": 1, "hostId": str(win.host_id)},
        ).status_code
        == 404
    )
    stop_r = client.post(
        "/v1/actions/stop",
        headers=headers,
        json={"schemaVersion": 1, "hostId": str(win.host_id)},
    )
    assert stop_r.status_code == 200
    assert stop_r.json()["hostId"] == str(win.host_id)


def test_sse_turn_mesh_slash_commands_repair_dispatch_and_ws_strict_validation() -> None:
    llm = _ScriptedLLM(
        [
            # Round 0: ask_target (neither host has presence, both R1 media)
            [
                ToolCallRequested(ToolCall(id="t1", name="media__play_pause", arguments={})),
                _done(),
            ],
            # Round 1: answer asking user
            [TextDelta("Which host?"), _done()],
            # Round 2: summary after repair_dispatch (must receive Role.TOOL)
            [TextDelta("Paused on WinNode."), _done()],
        ]
    )
    foreign_owner = Owner(
        owner_id=uuid.uuid4(),
        issuer="https://auth.test.xuna",
        subject="owner-foreign-ws",
    )
    client, issuer, owner, host_repo, _, _, _, app = _build_mesh_app(
        llm, extra_owners=[foreign_owner]
    )
    token = issuer.token("owner-mesh-1")
    headers = {"Authorization": f"Bearer {token}"}
    hub = app.state.host_session_hub

    mac = _make_host(owner.owner_id, "MacNode", platform=HostPlatform.MACOS)
    win = _make_host(owner.owner_id, "WinNode", platform=HostPlatform.WINDOWS)

    async def _seed() -> None:
        await host_repo.save_host(mac)
        await host_repo.save_host(win)
        await host_repo.save_manifest(Manifest(mac.host_id, (_media_tool(),), NOW))
        await host_repo.save_manifest(Manifest(win.host_id, (_media_tool(),), NOW))
        await host_repo.set_grants(
            mac.host_id, [Grant(mac.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW)]
        )
        await host_repo.set_grants(
            win.host_id, [Grant(win.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW)]
        )
        await _connect_fake_host(hub, owner.owner_id, mac)
        await _connect_fake_host(hub, owner.owner_id, win)

    asyncio.run(_seed())

    async def _fake_send(
        hid: str, method: str, params: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        return {"_meta": {"xuna/status": "ok"}, "structuredContent": {"playing": False}}

    hub.send_request = _fake_send

    # 1. Invalid clientSubnetHash on TurnBody -> 422
    r_bad_subnet = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={"schemaVersion": 1, "text": "hi", "clientSubnetHash": "bad!"},
    )
    assert r_bad_subnet.status_code == 422

    # 2. Foreign targetHostId -> 404
    r_foreign_target = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={
            "schemaVersion": 1,
            "text": "pause",
            "clientCapabilities": ["host_actions_v1"],
            "targetHostId": str(uuid.uuid4()),
        },
    )
    assert r_foreign_target.status_code == 404

    # 3. /may WinNode emits host_focus_changed without invoking LLM
    llm_count_before = len(llm.requests)
    slash_r = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={
            "schemaVersion": 1,
            "text": "/may WinNode",
            "clientCapabilities": ["host_actions_v1"],
        },
    )
    assert slash_r.status_code == 200
    slash_evs = _parse_sse(slash_r.text)
    focus_ev = next((d for name, d in slash_evs if name == "host_focus_changed"), None)
    assert focus_ev is not None and focus_ev["hostId"] == str(win.host_id)
    assert len(llm.requests) == llm_count_before

    # /may auto clears focus without invoking LLM
    auto_r = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={
            "schemaVersion": 1,
            "text": "/may auto",
            "clientCapabilities": ["host_actions_v1"],
        },
    )
    auto_evs = _parse_sse(auto_r.text)
    auto_focus = next((d for name, d in auto_evs if name == "host_focus_changed"), None)
    assert auto_focus is not None and auto_focus["hostId"] is None
    assert len(llm.requests) == llm_count_before

    # 4. SSE Turn 1 -> ask_target -> Turn 2 ("WinNode") repair_dispatch -> Role.TOOL in summary
    t1_r = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={
            "schemaVersion": 1,
            "text": "pause music",
            "clientCapabilities": ["host_actions_v1"],
        },
    )
    t1_evs = _parse_sse(t1_r.text)
    tr_ev = next((d for name, d in t1_evs if name == "action_target_required"), None)
    assert tr_ev is not None and tr_ev["reasonCode"] == "ask_target"

    t2_r = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={
            "schemaVersion": 1,
            "text": "WinNode",
            "clientCapabilities": ["host_actions_v1"],
        },
    )
    t2_evs = _parse_sse(t2_r.text)
    act_res = next((d for name, d in t2_evs if name == "action_result"), None)
    assert act_res is not None
    assert act_res["status"] == "ok"
    assert act_res["hostId"] == str(win.host_id)
    assert act_res["targetPlatform"] == "windows"
    assert act_res["targetReason"]
    assert any(m.role == Role.TOOL for m in llm.requests[-1].messages)

    # 5. Voice WS strict validation on hello & turn.commit + 2-tier action.stop
    with client.websocket_connect("/v1/voice") as bad_ws:
        bad_ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": token},
                "petId": PET_HEX,
                "extraField": True,
            }
        )
        err = bad_ws.receive_json()
        assert err == {"type": "error", "code": "validation_failed"}

    with client.websocket_connect("/v1/voice") as ws:
        ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": token},
                "petId": PET_HEX,
                "speak": False,
                "clientCapabilities": ["host_actions_v1"],
            }
        )
        assert ws.receive_json()["type"] == "hello.ok"

        ws.send_json({"type": "turn.commit", "text": "hi", "clientSubnetHash": "invalid!"})
        assert ws.receive_json() == {"type": "error", "code": "validation_failed"}

        # Socket stays open: send action.stop with foreign hostId -> not_found, then valid hostId
        ws.send_json({"type": "action.stop", "hostId": str(uuid.uuid4())})
        assert ws.receive_json() == {"type": "error", "code": "not_found"}

        ws.send_json({"type": "action.stop", "hostId": str(mac.host_id)})
        stopped = ws.receive_json()
        assert stopped["type"] == "action_stopped"
        assert stopped["hostId"] == str(mac.host_id)
        ws.send_json({"type": "bye"})


def test_m23_remediation_zero_uuid_pinned_unsupported_direct_estop_and_repair_omits_caps() -> None:
    import json

    from starlette.websockets import WebSocketDisconnect

    from xuna_backend.modules.actions.application.job_runner import AutonomousJobRunner

    zero_uuid = "00000000-0000-0000-0000-000000000000"
    llm = _ScriptedLLM(
        [
            # Turn 1 (ui.inspect -> ask_target)
            [
                ToolCallRequested(ToolCall(id="u5_1", name="ui__inspect", arguments={})),
                _done(),
            ],
            [TextDelta("Which host?"), _done()],
            # Turn 2 summary after repair_dispatch omitting clientCapabilities
            [TextDelta("Inspected WinNode."), _done()],
        ]
    )
    client, issuer, owner, host_repo, _, _, audit_log, app = _build_mesh_app(llm)
    token = issuer.token("owner-mesh-1")
    headers = {"Authorization": f"Bearer {token}"}
    hub = app.state.host_session_hub

    mac = _make_host(owner.owner_id, "MacNode", platform=HostPlatform.MACOS)
    win = _make_host(owner.owner_id, "WinNode", platform=HostPlatform.WINDOWS)
    ui_tool = ToolDescriptor(
        name="ui.inspect",
        description="Inspect UI",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=ToolFamily.UI,
        risk=RiskTier.R1,
        observes=True,
    )
    media_next_unsupported = ToolDescriptor(
        name="media.next",
        description="Next track",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=ToolFamily.MEDIA,
        risk=RiskTier.R1,
        permission=ToolPermissionState.UNSUPPORTED,
    )
    media_next_granted = ToolDescriptor(
        name="media.next",
        description="Next track",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=ToolFamily.MEDIA,
        risk=RiskTier.R1,
        permission=ToolPermissionState.GRANTED,
    )

    async def _seed() -> None:
        await host_repo.save_host(mac)
        await host_repo.save_host(win)
        await host_repo.save_manifest(
            Manifest(mac.host_id, (_media_tool(), ui_tool, media_next_unsupported), NOW)
        )
        await host_repo.save_manifest(
            Manifest(win.host_id, (_media_tool(), ui_tool, media_next_granted), NOW)
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(mac.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),
                Grant(mac.host_id, ToolFamily.UI, RiskTier.R1, True, NOW),
            ],
        )
        await host_repo.set_grants(
            win.host_id,
            [
                Grant(win.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),
                Grant(win.host_id, ToolFamily.UI, RiskTier.R1, True, NOW),
            ],
        )
        await _connect_fake_host(hub, owner.owner_id, mac)
        await _connect_fake_host(hub, owner.owner_id, win)

    asyncio.run(_seed())

    async def _fake_send(
        hid: str, method: str, params: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        return {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}}

    hub.send_request = _fake_send

    # 1. Zero UUID returns 422 validation_failed on HTTP endpoints (never 500)
    for path, payload in (
        ("/v1/actions/stop", {"schemaVersion": 1, "hostId": zero_uuid}),
        ("/v1/actions/focus", {"schemaVersion": 1, "hostId": zero_uuid}),
        (
            f"/v1/conversations/{PET_HEX}/turns",
            {"schemaVersion": 1, "text": "hi", "targetHostId": zero_uuid},
        ),
    ):
        resp = client.post(path, headers=headers, json=payload)
        assert resp.status_code == 422, (path, resp.text)
        assert resp.json()["code"] == "validation_failed"

    # 2. Zero UUID on WS hello -> validation_failed + close 1008
    with client.websocket_connect("/v1/voice") as bad_ws:
        bad_ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": token},
                "petId": PET_HEX,
                "targetHostId": zero_uuid,
            }
        )
        assert bad_ws.receive_json() == {"type": "error", "code": "validation_failed"}
        try:
            bad_ws.receive_json()
            raise AssertionError("Expected WebSocket close 1008 after invalid hello")
        except WebSocketDisconnect as exc:
            assert exc.code == 1008

    # 3. Zero UUID on WS turn.commit (top-level & context) -> validation_failed & socket stays open
    with client.websocket_connect("/v1/voice") as ws:
        ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": token},
                "petId": PET_HEX,
                "speak": False,
                "clientCapabilities": ["host_actions_v1"],
            }
        )
        assert ws.receive_json()["type"] == "hello.ok"

        ws.send_json({"type": "turn.commit", "text": "hi", "targetHostId": zero_uuid})
        assert ws.receive_json() == {"type": "error", "code": "validation_failed"}

        ws.send_json({"type": "turn.commit", "text": "hi", "context": {"targetHostId": zero_uuid}})
        assert ws.receive_json() == {"type": "error", "code": "validation_failed"}

        # Socket stays open for valid action.stop
        ws.send_json({"type": "action.stop", "hostId": str(mac.host_id)})
        assert ws.receive_json()["type"] == "action_stopped"
        ws.send_json({"type": "bye"})

    # 4. Pinned UNSUPPORTED tool -> ActionResultEvent(status="unsupported"); absent -> no_host
    async def _check_pinned_unsupported_and_direct_estop() -> None:
        job_store = InMemoryAutonomousJobStore()
        runner = AutonomousJobRunner(
            job_store=job_store,
            clock_utc=lambda: NOW,
        )
        orch = Orchestrator(
            host_repo=host_repo,
            hub=hub,
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=audit_log,
            job_runner=runner,
            clock=lambda: NOW,
        )
        events: list[object] = []

        async def _emit(ev: object) -> None:
            events.append(ev)

        ctx_pin_mac = ActionContext(
            owner_id=owner.owner_id,
            pet_id=PET_HEX,
            locale=Locale.EN,
            client_caps=("host_actions_v1",),
            board_available=False,
            emit_event=_emit,
            pinned_host_id=mac.host_id,
            pinned_target_reason="Explicit target MacNode",
        )
        # 4a. Pinned UNSUPPORTED on MacNode (even while WinNode grants media.next)
        tool_msg = await orch.handle_call(
            ToolCall(id="unsup-1", name="media__next", arguments={}), ctx_pin_mac
        )
        assert json.loads(tool_msg.content)["status"] == "unsupported"
        res_ev = next(e for e in events if isinstance(e, ActionResultEvent))
        assert res_ev.status == "unsupported"
        assert res_ev.host_id == str(mac.host_id)
        assert any(
            e.action == AuditAction.HOST_TOOL_DENIED and e.metadata.get("reason") == "unsupported"
            for e in audit_log.events
        )

        # 4b. Pinned absent tool on MacNode (descriptor is None, while WinNode has app.launch)
        events.clear()
        await host_repo.save_manifest(
            Manifest(win.host_id, (_media_tool(), ui_tool, media_next_granted, _app_tool()), NOW)
        )
        await host_repo.set_grants(
            win.host_id,
            [
                Grant(win.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),
                Grant(win.host_id, ToolFamily.UI, RiskTier.R1, True, NOW),
                Grant(win.host_id, ToolFamily.APP, RiskTier.R2, True, NOW),
            ],
        )
        await orch.handle_call(
            ToolCall(id="absent-1", name="app__launch", arguments={"id": "editor_x"}),
            ctx_pin_mac,
        )
        tr_ev = next(e for e in events if isinstance(e, ActionTargetRequiredEvent))
        assert tr_ev.reason_code == "no_host"
        assert not any(isinstance(e, ActionResultEvent) for e in events)

        # 5. Direct Orchestrator.emergency_stop cancels job_runner jobs (Tier 1 & Tier 2)
        job_mac = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner.owner_id,
            pet_id=PET_HEX,
            host_id=mac.host_id,
            session_id="s-mac",
            title="Mac job",
            goal="Mac goal",
            state=AutonomousJobState.RUNNING,
            created_at=NOW,
            updated_at=NOW,
        )
        job_win = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner.owner_id,
            pet_id=PET_HEX,
            host_id=win.host_id,
            session_id="s-win",
            title="Win job",
            goal="Win goal",
            state=AutonomousJobState.RUNNING,
            created_at=NOW,
            updated_at=NOW,
        )
        await job_store.save(job_mac)
        await job_store.save(job_win)

        await orch.emergency_stop(owner_id=owner.owner_id, reason="direct_t1", host_id=mac.host_id)
        assert (await job_store.get(job_mac.job_id)).state == AutonomousJobState.CANCELLED  # type: ignore[union-attr]
        assert (await job_store.get(job_win.job_id)).state == AutonomousJobState.RUNNING  # type: ignore[union-attr]

        await orch.emergency_stop(owner_id=owner.owner_id, reason="direct_t2", host_id=None)
        assert (await job_store.get(job_win.job_id)).state == AutonomousJobState.CANCELLED  # type: ignore[union-attr]

    asyncio.run(_check_pinned_unsupported_and_direct_estop())

    # 6. repair_dispatch when Turn 2 omits clientCapabilities still loads U5 consent & runs tool
    from xuna_backend.modules.consent.domain.consent import ConsentState

    u5_consent = ConsentState(
        aggregate_metrics=False,
        experience_events=False,
        conversation_text=True,
        conversation_voice=True,
        owner_documents=False,
        host_screen=True,
        version=2,
        updated_at=NOW,
    )

    class _U5ConsentRepo:
        async def current(self, _owner_id: uuid.UUID) -> ConsentState:
            return u5_consent

    app.state.consent_repository = lambda _s: _U5ConsentRepo()

    t1_r = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={
            "schemaVersion": 1,
            "text": "inspect screen",
            "clientCapabilities": ["host_actions_v1"],
        },
    )
    t1_evs = _parse_sse(t1_r.text)
    t1_tr = next((d for name, d in t1_evs if name == "action_target_required"), None)
    assert t1_tr is not None and t1_tr["reasonCode"] == "ask_target"

    t2_r = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={"schemaVersion": 1, "text": "WinNode"},
    )
    t2_evs = _parse_sse(t2_r.text)
    act_res = next((d for name, d in t2_evs if name == "action_result"), None)
    assert act_res is not None
    assert act_res["status"] == "ok"
    assert act_res["hostId"] == str(win.host_id)


def test_emergency_stop_propagates_job_runner_errors_and_keeps_dummy_db_green() -> None:
    from contextlib import asynccontextmanager

    import pytest

    from xuna_backend.modules.actions.infrastructure.job_store import SqlAutonomousJobStore

    llm = _ScriptedLLM([])
    client, issuer, owner, host_repo, _, _, audit_log, app = _build_mesh_app(llm)
    token = issuer.token("owner-mesh-1")
    headers = {"Authorization": f"Bearer {token}"}
    mac = _make_host(owner.owner_id, "MacNode", platform=HostPlatform.MACOS)
    asyncio.run(host_repo.save_host(mac))

    # 1. _DummyDatabase-backed HTTP and WS E-Stop still succeed via SqlAutonomousJobStore
    http_ok = client.post(
        "/v1/actions/stop",
        headers=headers,
        json={"schemaVersion": 1, "hostId": str(mac.host_id)},
    )
    assert http_ok.status_code == 200
    assert http_ok.json()["stopped"] is True

    with client.websocket_connect("/v1/voice") as ws_ok:
        ws_ok.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": token},
                "petId": PET_HEX,
                "speak": False,
            }
        )
        assert ws_ok.receive_json()["type"] == "hello.ok"
        ws_ok.send_json({"type": "action.stop", "hostId": str(mac.host_id)})
        stopped_ev = ws_ok.receive_json()
        assert stopped_ev["type"] == "action_stopped"
        assert stopped_ev["stopped"] is True
        ws_ok.send_json({"type": "bye"})

    # 2. SqlAutonomousJobStore propagates real execute() errors (never swallows them)
    class _FailingSession:
        async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("db execute failed")

    @asynccontextmanager
    async def _failing_sessions() -> Any:
        yield _FailingSession()

    failing_store = SqlAutonomousJobStore(_failing_sessions)
    with pytest.raises(RuntimeError, match="db execute failed"):
        asyncio.run(failing_store.list_for_owner(owner.owner_id))

    # 3. Direct Orchestrator.emergency_stop propagates RuntimeError from job_runner
    class _FailingJobRunner:
        def __init__(self) -> None:
            self.calls = 0

        async def cancel_all_for_owner(
            self,
            _owner_id: uuid.UUID,
            *,
            reason: str = "owner_app",
            host_id: HostId | None = None,
        ) -> int:
            self.calls += 1
            raise RuntimeError(f"job cancel failed: {reason} {host_id}")

    failing_runner = _FailingJobRunner()
    orch = Orchestrator(
        host_repo=host_repo,
        hub=app.state.host_session_hub,
        approval_repo=InMemoryApprovalRepository(),
        approval_broker=InMemoryApprovalBroker(),
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=audit_log,
        job_runner=failing_runner,  # type: ignore[arg-type]
        clock=lambda: NOW,
    )
    with pytest.raises(RuntimeError, match="job cancel failed"):
        asyncio.run(
            orch.emergency_stop(
                owner_id=owner.owner_id,
                reason="direct_fail",
                host_id=mac.host_id,
            )
        )
    assert failing_runner.calls == 1

    # 4. HTTP POST /v1/actions/stop propagates / returns 500 internal_error (never stopped=True)
    app.state.job_runner = failing_runner
    no_raise_client = TestClient(app, raise_server_exceptions=False)
    http_err = no_raise_client.post(
        "/v1/actions/stop",
        headers=headers,
        json={"schemaVersion": 1, "hostId": str(mac.host_id)},
    )
    assert http_err.status_code == 500
    assert http_err.json()["code"] == "internal_error"
    assert "stopped" not in http_err.json()
    assert failing_runner.calls == 2

    # 5. Voice WS action.stop calls cancel_all_for_owner once and does NOT emit action_stopped
    with (
        pytest.raises(RuntimeError, match="job cancel failed"),
        client.websocket_connect("/v1/voice") as ws_fail,
    ):
        ws_fail.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": token},
                "petId": PET_HEX,
                "speak": False,
            }
        )
        assert ws_fail.receive_json()["type"] == "hello.ok"
        ws_fail.send_json({"type": "action.stop", "hostId": str(mac.host_id)})
        ws_fail.receive_json()
    assert failing_runner.calls == 3


def test_short_circuit_wol_promotes_pending_intent_and_dispatches_via_chat_turn() -> None:
    from xuna_backend.modules.actions.application.cross_host_pipelines import (
        handle_host_ready_wake_queue,
    )
    from xuna_backend.modules.actions.domain.wake_queue import InMemoryWakeQueueStore

    llm = _ScriptedLLM(
        [
            [
                ToolCallRequested(
                    ToolCall(id="sc_wol_1", name="app__launch", arguments={"id": "app_x"})
                ),
                _done(),
            ],
            [TextDelta("Launched app_x on WinWol."), _done()],
        ]
    )
    client, issuer, owner, host_repo, approval_repo, _, _, app = _build_mesh_app(llm)
    token = issuer.token("owner-mesh-1")
    headers = {"Authorization": f"Bearer {token}"}
    hub = app.state.host_session_hub
    wake_queue = InMemoryWakeQueueStore()
    app.state.wake_queue = wake_queue
    pending_store = app.state.pending_intent_store

    win = Host(
        host_id=HostId.generate(),
        owner_id=owner.owner_id,
        alias=HostAlias("WinWol"),
        platform=HostPlatform.WINDOWS,
        hostname="winwol.local",
        os_version="11",
        agent_version="0.2.0",
        public_key=FAKE_PUBKEY,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        wol_mac="AA:BB:CC:DD:EE:99",
    )

    async def _seed_offline() -> None:
        await host_repo.save_host(win)
        await host_repo.save_manifest(Manifest(win.host_id, (_app_tool(),), NOW))
        await host_repo.set_grants(
            win.host_id, [Grant(win.host_id, ToolFamily.APP, RiskTier.R2, True, NOW)]
        )

    asyncio.run(_seed_offline())

    # Turn 1: single offline host short-circuits and enqueues DeferredWakeStep(tool="")
    t1_r = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={
            "schemaVersion": 1,
            "text": "launch app_x @WinWol",
            "clientCapabilities": ["host_actions_v1"],
        },
    )
    t1_evs = _parse_sse(t1_r.text)
    t1_tr = next((d for name, d in t1_evs if name == "action_target_required"), None)
    t1_wake = next((d for name, d in t1_evs if name == "host_wake_request"), None)
    assert t1_tr is not None and t1_tr["reasonCode"] == "host_offline"
    assert t1_wake is not None and t1_wake["macAddress"] == "AA:BB:CC:DD:EE:99"
    assert len(wake_queue.list_active(owner.owner_id, NOW)) == 1

    # Host wakes up -> handle_host_ready_wake_queue promotes tool="" step to PendingIntentStore
    win_calls: list[dict[str, object]] = []

    async def _connect_and_promote() -> None:
        async def _send(raw: str) -> None:
            import json as _json

            frame = _json.loads(raw)
            if frame.get("method") == "tools/call":
                win_calls.append(frame["params"])
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}},
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(win.host_id), object(), _send, _close, _close, owner_id=owner.owner_id
        )
        promoted = await handle_host_ready_wake_queue(
            owner_id=owner.owner_id,
            host_id=win.host_id,
            wake_queue=wake_queue,
            pending_intent_store=pending_store,
        )
        assert promoted == 1
        # Never executes R2 app.launch silently on wake
        assert win_calls == []

    asyncio.run(_connect_and_promote())

    # Turn 2: reply "WinWol" without clientCapabilities -> repair_dispatch restores text & tools
    def _approve_when_pending() -> None:
        import time

        for _ in range(200):
            pending_list = list(approval_repo._items.values())
            for req in pending_list:
                if req.state == ApprovalState.PENDING:
                    client.post(
                        f"/v1/actions/approvals/{req.approval_id}",
                        headers=headers,
                        json={"schemaVersion": 1, "decision": "approve"},
                    )
                    return
            time.sleep(0.01)

    import threading

    approver = threading.Thread(target=_approve_when_pending, daemon=True)
    approver.start()
    t2_r = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={"schemaVersion": 1, "text": "WinWol"},
    )
    approver.join(timeout=2.0)
    t2_evs = _parse_sse(t2_r.text)
    pending_ev = next((d for name, d in t2_evs if name == "action_pending"), None)
    result_ev = next((d for name, d in t2_evs if name == "action_result"), None)
    assert pending_ev is not None
    assert result_ev is not None and result_ev["status"] == "ok"
    assert len(win_calls) == 1
    assert win_calls[0]["name"] == "app.launch"
    assert llm.requests[0].messages[-1].content == "launch app_x @WinWol"


def test_wol_multi_host_partial_offline_full_queue_redaction_and_expert_repair_mode() -> None:
    import json as _json

    from xuna_backend.modules.actions.application.cross_host_pipelines import (
        handle_host_ready_wake_queue,
    )
    from xuna_backend.modules.actions.application.mesh_call_helper import (
        handle_pre_route_target_required,
    )
    from xuna_backend.modules.actions.application.mesh_router import (
        PreRouteOutcome,
        TargetReason,
        TargetReasonCode,
    )
    from xuna_backend.modules.actions.domain.entities import HostWakeRequestEvent
    from xuna_backend.modules.actions.domain.mesh_scoring import (
        CandidateUnavailableReason,
        ScoredHostCandidate,
    )
    from xuna_backend.modules.actions.domain.wake_queue import InMemoryWakeQueueStore

    chat_llm = _ScriptedLLM([[TextDelta("Chat reply"), _done()]])
    code_llm = _ScriptedLLM(
        [
            [TextDelta("Expert HTTP code reply"), _done()],
            [TextDelta("Expert WS code reply"), _done()],
        ]
    )
    client, issuer, owner, host_repo, _, _, _, app = _build_mesh_app(chat_llm)
    app.state.code_text_generator = code_llm
    app.state.expert_text_generator = code_llm
    token = issuer.token("owner-mesh-1")
    headers = {"Authorization": f"Bearer {token}"}
    hub = app.state.host_session_hub
    wake_queue = InMemoryWakeQueueStore()
    app.state.wake_queue = wake_queue
    pending_store = app.state.pending_intent_store

    mac = _make_host(owner.owner_id, "MacOnline", platform=HostPlatform.MACOS)
    win = Host(
        host_id=HostId.generate(),
        owner_id=owner.owner_id,
        alias=HostAlias("WinSleep"),
        platform=HostPlatform.WINDOWS,
        hostname="winsleep.local",
        os_version="11",
        agent_version="0.2.0",
        public_key=FAKE_PUBKEY,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        wol_mac="AA:BB:CC:DD:EE:44",
    )

    async def _seed() -> None:
        await host_repo.save_host(mac)
        await host_repo.save_host(win)
        await host_repo.save_manifest(Manifest(mac.host_id, (_app_tool(),), NOW))
        await host_repo.save_manifest(Manifest(win.host_id, (_app_tool(),), NOW))
        await host_repo.set_grants(
            mac.host_id, [Grant(mac.host_id, ToolFamily.APP, RiskTier.R2, True, NOW)]
        )
        await host_repo.set_grants(
            win.host_id, [Grant(win.host_id, ToolFamily.APP, RiskTier.R2, True, NOW)]
        )
        await _connect_fake_host(hub, owner.owner_id, mac)

    asyncio.run(_seed())

    # 1. (8f + 8c) Multi-host partial-offline WoL with online MacOnline & secret in text
    secret = "sk-" + "multiHostSecret123456"
    t_multi = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={
            "schemaVersion": 1,
            "text": f"@WinSleep hello {secret}",
            "clientCapabilities": ["host_actions_v1"],
        },
    )
    multi_evs = _parse_sse(t_multi.text)
    tr_multi = next((d for name, d in multi_evs if name == "action_target_required"), None)
    wake_multi = next((d for name, d in multi_evs if name == "host_wake_request"), None)
    assert tr_multi is not None and tr_multi["reasonCode"] == "host_offline"
    assert wake_multi is not None and wake_multi["macAddress"] == "AA:BB:CC:DD:EE:44"
    steps_1 = wake_queue.list_active(owner.owner_id, NOW)
    assert len(steps_1) == 1
    assert secret not in steps_1[0].summary
    assert "[REDACTED]" in steps_1[0].summary
    assert len(chat_llm.requests) == 0

    # 2. (8b) Fill remaining 2 slots; 4th offline turn surfaces wake_queue_full & suppresses wake
    for idx in range(2):
        r_fill = client.post(
            f"/v1/conversations/{PET_HEX}/turns",
            headers=headers,
            json={
                "schemaVersion": 1,
                "text": f"@WinSleep step {idx}",
                "clientCapabilities": ["host_actions_v1"],
            },
        )
        assert any(name == "host_wake_request" for name, _ in _parse_sse(r_fill.text))

    r_full = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={
            "schemaVersion": 1,
            "text": "@WinSleep overflow",
            "clientCapabilities": ["host_actions_v1"],
        },
    )
    full_evs = _parse_sse(r_full.text)
    tr_full = next((d for name, d in full_evs if name == "action_target_required"), None)
    assert tr_full is not None
    assert tr_full["reasonCode"] == "host_offline"
    assert tr_full.get("errorCode") == "wake_queue_full"
    assert "wake_queue_full" in tr_full["targetReason"]
    assert not any(name == "host_wake_request" for name, _ in full_evs)

    # Also verify mesh_call_helper surfaces wake_queue_full in both event and tool payload
    async def _check_call_helper_full() -> None:
        helper_evs: list[object] = []

        async def _emit(ev: object) -> None:
            helper_evs.append(ev)

        cand = ScoredHostCandidate(
            host=win,
            score=0.0,
            eligible=False,
            online=False,
            unavailable_reason=CandidateUnavailableReason.HOST_OFFLINE,
            reason="WinSleep is offline",
            wol_mac=win.wol_mac,
            wol_available=True,
        )
        outcome = PreRouteOutcome(
            kind="target_required",
            reason_code="host_offline",
            target_reason=TargetReason(
                reason_code=TargetReasonCode.HOST_OFFLINE,
                summary="WinSleep is offline",
            ),
            candidates=(cand,),
        )
        msg = await handle_pre_route_target_required(
            outcome=outcome,
            call_id="helper_full",
            tool_name="app.launch",
            clean_args={"id": "app_x"},
            reference_descriptor=_app_tool(),
            owner_id=owner.owner_id,
            pet_id=PET_HEX,
            emit_event=_emit,
            pending_store=pending_store,
            now=NOW,
            client_caps=("host_actions_v1",),
            wake_queue=wake_queue,
        )
        payload = _json.loads(msg.content)
        assert payload["status"] == "host_offline"
        assert payload["errorCode"] == "wake_queue_full"
        assert "wake_queue_full" in payload["targetReason"]
        assert not any(isinstance(e, HostWakeRequestEvent) for e in helper_evs)
        tr_ev = next(e for e in helper_evs if isinstance(e, ActionTargetRequiredEvent))
        assert tr_ev.error_code == "wake_queue_full"
        assert "wake_queue_full" in tr_ev.target_reason

    asyncio.run(_check_call_helper_full())

    # 3. (8d) HTTP + WS Turn 2 repair_dispatch routes code/expert turns to TurnMode.EXPERT
    wake_queue.clear_owner(owner.owner_id)
    pending_store.clear(owner.owner_id, PET_HEX)

    expert_prompt = "viết script python kiểm tra log @WinSleep"
    t1_http = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={
            "schemaVersion": 1,
            "text": expert_prompt,
            "clientCapabilities": ["host_actions_v1"],
        },
    )
    assert any(name == "host_wake_request" for name, _ in _parse_sse(t1_http.text))

    win_token = object()

    async def _wake_win() -> None:
        async def _noop_send(_txt: str) -> None:
            return None

        async def _noop_close() -> None:
            return None

        await hub.open_session(
            str(win.host_id),
            win_token,
            _noop_send,
            _noop_close,
            _noop_close,
            owner_id=owner.owner_id,
        )
        promoted = await handle_host_ready_wake_queue(
            owner_id=owner.owner_id,
            host_id=win.host_id,
            wake_queue=wake_queue,
            pending_intent_store=pending_store,
        )
        assert promoted == 1

    asyncio.run(_wake_win())

    t2_http = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        headers=headers,
        json={"schemaVersion": 1, "text": "WinSleep", "mode": "auto"},
    )
    t2_http_evs = _parse_sse(t2_http.text)
    t2_done = next((d for name, d in t2_http_evs if name == "finished"), None)
    assert t2_done is not None and t2_done["mode"] == "expert"
    assert len(code_llm.requests) == 1
    assert code_llm.requests[0].messages[-1].content == expert_prompt

    # Now test the same flow over WebSocket /v1/voice
    hub.close_session(str(win.host_id), win_token)
    wake_queue.clear_owner(owner.owner_id)
    pending_store.clear(owner.owner_id, PET_HEX)

    with client.websocket_connect("/v1/voice") as ws:
        ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": token},
                "petId": PET_HEX,
                "speak": False,
                "clientCapabilities": ["host_actions_v1"],
            }
        )
        assert ws.receive_json()["type"] == "hello.ok"

        ws.send_json({"type": "turn.commit", "text": expert_prompt})
        ws_t1_evs: list[dict[str, Any]] = []
        while True:
            ev = ws.receive_json()
            ws_t1_evs.append(ev)
            if ev["type"] == "reply.finished":
                break
        assert any(e["type"] == "host_wake_request" for e in ws_t1_evs)

        asyncio.run(_wake_win())

        ws.send_json({"type": "turn.commit", "text": "WinSleep", "mode": "auto"})
        ws_t2_evs: list[dict[str, Any]] = []
        while True:
            ev = ws.receive_json()
            ws_t2_evs.append(ev)
            if ev["type"] == "reply.finished":
                break
        fin_ev = next(e for e in ws_t2_evs if e["type"] == "reply.finished")
        assert fin_ev["mode"] == "expert"
        assert len(code_llm.requests) == 2
        assert code_llm.requests[1].messages[-1].content == expert_prompt
        ws.send_json({"type": "bye"})


def test_tier5_schema_version_rejects_boolean_across_http_and_ws_models() -> None:
    """Tier 5 (Gap 9): schemaVersion=True must be rejected while valid enums/datetimes parse."""
    import pytest
    from pydantic import ValidationError

    from xuna_backend.modules.actions.api.mesh_routes import SetHostFocusBody
    from xuna_backend.modules.conversation.api.routes import TurnBody
    from xuna_backend.modules.conversation.application.mesh_turn_helper import parse_ws_hello
    from xuna_backend.modules.hosts.api.schemas import PatchHostRequest

    for cls, payload in (
        (SetHostFocusBody, {"schemaVersion": True}),
        (PatchHostRequest, {"schemaVersion": True, "alias": "Mac"}),
        (TurnBody, {"schemaVersion": True, "text": "hello"}),
    ):
        with pytest.raises(ValidationError):
            cls.model_validate(payload)

    with pytest.raises(ValidationError):
        parse_ws_hello({"type": "hello", "schemaVersion": True, "auth": {}, "petId": "p1"})

    # Valid JSON enums & datetimes still parse cleanly without strict-mode breakage
    tb = TurnBody.model_validate(
        {
            "schemaVersion": 1,
            "text": "hello",
            "locale": "vi",
            "clientTime": "2026-11-21T07:00:00Z",
        }
    )
    assert tb.schemaVersion == 1
    assert tb.locale is not None and tb.locale.value == "vi"
    assert tb.clientTime is not None
