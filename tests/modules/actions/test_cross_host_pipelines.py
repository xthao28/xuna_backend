"""Integration tests for M4.2 Phase 3 Cross-Machine Superpowers (pipelines, WoL, R3, Battery)."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.helpers import MemoryAuditLog, SimulatedDevice

from xuna_backend.modules.actions.application.approval_and_board_service import (
    ApprovalAndBoardService,
)
from xuna_backend.modules.actions.application.catalog import (
    CAP_HOST_ACTIONS_V1,
    MAX_CATALOG_TOOLS,
    build_tool_catalog,
)
from xuna_backend.modules.actions.application.compactor import SubGoalCompactor
from xuna_backend.modules.actions.application.cross_host_pipelines import (
    handle_host_ready_wake_queue,
)
from xuna_backend.modules.actions.application.job_runner import (
    AutonomousJobRunner,
    OrchestratorJobStepExecutor,
)
from xuna_backend.modules.actions.application.mesh_router import InMemoryPendingIntentStore
from xuna_backend.modules.actions.application.orchestrator import ActionContext, Orchestrator
from xuna_backend.modules.actions.application.planner_executor import JobPlannerExecutor
from xuna_backend.modules.actions.application.tool_discovery import (
    MESH_CLIPBOARD_TRANSFER_FUNCTION,
    MESH_FIND_RESOURCE_FUNCTION,
    MESH_FLEET_GIT_STATUS_FUNCTION,
    MESH_FLEET_VITALS_FUNCTION,
    MESH_SMART_PAUSE_MEDIA_FUNCTION,
    MESH_STATUS_FUNCTION,
    MESH_TELEPORT_FUNCTION,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionResultEvent,
    ActionStreamEvent,
    ActionTargetRequiredEvent,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScope,
    HostWakeRequestEvent,
    LocalActionRequestEvent,
)
from xuna_backend.modules.actions.domain.job import AutonomousJobState
from xuna_backend.modules.actions.domain.redaction import MAX_CLIPBOARD_WRITE_CHARS
from xuna_backend.modules.actions.domain.wake_queue import (
    MAX_DEFERRED_WAKE_STEPS_PER_OWNER,
    WAKE_QUEUE_TTL,
    DeferredWakeStep,
    InMemoryWakeQueueStore,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.job_store import InMemoryAutonomousJobStore
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.domain.crypto import approval_proof_message, args_sha256
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
from xuna_backend.modules.hosts.domain.twin import HostPresence, HostResources
from xuna_backend.modules.hosts.infrastructure.session_hub import InMemoryHostSessionHub
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier

NOW = datetime(2026, 11, 20, 18, 0, 0, tzinfo=UTC)
PUB = b"\x04" + b"\x22" * 64


class _MemHostRepo:
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

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifests.get(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        self.manifests[manifest.host_id] = manifest

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self.grants.get(host_id, []))

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self.grants[host_id] = list(grants)

    async def suspend_r2_plus_grants_for_owner(self, owner_id: uuid.UUID, now: datetime) -> int:
        return 0


def _host(
    owner_id: uuid.UUID,
    alias: str,
    platform: HostPlatform,
    *,
    kind: HostKind = HostKind.DESKTOP,
    public_key: bytes = PUB,
    wol_mac: str | None = None,
) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        host_kind=kind,
        hostname=alias.lower(),
        os_version="15.0",
        agent_version="0.2.0",
        public_key=public_key,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        wol_mac=wol_mac,
    )


def _tool(
    name: str,
    family: ToolFamily,
    risk: RiskTier,
    *,
    observes: bool = False,
    idempotent: bool | None = None,
    props: dict[str, object] | None = None,
    required: list[str] | None = None,
) -> ToolDescriptor:
    schema: dict[str, object] = {
        "type": "object",
        "properties": props or {},
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return ToolDescriptor(
        name=name,
        description=f"Tool {name}",
        input_schema=schema,
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=family,
        risk=risk,
        observes=observes,
        idempotent=idempotent,
    )


def _all_grants(host_id: HostId, max_risk: RiskTier = RiskTier.R3) -> list[Grant]:
    return [
        Grant(host_id=host_id, family=fam, max_risk=max_risk, enabled=True, updated_at=NOW)
        for fam in (
            ToolFamily.HOST,
            ToolFamily.MEDIA,
            ToolFamily.APP,
            ToolFamily.URL,
            ToolFamily.CU,
            ToolFamily.INPUT,
        )
    ]


def _make_orch(
    *,
    repo: _MemHostRepo,
    hub: InMemoryHostSessionHub,
    audit: MemoryAuditLog,
    approval_repo: InMemoryApprovalRepository | None = None,
    approval_broker: InMemoryApprovalBroker | None = None,
    local_broker: InMemoryLocalResultBroker | None = None,
    scope_store: InMemoryApprovalScopeStore | None = None,
    pending_store: InMemoryPendingIntentStore | None = None,
    wake_queue: InMemoryWakeQueueStore | None = None,
    verifier: P256SignatureVerifier | None = None,
) -> Orchestrator:
    return Orchestrator(
        host_repo=repo,  # type: ignore[arg-type]
        hub=hub,  # type: ignore[arg-type]
        approval_repo=approval_repo or InMemoryApprovalRepository(),
        approval_broker=approval_broker or InMemoryApprovalBroker(),
        local_result_broker=local_broker or InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=audit,
        scope_store=scope_store or InMemoryApprovalScopeStore(),
        verifier=verifier,
        pending_intent_store=pending_store,
        wake_queue=wake_queue,
        clock=lambda: NOW,
    )


def test_catalog_pinned_hides_discovery_but_exposes_5_cross_host_pipelines() -> None:
    owner = uuid.uuid4()
    mac = _host(owner, "Mac", HostPlatform.MACOS)
    win = _host(owner, "Win", HostPlatform.WINDOWS)
    m_mac = Manifest(
        host_id=mac.host_id,
        tools=(_tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1),),
        updated_at=NOW,
    )
    m_win = Manifest(
        host_id=win.host_id,
        tools=(_tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1),),
        updated_at=NOW,
    )
    grants = {mac.host_id: _all_grants(mac.host_id), win.host_id: _all_grants(win.host_id)}

    pinned = build_tool_catalog(
        hosts=[mac, win],
        manifests_by_host={mac.host_id: m_mac, win.host_id: m_win},
        grants_by_host=grants,
        online_host_ids={str(mac.host_id), str(win.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1,),
        pinned_host_id=win.host_id,
        include_mesh_tools=True,
    )
    names = {s.name for s in pinned}
    assert len(pinned) <= MAX_CATALOG_TOOLS
    assert MESH_STATUS_FUNCTION not in names
    assert MESH_FIND_RESOURCE_FUNCTION not in names
    assert {
        MESH_CLIPBOARD_TRANSFER_FUNCTION,
        MESH_TELEPORT_FUNCTION,
        MESH_SMART_PAUSE_MEDIA_FUNCTION,
        MESH_FLEET_VITALS_FUNCTION,
        MESH_FLEET_GIT_STATUS_FUNCTION,
    } <= names


def test_clipboard_transfer_desktop_to_phone_local_action_dual_redaction() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()

        mac = _host(owner, "MacDesk", HostPlatform.MACOS)
        phone = _host(owner, "OwnerPhone", HostPlatform.IOS, kind=HostKind.MOBILE)
        await repo.save_host(mac)
        await repo.save_host(phone)
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool(
                        "selection.read",
                        ToolFamily.APP,
                        RiskTier.R1,
                        observes=True,
                        props={"maxChars": {"type": "integer", "minimum": 80, "maximum": 1200}},
                    ),
                ),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(Manifest(host_id=phone.host_id, tools=(), updated_at=NOW))
        await repo.set_grants(mac.host_id, _all_grants(mac.host_id))
        await repo.set_grants(phone.host_id, _all_grants(phone.host_id))

        secret = "sk-" + "liveSecret987654321"
        raw_payload = f"Token={secret} " + ("m" * 1350)

        async def _send(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "ok"},
                        "structuredContent": {"text": raw_payload},
                    },
                )

        async def _close() -> None:
            return None

        await hub.open_session(str(mac.host_id), object(), _send, _close, _close, owner_id=owner)

        orch = _make_orch(
            repo=repo,
            hub=hub,
            audit=audit,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_broker=local_broker,
        )

        events: list[ActionStreamEvent] = []
        local_calls: list[LocalActionRequestEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                assert ev.mission_envelope is not None
                assert len(ev.mission_envelope["steps"]) == 2  # type: ignore[arg-type]
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )
            elif isinstance(ev, LocalActionRequestEvent):
                local_calls.append(ev)
                local_broker.report(ev.invocation_id, owner, "ok", {"ok": True})

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
            self_host_id=str(phone.host_id),
        )
        msg = await orch.handle_call(
            ToolCall(
                id="cb_phone",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"sourceHost": "MacDesk", "targetHost": "OwnerPhone"},
            ),
            ctx,
        )
        out = json.loads(msg.content)
        assert out["status"] == "ok"
        assert len(local_calls) == 1
        written = str(local_calls[0].arguments["text"])
        assert len(written) <= 1200
        assert secret not in written
        assert "[REDACTED]" in written
        assert all(secret not in json.dumps(e.metadata) for e in audit.events)

        # Reject binary/NUL payload cleanly with invalid_args
        bad_msg = await orch.handle_call(
            ToolCall(
                id="cb_bad",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"targetHost": "OwnerPhone", "text": "bad\x00blob"},
            ),
            ctx,
        )
        assert json.loads(bad_msg.content)["status"] == "invalid_args"

    asyncio.run(_run())


def test_teleport_and_smart_fanout_vitals_and_git_summary_isolation() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()

        mac = _host(owner, "MacA", HostPlatform.MACOS)
        win = _host(owner, "WinB", HostPlatform.WINDOWS)
        lin = _host(owner, "LinC", HostPlatform.LINUX, kind=HostKind.SERVER)
        for h in (mac, win, lin):
            await repo.save_host(h)
            await repo.save_manifest(
                Manifest(
                    host_id=h.host_id,
                    tools=(
                        _tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1, idempotent=False),
                        _tool("media.now_playing", ToolFamily.MEDIA, RiskTier.R0),
                        _tool("system.vitals", ToolFamily.HOST, RiskTier.R1, observes=True),
                        _tool(
                            "workspace.git_summary",
                            ToolFamily.APP,
                            RiskTier.R1,
                            observes=True,
                            props={"project": {"type": "string"}},
                            required=["project"],
                        ),
                    ),
                    updated_at=NOW,
                )
            )
        await repo.set_grants(mac.host_id, _all_grants(mac.host_id))
        await repo.set_grants(win.host_id, _all_grants(win.host_id))
        # LinC lacks app.R1 so fleet_git_status denies LinC without failing MacA/WinB
        await repo.set_grants(
            lin.host_id,
            [
                Grant(
                    host_id=lin.host_id,
                    family=ToolFamily.HOST,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )

        git_projects_called: list[tuple[str, str]] = []

        def _make_send(h: Host) -> Any:
            async def _send(raw: str) -> None:
                frame = json.loads(raw)
                if frame.get("method") != "tools/call":
                    return
                params = frame["params"]
                tname = params["name"]
                if tname == "media.now_playing":
                    hub.on_response(
                        str(h.host_id),
                        frame["id"],
                        {"_meta": {"xuna/status": "ok"}, "structuredContent": {"playing": True}},
                    )
                elif tname == "workspace.git_summary":
                    proj = str(params["arguments"].get("project", ""))
                    git_projects_called.append((h.alias.value, proj))
                    hub.on_response(
                        str(h.host_id),
                        frame["id"],
                        {
                            "_meta": {"xuna/status": "ok"},
                            "structuredContent": {"branch": "main", "dirty": False},
                        },
                    )
                else:
                    hub.on_response(
                        str(h.host_id),
                        frame["id"],
                        {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}},
                    )

            return _send

        async def _close() -> None:
            return None

        for h in (mac, win, lin):
            await hub.open_session(
                str(h.host_id), object(), _make_send(h), _close, _close, owner_id=owner
            )

        hub.update_presence(
            str(mac.host_id),
            HostPresence(
                idle_seconds=5,
                screen_locked=False,
                media_playing=False,
                updated_at=NOW,
                battery_percent=12,
                is_charging=False,
            ),
        )
        hub.update_presence(
            str(win.host_id),
            HostPresence(
                idle_seconds=5,
                screen_locked=False,
                media_playing=False,
                updated_at=NOW,
                battery_percent=None,
                is_charging=None,
            ),
        )
        hub.update_resources(
            str(mac.host_id),
            HostResources(installed_apps=(), workspaces=("repo_alpha",), updated_at=NOW),
        )
        hub.set_focus_state(owner, focused_host_id=mac.host_id, source="app", now=NOW)

        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
            pinned_host_id=mac.host_id,
        )

        # 1. smart_pause_media: MacA/WinB not playing, LinC ungranted -> no_active_media
        pause_msg = await orch.handle_call(
            ToolCall(id="p1", name=MESH_SMART_PAUSE_MEDIA_FUNCTION, arguments={}), ctx
        )
        assert json.loads(pause_msg.content)["status"] == "no_active_media"
        focus_after = hub.get_focus_state(owner)
        assert focus_after is not None and focus_after.focused_host_id == mac.host_id

        # 2. fleet_vitals: surfaces 12% unplugged MacA + WinB/LinC alternatives
        vitals_msg = await orch.handle_call(
            ToolCall(id="v1", name=MESH_FLEET_VITALS_FUNCTION, arguments={}), ctx
        )
        vitals = json.loads(vitals_msg.content)
        assert vitals["status"] == "ok"
        assert any(
            "MacA" in a and ("WinB" in a or "LinC" in a) for a in vitals["batteryAdvisories"]
        )

        # 3. fleet_git_status: passes non-empty project (repo_alpha/recent), isolates denied LinC
        git_msg = await orch.handle_call(
            ToolCall(id="g1", name=MESH_FLEET_GIT_STATUS_FUNCTION, arguments={}), ctx
        )
        git_out = json.loads(git_msg.content)
        assert ("MacA", "repo_alpha") in git_projects_called
        assert ("WinB", "recent") in git_projects_called
        by_alias = {r["alias"]: r for r in git_out["results"]}
        assert by_alias["MacA"]["status"] == "ok"
        assert by_alias["LinC"]["status"] == "denied"

    asyncio.run(_run())


def test_teleport_https_guard_and_workspace_capsule_restore() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id))
            await repo.save_manifest(
                Manifest(
                    host_id=h.host_id,
                    tools=(
                        _tool("browser.current_url", ToolFamily.URL, RiskTier.R2, observes=True),
                        _tool(
                            "url.open",
                            ToolFamily.URL,
                            RiskTier.R2,
                            props={"url": {"type": "string"}},
                            required=["url"],
                        ),
                        _tool(
                            "workspace.capture_capsule",
                            ToolFamily.APP,
                            RiskTier.R1,
                            observes=True,
                            props={"project": {"type": "string"}},
                        ),
                        _tool(
                            "workspace.restore_capsule",
                            ToolFamily.APP,
                            RiskTier.R2,
                            props={
                                "capsuleId": {"type": "string"},
                                "project": {"type": "string"},
                            },
                        ),
                    ),
                    updated_at=NOW,
                )
            )

        urls = iter(["https://example.org/docs", "file:///etc/passwd"])
        win_calls: list[dict[str, object]] = []

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") != "tools/call":
                return
            name = frame["params"]["name"]
            if name == "browser.current_url":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"url": next(urls)}},
                )
            else:
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "ok"},
                        "structuredContent": {"capsuleId": "cap_1", "project": "xuna_proj"},
                    },
                )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") != "tools/call":
                return
            win_calls.append(frame["params"])
            hub.on_response(
                str(win.host_id),
                frame["id"],
                {"_meta": {"xuna/status": "ok"}, "structuredContent": {"opened": True}},
            )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )
        hub.update_resources(
            str(mac.host_id),
            HostResources(installed_apps=(), workspaces=("xuna_proj",), updated_at=NOW),
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        ok_msg = await orch.handle_call(
            ToolCall(
                id="tp1",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst"},
            ),
            ctx,
        )
        assert json.loads(ok_msg.content)["status"] == "ok"
        assert {c["name"] for c in win_calls} == {"url.open", "workspace.restore_capsule"}
        pending = next(e for e in events if isinstance(e, ActionPendingEvent))
        assert pending.mission_envelope is not None
        assert len(pending.mission_envelope["steps"]) == 4  # type: ignore[arg-type]

        # Non-HTTPS URL fails closed without invoking anything on WinDst
        before = len(win_calls)
        bad_msg = await orch.handle_call(
            ToolCall(
                id="tp2",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst"},
            ),
            ctx,
        )
        assert json.loads(bad_msg.content)["status"] == "invalid_args"
        assert len(win_calls) == before

    asyncio.run(_run())


def test_r3_signature_required_with_battery_advisory_and_wol_job_waiting_wake() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        phone_dev = SimulatedDevice()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        pending_store = InMemoryPendingIntentStore()
        wake_queue = InMemoryWakeQueueStore()
        verifier = P256SignatureVerifier()

        phone = _host(
            owner,
            "OwnerPhone",
            HostPlatform.IOS,
            kind=HostKind.MOBILE,
            public_key=phone_dev.public_key,
        )
        low_mac = _host(owner, "LowLaptop", HostPlatform.MACOS)
        win_desk = _host(owner, "WinDesktop", HostPlatform.WINDOWS, wol_mac="AA:BB:CC:DD:EE:11")
        for h in (phone, low_mac, win_desk):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id, RiskTier.R3))

        cu_start_tool = _tool(
            "cu.session_start",
            ToolFamily.CU,
            RiskTier.R3,
            observes=True,
            props={
                "goal": {"type": "string"},
                "apps": {"type": "array", "items": {"type": "string"}},
                "domains": {"type": "array", "items": {"type": "string"}},
            },
            required=["goal"],
        )
        await repo.save_manifest(
            Manifest(host_id=low_mac.host_id, tools=(cu_start_tool,), updated_at=NOW)
        )
        await repo.save_manifest(
            Manifest(host_id=win_desk.host_id, tools=(cu_start_tool,), updated_at=NOW)
        )

        async def _send_ok(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(low_mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"sessionId": "s1"}},
                )

        async def _close() -> None:
            return None

        win_token = object()
        await hub.open_session(
            str(low_mac.host_id), object(), _send_ok, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win_desk.host_id), win_token, _send_ok, _close, _close, owner_id=owner
        )
        hub.update_presence(
            str(low_mac.host_id),
            HostPresence(
                idle_seconds=4,
                screen_locked=False,
                media_playing=False,
                updated_at=NOW,
                battery_percent=15,
                is_charging=False,
            ),
        )
        hub.update_presence(
            str(win_desk.host_id),
            HostPresence(
                idle_seconds=4,
                screen_locked=False,
                media_playing=False,
                updated_at=NOW,
                battery_percent=None,
                is_charging=None,
            ),
        )

        orch = _make_orch(
            repo=repo,
            hub=hub,
            audit=audit,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_broker=local_broker,
            pending_store=pending_store,
            wake_queue=wake_queue,
            verifier=verifier,
        )
        svc = ApprovalAndBoardService(
            host_repo=repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            scope_store=None,
            verifier=verifier,
            clock=lambda: NOW,
        )

        pending_events: list[ActionPendingEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            if isinstance(ev, ActionPendingEvent):
                pending_events.append(ev)
                # Missing signature on R3 with enrolled phone MUST fail closed
                with pytest.raises(ValueError, match="approval_signature_invalid"):
                    await svc.decide_approval(
                        approval_id=uuid.UUID(ev.approval_id),
                        owner_id=owner,
                        decision=ApprovalDecision.APPROVE,
                        scope=ApprovalScope.ONCE,
                        signature=None,
                    )
                msg = approval_proof_message(
                    ev.approval_id, ev.args_sha256, ev.host_id or "", ev.expires_at
                )
                sig = phone_dev.sign(msg)
                await svc.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                    scope=ApprovalScope.ONCE,
                    signature=sig,
                    signer_public_key=phone_dev.public_key,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
            pinned_host_id=low_mac.host_id,
        )
        res_msg = await orch.handle_call(
            ToolCall(id="cu1", name="cu__session_start", arguments={"goal": "Heavy CU task"}), ctx
        )
        assert json.loads(res_msg.content)["status"] == "ok"
        assert len(pending_events) == 1
        assert pending_events[0].battery_advisories
        assert "LowLaptop" in pending_events[0].battery_advisories[0]
        assert "WinDesktop" in pending_events[0].battery_advisories[0]

        # Disconnect WinDesktop and verify AutonomousJob transitions to WAITING_WAKE, then resumes
        hub.close_session(str(win_desk.host_id), win_token)
        job_store = InMemoryAutonomousJobStore()

        class _DummyGen:
            async def stream(self, req: Any) -> Any:
                return
                yield req

        step_exec = OrchestratorJobStepExecutor(
            orchestrator_factory=lambda: orch,
            host_repo_factory=lambda: repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            audit_factory=lambda: audit,
            generator_factory=lambda: _DummyGen(),  # type: ignore[arg-type,return-value]
            planner=JobPlannerExecutor(),
            compactor=SubGoalCompactor(),
            wake_queue=wake_queue,
            clock_utc=lambda: NOW,
        )
        runner = AutonomousJobRunner(
            job_store=job_store, step_executor=step_exec, clock_utc=lambda: NOW
        )
        job = await runner.start_job(
            job_id=uuid.uuid4(),
            owner_id=owner,
            host_id=win_desk.host_id,
            goal="Step 1: run\nStep 2: verify",
        )
        await asyncio.sleep(0.05)
        waiting = await job_store.get(job.job_id)
        assert waiting is not None and waiting.state == AutonomousJobState.WAITING_WAKE

        # Also enqueue an interactive DeferredWakeStep and test on_host_ready promotion
        wake_queue.enqueue(
            DeferredWakeStep(
                step_id=uuid.uuid4(),
                owner_id=owner,
                pet_id="pet-1",
                host_id=win_desk.host_id,
                host_alias="WinDesktop",
                tool="app.launch",
                arguments={"id": "app_x"},
                risk=RiskTier.R2,
                created_at=NOW,
                expires_at=NOW + WAKE_QUEUE_TTL,
                summary="Launch app_x",
                client_caps=(CAP_HOST_ACTIONS_V1,),
            ),
            NOW,
        )
        promoted = await handle_host_ready_wake_queue(
            owner_id=owner,
            host_id=win_desk.host_id,
            wake_queue=wake_queue,
            pending_intent_store=pending_store,
            job_runner=runner,
            now=NOW + timedelta(seconds=10),
        )
        assert promoted >= 2
        assert pending_store.get(owner, "pet-1", NOW + timedelta(seconds=10)) is not None
        await runner.cancel_all_for_owner(owner)
        await runner.shutdown()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Iteration 2 Rule H7 Fail-Closed & Negative Regression Tests
# ---------------------------------------------------------------------------


def test_teleport_workspace_mode_without_executable_target_fails_closed() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        await repo.save_host(mac)
        await repo.save_host(win)
        await repo.set_grants(mac.host_id, _all_grants(mac.host_id))
        await repo.set_grants(win.host_id, _all_grants(win.host_id))
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("workspace.capture_capsule", ToolFamily.APP, RiskTier.R1, observes=True),
                ),
                updated_at=NOW,
            )
        )
        # WinDst has app.launch ONLY (no workspace.open or workspace.restore_capsule)
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "app.launch",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"id": {"type": "string"}, "openTarget": {"type": "string"}},
                    ),
                ),
                updated_at=NOW,
            )
        )

        mac_calls: list[str] = []
        win_calls: list[str] = []

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                mac_calls.append(frame["params"]["name"])
                hub.on_response(str(mac.host_id), frame["id"], {"_meta": {"xuna/status": "ok"}})

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                win_calls.append(frame["params"]["name"])
                hub.on_response(str(win.host_id), frame["id"], {"_meta": {"xuna/status": "ok"}})

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )

        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        msg = await orch.handle_call(
            ToolCall(
                id="tp_no_exec",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "workspace"},
            ),
            ctx,
        )
        payload = json.loads(msg.content)
        assert payload["status"] != "ok"
        assert mac_calls == []
        assert win_calls == []
        assert not any(
            e.action == AuditAction.HOST_TOOL_INVOKED and e.metadata.get("tool") == "url.open"
            for e in audit.events
        )

    asyncio.run(_run())


def test_teleport_binds_r2_approval_args_sha256_on_every_step() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id))
            await repo.save_manifest(
                Manifest(
                    host_id=h.host_id,
                    tools=(
                        _tool("browser.current_url", ToolFamily.URL, RiskTier.R2, observes=True),
                        _tool(
                            "url.open",
                            ToolFamily.URL,
                            RiskTier.R2,
                            props={"url": {"type": "string"}},
                            required=["url"],
                        ),
                        _tool(
                            "workspace.capture_capsule",
                            ToolFamily.APP,
                            RiskTier.R1,
                            observes=True,
                        ),
                        _tool(
                            "workspace.restore_capsule",
                            ToolFamily.APP,
                            RiskTier.R2,
                            props={"capsuleId": {"type": "string"}, "project": {"type": "string"}},
                        ),
                    ),
                    updated_at=NOW,
                )
            )

        observed_calls: list[dict[str, Any]] = []

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") != "tools/call":
                return
            observed_calls.append(frame["params"])
            name = frame["params"]["name"]
            sc = (
                {"url": "https://example.org/app"}
                if name == "browser.current_url"
                else {"capsuleId": "cap_9", "project": "proj_x"}
            )
            hub.on_response(
                str(mac.host_id),
                frame["id"],
                {"_meta": {"xuna/status": "ok"}, "structuredContent": sc},
            )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") != "tools/call":
                return
            observed_calls.append(frame["params"])
            hub.on_response(
                str(win.host_id),
                frame["id"],
                {"_meta": {"xuna/status": "ok"}, "structuredContent": {"restored": True}},
            )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )
        hub.update_resources(
            str(mac.host_id),
            HostResources(installed_apps=(), workspaces=("proj_x",), updated_at=NOW),
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        pending_count = 0

        async def _emit(ev: ActionStreamEvent) -> None:
            nonlocal pending_count
            if isinstance(ev, ActionPendingEvent):
                pending_count += 1
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        res = await orch.handle_call(
            ToolCall(
                id="tp_bind",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst"},
            ),
            ctx,
        )
        assert json.loads(res.content)["status"] == "ok"
        assert pending_count == 1
        r2_calls = [c for c in observed_calls if c["name"] != "workspace.capture_capsule"]
        assert {c["name"] for c in r2_calls} == {
            "browser.current_url",
            "url.open",
            "workspace.restore_capsule",
        }
        for c in r2_calls:
            app_meta = c["_meta"].get("xuna/approval")
            assert app_meta is not None, f"Missing xuna/approval on {c['name']}"
            assert app_meta["argsSha256"] == args_sha256(c["arguments"])
            assert datetime.fromisoformat(app_meta["expiresAt"]) > NOW

    asyncio.run(_run())


def test_teleport_r3_target_step_requires_phone_p256_signature_per_step() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        phone_dev = SimulatedDevice()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        verifier = P256SignatureVerifier()

        phone = _host(
            owner,
            "OwnerPhone",
            HostPlatform.IOS,
            kind=HostKind.MOBILE,
            public_key=phone_dev.public_key,
        )
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (phone, mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id, RiskTier.R3))

        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_tool("browser.current_url", ToolFamily.URL, RiskTier.R2, observes=True),),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "url.open",
                        ToolFamily.URL,
                        RiskTier.R3,
                        props={"url": {"type": "string"}},
                        required=["url"],
                    ),
                ),
                updated_at=NOW,
            )
        )

        win_calls: list[dict[str, Any]] = []

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "ok"},
                        "structuredContent": {"url": "https://example.org/r3"},
                    },
                )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                win_calls.append(frame["params"])
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"opened": True}},
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )

        orch = _make_orch(repo=repo, hub=hub, audit=audit, verifier=verifier)

        async def _emit(ev: ActionStreamEvent) -> None:
            if isinstance(ev, ActionPendingEvent):
                if ev.risk == RiskTier.R3.value:
                    with pytest.raises(ValueError, match="approval_signature_invalid"):
                        await orch.decide_approval(
                            approval_id=uuid.UUID(ev.approval_id),
                            owner_id=owner,
                            decision=ApprovalDecision.APPROVE,
                            scope=ApprovalScope.ONCE,
                            signature=None,
                        )
                    msg = approval_proof_message(
                        ev.approval_id, ev.args_sha256, ev.host_id or "", ev.expires_at
                    )
                    sig = phone_dev.sign(msg)
                    await orch.decide_approval(
                        approval_id=uuid.UUID(ev.approval_id),
                        owner_id=owner,
                        decision=ApprovalDecision.APPROVE,
                        scope=ApprovalScope.ONCE,
                        signature=sig,
                        signer_public_key=phone_dev.public_key,
                    )
                else:
                    await orch.decide_approval(
                        approval_id=uuid.UUID(ev.approval_id),
                        owner_id=owner,
                        decision=ApprovalDecision.APPROVE,
                    )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        res = await orch.handle_call(
            ToolCall(
                id="tp_r3",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "url"},
            ),
            ctx,
        )
        assert json.loads(res.content)["status"] == "ok"
        assert len(win_calls) == 1
        app_meta = win_calls[0]["_meta"]["xuna/approval"]
        assert app_meta["argsSha256"] == args_sha256({"url": "https://example.org/r3"})
        proof_msg = approval_proof_message(
            app_meta["approvalId"],
            app_meta["argsSha256"],
            str(win.host_id),
            app_meta["expiresAt"],
        )
        import base64 as _b64

        assert verifier.verify(
            phone_dev.public_key, proof_msg, _b64.b64decode(app_meta["signature"])
        )

    asyncio.run(_run())


def test_teleport_prefers_workspace_open_with_toast_and_surfaces_restore_fields() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        await repo.save_host(mac)
        await repo.save_host(win)
        await repo.set_grants(mac.host_id, _all_grants(mac.host_id))
        await repo.set_grants(win.host_id, _all_grants(win.host_id))
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("workspace.capture_capsule", ToolFamily.APP, RiskTier.R1, observes=True),
                ),
                updated_at=NOW,
            )
        )
        # Case A: M4.1 Desktop-shaped target (workspace.open + restore_capsule + host.notify)
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "workspace.open",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"project": {"type": "string"}},
                    ),
                    _tool(
                        "workspace.restore_capsule",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"capsuleId": {"type": "string"}, "project": {"type": "string"}},
                    ),
                    _tool(
                        "host.notify",
                        ToolFamily.HOST,
                        RiskTier.R1,
                        props={"title": {"type": "string"}, "body": {"type": "string"}},
                    ),
                ),
                updated_at=NOW,
            )
        )

        win_tools_called: list[str] = []

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "ok"},
                        "structuredContent": {
                            "capsuleId": "cap-src-1",
                            "project": "xuna_pet",
                            "branch": "main",
                            "dirtyCount": 2,
                            "nextStepHint": "Review changes",
                        },
                    },
                )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") != "tools/call":
                return
            tname = frame["params"]["name"]
            win_tools_called.append(tname)
            if tname == "workspace.restore_capsule":
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {
                        "_meta": {
                            "xuna/status": "invalid_args",
                            "xuna/errorCode": "capsule_not_found",
                        },
                        "structuredContent": {"errorCode": "capsule_not_found"},
                    },
                )
            else:
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "ok"},
                        "structuredContent": {
                            "restored": True,
                            "capsuleId": "cap-dst-2",
                            "nextStepHint": "Resume build",
                            "briefingVi": "Da khoi phuc",
                        },
                    },
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit)

        async def _emit(ev: ActionStreamEvent) -> None:
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        msg_a = await orch.handle_call(
            ToolCall(
                id="tp_open",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "workspace"},
            ),
            ctx,
        )
        out_a = json.loads(msg_a.content)
        assert out_a["status"] == "ok"
        assert win_tools_called == ["workspace.open", "host.notify"]
        assert out_a["output"]["capsuleId"] == "cap-src-1"
        assert out_a["output"]["nextStepHint"] == "Review changes"

        # Case B: Target with only workspace.restore_capsule fails honestly on capsule_not_found
        win_tools_called.clear()
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "workspace.restore_capsule",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"capsuleId": {"type": "string"}, "project": {"type": "string"}},
                    ),
                ),
                updated_at=NOW,
            )
        )
        msg_b = await orch.handle_call(
            ToolCall(
                id="tp_restore_fail",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "workspace"},
            ),
            ctx,
        )
        out_b = json.loads(msg_b.content)
        assert out_b["status"] == "invalid_args"
        assert out_b["output"]["errorCode"] == "capsule_not_found"

    asyncio.run(_run())


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://example.org/a\nb",
        "https://example.org/a\rb",
        "https://example.org/a\tb",
        "https://example.org/a\x07b",
        "https://example.org/a\x1bb",
        "https://example.org/a\x7fb",
    ],
)
def test_teleport_rejects_url_control_characters_without_target_dispatch(bad_url: str) -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id))
            await repo.save_manifest(
                Manifest(
                    host_id=h.host_id,
                    tools=(
                        _tool("browser.current_url", ToolFamily.URL, RiskTier.R1, observes=True),
                        _tool(
                            "url.open",
                            ToolFamily.URL,
                            RiskTier.R2,
                            props={"url": {"type": "string"}},
                            required=["url"],
                        ),
                    ),
                    updated_at=NOW,
                )
            )
        win_calls: list[str] = []

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"url": bad_url}},
                )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                win_calls.append(frame["params"]["name"])

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit)

        async def _emit(ev: ActionStreamEvent) -> None:
            return None

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        msg = await orch.handle_call(
            ToolCall(
                id="tp_ctrl",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "url"},
            ),
            ctx,
        )
        out = json.loads(msg.content)
        assert out["status"] == "invalid_args"
        assert out["output"]["errorCode"] == "url_invalid"
        assert win_calls == []

    asyncio.run(_run())


def test_smart_pause_media_all_playing_fail_returns_failed_not_ok() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacA", HostPlatform.MACOS)
        await repo.save_host(mac)
        await repo.set_grants(mac.host_id, _all_grants(mac.host_id))
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1, idempotent=False),),
                updated_at=NOW,
            )
        )

        async def _send(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "failed", "xuna/errorCode": "media_error"},
                        "structuredContent": {"errorCode": "media_error"},
                    },
                )

        async def _close() -> None:
            return None

        await hub.open_session(str(mac.host_id), object(), _send, _close, _close, owner_id=owner)
        hub.update_presence(
            str(mac.host_id),
            HostPresence(idle_seconds=2, screen_locked=False, media_playing=True, updated_at=NOW),
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        msg = await orch.handle_call(
            ToolCall(id="sp_fail", name=MESH_SMART_PAUSE_MEDIA_FUNCTION, arguments={}), ctx
        )
        payload = json.loads(msg.content)
        assert payload["status"] == "failed"
        assert payload["errorCode"] == "media_error"
        res_ev = next(e for e in events if isinstance(e, ActionResultEvent))
        assert res_ev.status == "failed"
        assert res_ev.gesture == "look_around"
        assert res_ev.error_code == "media_error"
        assert res_ev.mission_envelope is not None
        step0 = res_ev.mission_envelope["steps"][0]  # type: ignore[index]
        assert step0["status"] == "failed"
        assert "Paused media on" not in step0["summary"]

    asyncio.run(_run())


def test_smart_pause_media_timeout_and_partial_success() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacA", HostPlatform.MACOS)
        win = _host(owner, "WinB", HostPlatform.WINDOWS)
        for h in (mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id))
            await repo.save_manifest(
                Manifest(
                    host_id=h.host_id,
                    tools=(
                        _tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1, idempotent=False),
                    ),
                    updated_at=NOW,
                )
            )

        mode_box = {"win_status": "timeout", "mac_status": "timeout"}

        def _make_send(h: Host, key: str) -> Any:
            async def _send(raw: str) -> None:
                frame = json.loads(raw)
                if frame.get("method") == "tools/call":
                    st = mode_box[key]
                    hub.on_response(
                        str(h.host_id),
                        frame["id"],
                        {"_meta": {"xuna/status": st, "xuna/errorCode": st}},
                    )

            return _send

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id),
            object(),
            _make_send(mac, "mac_status"),
            _close,
            _close,
            owner_id=owner,
        )
        await hub.open_session(
            str(win.host_id),
            object(),
            _make_send(win, "win_status"),
            _close,
            _close,
            owner_id=owner,
        )
        for h in (mac, win):
            hub.update_presence(
                str(h.host_id),
                HostPresence(
                    idle_seconds=2, screen_locked=False, media_playing=True, updated_at=NOW
                ),
            )
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        msg_to = await orch.handle_call(
            ToolCall(id="sp_to", name=MESH_SMART_PAUSE_MEDIA_FUNCTION, arguments={}), ctx
        )
        out_to = json.loads(msg_to.content)
        assert out_to["status"] == "timeout"
        assert out_to["errorCode"] == "timeout"
        assert next(e for e in events if isinstance(e, ActionResultEvent)).status == "timeout"

        events.clear()
        mode_box["mac_status"] = "ok"
        mode_box["win_status"] = "failed"
        msg_partial = await orch.handle_call(
            ToolCall(id="sp_part", name=MESH_SMART_PAUSE_MEDIA_FUNCTION, arguments={}), ctx
        )
        out_part = json.loads(msg_partial.content)
        assert out_part["status"] == "ok"
        by_alias = {p["alias"]: p["status"] for p in out_part["pausedHosts"]}
        assert by_alias == {"MacA": "ok", "WinB": "failed"}

    asyncio.run(_run())


def test_smart_pause_media_all_denied_aligns_payload_and_event_status() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacDenied", HostPlatform.MACOS)
        await repo.save_host(mac)
        await repo.set_grants(mac.host_id, [])
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1, idempotent=False),),
                updated_at=NOW,
            )
        )

        async def _send(_raw: str) -> None:
            return None

        async def _close() -> None:
            return None

        await hub.open_session(str(mac.host_id), object(), _send, _close, _close, owner_id=owner)
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        msg = await orch.handle_call(
            ToolCall(id="sp_den", name=MESH_SMART_PAUSE_MEDIA_FUNCTION, arguments={}), ctx
        )
        payload = json.loads(msg.content)
        assert payload["status"] == "denied"
        assert payload["errorCode"] == "grant_denied"
        res_ev = next(e for e in events if isinstance(e, ActionResultEvent))
        assert res_ev.status == "denied"

    asyncio.run(_run())


@pytest.mark.parametrize(
    "fn_name",
    [
        MESH_SMART_PAUSE_MEDIA_FUNCTION,
        MESH_FLEET_VITALS_FUNCTION,
        MESH_FLEET_GIT_STATUS_FUNCTION,
    ],
)
def test_fanout_invalid_host_ids_returns_invalid_args(fn_name: str) -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        orch = _make_orch(repo=repo, hub=hub, audit=audit)

        async def _emit(_ev: ActionStreamEvent) -> None:
            return None

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        msg = await orch.handle_call(
            ToolCall(id="bad_ids", name=fn_name, arguments={"hostIds": ["not-a-uuid"]}), ctx
        )
        assert json.loads(msg.content)["status"] == "invalid_args"

    asyncio.run(_run())


def test_fleet_vitals_and_git_status_zero_success_return_honest_status() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacV", HostPlatform.MACOS)
        await repo.save_host(mac)
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("system.vitals", ToolFamily.HOST, RiskTier.R1, observes=True),
                    _tool(
                        "workspace.git_summary",
                        ToolFamily.APP,
                        RiskTier.R1,
                        observes=True,
                        props={"project": {"type": "string"}},
                    ),
                ),
                updated_at=NOW,
            )
        )
        await repo.set_grants(mac.host_id, _all_grants(mac.host_id))

        mode_box = {"status": "timeout"}

        async def _send(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                st = mode_box["status"]
                if st == "timeout":
                    raise TimeoutError("probe timeout")
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": st, "xuna/errorCode": st}},
                )

        async def _close() -> None:
            return None

        await hub.open_session(str(mac.host_id), object(), _send, _close, _close, owner_id=owner)
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        # 1. All denied (U5 consent false)
        ctx_denied = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
            host_screen_consent=False,
        )
        for fn in (MESH_FLEET_VITALS_FUNCTION, MESH_FLEET_GIT_STATUS_FUNCTION):
            events.clear()
            msg = await orch.handle_call(
                ToolCall(id=f"{fn}_den", name=fn, arguments={}), ctx_denied
            )
            out = json.loads(msg.content)
            assert out["status"] == "denied"
            assert events[-1].status == "denied"  # type: ignore[union-attr]

        # 2. All timeout / failed
        ctx_ok = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
            host_screen_consent=True,
        )
        for expected_st in ("timeout", "failed"):
            mode_box["status"] = expected_st
            for fn in (MESH_FLEET_VITALS_FUNCTION, MESH_FLEET_GIT_STATUS_FUNCTION):
                events.clear()
                msg = await orch.handle_call(
                    ToolCall(id=f"{fn}_{expected_st}", name=fn, arguments={}), ctx_ok
                )
                out = json.loads(msg.content)
                assert out["status"] == expected_st
                assert out["errorCode"] == expected_st
                assert events[-1].status == expected_st  # type: ignore[union-attr]

        # 3. 0-host filter
        for fn in (MESH_FLEET_VITALS_FUNCTION, MESH_FLEET_GIT_STATUS_FUNCTION):
            events.clear()
            msg = await orch.handle_call(
                ToolCall(id=f"{fn}_empty", name=fn, arguments={"hostIds": [str(uuid.uuid4())]}),
                ctx_ok,
            )
            out = json.loads(msg.content)
            assert out["status"] == "failed"
            assert out["errorCode"] == "no_host"
            assert events[-1].status == "failed"  # type: ignore[union-attr]

    asyncio.run(_run())


def test_invoke_host_tool_normalizes_malformed_xuna_status_for_audit() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        await repo.save_host(mac)
        await repo.set_grants(mac.host_id, _all_grants(mac.host_id))
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool(
                        "clipboard.write",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"text": {"type": "string"}},
                        required=["text"],
                    ),
                ),
                updated_at=NOW,
            )
        )

        async def _send(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "bad status! 🚀", "xuna/errorCode": "weird_err"},
                        "structuredContent": {},
                    },
                )

        async def _close() -> None:
            return None

        await hub.open_session(str(mac.host_id), object(), _send, _close, _close, owner_id=owner)
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        msg = await orch.handle_call(
            ToolCall(
                id="cb_malformed",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"targetHost": "MacSrc", "text": "hello"},
            ),
            ctx,
        )
        out = json.loads(msg.content)
        assert out["status"] == "failed"
        res_ev = next(e for e in events if isinstance(e, ActionResultEvent))
        assert res_ev.status == "failed"
        assert audit.events[-1].metadata["status"] == "failed"

    asyncio.run(_run())


def test_clipboard_transfer_failed_source_read_strips_secrets_and_inline_4000_cap_and_r3() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        phone_dev = SimulatedDevice()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        verifier = P256SignatureVerifier()

        phone = _host(
            owner,
            "OwnerPhone",
            HostPlatform.IOS,
            kind=HostKind.MOBILE,
            public_key=phone_dev.public_key,
        )
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (phone, mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id, RiskTier.R3))

        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("selection.read", ToolFamily.APP, RiskTier.R1, observes=True),
                    _tool("clipboard.read", ToolFamily.APP, RiskTier.R2, observes=True),
                ),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "clipboard.write",
                        ToolFamily.APP,
                        RiskTier.R3,
                        props={"text": {"type": "string", "maxLength": 4000}},
                        required=["text"],
                    ),
                ),
                updated_at=NOW,
            )
        )

        secret = "sk-" + "liveSecret9999999999"

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "failed", "xuna/errorCode": "read_err"},
                        "structuredContent": {"text": f"partial {secret}", "errorCode": "read_err"},
                    },
                )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}},
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit, verifier=verifier)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                if ev.risk == RiskTier.R3.value:
                    with pytest.raises(ValueError, match="approval_signature_invalid"):
                        await orch.decide_approval(
                            approval_id=uuid.UUID(ev.approval_id),
                            owner_id=owner,
                            decision=ApprovalDecision.APPROVE,
                            scope=ApprovalScope.ONCE,
                            signature=None,
                        )
                    msg = approval_proof_message(
                        ev.approval_id, ev.args_sha256, ev.host_id or "", ev.expires_at
                    )
                    sig = phone_dev.sign(msg)
                    await orch.decide_approval(
                        approval_id=uuid.UUID(ev.approval_id),
                        owner_id=owner,
                        decision=ApprovalDecision.APPROVE,
                        scope=ApprovalScope.ONCE,
                        signature=sig,
                        signer_public_key=phone_dev.public_key,
                    )
                else:
                    await orch.decide_approval(
                        approval_id=uuid.UUID(ev.approval_id),
                        owner_id=owner,
                        decision=ApprovalDecision.APPROVE,
                    )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )

        # 1. Failed source read strips unredacted secret text
        fail_msg = await orch.handle_call(
            ToolCall(
                id="cb_src_fail",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst"},
            ),
            ctx,
        )
        assert secret not in fail_msg.content
        assert secret not in json.dumps([e.wire_payload() for e in events])
        fail_out = json.loads(fail_msg.content)
        assert fail_out["status"] == "failed"
        assert fail_out["output"] == {"errorCode": "read_err"}

        # 2. Inline text up to 4000 chars succeeds & manifest R3 triggers R3 phone signature
        events.clear()
        msg_4000 = await orch.handle_call(
            ToolCall(
                id="cb_4000",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"targetHost": "WinDst", "text": "a" * MAX_CLIPBOARD_WRITE_CHARS},
            ),
            ctx,
        )
        assert json.loads(msg_4000.content)["status"] == "ok"
        p_ev = next(e for e in events if isinstance(e, ActionPendingEvent))
        assert p_ev.risk == RiskTier.R3.value
        assert p_ev.mission_envelope is not None
        assert p_ev.mission_envelope["highestRisk"] == "r3"

        # 3. Inline text 4001 chars fails closed with invalid_args
        msg_4001 = await orch.handle_call(
            ToolCall(
                id="cb_4001",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"targetHost": "WinDst", "text": "a" * (MAX_CLIPBOARD_WRITE_CHARS + 1)},
            ),
            ctx,
        )
        assert json.loads(msg_4001.content)["status"] == "invalid_args"

    asyncio.run(_run())


def test_clipboard_transfer_mode_clipboard_authorizes_r2_read() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id))
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_tool("clipboard.read", ToolFamily.APP, RiskTier.R2, observes=True),),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "clipboard.write",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"text": {"type": "string"}},
                        required=["text"],
                    ),
                ),
                updated_at=NOW,
            )
        )

        mac_calls: list[dict[str, Any]] = []

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                mac_calls.append(frame["params"])
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"text": "copied text"}},
                )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}},
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit)

        async def _emit(ev: ActionStreamEvent) -> None:
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        msg = await orch.handle_call(
            ToolCall(
                id="cb_mode_clip",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "mode": "clipboard"},
            ),
            ctx,
        )
        assert json.loads(msg.content)["status"] == "ok"
        assert len(mac_calls) == 1
        assert mac_calls[0]["_meta"]["xuna/approval"]["argsSha256"] == args_sha256({})

    asyncio.run(_run())


def test_decide_approval_r4_and_verifier_none_fail_closed() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        phone_dev = SimulatedDevice()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()

        phone = _host(
            owner,
            "OwnerPhone",
            HostPlatform.IOS,
            kind=HostKind.MOBILE,
            public_key=phone_dev.public_key,
        )
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        await repo.save_host(phone)
        await repo.save_host(mac)

        svc_with_verifier = ApprovalAndBoardService(
            host_repo=repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            scope_store=None,
            verifier=P256SignatureVerifier(),
            clock=lambda: NOW,
        )
        svc_no_verifier = ApprovalAndBoardService(
            host_repo=repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            scope_store=None,
            verifier=None,
            clock=lambda: NOW,
        )

        async def _new_req(risk: RiskTier) -> ApprovalRequest:
            req = ApprovalRequest(
                approval_id=uuid.uuid4(),
                owner_id=owner,
                host_id=mac.host_id,
                target_alias="MacSrc",
                tool="system.shutdown",
                args_sha256=args_sha256({}),
                risk=risk,
                summary="High risk action",
                created_at=NOW,
                expires_at=NOW + timedelta(seconds=60),
            )
            await approval_repo.save(req)
            await approval_broker.create(req)
            return req

        # 1. R4 requires phone signature; None fails closed, valid phone sig succeeds
        r4_req = await _new_req(RiskTier.R4)
        with pytest.raises(ValueError, match="approval_signature_invalid"):
            await svc_with_verifier.decide_approval(
                approval_id=r4_req.approval_id,
                owner_id=owner,
                decision=ApprovalDecision.APPROVE,
                signature=None,
            )
        r4_msg = approval_proof_message(
            str(r4_req.approval_id),
            r4_req.args_sha256,
            str(mac.host_id),
            r4_req.expires_at.astimezone(UTC).isoformat(),
        )
        approved_r4 = await svc_with_verifier.decide_approval(
            approval_id=r4_req.approval_id,
            owner_id=owner,
            decision=ApprovalDecision.APPROVE,
            signature=phone_dev.sign(r4_msg),
            signer_public_key=phone_dev.public_key,
        )
        assert approved_r4 is not None and approved_r4.state.value == "approved"

        # 2. verifier=None with enrolled phone + R3/R4 fails closed even with 64-byte signature
        for r_tier in (RiskTier.R3, RiskTier.R4):
            req_nv = await _new_req(r_tier)
            with pytest.raises(ValueError, match="approval_signature_invalid"):
                await svc_no_verifier.decide_approval(
                    approval_id=req_nv.approval_id,
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                    signature=b"\x00" * 64,
                )
            with pytest.raises(ValueError, match="approval_signature_invalid"):
                await svc_no_verifier.decide_approval(
                    approval_id=req_nv.approval_id,
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                    signature=b"\x00" * 64,
                    signer_public_key=phone_dev.public_key,
                )

    asyncio.run(_run())


def test_waiting_wake_job_expires_after_180s_ttl_on_host_ready() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        wake_queue = InMemoryWakeQueueStore()
        pending_store = InMemoryPendingIntentStore()
        win = _host(owner, "WinSleep", HostPlatform.WINDOWS, wol_mac="AA:BB:CC:DD:EE:22")
        await repo.save_host(win)
        await repo.set_grants(win.host_id, _all_grants(win.host_id))
        await repo.save_manifest(Manifest(host_id=win.host_id, tools=(), updated_at=NOW))

        orch = _make_orch(repo=repo, hub=hub, audit=audit, wake_queue=wake_queue)
        job_store = InMemoryAutonomousJobStore()

        class _DummyGen:
            async def stream(self, req: Any) -> Any:
                return
                yield req

        step_exec = OrchestratorJobStepExecutor(
            orchestrator_factory=lambda: orch,
            host_repo_factory=lambda: repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            audit_factory=lambda: audit,
            generator_factory=lambda: _DummyGen(),  # type: ignore[arg-type,return-value]
            planner=JobPlannerExecutor(),
            compactor=SubGoalCompactor(),
            wake_queue=wake_queue,
            clock_utc=lambda: NOW,
        )
        runner = AutonomousJobRunner(
            job_store=job_store, step_executor=step_exec, clock_utc=lambda: NOW
        )
        job = await runner.start_job(
            job_id=uuid.uuid4(),
            owner_id=owner,
            host_id=win.host_id,
            goal="Step 1: run",
        )
        await asyncio.sleep(0.05)
        waiting = await job_store.get(job.job_id)
        assert waiting is not None and waiting.state == AutonomousJobState.WAITING_WAKE

        handled = await handle_host_ready_wake_queue(
            owner_id=owner,
            host_id=win.host_id,
            wake_queue=wake_queue,
            pending_intent_store=pending_store,
            job_runner=runner,
            now=NOW + timedelta(seconds=181),
        )
        assert handled == 0
        expired = await job_store.get(job.job_id)
        assert expired is not None
        assert expired.state == AutonomousJobState.FAILED
        assert expired.stop_reason == "wake_ttl_expired"
        await runner.shutdown()

    asyncio.run(_run())


def test_emit_offline_or_wake_queue_full_and_clipboard_offline_wol_args() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        wake_queue = InMemoryWakeQueueStore()
        mac = _host(owner, "MacOnline", HostPlatform.MACOS)
        win = _host(owner, "WinOffline", HostPlatform.WINDOWS, wol_mac="AA:BB:CC:DD:EE:33")
        await repo.save_host(mac)
        await repo.save_host(win)
        await repo.set_grants(mac.host_id, _all_grants(mac.host_id))
        await repo.set_grants(win.host_id, _all_grants(win.host_id))
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_tool("selection.read", ToolFamily.APP, RiskTier.R1, observes=True),),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "clipboard.write",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"text": {"type": "string", "maxLength": 4000}},
                    ),
                ),
                updated_at=NOW,
            )
        )

        async def _noop_send(_raw: str) -> None:
            return None

        async def _noop_close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _noop_send, _noop_close, _noop_close, owner_id=owner
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit, wake_queue=wake_queue)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )

        # 1. Inline text stores redacted {"text": ...} in DeferredWakeStep (never {})
        secret = "sk-" + "offlineSecret123456"
        msg_inline = await orch.handle_call(
            ToolCall(
                id="cb_off_inline",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"targetHost": "WinOffline", "text": f"key={secret}"},
            ),
            ctx,
        )
        assert json.loads(msg_inline.content)["status"] == "host_offline"
        assert any(isinstance(e, HostWakeRequestEvent) for e in events)
        step1 = wake_queue.list_active(owner, NOW)[0]
        assert step1.tool == "clipboard.write"
        assert step1.arguments == {"text": "key=[REDACTED]"}
        assert secret not in json.dumps(step1.arguments)

        # 2. Cross-host clipboard without inline text stores MESH_CLIPBOARD_TRANSFER_FUNCTION + args
        events.clear()
        msg_cross = await orch.handle_call(
            ToolCall(
                id="cb_off_cross",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={
                    "sourceHost": "MacOnline",
                    "targetHost": "WinOffline",
                    "mode": "selection",
                },
            ),
            ctx,
        )
        assert json.loads(msg_cross.content)["status"] == "host_offline"
        step2 = wake_queue.list_active(owner, NOW)[1]
        assert step2.tool == MESH_CLIPBOARD_TRANSFER_FUNCTION
        assert step2.arguments["sourceHost"] == "MacOnline"
        assert step2.arguments["targetHost"] == "WinOffline"

        # 3. Fill 3rd slot, then 4th triggers WakeQueueFullError -> wake_queue_full, 0 wake events
        await orch.handle_call(
            ToolCall(
                id="cb_off_3",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"targetHost": "WinOffline", "text": "slot 3"},
            ),
            ctx,
        )
        assert len(wake_queue.list_active(owner, NOW)) == MAX_DEFERRED_WAKE_STEPS_PER_OWNER
        events.clear()
        msg_full = await orch.handle_call(
            ToolCall(
                id="cb_off_4",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"targetHost": "WinOffline", "text": "slot 4"},
            ),
            ctx,
        )
        out_full = json.loads(msg_full.content)
        assert out_full["status"] == "host_offline"
        assert out_full["reasonCode"] == "host_offline"
        assert out_full["errorCode"] == "wake_queue_full"
        assert "wake_queue_full" in out_full["targetReason"]
        assert not any(isinstance(e, HostWakeRequestEvent) for e in events)
        tr_ev = next(e for e in events if isinstance(e, ActionTargetRequiredEvent))
        assert tr_ev.reason_code == "host_offline"
        assert "wake_queue_full" in tr_ev.target_reason

    asyncio.run(_run())


def test_battery_advisor_omits_incapable_online_host_and_cu_rejects_blank_goal() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        low_mac = _host(owner, "LowMac", HostPlatform.MACOS)
        no_cu_win = _host(owner, "NoCuWin", HostPlatform.WINDOWS)
        unsup_lin = _host(owner, "UnsupLin", HostPlatform.LINUX, kind=HostKind.SERVER)
        for h in (low_mac, no_cu_win, unsup_lin):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id, RiskTier.R2))

        cu_tool = _tool(
            "cu.session_start",
            ToolFamily.CU,
            RiskTier.R2,
            observes=True,
            props={"goal": {"type": "string"}},
            required=["goal"],
        )
        cu_unsup = ToolDescriptor(
            name="cu.session_start",
            description="Unsupported CU",
            input_schema={
                "type": "object",
                "properties": {"goal": {"type": "string"}},
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.CU,
            risk=RiskTier.R2,
            permission=ToolPermissionState.UNSUPPORTED,
        )
        await repo.save_manifest(
            Manifest(host_id=low_mac.host_id, tools=(cu_tool,), updated_at=NOW)
        )
        await repo.save_manifest(
            Manifest(
                host_id=no_cu_win.host_id,
                tools=(_tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1),),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(host_id=unsup_lin.host_id, tools=(cu_unsup,), updated_at=NOW)
        )

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(low_mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"sessionId": "s1"}},
                )

        async def _noop_send(_raw: str) -> None:
            return None

        async def _close() -> None:
            return None

        await hub.open_session(
            str(low_mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(no_cu_win.host_id), object(), _noop_send, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(unsup_lin.host_id), object(), _noop_send, _close, _close, owner_id=owner
        )
        hub.update_presence(
            str(low_mac.host_id),
            HostPresence(
                idle_seconds=4,
                screen_locked=False,
                media_playing=False,
                updated_at=NOW,
                battery_percent=12,
                is_charging=False,
            ),
        )
        hub.update_presence(
            str(no_cu_win.host_id),
            HostPresence(
                idle_seconds=4,
                screen_locked=False,
                media_playing=False,
                updated_at=NOW,
                battery_percent=None,
                is_charging=None,
            ),
        )

        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
            pinned_host_id=low_mac.host_id,
        )

        # 1. Whitespace-only goal fails closed with invalid_args / missing_goal and 0 pending events
        blank_msg = await orch.handle_call(
            ToolCall(id="cu_blank", name="cu__session_start", arguments={"goal": "   "}), ctx
        )
        blank_out = json.loads(blank_msg.content)
        assert blank_out["status"] == "invalid_args"
        assert blank_out["output"]["errorCode"] == "missing_goal"
        assert events == []

        # 2. Valid goal emits batteryAdvisories excluding NoCuWin and UnsupLin
        ok_msg = await orch.handle_call(
            ToolCall(id="cu_valid", name="cu__session_start", arguments={"goal": "Run task"}), ctx
        )
        assert json.loads(ok_msg.content)["status"] == "ok"
        p_ev = next(e for e in events if isinstance(e, ActionPendingEvent))
        assert p_ev.battery_advisories
        assert "LowMac" in p_ev.battery_advisories[0]
        assert "NoCuWin" not in p_ev.battery_advisories[0]
        assert "UnsupLin" not in p_ev.battery_advisories[0]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Iteration 3 Rule H7 Fail-Closed & Negative Regression Tests (Defects 1-8)
# ---------------------------------------------------------------------------


def test_fanout_valid_host_ids_filters_smart_pause_vitals_and_git_status() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacA", HostPlatform.MACOS)
        win = _host(owner, "WinB", HostPlatform.WINDOWS)
        for h in (mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id))
            await repo.save_manifest(
                Manifest(
                    host_id=h.host_id,
                    tools=(
                        _tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1, idempotent=False),
                        _tool("system.vitals", ToolFamily.HOST, RiskTier.R1, observes=True),
                        _tool(
                            "workspace.git_summary",
                            ToolFamily.APP,
                            RiskTier.R1,
                            observes=True,
                            props={"project": {"type": "string"}},
                        ),
                    ),
                    updated_at=NOW,
                )
            )

        mac_calls: list[str] = []
        win_calls: list[str] = []

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                mac_calls.append(frame["params"]["name"])
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}},
                )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                tname = frame["params"]["name"]
                win_calls.append(tname)
                sc: dict[str, object] = (
                    {"cpuPercent": 15}
                    if tname == "system.vitals"
                    else (
                        {"branch": "main", "dirtyCount": 0}
                        if tname == "workspace.git_summary"
                        else {"paused": True}
                    )
                )
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": sc},
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )
        for h in (mac, win):
            hub.update_presence(
                str(h.host_id),
                HostPresence(
                    idle_seconds=1,
                    screen_locked=False,
                    media_playing=True,
                    updated_at=NOW,
                ),
            )

        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        target_filter = {"hostIds": [str(win.host_id)]}

        pause_msg = await orch.handle_call(
            ToolCall(id="f_pause", name=MESH_SMART_PAUSE_MEDIA_FUNCTION, arguments=target_filter),
            ctx,
        )
        pause_out = json.loads(pause_msg.content)
        assert pause_out["status"] == "ok"
        assert [p["hostId"] for p in pause_out["pausedHosts"]] == [str(win.host_id)]
        assert mac_calls == []

        vitals_msg = await orch.handle_call(
            ToolCall(id="f_vitals", name=MESH_FLEET_VITALS_FUNCTION, arguments=target_filter),
            ctx,
        )
        vitals_out = json.loads(vitals_msg.content)
        assert vitals_out["status"] == "ok"
        assert [r["hostId"] for r in vitals_out["hosts"]] == [str(win.host_id)]
        assert vitals_out["hosts"][0]["vitalsStatus"] == "ok"
        assert mac_calls == []

        git_msg = await orch.handle_call(
            ToolCall(id="f_git", name=MESH_FLEET_GIT_STATUS_FUNCTION, arguments=target_filter),
            ctx,
        )
        git_out = json.loads(git_msg.content)
        assert git_out["status"] == "ok"
        assert [r["hostId"] for r in git_out["results"]] == [str(win.host_id)]
        assert git_out["results"][0]["status"] == "ok"
        assert mac_calls == []
        assert win_calls == ["media.play_pause", "system.vitals", "workspace.git_summary"]

    asyncio.run(_run())


def test_teleport_url_without_open_in_app_fails_closed_when_target_lacks_url_open() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (mac, win):
            await repo.save_host(h)
        await repo.set_grants(mac.host_id, _all_grants(mac.host_id))
        # Target grants APP.R2 (app.launch), NOT URL (no url.open grant)
        await repo.set_grants(
            win.host_id,
            [Grant(win.host_id, ToolFamily.APP, RiskTier.R2, True, NOW)],
        )
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_tool("browser.current_url", ToolFamily.URL, RiskTier.R2, observes=True),),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "app.launch",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"id": {"type": "string"}, "openTarget": {"type": "string"}},
                    ),
                ),
                updated_at=NOW,
            )
        )

        target_calls: list[str] = []

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "ok"},
                        "structuredContent": {"url": "https://example.org/doc"},
                    },
                )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                target_calls.append(frame["params"]["name"])
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}},
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )

        # kind="url", openInAppId=None -> must fail closed as denied with 0 prompts/calls
        denied_msg = await orch.handle_call(
            ToolCall(
                id="tp_url_denied",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "url"},
            ),
            ctx,
        )
        denied_out = json.loads(denied_msg.content)
        assert denied_out["status"] == "denied"
        assert denied_out["output"]["errorCode"] == "grant_denied"
        assert events == []
        assert target_calls == []

        # Explicit openInAppId with app.launch grant succeeds
        ok_msg = await orch.handle_call(
            ToolCall(
                id="tp_url_app",
                name=MESH_TELEPORT_FUNCTION,
                arguments={
                    "sourceHost": "MacSrc",
                    "targetHost": "WinDst",
                    "kind": "url",
                    "openInAppId": "browser_x",
                },
            ),
            ctx,
        )
        assert json.loads(ok_msg.content)["status"] == "ok"
        assert target_calls == ["app.launch"]

    asyncio.run(_run())


def test_fleet_vitals_and_git_status_preserve_meta_xuna_status_timeout() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacA", HostPlatform.MACOS)
        await repo.save_host(mac)
        await repo.set_grants(mac.host_id, _all_grants(mac.host_id))
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("system.vitals", ToolFamily.HOST, RiskTier.R1, observes=True),
                    _tool(
                        "workspace.git_summary",
                        ToolFamily.APP,
                        RiskTier.R1,
                        observes=True,
                        props={"project": {"type": "string"}},
                    ),
                ),
                updated_at=NOW,
            )
        )

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "timeout", "xuna/errorCode": "host_timed_out"},
                        "structuredContent": {},
                    },
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )

        vitals_msg = await orch.handle_call(
            ToolCall(id="v_to", name=MESH_FLEET_VITALS_FUNCTION, arguments={}),
            ctx,
        )
        vitals_out = json.loads(vitals_msg.content)
        assert vitals_out["status"] == "timeout"
        assert vitals_out["errorCode"] == "host_timed_out"
        assert vitals_out["hosts"][0]["vitalsStatus"] == "timeout"
        assert isinstance(events[-1], ActionResultEvent)
        assert events[-1].status == "timeout"

        events.clear()
        git_msg = await orch.handle_call(
            ToolCall(id="g_to", name=MESH_FLEET_GIT_STATUS_FUNCTION, arguments={}),
            ctx,
        )
        git_out = json.loads(git_msg.content)
        assert git_out["status"] == "timeout"
        assert git_out["errorCode"] == "host_timed_out"
        assert git_out["results"][0]["status"] == "timeout"
        assert isinstance(events[-1], ActionResultEvent)
        assert events[-1].status == "timeout"

    asyncio.run(_run())


def test_teleport_workspace_capture_rejection_and_expiry_emit_action_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xuna_backend.modules.actions.application import cross_host_pipelines as chp

    async def _run() -> None:
        owner = uuid.uuid4()
        phone_dev = SimulatedDevice()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        verifier = P256SignatureVerifier()

        phone = _host(
            owner,
            "OwnerPhone",
            HostPlatform.IOS,
            kind=HostKind.MOBILE,
            public_key=phone_dev.public_key,
        )
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (phone, mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id, RiskTier.R3))
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "workspace.open",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"project": {"type": "string"}},
                    ),
                ),
                updated_at=NOW,
            )
        )

        mac_calls: list[str] = []

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                mac_calls.append(frame["params"]["name"])

        async def _noop(_raw: str) -> None:
            return None

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(str(win.host_id), object(), _noop, _close, _close, owner_id=owner)
        orch = _make_orch(repo=repo, hub=hub, audit=audit, verifier=verifier)

        # Case 1: R2 workspace.capture_capsule rejection emits ActionResultEvent(rejected)
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("workspace.capture_capsule", ToolFamily.APP, RiskTier.R2, observes=True),
                ),
                updated_at=NOW,
            )
        )
        events: list[ActionStreamEvent] = []

        async def _emit_reject(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.REJECT,
                )

        ctx_rej = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit_reject,
        )
        rej_msg = await orch.handle_call(
            ToolCall(
                id="tp_ws_rej",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "workspace"},
            ),
            ctx_rej,
        )
        assert json.loads(rej_msg.content)["status"] == "rejected"
        assert sum(1 for e in events if isinstance(e, ActionPendingEvent)) == 1
        res_rej = [e for e in events if isinstance(e, ActionResultEvent)]
        assert len(res_rej) == 1
        assert res_rej[0].status == "rejected"
        assert res_rej[0].error_code == "rejected"
        assert res_rej[0].target_platform == "macos"
        assert res_rej[0].mission_envelope is not None
        assert res_rej[0].mission_envelope["steps"][0]["status"] == "denied"
        assert mac_calls == []

        # Case 2: R3 workspace.capture_capsule expiry emits ActionResultEvent(expired)
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("workspace.capture_capsule", ToolFamily.APP, RiskTier.R3, observes=True),
                ),
                updated_at=NOW,
            )
        )
        monkeypatch.setattr(chp, "APPROVAL_TTL_SECONDS", 0.02)
        events.clear()

        async def _emit_expire(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx_exp = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit_expire,
        )
        exp_msg = await orch.handle_call(
            ToolCall(
                id="tp_ws_exp",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "workspace"},
            ),
            ctx_exp,
        )
        assert json.loads(exp_msg.content)["status"] == "expired"
        assert sum(1 for e in events if isinstance(e, ActionPendingEvent)) == 1
        res_exp = [e for e in events if isinstance(e, ActionResultEvent)]
        assert len(res_exp) == 1
        assert res_exp[0].status == "expired"
        assert res_exp[0].error_code == "expired"
        assert res_exp[0].mission_envelope is not None
        assert res_exp[0].mission_envelope["steps"][0]["status"] == "denied"
        assert mac_calls == []

    asyncio.run(_run())


def test_teleport_failed_source_read_strips_unredacted_secrets() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id))
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("browser.current_url", ToolFamily.URL, RiskTier.R1, observes=True),
                    _tool("workspace.capture_capsule", ToolFamily.APP, RiskTier.R1, observes=True),
                ),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "url.open",
                        ToolFamily.URL,
                        RiskTier.R2,
                        props={"url": {"type": "string"}},
                        required=["url"],
                    ),
                    _tool(
                        "workspace.open",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"project": {"type": "string"}},
                    ),
                ),
                updated_at=NOW,
            )
        )

        url_secret = "sk-" + "urlSecretLeak123456"
        ws_secret = "ghp_" + "wsSecretLeak123456"

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") != "tools/call":
                return
            tname = frame["params"]["name"]
            if tname == "browser.current_url":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "failed", "xuna/errorCode": "url_read_err"},
                        "structuredContent": {
                            "url": f"https://example.org/?token={url_secret}",
                            "partial": f'password="{url_secret}"',
                        },
                    },
                )
            else:
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "failed", "xuna/errorCode": "ws_read_err"},
                        "structuredContent": {
                            "nextStepHint": f"token={ws_secret}",
                            "briefingVi": f"secret {ws_secret}",
                        },
                    },
                )

        async def _noop(_raw: str) -> None:
            return None

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(str(win.host_id), object(), _noop, _close, _close, owner_id=owner)
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )

        url_msg = await orch.handle_call(
            ToolCall(
                id="tp_fail_url",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "url"},
            ),
            ctx,
        )
        assert url_secret not in url_msg.content
        assert url_secret not in json.dumps([e.wire_payload() for e in events])
        assert json.loads(url_msg.content) == {
            "status": "failed",
            "output": {"errorCode": "url_read_err"},
        }

        events.clear()
        ws_msg = await orch.handle_call(
            ToolCall(
                id="tp_fail_ws",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "workspace"},
            ),
            ctx,
        )
        assert ws_secret not in ws_msg.content
        assert ws_secret not in json.dumps([e.wire_payload() for e in events])
        assert json.loads(ws_msg.content) == {
            "status": "failed",
            "output": {"errorCode": "ws_read_err"},
        }

    asyncio.run(_run())


def test_decide_approval_r4_without_phone_and_bogus_sig_always_fail_closed() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        other_dev = SimulatedDevice()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()

        # Desktop host ONLY (phone_keys == [])
        mac = _host(owner, "MacOnly", HostPlatform.MACOS)
        await repo.save_host(mac)

        svc = ApprovalAndBoardService(
            host_repo=repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            scope_store=None,
            verifier=P256SignatureVerifier(),
            clock=lambda: NOW,
        )

        async def _new_req(risk: RiskTier) -> ApprovalRequest:
            req = ApprovalRequest(
                approval_id=uuid.uuid4(),
                owner_id=owner,
                host_id=mac.host_id,
                target_alias="MacOnly",
                tool="system.shutdown",
                args_sha256=args_sha256({}),
                risk=risk,
                summary="High risk action",
                created_at=NOW,
                expires_at=NOW + timedelta(seconds=60),
            )
            await approval_repo.save(req)
            await approval_broker.create(req)
            return req

        # 1. R4 without phone enrolled fails closed on None, bogus 64B sig, and unenrolled key
        for bad_sig, bad_pk in (
            (None, None),
            (b"\x00" * 64, None),
        ):
            r4_req = await _new_req(RiskTier.R4)
            with pytest.raises(ValueError, match="approval_signature_invalid"):
                await svc.decide_approval(
                    approval_id=r4_req.approval_id,
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                    signature=bad_sig,
                    signer_public_key=bad_pk,
                )

        r4_unenrolled = await _new_req(RiskTier.R4)
        unenrolled_msg = approval_proof_message(
            str(r4_unenrolled.approval_id),
            r4_unenrolled.args_sha256,
            str(mac.host_id),
            r4_unenrolled.expires_at.astimezone(UTC).isoformat(),
        )
        with pytest.raises(ValueError, match="approval_signature_invalid"):
            await svc.decide_approval(
                approval_id=r4_unenrolled.approval_id,
                owner_id=owner,
                decision=ApprovalDecision.APPROVE,
                signature=other_dev.sign(unenrolled_msg),
                signer_public_key=other_dev.public_key,
            )

        # 2. R3 with bogus 64B signature and signer_public_key=None fails closed even if no phone
        r3_bogus = await _new_req(RiskTier.R3)
        with pytest.raises(ValueError, match="approval_signature_invalid"):
            await svc.decide_approval(
                approval_id=r3_bogus.approval_id,
                owner_id=owner,
                decision=ApprovalDecision.APPROVE,
                signature=b"\x00" * 64,
                signer_public_key=None,
            )

    asyncio.run(_run())


def test_battery_advisor_excludes_online_host_lacking_cu_grant() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        low_mac = _host(owner, "LowMac", HostPlatform.MACOS)
        denied_win = _host(owner, "DeniedWin", HostPlatform.WINDOWS)
        await repo.save_host(low_mac)
        await repo.save_host(denied_win)
        await repo.set_grants(low_mac.host_id, _all_grants(low_mac.host_id, RiskTier.R2))
        # DeniedWin has cu.session_start in manifest, but NO ToolFamily.CU grant!
        await repo.set_grants(
            denied_win.host_id,
            [Grant(denied_win.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW)],
        )
        cu_tool = _tool(
            "cu.session_start",
            ToolFamily.CU,
            RiskTier.R2,
            observes=True,
            props={"goal": {"type": "string"}},
            required=["goal"],
        )
        await repo.save_manifest(
            Manifest(host_id=low_mac.host_id, tools=(cu_tool,), updated_at=NOW)
        )
        await repo.save_manifest(
            Manifest(host_id=denied_win.host_id, tools=(cu_tool,), updated_at=NOW)
        )

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(low_mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"sessionId": "s1"}},
                )

        async def _noop(_raw: str) -> None:
            return None

        async def _close() -> None:
            return None

        await hub.open_session(
            str(low_mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(denied_win.host_id), object(), _noop, _close, _close, owner_id=owner
        )
        hub.update_presence(
            str(low_mac.host_id),
            HostPresence(
                idle_seconds=2,
                screen_locked=False,
                media_playing=False,
                updated_at=NOW,
                battery_percent=12,
                is_charging=False,
            ),
        )
        hub.update_presence(
            str(denied_win.host_id),
            HostPresence(
                idle_seconds=2,
                screen_locked=False,
                media_playing=False,
                updated_at=NOW,
                battery_percent=95,
                is_charging=True,
            ),
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
            pinned_host_id=low_mac.host_id,
        )
        msg = await orch.handle_call(
            ToolCall(id="cu_nogrant", name="cu__session_start", arguments={"goal": "Check"}), ctx
        )
        assert json.loads(msg.content)["status"] == "ok"
        p_ev = next(e for e in events if isinstance(e, ActionPendingEvent))
        for adv in p_ev.battery_advisories:
            assert "DeniedWin" not in adv

    asyncio.run(_run())


def test_teleport_mixed_r2_r3_mission_enforces_r3_on_upfront_prompt() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        phone_dev = SimulatedDevice()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        verifier = P256SignatureVerifier()

        phone = _host(
            owner,
            "OwnerPhone",
            HostPlatform.IOS,
            kind=HostKind.MOBILE,
            public_key=phone_dev.public_key,
        )
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (phone, mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id, RiskTier.R3))
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_tool("browser.current_url", ToolFamily.URL, RiskTier.R2, observes=True),),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "url.open",
                        ToolFamily.URL,
                        RiskTier.R3,
                        props={"url": {"type": "string"}},
                        required=["url"],
                    ),
                ),
                updated_at=NOW,
            )
        )

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {
                        "_meta": {"xuna/status": "ok"},
                        "structuredContent": {"url": "https://example.org/mixed"},
                    },
                )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"opened": True}},
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )
        orch = _make_orch(repo=repo, hub=hub, audit=audit, verifier=verifier)
        pending_risks: list[tuple[str, str]] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            if isinstance(ev, ActionPendingEvent):
                assert ev.mission_envelope is not None
                pending_risks.append((ev.risk, str(ev.mission_envelope["highestRisk"])))
                with pytest.raises(ValueError, match="r3_requires_once_scope"):
                    await orch.decide_approval(
                        approval_id=uuid.UUID(ev.approval_id),
                        owner_id=owner,
                        decision=ApprovalDecision.APPROVE,
                        scope=ApprovalScope.CONVERSATION_15M,
                    )
                with pytest.raises(ValueError, match="approval_signature_invalid"):
                    await orch.decide_approval(
                        approval_id=uuid.UUID(ev.approval_id),
                        owner_id=owner,
                        decision=ApprovalDecision.APPROVE,
                        scope=ApprovalScope.ONCE,
                        signature=None,
                    )
                msg = approval_proof_message(
                    ev.approval_id, ev.args_sha256, ev.host_id or "", ev.expires_at
                )
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                    scope=ApprovalScope.ONCE,
                    signature=phone_dev.sign(msg),
                    signer_public_key=phone_dev.public_key,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )
        res = await orch.handle_call(
            ToolCall(
                id="tp_mixed_r3",
                name=MESH_TELEPORT_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "kind": "url"},
            ),
            ctx,
        )
        assert json.loads(res.content)["status"] == "ok"
        assert pending_risks == [("R3", "r3"), ("R3", "r3")]

    asyncio.run(_run())


def test_waiting_wake_db_job_resumes_with_empty_ram_queue_before_180s_and_expires_at_180s() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        wake_queue = InMemoryWakeQueueStore()
        pending_store = InMemoryPendingIntentStore()
        win = _host(owner, "WinSleep", HostPlatform.WINDOWS, wol_mac="AA:BB:CC:DD:EE:55")
        await repo.save_host(win)
        await repo.set_grants(win.host_id, _all_grants(win.host_id))
        await repo.save_manifest(Manifest(host_id=win.host_id, tools=(), updated_at=NOW))

        orch = _make_orch(repo=repo, hub=hub, audit=audit, wake_queue=wake_queue)
        job_store = InMemoryAutonomousJobStore()

        class _DummyGen:
            async def stream(self, req: Any) -> Any:
                return
                yield req

        step_exec = OrchestratorJobStepExecutor(
            orchestrator_factory=lambda: orch,
            host_repo_factory=lambda: repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            audit_factory=lambda: audit,
            generator_factory=lambda: _DummyGen(),  # type: ignore[arg-type,return-value]
            planner=JobPlannerExecutor(),
            compactor=SubGoalCompactor(),
            wake_queue=None,
            clock_utc=lambda: NOW,
        )
        runner = AutonomousJobRunner(
            job_store=job_store, step_executor=step_exec, clock_utc=lambda: NOW
        )

        # 1. 10s-old WAITING_WAKE DB job resumes even when RAM wake_queue is empty
        job_ok = await runner.start_job(
            job_id=uuid.uuid4(),
            owner_id=owner,
            host_id=win.host_id,
            goal="Step 1: resume after restart",
        )
        await asyncio.sleep(0.05)
        w_ok = await job_store.get(job_ok.job_id)
        assert w_ok is not None and w_ok.state == AutonomousJobState.WAITING_WAKE

        handled_10s = await handle_host_ready_wake_queue(
            owner_id=owner,
            host_id=win.host_id,
            wake_queue=InMemoryWakeQueueStore(),
            pending_intent_store=pending_store,
            job_runner=runner,
            now=NOW + timedelta(seconds=10),
        )
        assert handled_10s == 1
        resumed = await job_store.get(job_ok.job_id)
        assert resumed is not None and resumed.state != AutonomousJobState.FAILED

        # 2. Exact 180s TTL expires with wake_ttl_expired
        job_exp = await runner.start_job(
            job_id=uuid.uuid4(),
            owner_id=owner,
            host_id=win.host_id,
            goal="Step 1: expire at 180s",
        )
        await asyncio.sleep(0.05)
        handled_180s = await handle_host_ready_wake_queue(
            owner_id=owner,
            host_id=win.host_id,
            wake_queue=InMemoryWakeQueueStore(),
            pending_intent_store=pending_store,
            job_runner=runner,
            now=NOW + timedelta(seconds=180),
        )
        assert handled_180s == 0
        expired = await job_store.get(job_exp.job_id)
        assert expired is not None
        assert expired.state == AutonomousJobState.FAILED
        assert expired.stop_reason == "wake_ttl_expired"
        await runner.shutdown()

    asyncio.run(_run())


def test_clipboard_mixed_r2_r3_mission_enforces_r3_and_keeps_r1_unprompted() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        phone_dev = SimulatedDevice()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        verifier = P256SignatureVerifier()
        scope_store = InMemoryApprovalScopeStore()

        phone = _host(
            owner,
            "OwnerPhone",
            HostPlatform.IOS,
            kind=HostKind.MOBILE,
            public_key=phone_dev.public_key,
        )
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (phone, mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id, RiskTier.R3))

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"text": "hello"}},
                )

        async def _send_win(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"written": True}},
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _send_win, _close, _close, owner_id=owner
        )
        orch = _make_orch(
            repo=repo, hub=hub, audit=audit, scope_store=scope_store, verifier=verifier
        )
        pending_risks: list[tuple[str, str]] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            if isinstance(ev, ActionPendingEvent):
                assert ev.mission_envelope is not None
                pending_risks.append((ev.risk, str(ev.mission_envelope["highestRisk"])))
                with pytest.raises(ValueError, match="r3_requires_once_scope"):
                    await orch.decide_approval(
                        approval_id=uuid.UUID(ev.approval_id),
                        owner_id=owner,
                        decision=ApprovalDecision.APPROVE,
                        scope=ApprovalScope.CONVERSATION_15M,
                    )
                with pytest.raises(ValueError, match="approval_signature_invalid"):
                    await orch.decide_approval(
                        approval_id=uuid.UUID(ev.approval_id),
                        owner_id=owner,
                        decision=ApprovalDecision.APPROVE,
                        scope=ApprovalScope.ONCE,
                        signature=None,
                    )
                msg = approval_proof_message(
                    ev.approval_id, ev.args_sha256, ev.host_id or "", ev.expires_at
                )
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                    scope=ApprovalScope.ONCE,
                    signature=phone_dev.sign(msg),
                    signer_public_key=phone_dev.public_key,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )

        # Case 1: clipboard.read (R2) + clipboard.write (R3), with pre-existing 15m scope
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("selection.read", ToolFamily.APP, RiskTier.R1, observes=True),
                    _tool("clipboard.read", ToolFamily.APP, RiskTier.R2, observes=True),
                ),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "clipboard.write",
                        ToolFamily.APP,
                        RiskTier.R3,
                        props={"text": {"type": "string"}},
                        required=["text"],
                    ),
                ),
                updated_at=NOW,
            )
        )
        await scope_store.grant_scope(owner, "pet-1", mac.host_id, "clipboard.read", NOW)
        assert await scope_store.has_valid_scope(owner, "pet-1", mac.host_id, "clipboard.read", NOW)
        res_r2_r3 = await orch.handle_call(
            ToolCall(
                id="clip_r2_r3",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "mode": "clipboard"},
            ),
            ctx,
        )
        assert json.loads(res_r2_r3.content)["status"] == "ok"
        assert pending_risks == [("R3", "r3"), ("R3", "r3")]

        # Case 2: selection.read (R1) + clipboard.write (R3) keeps R1 unprompted
        pending_risks.clear()
        res_r1_r3 = await orch.handle_call(
            ToolCall(
                id="clip_r1_r3",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "mode": "selection"},
            ),
            ctx,
        )
        assert json.loads(res_r1_r3.content)["status"] == "ok"
        assert pending_risks == [("R3", "r3")]

        # Case 3: Inverse clipboard.read (R3) + clipboard.write (R2), with pre-existing 15m scope
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_tool("clipboard.read", ToolFamily.APP, RiskTier.R3, observes=True),),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "clipboard.write",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"text": {"type": "string"}},
                        required=["text"],
                    ),
                ),
                updated_at=NOW,
            )
        )
        await scope_store.grant_scope(owner, "pet-1", win.host_id, "clipboard.write", NOW)
        assert await scope_store.has_valid_scope(
            owner, "pet-1", win.host_id, "clipboard.write", NOW
        )
        pending_risks.clear()
        res_r3_r2 = await orch.handle_call(
            ToolCall(
                id="clip_r3_r2",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "mode": "clipboard"},
            ),
            ctx,
        )
        assert json.loads(res_r3_r2.content)["status"] == "ok"
        assert pending_risks == [("R3", "r3"), ("R3", "r3")]

    asyncio.run(_run())


def test_clipboard_failed_source_read_redacts_error_code() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        audit = MemoryAuditLog()
        mac = _host(owner, "MacSrc", HostPlatform.MACOS)
        win = _host(owner, "WinDst", HostPlatform.WINDOWS)
        for h in (mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id))
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("selection.read", ToolFamily.APP, RiskTier.R1, observes=True),
                    _tool("clipboard.read", ToolFamily.APP, RiskTier.R2, observes=True),
                ),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "clipboard.write",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"text": {"type": "string"}},
                        required=["text"],
                    ),
                ),
                updated_at=NOW,
            )
        )

        clip_secret = "sk-" + "clipErrSecret123456"
        sel_secret = "sk-" + "selErrSecret123456"

        async def _send_mac(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") != "tools/call":
                return
            tname = frame["params"]["name"]
            secret = clip_secret if tname == "clipboard.read" else sel_secret
            hub.on_response(
                str(mac.host_id),
                frame["id"],
                {
                    "_meta": {"xuna/status": "failed", "xuna/errorCode": f"err_{secret}"},
                    "structuredContent": {},
                },
            )

        async def _noop(_raw: str) -> None:
            return None

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_mac, _close, _close, owner_id=owner
        )
        await hub.open_session(str(win.host_id), object(), _noop, _close, _close, owner_id=owner)
        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )

        clip_msg = await orch.handle_call(
            ToolCall(
                id="clip_err_redact",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "mode": "clipboard"},
            ),
            ctx,
        )
        assert clip_secret not in clip_msg.content
        assert json.loads(clip_msg.content) == {
            "status": "failed",
            "output": {"errorCode": "err_[REDACTED]"},
        }
        clip_res = [e for e in events if isinstance(e, ActionResultEvent)]
        assert len(clip_res) == 1
        assert clip_res[0].error_code == "err_[REDACTED]"
        assert clip_secret not in json.dumps([e.wire_payload() for e in events])

        events.clear()
        sel_msg = await orch.handle_call(
            ToolCall(
                id="sel_err_redact",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={"sourceHost": "MacSrc", "targetHost": "WinDst", "mode": "selection"},
            ),
            ctx,
        )
        assert sel_secret not in sel_msg.content
        assert json.loads(sel_msg.content) == {
            "status": "failed",
            "output": {"errorCode": "err_[REDACTED]"},
        }
        sel_res = [e for e in events if isinstance(e, ActionResultEvent)]
        assert len(sel_res) == 1
        assert sel_res[0].error_code == "err_[REDACTED]"
        assert sel_secret not in json.dumps([e.wire_payload() for e in events])

    asyncio.run(_run())


def test_tier5_smart_pause_offline_fleet_git_offline_and_post_approval_bad_selection() -> None:
    """Tier 5 (Gaps 2, 4, 7): fail closed on offline fan-out & post-approval invalid text."""

    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        repo = _MemHostRepo()
        audit = MemoryAuditLog()

        mac = _host(owner, "MacOffline", HostPlatform.MACOS)
        win = _host(owner, "WinTarget", HostPlatform.WINDOWS)
        for h in (mac, win):
            await repo.save_host(h)
            await repo.set_grants(h.host_id, _all_grants(h.host_id, RiskTier.R3))

        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1),
                    _tool("workspace.git_summary", ToolFamily.APP, RiskTier.R1, observes=True),
                    _tool(
                        "selection.read",
                        ToolFamily.APP,
                        RiskTier.R2,
                        observes=True,
                        props={"maxChars": {"type": "integer"}},
                    ),
                ),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool(
                        "clipboard.write",
                        ToolFamily.APP,
                        RiskTier.R2,
                        props={"text": {"type": "string"}},
                        required=["text"],
                    ),
                ),
                updated_at=NOW,
            )
        )

        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )

        # Gap 2: smart_pause_media with explicit offline hostId -> failed host_offline (never ok)
        pause_off_msg = await orch.handle_call(
            ToolCall(
                id="sp_off",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(mac.host_id)]},
            ),
            ctx,
        )
        pause_off = json.loads(pause_off_msg.content)
        assert pause_off["status"] == "failed"
        assert pause_off["errorCode"] == "host_offline"
        assert pause_off["skippedHosts"] == [
            {"hostId": str(mac.host_id), "alias": "MacOffline", "status": "offline"}
        ]
        sp_res = [e for e in events if isinstance(e, ActionResultEvent)]
        assert len(sp_res) == 1
        assert sp_res[0].status == "failed"
        assert sp_res[0].error_code == "host_offline"
        assert sp_res[0].mission_envelope is not None

        # Unknown hostId -> failed no_host
        events.clear()
        pause_none_msg = await orch.handle_call(
            ToolCall(
                id="sp_none",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(uuid.uuid4())]},
            ),
            ctx,
        )
        pause_none = json.loads(pause_none_msg.content)
        assert pause_none["status"] == "failed"
        assert pause_none["errorCode"] == "no_host"

        # Gap 7: fleet_git_status when matching hosts are all offline -> failed host_offline
        events.clear()
        git_off_msg = await orch.handle_call(
            ToolCall(
                id="git_off",
                name=MESH_FLEET_GIT_STATUS_FUNCTION,
                arguments={"hostIds": [str(mac.host_id)]},
            ),
            ctx,
        )
        git_off = json.loads(git_off_msg.content)
        assert git_off["status"] == "failed"
        assert git_off["errorCode"] == "host_offline"
        assert git_off["skipped"] == [
            {"hostId": str(mac.host_id), "alias": "MacOffline", "status": "offline"}
        ]
        git_res = [e for e in events if isinstance(e, ActionResultEvent)]
        assert len(git_res) == 1
        assert git_res[0].status == "failed"
        assert git_res[0].error_code == "host_offline"

        # Gap 4: selection.read returns whitespace after R2 approval -> ActionResultEvent emitted
        async def _send_bad_sel(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"text": "   "}},
                )

        async def _noop(_raw: str) -> None:
            return None

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _send_bad_sel, _close, _close, owner_id=owner
        )
        await hub.open_session(str(win.host_id), object(), _noop, _close, _close, owner_id=owner)

        events.clear()
        bad_sel_msg = await orch.handle_call(
            ToolCall(
                id="clip_bad_sel",
                name=MESH_CLIPBOARD_TRANSFER_FUNCTION,
                arguments={
                    "sourceHost": "MacOffline",
                    "targetHost": "WinTarget",
                    "mode": "selection",
                },
            ),
            ctx,
        )
        assert json.loads(bad_sel_msg.content) == {
            "status": "invalid_args",
            "output": {"errorCode": "invalid_args"},
        }
        pending_evs = [e for e in events if isinstance(e, ActionPendingEvent)]
        result_evs = [e for e in events if isinstance(e, ActionResultEvent)]
        assert len(pending_evs) == 1
        assert len(result_evs) == 1
        assert result_evs[0].status == "failed"
        assert result_evs[0].error_code == "invalid_args"
        assert result_evs[0].mission_envelope is not None

    asyncio.run(_run())


def test_tier5_smart_pause_mixed_denied_offline_probe_failures_and_idle_preservation() -> None:
    """Tier 5 (Gap 2 closure): mixed denied+offline/timeout fail closed; confirmed idle passes."""

    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        repo = _MemHostRepo()
        audit = MemoryAuditLog()

        off_mac = _host(owner, "AlphaMac", HostPlatform.MACOS)
        den_win = _host(owner, "BetaWin", HostPlatform.WINDOWS)
        probe_lin = _host(owner, "GammaLin", HostPlatform.LINUX)
        idle_mac = _host(owner, "DeltaMac", HostPlatform.MACOS)
        for h in (off_mac, den_win, probe_lin, idle_mac):
            await repo.save_host(h)

        # off_mac, probe_lin, idle_mac have full grants; den_win has NO media grant
        await repo.set_grants(off_mac.host_id, _all_grants(off_mac.host_id, RiskTier.R2))
        await repo.set_grants(den_win.host_id, [])
        await repo.set_grants(probe_lin.host_id, _all_grants(probe_lin.host_id, RiskTier.R2))
        await repo.set_grants(idle_mac.host_id, _all_grants(idle_mac.host_id, RiskTier.R2))

        for h in (off_mac, den_win, idle_mac):
            await repo.save_manifest(
                Manifest(
                    host_id=h.host_id,
                    tools=(
                        _tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1, idempotent=False),
                    ),
                    updated_at=NOW,
                )
            )
        await repo.save_manifest(
            Manifest(
                host_id=probe_lin.host_id,
                tools=(
                    _tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1, idempotent=False),
                    _tool("media.now_playing", ToolFamily.MEDIA, RiskTier.R0, observes=True),
                ),
                updated_at=NOW,
            )
        )

        probe_mode = {"status": "timeout"}

        async def _send_probe_lin(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                st = probe_mode["status"]
                hub.on_response(
                    str(probe_lin.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": st, "xuna/errorCode": st}},
                )

        async def _noop(_raw: str) -> None:
            return None

        async def _close() -> None:
            return None

        # Open sessions for den_win (online, denied) and probe_lin (online, no HostTwin presence)
        await hub.open_session(
            str(den_win.host_id), object(), _noop, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(probe_lin.host_id), object(), _send_probe_lin, _close, _close, owner_id=owner
        )

        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )

        # 1. Mixed [offline_host, denied_host] -> MUST fail closed (denied / grant_denied)
        msg_mix = await orch.handle_call(
            ToolCall(
                id="sp_mix_off_den",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(off_mac.host_id), str(den_win.host_id)]},
            ),
            ctx,
        )
        out_mix = json.loads(msg_mix.content)
        assert out_mix["status"] == "denied"
        assert out_mix["errorCode"] == "grant_denied"
        assert out_mix["pausedHosts"] == []
        assert out_mix["skippedHosts"] == [
            {"hostId": str(off_mac.host_id), "alias": "AlphaMac", "status": "offline"}
        ]
        assert out_mix["deniedHosts"] == [
            {
                "hostId": str(den_win.host_id),
                "alias": "BetaWin",
                "status": "denied",
                "errorCode": "grant_denied",
            }
        ]
        ev_mix = next(e for e in events if isinstance(e, ActionResultEvent))
        assert ev_mix.status == "denied"
        assert ev_mix.error_code == "grant_denied"

        # 2. All online hosts without HostTwin presence return probe timeout/error -> failed
        for p_status in ("timeout", "failed"):
            probe_mode["status"] = p_status
            events.clear()
            msg_probe = await orch.handle_call(
                ToolCall(
                    id=f"sp_probe_{p_status}",
                    name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                    arguments={"hostIds": [str(probe_lin.host_id)]},
                ),
                ctx,
            )
            out_probe = json.loads(msg_probe.content)
            assert out_probe["status"] == "failed"
            assert out_probe["errorCode"] == "probe_failed"
            assert out_probe["pausedHosts"] == []
            ev_probe = next(e for e in events if isinstance(e, ActionResultEvent))
            assert ev_probe.status == "failed"
            assert ev_probe.error_code == "probe_failed"
            assert ev_probe.mission_envelope is not None
            assert all(s["status"] != "ok" for s in ev_probe.mission_envelope["steps"])  # type: ignore[index]

        # 3. Fleet-wide denied + probe timeout (zero idle) -> fails closed (denied)
        events.clear()
        probe_mode["status"] = "timeout"
        msg_fleet_mix = await orch.handle_call(
            ToolCall(id="sp_fleet_den_to", name=MESH_SMART_PAUSE_MEDIA_FUNCTION, arguments={}),
            ctx,
        )
        out_fleet_mix = json.loads(msg_fleet_mix.content)
        assert out_fleet_mix["status"] == "denied"
        assert out_fleet_mix["errorCode"] == "grant_denied"
        ev_fleet_mix = next(e for e in events if isinstance(e, ActionResultEvent))
        assert ev_fleet_mix.status == "denied"

        # 4. At least 1 host confirmed idle -> preserves existing no_active_media / ok contract
        await hub.open_session(
            str(idle_mac.host_id), object(), _noop, _close, _close, owner_id=owner
        )
        hub.update_presence(
            str(idle_mac.host_id),
            HostPresence(idle_seconds=5, screen_locked=False, media_playing=False, updated_at=NOW),
        )

        # 4a. [idle_host, denied_host]
        events.clear()
        msg_idle_den = await orch.handle_call(
            ToolCall(
                id="sp_idle_den",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(idle_mac.host_id), str(den_win.host_id)]},
            ),
            ctx,
        )
        out_idle_den = json.loads(msg_idle_den.content)
        assert out_idle_den["status"] == "no_active_media"
        assert next(e for e in events if isinstance(e, ActionResultEvent)).status == "ok"

        # 4b. Pure [idle_host]
        events.clear()
        msg_pure_idle = await orch.handle_call(
            ToolCall(
                id="sp_pure_idle",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(idle_mac.host_id)]},
            ),
            ctx,
        )
        out_pure_idle = json.loads(msg_pure_idle.content)
        assert out_pure_idle["status"] == "no_active_media"
        assert next(e for e in events if isinstance(e, ActionResultEvent)).status == "ok"

    asyncio.run(_run())


def test_tier5_smart_pause_unsupported_and_mixed_idle_probe_consistency() -> None:
    """Tier 5: unsupported targets fail closed; mixed [idle, probe_timeout] fails closed cleanly."""

    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        repo = _MemHostRepo()
        audit = MemoryAuditLog()

        phone_unsupported = _host(owner, "PhoneNoMedia", HostPlatform.IOS)
        probe_lin = _host(owner, "ProbeLin", HostPlatform.LINUX)
        idle_mac = _host(owner, "IdleMac", HostPlatform.MACOS)
        off_mac = _host(owner, "OfflineMac", HostPlatform.MACOS)
        den_win = _host(owner, "DeniedWin", HostPlatform.WINDOWS)
        for h in (phone_unsupported, probe_lin, idle_mac, off_mac, den_win):
            await repo.save_host(h)

        await repo.set_grants(
            phone_unsupported.host_id, _all_grants(phone_unsupported.host_id, RiskTier.R2)
        )
        await repo.set_grants(probe_lin.host_id, _all_grants(probe_lin.host_id, RiskTier.R2))
        await repo.set_grants(idle_mac.host_id, _all_grants(idle_mac.host_id, RiskTier.R2))
        await repo.set_grants(off_mac.host_id, _all_grants(off_mac.host_id, RiskTier.R2))
        await repo.set_grants(den_win.host_id, [])

        # phone_unsupported lacks media.play_pause completely
        await repo.save_manifest(
            Manifest(host_id=phone_unsupported.host_id, tools=(), updated_at=NOW)
        )
        await repo.save_manifest(
            Manifest(
                host_id=probe_lin.host_id,
                tools=(
                    _tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1, idempotent=False),
                    _tool("media.now_playing", ToolFamily.MEDIA, RiskTier.R0, observes=True),
                ),
                updated_at=NOW,
            )
        )
        for h in (idle_mac, off_mac, den_win):
            await repo.save_manifest(
                Manifest(
                    host_id=h.host_id,
                    tools=(
                        _tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1, idempotent=False),
                    ),
                    updated_at=NOW,
                )
            )

        probe_mode = {"status": "unsupported"}

        async def _send_probe_lin(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                st = probe_mode["status"]
                hub.on_response(
                    str(probe_lin.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": st, "xuna/errorCode": st}},
                )

        async def _noop(_raw: str) -> None:
            return None

        async def _close() -> None:
            return None

        tok_phone = object()
        tok_probe = object()
        await hub.open_session(
            str(phone_unsupported.host_id), tok_phone, _noop, _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(probe_lin.host_id), tok_probe, _send_probe_lin, _close, _close, owner_id=owner
        )

        orch = _make_orch(repo=repo, hub=hub, audit=audit)
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.EN,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
        )

        # Case A1: probe_returns_unsupported (media.now_playing returns unsupported, 0 idle)
        probe_mode["status"] = "unsupported"
        msg_probe_unsup = await orch.handle_call(
            ToolCall(
                id="sp_probe_unsup",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(probe_lin.host_id)]},
            ),
            ctx,
        )
        out_probe_unsup = json.loads(msg_probe_unsup.content)
        assert out_probe_unsup["status"] == "unsupported"
        assert out_probe_unsup["status"] != "no_active_media"
        assert out_probe_unsup["errorCode"] == "unsupported"
        assert out_probe_unsup["skippedHosts"] == [
            {"hostId": str(probe_lin.host_id), "alias": "ProbeLin", "status": "unsupported"}
        ]
        ev_probe_unsup = next(e for e in events if isinstance(e, ActionResultEvent))
        assert ev_probe_unsup.status == "unsupported"
        assert ev_probe_unsup.error_code == "unsupported"
        assert ev_probe_unsup.mission_envelope is not None
        assert [s["status"] for s in ev_probe_unsup.mission_envelope["steps"]] == ["failed"]  # type: ignore[index]

        # Case A2: play_pause_unsupported (active_online host lacks media.play_pause)
        events.clear()
        msg_pp_unsup = await orch.handle_call(
            ToolCall(
                id="sp_pp_unsup",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(phone_unsupported.host_id)]},
            ),
            ctx,
        )
        out_pp_unsup = json.loads(msg_pp_unsup.content)
        assert out_pp_unsup["status"] == "unsupported"
        assert out_pp_unsup["status"] != "no_active_media"
        assert out_pp_unsup["errorCode"] == "unsupported"
        assert out_pp_unsup["skippedHosts"] == [
            {
                "hostId": str(phone_unsupported.host_id),
                "alias": "PhoneNoMedia",
                "status": "unsupported",
            }
        ]
        ev_pp_unsup = next(e for e in events if isinstance(e, ActionResultEvent))
        assert ev_pp_unsup.status == "unsupported"
        assert ev_pp_unsup.error_code == "unsupported"
        assert ev_pp_unsup.mission_envelope is not None
        assert [s["status"] for s in ev_pp_unsup.mission_envelope["steps"]] == ["failed"]  # type: ignore[index]

        # Bring idle_mac and den_win online for Case B and preservation checks
        tok_idle = object()
        tok_den = object()
        await hub.open_session(
            str(idle_mac.host_id), tok_idle, _noop, _close, _close, owner_id=owner
        )
        hub.update_presence(
            str(idle_mac.host_id),
            HostPresence(idle_seconds=5, screen_locked=False, media_playing=False, updated_at=NOW),
        )
        await hub.open_session(str(den_win.host_id), tok_den, _noop, _close, _close, owner_id=owner)

        # Case B: mixed_idle_and_probe_timeout -> fails closed with consistent MissionSteps
        events.clear()
        probe_mode["status"] = "timeout"
        msg_idle_to = await orch.handle_call(
            ToolCall(
                id="sp_idle_to",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(idle_mac.host_id), str(probe_lin.host_id)]},
            ),
            ctx,
        )
        out_idle_to = json.loads(msg_idle_to.content)
        assert out_idle_to["status"] == "failed"
        assert out_idle_to["errorCode"] == "probe_failed"
        assert out_idle_to["pausedHosts"] == []
        ev_idle_to = next(e for e in events if isinstance(e, ActionResultEvent))
        assert ev_idle_to.status == "failed"
        assert ev_idle_to.error_code == "probe_failed"
        assert ev_idle_to.mission_envelope is not None
        step_statuses_to = [s["status"] for s in ev_idle_to.mission_envelope["steps"]]  # type: ignore[index]
        assert "failed" in step_statuses_to
        assert "ok" in step_statuses_to

        # Preservation: [idle_host, denied_host] -> ok / no_active_media with OK + DENIED steps
        events.clear()
        msg_idle_den = await orch.handle_call(
            ToolCall(
                id="sp_idle_den_steps",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(idle_mac.host_id), str(den_win.host_id)]},
            ),
            ctx,
        )
        out_idle_den = json.loads(msg_idle_den.content)
        assert out_idle_den["status"] == "no_active_media"
        ev_idle_den = next(e for e in events if isinstance(e, ActionResultEvent))
        assert ev_idle_den.status == "ok"
        assert ev_idle_den.mission_envelope is not None
        step_statuses_den = {s["status"] for s in ev_idle_den.mission_envelope["steps"]}  # type: ignore[index]
        assert step_statuses_den == {"ok", "denied"}

        # Preservation: [idle_host] -> ok / no_active_media with [OK] step
        events.clear()
        msg_idle_only = await orch.handle_call(
            ToolCall(
                id="sp_idle_only_steps",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(idle_mac.host_id)]},
            ),
            ctx,
        )
        out_idle_only = json.loads(msg_idle_only.content)
        assert out_idle_only["status"] == "no_active_media"
        ev_idle_only = next(e for e in events if isinstance(e, ActionResultEvent))
        assert ev_idle_only.status == "ok"
        assert ev_idle_only.mission_envelope is not None
        assert [s["status"] for s in ev_idle_only.mission_envelope["steps"]] == ["ok"]  # type: ignore[index]

        # Preservation: [off_mac, idle_mac] with filter_ids=None (fleet w/o probe failures)
        hub.close_session(str(probe_lin.host_id), tok_probe)
        hub.close_session(str(den_win.host_id), tok_den)
        hub.close_session(str(phone_unsupported.host_id), tok_phone)
        events.clear()
        msg_fleet_off_idle = await orch.handle_call(
            ToolCall(id="sp_fleet_off_idle", name=MESH_SMART_PAUSE_MEDIA_FUNCTION, arguments={}),
            ctx,
        )
        out_fleet_off_idle = json.loads(msg_fleet_off_idle.content)
        assert out_fleet_off_idle["status"] == "no_active_media"
        ev_fleet_off_idle = next(e for e in events if isinstance(e, ActionResultEvent))
        assert ev_fleet_off_idle.status == "ok"
        assert ev_fleet_off_idle.mission_envelope is not None
        assert [s["status"] for s in ev_fleet_off_idle.mission_envelope["steps"]] == ["ok"]  # type: ignore[index]

        # Preservation: [off_mac, idle_mac] w/ hostIds emits OK + FAILED (never solitary FAILED)
        events.clear()
        msg_explicit_off_idle = await orch.handle_call(
            ToolCall(
                id="sp_explicit_off_idle",
                name=MESH_SMART_PAUSE_MEDIA_FUNCTION,
                arguments={"hostIds": [str(off_mac.host_id), str(idle_mac.host_id)]},
            ),
            ctx,
        )
        out_explicit_off_idle = json.loads(msg_explicit_off_idle.content)
        assert out_explicit_off_idle["status"] == "no_active_media"
        ev_explicit_off_idle = next(e for e in events if isinstance(e, ActionResultEvent))
        assert ev_explicit_off_idle.status == "ok"
        assert ev_explicit_off_idle.mission_envelope is not None
        assert {s["status"] for s in ev_explicit_off_idle.mission_envelope["steps"]} == {  # type: ignore[index]
            "ok",
            "failed",
        }

    asyncio.run(_run())
