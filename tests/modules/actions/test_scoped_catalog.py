"""Tests for single-host scoped catalog, job_runner hard pinning, and mesh meta-tools."""

import asyncio
import json
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests.helpers import MemoryAuditLog

from xuna_backend.modules.actions.application.catalog import (
    CAP_HOST_ACTIONS_V1,
    build_tool_catalog,
)
from xuna_backend.modules.actions.application.compactor import SubGoalCompactor
from xuna_backend.modules.actions.application.job_runner import (
    AutonomousJobRunner,
    OrchestratorJobStepExecutor,
)
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    Orchestrator,
)
from xuna_backend.modules.actions.application.planner_executor import JobPlannerExecutor
from xuna_backend.modules.actions.application.tool_discovery import (
    MESH_FIND_RESOURCE_FUNCTION,
    MESH_STATUS_FUNCTION,
    META_FUNCTION_NAMES,
    MetaToolArgumentError,
    encode_mesh_meta_payload,
    parse_mesh_find_resource_args,
    parse_mesh_status_args,
)
from xuna_backend.modules.actions.domain.entities import MAX_TOOL_OUTPUT_BYTES
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.job_store import InMemoryAutonomousJobStore
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import (
    ChatFinished,
    ChatRequest,
    Locale,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)
from xuna_backend.modules.conversation.application.chat_turn import build_request
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
from xuna_backend.modules.hosts.infrastructure.session_hub import InMemoryHostSessionHub

NOW = datetime(2026, 11, 20, 19, 0, 0, tzinfo=UTC)
PUB = b"\x04" + b"\x44" * 64


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


def _host(owner_id: uuid.UUID, alias: str, platform: HostPlatform) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname=alias.lower(),
        os_version="15.0",
        agent_version="0.2.0",
        public_key=PUB,
        status=HostStatus.ACTIVE,
        created_at=NOW,
    )


def _tool(
    name: str, family: ToolFamily = ToolFamily.APP, risk: RiskTier = RiskTier.R1
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=family,
        risk=risk,
    )


def test_scoped_catalog_and_mesh_meta_tool_gating() -> None:
    owner = uuid.uuid4()
    mac = _host(owner, "Mac", HostPlatform.MACOS)
    win = _host(owner, "Win", HostPlatform.WINDOWS)
    m_mac = Manifest(
        host_id=mac.host_id,
        tools=(_tool("media.play_pause", ToolFamily.MEDIA), _tool("window.mac_only")),
        updated_at=NOW,
    )
    m_win = Manifest(
        host_id=win.host_id,
        tools=(_tool("media.play_pause", ToolFamily.MEDIA), _tool("window.win_only")),
        updated_at=NOW,
    )
    grants = {
        mac.host_id: (
            Grant(mac.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),
            Grant(mac.host_id, ToolFamily.APP, RiskTier.R1, True, NOW),
        ),
        win.host_id: (
            Grant(win.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),
            Grant(win.host_id, ToolFamily.APP, RiskTier.R1, True, NOW),
        ),
    }

    # Unpinned with 2 active hosts + include_mesh_tools=True -> exposes mesh__*
    unpinned = build_tool_catalog(
        hosts=[mac, win],
        manifests_by_host={mac.host_id: m_mac, win.host_id: m_win},
        grants_by_host=grants,
        online_host_ids={str(mac.host_id), str(win.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1,),
        include_mesh_tools=True,
    )
    unpinned_names = {s.name for s in unpinned}
    assert {MESH_STATUS_FUNCTION, MESH_FIND_RESOURCE_FUNCTION} <= unpinned_names
    assert {"window__mac_only", "window__win_only"} <= unpinned_names

    # Pinned to win -> includes ONLY win's tools and omits mesh__*
    pinned = build_tool_catalog(
        hosts=[mac, win],
        manifests_by_host={mac.host_id: m_mac, win.host_id: m_win},
        grants_by_host=grants,
        online_host_ids={str(mac.host_id), str(win.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1,),
        pinned_host_id=win.host_id,
        include_mesh_tools=True,
    )
    pinned_names = {s.name for s in pinned}
    assert "window__win_only" in pinned_names
    assert "window__mac_only" not in pinned_names
    assert MESH_STATUS_FUNCTION not in pinned_names

    # Single active host -> omits mesh__* even with include_mesh_tools=True
    single = build_tool_catalog(
        hosts=[mac],
        manifests_by_host={mac.host_id: m_mac},
        grants_by_host=grants,
        online_host_ids={str(mac.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1,),
        include_mesh_tools=True,
    )
    assert MESH_STATUS_FUNCTION not in {s.name for s in single}


def test_mesh_meta_tool_argument_parsers_and_orchestrator_dispatch() -> None:
    assert {MESH_STATUS_FUNCTION, MESH_FIND_RESOURCE_FUNCTION} <= META_FUNCTION_NAMES

    with pytest.raises(MetaToolArgumentError):
        parse_mesh_find_resource_args({"query": ""})
    with pytest.raises(MetaToolArgumentError):
        parse_mesh_find_resource_args({"query": "q" * 65})
    with pytest.raises(MetaToolArgumentError):
        parse_mesh_find_resource_args({"query": "x", "extra": 1})
    with pytest.raises(MetaToolArgumentError):
        parse_mesh_status_args({"hostIds": ["not-a-uuid"]})
    assert parse_mesh_status_args({}) == (None, False)
    assert parse_mesh_find_resource_args({"query": "Apollo", "kind": "workspace"}) == (
        "Apollo",
        "workspace",
    )

    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        mac = _host(owner, "Mac", HostPlatform.MACOS)
        await repo.save_host(mac)
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_tool("host.status", ToolFamily.HOST, RiskTier.R0),),
                updated_at=NOW,
            )
        )
        await repo.set_grants(
            mac.host_id, [Grant(mac.host_id, ToolFamily.HOST, RiskTier.R0, True, NOW)]
        )

        async def _send(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(mac.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}},
                )

        async def _close() -> None:
            return None

        await hub.open_session(str(mac.host_id), object(), _send, _close, _close, owner_id=owner)

        orch = Orchestrator(
            host_repo=repo,
            hub=hub,
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=MemoryAuditLog(),
            clock=lambda: NOW,
        )

        async def _sink(_ev: object) -> None:
            return None

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_sink,
        )
        msg = await orch.handle_call(
            ToolCall(id="m1", name=MESH_STATUS_FUNCTION, arguments={"includeVitals": False}),
            ctx,
        )
        parsed = json.loads(msg.content)
        assert parsed["status"] == "ok"
        assert parsed["hosts"][0]["hostId"] == str(mac.host_id)
        assert parsed["hosts"][0]["probeStatus"] == "ok"

    asyncio.run(_run())


def test_job_runner_hard_pinned_host_ignores_focus_on_other_host() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        repo = _MemHostRepo()
        hub = InMemoryHostSessionHub()
        mac = _host(owner, "MacFocus", HostPlatform.MACOS)
        win = _host(owner, "WinPinned", HostPlatform.WINDOWS)
        await repo.save_host(mac)
        await repo.save_host(win)
        await repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(
                    _tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1),
                    _tool("window.mac_only", ToolFamily.APP, RiskTier.R1),
                ),
                updated_at=NOW,
            )
        )
        await repo.save_manifest(
            Manifest(
                host_id=win.host_id,
                tools=(
                    _tool("media.play_pause", ToolFamily.MEDIA, RiskTier.R1),
                    _tool("window.win_only", ToolFamily.APP, RiskTier.R1),
                ),
                updated_at=NOW,
            )
        )
        for h in (mac, win):
            await repo.set_grants(
                h.host_id,
                [
                    Grant(h.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),
                    Grant(h.host_id, ToolFamily.APP, RiskTier.R1, True, NOW),
                ],
            )

        calls_by_host: list[str] = []

        def _make_send(h: Host) -> Any:
            async def _send(raw: str) -> None:
                frame = json.loads(raw)
                if frame.get("method") == "tools/call":
                    calls_by_host.append(str(h.host_id))
                    hub.on_response(
                        str(h.host_id),
                        frame["id"],
                        {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}},
                    )

            return _send

        async def _close() -> None:
            return None

        await hub.open_session(
            str(mac.host_id), object(), _make_send(mac), _close, _close, owner_id=owner
        )
        await hub.open_session(
            str(win.host_id), object(), _make_send(win), _close, _close, owner_id=owner
        )
        hub.set_focus_state(owner, focused_host_id=mac.host_id, source="app", now=NOW)

        seen_requests: list[ChatRequest] = []

        class _Gen:
            async def stream(self, req: ChatRequest) -> Any:
                seen_requests.append(req)
                if len(seen_requests) == 1:
                    yield ToolCallRequested(
                        ToolCall(id="j1", name="media__play_pause", arguments={})
                    )
                else:
                    yield TextDelta("Done.")
                yield ChatFinished(
                    provider="fake", model="m1", input_tokens=5, output_tokens=5, latency_ms=1
                )

        audit = MemoryAuditLog()
        orch = Orchestrator(
            host_repo=repo,
            hub=hub,
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            scope_store=InMemoryApprovalScopeStore(),
            clock=lambda: NOW,
        )
        store = InMemoryAutonomousJobStore()
        step_exec = OrchestratorJobStepExecutor(
            orchestrator_factory=lambda: orch,
            host_repo_factory=lambda: repo,
            hub=hub,
            audit_factory=lambda: audit,
            generator_factory=lambda: _Gen(),
            planner=JobPlannerExecutor(),
            compactor=SubGoalCompactor(),
        )
        runner = AutonomousJobRunner(job_store=store, step_executor=step_exec)
        job = await runner.start_job(
            job_id=uuid.uuid4(),
            owner_id=owner,
            host_id=win.host_id,
            goal="Step 1: pause media\nStep 2: verify",
        )
        for _ in range(40):
            await asyncio.sleep(0.02)
            cur = await store.get(job.job_id)
            if cur is not None and (cur.is_terminal or calls_by_host):
                break
        await runner.shutdown()

        assert calls_by_host == [str(win.host_id)]
        assert seen_requests
        offered = {t.name for t in seen_requests[0].tools}
        assert "window__win_only" in offered
        assert "window__mac_only" not in offered

    asyncio.run(_run())


def test_build_request_omits_roster_without_tools_and_keeps_loc_guardrails() -> None:
    req_no_tools = build_request(
        "hello", tools=(), host_roster="[HOST_ROSTER]\n- Mac\n[/HOST_ROSTER]"
    )
    assert "[HOST_ROSTER]" not in req_no_tools.messages[0].content

    backend_root = Path(__file__).resolve().parents[3]
    git_bin = shutil.which("git") or "git"
    diff_out = subprocess.check_output(  # noqa: S603 - fixed argv, test-only
        [
            git_bin,
            "diff",
            "--numstat",
            "src/xuna_backend/modules/actions/application/orchestrator.py",
            "src/xuna_backend/modules/actions/application/workflows.py",
        ],
        cwd=backend_root,
        text=True,
    ).strip()
    for line in diff_out.splitlines():
        added_s, deleted_s, fname = line.split("\t")
        net = int(added_s) - int(deleted_s)
        if fname.endswith("workflows.py"):
            assert net == 0, f"workflows.py net LOC must be 0, got {net}"
        elif fname.endswith("orchestrator.py"):
            assert net < 50, f"orchestrator.py net LOC must be < 50, got {net}"


def test_encode_mesh_meta_payload_enforces_max_bytes_on_probe_errors_and_advisories() -> None:
    huge_errors = [
        {"hostId": str(uuid.uuid4()), "alias": f"Host{i}", "status": "error", "errorCode": "x" * 64}
        for i in range(80)
    ]
    out = encode_mesh_meta_payload(
        {
            "status": "ok",
            "query": "apollo",
            "matches": [{"id": "m1"}],
            "probeErrors": huge_errors,
            "batteryAdvisories": ["Low battery on A; alternative hosts available: B" * 5] * 20,
        }
    )
    assert len(out.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES
    parsed = json.loads(out)
    assert parsed["status"] == "ok"
    assert parsed["truncated"] is True
    assert parsed["matches"] == [{"id": "m1"}]

    # Extreme small max_bytes still respects limit and preserves status + hosts/matches keys
    tiny = encode_mesh_meta_payload(
        {
            "status": "ok",
            "hosts": [{"hostId": "h1", "blob": "x" * 200}],
            "probeErrors": huge_errors,
        },
        max_bytes=64,
    )
    assert len(tiny.encode("utf-8")) <= 64
    parsed_tiny = json.loads(tiny)
    assert parsed_tiny["status"] == "ok"
    assert parsed_tiny["truncated"] is True
    assert parsed_tiny["hosts"] == []
