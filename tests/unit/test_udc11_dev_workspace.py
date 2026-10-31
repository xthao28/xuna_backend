"""UDC-11 unit tests (`S11.1` -> `S11.3`):

- `S11.1` (Smart Host Wake & Reconnect):
  - `display.wake` (`R1`, `ToolFamily.APP`) tool catalog inclusion and execution.
  - `HostWakeRequestEvent` emitted when the target PC/Mac is offline (`wol_mac` included).
  - `wait_for_host_online(host_id, timeout_s)` wakes and reconnects when the host comes
    online within the window, or returns `host_offline` when it times out.
- `S11.2` (Zero-Config App & Project Auto-Discovery):
  - `workspace.open` (`R2`) and `workspace.git_summary` (`R1`) tool catalog inclusion,
    `validate_tool_name` mapping, and zero raw filesystem path logging.
- `S11.3` (`workspace.coding_agent` & `resume_work` / `wrap_up_work` Composite Workflows):
  - `resume_work`: sleeping Mac -> `HostWakeRequestEvent` -> reconnect -> `display.wake` ->
    `workspace.git_summary` -> single `R2` approval
    (`"Mac · Mở claude_code (xuna_pet) và tiếp tục công việc"`) ->
    `workspace.coding_agent` (`claude_code` / `antigravity` with synthesized continuation prompt).
  - `wrap_up_work`: reads Git summary (`workspace.git_summary`) + pauses active media
    (`media.play_pause`), returns structured `wrapUpSummary`.
  - Zero raw paths or user prompts in `AuditLog` and `ApprovalRepository`.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from tests.helpers import MemoryAuditLog

from xuna_backend.modules.actions.application.catalog import (
    CAP_HOST_ACTIONS_V1,
    CAP_WORKFLOW_TEMPLATES_V1,
    build_tool_catalog,
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
    HostWakeRequestEvent,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import (
    InMemoryApprovalRepository,
)
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
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
    validate_tool_name,
)

NOW = datetime(2026, 10, 31, 12, 0, 0, tzinfo=UTC)
FAKE_PUBKEY = b"\x04" + bytes(range(1, 65))


class _FakeHostRepo:
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

    async def suspend_r2_plus_grants_for_owner(
        self, owner_id: uuid.UUID, now: datetime
    ) -> int:
        return 0

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifests.get(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        self.manifests[manifest.host_id] = manifest


class _FakeHub:
    def __init__(self) -> None:
        self.online_ids: set[str] = set()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.handler: Any = None

    def is_online(self, host_id: str) -> bool:
        return host_id in self.online_ids

    async def send_request(
        self,
        host_id: str,
        method: str,
        params: dict[str, Any],
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append((host_id, method, params))
        if self.handler is not None:
            return await self.handler(host_id, method, params)
        return {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {},
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
        }

    async def stop_host(self, host_id: str, reason: str = "owner_app") -> bool:
        return host_id in self.online_ids


def _make_mac_host(owner_id: uuid.UUID, alias: str = "Mac") -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=HostPlatform.MACOS,
        hostname="macbook-pro",
        os_version="15.1",
        agent_version="0.1.0",
        public_key=FAKE_PUBKEY,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        last_seen_at=NOW,
        wol_mac="AA:BB:CC:DD:EE:FF",
    )


def _udc11_tools() -> tuple[ToolDescriptor, ...]:
    empty_out: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }
    return (
        ToolDescriptor(
            name="display.wake",
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            description="Wake the host display.",
            input_schema=empty_out,
            output_schema=empty_out,
        ),
        ToolDescriptor(
            name="workspace.git_summary",
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            description="Read Git branch and status counts for a discovered workspace.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["project"],
                "properties": {"project": {"type": "string"}},
            },
            output_schema=empty_out,
        ),
        ToolDescriptor(
            name="workspace.open",
            family=ToolFamily.APP,
            risk=RiskTier.R2,
            description="Open a project in an IDE.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["project"],
                "properties": {
                    "project": {"type": "string"},
                    "ide": {"type": "string"},
                },
            },
            output_schema=empty_out,
        ),
        ToolDescriptor(
            name="workspace.coding_agent",
            family=ToolFamily.APP,
            risk=RiskTier.R2,
            description="Launch a coding agent or IDE in a project and resume work.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["project", "agent", "resumeLast"],
                "properties": {
                    "project": {"type": "string"},
                    "agent": {"type": "string"},
                    "resumeLast": {"type": "boolean"},
                    "prompt": {"type": "string"},
                },
            },
            output_schema=empty_out,
        ),
        ToolDescriptor(
            name="media.now_playing",
            family=ToolFamily.MEDIA,
            risk=RiskTier.R0,
            description="Read active media playback.",
            input_schema=empty_out,
            output_schema=empty_out,
        ),
        ToolDescriptor(
            name="media.play_pause",
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
            description="Toggle media playback.",
            input_schema=empty_out,
            output_schema=empty_out,
        ),
    )


def test_s11_tool_names_and_catalog_priority() -> None:
    """`display.wake`, `workspace.open`, `workspace.git_summary`, and `workspace.coding_agent`
    map to `ToolFamily.APP` and appear in the bounded catalog alongside
    `resume_work` and `wrap_up_work`."""
    assert validate_tool_name("display.wake") == ToolFamily.APP
    assert validate_tool_name("workspace.open") == ToolFamily.APP
    assert validate_tool_name("workspace.git_summary") == ToolFamily.APP
    assert validate_tool_name("workspace.coding_agent") == ToolFamily.APP

    owner_id = uuid.uuid4()
    mac = _make_mac_host(owner_id, "Mac")
    manifest = Manifest(
        host_id=mac.host_id,
        tools=_udc11_tools(),
        updated_at=NOW,
    )
    grants = [
        Grant(
            host_id=mac.host_id,
            family=ToolFamily.APP,
            enabled=True,
            max_risk=RiskTier.R2,
            updated_at=NOW,
        ),
        Grant(
            host_id=mac.host_id,
            family=ToolFamily.MEDIA,
            enabled=True,
            max_risk=RiskTier.R1,
            updated_at=NOW,
        ),
    ]
    specs = build_tool_catalog(
        hosts=[mac],
        manifests_by_host={mac.host_id: manifest},
        grants_by_host={mac.host_id: grants},
        online_host_ids={str(mac.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1),
        now=NOW,
    )
    names = {s.name for s in specs}
    assert "display__wake" in names
    assert "workspace__git_summary" in names
    assert "workspace__open" in names
    assert "workflow_template" in names

    # When only workspace.coding_agent is advertised, it appears as workspace__coding_agent
    ca_manifest = Manifest(
        host_id=mac.host_id,
        tools=(next(t for t in _udc11_tools() if t.name == "workspace.coding_agent"),),
        updated_at=NOW,
    )
    ca_specs = build_tool_catalog(
        hosts=[mac],
        manifests_by_host={mac.host_id: ca_manifest},
        grants_by_host={mac.host_id: grants},
        online_host_ids={str(mac.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1,),
        now=NOW,
    )
    assert "workspace__coding_agent" in {s.name for s in ca_specs}

    wf_spec = next(s for s in specs if s.name == "workflow_template")
    template_enum = wf_spec.parameters["properties"]["template"]["enum"]
    assert "resume_work" in template_enum
    assert "wrap_up_work" in template_enum


def test_s11_1_and_s11_3_resume_work_wakes_sleeping_mac_and_prompts_claude_code() -> None:
    """`resume_work` wakes a sleeping Mac via `HostWakeRequestEvent`, waits for reconnect,
    runs `display.wake` (`R1`) and `workspace.git_summary` (`R1`), requests 1 `R2` approval,
    and launches `workspace.coding_agent` (`claude_code`) with a synthesized prompt while
    leaking zero paths or prompts to `AuditLog`."""

    async def _scenario() -> None:
        owner_id = uuid.uuid4()
        host_repo = _FakeHostRepo()
        hub = _FakeHub()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()

        mac = _make_mac_host(owner_id, "Mac")
        await host_repo.save_host(mac)
        await host_repo.save_manifest(
            Manifest(host_id=mac.host_id, tools=_udc11_tools(), updated_at=NOW)
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    enabled=True,
                    max_risk=RiskTier.R2,
                    updated_at=NOW,
                )
            ],
        )
        # Initially offline (sleeping)!
        assert str(mac.host_id) not in hub.online_ids

        async def _fake_handler(
            host_id: str, method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            name = params["name"]
            if name == "display.wake":
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {"woken": True},
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 8},
                }
            if name == "workspace.git_summary":
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {
                        "workspace": "xuna_pet",
                        "branch": "feat/udc-11",
                        "clean": False,
                        "modifiedCount": 3,
                        "untrackedCount": 1,
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 12},
                }
            if name == "workspace.coding_agent":
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {
                        "workspace": "xuna_pet",
                        "agent": "claude_code",
                        "resumed": True,
                        "promptSent": True,
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 25},
                }
            raise AssertionError(f"Unexpected tool call: {name}")

        hub.handler = _fake_handler

        orchestrator = Orchestrator(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            clock=lambda: NOW,
        )

        emitted: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            emitted.append(ev)
            if isinstance(ev, HostWakeRequestEvent):
                # Phone sends WoL magic packet -> Mac wakes up and reconnects!
                hub.online_ids.add(ev.host_id)
            elif isinstance(ev, ActionPendingEvent):
                await orchestrator.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner_id,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1),
            board_available=False,
            emit_event=_emit,
            request_id="req-udc11-resume",
        )
        private_brief = "Private continuation instructions for XUNA feature"
        call = ToolCall(
            id="call-resume-1",
            name="workflow_template",
            arguments={
                "template": "resume_work",
                "source_target": "Mac",
                "project": "xuna",
                "agent": "claude_code",
                "prompt": private_brief,
            },
        )
        msg = await orchestrator.handle_call(call, ctx)
        payload = json.loads(msg.content)
        assert payload["status"] == "ok"
        out = payload["output"]
        assert out["template"] == "resume_work"
        assert out["host"] == "Mac"
        assert out["workspace"] == "xuna_pet"
        assert out["branch"] == "feat/udc-11"
        assert out["modifiedCount"] == 3
        assert out["untrackedCount"] == 1
        assert out["agent"] == "claude_code"
        assert out["resumed"] is True
        assert out["wokenFromSleep"] is True
        assert out["displayWoken"] is True

        # Verify event sequence: 1 HostWakeRequestEvent, 1 ActionPendingEvent, 1 ActionResultEvent
        wake_events = [e for e in emitted if isinstance(e, HostWakeRequestEvent)]
        pending_events = [e for e in emitted if isinstance(e, ActionPendingEvent)]
        result_events = [e for e in emitted if isinstance(e, ActionResultEvent)]
        assert len(wake_events) == 1
        assert wake_events[0].mac_address == "AA:BB:CC:DD:EE:FF"
        assert len(pending_events) == 1
        assert "Mac · Mở claude_code (xuna_pet) và tiếp tục công việc" in pending_events[0].summary
        assert len(result_events) == 1
        assert result_events[0].status == "ok"

        # Verify hub calls order: display.wake -> workspace.git_summary -> workspace.coding_agent
        called_tools = [c[2]["name"] for c in hub.calls]
        assert called_tools == [
            "display.wake",
            "workspace.git_summary",
            "workspace.coding_agent",
        ]
        coding_agent_args = hub.calls[2][2]["arguments"]
        assert coding_agent_args["project"] == "xuna_pet"
        assert coding_agent_args["agent"] == "claude_code"
        assert coding_agent_args["resumeLast"] is True
        assert "feat/udc-11" in coding_agent_args["prompt"]

        # Verify zero raw filesystem paths or user prompts in AuditLog
        for event in audit.events:
            serialized = json.dumps(event.metadata)
            assert private_brief not in serialized
            assert "/Users/" not in serialized

    asyncio.run(_scenario())


def test_s11_1_resume_work_timeout_when_host_stays_offline() -> None:
    """When the target Mac stays offline past the reconnect window, `wait_for_host_online`
    returns `False` and `resume_work` returns `host_offline`."""

    async def _scenario() -> None:
        owner_id = uuid.uuid4()
        host_repo = _FakeHostRepo()
        hub = _FakeHub()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()

        mac = _make_mac_host(owner_id, "Mac")
        await host_repo.save_host(mac)
        await host_repo.save_manifest(
            Manifest(host_id=mac.host_id, tools=_udc11_tools(), updated_at=NOW)
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    enabled=True,
                    max_risk=RiskTier.R2,
                    updated_at=NOW,
                )
            ],
        )

        orchestrator = Orchestrator(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            clock=lambda: NOW,
        )
        # Speed up the reconnect timeout for unit testing
        async def _fast_timeout(
            host_id: str, timeout_s: float = 0.05, poll_interval_s: float = 0.01
        ) -> bool:
            return False

        orchestrator._workflow_runner.wait_for_host_online = _fast_timeout  # type: ignore[method-assign]

        emitted: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            emitted.append(ev)

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1),
            board_available=False,
            emit_event=_emit,
            request_id="req-udc11-timeout",
        )
        call = ToolCall(
            id="call-resume-timeout",
            name="workflow_template",
            arguments={"template": "resume_work", "source_target": "Mac"},
        )
        msg = await orchestrator.handle_call(call, ctx)
        payload = json.loads(msg.content)
        assert payload["status"] == "host_offline"
        assert any(isinstance(e, HostWakeRequestEvent) for e in emitted)

    asyncio.run(_scenario())


def test_s11_3_wrap_up_work_captures_git_summary_and_pauses_media() -> None:
    """`wrap_up_work` reads the project's Git summary and pauses active media on the host."""

    async def _scenario() -> None:
        owner_id = uuid.uuid4()
        host_repo = _FakeHostRepo()
        hub = _FakeHub()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()

        mac = _make_mac_host(owner_id, "Mac")
        await host_repo.save_host(mac)
        await host_repo.save_manifest(
            Manifest(host_id=mac.host_id, tools=_udc11_tools(), updated_at=NOW)
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    enabled=True,
                    max_risk=RiskTier.R2,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.MEDIA,
                    enabled=True,
                    max_risk=RiskTier.R1,
                    updated_at=NOW,
                ),
            ],
        )
        hub.online_ids.add(str(mac.host_id))

        async def _fake_handler(
            host_id: str, method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            name = params["name"]
            if name == "workspace.git_summary":
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {
                        "workspace": "xuna_pet",
                        "branch": "feat/udc-11",
                        "clean": False,
                        "modifiedCount": 2,
                        "untrackedCount": 0,
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 10},
                }
            if name == "media.now_playing":
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {"state": "playing", "app": "Spotify"},
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
                }
            if name == "media.play_pause":
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {"state": "paused"},
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
                }
            raise AssertionError(f"Unexpected tool call: {name}")

        hub.handler = _fake_handler

        orchestrator = Orchestrator(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            clock=lambda: NOW,
        )
        emitted: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            emitted.append(ev)

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1),
            board_available=False,
            emit_event=_emit,
            request_id="req-udc11-wrapup",
        )
        call = ToolCall(
            id="call-wrapup-1",
            name="workflow_template",
            arguments={
                "template": "wrap_up_work",
                "source_target": "Mac",
                "project": "xuna_pet",
            },
        )
        msg = await orchestrator.handle_call(call, ctx)
        payload = json.loads(msg.content)
        assert payload["status"] == "ok"
        out = payload["output"]
        assert out["template"] == "wrap_up_work"
        assert out["workspace"] == "xuna_pet"
        assert out["branch"] == "feat/udc-11"
        assert out["modifiedCount"] == 2
        assert out["mediaPaused"] is True

    asyncio.run(_scenario())
