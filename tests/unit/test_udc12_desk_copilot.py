"""UDC-12 unit and integration tests (`S12.2` -> `S12.5`):

- `S12.2` & `S12.3`:
  - Validates tool name mapping and deterministic summaries (`ActionSummary`) for
    `selection.read`, `host.lock_screen`, `workspace.watch_task`,
    `mcp.list_servers`, and `mcp.call`.
- `S12.4`:
  - `extract_custom_workflow_candidate` parses Vietnamese routine teaching statements
    (`"Khi mình nói '<trigger>' thì <actions>"`), caps steps at `<= 5`, and rejects `R3`/`R4`
    or forbidden shell/input tools (`ValueError("custom_workflow_risk_forbidden")` /
    `ValueError("forbidden tool prefix")`).
  - `CompositeWorkflowRunner` executes `custom_workflow` with a single consolidated `R2`
    approval covering all steps in the routine.
- `S12.5`:
  - `CompositeWorkflowRunner` executes `focus_session` (`R1`: pauses active media, reads
    workspace Git summary, sends focus notification) and `away_guard` (`R1`: pauses active
    media, reads workspace Git summary, locks screen via `host.lock_screen`).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.helpers import MemoryAuditLog

from xuna_backend.modules.actions.application.workflows import (
    CompositeWorkflowRunner,
    CustomWorkflowDefinition,
    CustomWorkflowStep,
    InMemoryCustomWorkflowStore,
    extract_custom_workflow_candidate,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
    ActionSummary,
    ApprovalDecision,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
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
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
        }

    async def stop_host(self, host_id: str, reason: str = "owner_app") -> bool:
        return host_id in self.online_ids


def _build_desk_copilot_manifest(host_id: HostId) -> Manifest:
    empty_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    any_obj_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "project": {"type": "string", "maxLength": 64},
            "ide": {"type": "string", "maxLength": 32},
            "id": {"type": "string", "maxLength": 64},
            "title": {"type": "string", "maxLength": 80},
            "body": {"type": "string", "maxLength": 240},
        },
        "additionalProperties": False,
    }
    return Manifest(
        host_id=host_id,
        tools=(
            ToolDescriptor(
                name="display.wake",
                description="Wake display.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R1,
            ),
            ToolDescriptor(
                name="workspace.open",
                description="Open discovered project.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R2,
                reversible=False,
            ),
            ToolDescriptor(
                name="workspace.git_summary",
                description="Read workspace Git summary.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R1,
            ),
            ToolDescriptor(
                name="workspace.watch_task",
                description="Watch workspace task status.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R1,
                observes=True,
            ),
            ToolDescriptor(
                name="selection.read",
                description="Read selected text with secret redaction.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R1,
                observes=True,
            ),
            ToolDescriptor(
                name="host.lock_screen",
                description="Lock screen immediately.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.HOST,
                risk=RiskTier.R1,
            ),
            ToolDescriptor(
                name="host.notify",
                description="Show desktop notification.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.HOST,
                risk=RiskTier.R1,
            ),
            ToolDescriptor(
                name="media.now_playing",
                description="Read media state.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.MEDIA,
                risk=RiskTier.R0,
            ),
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
                description="Launch app.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R2,
                reversible=False,
            ),
            ToolDescriptor(
                name="mcp.list_servers",
                description="List local MCP servers.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R1,
            ),
            ToolDescriptor(
                name="mcp.call",
                description="Invoke local MCP server tool.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R2,
                reversible=False,
            ),
        ),
        updated_at=NOW,
    )


def test_udc12_tool_names_and_summaries() -> None:
    assert (
        validate_tool_name("selection.read", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )
    assert (
        validate_tool_name("host.lock_screen", expected_family=ToolFamily.HOST)
        == ToolFamily.HOST
    )
    assert (
        validate_tool_name("workspace.watch_task", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )
    assert (
        validate_tool_name("mcp.list_servers", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )
    assert (
        validate_tool_name("mcp.call", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )

    assert (
        ActionSummary.format_pending("selection.read", {}, "MacBook", Locale.VI)
        == "Đọc đoạn văn bản đang chọn trên MacBook"
    )
    assert (
        ActionSummary.format_pending("host.lock_screen", {}, "MacBook", Locale.VI)
        == "Khóa màn hình MacBook"
    )
    assert (
        ActionSummary.format_pending(
            "mcp.call",
            {"server": "github", "tool": "list_prs"},
            "MacBook",
            Locale.VI,
        )
        == "Gọi công cụ MCP github.list_prs trên MacBook"
    )


def test_udc12_teachable_custom_workflow_extraction_and_bounds() -> None:
    owner_id = uuid.uuid4()
    candidate = extract_custom_workflow_candidate(
        owner_id,
        "Khi mình nói 'chế độ chiến' thì bật màn hình, mở dự án xuna_pet, mở spotify và dừng nhạc",
        default_host_alias="MacBook",
    )
    assert candidate is not None
    assert candidate.trigger_phrase == "chế độ chiến"
    assert candidate.host_alias == "MacBook"
    assert len(candidate.steps) == 4
    assert [s.tool for s in candidate.steps] == [
        "display.wake",
        "workspace.open",
        "app.launch",
        "media.play_pause",
    ]
    assert candidate.effective_risk == RiskTier.R2

    # Rejects R3 step inside CustomWorkflowStep
    with pytest.raises(ValueError, match="custom_workflow_risk_forbidden"):
        CustomWorkflowStep(tool="cu.click", arguments={}, risk=RiskTier.R3)

    # Rejects > 5 steps in CustomWorkflowDefinition
    with pytest.raises(ValueError, match="custom_workflow_step_limit"):
        CustomWorkflowDefinition(
            owner_id=owner_id,
            name="too_many",
            trigger_phrase="quá nhiều",
            steps=tuple(
                CustomWorkflowStep(tool="display.wake", arguments={}, risk=RiskTier.R1)
                for _ in range(6)
            ),
        )


def test_udc12_custom_workflow_and_focus_away_guard_execution() -> None:
    async def _scenario() -> None:
        owner_id = uuid.uuid4()
        host_repo = _FakeHostRepo()
        hub = _FakeHub()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()
        custom_store = InMemoryCustomWorkflowStore()

        mac = Host(
            host_id=HostId.generate(),
            owner_id=owner_id,
            alias=HostAlias("MacBook"),
            platform=HostPlatform.MACOS,
            hostname="macbook.local",
            os_version="15.1",
            agent_version="0.1.0",
            public_key=FAKE_PUBKEY,
            status=HostStatus.ACTIVE,
            created_at=NOW,
            last_seen_at=NOW,
        )
        await host_repo.save_host(mac)
        await host_repo.save_manifest(_build_desk_copilot_manifest(mac.host_id))
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.MEDIA,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.HOST,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=NOW,
                ),
            ],
        )
        hub.online_ids.add(str(mac.host_id))

        async def _hub_handler(
            host_id: str, method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            name = params.get("name")
            if name == "media.now_playing":
                return {
                    "content": [{"type": "text", "text": "playing"}],
                    "structuredContent": {"state": "playing", "app": "Spotify"},
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
                }
            if name == "workspace.git_summary":
                return {
                    "content": [{"type": "text", "text": "xuna_pet"}],
                    "structuredContent": {
                        "workspace": "xuna_pet",
                        "branch": "main",
                        "modifiedCount": 2,
                        "untrackedCount": 1,
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 6},
                }
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {},
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
            }

        hub.handler = _hub_handler

        # Save a taught custom workflow and run it with 1 consolidated R2 approval
        candidate = extract_custom_workflow_candidate(
            owner_id,
            "Khi mình nói 'chế độ chiến' thì bật màn hình, mở dự án xuna_pet và dừng nhạc",
            default_host_alias="MacBook",
        )
        assert candidate is not None
        custom_store.save(candidate)

        runner = CompositeWorkflowRunner(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            audit=audit,
            custom_workflow_store=custom_store,
            clock=lambda: NOW,
        )

        emitted: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            emitted.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await approval_broker.decide(
                    uuid.UUID(ev.approval_id),
                    owner_id,
                    ApprovalDecision.APPROVE,
                    NOW,
                )

        custom_msg = await runner.handle_template_call(
            ToolCall(
                id="call-custom-1",
                name="workflow_template",
                arguments={
                    "template": "custom_workflow",
                    "workflow_name": "chế độ chiến",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit,
            request_id="req-custom",
        )
        custom_payload = json.loads(custom_msg.content)
        assert custom_payload["status"] == "ok"
        assert custom_payload["output"]["completedSteps"] == [
            "display.wake",
            "workspace.open",
            "media.play_pause",
        ]
        # Exactly 1 consolidated plan approval was requested for the 3-step routine
        pending_events = [e for e in emitted if isinstance(e, ActionPendingEvent)]
        assert len(pending_events) == 1

        # S12.5: focus_session (R1, no approval prompt needed)
        emitted.clear()
        focus_msg = await runner.handle_template_call(
            ToolCall(
                id="call-focus-1",
                name="workflow_template",
                arguments={
                    "template": "focus_session",
                    "source_target": "MacBook",
                    "duration_minutes": 25,
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit,
            request_id="req-focus",
        )
        focus_payload = json.loads(focus_msg.content)
        assert focus_payload["status"] == "ok"
        assert focus_payload["output"]["durationMinutes"] == 25
        assert focus_payload["output"]["mediaPaused"] is True
        assert focus_payload["output"]["workspace"] == "xuna_pet"

        # S12.5: away_guard (R1: pauses media, saves workspace summary, locks screen)
        away_msg = await runner.handle_template_call(
            ToolCall(
                id="call-away-1",
                name="workflow_template",
                arguments={
                    "template": "away_guard",
                    "source_target": "MacBook",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit,
            request_id="req-away",
        )
        away_payload = json.loads(away_msg.content)
        assert away_payload["status"] == "ok"
        assert away_payload["output"]["locked"] is True
        assert away_payload["output"]["mediaPaused"] is True
        assert away_payload["output"]["workspace"] == "xuna_pet"

    asyncio.run(_scenario())
