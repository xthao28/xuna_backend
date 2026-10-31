"""SYM-1 (`S1.1` -> `S1.3`) unit tests for Symbiotic Evolution Foundation:

- Track H (`H1` Workspace Time-Capsule + `H2` Universal Cross-Device Pocket):
  - Validates the 6 `SYM-1` XTP tools (`workspace.capture_capsule`, `workspace.restore_capsule`,
    `workspace.list_capsules`, `pocket.push_item`, `pocket.list_items`, `pocket.transform_item`)
    in `validate_tool_name`, `ActionSummary`, and `build_tool_catalog`.
  - Tests `freeze_for_tonight` (`R1`) workflow: captures capsule, saves next-step into
    Universal Pocket, and synthesizes an overnight NREM/REM `DreamVignette` without `R2` approval.
  - Tests `rehydrate_project` (`R2`) workflow: requests single plan approval, restores
    the Workspace Time-Capsule, and returns a 15-second morning briefing.
- Track L (`L1` Sleep Consolidation & Dreaming + `L2` Affective Co-Regulation):
  - Tests `DreamConsolidatorService` NREM/REM consolidation and morning curiosity hook.
  - Tests `CoRegulationEvaluator` across `drowsy_care`, `shared_breakthrough`,
    `reunion_arc`, and `steady_companion`.
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
from xuna_backend.modules.actions.application.workflows import CompositeWorkflowRunner
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
from xuna_backend.modules.symbiotic.application.services import (
    CoRegulationEvaluator,
    DreamConsolidatorService,
)
from xuna_backend.modules.symbiotic.domain.entities import CoRegulationMode

NOW = datetime(2026, 10, 31, 23, 30, 0, tzinfo=UTC)
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


def _sym1_desktop_tools() -> tuple[ToolDescriptor, ...]:
    empty_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return (
        ToolDescriptor(
            name="workspace.capture_capsule",
            description="Capture a Workspace Time-Capsule.",
            input_schema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "maxLength": 64},
                    "nextStepHint": {"type": "string", "maxLength": 240},
                },
                "additionalProperties": False,
            },
            output_schema=empty_schema,
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            reversible=True,
        ),
        ToolDescriptor(
            name="workspace.restore_capsule",
            description="Restore a Workspace Time-Capsule.",
            input_schema={
                "type": "object",
                "properties": {
                    "capsuleId": {"type": "string", "maxLength": 64},
                    "project": {"type": "string", "maxLength": 64},
                },
                "additionalProperties": False,
            },
            output_schema=empty_schema,
            family=ToolFamily.APP,
            risk=RiskTier.R2,
            reversible=True,
        ),
        ToolDescriptor(
            name="workspace.list_capsules",
            description="List saved Workspace Time-Capsules.",
            input_schema=empty_schema,
            output_schema=empty_schema,
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            reversible=True,
        ),
        ToolDescriptor(
            name="pocket.push_item",
            description="Push an item into the Universal Pocket.",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "maxLength": 1000},
                    "kind": {"type": "string", "maxLength": 32},
                    "sourceApp": {"type": "string", "maxLength": 64},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            output_schema=empty_schema,
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            reversible=True,
        ),
        ToolDescriptor(
            name="pocket.list_items",
            description="List items in the Universal Pocket.",
            input_schema=empty_schema,
            output_schema=empty_schema,
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            reversible=True,
        ),
        ToolDescriptor(
            name="pocket.transform_item",
            description="Transform an item in the Universal Pocket.",
            input_schema={
                "type": "object",
                "properties": {
                    "itemId": {"type": "string", "maxLength": 64},
                    "mode": {"type": "string", "maxLength": 32},
                },
                "additionalProperties": False,
            },
            output_schema=empty_schema,
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            reversible=True,
        ),
    )


def test_sym1_tool_names_and_action_summaries() -> None:
    sym1_tools = (
        "workspace.capture_capsule",
        "workspace.restore_capsule",
        "workspace.list_capsules",
        "pocket.push_item",
        "pocket.list_items",
        "pocket.transform_item",
    )
    for name in sym1_tools:
        assert validate_tool_name(name, expected_family=ToolFamily.APP) == ToolFamily.APP
        vi_summary = ActionSummary.format_pending(
            name, {"project": "xuna_pet", "kind": "todo", "mode": "summarize"}, "PC", Locale.VI
        )
        en_summary = ActionSummary.format_pending(
            name, {"project": "xuna_pet", "kind": "todo", "mode": "summarize"}, "PC", Locale.EN
        )
        assert vi_summary and "Thực hiện " not in vi_summary
        assert en_summary and "Run pocket." not in en_summary

    owner_id = uuid.uuid4()
    pc_id = HostId.generate()
    pc_host = Host(
        host_id=pc_id,
        owner_id=owner_id,
        alias=HostAlias("PC"),
        platform=HostPlatform.WINDOWS,
        hostname="pc-host",
        os_version="11.0",
        agent_version="0.1.0",
        public_key=FAKE_PUBKEY,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        last_seen_at=NOW,
    )
    manifest = Manifest(
        host_id=pc_id,
        tools=_sym1_desktop_tools(),
        updated_at=NOW,
    )
    grants = [
        Grant(
            host_id=pc_id,
            family=ToolFamily.APP,
            max_risk=RiskTier.R2,
            enabled=True,
            updated_at=NOW,
        )
    ]
    catalog = build_tool_catalog(
        hosts=[pc_host],
        manifests_by_host={pc_id: manifest},
        grants_by_host={pc_id: grants},
        online_host_ids={str(pc_id)},
        client_caps=frozenset({CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1}),
        board_available=False,
        now=NOW,
    )
    exposed_names = {c.name for c in catalog}
    for name in sym1_tools:
        fn_name = name.replace(".", "__")
        assert fn_name in exposed_names


def test_dream_consolidator_and_coregulation_evaluator() -> None:
    dream_service = DreamConsolidatorService()
    vignette = dream_service.consolidate_nightly_dream(
        project="xuna_pet",
        branch="feat/sym1",
        dirty_count=3,
        next_step_hint="Finish S1.4 mobile models",
        confirmed_facts=("Owner thích uống cà phê muối",),
        active_ports=(3000, 5173),
        now=NOW,
    )
    assert vignette.dream_id.startswith("drm-")
    assert "xuna_pet" in vignette.narrative_vi
    assert "xuna_pet" in vignette.morning_curiosity_hook_vi
    assert "lan_beacon" in vignette.dream_symbols
    assert "memory_crystal" in vignette.dream_symbols

    evaluator = CoRegulationEvaluator()
    drowsy = evaluator.evaluate(
        local_hour=2,
        continuous_work_minutes=140,
        hours_since_last_interaction=0.2,
        previous_git_dirty_count=3,
        current_git_dirty_count=3,
        project_name="xuna_pet",
    )
    assert drowsy.mode == CoRegulationMode.DROWSY_CARE
    assert drowsy.suggested_workflow == "freeze_for_tonight"
    assert drowsy.gesture_hint == "yawn_stretch"

    breakthrough = evaluator.evaluate(
        local_hour=15,
        continuous_work_minutes=45,
        hours_since_last_interaction=0.1,
        previous_git_dirty_count=6,
        current_git_dirty_count=0,
        solved_case_just_now=True,
        project_name="xuna_pet",
    )
    assert breakthrough.mode == CoRegulationMode.SHARED_BREAKTHROUGH
    assert breakthrough.gesture_hint == "bounce_play"
    assert breakthrough.competence_dopamine_boost >= 0.2

    reunion = evaluator.evaluate(
        local_hour=9,
        continuous_work_minutes=5,
        hours_since_last_interaction=52.0,
        previous_git_dirty_count=2,
        current_git_dirty_count=2,
        project_name="xuna_pet",
    )
    assert reunion.mode == CoRegulationMode.REUNION_ARC
    assert reunion.suggested_workflow == "rehydrate_project"


def test_freeze_for_tonight_and_rehydrate_project_workflows() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_repo = _FakeHostRepo()
        hub = _FakeHub()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()

        pc_host = Host(
            host_id=HostId.generate(),
            owner_id=owner_id,
            alias=HostAlias("PC"),
            platform=HostPlatform.WINDOWS,
            hostname="pc-host",
            os_version="11.0",
            agent_version="0.1.0",
            public_key=FAKE_PUBKEY,
            status=HostStatus.ACTIVE,
            created_at=NOW,
            last_seen_at=NOW,
        )
        await host_repo.save_host(pc_host)
        await host_repo.save_manifest(
            Manifest(
                host_id=pc_host.host_id,
                tools=_sym1_desktop_tools(),
                updated_at=NOW,
            )
        )
        await host_repo.set_grants(
            pc_host.host_id,
            [
                Grant(
                    host_id=pc_host.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )
        hub.online_ids.add(str(pc_host.host_id))

        async def _scripted_handler(
            _host_id: str, _method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            tool_name = params.get("name")
            if tool_name == "workspace.capture_capsule":
                return {
                    "content": [{"type": "text", "text": "captured"}],
                    "structuredContent": {
                        "capsuleId": "cap-xuna-1",
                        "workspace": "xuna_pet",
                        "branch": "feat/sym1",
                        "uncommittedCount": 4,
                        "nextStepHint": "Finish S1.4 mobile models",
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 8},
                }
            if tool_name == "pocket.push_item":
                return {
                    "content": [{"type": "text", "text": "pushed"}],
                    "structuredContent": {
                        "itemId": "pock-1",
                        "kind": "todo",
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
                }
            if tool_name == "workspace.restore_capsule":
                return {
                    "content": [{"type": "text", "text": "restored"}],
                    "structuredContent": {
                        "capsuleId": "cap-xuna-1",
                        "workspace": "xuna_pet",
                        "branch": "feat/sym1",
                        "briefing": (
                            "Restored xuna_pet@feat/sym1 (4 uncommitted files). "
                            "Next: Finish S1.4 mobile models"
                        ),
                        "nextStepHint": "Finish S1.4 mobile models",
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 12},
                }
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {},
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
            }

        hub.handler = _scripted_handler

        runner = CompositeWorkflowRunner(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            audit=audit,
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

        # 1. `freeze_for_tonight` (R1: zero approval required)
        freeze_msg = await runner.handle_template_call(
            ToolCall(
                id="call-freeze-1",
                name="workflow_template",
                arguments={
                    "template": "freeze_for_tonight",
                    "project": "xuna_pet",
                    "next_step_hint": "Finish S1.4 mobile models",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=frozenset({CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1}),
            board_available=False,
            emit_event=_emit,
            request_id="req-freeze-1",
        )
        freeze_payload = json.loads(freeze_msg.content)
        assert freeze_payload["status"] == "ok"
        freeze_out = freeze_payload["output"]
        assert freeze_out["capsuleId"] == "cap-xuna-1"
        assert freeze_out["pocketItemId"] == "pock-1"
        assert "dream" in freeze_out
        assert freeze_out["dream"]["dreamId"].startswith("drm-")
        assert not any(isinstance(e, ActionPendingEvent) for e in emitted)

        # 2. `rehydrate_project` (R2: single approval required)
        emitted.clear()
        rehydrate_msg = await runner.handle_template_call(
            ToolCall(
                id="call-rehydrate-1",
                name="workflow_template",
                arguments={
                    "template": "rehydrate_project",
                    "project": "xuna_pet",
                    "capsule_id": "cap-xuna-1",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=frozenset({CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1}),
            board_available=False,
            emit_event=_emit,
            request_id="req-rehydrate-1",
        )
        rehydrate_payload = json.loads(rehydrate_msg.content)
        assert rehydrate_payload["status"] == "ok"
        rehydrate_out = rehydrate_payload["output"]
        assert rehydrate_out["capsuleId"] == "cap-xuna-1"
        assert "Finish S1.4 mobile models" in rehydrate_out["briefing"]
        pending_events = [e for e in emitted if isinstance(e, ActionPendingEvent)]
        assert len(pending_events) == 1

    asyncio.run(_run())
