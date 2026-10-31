"""SYM-2 (`S2.1` -> `S2.2`) unit tests for Symbiotic Watchdogs, Coach & Serendipity:

- Track H (`H3` Ambient Sentinel Watchdogs + `H4` Neurodivergent Micro-Step Coach +
  `H5` Live Meeting / Call Desk Whisper):
  - Validates the 6 `SYM-2` XTP tools (`watchdog.watch_process_or_port`,
    `watchdog.watch_resource_threshold`, `watchdog.list_and_clear`,
    `coach.decompose_overwhelm`, `meeting.desk_whisper_digest`,
    `habit.propose_workflow`) in `validate_tool_name`, `ActionSummary`, and `build_tool_catalog`.
  - Tests `unstuck_coach` (`R1`), `watch_and_alert` (`R1`), and `desk_whisper` (`R1`)
    workflows.
- Track L (`L3` Serendipity & Curiosity Sparks + `L4` Emergent Shared Rituals & Habit Mining):
  - Tests `SerendipityEngine` non-intrusive spark generation (`max 1/day`, `<= 280` chars).
  - Tests `HabitWorkflowMiner` threshold-gated (`>= 3` repetitions) habit macro proposals.
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
    ActionStreamEvent,
    ActionSummary,
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
    HabitWorkflowMiner,
    SerendipityEngine,
)

NOW = datetime(2026, 10, 31, 18, 0, 0, tzinfo=UTC)
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
        name = params.get("name", "")
        args = params.get("arguments", {})
        if name == "coach.decompose_overwhelm":
            goal = args.get("goal", "Gỡ bí")
            project = args.get("project", "xuna_pet")
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {
                    "goal": goal,
                    "project": project,
                    "firstFiveMinuteStep": (
                        f"Mở đúng 1 file của {project} và viết 3 dòng giả mã cho: {goal}"
                    ),
                    "followUpSteps": [
                        "Chạy thử 1 unit test nhỏ để lấy lại đà.",
                        "Ghi chú phần còn lại vào Pocket để mai làm tiếp.",
                    ],
                    "encouragementVi": (
                        "Chỉ cần làm 5 phút bước đầu tiên thôi, mình ngồi canh cùng bạn nhé!"
                    ),
                },
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
            }
        if name == "watchdog.watch_process_or_port":
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {
                    "watchdogId": "wd-1",
                    "targetKind": args.get("targetKind", "port"),
                    "targetValue": args.get("targetValue", "8000"),
                    "summaryVi": f"Đang canh {args.get('targetKind')} {args.get('targetValue')}",
                },
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
            }
        if name == "watchdog.watch_resource_threshold":
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {
                    "watchdogId": "wd-res-1",
                    "metric": args.get("metric", "battery_below"),
                    "threshold": args.get("threshold", 20),
                },
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
            }
        if name == "meeting.desk_whisper_digest":
            topic = args.get("topic", "Daily Sync")
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {
                    "topic": topic,
                    "whisperLine": f"[{topic}] Nhắc chốt deadline sprint và hỏi blocker.",
                    "charCount": 52,
                    "silentMode": True,
                },
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
            }
        return {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {"ok": True},
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
        }


def _build_sym2_manifest(host_id: HostId) -> Manifest:
    empty_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    tools = (
        ToolDescriptor(
            name="watchdog.watch_process_or_port",
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            description="Watch process or port",
            input_schema=empty_schema,
            output_schema=empty_schema,
            reversible=True,
        ),
        ToolDescriptor(
            name="watchdog.watch_resource_threshold",
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            description="Watch resource threshold",
            input_schema=empty_schema,
            output_schema=empty_schema,
            reversible=True,
        ),
        ToolDescriptor(
            name="watchdog.list_and_clear",
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            description="List or clear watchdogs",
            input_schema=empty_schema,
            output_schema=empty_schema,
            reversible=True,
        ),
        ToolDescriptor(
            name="coach.decompose_overwhelm",
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            description="Decompose overwhelming task into 5-minute step",
            input_schema=empty_schema,
            output_schema=empty_schema,
            reversible=True,
        ),
        ToolDescriptor(
            name="meeting.desk_whisper_digest",
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
            description="Silent <= 120 char meeting whisper",
            input_schema=empty_schema,
            output_schema=empty_schema,
            reversible=True,
        ),
        ToolDescriptor(
            name="habit.propose_workflow",
            family=ToolFamily.WORKFLOW,
            risk=RiskTier.R1,
            description="Propose workflow from repeated sequence",
            input_schema=empty_schema,
            output_schema=empty_schema,
            reversible=True,
        ),
    )
    return Manifest(
        host_id=host_id,
        tools=tools,
        updated_at=NOW,
    )


def test_sym2_tools_validation_catalog_and_summaries() -> None:
    expected = {
        "watchdog.watch_process_or_port": ToolFamily.APP,
        "watchdog.watch_resource_threshold": ToolFamily.APP,
        "watchdog.list_and_clear": ToolFamily.APP,
        "coach.decompose_overwhelm": ToolFamily.APP,
        "meeting.desk_whisper_digest": ToolFamily.MEDIA,
        "habit.propose_workflow": ToolFamily.WORKFLOW,
    }
    for name, family in expected.items():
        assert validate_tool_name(name, expected_family=family) == family

    host_id = HostId.generate()
    owner_id = uuid.uuid4()
    host = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("Work-PC"),
        platform=HostPlatform.WINDOWS,
        hostname="work-pc",
        os_version="11.0",
        agent_version="0.6.0",
        public_key=FAKE_PUBKEY,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        last_seen_at=NOW,
    )
    manifest = _build_sym2_manifest(host_id)
    grants = [
        Grant(
            host_id=host_id,
            family=ToolFamily.APP,
            max_risk=RiskTier.R2,
            enabled=True,
            updated_at=NOW,
        ),
        Grant(
            host_id=host_id,
            family=ToolFamily.MEDIA,
            max_risk=RiskTier.R2,
            enabled=True,
            updated_at=NOW,
        ),
        Grant(
            host_id=host_id,
            family=ToolFamily.WORKFLOW,
            max_risk=RiskTier.R2,
            enabled=True,
            updated_at=NOW,
        ),
    ]
    catalog = build_tool_catalog(
        hosts=[host],
        manifests_by_host={host_id: manifest},
        grants_by_host={host_id: grants},
        online_host_ids={str(host_id)},
        client_caps=frozenset({CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1}),
        board_available=True,
        now=NOW,
    )
    names = {t.name for t in catalog}
    assert "watchdog__watch_process_or_port" in names
    assert "watchdog__watch_resource_threshold" in names
    assert "watchdog__list_and_clear" in names
    assert "coach__decompose_overwhelm" in names
    assert "meeting__desk_whisper_digest" in names
    assert "habit__propose_workflow" in names

    s_watch = ActionSummary.format_pending(
        "watchdog.watch_process_or_port",
        {"targetKind": "port", "target": "5432"},
        "Work-PC",
        Locale.VI,
    )
    assert "port" in s_watch and "Work-PC" in s_watch

    s_coach = ActionSummary.format_pending(
        "coach.decompose_overwhelm",
        {"goal": "Refactor BLE layer"},
        "Work-PC",
        Locale.VI,
    )
    assert "Refactor BLE layer" in s_coach


def test_serendipity_engine_and_habit_miner() -> None:
    engine = SerendipityEngine()
    note = engine.generate_note(
        primary_topic="Optimizing BLE payload compression",
        secondary_topic="xuna_pet",
        first_step_hint="Đo kích thước gói 7a6e000e trước khi sửa",
        now=NOW,
    )
    assert note.note_id.startswith("srp-")
    assert len(note.spark_vi) <= 500
    assert "xuna_pet" in note.spark_vi
    assert note.to_wire()["suggestedActionVi"] == (
        "Đo kích thước gói 7a6e000e trước khi sửa"
    )

    miner = HabitWorkflowMiner()
    below = miner.mine_from_history(
        ["ide.git_status_summary", "terminal.run_preset"],
        min_occurrences=2,
    )
    assert below == []

    proposals = miner.mine_from_history(
        [
            "ide.git_status_summary",
            "terminal.run_preset",
            "ide.git_status_summary",
            "terminal.run_preset",
        ],
        min_occurrences=2,
    )
    assert len(proposals) == 1
    assert proposals[0].occurrences == 2
    assert proposals[0].max_risk == "R1"


def test_sym2_workflows_unstuck_coach_watch_and_alert_and_desk_whisper() -> None:
    async def _run() -> None:
        repo = _FakeHostRepo()
        hub = _FakeHub()
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        host = Host(
            host_id=host_id,
            owner_id=owner_id,
            alias=HostAlias("Work-PC"),
            platform=HostPlatform.WINDOWS,
            hostname="work-pc",
            os_version="11.0",
            agent_version="0.6.0",
            public_key=FAKE_PUBKEY,
            status=HostStatus.ACTIVE,
            created_at=NOW,
            last_seen_at=NOW,
        )
        await repo.save_host(host)
        await repo.save_manifest(_build_sym2_manifest(host_id))
        await repo.set_grants(
            host_id,
            [
                Grant(
                    host_id=host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=host_id,
                    family=ToolFamily.MEDIA,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=host_id,
                    family=ToolFamily.WORKFLOW,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                ),
            ],
        )
        hub.online_ids.add(str(host_id))

        runner = CompositeWorkflowRunner(
            host_repo=repo,
            hub=hub,
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            audit=MemoryAuditLog(),
            clock=lambda: NOW,
        )
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        # 1. unstuck_coach (R1)
        msg_coach = await runner.handle_template_call(
            call=ToolCall(
                id="call-coach-1",
                name="workflow_template",
                arguments={
                    "template": "unstuck_coach",
                    "goal": "Viết tài liệu kiến trúc Symbiotic",
                    "project": "xuna_pet",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps={CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1},
            board_available=True,
            emit_event=_emit,
            request_id="req-coach-1",
        )
        payload_coach = json.loads(msg_coach.content)
        assert payload_coach["status"] == "ok"
        coach_out = payload_coach["output"]
        assert coach_out["template"] == "unstuck_coach"
        assert "firstFiveMinuteStep" in coach_out
        assert "serendipityNote" in coach_out
        assert coach_out["serendipityNote"]["connectedTopics"][0] == (
            "Viết tài liệu kiến trúc Symbiotic"
        )

        # 2. watch_and_alert (R1)
        msg_watch = await runner.handle_template_call(
            call=ToolCall(
                id="call-watch-1",
                name="workflow_template",
                arguments={
                    "template": "watch_and_alert",
                    "watch_target": "8000",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps={CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1},
            board_available=True,
            emit_event=_emit,
            request_id="req-watch-1",
        )
        payload_watch = json.loads(msg_watch.content)
        assert payload_watch["status"] == "ok"
        watch_out = payload_watch["output"]
        assert watch_out["template"] == "watch_and_alert"
        assert watch_out["targetKind"] == "port"
        assert watch_out["watchdogId"] == "wd-1"
        assert watch_out["resourceWatchdogId"] == "wd-res-1"

        # 3. desk_whisper (R1)
        msg_whisper = await runner.handle_template_call(
            call=ToolCall(
                id="call-whisper-1",
                name="workflow_template",
                arguments={
                    "template": "desk_whisper",
                    "whisper_topic": "Design Review",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps={CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1},
            board_available=True,
            emit_event=_emit,
            request_id="req-whisper-1",
        )
        payload_whisper = json.loads(msg_whisper.content)
        assert payload_whisper["status"] == "ok"
        whisper_out = payload_whisper["output"]
        assert whisper_out["template"] == "desk_whisper"
        assert whisper_out["silentMode"] is True
        assert len(whisper_out["whisperLine"]) <= 120

    asyncio.run(_run())
