"""Tests for ActionPendingEvent missionEnvelope wire serialization and propagation (Wave E M1)."""

import asyncio
import json
import uuid
from datetime import UTC, datetime

from xuna_backend.modules.actions.application.catalog import (
    CAP_HOST_ACTIONS_V1,
    build_tool_catalog,
)
from xuna_backend.modules.actions.application.job_runner import (
    AutonomousJobRunner,
    JobStepOutcome,
)
from xuna_backend.modules.actions.application.orchestrator import ActionContext, Orchestrator
from xuna_backend.modules.actions.domain.entities import ActionPendingEvent, ActionStreamEvent
from xuna_backend.modules.actions.domain.job import AutonomousJob
from xuna_backend.modules.actions.domain.tool_names import AUTONOMOUS_JOB_FUNCTION_NAME
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.job_store import InMemoryAutonomousJobStore
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.audit.domain.event import AuditEvent
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
)


def test_action_pending_wire_payload_without_mission_envelope() -> None:
    """Standard 1-step action pending event omits missionEnvelope on the wire."""
    approval_id = str(uuid.uuid4())
    event = ActionPendingEvent(
        approval_id=approval_id,
        target="MacBook Pro",
        summary="Phát nhạc trên Spotify",
        risk="R2",
        expires_in=120,
        host_id="host-123",
        args_sha256="abc123sha",
        expires_at="2026-11-15T12:00:00Z",
        standing_tool="media.play",
        standing_arguments={"query": "lofi"},
        standing_approved_count=3,
        mission_envelope=None,
    )

    payload = event.wire_payload()

    assert payload["approvalId"] == approval_id
    assert payload["target"] == "MacBook Pro"
    assert payload["summary"] == "Phát nhạc trên Spotify"
    assert payload["risk"] == "r2"  # strictly lowercase for mobile ActionEventMapper
    assert payload["expiresIn"] == 120
    assert payload["hostId"] == "host-123"
    assert payload["argsSha256"] == "abc123sha"
    assert payload["expiresAt"] == "2026-11-15T12:00:00Z"
    assert payload["standing"] == {
        "tool": "media.play",
        "arguments": {"query": "lofi"},
        "approvedCount": 3,
    }
    # Backward compatibility: missionEnvelope key MUST NOT exist when None
    assert "missionEnvelope" not in payload


def test_action_pending_wire_payload_with_mission_envelope() -> None:
    """Multi-step mission / AutonomousJob event includes missionEnvelope on the wire."""
    approval_id = str(uuid.uuid4())
    envelope = {
        "goal": "Viết email báo cáo quý 4 và gửi cho sếp",
        "apps": ["mail", "calc"],
        "domains": ["internal.company.com"],
        "maxSteps": 20,
        "maxMinutes": 10,
        "planSteps": [
            "Mở ứng dụng tính toán để kiểm tra số liệu",
            "Mở Mail và soạn nội dung bản thảo",
            "Gửi email báo cáo",
        ],
    }
    event = ActionPendingEvent(
        approval_id=approval_id,
        target="Office PC",
        summary="Bắt đầu phiên điều khiển: Viết email báo cáo",
        risk="R3",
        expires_in=180,
        host_id="host-456",
        args_sha256="xyz789sha",
        expires_at="2026-11-15T12:03:00Z",
        mission_envelope=envelope,
    )

    payload = event.wire_payload()

    assert payload["approvalId"] == approval_id
    assert payload["target"] == "Office PC"
    assert payload["summary"] == "Bắt đầu phiên điều khiển: Viết email báo cáo"
    assert payload["risk"] == "r3"
    assert payload["expiresIn"] == 180
    assert payload["hostId"] == "host-456"
    assert payload["argsSha256"] == "xyz789sha"
    assert payload["expiresAt"] == "2026-11-15T12:03:00Z"
    assert "standing" not in payload
    # Wire compatibility: missionEnvelope must be present with identical dict
    assert "missionEnvelope" in payload
    assert payload["missionEnvelope"] == envelope


class _AuditStub:
    async def record(self, event: AuditEvent) -> None:
        pass

    async def recent(self, owner_id: uuid.UUID, limit: int) -> list[AuditEvent]:
        return []


class _HostRepoStub:
    def __init__(self, hosts: list[Host], manifests: dict[HostId, Manifest]) -> None:
        self._hosts = {h.host_id: h for h in hosts}
        self._manifests = manifests
        self._grants: dict[HostId, list[Grant]] = {}

    async def get_host(self, host_id: HostId) -> Host | None:
        return self._hosts.get(host_id)

    async def save_host(self, host: Host) -> None:
        self._hosts[host.host_id] = host

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        return [h for h in self._hosts.values() if h.owner_id == owner_id]

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self._grants.get(host_id, []))

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self._grants[host_id] = list(grants)

    async def suspend_r2_plus_grants_for_owner(self, owner_id: uuid.UUID, now: datetime) -> int:
        return 0

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self._manifests.get(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        self._manifests[manifest.host_id] = manifest


class _HubStub:
    def __init__(self, online_ids: set[str]) -> None:
        self.online_ids = online_ids

    def is_online(self, host_id: str) -> bool:
        return host_id in self.online_ids

    async def send_request(
        self,
        host_id: str,
        method: str,
        params: dict[str, object],
        timeout_s: float = 10.0,
    ) -> dict[str, object]:
        return {"_meta": {"xuna/status": "ok"}, "structuredContent": {}}

    async def send_notification(self, host_id: str, method: str, params: dict[str, object]) -> bool:
        return True

    async def broadcast_stop(self, host_ids: list[str], reason: str) -> int:
        return 0


def test_autonomous_job_chat_tool_lifecycle_and_fail_closed() -> None:
    """Chat-level `autonomous_job` starts, queries, pauses, steers, cancels and fails closed."""

    async def _run() -> None:
        now = datetime(2026, 11, 16, 12, 0, 0, tzinfo=UTC)
        owner_id = uuid.uuid4()
        host_id = HostId(uuid.uuid4())
        host = Host(
            host_id=host_id,
            owner_id=owner_id,
            alias=HostAlias("MacBook"),
            platform=HostPlatform.MACOS,
            hostname="macbook.local",
            os_version="12.6",
            agent_version="0.1.0",
            public_key=b"\x04" + bytes(range(1, 65)),
            status=HostStatus.ACTIVE,
            created_at=now,
        )
        tool_desc = ToolDescriptor(
            name="media.now_playing",
            description="Now playing",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        )
        cu_desc = ToolDescriptor(
            name="cu.session_start",
            description="Start computer-use session",
            input_schema={
                "type": "object",
                "properties": {"goal": {"type": "string"}},
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.CU,
            risk=RiskTier.R3,
        )
        manifest = Manifest(host_id=host_id, tools=(tool_desc, cu_desc), updated_at=now)
        grants = [
            Grant(
                host_id=host_id,
                family=ToolFamily.MEDIA,
                enabled=True,
                max_risk=RiskTier.R2,
                updated_at=now,
            ),
            Grant(
                host_id=host_id,
                family=ToolFamily.CU,
                enabled=True,
                max_risk=RiskTier.R3,
                updated_at=now,
            ),
        ]

        # 1. Catalog exposes `autonomous_job` only when online host has CU mission capability
        offline_specs = build_tool_catalog(
            hosts=[host],
            manifests_by_host={host_id: manifest},
            grants_by_host={host_id: grants},
            online_host_ids=set(),
            client_caps=[CAP_HOST_ACTIONS_V1],
            host_screen_consent=True,
            include_autonomous_job=True,
            now=now,
        )
        assert offline_specs == ()

        online_specs = build_tool_catalog(
            hosts=[host],
            manifests_by_host={host_id: manifest},
            grants_by_host={host_id: grants},
            online_host_ids={str(host_id)},
            client_caps=[CAP_HOST_ACTIONS_V1],
            host_screen_consent=True,
            include_autonomous_job=True,
            now=now,
        )
        assert any(s.name == AUTONOMOUS_JOB_FUNCTION_NAME for s in online_specs)

        # 2. Fail-closed when job_runner is not configured
        repo = _HostRepoStub([host], {host_id: manifest})
        await repo.set_grants(host_id, grants)
        hub = _HubStub({str(host_id)})
        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="d945c77026fd4f4a8a899bbff4aeed86",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
            request_id="req-1",
        )
        unconfigured_orch = Orchestrator(
            host_repo=repo,
            hub=hub,
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=_AuditStub(),
            clock=lambda: now,
        )
        res_unconf = await unconfigured_orch.handle_call(
            ToolCall(
                id="c0",
                name="autonomous_job",
                arguments={"action": "start", "goal": "Tổng hợp ghi chú"},
            ),
            ctx,
        )
        assert json.loads(res_unconf.content)["output"]["errorCode"] == "executor_not_configured"

        # 3. Configured runner: start -> status -> steer -> pause -> cancel
        gate = asyncio.Event()

        async def _slow_step(job: AutonomousJob) -> JobStepOutcome:
            await gate.wait()
            return JobStepOutcome(kind="completed", stop_reason="done")

        job_store = InMemoryAutonomousJobStore()
        runner = AutonomousJobRunner(
            job_store=job_store,
            step_executor=_slow_step,
            clock_utc=lambda: now,
        )
        orch = Orchestrator(
            host_repo=repo,
            hub=hub,
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=_AuditStub(),
            clock=lambda: now,
            job_runner=runner,
        )

        start_msg = await orch.handle_call(
            ToolCall(
                id="c1",
                name="autonomous_job",
                arguments={"action": "start", "goal": "Tổng hợp ghi chú tuần và tạo nháp"},
            ),
            ctx,
        )
        started = json.loads(start_msg.content)
        assert started["status"] == "ok"
        job_id = started["output"]["jobId"]
        assert started["output"]["state"] == "running"

        status_msg = await orch.handle_call(
            ToolCall(id="c2", name="autonomous_job", arguments={"action": "status"}),
            ctx,
        )
        assert json.loads(status_msg.content)["output"]["jobId"] == job_id

        steer_msg = await orch.handle_call(
            ToolCall(
                id="c3",
                name="autonomous_job",
                arguments={"action": "steer", "instruction": "Chỉ lấy ghi chú tháng 11"},
            ),
            ctx,
        )
        assert json.loads(steer_msg.content)["status"] == "ok"

        pause_msg = await orch.handle_call(
            ToolCall(id="c4", name="autonomous_job", arguments={"action": "pause"}),
            ctx,
        )
        assert json.loads(pause_msg.content)["output"]["state"] == "paused"

        cancel_msg = await orch.handle_call(
            ToolCall(
                id="c5",
                name="autonomous_job",
                arguments={"action": "cancel", "reason": "owner_takeover"},
            ),
            ctx,
        )
        cancelled = json.loads(cancel_msg.content)
        assert cancelled["output"]["state"] == "cancelled"
        assert cancelled["output"]["stopReason"] == "owner_takeover"

    asyncio.run(_run())
