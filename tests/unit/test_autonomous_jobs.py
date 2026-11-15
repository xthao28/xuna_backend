"""Unit and integration tests for AutonomousJob queue, runner, and APIs (Wave D M1).

Covers:
1. Domain state machine valid and invalid transitions, sub-goals, and serialization.
2. InMemoryAutonomousJobStore operations (save, get, active filtering, count, pagination).
3. AutonomousJobRunner detached lifecycle, step budgets, pause, resume, cancel, steer.
4. R3 Checkpoint pause on unexempt action, pending approval attachment, and auto-resume.
5. Consent U5 Zero-Retention: base64 screenshots never persist in checkpoint_data.
6. Phone-lock heartbeat resilience: runner touches session heartbeat during detached execution.
7. REST API endpoints (`/v1/actions/jobs`) with authentication and owner isolation.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from starlette.testclient import TestClient
from tests.helpers import LocalIssuer, MemoryAuditLog

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.application.compactor import SubGoalCompactor
from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    InMemoryComputerUseSessionStore,
)
from xuna_backend.modules.actions.application.job_runner import (
    AutonomousJobRunner,
    JobStepOutcome,
    OrchestratorJobStepExecutor,
    sanitize_checkpoint_data,
)
from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.application.planner_executor import JobPlannerExecutor
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    SubGoal,
    SubGoalStatus,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.job_store import InMemoryAutonomousJobStore
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
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

# ---------------------------------------------------------------------------
# Test Fakes
# ---------------------------------------------------------------------------


class _DummyDbSession:
    async def commit(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def flush(self) -> None:
        pass


class _DummyDatabase:
    @asynccontextmanager
    async def sessions(self) -> AsyncIterator[_DummyDbSession]:
        yield _DummyDbSession()

    async def dispose(self) -> None:
        pass


class _FakeOwnerRepo:
    def __init__(self, owner: Owner) -> None:
        self._owners: dict[str, Owner] = {owner.subject: owner}

    async def get_or_create(self, claims: Any) -> Owner:
        sub = getattr(claims, "subject", None) or getattr(claims, "sub", str(claims))
        if sub not in self._owners:
            self._owners[sub] = Owner(
                owner_id=uuid.uuid4(),
                issuer=getattr(claims, "issuer", "https://auth.test.xuna"),
                subject=sub,
            )
        return self._owners[sub]

    async def get_by_subject(self, issuer: str, subject: str) -> Owner | None:
        return self._owners.get(subject)

    async def get(self, owner_id: uuid.UUID) -> Owner | None:
        for o in self._owners.values():
            if o.owner_id == owner_id:
                return o
        return None


# ---------------------------------------------------------------------------
# 1. Domain State Machine & Serialization
# ---------------------------------------------------------------------------


def test_job_state_machine_valid_transitions() -> None:
    job_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    job = AutonomousJob(
        job_id=job_id,
        owner_id=owner_id,
        pet_id="xuna_pet",
        host_id=None,
        session_id=None,
        title="Test Task",
        goal="Perform complex task",
        state=AutonomousJobState.QUEUED,
    )
    assert job.is_active is True
    assert job.is_terminal is False

    # QUEUED -> RUNNING
    running = job.with_state(AutonomousJobState.RUNNING)
    assert running.state == AutonomousJobState.RUNNING
    assert running.started_at is not None

    # RUNNING -> AWAITING_R3_CHECKPOINT
    approval_id = uuid.uuid4()
    awaiting = running.with_state(
        AutonomousJobState.AWAITING_R3_CHECKPOINT,
        pending_approval_id=approval_id,
    )
    assert awaiting.state == AutonomousJobState.AWAITING_R3_CHECKPOINT
    assert awaiting.pending_approval_id == approval_id
    assert awaiting.is_active is True

    # AWAITING_R3_CHECKPOINT -> RUNNING (resume)
    resumed = awaiting.with_state(AutonomousJobState.RUNNING)
    assert resumed.state == AutonomousJobState.RUNNING
    assert resumed.pending_approval_id is None

    # RUNNING -> PAUSED -> RUNNING
    paused = resumed.with_state(AutonomousJobState.PAUSED)
    assert paused.state == AutonomousJobState.PAUSED
    assert paused.is_active is True

    unpaused = paused.with_state(AutonomousJobState.RUNNING)
    assert unpaused.state == AutonomousJobState.RUNNING

    # RUNNING -> COMPLETED (terminal)
    completed = unpaused.with_state(AutonomousJobState.COMPLETED)
    assert completed.state == AutonomousJobState.COMPLETED
    assert completed.is_terminal is True
    assert completed.is_active is False
    assert completed.completed_at is not None


def test_job_state_machine_invalid_transitions_raise() -> None:
    job = AutonomousJob(
        job_id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        pet_id="xuna_pet",
        host_id=None,
        session_id=None,
        title="Test",
        goal="Test goal",
        state=AutonomousJobState.COMPLETED,
    )
    assert job.is_terminal is True

    with pytest.raises(ValueError, match="Invalid job state transition"):
        job.with_state(AutonomousJobState.RUNNING)

    cancelled = AutonomousJob(
        job_id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        pet_id="xuna_pet",
        host_id=None,
        session_id=None,
        title="Test",
        goal="Test goal",
        state=AutonomousJobState.CANCELLED,
    )
    with pytest.raises(ValueError, match="Invalid job state transition"):
        cancelled.with_state(AutonomousJobState.PAUSED)


def test_sub_goal_progression_and_serialization() -> None:
    now = datetime(2026, 11, 15, 12, 0, 0, tzinfo=UTC)
    sg = SubGoal(
        id="sg-1",
        title="Launch Target App",
        description="Verify and open desktop tool",
        status=SubGoalStatus.PENDING,
    )
    sg_dict = sg.to_dict()
    assert sg_dict["id"] == "sg-1"
    assert sg_dict["status"] == "pending"
    assert sg_dict["toolCallCount"] == 0

    reconstructed = SubGoal.from_dict(sg_dict)
    assert reconstructed.id == sg.id
    assert reconstructed.title == sg.title
    assert reconstructed.status == SubGoalStatus.PENDING

    # Advance to completed
    completed_sg = SubGoal(
        id=sg.id,
        title=sg.title,
        description=sg.description,
        status=SubGoalStatus.COMPLETED,
        tool_call_count=3,
        result_summary="App opened successfully",
        started_at=now,
        completed_at=now + timedelta(seconds=5),
    )
    comp_dict = completed_sg.to_dict()
    assert comp_dict["status"] == "completed"
    assert comp_dict["toolCallCount"] == 3
    assert comp_dict["resultSummary"] == "App opened successfully"

    parsed_comp = SubGoal.from_dict(comp_dict)
    assert parsed_comp.status == SubGoalStatus.COMPLETED
    assert parsed_comp.tool_call_count == 3


def test_autonomous_job_to_dict_camel_case() -> None:
    job_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    now = datetime(2026, 11, 15, 12, 0, 0, tzinfo=UTC)
    job = AutonomousJob(
        job_id=job_id,
        owner_id=owner_id,
        pet_id="xuna_pet",
        host_id=HostId(uuid.uuid4()),
        session_id="cu-session-123",
        title="Autonomous Research",
        goal="Browse documents and synthesize findings",
        state=AutonomousJobState.RUNNING,
        sub_goals=(
            SubGoal(id="1", title="Step 1", description="desc", status=SubGoalStatus.COMPLETED),
        ),
        active_sub_goal_index=0,
        max_steps=50,
        steps_used=5,
        created_at=now,
        updated_at=now,
        started_at=now,
    )
    data = job.to_dict()
    assert data["jobId"] == str(job_id)
    assert data["ownerId"] == str(owner_id)
    assert data["petId"] == "xuna_pet"
    assert data["sessionId"] == "cu-session-123"
    assert data["state"] == "running"
    assert data["maxSteps"] == 50
    assert data["stepsUsed"] == 5
    assert len(data["subGoals"]) == 1
    assert data["subGoals"][0]["title"] == "Step 1"


# ---------------------------------------------------------------------------
# 2. In-Memory Store Operations
# ---------------------------------------------------------------------------


def test_in_memory_job_store_crud_and_queries() -> None:
    async def _test() -> None:
        store = InMemoryAutonomousJobStore()
        owner1 = uuid.uuid4()
        owner2 = uuid.uuid4()

        job1 = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner1,
            pet_id="pet-1",
            host_id=None,
            session_id=None,
            title="Job 1",
            goal="Goal 1",
            state=AutonomousJobState.QUEUED,
        )
        job2 = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner1,
            pet_id="pet-1",
            host_id=None,
            session_id=None,
            title="Job 2",
            goal="Goal 2",
            state=AutonomousJobState.RUNNING,
        )
        job3 = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner2,
            pet_id="pet-2",
            host_id=None,
            session_id=None,
            title="Job 3",
            goal="Goal 3",
            state=AutonomousJobState.RUNNING,
        )

        await store.save(job1)
        await store.save(job2)
        await store.save(job3)

        # Retrieval
        assert await store.get(job1.job_id) == job1
        assert await store.get(uuid.uuid4()) is None

        # Active job for owner
        active = await store.get_active_for_owner(owner1)
        assert active is not None
        assert active.job_id in {job1.job_id, job2.job_id}

        # List & count for owner
        owner1_jobs = await store.list_for_owner(owner1)
        assert len(owner1_jobs) == 2
        assert await store.count_for_owner(owner1) == 2

        # Filter by state
        running_jobs = await store.list_for_owner(owner1, states=(AutonomousJobState.RUNNING,))
        assert len(running_jobs) == 1
        assert running_jobs[0].job_id == job2.job_id
        assert await store.count_for_owner(owner1, states=(AutonomousJobState.RUNNING,)) == 1

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 3. Autonomous Runner Lifecycle, Honest Fail-Closed & Real Orchestrator Loop
# ---------------------------------------------------------------------------


class _JobHostRepo(HostRepository):
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

    async def suspend_r2_plus_grants_for_owner(
        self, owner_id: uuid.UUID, now: datetime
    ) -> int:
        return 0


class _JobSessionHub(HostSessionHub):
    def __init__(self, *, online: bool = True) -> None:
        self.online = online
        self.invocations: list[tuple[str, dict[str, Any]]] = []

    def register_pending(
        self,
        pending_id: str,
        on_enrolled: Callable[[HostId, HostAlias], Awaitable[None]],
    ) -> None:
        pass

    def unregister_pending(self, pending_id: str) -> None:
        pass

    async def notify_enrolled(
        self, pending_id: str, host_id: HostId, alias: HostAlias
    ) -> None:
        pass

    async def open_session(
        self,
        host_id: str,
        token: object,
        send_text: Callable[[str], Awaitable[None]],
        close_duplicate: Callable[[], Awaitable[None]],
        close_revoked: Callable[[], Awaitable[None]],
    ) -> bool:
        return True

    def close_session(self, host_id: str, token: object) -> bool:
        return True

    def on_response(
        self, host_id: str, request_id: int | str, result: dict[str, object]
    ) -> None:
        pass

    def on_error_response(
        self, host_id: str, request_id: int | str, code: int, message: str
    ) -> None:
        pass

    def is_online(self, host_id: str) -> bool:
        return self.online

    async def send_request(
        self,
        host_id: str,
        method: str,
        params: dict[str, object],
        timeout_s: float,
    ) -> dict[str, object]:
        tool_name = str(params.get("name") or "")
        raw_args = params.get("arguments")
        args_dict = dict(raw_args) if isinstance(raw_args, dict) else {}
        self.invocations.append((tool_name, args_dict))
        return {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {"revealedPath": str(args_dict.get("path", ""))},
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 12},
        }


class _ScriptedJobGenerator:
    def __init__(self, turns: Sequence[Sequence[ChatEvent]]) -> None:
        self._turns = list(turns)
        self.requests: list[ChatRequest] = []

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        events = self._turns.pop(0) if self._turns else [TextDelta(text="Done.")]
        for ev in events:
            yield ev
        yield ChatFinished(
            provider="test",
            model="scripted",
            input_tokens=32,
            output_tokens=16,
            latency_ms=5,
        )


def test_runner_without_executor_fails_closed_rule_h1() -> None:
    """Rule H1 & H7: A runner without a real step_executor must fail closed, never fake ok."""

    async def _test() -> None:
        store = InMemoryAutonomousJobStore()
        runner = AutonomousJobRunner(job_store=store)
        job_id = uuid.uuid4()
        owner_id = uuid.uuid4()

        job = await runner.start_job(
            job_id=job_id,
            owner_id=owner_id,
            goal="Run a complete autonomous task",
            max_steps=50,
        )
        assert job.state == AutonomousJobState.RUNNING
        assert job.max_steps == 50

        for _ in range(30):
            await asyncio.sleep(0.02)
            current = await store.get(job_id)
            if current and current.is_terminal:
                break

        final = await store.get(job_id)
        assert final is not None
        assert final.state == AutonomousJobState.FAILED
        assert final.stop_reason == "executor_not_configured"
        assert final.is_terminal is True
        await runner.shutdown()

    asyncio.run(_test())


def test_orchestrator_job_step_executor_real_host_tools_and_fail_closed() -> None:
    """Verifies OrchestratorJobStepExecutor fail-closed (no_host, no_tool_executed) & real run."""

    async def _test() -> None:
        now = datetime(2026, 11, 15, 12, 0, 0, tzinfo=UTC)
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        host_repo = _JobHostRepo()
        host = Host(
            host_id=host_id,
            owner_id=owner_id,
            alias=HostAlias("Studio Mac"),
            platform=HostPlatform.MACOS,
            hostname="studio-mac",
            os_version="15.1",
            agent_version="0.2.0",
            public_key=b"\x04" + bytes(range(1, 65)),
            status=HostStatus.ACTIVE,
            created_at=now,
        )
        await host_repo.save_host(host)
        descriptor = ToolDescriptor(
            name="finder.reveal",
            description="Reveal path in Finder",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"revealedPath": {"type": "string"}},
                "additionalProperties": False,
            },
            family=ToolFamily.FILES,
            risk=RiskTier.R1,
            permission=ToolPermissionState.GRANTED,
            policy_version=1,
        )
        await host_repo.save_manifest(
            Manifest(
                host_id=host_id,
                tools=(descriptor,),
                updated_at=now,
            )
        )
        await host_repo.set_grants(
            host_id,
            [
                Grant(
                    host_id=host_id,
                    family=ToolFamily.FILES,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=now,
                )
            ],
        )

        # Case A: Offline host -> fails closed with stop_reason == "no_host"
        offline_hub = _JobSessionHub(online=False)
        audit = MemoryAuditLog()
        offline_orch = Orchestrator(
            host_repo=host_repo,
            hub=offline_hub,
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            scope_store=InMemoryApprovalScopeStore(),
        )
        gen_unused = _ScriptedJobGenerator([])
        store_a = InMemoryAutonomousJobStore()
        exec_a = OrchestratorJobStepExecutor(
            orchestrator_factory=lambda: offline_orch,
            host_repo_factory=lambda: host_repo,
            hub=offline_hub,
            audit_factory=lambda: audit,
            generator_factory=lambda: gen_unused,
            planner=JobPlannerExecutor(),
            compactor=SubGoalCompactor(),
        )
        runner_a = AutonomousJobRunner(job_store=store_a, step_executor=exec_a)
        job_a = await runner_a.start_job(
            job_id=uuid.uuid4(),
            owner_id=owner_id,
            host_id=host_id,
            goal="Inspect files then verify result",
        )
        for _ in range(25):
            await asyncio.sleep(0.02)
            cur = await store_a.get(job_a.job_id)
            if cur and cur.is_terminal:
                break
        final_a = await store_a.get(job_a.job_id)
        assert final_a is not None
        assert final_a.state == AutonomousJobState.FAILED
        assert final_a.stop_reason == "no_host"
        await runner_a.shutdown()

        # Case B: Online host, LLM claims completion without calling any tool -> "no_tool_executed"
        online_hub = _JobSessionHub(online=True)
        online_orch = Orchestrator(
            host_repo=host_repo,
            hub=online_hub,
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            scope_store=InMemoryApprovalScopeStore(),
        )
        gen_no_tools = _ScriptedJobGenerator([[TextDelta(text="I finished everything!")]])
        store_b = InMemoryAutonomousJobStore()
        exec_b = OrchestratorJobStepExecutor(
            orchestrator_factory=lambda: online_orch,
            host_repo_factory=lambda: host_repo,
            hub=online_hub,
            audit_factory=lambda: audit,
            generator_factory=lambda: gen_no_tools,
            planner=JobPlannerExecutor(),
            compactor=SubGoalCompactor(),
        )
        runner_b = AutonomousJobRunner(job_store=store_b, step_executor=exec_b)
        job_b = await runner_b.start_job(
            job_id=uuid.uuid4(),
            owner_id=owner_id,
            host_id=host_id,
            goal="Inspect files then verify result",
        )
        for _ in range(25):
            await asyncio.sleep(0.02)
            cur = await store_b.get(job_b.job_id)
            if cur and cur.is_terminal:
                break
        final_b = await store_b.get(job_b.job_id)
        assert final_b is not None
        assert final_b.state == AutonomousJobState.FAILED
        assert final_b.stop_reason == "no_tool_executed"
        assert len(online_hub.invocations) == 0
        await runner_b.shutdown()

        # Case C: Real 3-sub-goal execution calling finder.reveal + preserving thought_signature
        scripted_turns: list[list[ChatEvent]] = []
        for sg_num in range(1, 4):
            scripted_turns.append(
                [
                    ToolCallRequested(
                        call=ToolCall(
                            id=f"call_sg_{sg_num}",
                            name="finder__reveal",
                            arguments={"path": f"/Users/thao/Documents/step_{sg_num}.txt"},
                            extra={"thought_signature": f"sig_sg_{sg_num}"},
                        )
                    )
                ]
            )
            scripted_turns.append([TextDelta(text=f"Sub-goal {sg_num} verified.")])

        gen_real = _ScriptedJobGenerator(scripted_turns)
        store_c = InMemoryAutonomousJobStore()
        exec_c = OrchestratorJobStepExecutor(
            orchestrator_factory=lambda: online_orch,
            host_repo_factory=lambda: host_repo,
            hub=online_hub,
            audit_factory=lambda: audit,
            generator_factory=lambda: gen_real,
            planner=JobPlannerExecutor(),
            compactor=SubGoalCompactor(),
        )
        runner_c = AutonomousJobRunner(job_store=store_c, step_executor=exec_c)
        job_c = await runner_c.start_job(
            job_id=uuid.uuid4(),
            owner_id=owner_id,
            host_id=host_id,
            goal="Reveal project report",
            max_steps=20,
        )
        for _ in range(50):
            await asyncio.sleep(0.02)
            cur = await store_c.get(job_c.job_id)
            if cur and cur.is_terminal:
                break

        final_c = await store_c.get(job_c.job_id)
        assert final_c is not None
        assert final_c.state == AutonomousJobState.COMPLETED
        assert final_c.stop_reason == "all_sub_goals_completed"
        assert len(final_c.sub_goals) == 3
        assert all(sg.status == SubGoalStatus.COMPLETED for sg in final_c.sub_goals)
        assert len(online_hub.invocations) == 3
        assert final_c.checkpoint_data is not None
        assert len(final_c.checkpoint_data.get("compacted_history", [])) == 3
        # Verify Wave A thought_signature survived into the follow-up turn's assistant ToolCall
        second_req = gen_real.requests[1]
        assistant_msgs = [m for m in second_req.messages if m.tool_calls]
        assert assistant_msgs[0].tool_calls[0].extra == {"thought_signature": "sig_sg_1"}
        await runner_c.shutdown()

    asyncio.run(_test())


def test_runner_respects_step_budget() -> None:
    async def _test() -> None:
        store = InMemoryAutonomousJobStore()

        # Step executor that generates infinite progress
        async def infinite_steps(j: AutonomousJob) -> JobStepOutcome:
            return JobStepOutcome(
                kind="progress",
                sub_goals=(
                    SubGoal(
                        id="1",
                        title="Working",
                        description="Work",
                        status=SubGoalStatus.IN_PROGRESS,
                    ),
                ),
                checkpoint_data={"step": j.steps_used + 1},
            )

        runner = AutonomousJobRunner(job_store=store, step_executor=infinite_steps)
        job_id = uuid.uuid4()
        owner_id = uuid.uuid4()

        # Budget of only 5 steps
        await runner.start_job(
            job_id=job_id,
            owner_id=owner_id,
            goal="Long task",
            max_steps=5,
        )

        for _ in range(30):
            await asyncio.sleep(0.02)
            current = await store.get(job_id)
            if current and current.is_terminal:
                break

        final = await store.get(job_id)
        assert final is not None
        assert final.state == AutonomousJobState.FAILED
        assert final.stop_reason == "max_steps_exceeded"
        assert final.steps_used >= 5
        await runner.shutdown()

    asyncio.run(_test())


def test_runner_pause_resume_steer_cancel() -> None:
    async def _test() -> None:
        store = InMemoryAutonomousJobStore()

        async def _paced_step(j: AutonomousJob) -> JobStepOutcome:
            await asyncio.sleep(0.03)
            return JobStepOutcome(
                kind="progress",
                checkpoint_data={"step": j.steps_used + 1},
            )

        runner = AutonomousJobRunner(job_store=store, step_executor=_paced_step)
        job_id = uuid.uuid4()
        owner_id = uuid.uuid4()

        await runner.start_job(
            job_id=job_id,
            owner_id=owner_id,
            goal="Test pause and resume",
            max_steps=50,
        )

        # Pause
        paused = await runner.pause_job(job_id)
        assert paused.state == AutonomousJobState.PAUSED
        assert paused.is_active is True

        # Steer while paused
        steered = await runner.steer_job(job_id, "Please focus on table 2 instead")
        assert steered.checkpoint_data is not None
        assert "steering_notes" in steered.checkpoint_data
        instruction = steered.checkpoint_data["steering_notes"][0]["instruction"]
        assert instruction == "Please focus on table 2 instead"

        # Resume
        resumed = await runner.resume_job(job_id)
        assert resumed.state == AutonomousJobState.RUNNING

        # Cancel
        cancelled = await runner.cancel_job(job_id, reason="owner_stopped")
        assert cancelled.state == AutonomousJobState.CANCELLED
        assert cancelled.stop_reason == "owner_stopped"
        assert cancelled.is_terminal is True
        await runner.shutdown()

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 4. R3 Checkpoint Pause & Auto-Resume
# ---------------------------------------------------------------------------


def test_runner_r3_checkpoint_pause_and_auto_resume() -> None:
    async def _test() -> None:
        store = InMemoryAutonomousJobStore()
        approval_id = uuid.uuid4()
        r3_encountered = False

        async def mock_r3_step_executor(job: AutonomousJob) -> JobStepOutcome:
            nonlocal r3_encountered
            if not r3_encountered:
                r3_encountered = True
                return JobStepOutcome(
                    kind="await_r3",
                    pending_approval_id=approval_id,
                    checkpoint_data={
                        "step_count": job.steps_used + 1,
                        "active_intent": "Submit external payment",
                    },
                )
            # Post-approval resumed step
            return JobStepOutcome(
                kind="completed",
                stop_reason="payment_confirmed",
            )

        runner = AutonomousJobRunner(job_store=store, step_executor=mock_r3_step_executor)
        job_id = uuid.uuid4()
        owner_id = uuid.uuid4()

        await runner.start_job(
            job_id=job_id,
            owner_id=owner_id,
            goal="Execute financial operation",
        )

        # Wait for transition to AWAITING_R3_CHECKPOINT
        for _ in range(20):
            await asyncio.sleep(0.02)
            current = await store.get(job_id)
            if current and current.state == AutonomousJobState.AWAITING_R3_CHECKPOINT:
                break

        paused_job = await store.get(job_id)
        assert paused_job is not None
        assert paused_job.state == AutonomousJobState.AWAITING_R3_CHECKPOINT
        assert paused_job.pending_approval_id == approval_id
        assert paused_job.checkpoint_data is not None
        assert paused_job.checkpoint_data["pending_approval_id"] == str(approval_id)

        # Simulate P-256 Owner Approval Event
        resumed_job = await runner.on_approval_resolved(approval_id, approved=True)
        assert resumed_job is not None
        assert resumed_job.state == AutonomousJobState.RUNNING
        assert resumed_job.pending_approval_id is None

        # Wait for completion
        for _ in range(20):
            await asyncio.sleep(0.02)
            current = await store.get(job_id)
            if current and current.state == AutonomousJobState.COMPLETED:
                break

        final = await store.get(job_id)
        assert final is not None
        assert final.state == AutonomousJobState.COMPLETED
        assert final.stop_reason == "payment_confirmed"
        await runner.shutdown()

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 5. Consent U5 Zero-Retention: Vision Payloads Never Persist
# ---------------------------------------------------------------------------


def test_sanitize_checkpoint_data_strips_vision_base64() -> None:
    data: dict[str, Any] = {
        "step_idx": 3,
        "action": "click",
        "image_base64": "/9j/4AAQSkZJRgABAQEASABIAAD/...",
        "screenshot_base64": "some_long_image_string",
        "nested": {
            "view_url": "data:image/jpeg;base64,1234567890",
            "safe_metadata": "button_ok",
        },
        "history": [
            {"note": "saw modal", "image_bytes": "binary_content"},
            {"safe_note": "text only"},
        ],
    }

    sanitized = sanitize_checkpoint_data(data)
    assert "image_base64" not in sanitized
    assert "screenshot_base64" not in sanitized
    assert sanitized["step_idx"] == 3
    assert sanitized["nested"]["safe_metadata"] == "button_ok"
    assert "view_url" not in sanitized["nested"]
    assert "image_bytes" not in sanitized["history"][0]
    assert sanitized["history"][1]["safe_note"] == "text only"


# ---------------------------------------------------------------------------
# 6. Phone-Lock Heartbeat Resilience
# ---------------------------------------------------------------------------


def test_runner_phone_lock_heartbeat_touching() -> None:
    async def _test() -> None:
        cu_store = InMemoryComputerUseSessionStore()
        store = InMemoryAutonomousJobStore()

        async def _heartbeat_step(j: AutonomousJob) -> JobStepOutcome:
            await asyncio.sleep(0.01)
            return JobStepOutcome(
                kind="completed",
                stop_reason="all_sub_goals_completed",
            )

        runner = AutonomousJobRunner(
            job_store=store,
            cu_session_store=cu_store,
            step_executor=_heartbeat_step,
        )

        owner_id = uuid.uuid4()
        session_id = "cu-session-background-test"
        now = datetime.now(UTC)

        session_state = ComputerUseSessionState(
            session_id=session_id,
            owner_id=owner_id,
            host_id=HostId(uuid.uuid4()),
            goal="Background research during lock",
            allowed_apps=("browser",),
            allowed_domains=("example.com",),
            max_steps=50,
            max_minutes=30,
            started_at_utc=now,
            last_heartbeat_at=now - timedelta(seconds=10),
        )
        await cu_store.start(session_state, now)

        job_id = uuid.uuid4()
        await runner.start_job(
            job_id=job_id,
            owner_id=owner_id,
            session_id=session_id,
            goal="Execute tasks while phone screen is locked",
            max_steps=3,
        )

        # Let the job run a couple steps
        await asyncio.sleep(0.05)

        refreshed_session = await cu_store.get(session_id)
        assert refreshed_session is not None
        # Verify that last_heartbeat_at was refreshed by runner
        assert refreshed_session.last_heartbeat_at >= now
        await runner.shutdown()

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 7. REST API Endpoints Tests (`/v1/actions/jobs`)
# ---------------------------------------------------------------------------


ApiClientFixture = tuple[
    TestClient, LocalIssuer, Owner, InMemoryAutonomousJobStore, AutonomousJobRunner
]


@pytest.fixture
def api_client() -> ApiClientFixture:
    issuer = LocalIssuer()
    owner = Owner(
        owner_id=uuid.uuid4(),
        issuer="https://auth.test.xuna",
        subject="test-owner-1",
    )
    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
        ),
        verifier=issuer.verifier(),
    )
    app.state.database = _DummyDatabase()
    app.state.owner_repository = lambda _s: _FakeOwnerRepo(owner)
    app.state.audit_log = lambda _s: MemoryAuditLog()
    app.state.cu_session_store = InMemoryComputerUseSessionStore()

    job_store = InMemoryAutonomousJobStore()

    async def _controlled_step_executor(job: AutonomousJob) -> JobStepOutcome:
        await asyncio.sleep(0.05)
        return JobStepOutcome(
            kind="progress",
            sub_goals=(
                SubGoal(
                    id="1",
                    title="Active Work",
                    description="Working",
                    status=SubGoalStatus.IN_PROGRESS,
                ),
            ),
            checkpoint_data={"current_sub_goal_idx": 0, "step_count": job.steps_used + 1},
        )

    job_runner = AutonomousJobRunner(
        job_store=job_store,
        cu_session_store=app.state.cu_session_store,
        step_executor=_controlled_step_executor,
    )
    app.state.job_store = job_store
    app.state.job_runner = job_runner

    client = TestClient(app)
    return client, issuer, owner, job_store, job_runner


def test_rest_api_job_lifecycle(api_client: ApiClientFixture) -> None:
    client, issuer, owner, store, runner = api_client
    headers = {"Authorization": f"Bearer {issuer.token('test-owner-1')}"}

    # 1. Create Job (POST /v1/actions/jobs)
    payload = {
        "schemaVersion": 1,
        "petId": "xuna_pet",
        "title": "Automated Research Task",
        "goal": "Collect and summarize documentation",
        "maxSteps": 45,
    }
    create_resp = client.post("/v1/actions/jobs", json=payload, headers=headers)
    assert create_resp.status_code == 201, create_resp.text
    job_data = create_resp.json()
    job_id = job_data["jobId"]
    assert job_data["state"] == "running"
    assert job_data["maxSteps"] == 45
    assert job_data["title"] == "Automated Research Task"

    # 2. Get Job Details (GET /v1/actions/jobs/{job_id})
    get_resp = client.get(f"/v1/actions/jobs/{job_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["jobId"] == job_id

    # 3. List Jobs (GET /v1/actions/jobs)
    list_resp = client.get("/v1/actions/jobs", headers=headers)
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] >= 1
    assert any(j["jobId"] == job_id for j in list_resp.json()["jobs"])

    # 4. Pause Job (POST /v1/actions/jobs/{job_id}/pause)
    pause_resp = client.post(f"/v1/actions/jobs/{job_id}/pause", headers=headers)
    assert pause_resp.status_code == 200
    assert pause_resp.json()["state"] == "paused"

    # 5. Steer Job (POST /v1/actions/jobs/{job_id}/steer)
    steer_resp = client.post(
        f"/v1/actions/jobs/{job_id}/steer",
        json={"schemaVersion": 1, "instruction": "Skip optional sections"},
        headers=headers,
    )
    assert steer_resp.status_code == 200
    assert "steering_notes" in steer_resp.json()["checkpointData"]

    # 6. Resume Job (POST /v1/actions/jobs/{job_id}/resume)
    resume_resp = client.post(
        f"/v1/actions/jobs/{job_id}/resume", json={"schemaVersion": 1}, headers=headers
    )
    assert resume_resp.status_code == 200
    assert resume_resp.json()["state"] == "running"

    # 7. Cancel Job (POST /v1/actions/jobs/{job_id}/cancel)
    cancel_resp = client.post(
        f"/v1/actions/jobs/{job_id}/cancel",
        json={"schemaVersion": 1, "reason": "user_cancelled"},
        headers=headers,
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["state"] == "cancelled"
    assert cancel_resp.json()["stopReason"] == "user_cancelled"


def test_rest_api_owner_isolation(api_client: ApiClientFixture) -> None:
    client, issuer, owner, store, runner = api_client
    owner1_headers = {"Authorization": f"Bearer {issuer.token('test-owner-1')}"}
    other_headers = {"Authorization": f"Bearer {issuer.token('unauthorized-owner')}"}

    create_resp = client.post(
        "/v1/actions/jobs",
        json={"schemaVersion": 1, "goal": "Confidential goal"},
        headers=owner1_headers,
    )
    assert create_resp.status_code == 201
    job_id = create_resp.json()["jobId"]

    # Other owner cannot access job1 -> 404
    other_resp = client.get(f"/v1/actions/jobs/{job_id}", headers=other_headers)
    assert other_resp.status_code == 404
