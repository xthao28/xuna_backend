"""Unit/integration tests for WatchCoordinator, job__wait, backoff, E-Stop, and Autopilot vs R3."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from tests.helpers import MemoryAuditLog

from xuna_backend.modules.actions.application.compactor import SubGoalCompactor
from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    InMemoryComputerUseSessionStore,
)
from xuna_backend.modules.actions.application.job_runner import (
    AutonomousJobRunner,
    OrchestratorJobStepExecutor,
)
from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.application.planner_executor import JobPlannerExecutor
from xuna_backend.modules.actions.application.wake_inbox import WakeInboxService
from xuna_backend.modules.actions.application.watch_coordinator import WatchCoordinator
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    SubGoal,
)
from xuna_backend.modules.actions.domain.watch import (
    JobWatch,
    WakeInboxKind,
    WakeSuggestedAction,
    WatchCompositeSignal,
    WatchMode,
    WatchStatus,
    WatchTarget,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.job_store import InMemoryAutonomousJobStore
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.actions.infrastructure.watch_repository import (
    InMemoryJobWatchRepository,
    InMemoryWakeInboxRepository,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)
from xuna_backend.modules.hosts.application.xtp_frames import WatchEventParamsModel
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
from xuna_backend.modules.hosts.domain.twin import HostPresence, HostResources, HostTwin


class _FakeHostRepo:
    def __init__(self, host: Host, manifest: Manifest, grants: list[Grant]) -> None:
        self._host = host
        self._manifest = manifest
        self._grants = grants

    async def get_host(self, host_id: HostId) -> Host | None:
        return self._host if host_id == self._host.host_id else None

    async def save_host(self, host: Host) -> None:
        self._host = host

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        return [self._host] if self._host.owner_id == owner_id else []

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self._grants) if host_id == self._host.host_id else []

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self._grants = list(grants)

    async def suspend_r2_plus_grants_for_owner(self, owner_id: uuid.UUID, now: datetime) -> int:
        return 0

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self._manifest if host_id == self._host.host_id else None

    async def save_manifest(self, manifest: Manifest) -> None:
        self._manifest = manifest


class _FakeHub:
    def __init__(self, host_id: HostId, owner_id: uuid.UUID) -> None:
        self.host_id = host_id
        self.owner_id = owner_id
        self.online = True
        self.sent: list[tuple[str, str, dict[str, object]]] = []
        self.twin = HostTwin(host_id=str(host_id), owner_id=owner_id)
        self.rpc_handler: Callable[[str, dict[str, object]], dict[str, object]] | None = None

    def is_online(self, host_id: str) -> bool:
        return self.online and host_id == str(self.host_id)

    def get_twin(self, host_id: str) -> HostTwin | None:
        return self.twin if host_id == str(self.host_id) and self.online else None

    def get_twins_for_owner(self, owner_id: uuid.UUID) -> list[HostTwin]:
        return [self.twin] if self.online and owner_id == self.owner_id else []

    async def send_request(
        self,
        host_id: str,
        method: str,
        params: dict[str, object],
        timeout_s: float,
    ) -> dict[str, object]:
        del timeout_s
        if not self.is_online(host_id):
            raise RuntimeError("host_offline")
        self.sent.append((host_id, method, params))
        if self.rpc_handler is not None:
            return self.rpc_handler(method, params)
        if method == "xuna/watchRegister":
            signals = params.get("signals")
            active = (
                list(signals)
                if isinstance(signals, list)
                else [str(params.get("mode", "ui_quiescence"))]
            )
            return {
                "schemaVersion": 1,
                "accepted": True,
                "watchId": str(params["watchId"]),
                "activeSignals": active,
                "eventDriven": True,
                "fallbackUsed": False,
            }
        if method == "xuna/watchCancel":
            return {"schemaVersion": 1, "cancelledCount": 1}
        if method == "xuna/get_policy":
            return {"trustPreset": "autopilot"}
        return {
            "structuredContent": {"status": "ok", "observed": True},
            "_meta": {"xuna/status": "ok"},
        }

    async def notify(self, host_id: str, method: str, params: dict[str, object]) -> None:
        self.sent.append((host_id, method, params))

    async def stop_host(self, host_id: str, reason: str = "owner_app") -> bool:
        self.sent.append((host_id, "xuna/stop", {"reason": reason}))
        return True


class _ScriptedGenerator:
    def __init__(
        self,
        turns: list[tuple[str, list[ToolCall]]],
        *,
        on_request: Callable[[ChatRequest], None] | None = None,
    ) -> None:
        self._turns = list(turns)
        self._on_request = on_request
        self.calls: list[ChatRequest] = []

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.calls.append(request)
        if self._on_request is not None:
            self._on_request(request)
        if not self._turns:
            text, tool_calls = "All verified.", []
        else:
            text, tool_calls = self._turns.pop(0)
        if text:
            yield TextDelta(text=text)
        for tc in tool_calls:
            yield ToolCallRequested(call=tc)
        yield ChatFinished(
            provider="fake",
            model="fake-1",
            latency_ms=5,
            input_tokens=10,
            output_tokens=5,
        )


def _make_host_and_tools(
    owner_id: uuid.UUID, host_id: HostId
) -> tuple[Host, Manifest, list[Grant]]:
    now = datetime.now(UTC)
    host = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("Workstation"),
        platform=HostPlatform.MACOS,
        hostname="workstation",
        os_version="14.0",
        agent_version="0.2.0",
        public_key=b"\x04" + bytes(range(1, 65)),
        status=HostStatus.ACTIVE,
        created_at=now,
    )
    inspect_desc = ToolDescriptor(
        name="ui.inspect",
        description="Inspect current window.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=ToolFamily.UI,
        risk=RiskTier.R0,
        observes=True,
        reversible=True,
        permission=ToolPermissionState.GRANTED,
    )
    mail_desc = ToolDescriptor(
        name="mail.send",
        description="Send an external email.",
        input_schema={
            "type": "object",
            "properties": {"to": {"type": "string", "maxLength": 80}},
            "required": ["to"],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=ToolFamily.APP,
        risk=RiskTier.R3,
        observes=False,
        reversible=False,
        permission=ToolPermissionState.GRANTED,
    )
    manifest = Manifest(host_id=host_id, tools=(inspect_desc, mail_desc), updated_at=now)
    grants = [
        Grant(
            host_id=host_id,
            family=ToolFamily.UI,
            max_risk=RiskTier.R0,
            enabled=True,
            updated_at=now,
        ),
        Grant(
            host_id=host_id,
            family=ToolFamily.APP,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=now,
        ),
    ]
    return host, manifest, grants


def test_bind_target_reads_cu_last_window_pid_window_id_and_never_guesses() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        cu_store = InMemoryComputerUseSessionStore()
        session = ComputerUseSessionState(
            session_id="cu_1",
            owner_id=owner_id,
            host_id=host_id,
            goal="Run task",
            allowed_apps=("editor", "browser"),
            allowed_domains=(),
            max_steps=20,
            max_minutes=15,
            started_at_utc=now,
            last_heartbeat_at=now,
        )
        Orchestrator._record_cu_window_context(
            session,
            app_id="editor",
            window_title="ignored",
            pid=48210,
            window_id=1042,
        )
        await cu_store.start(session, now)

        coord = WatchCoordinator(
            watch_repo=InMemoryJobWatchRepository(),
            job_store=InMemoryAutonomousJobStore(),
            hub=hub,  # type: ignore[arg-type]
            cu_session_store=cu_store,
        )
        target, err = await coord.bind_target(
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.COMPOSITE,
            signals=(WatchCompositeSignal.UI_QUIESCENCE, WatchCompositeSignal.PROCESS_IDLE),
            session_id="cu_1",
            workspace_alias="xuna_pet",
        )
        assert err is None
        assert target is not None
        assert target.pid == 48210
        assert target.window_id == 1042
        assert target.app_id == "editor"
        assert target.workspace_alias == "xuna_pet"

        # Multiple allowed_apps without last_window or presence must fail closed (H3)
        session2 = ComputerUseSessionState(
            session_id="cu_2",
            owner_id=owner_id,
            host_id=host_id,
            goal="Ambiguous",
            allowed_apps=("app_a", "app_b"),
            allowed_domains=(),
            max_steps=10,
            max_minutes=10,
            started_at_utc=now,
            last_heartbeat_at=now,
        )
        await cu_store.start(session2, now)
        target2, err2 = await coord.bind_target(
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.UI_QUIESCENCE,
            signals=(),
            session_id="cu_2",
        )
        assert target2 is None
        assert err2 == "target_not_found"

        # Missing workspaceAlias for fs_quiescence fails closed
        target3, err3 = await coord.bind_target(
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.FS_QUIESCENCE,
            signals=(),
        )
        assert target3 is None
        assert err3 == "workspace_not_found"

        # Hard-denied app target fails closed policy_blocked
        hub.twin = replace(
            hub.twin,
            presence=HostPresence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                updated_at=now,
                foreground_app_id="terminal",
            ),
        )
        target4, err4 = await coord.bind_target(
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.UI_QUIESCENCE,
            signals=(),
        )
        assert target4 is None
        assert err4 == "policy_blocked"

    asyncio.run(_run())


def test_job_wait_no_action_backoff_and_auto_continue_r3_checkpoint() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        host, manifest, grants = _make_host_and_tools(owner_id, host_id)
        host_repo = _FakeHostRepo(host, manifest, grants)
        hub = _FakeHub(host_id, owner_id)
        hub.twin = replace(
            hub.twin,
            presence=HostPresence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                updated_at=now,
                foreground_app_id="editor",
            ),
            resources=HostResources(
                installed_apps=("editor",),
                workspaces=("xuna_pet",),
                updated_at=now,
            ),
        )
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        inbox_repo = InMemoryWakeInboxRepository()
        wake_inbox = WakeInboxService(repo=inbox_repo)
        cu_store = InMemoryComputerUseSessionStore()

        # Scripted LLM:
        # Step 1: call job__wait (quiescence=8, auto_continue)
        # Step 2 (after 1st wake): call ui__inspect (task still running -> NO_ACTION)
        # Step 3: call job__wait again -> backoff should scale 8 -> 12 silently (0 inbox items!)
        # Step 4 (after 2nd wake): call ui__inspect -> complete sub-goal 1 -> auto-continue to sg 2
        # Step 5: call mail__send (high-impact R3 -> approval_timeout -> AWAITING_R3_CHECKPOINT)
        generator = _ScriptedGenerator(
            turns=[
                (
                    "",
                    [
                        ToolCall(
                            id="c1",
                            name="job__wait",
                            arguments={
                                "mode": "composite",
                                "signals": ["ui_quiescence", "process_idle"],
                                "doneCriteria": "Build finishes",
                                "quiescenceSeconds": 8,
                                "continuationMode": "auto_continue",
                            },
                        )
                    ],
                ),
                ("", [ToolCall(id="c2", name="ui__inspect", arguments={})]),
                (
                    "",
                    [
                        ToolCall(
                            id="c3",
                            name="job__wait",
                            arguments={
                                "mode": "composite",
                                "signals": ["ui_quiescence", "process_idle"],
                                "doneCriteria": "Build finishes",
                                "quiescenceSeconds": 8,
                                "continuationMode": "auto_continue",
                            },
                        )
                    ],
                ),
                ("", [ToolCall(id="c4", name="ui__inspect", arguments={})]),
                ("Build completed.", []),
                ("", [ToolCall(id="c5", name="mail__send", arguments={"to": "a@example.com"})]),
            ]
        )

        approval_broker = InMemoryApprovalBroker()
        # Force approval timeout immediately for mail.send R3 test
        orig_wait = approval_broker.wait

        async def _fast_timeout_wait(approval_id: uuid.UUID, timeout_s: float) -> Any:
            return await orig_wait(approval_id, 0.01)

        approval_broker.wait = _fast_timeout_wait  # type: ignore[method-assign]

        def _make_orch() -> Orchestrator:
            return Orchestrator(
                host_repo=host_repo,  # type: ignore[arg-type]
                hub=hub,  # type: ignore[arg-type]
                approval_repo=InMemoryApprovalRepository(),
                approval_broker=approval_broker,
                local_result_broker=InMemoryLocalResultBroker(),
                last_target_store=InMemoryLastTargetStore(),
                audit=MemoryAuditLog(),
                scope_store=InMemoryApprovalScopeStore(),
                cu_session_store=cu_store,
            )

        compactor = SubGoalCompactor()
        planner = JobPlannerExecutor(compactor=compactor)
        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            cu_session_store=cu_store,
            wake_inbox=wake_inbox,
        )
        runner = AutonomousJobRunner(
            job_store=job_store,
            cu_session_store=cu_store,
            planner=planner,
            compactor=compactor,
            watch_coordinator=coord,
        )
        coord.set_job_runner(runner)
        executor = OrchestratorJobStepExecutor(
            orchestrator_factory=_make_orch,
            host_repo_factory=lambda: host_repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            audit_factory=MemoryAuditLog,
            generator_factory=lambda: generator,
            planner=planner,
            compactor=compactor,
            on_pending_approval=runner.mark_live_pending_approval,
            watch_coordinator=coord,
            wake_inbox=wake_inbox,
        )
        runner._step_executor = executor

        job_id = uuid.uuid4()
        await runner.start_job(
            job_id=job_id,
            owner_id=owner_id,
            host_id=host_id,
            title="Build and send",
            goal="Wait for build then send email",
            sub_goals=[
                SubGoal(id="sg1", title="Build", description="Wait for build"),
                SubGoal(id="sg2", title="Notify", description="Send email"),
            ],
        )

        # Wait for job to enter WAITING_WATCH (watch #1)
        for _ in range(50):
            cur = await job_store.get(job_id)
            if cur is not None and cur.state == AutonomousJobState.WAITING_WATCH:
                break
            await asyncio.sleep(0.01)

        w1 = await watch_repo.get_active_for_job(job_id)
        assert w1 is not None
        assert w1.quiescence_seconds == 8
        assert len(await wake_inbox.list_items(owner_id)) == 0

        # Trigger 1st watchEvent (NO_ACTION -> re-watch with backoff 12s, 0 inbox items!)
        await coord.on_watch_event(
            owner_id,
            host_id,
            WatchEventParamsModel(
                schemaVersion=1,
                watchId=str(w1.watch_id),
                jobId=str(job_id),
                kind="settled",
                observedSignals=["ui_quiescence", "process_idle"],
                elapsedSeconds=10,
                changeCount=2,
                digestChanged=True,
                occurredAt=datetime.now(UTC).isoformat(),
            ),
        )

        for _ in range(50):
            w2 = await watch_repo.get_active_for_job(job_id)
            if w2 is not None and w2.watch_id != w1.watch_id:
                break
            await asyncio.sleep(0.01)

        w2 = await watch_repo.get_active_for_job(job_id)
        assert w2 is not None
        assert w2.quiescence_seconds == 12
        assert w2.wake_checks_used == 1
        assert wake_inbox.silent_no_action_count == 1
        assert len(await wake_inbox.list_items(owner_id)) == 0

        # Trigger 2nd watchEvent -> completes sg1 -> auto_continue -> hits mail.send R3 checkpoint
        await coord.on_watch_event(
            owner_id,
            host_id,
            WatchEventParamsModel(
                schemaVersion=1,
                watchId=str(w2.watch_id),
                jobId=str(job_id),
                kind="settled",
                observedSignals=["ui_quiescence", "process_idle"],
                elapsedSeconds=25,
                changeCount=5,
                digestChanged=True,
                occurredAt=datetime.now(UTC).isoformat(),
            ),
        )

        for _ in range(100):
            final_job = await job_store.get(job_id)
            inbox_items = await wake_inbox.list_items(owner_id)
            if (
                final_job is not None
                and final_job.state == AutonomousJobState.AWAITING_R3_CHECKPOINT
                and inbox_items
            ):
                break
            await asyncio.sleep(0.01)

        final_job = await job_store.get(job_id)
        assert final_job is not None
        assert final_job.state == AutonomousJobState.AWAITING_R3_CHECKPOINT
        inbox_items = await wake_inbox.list_items(owner_id)
        assert any(
            i.kind == WakeInboxKind.JOB_NEEDS_DECISION
            and WakeSuggestedAction.OPEN_APPROVAL in i.suggested_actions
            for i in inbox_items
        )
        await runner.shutdown()

    asyncio.run(_run())


def test_continuation_modes_report_only_and_report_and_ask() -> None:
    async def _run() -> None:
        for mode_str, expected_stop_reason, expected_inbox_kind in (
            ("report_only", "report_only", WakeInboxKind.JOB_REPORT),
            ("report_and_ask", "awaiting_owner_decision", WakeInboxKind.JOB_NEEDS_DECISION),
        ):
            owner_id = uuid.uuid4()
            host_id = HostId.generate()
            now = datetime.now(UTC)
            host, manifest, grants = _make_host_and_tools(owner_id, host_id)
            host_repo = _FakeHostRepo(host, manifest, grants)
            hub = _FakeHub(host_id, owner_id)
            hub.twin = replace(
                hub.twin,
                presence=HostPresence(
                    idle_seconds=0,
                    screen_locked=False,
                    media_playing=False,
                    updated_at=now,
                    foreground_app_id="editor",
                ),
            )
            job_store = InMemoryAutonomousJobStore()
            watch_repo = InMemoryJobWatchRepository()
            wake_inbox = WakeInboxService(repo=InMemoryWakeInboxRepository())
            generator = _ScriptedGenerator(
                turns=[
                    (
                        "",
                        [
                            ToolCall(
                                id="w1",
                                name="job__wait",
                                arguments={
                                    "mode": "ui_quiescence",
                                    "doneCriteria": "Sub-goal 1 done",
                                    "continuationMode": mode_str,
                                },
                            )
                        ],
                    ),
                    ("", [ToolCall(id="i1", name="ui__inspect", arguments={})]),
                    ("Sub-goal 1 finished.", []),
                ]
            )
            compactor = SubGoalCompactor()
            planner = JobPlannerExecutor(compactor=compactor)
            coord = WatchCoordinator(
                watch_repo=watch_repo,
                job_store=job_store,
                hub=hub,  # type: ignore[arg-type]
                wake_inbox=wake_inbox,
            )
            runner = AutonomousJobRunner(
                job_store=job_store,
                planner=planner,
                compactor=compactor,
                watch_coordinator=coord,
            )
            coord.set_job_runner(runner)

            def _make_orch(hr: _FakeHostRepo = host_repo, hb: _FakeHub = hub) -> Orchestrator:
                return Orchestrator(
                    host_repo=hr,  # type: ignore[arg-type]
                    hub=hb,  # type: ignore[arg-type]
                    approval_repo=InMemoryApprovalRepository(),
                    approval_broker=InMemoryApprovalBroker(),
                    local_result_broker=InMemoryLocalResultBroker(),
                    last_target_store=InMemoryLastTargetStore(),
                    audit=MemoryAuditLog(),
                    scope_store=InMemoryApprovalScopeStore(),
                )

            runner._step_executor = OrchestratorJobStepExecutor(
                orchestrator_factory=_make_orch,
                host_repo_factory=lambda hr=host_repo: hr,  # type: ignore[arg-type, misc]
                hub=hub,  # type: ignore[arg-type]
                audit_factory=MemoryAuditLog,
                generator_factory=lambda gen=generator: gen,  # type: ignore[misc]
                planner=planner,
                compactor=compactor,
                watch_coordinator=coord,
                wake_inbox=wake_inbox,
            )

            job_id = uuid.uuid4()
            await runner.start_job(
                job_id=job_id,
                owner_id=owner_id,
                host_id=host_id,
                title=f"Mode {mode_str}",
                goal="Two phase task",
                sub_goals=[
                    SubGoal(id="sg1", title="Phase 1", description="Watch phase"),
                    SubGoal(id="sg2", title="Phase 2", description="Follow-up"),
                ],
            )
            for _ in range(50):
                cur = await job_store.get(job_id)
                if cur is not None and cur.state == AutonomousJobState.WAITING_WATCH:
                    break
                await asyncio.sleep(0.01)
            active_w = await watch_repo.get_active_for_job(job_id)
            assert active_w is not None
            await coord.on_watch_event(
                owner_id,
                host_id,
                WatchEventParamsModel(
                    schemaVersion=1,
                    watchId=str(active_w.watch_id),
                    jobId=str(job_id),
                    kind="settled",
                    observedSignals=["ui_quiescence"],
                    elapsedSeconds=9,
                    changeCount=1,
                    digestChanged=True,
                    occurredAt=datetime.now(UTC).isoformat(),
                ),
            )
            for _ in range(50):
                done_job = await job_store.get(job_id)
                if done_job is not None and done_job.state == AutonomousJobState.PAUSED:
                    break
                await asyncio.sleep(0.01)
            done_job = await job_store.get(job_id)
            assert done_job is not None
            assert done_job.state == AutonomousJobState.PAUSED
            assert done_job.stop_reason == expected_stop_reason
            items = await wake_inbox.list_items(owner_id)
            assert len(items) == 1
            assert items[0].kind == expected_inbox_kind
            await runner.shutdown()

    asyncio.run(_run())


def test_sweep_expiry_heartbeat_rehydration_fail_closed_and_estop() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        clock_box = [now]
        hub = _FakeHub(host_id, owner_id)
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        inbox_repo = InMemoryWakeInboxRepository()
        wake_inbox = WakeInboxService(repo=inbox_repo, clock_utc=lambda: clock_box[0])
        cu_store = InMemoryComputerUseSessionStore()

        cu = ComputerUseSessionState(
            session_id="cu_keep",
            owner_id=owner_id,
            host_id=host_id,
            goal="Keep alive",
            allowed_apps=("editor",),
            allowed_domains=(),
            max_steps=20,
            max_minutes=30,
            started_at_utc=now,
            last_heartbeat_at=now,
        )
        await cu_store.start(cu, now)

        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            cu_session_store=cu_store,
            wake_inbox=wake_inbox,
            clock_utc=lambda: clock_box[0],
        )

        # 1. Sweep preserves CU heartbeat while unexpired, and expires overdue watches as PAUSED
        job_ok = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id="cu_keep",
            title="Active watch",
            goal="Wait",
            state=AutonomousJobState.WAITING_WATCH,
        )
        await job_store.save(job_ok)
        w_ok = JobWatch(
            watch_id=uuid.uuid4(),
            job_id=job_ok.job_id,
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Wait 60s",
            expires_at=now + timedelta(seconds=60),
            created_at=now,
            updated_at=now,
        )
        await watch_repo.save(w_ok)

        clock_box[0] = now + timedelta(seconds=10)
        assert await coord.sweep_once() == 0
        refreshed_cu = await cu_store.get("cu_keep")
        assert refreshed_cu is not None
        assert refreshed_cu.last_heartbeat_at == clock_box[0]

        # Advance past expiry -> sweep marks watch EXPIRED, job PAUSED (NEVER COMPLETED!)
        clock_box[0] = now + timedelta(seconds=65)
        assert await coord.sweep_once() == 1
        expired_job = await job_store.get(job_ok.job_id)
        assert expired_job is not None
        assert expired_job.state == AutonomousJobState.PAUSED
        assert expired_job.stop_reason == "watch_timeout"
        items = await wake_inbox.list_items(owner_id)
        assert items[0].kind == WakeInboxKind.JOB_TIMEOUT
        assert WakeSuggestedAction.EXTEND_WATCH in items[0].suggested_actions

        # 2. Rehydration on host_ready: unexpired watch re-registers; rejected watch fails closed
        job_rehyd = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Rehydrate",
            goal="Wait across disconnect",
            state=AutonomousJobState.WAITING_WATCH,
        )
        await job_store.save(job_rehyd)
        w_rehyd = JobWatch(
            watch_id=uuid.uuid4(),
            job_id=job_rehyd.job_id,
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Resume after reconnect",
            expires_at=clock_box[0] + timedelta(seconds=120),
            created_at=clock_box[0],
            updated_at=clock_box[0],
        )
        await watch_repo.save(w_rehyd)
        hub.sent.clear()
        rehydrated = await coord.on_host_ready(owner_id, host_id)
        assert rehydrated == 1
        assert any(m == "xuna/watchRegister" for _, m, _ in hub.sent)

        # target_lost fails closed -> FAILED (NEVER COMPLETED)
        await coord.on_watch_event(
            owner_id,
            host_id,
            WatchEventParamsModel(
                schemaVersion=1,
                watchId=str(w_rehyd.watch_id),
                jobId=str(job_rehyd.job_id),
                kind="target_lost",
                observedSignals=[],
                elapsedSeconds=12,
                changeCount=0,
                digestChanged=False,
                occurredAt=clock_box[0].isoformat(),
            ),
        )
        lost_job = await job_store.get(job_rehyd.job_id)
        assert lost_job is not None
        assert lost_job.state == AutonomousJobState.FAILED
        assert lost_job.stop_reason == "target_lost"

        # 3. E-Stop cancels active watches + sends xuna/watchCancel {reason: "estop"}
        job_estop = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Stoppable",
            goal="Wait",
            state=AutonomousJobState.WAITING_WATCH,
        )
        await job_store.save(job_estop)
        w_estop = JobWatch(
            watch_id=uuid.uuid4(),
            job_id=job_estop.job_id,
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Wait",
            expires_at=clock_box[0] + timedelta(seconds=120),
            created_at=clock_box[0],
            updated_at=clock_box[0],
        )
        await watch_repo.save(w_estop)
        runner = AutonomousJobRunner(
            job_store=job_store,
            watch_coordinator=coord,
        )
        hub.sent.clear()
        cancelled_jobs = await runner.cancel_all_for_owner(owner_id, reason="owner_app")
        assert cancelled_jobs >= 1
        w_estop_after = await watch_repo.get(w_estop.watch_id)
        assert w_estop_after is not None
        assert w_estop_after.status == WatchStatus.CANCELLED
        assert any(
            m == "xuna/watchCancel" and p == {"schemaVersion": 1, "reason": "estop"}
            for _, m, p in hub.sent
        )

    asyncio.run(_run())


def test_max_wake_checks_exhaustion_fails_closed_consistent_with_wake_inbox() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        host, manifest, grants = _make_host_and_tools(owner_id, host_id)
        host_repo = _FakeHostRepo(host, manifest, grants)
        hub = _FakeHub(host_id, owner_id)
        hub.twin = replace(
            hub.twin,
            presence=HostPresence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                updated_at=now,
                foreground_app_id="editor",
            ),
        )
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        wake_inbox = WakeInboxService(repo=InMemoryWakeInboxRepository())
        cu_store = InMemoryComputerUseSessionStore()

        generator = _ScriptedGenerator(
            turns=[
                (
                    "",
                    [
                        ToolCall(
                            id="w1",
                            name="job__wait",
                            arguments={
                                "mode": "ui_quiescence",
                                "doneCriteria": "Build settles",
                                "continuationMode": "auto_continue",
                            },
                        )
                    ],
                ),
                ("", [ToolCall(id="i1", name="ui__inspect", arguments={})]),
                (
                    "",
                    [
                        ToolCall(
                            id="w2",
                            name="job__wait",
                            arguments={
                                "mode": "ui_quiescence",
                                "doneCriteria": "Build settles",
                                "continuationMode": "auto_continue",
                            },
                        )
                    ],
                ),
            ]
        )

        def _make_orch() -> Orchestrator:
            return Orchestrator(
                host_repo=host_repo,  # type: ignore[arg-type]
                hub=hub,  # type: ignore[arg-type]
                approval_repo=InMemoryApprovalRepository(),
                approval_broker=InMemoryApprovalBroker(),
                local_result_broker=InMemoryLocalResultBroker(),
                last_target_store=InMemoryLastTargetStore(),
                audit=MemoryAuditLog(),
                scope_store=InMemoryApprovalScopeStore(),
                cu_session_store=cu_store,
            )

        compactor = SubGoalCompactor()
        planner = JobPlannerExecutor(compactor=compactor)
        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            cu_session_store=cu_store,
            wake_inbox=wake_inbox,
        )
        runner = AutonomousJobRunner(
            job_store=job_store,
            cu_session_store=cu_store,
            planner=planner,
            compactor=compactor,
            watch_coordinator=coord,
        )
        coord.set_job_runner(runner)
        runner._step_executor = OrchestratorJobStepExecutor(
            orchestrator_factory=_make_orch,
            host_repo_factory=lambda: host_repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            audit_factory=MemoryAuditLog,
            generator_factory=lambda: generator,
            planner=planner,
            compactor=compactor,
            watch_coordinator=coord,
            wake_inbox=wake_inbox,
        )

        job_id = uuid.uuid4()
        sg = SubGoal(id="sg1", title="Build", description="Wait for build")
        pre_job = AutonomousJob(
            job_id=job_id,
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Check limit test",
            goal="Wait for build",
            state=AutonomousJobState.PAUSED,
            sub_goals=(sg,),
            checkpoint_data={
                "current_sub_goal_idx": 0,
                "step_count": 0,
                "sub_goals": [sg.to_dict()],
                "compacted_history": [],
                "active_turns": [],
                "max_wake_checks": 1,
                "wake_checks_used": 0,
            },
            created_at=now,
            updated_at=now,
        )
        await job_store.save(pre_job)
        await runner.start_job(
            job_id=job_id,
            owner_id=owner_id,
            host_id=host_id,
            title="Check limit test",
            goal="Wait for build",
            sub_goals=[sg],
        )

        for _ in range(50):
            cur = await job_store.get(job_id)
            if cur is not None and cur.state == AutonomousJobState.WAITING_WATCH:
                break
            await asyncio.sleep(0.01)

        w1 = await watch_repo.get_active_for_job(job_id)
        assert w1 is not None
        assert w1.max_wake_checks == 1

        await coord.on_watch_event(
            owner_id,
            host_id,
            WatchEventParamsModel(
                schemaVersion=1,
                watchId=str(w1.watch_id),
                jobId=str(job_id),
                kind="settled",
                observedSignals=["ui_quiescence"],
                elapsedSeconds=10,
                changeCount=1,
                digestChanged=True,
                occurredAt=datetime.now(UTC).isoformat(),
            ),
        )

        for _ in range(100):
            final_job = await job_store.get(job_id)
            if final_job is not None and final_job.state == AutonomousJobState.FAILED:
                break
            await asyncio.sleep(0.01)

        final_job = await job_store.get(job_id)
        assert final_job is not None
        assert final_job.state == AutonomousJobState.FAILED
        assert final_job.stop_reason == "max_wake_checks_exceeded"
        items = await wake_inbox.list_items(owner_id)
        assert len(items) == 1
        assert items[0].kind == WakeInboxKind.JOB_FAILED
        assert items[0].suggested_actions == (WakeSuggestedAction.DISMISS,)
        assert WakeSuggestedAction.EXTEND_WATCH not in items[0].suggested_actions
        assert "Paused" not in items[0].summary
        await runner.shutdown()

    asyncio.run(_run())


def test_on_watch_event_and_expire_at_max_wake_checks_omit_extend_watch() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        clock_box = [now]
        hub = _FakeHub(host_id, owner_id)
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        wake_inbox = WakeInboxService(
            repo=InMemoryWakeInboxRepository(), clock_utc=lambda: clock_box[0]
        )
        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            wake_inbox=wake_inbox,
            clock_utc=lambda: clock_box[0],
        )

        # (a) on_watch_event when wake_checks_used == max_wake_checks -> FAILED + JOB_FAILED
        job_a = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Event at limit",
            goal="Wait",
            state=AutonomousJobState.WAITING_WATCH,
        )
        await job_store.save(job_a)
        watch_a = JobWatch(
            watch_id=uuid.uuid4(),
            job_id=job_a.job_id,
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Wait",
            max_wake_checks=2,
            wake_checks_used=2,
            status=WatchStatus.ACTIVE,
            expires_at=now + timedelta(seconds=60),
            created_at=now,
            updated_at=now,
        )
        await watch_repo.save(watch_a)
        await coord.on_watch_event(
            owner_id,
            host_id,
            WatchEventParamsModel(
                schemaVersion=1,
                watchId=str(watch_a.watch_id),
                jobId=str(job_a.job_id),
                kind="settled",
                observedSignals=["timer"],
                elapsedSeconds=10,
                changeCount=1,
                digestChanged=False,
                occurredAt=now.isoformat(),
            ),
        )
        saved_watch_a = await watch_repo.get(watch_a.watch_id)
        saved_job_a = await job_store.get(job_a.job_id)
        assert saved_watch_a is not None and saved_watch_a.status == WatchStatus.FAILED
        assert saved_watch_a.last_event_kind == "max_wake_checks_exceeded"
        assert saved_job_a is not None and saved_job_a.state == AutonomousJobState.FAILED
        assert saved_job_a.stop_reason == "max_wake_checks_exceeded"
        items_a = await wake_inbox.list_items(owner_id)
        assert len(items_a) == 1
        assert items_a[0].kind == WakeInboxKind.JOB_FAILED
        assert items_a[0].suggested_actions == (WakeSuggestedAction.DISMISS,)

        # (b) sweep_once expiry when wake_checks_used == max_wake_checks omits EXTEND_WATCH
        owner_b = uuid.uuid4()
        job_b = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner_b,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Expire at limit",
            goal="Wait",
            state=AutonomousJobState.WAITING_WATCH,
        )
        await job_store.save(job_b)
        watch_b = JobWatch(
            watch_id=uuid.uuid4(),
            job_id=job_b.job_id,
            owner_id=owner_b,
            host_id=host_id,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Wait",
            max_wake_checks=2,
            wake_checks_used=2,
            status=WatchStatus.ACTIVE,
            expires_at=now + timedelta(seconds=30),
            created_at=now,
            updated_at=now,
        )
        await watch_repo.save(watch_b)
        clock_box[0] = now + timedelta(seconds=45)
        assert await coord.sweep_once() == 1
        items_b = await wake_inbox.list_items(owner_b)
        assert len(items_b) == 1
        assert WakeSuggestedAction.EXTEND_WATCH not in items_b[0].suggested_actions
        assert items_b[0].suggested_actions == (WakeSuggestedAction.DISMISS,)

    asyncio.run(_run())


def test_record_cu_window_context_clears_stale_pid_and_window_id_on_app_switch() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        cu_store = InMemoryComputerUseSessionStore()
        session = ComputerUseSessionState(
            session_id="cu_switch",
            owner_id=owner_id,
            host_id=host_id,
            goal="Switch apps",
            allowed_apps=("editor", "browser"),
            allowed_domains=(),
            max_steps=20,
            max_minutes=15,
            started_at_utc=now,
            last_heartbeat_at=now,
        )
        Orchestrator._record_cu_window_context(
            session,
            app_id="editor",
            window_title="a.py",
            pid=111,
            window_id=10,
            clear_title_if_app_changed=True,
        )
        assert session.last_window.get("pid") == "111"
        assert session.last_window.get("windowId") == "10"

        Orchestrator._record_cu_window_context(
            session,
            app_id="browser",
            window_title="Docs",
            clear_title_if_app_changed=True,
        )
        assert session.last_app_id == "browser"
        assert session.last_window_title == "Docs"
        assert "pid" not in session.last_window
        assert "windowId" not in session.last_window

        await cu_store.start(session, now)
        coord = WatchCoordinator(
            watch_repo=InMemoryJobWatchRepository(),
            job_store=InMemoryAutonomousJobStore(),
            hub=hub,  # type: ignore[arg-type]
            cu_session_store=cu_store,
        )
        target, err = await coord.bind_target(
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.UI_QUIESCENCE,
            signals=(),
            session_id="cu_switch",
        )
        assert err is None
        assert target is not None
        assert target.app_id == "browser"
        assert target.pid is None
        assert target.window_id is None

        # Switching back with pid only pops windowId
        Orchestrator._record_cu_window_context(
            session,
            app_id="browser",
            window_id=20,
        )
        assert session.last_window.get("windowId") == "20"
        Orchestrator._record_cu_window_context(
            session,
            app_id="editor",
            pid=222,
            clear_title_if_app_changed=True,
        )
        assert session.last_window.get("pid") == "222"
        assert "windowId" not in session.last_window

    asyncio.run(_run())


def test_pause_job_cancels_active_watch_and_notifies_host() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        host, manifest, grants = _make_host_and_tools(owner_id, host_id)
        host_repo = _FakeHostRepo(host, manifest, grants)
        hub = _FakeHub(host_id, owner_id)
        hub.twin = replace(
            hub.twin,
            presence=HostPresence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                updated_at=now,
                foreground_app_id="editor",
            ),
        )
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        wake_inbox = WakeInboxService(repo=InMemoryWakeInboxRepository())
        generator = _ScriptedGenerator(
            turns=[
                (
                    "",
                    [
                        ToolCall(
                            id="w1",
                            name="job__wait",
                            arguments={
                                "mode": "ui_quiescence",
                                "doneCriteria": "Build settles",
                                "continuationMode": "auto_continue",
                            },
                        )
                    ],
                )
            ]
        )

        def _make_orch() -> Orchestrator:
            return Orchestrator(
                host_repo=host_repo,  # type: ignore[arg-type]
                hub=hub,  # type: ignore[arg-type]
                approval_repo=InMemoryApprovalRepository(),
                approval_broker=InMemoryApprovalBroker(),
                local_result_broker=InMemoryLocalResultBroker(),
                last_target_store=InMemoryLastTargetStore(),
                audit=MemoryAuditLog(),
                scope_store=InMemoryApprovalScopeStore(),
            )

        compactor = SubGoalCompactor()
        planner = JobPlannerExecutor(compactor=compactor)
        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            wake_inbox=wake_inbox,
        )
        runner = AutonomousJobRunner(
            job_store=job_store,
            planner=planner,
            compactor=compactor,
            watch_coordinator=coord,
        )
        coord.set_job_runner(runner)
        runner._step_executor = OrchestratorJobStepExecutor(
            orchestrator_factory=_make_orch,
            host_repo_factory=lambda: host_repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            audit_factory=MemoryAuditLog,
            generator_factory=lambda: generator,
            planner=planner,
            compactor=compactor,
            watch_coordinator=coord,
            wake_inbox=wake_inbox,
        )

        job_id = uuid.uuid4()
        await runner.start_job(
            job_id=job_id,
            owner_id=owner_id,
            host_id=host_id,
            title="Pause watch job",
            goal="Wait and pause",
            sub_goals=[SubGoal(id="sg1", title="Wait", description="Wait for UI")],
        )
        for _ in range(50):
            cur = await job_store.get(job_id)
            if cur is not None and cur.state == AutonomousJobState.WAITING_WATCH:
                break
            await asyncio.sleep(0.01)

        w1 = await watch_repo.get_active_for_job(job_id)
        assert w1 is not None
        hub.sent.clear()

        paused = await runner.pause_job(job_id)
        assert paused.state == AutonomousJobState.PAUSED
        cancelled_watch = await watch_repo.get(w1.watch_id)
        assert cancelled_watch is not None
        assert cancelled_watch.status == WatchStatus.CANCELLED
        assert any(
            m == "xuna/watchCancel"
            and p.get("reason") == "owner_cancel"
            and p.get("watchId") == str(w1.watch_id)
            for _, m, p in hub.sent
        )
        await runner.shutdown()

    asyncio.run(_run())


def test_concurrent_estop_vs_on_watch_event_never_resurrects_cancelled_job() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        wake_inbox = WakeInboxService(repo=InMemoryWakeInboxRepository())

        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            wake_inbox=wake_inbox,
        )
        runner = AutonomousJobRunner(
            job_store=job_store,
            watch_coordinator=coord,
            wake_inbox=wake_inbox,
        )
        coord.set_job_runner(runner)

        job_id = uuid.uuid4()
        watch_id = uuid.uuid4()
        job = AutonomousJob(
            job_id=job_id,
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="E-Stop Race Job",
            goal="Wait safely",
            state=AutonomousJobState.WAITING_WATCH,
            created_at=now,
            updated_at=now,
        )
        watch = JobWatch(
            watch_id=watch_id,
            job_id=job_id,
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Wait",
            expires_at=now + timedelta(seconds=120),
            created_at=now,
            updated_at=now,
        )
        await job_store.save(job)
        await watch_repo.save(watch)

        # Force `job_store.get` to yield cleanly mid-`on_watch_event` so E-Stop interleaves
        orig_get = job_store.get
        entered_get = asyncio.Event()

        async def _yielding_get(jid: uuid.UUID) -> AutonomousJob | None:
            entered_get.set()
            await asyncio.sleep(0.02)
            return await orig_get(jid)

        job_store.get = _yielding_get  # type: ignore[method-assign]

        event_task = asyncio.create_task(
            coord.on_watch_event(
                owner_id,
                host_id,
                WatchEventParamsModel(
                    schemaVersion=1,
                    watchId=str(watch_id),
                    jobId=str(job_id),
                    kind="settled",
                    observedSignals=["timer"],
                    elapsedSeconds=5,
                    changeCount=1,
                    digestChanged=False,
                    occurredAt=now.isoformat(),
                ),
            )
        )
        await entered_get.wait()
        await runner.cancel_all_for_owner(owner_id, reason="estop")
        await event_task
        await asyncio.sleep(0.02)

        final_job = await orig_get(job_id)
        final_watch = await watch_repo.get(watch_id)
        assert final_job is not None
        assert final_job.state == AutonomousJobState.CANCELLED
        assert final_watch is not None
        assert final_watch.status == WatchStatus.CANCELLED
        await runner.shutdown()

    asyncio.run(_run())


def test_concurrent_estop_vs_register_from_job_wait_never_overwrites_cancelled() -> None:
    async def _run(host_accepts: bool) -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        wake_inbox = WakeInboxService(repo=InMemoryWakeInboxRepository())

        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            wake_inbox=wake_inbox,
        )
        runner = AutonomousJobRunner(
            job_store=job_store,
            watch_coordinator=coord,
            wake_inbox=wake_inbox,
        )
        coord.set_job_runner(runner)

        job_id = uuid.uuid4()
        sg = SubGoal(id="sg1", title="Wait", description="Wait for host")
        job = AutonomousJob(
            job_id=job_id,
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Register Race Job",
            goal="Wait safely",
            state=AutonomousJobState.RUNNING,
            sub_goals=(sg,),
            created_at=now,
            updated_at=now,
        )
        await job_store.save(job)

        register_started = asyncio.Event()
        orig_send = hub.send_request

        async def _slow_send_request(
            hid: str, method: str, params: dict[str, object], timeout_s: float
        ) -> dict[str, object]:
            if method == "xuna/watchRegister":
                register_started.set()
                await asyncio.sleep(0.03)
                hub.sent.append((hid, method, params))
                if host_accepts:
                    return {
                        "schemaVersion": 1,
                        "accepted": True,
                        "watchId": str(params["watchId"]),
                        "activeSignals": ["timer"],
                        "eventDriven": False,
                        "fallbackUsed": False,
                    }
                return {
                    "schemaVersion": 1,
                    "accepted": False,
                    "watchId": str(params["watchId"]),
                    "errorCode": "stopped",
                }
            return await orig_send(hid, method, params, timeout_s)

        hub.send_request = _slow_send_request  # type: ignore[method-assign]

        reg_task = asyncio.create_task(
            coord.register_from_job_wait(
                job=job,
                clean_args={
                    "mode": "timer",
                    "doneCriteria": "Finish",
                    "quiescenceSeconds": 4,
                    "maxWaitSeconds": 60,
                },
                sub_goals=(sg,),
                active_sub_goal_index=0,
                active_turns=[],
                compacted_history=[],
                assistant_call=ToolCall(id="c_reg", name="job__wait", arguments={}),
                assistant_text="",
                steering_notes=[],
                now=now,
            )
        )
        await register_started.wait()
        await runner.cancel_all_for_owner(owner_id, reason="estop")
        outcome = await reg_task

        assert outcome.accepted is False
        assert outcome.error_code == "cancelled"
        final_job = await job_store.get(job_id)
        final_watch = await watch_repo.get_latest_for_job(job_id)
        assert final_job is not None
        assert final_job.state == AutonomousJobState.CANCELLED
        assert final_watch is not None
        assert final_watch.status == WatchStatus.CANCELLED
        if host_accepts:
            assert any(
                m == "xuna/watchCancel"
                and p.get("reason") == "estop"
                and p.get("watchId") == str(final_watch.watch_id)
                for _, m, p in hub.sent
            )
        await runner.shutdown()

    asyncio.run(_run(host_accepts=False))
    asyncio.run(_run(host_accepts=True))


def test_on_host_ready_does_not_rehydrate_paused_job_and_cancels_stale_watch() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
        )

        paused_job = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Paused Job",
            goal="Paused before reconnect",
            state=AutonomousJobState.PAUSED,
            stop_reason="watch_timeout",
            created_at=now,
            updated_at=now,
        )
        stale_watch = JobWatch(
            watch_id=uuid.uuid4(),
            job_id=paused_job.job_id,
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Wait",
            status=WatchStatus.ACTIVE,
            expires_at=now + timedelta(seconds=120),
            created_at=now,
            updated_at=now,
        )
        await job_store.save(paused_job)
        await watch_repo.save(stale_watch)

        rehydrated = await coord.on_host_ready(owner_id, host_id)
        assert rehydrated == 0
        assert not any(m == "xuna/watchRegister" for _, m, _ in hub.sent)
        updated_watch = await watch_repo.get(stale_watch.watch_id)
        assert updated_watch is not None
        assert updated_watch.status == WatchStatus.CANCELLED
        updated_job = await job_store.get(paused_job.job_id)
        assert updated_job is not None
        assert updated_job.state == AutonomousJobState.PAUSED

    asyncio.run(_run())


def test_probe_1a_on_watch_event_vs_estop_during_resume_job() -> None:
    """Reproduces Challenger 2 Probe 1A: E-Stop during resume_job._store.get + watchCancel yield."""

    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        wake_inbox = WakeInboxService(repo=InMemoryWakeInboxRepository())

        async def _completing_step(j: AutonomousJob) -> Any:
            from xuna_backend.modules.actions.domain.job import JobStepOutcome

            return JobStepOutcome(kind="completed", stop_reason="completed")

        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            wake_inbox=wake_inbox,
        )
        runner = AutonomousJobRunner(
            job_store=job_store,
            step_executor=_completing_step,
            watch_coordinator=coord,
            wake_inbox=wake_inbox,
        )
        coord.set_job_runner(runner)

        job_id = uuid.uuid4()
        watch_id = uuid.uuid4()
        job = AutonomousJob(
            job_id=job_id,
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Probe 1A Job",
            goal="Wait safely",
            state=AutonomousJobState.WAITING_WATCH,
            created_at=now,
            updated_at=now,
        )
        watch = JobWatch(
            watch_id=watch_id,
            job_id=job_id,
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Wait",
            expires_at=now + timedelta(seconds=120),
            created_at=now,
            updated_at=now,
        )
        await job_store.save(job)
        await watch_repo.save(watch)

        # Yield inside resume_job's store.get (triggered after watch is marked TRIGGERED)
        orig_get = job_store.get
        resume_get_entered = asyncio.Event()

        async def _yielding_resume_get(jid: uuid.UUID) -> AutonomousJob | None:
            w_now = await watch_repo.get(watch_id)
            if (
                w_now is not None
                and w_now.status == WatchStatus.TRIGGERED
                and not resume_get_entered.is_set()
            ):
                resume_get_entered.set()
                await asyncio.sleep(0.03)
            return await orig_get(jid)

        job_store.get = _yielding_resume_get  # type: ignore[method-assign]

        # Also yield inside hub.send_request("xuna/watchCancel")
        orig_send = hub.send_request

        async def _slow_cancel_send(
            hid: str, method: str, params: dict[str, object], timeout_s: float
        ) -> dict[str, object]:
            if method == "xuna/watchCancel":
                await asyncio.sleep(0.03)
            return await orig_send(hid, method, params, timeout_s)

        hub.send_request = _slow_cancel_send  # type: ignore[method-assign]

        event_task = asyncio.create_task(
            coord.on_watch_event(
                owner_id,
                host_id,
                WatchEventParamsModel(
                    schemaVersion=1,
                    watchId=str(watch_id),
                    jobId=str(job_id),
                    kind="settled",
                    observedSignals=["timer"],
                    elapsedSeconds=5,
                    changeCount=1,
                    digestChanged=False,
                    occurredAt=now.isoformat(),
                ),
            )
        )
        await resume_get_entered.wait()
        estop_count = await runner.cancel_all_for_owner(owner_id, reason="estop")
        await event_task
        await asyncio.sleep(0.03)

        final_job = await orig_get(job_id)
        assert final_job is not None
        assert estop_count >= 1
        assert final_job.state == AutonomousJobState.CANCELLED
        await runner.shutdown()

    asyncio.run(_run())


def test_probe_1e_on_watch_event_error_and_timeout_vs_estop_never_overwrite_cancelled() -> None:
    """Reproduces Challenger 2 Probe 1E for error/target_lost and timeout/sweep_once vs E-Stop."""

    async def _run_error_case() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        wake_inbox = WakeInboxService(repo=InMemoryWakeInboxRepository())

        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            wake_inbox=wake_inbox,
        )
        runner = AutonomousJobRunner(
            job_store=job_store,
            watch_coordinator=coord,
            wake_inbox=wake_inbox,
        )
        coord.set_job_runner(runner)

        job_id = uuid.uuid4()
        watch_id = uuid.uuid4()
        job = AutonomousJob(
            job_id=job_id,
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Probe 1E Error Job",
            goal="Wait safely",
            state=AutonomousJobState.WAITING_WATCH,
            created_at=now,
            updated_at=now,
        )
        watch = JobWatch(
            watch_id=watch_id,
            job_id=job_id,
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Wait",
            expires_at=now + timedelta(seconds=120),
            created_at=now,
            updated_at=now,
        )
        await job_store.save(job)
        await watch_repo.save(watch)

        orig_watch_save = watch_repo.save
        entered_save = asyncio.Event()

        async def _yielding_watch_save(w: JobWatch) -> None:
            if w.status == WatchStatus.FAILED and not entered_save.is_set():
                entered_save.set()
                await asyncio.sleep(0.03)
            await orig_watch_save(w)

        watch_repo.save = _yielding_watch_save  # type: ignore[method-assign]

        event_task = asyncio.create_task(
            coord.on_watch_event(
                owner_id,
                host_id,
                WatchEventParamsModel(
                    schemaVersion=1,
                    watchId=str(watch_id),
                    jobId=str(job_id),
                    kind="error",
                    observedSignals=["timer"],
                    elapsedSeconds=2,
                    changeCount=0,
                    digestChanged=False,
                    occurredAt=now.isoformat(),
                    errorCode="target_crashed",
                ),
            )
        )
        await entered_save.wait()
        estop_task = asyncio.create_task(runner.cancel_all_for_owner(owner_id, reason="estop"))
        await asyncio.gather(event_task, estop_task)

        final_job = await job_store.get(job_id)
        assert final_job is not None
        assert final_job.state == AutonomousJobState.CANCELLED
        await runner.shutdown()

    async def _run_timeout_sweep_case() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        wake_inbox = WakeInboxService(repo=InMemoryWakeInboxRepository())

        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            wake_inbox=wake_inbox,
            clock_utc=lambda: now + timedelta(seconds=200),
        )
        runner = AutonomousJobRunner(
            job_store=job_store,
            watch_coordinator=coord,
            wake_inbox=wake_inbox,
        )
        coord.set_job_runner(runner)

        job_id = uuid.uuid4()
        watch_id = uuid.uuid4()
        job = AutonomousJob(
            job_id=job_id,
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Probe 1E Timeout Job",
            goal="Wait safely",
            state=AutonomousJobState.WAITING_WATCH,
            created_at=now,
            updated_at=now,
        )
        watch = JobWatch(
            watch_id=watch_id,
            job_id=job_id,
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Wait",
            expires_at=now + timedelta(seconds=60),
            created_at=now,
            updated_at=now,
        )
        await job_store.save(job)
        await watch_repo.save(watch)

        orig_watch_save = watch_repo.save
        entered_expire_save = asyncio.Event()

        async def _yielding_expire_save(w: JobWatch) -> None:
            if w.status == WatchStatus.EXPIRED and not entered_expire_save.is_set():
                entered_expire_save.set()
                await asyncio.sleep(0.02)
            await orig_watch_save(w)

        watch_repo.save = _yielding_expire_save  # type: ignore[method-assign]

        cancel_rpc_started = asyncio.Event()
        orig_send = hub.send_request

        async def _slow_expired_cancel(
            hid: str, method: str, params: dict[str, object], timeout_s: float
        ) -> dict[str, object]:
            if method == "xuna/watchCancel" and params.get("reason") == "expired":
                cancel_rpc_started.set()
                await asyncio.sleep(0.05)
            return await orig_send(hid, method, params, timeout_s)

        hub.send_request = _slow_expired_cancel  # type: ignore[method-assign]

        sweep_task = asyncio.create_task(coord.sweep_once())
        await entered_expire_save.wait()
        # Run E-Stop while sweep_once is yielding across watch_repo.save and/or watchCancel RPC
        estop_task = asyncio.create_task(runner.cancel_all_for_owner(owner_id, reason="estop"))
        await asyncio.gather(sweep_task, estop_task)

        final_job = await job_store.get(job_id)
        assert final_job is not None
        assert final_job.state == AutonomousJobState.CANCELLED
        assert cancel_rpc_started.is_set()
        await runner.shutdown()

    asyncio.run(_run_error_case())
    asyncio.run(_run_timeout_sweep_case())


def test_shared_per_job_lock_and_terminal_state_monotonicity() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()

        # Both linkage orders must share the exact same lock instance per job_id
        runner_first = AutonomousJobRunner(job_store=job_store)
        coord_second = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            job_runner=runner_first,
        )
        runner_first.set_watch_coordinator(coord_second)
        jid = uuid.uuid4()
        assert runner_first._lock_for_job(jid) is coord_second._lock_for_job(jid)

        # Terminal-state monotonicity on InMemoryAutonomousJobStore
        cancelled = AutonomousJob(
            job_id=jid,
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Monotonic Job",
            goal="Stay cancelled",
            state=AutonomousJobState.CANCELLED,
            created_at=now,
            updated_at=now,
        )
        await job_store.save(cancelled)
        await job_store.save(replace(cancelled, state=AutonomousJobState.RUNNING))
        await job_store.save(replace(cancelled, state=AutonomousJobState.FAILED))
        await job_store.save(replace(cancelled, state=AutonomousJobState.COMPLETED))
        stored = await job_store.get(jid)
        assert stored is not None
        assert stored.state == AutonomousJobState.CANCELLED

    asyncio.run(_run())
