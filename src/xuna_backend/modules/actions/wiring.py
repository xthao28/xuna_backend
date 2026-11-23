"""Composition root wiring for the `actions` module (UDC-4 §5.2, UDC-7, UDC-8)."""

from fastapi import FastAPI

from xuna_backend.modules.actions.api.job_routes import router as job_router
from xuna_backend.modules.actions.api.mesh_routes import router as mesh_router
from xuna_backend.modules.actions.api.routes import (
    build_short_session_audit_log,
    build_short_session_host_repository,
    build_stream_orchestrator,
)
from xuna_backend.modules.actions.api.routes import router as actions_router
from xuna_backend.modules.actions.api.watch_routes import router as watch_router
from xuna_backend.modules.actions.application.compactor import SubGoalCompactor
from xuna_backend.modules.actions.application.cross_host_pipelines import (
    handle_host_ready_wake_queue,
)
from xuna_backend.modules.actions.application.job_runner import (
    AutonomousJobRunner,
    OrchestratorJobStepExecutor,
)
from xuna_backend.modules.actions.application.mesh_prober import MeshProber
from xuna_backend.modules.actions.application.mesh_router import InMemoryPendingIntentStore
from xuna_backend.modules.actions.application.planner_executor import JobPlannerExecutor
from xuna_backend.modules.actions.application.wake_inbox import WakeInboxService
from xuna_backend.modules.actions.application.watch_coordinator import WatchCoordinator
from xuna_backend.modules.actions.application.yield_detach import YieldDetachService
from xuna_backend.modules.actions.domain.wake_queue import InMemoryWakeQueueStore
from xuna_backend.modules.actions.infrastructure.approval_scope_store import (
    SqlApprovalScopeStore,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.cu_session_store import (
    SqlComputerUseSessionStore,
)
from xuna_backend.modules.actions.infrastructure.habit_repository import SqlHabitRepository
from xuna_backend.modules.actions.infrastructure.job_store import SqlAutonomousJobStore
from xuna_backend.modules.actions.infrastructure.repository import SqlApprovalRepository
from xuna_backend.modules.actions.infrastructure.standing_repository import (
    SqlStandingApprovalRepository,
)
from xuna_backend.modules.actions.infrastructure.watch_repository import (
    SqlJobWatchRepository,
    SqlWakeInboxRepository,
)


def wire_actions(app: FastAPI) -> None:
    app.state.approval_repository = SqlApprovalRepository
    app.state.standing_approval_repository = SqlStandingApprovalRepository
    app.state.habit_repository = SqlHabitRepository
    app.state.approval_broker = InMemoryApprovalBroker()
    app.state.local_result_broker = InMemoryLocalResultBroker()
    app.state.last_target_store = InMemoryLastTargetStore()
    if not hasattr(app.state, "pending_intent_store"):
        app.state.pending_intent_store = InMemoryPendingIntentStore()
    if not hasattr(app.state, "wake_queue"):
        app.state.wake_queue = InMemoryWakeQueueStore()
    if not hasattr(app.state, "mesh_prober") and hasattr(app.state, "host_session_hub"):
        app.state.mesh_prober = MeshProber(hub=app.state.host_session_hub)
    app.state.approval_scope_store = SqlApprovalScopeStore(lambda: app.state.database.sessions())
    # UDC-17 S17.6: durable sessions. The database is created in the app lifespan, so the
    # store reaches it at call time.
    app.state.cu_session_store = SqlComputerUseSessionStore(lambda: app.state.database.sessions())
    # Wave D M1/M2: Autonomous job store and detached runner wired to real Orchestrator + LLM
    if not hasattr(app.state, "job_store"):
        app.state.job_store = SqlAutonomousJobStore(lambda: app.state.database.sessions())
    if not hasattr(app.state, "job_runner"):
        compactor = SubGoalCompactor()
        planner = JobPlannerExecutor(compactor=compactor)
        runner = AutonomousJobRunner(
            job_store=app.state.job_store,
            cu_session_store=app.state.cu_session_store,
            planner=planner,
            compactor=compactor,
        )
        step_executor = OrchestratorJobStepExecutor(
            orchestrator_factory=lambda: build_stream_orchestrator(app.state),
            host_repo_factory=lambda: build_short_session_host_repository(app.state),
            hub=app.state.host_session_hub,
            audit_factory=lambda: build_short_session_audit_log(app.state),
            generator_factory=lambda: (
                getattr(app.state, "expert_text_generator", None)
                or getattr(app.state, "text_generator", None)
            ),
            planner=planner,
            compactor=compactor,
            on_pending_approval=runner.mark_live_pending_approval,
            wake_queue=app.state.wake_queue,
        )
        runner._step_executor = step_executor
        app.state.job_runner = runner

    if not hasattr(app.state, "watch_repository"):
        app.state.watch_repository = SqlJobWatchRepository(lambda: app.state.database.sessions())
    if not hasattr(app.state, "wake_inbox_repository"):
        app.state.wake_inbox_repository = SqlWakeInboxRepository(
            lambda: app.state.database.sessions()
        )
    if not hasattr(app.state, "wake_inbox_service"):
        app.state.wake_inbox_service = WakeInboxService(repo=app.state.wake_inbox_repository)
    if not hasattr(app.state, "watch_coordinator"):
        app.state.watch_coordinator = WatchCoordinator(
            watch_repo=app.state.watch_repository,
            job_store=app.state.job_store,
            hub=app.state.host_session_hub,
            cu_session_store=app.state.cu_session_store,
            wake_inbox=app.state.wake_inbox_service,
            job_runner=getattr(app.state, "job_runner", None),
        )
    if not hasattr(app.state, "yield_detach_service"):
        app.state.yield_detach_service = YieldDetachService(
            job_store=app.state.job_store,
            watch_coordinator=app.state.watch_coordinator,
            hub=app.state.host_session_hub,
            cu_session_store=app.state.cu_session_store,
        )

    runner_obj = getattr(app.state, "job_runner", None)
    if runner_obj is not None:
        if hasattr(runner_obj, "set_watch_coordinator"):
            runner_obj.set_watch_coordinator(app.state.watch_coordinator)
        if hasattr(app.state.watch_coordinator, "set_job_runner") and (
            getattr(app.state.watch_coordinator, "_job_runner", None) is None
        ):
            app.state.watch_coordinator.set_job_runner(runner_obj)
        if hasattr(runner_obj, "set_wake_inbox"):
            runner_obj.set_wake_inbox(app.state.wake_inbox_service)
        step_exec = getattr(runner_obj, "_step_executor", None)
        if step_exec is not None:
            if hasattr(step_exec, "set_watch_coordinator"):
                step_exec.set_watch_coordinator(app.state.watch_coordinator)
            if hasattr(step_exec, "set_wake_inbox"):
                step_exec.set_wake_inbox(app.state.wake_inbox_service)

    async def _on_host_ready(owner_id: object, host_id: object) -> None:
        await handle_host_ready_wake_queue(
            owner_id=owner_id,  # type: ignore[arg-type]
            host_id=host_id,  # type: ignore[arg-type]
            wake_queue=getattr(app.state, "wake_queue", None),
            pending_intent_store=getattr(app.state, "pending_intent_store", None),
            job_runner=getattr(app.state, "job_runner", None),
        )
        watch_coord = getattr(app.state, "watch_coordinator", None)
        if watch_coord is not None:
            await watch_coord.on_host_ready(
                owner_id=owner_id,
                host_id=host_id,
            )

    async def _on_host_watch_event(owner_id: object, host_id: object, params: object) -> None:
        watch_coord = getattr(app.state, "watch_coordinator", None)
        if watch_coord is not None:
            await watch_coord.on_watch_event(
                owner_id=owner_id,
                host_id=host_id,
                event=params,
            )

    app.state.on_host_ready = _on_host_ready
    app.state.on_host_watch_event = _on_host_watch_event
    app.include_router(actions_router)
    app.include_router(job_router)
    app.include_router(mesh_router)
    app.include_router(watch_router)
