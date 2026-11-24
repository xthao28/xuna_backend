"""Asynchronous detached runner for `AutonomousJob` (Wave D M1).

Executes multi-step tasks (40-50 steps) in background tasks detached from any single HTTP
SSE connection. Maintains CU session heartbeats so phone screen-lock does not kill active
sessions, safely transitions to `AWAITING_R3_CHECKPOINT` on unexempt R3 actions, and resumes
upon P-256 approval.
"""

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Collection
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from xuna_backend.modules.actions.application.catalog import (
    CAP_HOST_ACTIONS_V1,
    load_tool_catalog_for_owner,
)
from xuna_backend.modules.actions.application.compactor import SubGoalCompactor
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    Orchestrator,
    tool_system_prompt,
)
from xuna_backend.modules.actions.application.planner_executor import JobPlannerExecutor
from xuna_backend.modules.actions.application.ports import (
    ComputerUseSessionStore,
)
from xuna_backend.modules.actions.application.wake_inbox import WakeInboxService
from xuna_backend.modules.actions.application.watch_coordinator import (
    ReentrantJobLock,
    WatchCoordinator,
    unify_job_locks,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
)
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    JobStepOutcome,
    PendingWakeInboxItem,
    SubGoal,
    SubGoalStatus,
)
from xuna_backend.modules.actions.domain.ports import AutonomousJobRepository
from xuna_backend.modules.actions.domain.schema_validator import (
    ArgumentValidationError,
    SchemaValidator,
)
from xuna_backend.modules.actions.domain.tool_names import from_function_name
from xuna_backend.modules.actions.domain.wake_queue import (
    WAKE_QUEUE_TTL,
    DeferredWakeStep,
    InMemoryWakeQueueStore,
    WakeQueueFullError,
)
from xuna_backend.modules.actions.domain.watch import (
    JOB_WAIT_FUNCTION_NAME,
    JOB_WAIT_SCHEMA,
    ContinuationMode,
    JobWatch,
    WakeInboxKind,
    WakeSuggestedAction,
    WatchCancelReason,
    WatchGestureCue,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.language_detector import detect_text_language
from xuna_backend.modules.cognition.domain.types import (
    ChatRequest,
    Message,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
)
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.entities import HostId, HostStatus, RiskTier

_logger = logging.getLogger(__name__)


def sanitize_checkpoint_data(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively removes base64 images and raw vision payloads (Consent U5 Zero-Retention)."""
    cleaned: dict[str, Any] = {}
    for k, v in data.items():
        if k in {"image_base64", "screenshot_base64", "image_bytes"}:
            continue
        if isinstance(v, str) and (
            v.startswith("data:image/") or (len(v) > 2048 and "base64" in k.lower())
        ):
            continue
        if isinstance(v, dict):
            cleaned[k] = sanitize_checkpoint_data(v)
        elif isinstance(v, list):
            cleaned[k] = [
                sanitize_checkpoint_data(item) if isinstance(item, dict) else item for item in v
            ]
        else:
            cleaned[k] = v
    return cleaned


class AutonomousJobRunner:
    """Manages detached execution loops for long-running autonomous jobs."""

    def __init__(
        self,
        job_store: AutonomousJobRepository,
        cu_session_store: ComputerUseSessionStore | None = None,
        step_executor: Callable[[AutonomousJob], Awaitable[JobStepOutcome]] | None = None,
        planner: JobPlannerExecutor | None = None,
        compactor: SubGoalCompactor | None = None,
        clock_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
        watch_coordinator: WatchCoordinator | None = None,
        wake_inbox: WakeInboxService | None = None,
        job_locks: dict[uuid.UUID, ReentrantJobLock] | None = None,
    ) -> None:
        self._store = job_store
        self._cu_store = cu_session_store
        self._step_executor = step_executor
        self._clock = clock_utc
        self._sleep = sleep_func
        self._compactor = compactor or SubGoalCompactor()
        self._planner = planner or JobPlannerExecutor(
            compactor=self._compactor, clock_utc=self._clock
        )
        self._watch_coordinator = watch_coordinator
        self._wake_inbox = wake_inbox
        self._active_tasks: dict[uuid.UUID, asyncio.Task[None]] = {}
        self._waiting_r3_jobs: dict[uuid.UUID, uuid.UUID] = {}
        self._job_locks: dict[uuid.UUID, ReentrantJobLock] = (
            job_locks if job_locks is not None else {}
        )
        self._cancelling_scopes: list[tuple[uuid.UUID, HostId | None]] = []
        self._cancelling_jobs: set[uuid.UUID] = set()
        if watch_coordinator is not None:
            self.set_watch_coordinator(watch_coordinator)

    def set_watch_coordinator(self, watch_coordinator: WatchCoordinator) -> None:
        self._watch_coordinator = watch_coordinator
        coord_locks = getattr(watch_coordinator, "_job_locks", None)
        if isinstance(coord_locks, dict):
            unify_job_locks(coord_locks, self._job_locks)
            self._job_locks = coord_locks
        coord_scopes = getattr(watch_coordinator, "_cancelling_scopes", None)
        if isinstance(coord_scopes, list) and coord_scopes is not self._cancelling_scopes:
            coord_scopes.extend(self._cancelling_scopes)
            self._cancelling_scopes = coord_scopes
        coord_jobs = getattr(watch_coordinator, "_cancelling_jobs", None)
        if isinstance(coord_jobs, set) and coord_jobs is not self._cancelling_jobs:
            coord_jobs.update(self._cancelling_jobs)
            self._cancelling_jobs = coord_jobs

    def _lock_for_job(self, job_id: uuid.UUID) -> ReentrantJobLock:
        if self._watch_coordinator is not None:
            return self._watch_coordinator._lock_for_job(job_id)
        lock = self._job_locks.get(job_id)
        if lock is None:
            lock = ReentrantJobLock()
            self._job_locks[job_id] = lock
        return lock

    def _is_cancel_pending(self, job: AutonomousJob) -> bool:
        if job.job_id in self._cancelling_jobs:
            return True
        return any(
            oid == job.owner_id and (hid is None or job.host_id == hid)
            for oid, hid in self._cancelling_scopes
        )

    def set_wake_inbox(self, wake_inbox: WakeInboxService) -> None:
        self._wake_inbox = wake_inbox

    @property
    def planner(self) -> JobPlannerExecutor:
        return self._planner

    @property
    def compactor(self) -> SubGoalCompactor:
        return self._compactor

    async def get_job(self, job_id: uuid.UUID) -> AutonomousJob | None:
        return await self._store.get(job_id)

    async def get_active_for_owner(self, owner_id: uuid.UUID) -> AutonomousJob | None:
        return await self._store.get_active_for_owner(owner_id)

    async def list_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        limit: int = 10,
        offset: int = 0,
        states: tuple[AutonomousJobState, ...] | None = None,
    ) -> list[AutonomousJob]:
        return await self._store.list_for_owner(
            owner_id=owner_id,
            limit=limit,
            offset=offset,
            states=states,
        )

    async def start_job(
        self,
        job_id: uuid.UUID,
        *,
        session_id: str | None = None,
        owner_id: uuid.UUID | None = None,
        user_id: uuid.UUID | None = None,
        pet_id: str = "xuna_pet",
        host_id: HostId | None = None,
        title: str | None = None,
        prompt: str | None = None,
        goal: str | None = None,
        budget_steps: int = 50,
        max_steps: int | None = None,
        sub_goals: tuple[SubGoal, ...] | list[dict[str, Any]] | None = None,
    ) -> AutonomousJob:
        """Creates or retrieves a job, transitions it to RUNNING, and starts background loop."""
        actual_owner = owner_id or user_id
        if actual_owner is None:
            raise ValueError("owner_id or user_id is required to start a job")

        actual_goal = goal or prompt or ""
        actual_title = title or (actual_goal[:40] if actual_goal else "Autonomous Job")
        step_limit = max_steps if max_steps is not None else budget_steps

        job = await self._store.get(job_id)
        now = self._clock()

        if job is None:
            parsed_sub_goals: tuple[SubGoal, ...] = ()
            if sub_goals:
                parsed_sub_goals = tuple(
                    SubGoal.from_dict(sg) if isinstance(sg, dict) else sg for sg in sub_goals
                )
            else:
                parsed_sub_goals = self._planner.generate_plan(
                    actual_goal, title=actual_title, max_steps=step_limit
                )

            initial_checkpoint = {
                "current_sub_goal_idx": 0,
                "step_count": 0,
                "sub_goals": [sg.to_dict() for sg in parsed_sub_goals],
                "compacted_history": [],
                "active_turns": [],
                "pending_approval_id": None,
            }

            job = AutonomousJob(
                job_id=job_id,
                owner_id=actual_owner,
                pet_id=pet_id,
                host_id=host_id,
                session_id=session_id,
                title=actual_title,
                goal=actual_goal,
                state=AutonomousJobState.RUNNING,
                sub_goals=parsed_sub_goals,
                max_steps=step_limit,
                checkpoint_data=self._compactor.sanitize_checkpoint(initial_checkpoint),
                created_at=now,
                updated_at=now,
                started_at=now,
            )
        else:
            job = job.with_state(AutonomousJobState.RUNNING, now=now)

        await self._store.save(job)
        self._spawn_loop_task(job_id)
        return job

    async def pause_job(self, job_id: uuid.UUID) -> AutonomousJob:
        """Pauses an active job and cancels any active host watch."""
        cancelled_watches: list[JobWatch] = []
        updated: AutonomousJob
        async with self._lock_for_job(job_id):
            job = await self._store.get(job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found")
            job.validate_transition(AutonomousJobState.PAUSED)

            self._cancel_task(job_id)
            if self._watch_coordinator is not None:
                cancelled_watches = (
                    await self._watch_coordinator.cancel_active_watches_for_job_db_locked(
                        job_id,
                        reason=WatchCancelReason.OWNER_CANCEL.value,
                        now=self._clock(),
                    )
                )
            latest = await self._store.get(job_id) or job
            if latest.is_terminal or latest.state == AutonomousJobState.PAUSED:
                updated = latest
            else:
                updated = latest.with_state(AutonomousJobState.PAUSED, now=self._clock())
                await self._store.save(updated)

        if cancelled_watches and self._watch_coordinator is not None:
            await self._watch_coordinator._send_watch_cancels(
                cancelled_watches,
                host_id=updated.host_id,
                reason=WatchCancelReason.OWNER_CANCEL.value,
            )
        return updated

    async def resume_job(
        self,
        job_id: uuid.UUID,
        approval_id: uuid.UUID | str | None = None,
    ) -> AutonomousJob:
        """Resumes a paused or checkpointed job, clearing pending approval if supplied."""
        async with self._lock_for_job(job_id):
            job = await self._store.get(job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found")

            if job.is_terminal or job.state not in {
                AutonomousJobState.AWAITING_R3_CHECKPOINT,
                AutonomousJobState.WAITING_WAKE,
                AutonomousJobState.WAITING_WATCH,
                AutonomousJobState.PAUSED,
                AutonomousJobState.QUEUED,
            }:
                raise ValueError(f"Cannot resume job in state {job.state.value}")

            if self._is_cancel_pending(job):
                return job

            now = self._clock()
            appr_uuid = uuid.UUID(str(approval_id)) if approval_id else None

            cp = dict(job.checkpoint_data or {})
            cp.pop("wake_expires_at", None)
            if appr_uuid or job.state == AutonomousJobState.AWAITING_R3_CHECKPOINT:
                cp["pending_approval_id"] = None

            updated = job.with_state(
                AutonomousJobState.RUNNING,
                now=now,
                pending_approval_id=None,
                checkpoint_data=sanitize_checkpoint_data(cp),
            )
            await self._store.save(updated)
            self._spawn_loop_task(job_id)
            return updated

    async def steer_job(self, job_id: uuid.UUID, steering_prompt: str) -> AutonomousJob:
        """Injects live human guidance into active job checkpoint data."""
        async with self._lock_for_job(job_id):
            job = await self._store.get(job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found")

            if job.is_terminal:
                raise ValueError(f"Cannot steer terminal job ({job.state.value})")

            now = self._clock()
            cp = dict(job.checkpoint_data or {})
            steering_notes = list(cp.get("steering_notes", []))
            steering_notes.append(
                {
                    "instruction": steering_prompt,
                    "steered_at": now.isoformat(),
                }
            )
            cp["steering_notes"] = steering_notes

            updated = job.with_state(
                job.state,
                now=now,
                checkpoint_data=sanitize_checkpoint_data(cp),
            )
            await self._store.save(updated)
            return updated

    async def cancel_job(
        self,
        job_id: uuid.UUID,
        reason: str = "user_cancelled",
    ) -> AutonomousJob:
        """Cancels a job and tears down its detached background loop."""
        cancelled_watches: list[JobWatch] = []
        updated: AutonomousJob
        self._cancelling_jobs.add(job_id)
        try:
            async with self._lock_for_job(job_id):
                job = await self._store.get(job_id)
                if job is None:
                    raise ValueError(f"Job {job_id} not found")

                self._cancel_task(job_id)
                now = self._clock()
                if self._watch_coordinator is not None:
                    cancelled_watches = (
                        await self._watch_coordinator.cancel_active_watches_for_job_db_locked(
                            job_id,
                            reason=WatchCancelReason.OWNER_CANCEL.value,
                            now=now,
                        )
                    )
                latest = await self._store.get(job_id) or job
                if latest.is_terminal:
                    updated = latest
                else:
                    updated = latest.with_state(
                        AutonomousJobState.CANCELLED,
                        now=now,
                        stop_reason=reason,
                    )
                    await self._store.save(updated)
        finally:
            self._cancelling_jobs.discard(job_id)

        if cancelled_watches and self._watch_coordinator is not None:
            await self._watch_coordinator._send_watch_cancels(
                cancelled_watches,
                host_id=updated.host_id,
                reason=WatchCancelReason.OWNER_CANCEL.value,
            )
        return updated

    async def cancel_all_for_owner(
        self,
        owner_id: uuid.UUID,
        reason: str = "owner_app",
        host_id: HostId | None = None,
    ) -> int:
        """Cancels active jobs belonging to `owner_id` (scoped to `host_id` on Tier-1 E-Stop)."""
        active_states = (
            AutonomousJobState.QUEUED,
            AutonomousJobState.RUNNING,
            AutonomousJobState.AWAITING_R3_CHECKPOINT,
            AutonomousJobState.WAITING_WAKE,
            AutonomousJobState.WAITING_WATCH,
            AutonomousJobState.PAUSED,
        )
        scope = (owner_id, host_id)
        self._cancelling_scopes.append(scope)
        count = 0
        pre_cancelled_watches: list[JobWatch] = []
        now = self._clock()
        try:

            async def _cancel_matching_jobs_locked() -> int:
                cancelled_now = 0
                jobs = await self._store.list_for_owner(owner_id, limit=100, states=active_states)
                matching_ids: set[uuid.UUID] = {
                    j.job_id
                    for j in jobs
                    if not j.is_terminal and (host_id is None or j.host_id == host_id)
                }
                if self._watch_coordinator is not None:
                    watch_job_ids = (
                        await self._watch_coordinator.list_active_watch_job_ids_for_owner(
                            owner_id, host_id=host_id
                        )
                    )
                    matching_ids.update(watch_job_ids)
                for jid in sorted(matching_ids, key=str):
                    async with self._lock_for_job(jid):
                        self._cancel_task(jid)
                        latest = await self._store.get(jid)
                        if (
                            latest is not None
                            and not latest.is_terminal
                            and latest.owner_id == owner_id
                            and (host_id is None or latest.host_id == host_id)
                        ):
                            updated = latest.with_state(
                                AutonomousJobState.CANCELLED,
                                now=self._clock(),
                                stop_reason=reason,
                            )
                            await self._store.save(updated)
                            cancelled_now += 1
                        if self._watch_coordinator is not None:
                            coord = self._watch_coordinator
                            pre_cancelled_watches.extend(
                                await coord.cancel_active_watches_for_job_db_locked(
                                    jid,
                                    reason=reason,
                                    now=now,
                                )
                            )
                return cancelled_now

            count += await _cancel_matching_jobs_locked()
            if self._watch_coordinator is not None:
                await self._watch_coordinator.finish_owner_cancel(
                    owner_id,
                    reason=reason,
                    host_id=host_id,
                    pre_cancelled=pre_cancelled_watches,
                    now=now,
                )
                # Re-sweep under lock in case a job started concurrently during network RPCs
                count += await _cancel_matching_jobs_locked()
        finally:
            with contextlib.suppress(ValueError):
                self._cancelling_scopes.remove(scope)
        return count

    async def mark_live_pending_approval(
        self,
        job_id: uuid.UUID,
        approval_id: uuid.UUID,
    ) -> None:
        """Reflects an in-flight `ActionPendingEvent` on the stored job immediately."""
        async with self._lock_for_job(job_id):
            current = await self._store.get(job_id)
            if (
                current is None
                or current.state != AutonomousJobState.RUNNING
                or self._is_cancel_pending(current)
            ):
                return
            now = self._clock()
            cp = dict(current.checkpoint_data or {})
            cp["pending_approval_id"] = str(approval_id)
            waiting = current.with_state(
                AutonomousJobState.AWAITING_R3_CHECKPOINT,
                now=now,
                pending_approval_id=approval_id,
                checkpoint_data=sanitize_checkpoint_data(cp),
            )
            await self._store.save(waiting)

    async def on_approval_resolved(
        self,
        approval_id: uuid.UUID,
        approved: bool,
    ) -> AutonomousJob | None:
        """Called when an R3 approval is decided; resumes or fails matching waiting job."""
        target_job_id = self._waiting_r3_jobs.pop(approval_id, None)
        if target_job_id is not None:
            if approved:
                return await self.resume_job(target_job_id, approval_id=approval_id)
            async with self._lock_for_job(target_job_id):
                now = self._clock()
                latest = await self._store.get(target_job_id)
                if latest and not latest.is_terminal:
                    failed = latest.with_state(
                        AutonomousJobState.FAILED,
                        now=now,
                        error_message="r3_approval_denied",
                        stop_reason="approval_denied",
                    )
                    await self._store.save(failed)
                    return failed
                return latest
        return None

    async def resume_waiting_wake_jobs(
        self,
        owner_id: uuid.UUID,
        host_id: HostId,
        *,
        allowed_job_ids: Collection[uuid.UUID] | None = None,
        now: datetime | None = None,
    ) -> int:
        """Resumes unexpired `WAITING_WAKE` jobs for `(owner_id, host_id)`, fails expired ones."""
        jobs = await self._store.list_for_owner(
            owner_id,
            limit=50,
            states=(AutonomousJobState.WAITING_WAKE,),
        )
        check_now = now or self._clock()
        resumed = 0
        for job in jobs:
            if job.host_id != host_id:
                continue
            wake_exp_raw = (job.checkpoint_data or {}).get("wake_expires_at")
            wake_expires_at: datetime | None = None
            if isinstance(wake_exp_raw, str):
                with contextlib.suppress(ValueError):
                    parsed_exp = datetime.fromisoformat(wake_exp_raw)
                    if parsed_exp.tzinfo is not None:
                        wake_expires_at = parsed_exp
            if wake_expires_at is None:
                wake_expires_at = job.updated_at + WAKE_QUEUE_TTL
            is_unexpired = check_now < wake_expires_at
            is_allowed = (
                True
                if (allowed_job_ids is not None and job.job_id in allowed_job_ids)
                else is_unexpired
            )
            if is_allowed:
                await self.resume_job(job.job_id)
                resumed += 1
            else:
                failed = job.with_state(
                    AutonomousJobState.FAILED,
                    now=check_now,
                    error_message="Wake-on-LAN window expired (180s TTL)",
                    stop_reason="wake_ttl_expired",
                )
                await self._store.save(failed)
        return resumed

    def _spawn_loop_task(self, job_id: uuid.UUID) -> None:
        self._cancel_task(job_id)
        task = asyncio.create_task(self._run_job_loop(job_id))
        self._active_tasks[job_id] = task

        def _cleanup(t: asyncio.Task[None]) -> None:
            if self._active_tasks.get(job_id) is t:
                self._active_tasks.pop(job_id, None)

        task.add_done_callback(_cleanup)

    def _cancel_task(self, job_id: uuid.UUID) -> None:
        task = self._active_tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()

    async def _run_job_loop(self, job_id: uuid.UUID) -> None:
        """Detached execution loop for an active job."""
        try:
            while True:
                job = await self._store.get(job_id)
                if job is None or job.state != AutonomousJobState.RUNNING:
                    break
                if self._is_cancel_pending(job):
                    break

                now = self._clock()

                # Keep CU session alive during execution so phone lock does not kill it
                if job.session_id and self._cu_store is not None:
                    try:
                        await self._cu_store.touch_heartbeat(job.session_id, job.owner_id, now)
                    except Exception as err:
                        _logger.debug(
                            "Failed touching heartbeat for session %s: %s", job.session_id, err
                        )

                # Check step budget
                if job.steps_used >= job.max_steps:
                    async with self._lock_for_job(job_id):
                        latest_budget = await self._store.get(job_id)
                        if (
                            latest_budget is not None
                            and not latest_budget.is_terminal
                            and latest_budget.state == AutonomousJobState.RUNNING
                            and not self._is_cancel_pending(latest_budget)
                        ):
                            all_done = bool(latest_budget.sub_goals) and all(
                                sg.status == SubGoalStatus.COMPLETED
                                for sg in latest_budget.sub_goals
                            )
                            final_state = (
                                AutonomousJobState.COMPLETED
                                if all_done
                                else AutonomousJobState.FAILED
                            )
                            completed = latest_budget.with_state(
                                final_state,
                                now=now,
                                stop_reason="max_steps_exceeded",
                                error_message="Step budget exhausted" if not all_done else None,
                            )
                            await self._store.save(completed)
                    break

                # Step execution
                if self._step_executor is not None:
                    outcome = await self._step_executor(job)
                else:
                    outcome = await self._default_step(job)

                should_break = False
                should_emit = False
                async with self._lock_for_job(job_id):
                    # Re-check store in case owner paused, cancelled, or steered mid-step
                    latest_after = await self._store.get(job_id)
                    if (
                        latest_after is None
                        or latest_after.is_terminal
                        or latest_after.state == AutonomousJobState.PAUSED
                        or self._is_cancel_pending(latest_after)
                    ):
                        break

                    now = self._clock()

                    raw_cp = dict(outcome.checkpoint_data or latest_after.checkpoint_data or {})
                    if (
                        latest_after.checkpoint_data
                        and "steering_notes" in latest_after.checkpoint_data
                        and "steering_notes" not in raw_cp
                    ):
                        raw_cp["steering_notes"] = latest_after.checkpoint_data["steering_notes"]
                    clean_cp = sanitize_checkpoint_data(raw_cp)
                    clean_cp = self._compactor.sanitize_checkpoint(clean_cp)

                    # Ensure base job is RUNNING before terminal/progress step if live-marked
                    base_job = (
                        latest_after.with_state(
                            AutonomousJobState.RUNNING,
                            now=now,
                            pending_approval_id=None,
                        )
                        if latest_after.state == AutonomousJobState.AWAITING_R3_CHECKPOINT
                        and outcome.kind != "await_r3"
                        else latest_after
                    )

                    if outcome.kind == "await_r3":
                        if outcome.pending_approval_id:
                            self._waiting_r3_jobs[outcome.pending_approval_id] = job.job_id
                        clean_cp["pending_approval_id"] = (
                            str(outcome.pending_approval_id)
                            if outcome.pending_approval_id
                            else None
                        )
                        updated = (
                            latest_after
                            if latest_after.state == AutonomousJobState.AWAITING_R3_CHECKPOINT
                            else latest_after.with_state(
                                AutonomousJobState.AWAITING_R3_CHECKPOINT,
                                now=now,
                                pending_approval_id=outcome.pending_approval_id,
                                checkpoint_data=clean_cp,
                                steps_used=job.steps_used + 1,
                            )
                        )
                        if latest_after.state == AutonomousJobState.AWAITING_R3_CHECKPOINT:
                            updated = AutonomousJob(
                                job_id=latest_after.job_id,
                                owner_id=latest_after.owner_id,
                                pet_id=latest_after.pet_id,
                                host_id=latest_after.host_id,
                                session_id=latest_after.session_id,
                                title=latest_after.title,
                                goal=latest_after.goal,
                                state=AutonomousJobState.AWAITING_R3_CHECKPOINT,
                                sub_goals=(
                                    outcome.sub_goals
                                    if outcome.sub_goals is not None
                                    else latest_after.sub_goals
                                ),
                                active_sub_goal_index=(
                                    outcome.active_sub_goal_index
                                    if outcome.active_sub_goal_index is not None
                                    else latest_after.active_sub_goal_index
                                ),
                                max_steps=latest_after.max_steps,
                                steps_used=job.steps_used + 1,
                                pending_approval_id=outcome.pending_approval_id,
                                checkpoint_data=clean_cp,
                                error_message=latest_after.error_message,
                                stop_reason=latest_after.stop_reason,
                                created_at=latest_after.created_at,
                                updated_at=now,
                                started_at=latest_after.started_at,
                                completed_at=latest_after.completed_at,
                            )
                        await self._store.save(updated)
                        should_emit = True
                        should_break = True  # Pauses execution until resume_job is invoked

                    elif outcome.kind == "waiting_wake":
                        wake_cp = dict(clean_cp)
                        wake_cp["wake_expires_at"] = (now + WAKE_QUEUE_TTL).isoformat()
                        updated = base_job.with_state(
                            AutonomousJobState.WAITING_WAKE,
                            now=now,
                            checkpoint_data=wake_cp,
                            stop_reason=outcome.stop_reason or "waiting_wake",
                        )
                        await self._store.save(updated)
                        should_emit = True
                        should_break = True

                    elif outcome.kind == "waiting_watch":
                        if latest_after.state == AutonomousJobState.WAITING_WATCH:
                            break
                        outcome_watch_id = clean_cp.get("active_watch_id")
                        last_resumed_id = (latest_after.checkpoint_data or {}).get(
                            "last_resumed_watch_id"
                        )
                        if outcome_watch_id is not None and last_resumed_id == outcome_watch_id:
                            break
                        sub_goals_val = (
                            outcome.sub_goals
                            if outcome.sub_goals is not None
                            else base_job.sub_goals
                        )
                        next_active_idx = (
                            outcome.active_sub_goal_index
                            if outcome.active_sub_goal_index is not None
                            else base_job.active_sub_goal_index
                        )
                        updated = base_job.with_state(
                            AutonomousJobState.WAITING_WATCH,
                            now=now,
                            steps_used=job.steps_used + 1,
                            sub_goals=sub_goals_val,
                            active_sub_goal_index=next_active_idx,
                            checkpoint_data=clean_cp,
                            stop_reason=outcome.stop_reason or "waiting_watch",
                        )
                        await self._store.save(updated)
                        should_emit = True
                        should_break = True

                    elif outcome.kind == "paused":
                        sub_goals_val = (
                            outcome.sub_goals
                            if outcome.sub_goals is not None
                            else base_job.sub_goals
                        )
                        next_active_idx = (
                            outcome.active_sub_goal_index
                            if outcome.active_sub_goal_index is not None
                            else base_job.active_sub_goal_index
                        )
                        updated = base_job.with_state(
                            AutonomousJobState.PAUSED,
                            now=now,
                            steps_used=job.steps_used + 1,
                            sub_goals=sub_goals_val,
                            active_sub_goal_index=next_active_idx,
                            checkpoint_data=clean_cp,
                            stop_reason=outcome.stop_reason or "paused",
                        )
                        await self._store.save(updated)
                        should_emit = True
                        should_break = True

                    elif outcome.kind == "progress":
                        sub_goals_val = (
                            outcome.sub_goals
                            if outcome.sub_goals is not None
                            else base_job.sub_goals
                        )
                        next_active_idx = (
                            outcome.active_sub_goal_index
                            if outcome.active_sub_goal_index is not None
                            else base_job.active_sub_goal_index
                        )
                        updated = base_job.with_state(
                            AutonomousJobState.RUNNING,
                            now=now,
                            steps_used=job.steps_used + 1,
                            active_sub_goal_index=next_active_idx,
                            sub_goals=sub_goals_val,
                            checkpoint_data=clean_cp,
                        )
                        await self._store.save(updated)
                        should_emit = True

                    elif outcome.kind == "completed":
                        sub_goals_val = (
                            outcome.sub_goals
                            if outcome.sub_goals is not None
                            else base_job.sub_goals
                        )
                        updated = base_job.with_state(
                            AutonomousJobState.COMPLETED,
                            now=now,
                            steps_used=job.steps_used + 1,
                            sub_goals=sub_goals_val,
                            checkpoint_data=clean_cp,
                            stop_reason=outcome.stop_reason or "completed",
                        )
                        await self._store.save(updated)
                        should_emit = True
                        should_break = True

                    elif outcome.kind == "failed":
                        sub_goals_val = (
                            outcome.sub_goals
                            if outcome.sub_goals is not None
                            else base_job.sub_goals
                        )
                        updated = base_job.with_state(
                            AutonomousJobState.FAILED,
                            now=now,
                            steps_used=job.steps_used + 1,
                            sub_goals=sub_goals_val,
                            checkpoint_data=clean_cp,
                            error_message=outcome.error_message or "Execution failed",
                            stop_reason=outcome.stop_reason or "failed",
                        )
                        await self._store.save(updated)
                        should_emit = True
                        should_break = True

                if should_emit:
                    await self._emit_pending_wake_inbox(outcome)
                if should_break:
                    break

                await self._sleep(0.01)

        except asyncio.CancelledError:
            _logger.info("Job loop cancelled for %s", job_id)
        except Exception as exc:
            _logger.exception("Unexpected error in autonomous job loop for %s: %s", job_id, exc)
            async with self._lock_for_job(job_id):
                now = self._clock()
                latest = await self._store.get(job_id)
                if latest and not latest.is_terminal and not self._is_cancel_pending(latest):
                    err_job = latest.with_state(
                        AutonomousJobState.FAILED,
                        now=now,
                        error_message=str(exc),
                        stop_reason="runtime_error",
                    )
                    await self._store.save(err_job)

    async def _emit_pending_wake_inbox(self, outcome: JobStepOutcome) -> None:
        pending = outcome.pending_wake_inbox
        if pending is None:
            return
        inbox = self._wake_inbox or getattr(self._step_executor, "_wake_inbox", None)
        if inbox is None:
            return
        await inbox.emit_item(
            owner_id=pending.owner_id,
            pet_id=pending.pet_id,
            kind=pending.kind,
            title=pending.title,
            summary=pending.summary,
            job_id=pending.job_id,
            host_id=pending.host_id,
            suggested_actions=pending.suggested_actions,
            gesture_cue=pending.gesture_cue,
        )

    async def _default_step(self, job: AutonomousJob) -> JobStepOutcome:
        """Fail-closed default step when no real step_executor is wired (Rule H1)."""
        return self._planner.step(job)

    async def shutdown(self) -> None:
        """Cancels all active tasks for graceful shutdown."""
        tasks = list(self._active_tasks.values())
        self._active_tasks.clear()
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class OrchestratorJobStepExecutor:
    """Production `step_executor` driving real LLM + `Orchestrator` host tools (Wave D).

    Enforces:
    - Rule H1 & H2: Fails closed (`no_host`, `provider_unavailable`, `no_tool_executed`) when no
      online host/tools exist, when the LLM provider fails, or when no real tool returns `ok`.
      Never fabricates synthetic tool completions.
    - Rule H4: Preserves real tool error codes and fails the active sub-goal on non-`ok` status.
    - Rule H5: App-agnostic orchestration using the owner's dynamic eligible host tool catalog.
    - Wave A Invariant: Preserves `ToolCall.extra` (`thought_signature`) across active turns.
    - Wave C Consent U5: Forwards `ui.observe_hybrid` JPEG screenshots strictly in-memory for the
      single subsequent Vision LLM turn and never writes base64 image data to `checkpoint_data`.
    """

    def __init__(
        self,
        *,
        orchestrator_factory: Callable[[], Orchestrator],
        host_repo_factory: Callable[[], HostRepository],
        hub: HostSessionHub,
        audit_factory: Callable[[], AuditLog],
        generator_factory: Callable[[], TextGenerator | None],
        planner: JobPlannerExecutor,
        compactor: SubGoalCompactor,
        on_pending_approval: Callable[[uuid.UUID, uuid.UUID], Awaitable[None]] | None = None,
        wake_queue: InMemoryWakeQueueStore | None = None,
        watch_coordinator: WatchCoordinator | None = None,
        wake_inbox: WakeInboxService | None = None,
        clock_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._orchestrator_factory = orchestrator_factory
        self._host_repo_factory = host_repo_factory
        self._hub = hub
        self._audit_factory = audit_factory
        self._generator_factory = generator_factory
        self._planner = planner
        self._compactor = compactor
        self._on_pending_approval = on_pending_approval
        self._wake_queue = wake_queue
        self._watch_coordinator = watch_coordinator
        self._wake_inbox = wake_inbox
        self._clock = clock_utc
        self._ephemeral_vision: dict[uuid.UUID, tuple[str, str]] = {}

    def set_watch_coordinator(self, watch_coordinator: WatchCoordinator) -> None:
        self._watch_coordinator = watch_coordinator

    def set_wake_inbox(self, wake_inbox: WakeInboxService) -> None:
        self._wake_inbox = wake_inbox

    async def __call__(self, job: AutonomousJob) -> JobStepOutcome:
        now = self._clock()
        generator = self._generator_factory()
        if generator is None:
            self._ephemeral_vision.pop(job.job_id, None)
            return JobStepOutcome(
                kind="failed",
                error_message="Text generator is not configured",
                stop_reason="provider_unavailable",
            )

        host_repo = self._host_repo_factory()
        if job.host_id is not None:
            target_host = await host_repo.get_host(job.host_id)
            if (
                target_host is None
                or target_host.owner_id != job.owner_id
                or target_host.status != HostStatus.ACTIVE
            ):
                self._ephemeral_vision.pop(job.job_id, None)
                return JobStepOutcome(
                    kind="failed",
                    error_message="Pinned target host is offline or unavailable",
                    stop_reason="no_host",
                )
            if not self._hub.is_online(str(job.host_id)):
                self._ephemeral_vision.pop(job.job_id, None)
                if target_host.wol_mac:
                    if self._wake_queue is not None:
                        try:
                            self._wake_queue.enqueue(
                                DeferredWakeStep(
                                    step_id=uuid.uuid4(),
                                    owner_id=job.owner_id,
                                    pet_id=job.pet_id,
                                    host_id=job.host_id,
                                    host_alias=target_host.alias.value,
                                    tool="autonomous.job",
                                    arguments={},
                                    risk=RiskTier.R2,
                                    created_at=now,
                                    expires_at=now + WAKE_QUEUE_TTL,
                                    job_id=job.job_id,
                                    summary=job.title[:120],
                                ),
                                now,
                            )
                        except WakeQueueFullError:
                            return JobStepOutcome(
                                kind="failed",
                                error_message="Deferred wake queue limit reached",
                                stop_reason="wake_queue_full",
                            )
                    return JobStepOutcome(
                        kind="waiting_wake",
                        stop_reason="host_offline",
                    )
                return JobStepOutcome(
                    kind="failed",
                    error_message="Pinned target host is offline or unavailable",
                    stop_reason="no_host",
                )

        tools = await load_tool_catalog_for_owner(
            owner_id=job.owner_id,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            host_repo=host_repo,
            hub=self._hub,
            self_host_id=None,
            host_screen_consent=True,
            now=now,
            audit=self._audit_factory(),
            pinned_host_id=job.host_id,
        )
        if not tools:
            self._ephemeral_vision.pop(job.job_id, None)
            return JobStepOutcome(
                kind="failed",
                error_message="No active online host or granted tools available for job",
                stop_reason="no_host",
            )

        if self._watch_coordinator is not None:
            wait_spec = ToolSpec(
                name=JOB_WAIT_FUNCTION_NAME,
                description=(
                    "Pause execution while waiting for an ongoing task on the host to settle "
                    "or complete, then wake up to inspect the real result. Zero tokens are "
                    "spent while waiting."
                ),
                parameters=JOB_WAIT_SCHEMA,
            )
            tools = (*tools, wait_spec)

        sub_goals = (
            list(job.sub_goals)
            if job.sub_goals
            else list(
                self._planner.generate_plan(job.goal, title=job.title, max_steps=job.max_steps)
            )
        )
        idx = job.active_sub_goal_index
        if idx >= len(sub_goals):
            self._ephemeral_vision.pop(job.job_id, None)
            return JobStepOutcome(
                kind="completed",
                sub_goals=tuple(sub_goals),
                active_sub_goal_index=max(len(sub_goals) - 1, 0),
                stop_reason="all_sub_goals_completed",
            )

        current_sg = sub_goals[idx]
        if current_sg.status == SubGoalStatus.PENDING:
            current_sg = SubGoal(
                id=current_sg.id,
                title=current_sg.title,
                description=current_sg.description,
                status=SubGoalStatus.IN_PROGRESS,
                tool_call_count=current_sg.tool_call_count,
                result_summary=current_sg.result_summary,
                started_at=now,
            )
            sub_goals[idx] = current_sg

        cp = dict(job.checkpoint_data or {})
        compacted_history = list(cp.get("compacted_history", []))
        active_turns = list(cp.get("active_turns", []))
        steering_notes = list(cp.get("steering_notes", []))

        locale = detect_text_language(job.goal)
        steering_block = ""
        if steering_notes:
            lines = [
                f"- {note.get('instruction', '')}"
                for note in steering_notes
                if isinstance(note, dict) and note.get("instruction")
            ]
            if lines:
                steering_block = "\n\n[Owner Live Steering]:\n" + "\n".join(lines)

        system_instruction = (
            f"{tool_system_prompt(locale, tools=tools)}\n\n"
            f"[Autonomous Job Goal]: {job.goal}\n"
            f"[Active Sub-Goal {idx + 1}/{len(sub_goals)}]: {current_sg.title} — "
            f"{current_sg.description}\n"
            "Call the required host tool(s) to execute and verify this sub-goal. "
            "Once at least one real tool has returned status='ok' and this sub-goal is "
            "genuinely satisfied, respond with a concise completion summary and no tool calls."
            f"{steering_block}"
        )

        # Ensure an initial user message begins the active sub-goal turn sequence
        has_user_turn = any(
            isinstance(t, dict) and t.get("role") == "user" and t.get("sub_goal_index") == idx
            for t in active_turns
        )
        turns_for_llm = list(active_turns)
        if not has_user_turn:
            user_prompt_turn: dict[str, Any] = {
                "sub_goal_index": idx,
                "role": "user",
                "content": (
                    f"Execute Sub-Goal {idx + 1}: " f"{current_sg.title} ({current_sg.description})"
                ),
            }
            active_turns.append(user_prompt_turn)
            turns_for_llm.append(user_prompt_turn)

        messages = list(
            self._compactor.build_context_messages(
                compacted_history=compacted_history,
                active_turns=turns_for_llm,
                system_instruction=system_instruction,
            )
        )

        # Ephemeral in-memory Vision attachment (Consent U5: consumed once, never persisted)
        ephemeral_img = self._ephemeral_vision.pop(job.job_id, None)
        if ephemeral_img is not None:
            img_b64, img_mime = ephemeral_img
            messages.append(
                Message(
                    role=Role.USER,
                    content="[Ephemeral window observation attached for current step]",
                    image_base64=img_b64,
                    image_mime_type=img_mime,
                )
            )

        request = ChatRequest(
            messages=tuple(messages),
            tools=tools,
            locale=locale,
            max_output_tokens=1200,
            timeout_s=30.0,
        )

        text_parts: list[str] = []
        requested_calls: list[ToolCall] = []
        try:
            async for event in generator.stream(request):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                elif isinstance(event, ToolCallRequested):
                    requested_calls.append(event.call)
        except ProviderUnavailableError as err:
            return JobStepOutcome(
                kind="failed",
                sub_goals=tuple(sub_goals),
                active_sub_goal_index=idx,
                error_message=str(err),
                stop_reason="provider_unavailable",
            )

        if not requested_calls:
            summary_text = "".join(text_parts).strip() or None
            job_with_sg = AutonomousJob(
                job_id=job.job_id,
                owner_id=job.owner_id,
                pet_id=job.pet_id,
                host_id=job.host_id,
                session_id=job.session_id,
                title=job.title,
                goal=job.goal,
                state=job.state,
                sub_goals=tuple(sub_goals),
                active_sub_goal_index=idx,
                max_steps=job.max_steps,
                steps_used=job.steps_used,
                pending_approval_id=job.pending_approval_id,
                checkpoint_data={
                    **cp,
                    "active_turns": active_turns,
                    "compacted_history": compacted_history,
                },
                created_at=job.created_at,
                updated_at=now,
                started_at=job.started_at,
            )
            completion_outcome = self._planner.complete_sub_goal_after_verified_tools(
                job_with_sg,
                outcome_summary=summary_text,
                now=now,
            )
            cont_mode = cp.get("active_watch_continuation_mode")
            had_watch = bool(
                cp.get("active_watch_id")
                or cp.get("last_resumed_watch_id")
                or cont_mode
                or cp.get("had_watch_in_job")
            )
            if completion_outcome.kind == "failed":
                pending_fail = (
                    PendingWakeInboxItem(
                        owner_id=job.owner_id,
                        pet_id=job.pet_id,
                        kind=WakeInboxKind.JOB_FAILED,
                        title=f"Verification failed: {job.title[:50]}",
                        summary=(
                            completion_outcome.error_message
                            or completion_outcome.stop_reason
                            or "No verified tool execution"
                        ),
                        job_id=job.job_id,
                        host_id=job.host_id,
                        suggested_actions=(WakeSuggestedAction.DISMISS,),
                        gesture_cue=WatchGestureCue.LOOK_AROUND,
                    )
                    if had_watch and self._wake_inbox is not None
                    else None
                )
                return replace(completion_outcome, pending_wake_inbox=pending_fail)

            cleaned_cp = dict(completion_outcome.checkpoint_data or {})
            if cp.get("had_watch_in_job"):
                cleaned_cp["had_watch_in_job"] = True
            for wk_key in (
                "active_watch_continuation_mode",
                "active_watch_id",
                "active_watch_done_criteria",
                "last_resumed_watch_id",
                "last_quiescence_seconds",
                "wake_checks_used",
            ):
                cleaned_cp.pop(wk_key, None)

            if completion_outcome.kind == "completed":
                pending_done = (
                    PendingWakeInboxItem(
                        owner_id=job.owner_id,
                        pet_id=job.pet_id,
                        kind=WakeInboxKind.JOB_REPORT,
                        title=f"Done: {job.title[:60]}",
                        summary=summary_text or f"Completed '{current_sg.title}'",
                        job_id=job.job_id,
                        host_id=job.host_id,
                        suggested_actions=(WakeSuggestedAction.DISMISS,),
                        gesture_cue=WatchGestureCue.BOUNCE_PLAY,
                    )
                    if had_watch and self._wake_inbox is not None
                    else None
                )
                return JobStepOutcome(
                    kind="completed",
                    sub_goals=completion_outcome.sub_goals,
                    active_sub_goal_index=completion_outcome.active_sub_goal_index,
                    checkpoint_data=cleaned_cp,
                    stop_reason=completion_outcome.stop_reason,
                    pending_wake_inbox=pending_done,
                )

            if completion_outcome.kind == "progress" and had_watch:
                if cont_mode == ContinuationMode.REPORT_ONLY.value:
                    pending_report = (
                        PendingWakeInboxItem(
                            owner_id=job.owner_id,
                            pet_id=job.pet_id,
                            kind=WakeInboxKind.JOB_REPORT,
                            title=f"Update: {job.title[:60]}",
                            summary=summary_text or f"Completed '{current_sg.title}'",
                            job_id=job.job_id,
                            host_id=job.host_id,
                            suggested_actions=(WakeSuggestedAction.DISMISS,),
                            gesture_cue=WatchGestureCue.BOUNCE_PLAY,
                        )
                        if self._wake_inbox is not None
                        else None
                    )
                    return JobStepOutcome(
                        kind="paused",
                        sub_goals=completion_outcome.sub_goals,
                        active_sub_goal_index=completion_outcome.active_sub_goal_index,
                        checkpoint_data=cleaned_cp,
                        stop_reason="report_only",
                        pending_wake_inbox=pending_report,
                    )
                if cont_mode == ContinuationMode.REPORT_AND_ASK.value:
                    pending_ask = (
                        PendingWakeInboxItem(
                            owner_id=job.owner_id,
                            pet_id=job.pet_id,
                            kind=WakeInboxKind.JOB_NEEDS_DECISION,
                            title=f"Ready to continue: {job.title[:50]}",
                            summary=summary_text or f"Completed '{current_sg.title}'",
                            job_id=job.job_id,
                            host_id=job.host_id,
                            suggested_actions=(
                                WakeSuggestedAction.CONTINUE_JOB,
                                WakeSuggestedAction.DISMISS,
                            ),
                            gesture_cue=WatchGestureCue.LOOK_AROUND,
                        )
                        if self._wake_inbox is not None
                        else None
                    )
                    return JobStepOutcome(
                        kind="paused",
                        sub_goals=completion_outcome.sub_goals,
                        active_sub_goal_index=completion_outcome.active_sub_goal_index,
                        checkpoint_data=cleaned_cp,
                        stop_reason="awaiting_owner_decision",
                        pending_wake_inbox=pending_ask,
                    )
            return JobStepOutcome(
                kind=completion_outcome.kind,
                sub_goals=completion_outcome.sub_goals,
                active_sub_goal_index=completion_outcome.active_sub_goal_index,
                checkpoint_data=cleaned_cp,
                stop_reason=completion_outcome.stop_reason,
            )

        # Handle internal `job__wait` before touching Orchestrator
        call = requested_calls[0]
        if call.name == JOB_WAIT_FUNCTION_NAME:
            if self._watch_coordinator is None:
                return JobStepOutcome(
                    kind="failed",
                    sub_goals=tuple(sub_goals),
                    active_sub_goal_index=idx,
                    error_message="Watch coordinator is not configured",
                    stop_reason="watch_unsupported",
                )
            try:
                clean_wait_args, _ = SchemaValidator.validate_arguments(
                    JOB_WAIT_SCHEMA, dict(call.arguments), allow_target=False
                )
            except ArgumentValidationError as err:
                return JobStepOutcome(
                    kind="failed",
                    sub_goals=tuple(sub_goals),
                    active_sub_goal_index=idx,
                    error_message=str(err),
                    stop_reason="invalid_args",
                )
            reg = await self._watch_coordinator.register_from_job_wait(
                job=job,
                clean_args=clean_wait_args,
                sub_goals=tuple(sub_goals),
                active_sub_goal_index=idx,
                active_turns=active_turns,
                compacted_history=compacted_history,
                assistant_call=call,
                assistant_text="".join(text_parts),
                steering_notes=steering_notes,
                now=now,
            )
            if not reg.accepted:
                return JobStepOutcome(
                    kind="failed",
                    sub_goals=tuple(sub_goals),
                    active_sub_goal_index=idx,
                    checkpoint_data=reg.checkpoint_data or cp,
                    error_message=reg.error_message or reg.error_code or "watch_failed",
                    stop_reason=reg.error_code or "watch_unsupported",
                    pending_wake_inbox=reg.pending_wake_inbox,
                )
            return JobStepOutcome(
                kind="waiting_watch",
                sub_goals=tuple(sub_goals),
                active_sub_goal_index=idx,
                checkpoint_data=reg.checkpoint_data,
                stop_reason="waiting_watch",
            )

        # Execute the tool call through the real Orchestrator pipeline
        resolved_tool_name = from_function_name(call.name) or call.name
        captured_approval_id: uuid.UUID | None = None

        async def _emit(ev: ActionStreamEvent) -> None:
            nonlocal captured_approval_id
            if isinstance(ev, ActionPendingEvent):
                try:
                    captured_approval_id = uuid.UUID(ev.approval_id)
                except ValueError:
                    captured_approval_id = None
                if captured_approval_id is not None and self._on_pending_approval is not None:
                    await self._on_pending_approval(job.job_id, captured_approval_id)

        orch = self._orchestrator_factory()
        job_apps_raw = cp.get("apps") or (
            call.arguments.get("apps") if isinstance(call.arguments, dict) else None
        )
        job_apps = [str(a) for a in job_apps_raw] if isinstance(job_apps_raw, list | tuple) else []
        job_domains_raw = cp.get("domains") or (
            call.arguments.get("domains") if isinstance(call.arguments, dict) else None
        )
        job_domains = (
            [str(d) for d in job_domains_raw] if isinstance(job_domains_raw, list | tuple) else []
        )
        job_mission_env: dict[str, object] = {
            "goal": job.goal,
            "apps": job_apps,
            "domains": job_domains,
            "maxSteps": job.max_steps,
            "maxMinutes": int(cp.get("max_minutes") or cp.get("maxMinutes") or 15),
            "planSteps": [sg.title if hasattr(sg, "title") else str(sg) for sg in job.sub_goals],
        }
        action_ctx = ActionContext(
            owner_id=job.owner_id,
            pet_id=job.pet_id,
            locale=locale,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
            request_id=f"job-{job.job_id}-step-{job.steps_used + 1}",
            self_host_id=None,
            host_screen_consent=True,
            mission_envelope=job_mission_env,
            pinned_host_id=job.host_id,
        )
        tool_msg = await orch.handle_call(call, action_ctx)

        if tool_msg.image_base64:
            self._ephemeral_vision[job.job_id] = (
                tool_msg.image_base64,
                tool_msg.image_mime_type,
            )

        try:
            parsed_result = json.loads(tool_msg.content)
            if not isinstance(parsed_result, dict):
                parsed_result = {"status": "failed", "errorCode": "invalid_tool_payload"}
        except ValueError:
            parsed_result = {"status": "failed", "errorCode": "invalid_tool_payload"}

        # If approval timed out while phone was locked, pause cleanly at R3 checkpoint
        if (
            parsed_result.get("status") in ("approval_timeout", "expired")
            and captured_approval_id is not None
        ):
            pending_r3 = (
                PendingWakeInboxItem(
                    owner_id=job.owner_id,
                    pet_id=job.pet_id,
                    kind=WakeInboxKind.JOB_NEEDS_DECISION,
                    title=f"Approval needed: {job.title[:50]}",
                    summary=f"Action {resolved_tool_name} requires owner approval.",
                    job_id=job.job_id,
                    host_id=job.host_id,
                    suggested_actions=(
                        WakeSuggestedAction.OPEN_APPROVAL,
                        WakeSuggestedAction.DISMISS,
                    ),
                    gesture_cue=WatchGestureCue.LOOK_AROUND,
                )
                if self._wake_inbox is not None
                and bool(
                    cp.get("last_resumed_watch_id")
                    or cp.get("active_watch_continuation_mode")
                    or cp.get("had_watch_in_job")
                )
                else None
            )
            checkpoint = {
                **cp,
                "current_sub_goal_idx": idx,
                "step_count": job.steps_used + 1,
                "sub_goals": [sg.to_dict() for sg in sub_goals],
                "compacted_history": compacted_history,
                "active_turns": active_turns,
                "pending_approval_id": str(captured_approval_id),
                "pending_tool": resolved_tool_name,
            }
            return JobStepOutcome(
                kind="await_r3",
                sub_goals=tuple(sub_goals),
                active_sub_goal_index=idx,
                pending_approval_id=captured_approval_id,
                checkpoint_data=self._compactor.sanitize_checkpoint(checkpoint),
                pending_wake_inbox=pending_r3,
            )

        assistant_turn_dict: dict[str, Any] = {
            "sub_goal_index": idx,
            "role": "assistant",
            "content": "".join(text_parts),
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": dict(call.arguments),
                    **({"extra": dict(call.extra)} if call.extra else {}),
                }
            ],
        }
        tool_turn_dict: dict[str, Any] = {
            "sub_goal_index": idx,
            "role": "tool",
            "tool_call_id": call.id,
            "tool_name": resolved_tool_name,
            "status": str(parsed_result.get("status", "")),
            "content": tool_msg.content,
        }
        updated_active_turns = [*active_turns, assistant_turn_dict, tool_turn_dict]

        job_for_eval = AutonomousJob(
            job_id=job.job_id,
            owner_id=job.owner_id,
            pet_id=job.pet_id,
            host_id=job.host_id,
            session_id=job.session_id,
            title=job.title,
            goal=job.goal,
            state=job.state,
            sub_goals=tuple(sub_goals),
            active_sub_goal_index=idx,
            max_steps=job.max_steps,
            steps_used=job.steps_used,
            pending_approval_id=job.pending_approval_id,
            checkpoint_data={
                **cp,
                "active_turns": updated_active_turns,
                "compacted_history": compacted_history,
            },
            created_at=job.created_at,
            updated_at=now,
            started_at=job.started_at,
        )
        outcome = self._planner.process_tool_result(
            job=job_for_eval,
            tool_name=resolved_tool_name,
            tool_result=parsed_result,
            mark_sub_goal_complete=False,
            now=now,
        )
        if outcome.kind == "progress":
            merged_cp = dict(outcome.checkpoint_data or {})
            merged_cp["active_turns"] = [
                self._compactor.sanitize_active_turn(t) for t in updated_active_turns
            ]
            if steering_notes:
                merged_cp["steering_notes"] = steering_notes
            for wk_key in (
                "wake_checks_used",
                "max_wake_checks",
                "last_quiescence_seconds",
                "active_watch_continuation_mode",
                "active_watch_done_criteria",
                "last_resumed_watch_id",
                "active_watch_id",
                "had_watch_in_job",
            ):
                if wk_key in cp and wk_key not in merged_cp:
                    merged_cp[wk_key] = cp[wk_key]
            return JobStepOutcome(
                kind="progress",
                sub_goals=outcome.sub_goals,
                active_sub_goal_index=outcome.active_sub_goal_index,
                checkpoint_data=self._compactor.sanitize_checkpoint(merged_cp),
            )
        return outcome
