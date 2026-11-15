"""Asynchronous detached runner for `AutonomousJob` (Wave D M1).

Executes multi-step tasks (40-50 steps) in background tasks detached from any single HTTP
SSE connection. Maintains CU session heartbeats so phone screen-lock does not kill active
sessions, safely transitions to `AWAITING_R3_CHECKPOINT` on unexempt R3 actions, and resumes
upon P-256 approval.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
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
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
)
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    JobStepOutcome,
    SubGoal,
    SubGoalStatus,
)
from xuna_backend.modules.actions.domain.ports import AutonomousJobRepository
from xuna_backend.modules.actions.domain.tool_names import from_function_name
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
)
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.entities import HostId, HostStatus, ToolFamily

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
        self._active_tasks: dict[uuid.UUID, asyncio.Task[None]] = {}
        self._waiting_r3_jobs: dict[uuid.UUID, uuid.UUID] = {}

    @property
    def planner(self) -> JobPlannerExecutor:
        return self._planner

    @property
    def compactor(self) -> SubGoalCompactor:
        return self._compactor

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
                parsed_sub_goals = self._planner.generate_plan(actual_goal, title=actual_title)

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
        """Pauses a running or queued job."""
        job = await self._store.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        self._cancel_task(job_id)
        now = self._clock()
        updated = job.with_state(AutonomousJobState.PAUSED, now=now)
        await self._store.save(updated)
        return updated

    async def resume_job(
        self,
        job_id: uuid.UUID,
        approval_id: uuid.UUID | str | None = None,
    ) -> AutonomousJob:
        """Resumes a paused or checkpointed job, clearing pending approval if supplied."""
        job = await self._store.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        if job.state not in {AutonomousJobState.AWAITING_R3_CHECKPOINT, AutonomousJobState.PAUSED}:
            raise ValueError(f"Cannot resume job in state {job.state.value}")

        now = self._clock()
        appr_uuid = uuid.UUID(str(approval_id)) if approval_id else None

        cp = dict(job.checkpoint_data or {})
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
        job = await self._store.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        self._cancel_task(job_id)
        now = self._clock()
        updated = job.with_state(
            AutonomousJobState.CANCELLED,
            now=now,
            stop_reason=reason,
        )
        await self._store.save(updated)
        return updated

    async def cancel_all_for_owner(
        self,
        owner_id: uuid.UUID,
        reason: str = "owner_app",
    ) -> int:
        """Cancels all active jobs belonging to `owner_id` (invoked on Emergency Stop)."""
        active_states = (
            AutonomousJobState.QUEUED,
            AutonomousJobState.RUNNING,
            AutonomousJobState.AWAITING_R3_CHECKPOINT,
            AutonomousJobState.PAUSED,
        )
        jobs = await self._store.list_for_owner(owner_id, limit=100, states=active_states)
        count = 0
        for job in jobs:
            if not job.is_terminal:
                await self.cancel_job(job.job_id, reason=reason)
                count += 1
        return count

    async def mark_live_pending_approval(
        self,
        job_id: uuid.UUID,
        approval_id: uuid.UUID,
    ) -> None:
        """Reflects an in-flight `ActionPendingEvent` on the stored job immediately."""
        current = await self._store.get(job_id)
        if current is None or current.state != AutonomousJobState.RUNNING:
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
                    all_done = bool(job.sub_goals) and all(
                        sg.status == SubGoalStatus.COMPLETED for sg in job.sub_goals
                    )
                    final_state = (
                        AutonomousJobState.COMPLETED if all_done else AutonomousJobState.FAILED
                    )
                    completed = job.with_state(
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

                # Re-check store in case owner paused, cancelled, or steered mid-step
                latest_after = await self._store.get(job_id)
                if latest_after is None or latest_after.is_terminal:
                    break
                if latest_after.state == AutonomousJobState.PAUSED:
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

                # Ensure base job is in RUNNING before terminal/progress transition if live-marked
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
                        str(outcome.pending_approval_id) if outcome.pending_approval_id else None
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
                    break  # Pauses execution until resume_job is invoked

                elif outcome.kind == "progress":
                    sub_goals_val = (
                        outcome.sub_goals if outcome.sub_goals is not None else base_job.sub_goals
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

                elif outcome.kind == "completed":
                    sub_goals_val = (
                        outcome.sub_goals if outcome.sub_goals is not None else base_job.sub_goals
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
                    break

                elif outcome.kind == "failed":
                    sub_goals_val = (
                        outcome.sub_goals if outcome.sub_goals is not None else base_job.sub_goals
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
                    break

                await self._sleep(0.01)

        except asyncio.CancelledError:
            _logger.info("Job loop cancelled for %s", job_id)
        except Exception as exc:
            _logger.exception("Unexpected error in autonomous job loop for %s: %s", job_id, exc)
            now = self._clock()
            latest = await self._store.get(job_id)
            if latest and not latest.is_terminal:
                err_job = latest.with_state(
                    AutonomousJobState.FAILED,
                    now=now,
                    error_message=str(exc),
                    stop_reason="runtime_error",
                )
                await self._store.save(err_job)

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
        self._clock = clock_utc
        self._ephemeral_vision: dict[uuid.UUID, tuple[str, str]] = {}

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
        pinned_defaults: dict[str, str] | None = None
        if job.host_id is not None:
            target_host = await host_repo.get_host(job.host_id)
            if (
                target_host is None
                or target_host.owner_id != job.owner_id
                or target_host.status != HostStatus.ACTIVE
                or not self._hub.is_online(str(job.host_id))
            ):
                self._ephemeral_vision.pop(job.job_id, None)
                return JobStepOutcome(
                    kind="failed",
                    error_message="Pinned target host is offline or unavailable",
                    stop_reason="no_host",
                )
            pinned_defaults = {f.value: target_host.alias.value for f in ToolFamily}

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
        )
        if not tools:
            self._ephemeral_vision.pop(job.job_id, None)
            return JobStepOutcome(
                kind="failed",
                error_message="No active online host or granted tools available for job",
                stop_reason="no_host",
            )

        sub_goals = (
            list(job.sub_goals)
            if job.sub_goals
            else list(self._planner.generate_plan(job.goal, title=job.title))
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
            f"{tool_system_prompt(locale)}\n\n"
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
                    f"Execute Sub-Goal {idx + 1}: "
                    f"{current_sg.title} ({current_sg.description})"
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
            return self._planner.complete_sub_goal_after_verified_tools(
                job_with_sg,
                outcome_summary=summary_text,
                now=now,
            )

        # Execute the tool call through the real Orchestrator pipeline
        call = requested_calls[0]
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
        action_ctx = ActionContext(
            owner_id=job.owner_id,
            pet_id=job.pet_id,
            locale=locale,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_emit,
            request_id=f"job-{job.job_id}-step-{job.steps_used + 1}",
            self_host_id=None,
            host_defaults=pinned_defaults,
            host_screen_consent=True,
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
        if parsed_result.get("status") == "approval_timeout" and captured_approval_id is not None:
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
            return JobStepOutcome(
                kind="progress",
                sub_goals=outcome.sub_goals,
                active_sub_goal_index=outcome.active_sub_goal_index,
                checkpoint_data=self._compactor.sanitize_checkpoint(merged_cp),
            )
        return outcome
