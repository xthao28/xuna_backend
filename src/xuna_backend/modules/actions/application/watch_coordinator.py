"""Coordinator for event-driven host watches, `job__wait`, rehydration, and E-Stop (AW-M2)."""

import asyncio
import contextlib
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    is_hard_denied_target,
)
from xuna_backend.modules.actions.application.ports import ComputerUseSessionStore
from xuna_backend.modules.actions.application.wake_inbox import WakeInboxService
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    PendingWakeInboxItem,
    SubGoal,
)
from xuna_backend.modules.actions.domain.ports import (
    AutonomousJobRepository,
    JobWatchRepository,
)
from xuna_backend.modules.actions.domain.watch import (
    DEFAULT_MAX_WAKE_CHECKS,
    DEFAULT_QUIESCENCE_SECONDS,
    DEFAULT_WATCH_WAIT_SECONDS,
    HARD_MAX_WAKE_CHECKS,
    JOB_WAIT_MIN_WAIT_SECONDS,
    MAX_QUIESCENCE_SECONDS,
    MAX_WATCH_WAIT_SECONDS,
    MIN_QUIESCENCE_SECONDS,
    MIN_WATCH_WAIT_SECONDS,
    ContinuationMode,
    JobWatch,
    WakeInboxKind,
    WakeSuggestedAction,
    WatchCancelReason,
    WatchCompositeSignal,
    WatchEventKind,
    WatchGestureCue,
    WatchMode,
    WatchStatus,
    WatchTarget,
    compute_backoff_quiescence,
    validate_workspace_alias,
)
from xuna_backend.modules.cognition.domain.types import ToolCall
from xuna_backend.modules.hosts.application.ports import HostSessionHub
from xuna_backend.modules.hosts.application.xtp_frames import (
    WatchCompositeSignalWire,
    WatchEventParamsModel,
    WatchModeWire,
    WatchTargetModel,
    build_watch_cancel_params,
    build_watch_register_params,
    parse_watch_register_result,
)
from xuna_backend.modules.hosts.domain.entities import HostId

_logger = logging.getLogger(__name__)
_SHORT_CODE_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")


class JobRunnerPort(Protocol):
    """Minimal job-resume contract needed by `WatchCoordinator`."""

    async def resume_job(
        self,
        job_id: uuid.UUID,
        approval_id: uuid.UUID | str | None = None,
    ) -> AutonomousJob: ...


class WatchNotFoundError(ValueError):
    """Raised when a requested job or watch does not exist for the owner."""


class WatchConflictError(ValueError):
    """Raised when a watch action conflicts with job/watch/host state."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True, slots=True)
class WatchRegistrationOutcome:
    """Result of registering a watch from `job__wait` or `yieldMs` detach."""

    accepted: bool
    watch: JobWatch | None = None
    active_signals: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    checkpoint_data: dict[str, Any] | None = None
    pending_wake_inbox: PendingWakeInboxItem | None = None


class ReentrantJobLock:
    """Task-reentrant async lock per `job_id` shared by WatchCoordinator and JobRunner.

    Prevents self-deadlock when `WatchCoordinator.on_watch_event` / `wake_now` calls
    `JobRunner.resume_job` while holding the per-`job_id` lock, while still serializing
    concurrent tasks (such as `cancel_all_for_owner` / E-Stop) on the same `job_id`.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._depth: int = 0

    async def __aenter__(self) -> "ReentrantJobLock":
        task = asyncio.current_task()
        if self._owner is task and task is not None:
            self._depth += 1
            return self
        await self._lock.acquire()
        self._owner = task
        self._depth = 1
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()


def unify_job_locks(
    primary: dict[uuid.UUID, ReentrantJobLock],
    secondary: dict[uuid.UUID, ReentrantJobLock],
) -> dict[uuid.UUID, ReentrantJobLock]:
    """Merges two per-`job_id` lock registries into `primary` and returns it."""
    if primary is secondary:
        return primary
    for jid, lock in secondary.items():
        primary.setdefault(jid, lock)
    return primary


def _cancel_wire_reason(is_estop: bool) -> Literal["estop", "owner_cancel"]:
    return "estop" if is_estop else "owner_cancel"


def _normalize_app_id(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip().lower().replace(" ", "_")[:64]
    if not cleaned or not _SHORT_CODE_RE.fullmatch(cleaned):
        return None
    return cleaned


def _parse_optional_int(raw: object, *, minimum: int, maximum: int) -> int | None:
    if isinstance(raw, int) and not isinstance(raw, bool) and minimum <= raw <= maximum:
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        val = int(raw.strip())
        if minimum <= val <= maximum:
            return val
    return None


class WatchCoordinator:
    """Manages durable `JobWatch` lifecycles, XTP registration, sweep, and rehydration."""

    def __init__(
        self,
        *,
        watch_repo: JobWatchRepository,
        job_store: AutonomousJobRepository,
        hub: HostSessionHub,
        cu_session_store: ComputerUseSessionStore | None = None,
        wake_inbox: WakeInboxService | None = None,
        job_runner: JobRunnerPort | None = None,
        job_locks: dict[uuid.UUID, ReentrantJobLock] | None = None,
        clock_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
        rpc_timeout_s: float = 5.0,
        sweep_interval_s: float = 10.0,
    ) -> None:
        self._watch_repo = watch_repo
        self._job_store = job_store
        self._hub = hub
        self._cu_store = cu_session_store
        self._wake_inbox = wake_inbox
        self._job_runner = job_runner
        self._clock = clock_utc
        self._rpc_timeout_s = rpc_timeout_s
        self._sweep_interval_s = sweep_interval_s
        self._job_locks: dict[uuid.UUID, ReentrantJobLock] = (
            job_locks if job_locks is not None else {}
        )
        self._cancelling_scopes: list[tuple[uuid.UUID, HostId | None]] = []
        self._cancelling_jobs: set[uuid.UUID] = set()
        self._sweep_task: asyncio.Task[None] | None = None
        if job_runner is not None:
            self._share_job_locks_with(job_runner)

    def _share_job_locks_with(self, peer: object) -> None:
        peer_locks = getattr(peer, "_job_locks", None)
        if isinstance(peer_locks, dict):
            unify_job_locks(self._job_locks, peer_locks)
            peer._job_locks = self._job_locks  # type: ignore[attr-defined]
        peer_scopes = getattr(peer, "_cancelling_scopes", None)
        if isinstance(peer_scopes, list) and peer_scopes is not self._cancelling_scopes:
            self._cancelling_scopes.extend(peer_scopes)
            peer._cancelling_scopes = self._cancelling_scopes  # type: ignore[attr-defined]
        peer_jobs = getattr(peer, "_cancelling_jobs", None)
        if isinstance(peer_jobs, set) and peer_jobs is not self._cancelling_jobs:
            self._cancelling_jobs.update(peer_jobs)
            peer._cancelling_jobs = self._cancelling_jobs  # type: ignore[attr-defined]

    def _is_cancel_pending(
        self,
        *,
        owner_id: uuid.UUID,
        host_id: HostId | None,
        job_id: uuid.UUID,
    ) -> bool:
        if job_id in self._cancelling_jobs:
            return True
        return any(
            oid == owner_id and (hid is None or host_id == hid)
            for oid, hid in self._cancelling_scopes
        )

    def set_job_runner(self, job_runner: JobRunnerPort) -> None:
        self._job_runner = job_runner
        self._share_job_locks_with(job_runner)

    def _lock_for_job(self, job_id: uuid.UUID) -> ReentrantJobLock:
        lock = self._job_locks.get(job_id)
        if lock is None:
            lock = ReentrantJobLock()
            self._job_locks[job_id] = lock
        return lock

    def start_sweep(self) -> None:
        """Starts the lightweight background 10s sweep (CU heartbeat + 15s-bounded expiry)."""
        if self._sweep_task is not None and not self._sweep_task.done():
            return
        self._sweep_task = asyncio.create_task(self._sweep_loop())

    async def stop_sweep(self) -> None:
        """Stops the background sweep cleanly during lifespan shutdown."""
        task = self._sweep_task
        self._sweep_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _sweep_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._sweep_interval_s)
                with contextlib.suppress(Exception):
                    await self.sweep_once()
        except asyncio.CancelledError:
            return

    async def sweep_once(self) -> int:
        """Runs one sweep tick: keeps CU heartbeats alive and expires overdue watches."""
        now = self._clock()
        if self._cu_store is not None:
            active_watches = await self._watch_repo.list_active_watches(limit=200)
            seen_jobs: set[uuid.UUID] = set()
            for watch in active_watches:
                if watch.expires_at <= now or watch.job_id in seen_jobs:
                    continue
                seen_jobs.add(watch.job_id)
                job = await self._job_store.get(watch.job_id)
                if (
                    job is not None
                    and job.state == AutonomousJobState.WAITING_WATCH
                    and job.session_id
                ):
                    with contextlib.suppress(Exception):
                        await self._cu_store.touch_heartbeat(job.session_id, job.owner_id, now)

        overdue = await self._watch_repo.list_expired_active(now, limit=200)
        expired_count = 0
        for watch in overdue:
            expired_target: JobWatch | None = None
            paused_job: AutonomousJob | None = None
            should_emit = False
            async with self._lock_for_job(watch.job_id):
                latest_watch = await self._watch_repo.get(watch.watch_id)
                if (
                    latest_watch is None
                    or latest_watch.status != WatchStatus.ACTIVE
                    or latest_watch.expires_at > now
                ):
                    continue
                expired_target = latest_watch
                paused_job, should_emit = await self._expire_watch_db_locked(
                    latest_watch,
                    now=now,
                    stop_reason="watch_timeout",
                    message="Watch timed out before completion",
                )
                expired_count += 1
            if expired_target is not None:
                await self._expire_watch_notify_outside_lock(
                    expired_target,
                    job=paused_job,
                    message="Watch timed out before completion",
                    stop_reason="watch_timeout",
                    notify_host=True,
                    emit_inbox=should_emit,
                )
        return expired_count

    async def bind_target(
        self,
        *,
        owner_id: uuid.UUID,
        host_id: HostId,
        mode: WatchMode,
        signals: tuple[WatchCompositeSignal, ...],
        session_id: str | None = None,
        workspace_alias: str | None = None,
    ) -> tuple[WatchTarget | None, str | None]:
        """Binds an OS-generic watch target from observed CU/host state; never guesses (H3)."""
        clean_workspace: str | None = None
        if workspace_alias is not None:
            try:
                clean_workspace = validate_workspace_alias(workspace_alias)
            except ValueError:
                return None, "invalid_args"

        if mode == WatchMode.TIMER:
            return WatchTarget(), None

        twin = self._hub.get_twin(str(host_id))
        if (
            clean_workspace is not None
            and twin is not None
            and twin.resources is not None
            and twin.resources.workspaces
            and clean_workspace not in twin.resources.workspaces
        ):
            return None, "workspace_not_found"

        active_cu: ComputerUseSessionState | None = None
        if self._cu_store is not None:
            if session_id:
                candidate_cu = await self._cu_store.get(session_id)
                if (
                    candidate_cu is not None
                    and candidate_cu.active
                    and candidate_cu.owner_id == owner_id
                    and candidate_cu.host_id == host_id
                ):
                    active_cu = candidate_cu
            if active_cu is None:
                active_cu = await self._cu_store.get_active_for_owner(owner_id, host_id=host_id)

        pid: int | None = None
        window_id: int | None = None
        app_id: str | None = None
        tab_id: str | None = None
        mcp_server_id: str | None = None
        protocol_task_id: str | None = None

        if active_cu is not None:
            lw = active_cu.last_window
            pid = _parse_optional_int(lw.get("pid"), minimum=1, maximum=2_147_483_647)
            window_id = _parse_optional_int(lw.get("windowId"), minimum=0, maximum=2_147_483_647)
            raw_app = active_cu.last_app_id
            if raw_app is not None and is_hard_denied_target(raw_app):
                return None, "policy_blocked"
            app_id = _normalize_app_id(raw_app)
            if app_id is None and len(active_cu.allowed_apps) == 1:
                only_app = active_cu.allowed_apps[0]
                if is_hard_denied_target(only_app):
                    return None, "policy_blocked"
                app_id = _normalize_app_id(only_app)
            raw_tab = lw.get("tabId")
            if isinstance(raw_tab, str) and raw_tab.strip():
                tab_id = raw_tab.strip()[:128]
            raw_mcp = lw.get("mcpServerId")
            if isinstance(raw_mcp, str):
                mcp_server_id = _normalize_app_id(raw_mcp)
            raw_task = lw.get("protocolTaskId")
            if isinstance(raw_task, str) and raw_task.strip():
                protocol_task_id = raw_task.strip()[:128]

        if app_id is None and twin is not None and twin.presence is not None:
            fg_app = twin.presence.foreground_app_id
            if fg_app is not None:
                if is_hard_denied_target(fg_app):
                    return None, "policy_blocked"
                app_id = _normalize_app_id(fg_app)

        if app_id is not None and is_hard_denied_target(app_id):
            return None, "policy_blocked"

        needs_fs = mode == WatchMode.FS_QUIESCENCE or WatchCompositeSignal.FS_QUIESCENCE in signals
        if needs_fs and clean_workspace is None:
            return None, "workspace_not_found"

        needs_ui_or_proc = mode in {
            WatchMode.UI_QUIESCENCE,
            WatchMode.UI_CHANGE,
            WatchMode.PROCESS_IDLE,
        } or any(
            s in {WatchCompositeSignal.UI_QUIESCENCE, WatchCompositeSignal.PROCESS_IDLE}
            for s in signals
        )
        if needs_ui_or_proc and pid is None and window_id is None and app_id is None:
            return None, "target_not_found"

        needs_cdp = mode == WatchMode.CDP_IDLE or WatchCompositeSignal.CDP_IDLE in signals
        if needs_cdp and tab_id is None and app_id is None:
            return None, "target_not_found"

        if mode == WatchMode.PROTOCOL_ASYNC and mcp_server_id is None and protocol_task_id is None:
            return None, "watch_unsupported"

        try:
            target = WatchTarget(
                pid=pid,
                window_id=window_id,
                app_id=app_id,
                workspace_alias=clean_workspace,
                tab_id=tab_id,
                mcp_server_id=mcp_server_id,
                protocol_task_id=protocol_task_id,
            )
        except ValueError:
            return None, "invalid_args"
        return target, None

    async def infer_watch_for_detach(
        self,
        *,
        owner_id: uuid.UUID,
        host_id: HostId,
        session_id: str | None = None,
    ) -> tuple[WatchMode, tuple[WatchCompositeSignal, ...], WatchTarget | None, str | None]:
        """Infers an appropriate OS-generic watch for a `yieldMs` auto-detached turn."""
        composite_signals = (WatchCompositeSignal.UI_QUIESCENCE, WatchCompositeSignal.PROCESS_IDLE)
        target, err = await self.bind_target(
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.COMPOSITE,
            signals=composite_signals,
            session_id=session_id,
        )
        if err is None and target is not None:
            return WatchMode.COMPOSITE, composite_signals, target, None
        if err == "policy_blocked":
            return WatchMode.COMPOSITE, composite_signals, None, err
        return WatchMode.TIMER, (), WatchTarget(), None

    async def register_from_job_wait(
        self,
        *,
        job: AutonomousJob,
        clean_args: dict[str, object],
        sub_goals: tuple[SubGoal, ...],
        active_sub_goal_index: int,
        active_turns: list[dict[str, Any]],
        compacted_history: list[dict[str, Any]],
        assistant_call: ToolCall,
        assistant_text: str,
        steering_notes: list[dict[str, Any]],
        now: datetime | None = None,
    ) -> WatchRegistrationOutcome:
        """Validates `job__wait`, persists `JobWatch`, transitions to `WAITING_WATCH`,
        and sends `xuna/watchRegister` under a per-`job_id` lock."""
        check_now = now or self._clock()
        if job.host_id is None:
            return WatchRegistrationOutcome(
                accepted=False,
                error_code="no_host",
                error_message="Job has no pinned host for watch registration",
            )
        host_id = job.host_id

        try:
            mode = WatchMode(str(clean_args["mode"]))
        except ValueError:
            return WatchRegistrationOutcome(
                accepted=False,
                error_code="invalid_args",
                error_message="Invalid watch mode",
            )

        raw_signals = clean_args.get("signals")
        signals: tuple[WatchCompositeSignal, ...] = ()
        if mode == WatchMode.COMPOSITE:
            if not isinstance(raw_signals, list) or not 2 <= len(raw_signals) <= 3:
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code="invalid_args",
                    error_message="composite mode requires 2..3 distinct signals",
                )
            try:
                parsed_sigs = tuple(WatchCompositeSignal(str(s)) for s in raw_signals)
            except ValueError:
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code="invalid_args",
                    error_message="Invalid composite watch signal",
                )
            if len(set(parsed_sigs)) != len(parsed_sigs):
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code="invalid_args",
                    error_message="composite signals must be distinct",
                )
            signals = parsed_sigs
        elif raw_signals is not None:
            return WatchRegistrationOutcome(
                accepted=False,
                error_code="invalid_args",
                error_message="signals is only allowed when mode is composite",
            )

        done_criteria = str(clean_args.get("doneCriteria") or "").strip()[:240]
        if not done_criteria:
            return WatchRegistrationOutcome(
                accepted=False,
                error_code="invalid_args",
                error_message="doneCriteria is required",
            )

        raw_cont = clean_args.get("continuationMode") or ContinuationMode.REPORT_AND_ASK.value
        try:
            continuation_mode = ContinuationMode(str(raw_cont))
        except ValueError:
            return WatchRegistrationOutcome(
                accepted=False,
                error_code="invalid_args",
                error_message="Invalid continuationMode",
            )

        raw_q = clean_args.get("quiescenceSeconds")
        quiescence = (
            int(raw_q)
            if isinstance(raw_q, int) and not isinstance(raw_q, bool)
            else DEFAULT_QUIESCENCE_SECONDS
        )
        quiescence = max(MIN_QUIESCENCE_SECONDS, min(quiescence, MAX_QUIESCENCE_SECONDS))

        raw_max_wait = clean_args.get("maxWaitSeconds")
        max_wait = (
            int(raw_max_wait)
            if isinstance(raw_max_wait, int) and not isinstance(raw_max_wait, bool)
            else DEFAULT_WATCH_WAIT_SECONDS
        )
        max_wait = max(JOB_WAIT_MIN_WAIT_SECONDS, min(max_wait, MAX_WATCH_WAIT_SECONDS))

        raw_ws = clean_args.get("workspaceAlias")
        workspace_alias = str(raw_ws) if isinstance(raw_ws, str) else None

        cp_existing = dict(job.checkpoint_data or {})
        wake_checks_used = int(cp_existing.get("wake_checks_used") or 0)
        max_wake_checks = int(cp_existing.get("max_wake_checks") or DEFAULT_MAX_WAKE_CHECKS)
        max_wake_checks = max(1, min(max_wake_checks, HARD_MAX_WAKE_CHECKS))

        prev_q_raw = cp_existing.get("last_quiescence_seconds")
        is_rewatch = (
            isinstance(prev_q_raw, int) and not isinstance(prev_q_raw, bool) and prev_q_raw > 0
        )
        if isinstance(prev_q_raw, int) and not isinstance(prev_q_raw, bool) and prev_q_raw > 0:
            quiescence = compute_backoff_quiescence(prev_q_raw)

        replaced_watches: list[JobWatch] = []
        async with self._lock_for_job(job.job_id):
            latest_before = await self._job_store.get(job.job_id) or job
            if latest_before.is_terminal or latest_before.state == AutonomousJobState.PAUSED:
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code="cancelled",
                    error_message="Job is no longer active for watch registration",
                    checkpoint_data=cp_existing,
                )
            if wake_checks_used >= max_wake_checks:
                pending_fail = (
                    PendingWakeInboxItem(
                        owner_id=job.owner_id,
                        pet_id=job.pet_id,
                        kind=WakeInboxKind.JOB_FAILED,
                        title=f"Check limit reached: {job.title[:60]}",
                        summary=(
                            f"Stopped after {wake_checks_used} watch verification checks "
                            "without completion."
                        ),
                        job_id=job.job_id,
                        host_id=host_id,
                        suggested_actions=(WakeSuggestedAction.DISMISS,),
                        gesture_cue=WatchGestureCue.LOOK_AROUND,
                    )
                    if self._wake_inbox is not None
                    else None
                )
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code="max_wake_checks_exceeded",
                    error_message="Maximum wake verification checks exceeded",
                    checkpoint_data=cp_existing,
                    pending_wake_inbox=pending_fail,
                )

            target, bind_err = await self.bind_target(
                owner_id=job.owner_id,
                host_id=host_id,
                mode=mode,
                signals=signals,
                session_id=job.session_id,
                workspace_alias=workspace_alias,
            )
            if bind_err is not None or target is None:
                code = bind_err or "target_not_found"
                pending_bind_fail = (
                    PendingWakeInboxItem(
                        owner_id=job.owner_id,
                        pet_id=job.pet_id,
                        kind=WakeInboxKind.JOB_FAILED,
                        title=f"Watch failed: {job.title[:60]}",
                        summary=f"Could not bind watch target ({code}).",
                        job_id=job.job_id,
                        host_id=host_id,
                        suggested_actions=(WakeSuggestedAction.DISMISS,),
                        gesture_cue=WatchGestureCue.LOOK_AROUND,
                    )
                    if self._wake_inbox is not None
                    else None
                )
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code=code,
                    error_message=f"Watch target binding failed: {code}",
                    checkpoint_data=cp_existing,
                    pending_wake_inbox=pending_bind_fail,
                )

            if not self._hub.is_online(str(host_id)):
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code="host_offline",
                    error_message="Pinned target host is offline",
                    checkpoint_data=cp_existing,
                )

            latest_before_save = await self._job_store.get(job.job_id) or latest_before
            if (
                latest_before_save.is_terminal
                or latest_before_save.state == AutonomousJobState.PAUSED
            ):
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code="cancelled",
                    error_message="Job was cancelled before watch registration",
                    checkpoint_data=cp_existing,
                )

            watch_id = uuid.uuid4()
            if is_rewatch and self._wake_inbox is not None:
                self._wake_inbox.record_no_action_silent(
                    job_id=job.job_id,
                    watch_id=watch_id,
                    wake_checks_used=wake_checks_used,
                )

            replaced_watches = await self.cancel_active_watches_for_job_db_locked(
                job.job_id,
                reason=WatchCancelReason.REPLACED.value,
                now=check_now,
            )

            expires_at = check_now + timedelta(seconds=max_wait)
            watch = JobWatch(
                watch_id=watch_id,
                job_id=job.job_id,
                owner_id=job.owner_id,
                host_id=host_id,
                mode=mode,
                signals=signals,
                target=target,
                done_criteria=done_criteria,
                continuation_mode=continuation_mode,
                quiescence_seconds=quiescence,
                max_wait_seconds=max_wait,
                max_wake_checks=max_wake_checks,
                wake_checks_used=wake_checks_used,
                status=WatchStatus.ACTIVE,
                expires_at=expires_at,
                created_at=check_now,
                updated_at=check_now,
            )

            assistant_turn_dict: dict[str, Any] = {
                "sub_goal_index": active_sub_goal_index,
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": [
                    {
                        "id": assistant_call.id,
                        "name": assistant_call.name,
                        "arguments": dict(assistant_call.arguments),
                        **({"extra": dict(assistant_call.extra)} if assistant_call.extra else {}),
                    }
                ],
            }
            tool_turn_dict: dict[str, Any] = {
                "sub_goal_index": active_sub_goal_index,
                "role": "tool",
                "tool_call_id": assistant_call.id,
                "tool_name": "job.wait",
                "status": "ok",
                "content": (
                    f'{{"status":"waiting_watch","watchId":"{watch_id}",'
                    f'"mode":"{mode.value}","quiescenceSeconds":{quiescence}}}'
                ),
            }
            updated_turns = [*active_turns, assistant_turn_dict, tool_turn_dict]

            next_cp: dict[str, Any] = {
                **cp_existing,
                "current_sub_goal_idx": active_sub_goal_index,
                "step_count": job.steps_used + 1,
                "sub_goals": [sg.to_dict() for sg in sub_goals],
                "compacted_history": compacted_history,
                "active_turns": updated_turns,
                "pending_approval_id": None,
                "wake_checks_used": wake_checks_used,
                "max_wake_checks": max_wake_checks,
                "last_quiescence_seconds": quiescence,
                "active_watch_id": str(watch_id),
                "active_watch_continuation_mode": continuation_mode.value,
                "active_watch_done_criteria": done_criteria,
                "last_resumed_watch_id": None,
                "had_watch_in_job": True,
            }
            if steering_notes:
                next_cp["steering_notes"] = steering_notes

            # Save watch + WAITING_WATCH BEFORE sending RPC so fast `watchEvent` never races.
            await self._watch_repo.save(watch)
            waiting_job = latest_before_save.with_state(
                AutonomousJobState.WAITING_WATCH,
                now=check_now,
                steps_used=job.steps_used + 1,
                sub_goals=sub_goals,
                active_sub_goal_index=active_sub_goal_index,
                checkpoint_data=next_cp,
                stop_reason="waiting_watch",
            )
            await self._job_store.save(waiting_job)

        if replaced_watches:
            await self._send_watch_cancels(
                replaced_watches,
                host_id=host_id,
                reason=WatchCancelReason.REPLACED.value,
            )

        ok, active_signals, err_code, err_msg = await self._send_watch_register(
            watch,
            min_wait_seconds=0,
            require_initial_change=(mode != WatchMode.TIMER),
            release_focus=True,
        )

        compensating_cancel_reason: Literal["estop", "owner_cancel"] | None = None
        emit_fail_inbox = False
        fail_code_out = ""
        fail_msg_out = ""
        async with self._lock_for_job(job.job_id):
            latest_watch = await self._watch_repo.get(watch.watch_id)
            latest_job = await self._job_store.get(job.job_id)
            if (
                latest_watch is None
                or latest_watch.status
                in {WatchStatus.CANCELLED, WatchStatus.EXPIRED, WatchStatus.FAILED}
                or latest_job is None
                or latest_job.is_terminal
                or latest_job.state == AutonomousJobState.PAUSED
            ):
                if ok and self._hub.is_online(str(host_id)):
                    compensating_cancel_reason = _cancel_wire_reason(
                        (
                            latest_job is not None
                            and latest_job.state == AutonomousJobState.CANCELLED
                        )
                        or (
                            latest_watch is not None
                            and latest_watch.status == WatchStatus.CANCELLED
                        )
                    )
            elif latest_watch.status == WatchStatus.TRIGGERED and ok:
                return WatchRegistrationOutcome(
                    accepted=True,
                    watch=latest_watch,
                    active_signals=active_signals,
                    checkpoint_data=next_cp,
                )
            elif not ok:
                failed_code = err_code or "watch_unsupported"
                failed_watch = replace(
                    latest_watch,
                    status=WatchStatus.FAILED,
                    last_event_kind=failed_code[:24],
                    updated_at=self._clock(),
                )
                await self._watch_repo.save(failed_watch)
                latest_job_after = await self._job_store.get(job.job_id)
                if (
                    latest_job_after is not None
                    and not latest_job_after.is_terminal
                    and latest_job_after.state != AutonomousJobState.PAUSED
                    and not self._is_cancel_pending(
                        owner_id=latest_job_after.owner_id,
                        host_id=latest_job_after.host_id,
                        job_id=latest_job_after.job_id,
                    )
                ):
                    failed_job = latest_job_after.with_state(
                        AutonomousJobState.FAILED,
                        now=self._clock(),
                        error_message=err_msg or failed_code,
                        stop_reason=failed_code,
                    )
                    await self._job_store.save(failed_job)
                    emit_fail_inbox = self._wake_inbox is not None
                fail_code_out = failed_code
                fail_msg_out = err_msg or failed_code
            else:
                return WatchRegistrationOutcome(
                    accepted=True,
                    watch=latest_watch,
                    active_signals=active_signals,
                    checkpoint_data=next_cp,
                )

        if compensating_cancel_reason is not None:
            with contextlib.suppress(Exception):
                await self._hub.send_request(
                    str(host_id),
                    "xuna/watchCancel",
                    build_watch_cancel_params(
                        reason=compensating_cancel_reason,
                        watch_id=str(watch.watch_id),
                        job_id=str(job.job_id),
                    ),
                    timeout_s=2.0,
                )
            return WatchRegistrationOutcome(
                accepted=False,
                error_code="cancelled",
                error_message="Job or watch cancelled during watch registration",
                checkpoint_data=next_cp,
            )

        if fail_code_out:
            if emit_fail_inbox and self._wake_inbox is not None:
                await self._wake_inbox.emit_item(
                    owner_id=job.owner_id,
                    pet_id=job.pet_id,
                    kind=WakeInboxKind.JOB_FAILED,
                    title=f"Watch failed: {job.title[:60]}",
                    summary=f"Host rejected watch registration ({fail_code_out}).",
                    job_id=job.job_id,
                    host_id=host_id,
                    suggested_actions=(WakeSuggestedAction.DISMISS,),
                    gesture_cue=WatchGestureCue.LOOK_AROUND,
                )
            return WatchRegistrationOutcome(
                accepted=False,
                error_code=fail_code_out,
                error_message=fail_msg_out,
                checkpoint_data=next_cp,
            )

        return WatchRegistrationOutcome(
            accepted=False,
            error_code="cancelled",
            error_message="Job or watch cancelled during watch registration",
            checkpoint_data=next_cp,
        )

    async def register_for_detached_job(
        self,
        *,
        job: AutonomousJob,
        mode: WatchMode,
        signals: tuple[WatchCompositeSignal, ...],
        target: WatchTarget,
        done_criteria: str,
        continuation_mode: ContinuationMode = ContinuationMode.REPORT_AND_ASK,
        quiescence_seconds: int = DEFAULT_QUIESCENCE_SECONDS,
        max_wait_seconds: int = DEFAULT_WATCH_WAIT_SECONDS,
    ) -> WatchRegistrationOutcome:
        """Registers a host watch for a `yieldMs` detached job (`job` starts in `RUNNING`)."""
        if job.host_id is None:
            existing = await self._job_store.get(job.job_id)
            if existing is not None and not existing.is_terminal:
                await self._job_store.save(
                    existing.with_state(
                        AutonomousJobState.FAILED,
                        now=self._clock(),
                        error_message="Detached job requires a host_id",
                        stop_reason="no_host",
                    )
                )
            return WatchRegistrationOutcome(
                accepted=False,
                error_code="no_host",
                error_message="Detached job requires a host_id",
            )
        host_id = job.host_id
        now = self._clock()
        watch_id = uuid.uuid4()
        async with self._lock_for_job(job.job_id):
            existing = await self._job_store.get(job.job_id)
            if existing is not None and (
                existing.is_terminal or existing.state == AutonomousJobState.PAUSED
            ):
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code="cancelled",
                    error_message="Job is no longer active for watch registration",
                )
            if not self._hub.is_online(str(host_id)):
                if existing is not None and not existing.is_terminal:
                    await self._job_store.save(
                        existing.with_state(
                            AutonomousJobState.FAILED,
                            now=now,
                            error_message="Host went offline before watch registration",
                            stop_reason="host_offline",
                        )
                    )
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code="host_offline",
                    error_message="Host went offline before watch registration",
                )
            watch = JobWatch(
                watch_id=watch_id,
                job_id=job.job_id,
                owner_id=job.owner_id,
                host_id=host_id,
                mode=mode,
                signals=signals,
                target=target,
                done_criteria=done_criteria[:240],
                continuation_mode=continuation_mode,
                quiescence_seconds=quiescence_seconds,
                max_wait_seconds=max_wait_seconds,
                max_wake_checks=DEFAULT_MAX_WAKE_CHECKS,
                wake_checks_used=0,
                status=WatchStatus.ACTIVE,
                expires_at=now + timedelta(seconds=max_wait_seconds),
                created_at=now,
                updated_at=now,
            )
            cp = dict(job.checkpoint_data or {})
            cp["active_watch_id"] = str(watch_id)
            cp["active_watch_continuation_mode"] = continuation_mode.value
            cp["active_watch_done_criteria"] = watch.done_criteria
            cp["last_quiescence_seconds"] = quiescence_seconds
            cp["wake_checks_used"] = 0
            cp["max_wake_checks"] = DEFAULT_MAX_WAKE_CHECKS
            cp["had_watch_in_job"] = True

            # Persist WAITING_WATCH job BEFORE watch so `0023` FK holds without pre-save.
            base_for_wait = existing or job
            waiting_job = base_for_wait.with_state(
                AutonomousJobState.WAITING_WATCH,
                now=now,
                checkpoint_data=cp,
                stop_reason="yield_ms_detached",
            )
            await self._job_store.save(waiting_job)
            await self._watch_repo.save(watch)

        ok, active_signals, err_code, err_msg = await self._send_watch_register(
            watch,
            min_wait_seconds=0,
            require_initial_change=False,
            release_focus=True,
        )

        compensating_cancel_reason: Literal["estop", "owner_cancel"] | None = None
        async with self._lock_for_job(job.job_id):
            latest_watch = await self._watch_repo.get(watch.watch_id)
            latest_job = await self._job_store.get(job.job_id)
            if (
                latest_watch is None
                or latest_watch.status
                in {WatchStatus.CANCELLED, WatchStatus.EXPIRED, WatchStatus.FAILED}
                or latest_job is None
                or latest_job.is_terminal
                or latest_job.state == AutonomousJobState.PAUSED
            ):
                if ok and self._hub.is_online(str(host_id)):
                    compensating_cancel_reason = _cancel_wire_reason(
                        (
                            latest_job is not None
                            and latest_job.state == AutonomousJobState.CANCELLED
                        )
                        or (
                            latest_watch is not None
                            and latest_watch.status == WatchStatus.CANCELLED
                        )
                    )
            elif latest_watch.status == WatchStatus.TRIGGERED and ok:
                return WatchRegistrationOutcome(
                    accepted=True,
                    watch=latest_watch,
                    active_signals=active_signals,
                    checkpoint_data=cp,
                )
            elif not ok:
                failed_code = err_code or "watch_unsupported"
                await self._watch_repo.save(
                    replace(
                        latest_watch,
                        status=WatchStatus.FAILED,
                        last_event_kind=failed_code[:24],
                        updated_at=self._clock(),
                    )
                )
                latest_job_after = await self._job_store.get(job.job_id)
                if (
                    latest_job_after is not None
                    and not latest_job_after.is_terminal
                    and latest_job_after.state != AutonomousJobState.PAUSED
                    and not self._is_cancel_pending(
                        owner_id=latest_job_after.owner_id,
                        host_id=latest_job_after.host_id,
                        job_id=latest_job_after.job_id,
                    )
                ):
                    await self._job_store.save(
                        latest_job_after.with_state(
                            AutonomousJobState.FAILED,
                            now=self._clock(),
                            error_message=err_msg or failed_code,
                            stop_reason=failed_code,
                        )
                    )
                return WatchRegistrationOutcome(
                    accepted=False,
                    error_code=failed_code,
                    error_message=err_msg or failed_code,
                )
            else:
                return WatchRegistrationOutcome(
                    accepted=True,
                    watch=latest_watch,
                    active_signals=active_signals,
                    checkpoint_data=cp,
                )

        if compensating_cancel_reason is not None:
            with contextlib.suppress(Exception):
                await self._hub.send_request(
                    str(host_id),
                    "xuna/watchCancel",
                    build_watch_cancel_params(
                        reason=compensating_cancel_reason,
                        watch_id=str(watch.watch_id),
                        job_id=str(job.job_id),
                    ),
                    timeout_s=2.0,
                )
        return WatchRegistrationOutcome(
            accepted=False,
            error_code="cancelled",
            error_message="Job or watch cancelled during watch registration",
        )

    async def on_watch_event(
        self,
        owner_id: uuid.UUID,
        host_id: HostId,
        event: WatchEventParamsModel,
    ) -> None:
        """Handles an inbound `xuna/watchEvent` push notification from a host agent."""
        watch_id = uuid.UUID(event.watchId)
        job_id = uuid.UUID(event.jobId)
        kind = WatchEventKind(event.kind)

        post_lock_timeout: tuple[JobWatch, AutonomousJob | None, bool, str] | None = None
        post_lock_fail: tuple[AutonomousJob, str, str] | None = None
        async with self._lock_for_job(job_id):
            watch = await self._watch_repo.get(watch_id)
            if watch is None or watch.owner_id != owner_id or watch.host_id != host_id:
                return
            if watch.job_id != job_id or watch.status != WatchStatus.ACTIVE:
                return

            job = await self._job_store.get(job_id)
            # Re-fetch watch after `await _job_store.get` in case E-Stop/cancel interleaved
            watch = await self._watch_repo.get(watch_id)
            if (
                watch is None
                or watch.status != WatchStatus.ACTIVE
                or job is None
                or job.is_terminal
                or self._is_cancel_pending(
                    owner_id=job.owner_id, host_id=job.host_id, job_id=job.job_id
                )
            ):
                return

            now = self._clock()
            if kind in {WatchEventKind.SETTLED, WatchEventKind.CHANGED}:
                if job.state != AutonomousJobState.WAITING_WATCH:
                    return
                if watch.wake_checks_used >= watch.max_wake_checks:
                    failed_watch = replace(
                        watch,
                        status=WatchStatus.FAILED,
                        last_event_kind="max_wake_checks_exceeded",
                        updated_at=now,
                    )
                    await self._watch_repo.save(failed_watch)
                    latest_job = await self._job_store.get(job_id)
                    if (
                        latest_job is not None
                        and not latest_job.is_terminal
                        and latest_job.can_transition_to(AutonomousJobState.FAILED)
                        and not self._is_cancel_pending(
                            owner_id=latest_job.owner_id,
                            host_id=latest_job.host_id,
                            job_id=latest_job.job_id,
                        )
                    ):
                        await self._job_store.save(
                            latest_job.with_state(
                                AutonomousJobState.FAILED,
                                now=now,
                                error_message="Maximum wake verification checks reached",
                                stop_reason="max_wake_checks_exceeded",
                            )
                        )
                        post_lock_fail = (
                            latest_job,
                            f"Check limit reached: {latest_job.title[:60]}",
                            (
                                f"Stopped after {watch.wake_checks_used} watch verification "
                                "checks without completion."
                            ),
                        )
                else:
                    new_checks = watch.wake_checks_used + 1
                    latest_watch = await self._watch_repo.get(watch_id)
                    latest_job = await self._job_store.get(job_id)
                    if (
                        latest_watch is None
                        or latest_watch.status != WatchStatus.ACTIVE
                        or latest_job is None
                        or latest_job.is_terminal
                        or latest_job.state != AutonomousJobState.WAITING_WATCH
                        or self._is_cancel_pending(
                            owner_id=latest_job.owner_id,
                            host_id=latest_job.host_id,
                            job_id=latest_job.job_id,
                        )
                    ):
                        return

                    triggered = replace(
                        latest_watch,
                        status=WatchStatus.TRIGGERED,
                        last_event_kind=kind.value,
                        wake_checks_used=new_checks,
                        updated_at=now,
                    )
                    await self._watch_repo.save(triggered)

                    cp = dict(latest_job.checkpoint_data or {})
                    cp["wake_checks_used"] = new_checks
                    cp["last_quiescence_seconds"] = latest_watch.quiescence_seconds
                    cp["active_watch_continuation_mode"] = latest_watch.continuation_mode.value
                    cp["active_watch_done_criteria"] = latest_watch.done_criteria
                    cp["last_resumed_watch_id"] = str(latest_watch.watch_id)
                    cp.pop("active_watch_id", None)

                    turns = list(cp.get("active_turns", []))
                    signals_text = ", ".join(event.observedSignals) or latest_watch.mode.value
                    turns.append(
                        {
                            "sub_goal_index": latest_job.active_sub_goal_index,
                            "role": "user",
                            "content": (
                                f"[Host watch '{kind.value}'] signals={signals_text}, "
                                f"elapsed={event.elapsedSeconds}s, changes={event.changeCount}, "
                                f"digestChanged={event.digestChanged}. "
                                f"Verify real host state against doneCriteria: "
                                f"{latest_watch.done_criteria} "
                                f"(check {new_checks}/{latest_watch.max_wake_checks}). "
                                "Call a real observation tool first; if still running, "
                                "call job__wait."
                            ),
                        }
                    )
                    cp["active_turns"] = turns

                    latest_before_job_save = await self._job_store.get(job_id)
                    if (
                        latest_before_job_save is None
                        or latest_before_job_save.is_terminal
                        or latest_before_job_save.state != AutonomousJobState.WAITING_WATCH
                    ):
                        return
                    updated_job = latest_before_job_save.with_state(
                        latest_before_job_save.state,
                        now=now,
                        checkpoint_data=cp,
                    )
                    await self._job_store.save(updated_job)

                    latest_before_resume = await self._job_store.get(job_id)
                    if (
                        latest_before_resume is None
                        or latest_before_resume.is_terminal
                        or latest_before_resume.state != AutonomousJobState.WAITING_WATCH
                    ):
                        return
                    if self._job_runner is not None:
                        await self._job_runner.resume_job(job_id)
                    else:
                        await self._job_store.save(
                            latest_before_resume.with_state(AutonomousJobState.RUNNING, now=now)
                        )
                    return
            elif kind == WatchEventKind.TIMEOUT:
                timeout_msg = f"Watch timed out after {event.elapsedSeconds}s"
                paused_job, should_emit = await self._expire_watch_db_locked(
                    watch,
                    now=now,
                    stop_reason="watch_timeout",
                    message=timeout_msg,
                )
                post_lock_timeout = (watch, paused_job, should_emit, timeout_msg)
            else:
                # target_lost or error -> fail closed (NEVER COMPLETED)
                err_code = (
                    event.errorCode
                    if kind == WatchEventKind.ERROR and event.errorCode
                    else kind.value
                )
                failed_watch = replace(
                    watch,
                    status=WatchStatus.FAILED,
                    last_event_kind=err_code[:24],
                    updated_at=now,
                )
                await self._watch_repo.save(failed_watch)
                latest_job = await self._job_store.get(job_id)
                if (
                    latest_job is not None
                    and not latest_job.is_terminal
                    and latest_job.can_transition_to(AutonomousJobState.FAILED)
                    and not self._is_cancel_pending(
                        owner_id=latest_job.owner_id,
                        host_id=latest_job.host_id,
                        job_id=latest_job.job_id,
                    )
                ):
                    failed_job = latest_job.with_state(
                        AutonomousJobState.FAILED,
                        now=now,
                        error_message=f"Watch failed: {err_code}",
                        stop_reason=err_code[:64],
                    )
                    await self._job_store.save(failed_job)
                    post_lock_fail = (
                        latest_job,
                        f"Watch stopped: {latest_job.title[:60]}",
                        f"Observation stopped ({err_code}) after {event.elapsedSeconds}s.",
                    )

        if post_lock_timeout is not None:
            t_watch, t_job, t_emit, t_msg = post_lock_timeout
            await self._expire_watch_notify_outside_lock(
                t_watch,
                job=t_job,
                message=t_msg,
                stop_reason="watch_timeout",
                notify_host=False,
                emit_inbox=t_emit,
            )
        elif post_lock_fail is not None and self._wake_inbox is not None:
            f_job, f_title, f_summary = post_lock_fail
            await self._wake_inbox.emit_item(
                owner_id=f_job.owner_id,
                pet_id=f_job.pet_id,
                kind=WakeInboxKind.JOB_FAILED,
                title=f_title,
                summary=f_summary,
                job_id=f_job.job_id,
                host_id=host_id,
                suggested_actions=(WakeSuggestedAction.DISMISS,),
                gesture_cue=WatchGestureCue.LOOK_AROUND,
            )

    async def on_host_ready(self, owner_id: uuid.UUID, host_id: HostId) -> int:
        """Rehydrates unexpired active watches when a host reconnects (`xuna/ready`)."""
        watches = await self._watch_repo.list_active_for_host(host_id, owner_id=owner_id)
        rehydrated = 0
        now = self._clock()
        for watch in watches:
            expired_timeout: tuple[JobWatch, AutonomousJob | None, bool] | None = None
            should_register = False
            remaining = MIN_WATCH_WAIT_SECONDS
            latest: JobWatch | None = None
            async with self._lock_for_job(watch.job_id):
                latest = await self._watch_repo.get(watch.watch_id)
                if latest is None or latest.status != WatchStatus.ACTIVE:
                    continue
                job = await self._job_store.get(latest.job_id)
                if job is None or job.state != AutonomousJobState.WAITING_WATCH:
                    await self._watch_repo.save(
                        replace(latest, status=WatchStatus.CANCELLED, updated_at=now)
                    )
                    continue
                if latest.expires_at <= now:
                    paused_job, should_emit = await self._expire_watch_db_locked(
                        latest,
                        now=now,
                        stop_reason="watch_timeout",
                        message="Watch expired while host was offline",
                    )
                    expired_timeout = (latest, paused_job, should_emit)
                else:
                    remaining = max(
                        MIN_WATCH_WAIT_SECONDS,
                        min(
                            int((latest.expires_at - now).total_seconds()),
                            MAX_WATCH_WAIT_SECONDS,
                        ),
                    )
                    should_register = True

            if expired_timeout is not None:
                t_watch, t_job, t_emit = expired_timeout
                await self._expire_watch_notify_outside_lock(
                    t_watch,
                    job=t_job,
                    message="Watch expired while host was offline",
                    stop_reason="watch_timeout",
                    notify_host=False,
                    emit_inbox=t_emit,
                )
                continue

            if not should_register or latest is None:
                continue

            ok, _sigs, err_code, err_msg = await self._send_watch_register(
                latest,
                min_wait_seconds=0,
                max_wait_override=remaining,
                require_initial_change=False,
                release_focus=True,
            )

            compensating_cancel_reason: Literal["estop", "owner_cancel"] | None = None
            post_lock_fail: tuple[AutonomousJob, str] | None = None
            async with self._lock_for_job(watch.job_id):
                latest_after = await self._watch_repo.get(latest.watch_id)
                job_after = await self._job_store.get(latest.job_id)
                if (
                    latest_after is None
                    or latest_after.status != WatchStatus.ACTIVE
                    or job_after is None
                    or job_after.state != AutonomousJobState.WAITING_WATCH
                ):
                    if ok and self._hub.is_online(str(host_id)):
                        compensating_cancel_reason = _cancel_wire_reason(
                            (
                                job_after is not None
                                and job_after.state == AutonomousJobState.CANCELLED
                            )
                            or (
                                latest_after is not None
                                and latest_after.status == WatchStatus.CANCELLED
                            )
                        )
                elif ok:
                    rehydrated += 1
                else:
                    code = err_code or "watch_unsupported"
                    await self._watch_repo.save(
                        replace(
                            latest_after,
                            status=WatchStatus.FAILED,
                            last_event_kind=code[:24],
                            updated_at=self._clock(),
                        )
                    )
                    job_after2 = await self._job_store.get(latest.job_id)
                    if (
                        job_after2 is not None
                        and not job_after2.is_terminal
                        and job_after2.state == AutonomousJobState.WAITING_WATCH
                        and not self._is_cancel_pending(
                            owner_id=job_after2.owner_id,
                            host_id=job_after2.host_id,
                            job_id=job_after2.job_id,
                        )
                    ):
                        await self._job_store.save(
                            job_after2.with_state(
                                AutonomousJobState.FAILED,
                                now=self._clock(),
                                error_message=err_msg or f"Watch rehydration rejected: {code}",
                                stop_reason=code[:64],
                            )
                        )
                        post_lock_fail = (job_after2, code)

            if compensating_cancel_reason is not None:
                with contextlib.suppress(Exception):
                    await self._hub.send_request(
                        str(host_id),
                        "xuna/watchCancel",
                        build_watch_cancel_params(
                            reason=compensating_cancel_reason,
                            watch_id=str(latest.watch_id),
                            job_id=str(latest.job_id),
                        ),
                        timeout_s=2.0,
                    )
            elif post_lock_fail is not None and self._wake_inbox is not None:
                f_job, f_code = post_lock_fail
                await self._wake_inbox.emit_item(
                    owner_id=f_job.owner_id,
                    pet_id=f_job.pet_id,
                    kind=WakeInboxKind.JOB_FAILED,
                    title=f"Watch failed on reconnect: {f_job.title[:50]}",
                    summary=f"Could not resume watch on host ({f_code}).",
                    job_id=f_job.job_id,
                    host_id=host_id,
                    suggested_actions=(WakeSuggestedAction.DISMISS,),
                    gesture_cue=WatchGestureCue.LOOK_AROUND,
                )
        return rehydrated

    async def extend_watch(
        self,
        *,
        job_id: uuid.UUID,
        owner_id: uuid.UUID,
        additional_seconds: int = 300,
    ) -> JobWatch:
        """Extends a paused/waiting watch TTL up to `900s` and re-registers on host."""
        bounded_wait = max(
            JOB_WAIT_MIN_WAIT_SECONDS, min(additional_seconds, MAX_WATCH_WAIT_SECONDS)
        )
        replaced_watches: list[JobWatch] = []
        notify_replaced = False
        async with self._lock_for_job(job_id):
            job = await self._job_store.get(job_id)
            if job is None or job.owner_id != owner_id:
                raise WatchNotFoundError("Job not found")
            if job.is_terminal:
                raise WatchConflictError("job_terminal", f"Job is terminal ({job.state.value})")
            if job.state not in {
                AutonomousJobState.PAUSED,
                AutonomousJobState.WAITING_WATCH,
            }:
                raise WatchConflictError(
                    "invalid_state", f"Cannot extend watch in state {job.state.value}"
                )
            prev_watch = await self._watch_repo.get_latest_for_job(job_id)
            if prev_watch is None:
                raise WatchConflictError("no_watch", "Job has no watch record to extend")
            if prev_watch.wake_checks_used >= prev_watch.max_wake_checks:
                raise WatchConflictError(
                    "max_wake_checks_reached", "Maximum wake checks reached for this watch"
                )
            if not self._hub.is_online(str(prev_watch.host_id)):
                raise WatchConflictError("host_offline", "Target host is offline")

            now = self._clock()
            notify_replaced = prev_watch.status == WatchStatus.ACTIVE
            replaced_watches = await self.cancel_active_watches_for_job_db_locked(
                job_id,
                reason=WatchCancelReason.REPLACED.value,
                now=now,
            )

            new_watch_id = uuid.uuid4()
            new_watch = JobWatch(
                watch_id=new_watch_id,
                job_id=job.job_id,
                owner_id=owner_id,
                host_id=prev_watch.host_id,
                mode=prev_watch.mode,
                signals=prev_watch.signals,
                target=prev_watch.target,
                done_criteria=prev_watch.done_criteria,
                continuation_mode=prev_watch.continuation_mode,
                quiescence_seconds=prev_watch.quiescence_seconds,
                max_wait_seconds=bounded_wait,
                max_wake_checks=prev_watch.max_wake_checks,
                wake_checks_used=prev_watch.wake_checks_used,
                status=WatchStatus.ACTIVE,
                expires_at=now + timedelta(seconds=bounded_wait),
                created_at=now,
                updated_at=now,
            )
            await self._watch_repo.save(new_watch)
            cp = dict(job.checkpoint_data or {})
            cp["active_watch_id"] = str(new_watch_id)
            updated_job = job.with_state(
                AutonomousJobState.WAITING_WATCH,
                now=now,
                checkpoint_data=cp,
                stop_reason="waiting_watch",
            )
            await self._job_store.save(updated_job)

        if notify_replaced and replaced_watches:
            await self._send_watch_cancels(
                replaced_watches,
                host_id=prev_watch.host_id,
                reason=WatchCancelReason.REPLACED.value,
            )

        ok, _sigs, err_code, err_msg = await self._send_watch_register(
            new_watch,
            min_wait_seconds=0,
            require_initial_change=False,
            release_focus=True,
        )

        compensating_cancel_reason: Literal["estop", "owner_cancel"] | None = None
        async with self._lock_for_job(job_id):
            latest_watch = await self._watch_repo.get(new_watch.watch_id)
            latest_job = await self._job_store.get(job_id)
            if (
                latest_watch is None
                or latest_watch.status
                in {WatchStatus.CANCELLED, WatchStatus.EXPIRED, WatchStatus.FAILED}
                or latest_job is None
                or latest_job.is_terminal
                or latest_job.state == AutonomousJobState.PAUSED
            ):
                if ok and self._hub.is_online(str(new_watch.host_id)):
                    compensating_cancel_reason = _cancel_wire_reason(
                        (
                            latest_job is not None
                            and latest_job.state == AutonomousJobState.CANCELLED
                        )
                        or (
                            latest_watch is not None
                            and latest_watch.status == WatchStatus.CANCELLED
                        )
                    )
            elif not ok:
                code = err_code or "watch_unsupported"
                await self._watch_repo.save(
                    replace(
                        latest_watch,
                        status=WatchStatus.FAILED,
                        last_event_kind=code[:24],
                        updated_at=self._clock(),
                    )
                )
                latest_job_after = await self._job_store.get(job_id)
                if (
                    latest_job_after is not None
                    and not latest_job_after.is_terminal
                    and latest_job_after.state != AutonomousJobState.PAUSED
                    and not self._is_cancel_pending(
                        owner_id=latest_job_after.owner_id,
                        host_id=latest_job_after.host_id,
                        job_id=latest_job_after.job_id,
                    )
                ):
                    await self._job_store.save(
                        latest_job_after.with_state(
                            AutonomousJobState.FAILED,
                            now=self._clock(),
                            error_message=err_msg or code,
                            stop_reason=code[:64],
                        )
                    )
                raise WatchConflictError(code, err_msg or code)
            else:
                return latest_watch

        if compensating_cancel_reason is not None:
            with contextlib.suppress(Exception):
                await self._hub.send_request(
                    str(new_watch.host_id),
                    "xuna/watchCancel",
                    build_watch_cancel_params(
                        reason=compensating_cancel_reason,
                        watch_id=str(new_watch.watch_id),
                        job_id=str(job_id),
                    ),
                    timeout_s=2.0,
                )
        raise WatchConflictError("job_cancelled", "Job or watch cancelled during extend")

    async def wake_now(self, *, job_id: uuid.UUID, owner_id: uuid.UUID) -> AutonomousJob:
        """Cancels any active watch and immediately triggers a wake verification step."""
        cancelled_watches: list[JobWatch] = []
        cancel_host_id: HostId | None = None
        result_job: AutonomousJob | None = None
        async with self._lock_for_job(job_id):
            job = await self._job_store.get(job_id)
            if job is None or job.owner_id != owner_id:
                raise WatchNotFoundError("Job not found")
            if job.is_terminal:
                raise WatchConflictError("job_terminal", f"Job is terminal ({job.state.value})")
            if job.state not in {
                AutonomousJobState.WAITING_WATCH,
                AutonomousJobState.PAUSED,
            }:
                raise WatchConflictError(
                    "invalid_state", f"Cannot wake job in state {job.state.value}"
                )

            now = self._clock()
            latest_watch = await self._watch_repo.get_latest_for_job(job_id)
            if latest_watch is not None and latest_watch.status == WatchStatus.ACTIVE:
                cancel_host_id = latest_watch.host_id
                cancelled_watches = await self.cancel_active_watches_for_job_db_locked(
                    job_id,
                    reason=WatchCancelReason.OWNER_CANCEL.value,
                    now=now,
                )

            cp = dict(job.checkpoint_data or {})
            cp.pop("active_watch_id", None)
            cp.pop("last_quiescence_seconds", None)
            if latest_watch is not None:
                cp["last_resumed_watch_id"] = str(latest_watch.watch_id)
            turns = list(cp.get("active_turns", []))
            turns.append(
                {
                    "sub_goal_index": job.active_sub_goal_index,
                    "role": "user",
                    "content": (
                        "[Owner requested immediate check] Inspect real host state now "
                        "using an observation tool and report the current status."
                    ),
                }
            )
            cp["active_turns"] = turns
            saved = job.with_state(job.state, now=now, checkpoint_data=cp)
            await self._job_store.save(saved)

            if self._job_runner is not None:
                result_job = await self._job_runner.resume_job(job_id)
            else:
                running = saved.with_state(AutonomousJobState.RUNNING, now=now)
                await self._job_store.save(running)
                result_job = running

        if cancelled_watches:
            await self._send_watch_cancels(
                cancelled_watches,
                host_id=cancel_host_id,
                reason=WatchCancelReason.OWNER_CANCEL.value,
            )
        return result_job

    async def get_job_watch_summary(self, *, job_id: uuid.UUID, owner_id: uuid.UUID) -> JobWatch:
        job = await self._job_store.get(job_id)
        if job is None or job.owner_id != owner_id:
            raise WatchNotFoundError("Job not found")
        watch = await self._watch_repo.get_latest_for_job(job_id)
        if watch is None:
            raise WatchNotFoundError("Watch not found for job")
        return watch

    async def cancel_watches_for_job(
        self,
        job_id: uuid.UUID,
        *,
        reason: WatchCancelReason = WatchCancelReason.OWNER_CANCEL,
    ) -> int:
        host_id: HostId | None = None
        cancelled: list[JobWatch] = []
        async with self._lock_for_job(job_id):
            job = await self._job_store.get(job_id)
            host_id = job.host_id if job is not None else None
            cancelled = await self.cancel_active_watches_for_job_db_locked(
                job_id,
                reason=reason.value,
                now=self._clock(),
            )
        if cancelled:
            await self._send_watch_cancels(cancelled, host_id=host_id, reason=reason.value)
        return len(cancelled)

    async def list_active_watch_job_ids_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        host_id: HostId | None = None,
    ) -> set[uuid.UUID]:
        active_watches = await self._watch_repo.list_active_watches(limit=200)
        return {
            w.job_id
            for w in active_watches
            if w.owner_id == owner_id and (host_id is None or w.host_id == host_id)
        }

    async def cancel_active_watches_for_job_db_locked(
        self,
        job_id: uuid.UUID,
        *,
        reason: str = "owner_cancel",
        now: datetime | None = None,
    ) -> list[JobWatch]:
        """Cancels active watches in DB only (caller holds `_lock_for_job(job_id)`)."""
        return await self._watch_repo.cancel_active_for_job(
            job_id,
            reason=reason,
            now=now or self._clock(),
        )

    async def _send_watch_cancels(
        self,
        cancelled: list[JobWatch],
        *,
        host_id: HostId | None,
        reason: str,
    ) -> None:
        if not cancelled:
            return
        hid = str(host_id or cancelled[0].host_id)
        if not self._hub.is_online(hid):
            return
        wire_reason: Literal["estop", "owner_cancel", "replaced", "expired", "job_ended"] = (
            "estop"
            if reason == "estop"
            else (
                "replaced"
                if reason == "replaced"
                else (
                    "expired"
                    if reason == "expired"
                    else ("job_ended" if reason == "job_ended" else "owner_cancel")
                )
            )
        )
        for watch in cancelled:
            with contextlib.suppress(Exception):
                await self._hub.send_request(
                    hid,
                    "xuna/watchCancel",
                    build_watch_cancel_params(
                        reason=wire_reason,
                        watch_id=str(watch.watch_id),
                        job_id=str(watch.job_id),
                    ),
                    timeout_s=2.0,
                )

    async def finish_owner_cancel(
        self,
        owner_id: uuid.UUID,
        *,
        reason: str = "estop",
        host_id: HostId | None = None,
        pre_cancelled: list[JobWatch] | None = None,
        now: datetime | None = None,
    ) -> int:
        """Sweeps remaining owner watches in DB and emits `xuna/watchCancel`."""
        ts = now or self._clock()
        cancelled_by_id: dict[uuid.UUID, JobWatch] = {w.watch_id: w for w in (pre_cancelled or ())}
        remaining = await self._watch_repo.cancel_active_for_owner(
            owner_id,
            host_id=host_id,
            reason=reason,
            now=ts,
        )
        for w in remaining:
            cancelled_by_id[w.watch_id] = w
        cancelled = list(cancelled_by_id.values())

        target_host_ids: set[str] = {str(w.host_id) for w in cancelled}
        if host_id is not None:
            target_host_ids.add(str(host_id))
        else:
            for twin in self._hub.get_twins_for_owner(owner_id):
                target_host_ids.add(str(twin.host_id))

        wire_reason = (
            WatchCancelReason.ESTOP.value
            if reason in {"estop", "owner_app", "owner_host", "owner_board", "owner_ws", "revoked"}
            else WatchCancelReason.OWNER_CANCEL.value
        )
        params = (
            build_watch_cancel_params(
                reason="estop" if wire_reason == "estop" else "owner_cancel",
                watch_id=None,
                job_id=None,
            )
            if wire_reason == "estop"
            else None
        )

        for hid in sorted(target_host_ids):
            if not self._hub.is_online(hid):
                continue
            if params is not None:
                with contextlib.suppress(Exception):
                    await self._hub.send_request(hid, "xuna/watchCancel", params, timeout_s=2.0)
            else:
                for watch in [w for w in cancelled if str(w.host_id) == hid]:
                    with contextlib.suppress(Exception):
                        await self._hub.send_request(
                            hid,
                            "xuna/watchCancel",
                            build_watch_cancel_params(
                                reason="owner_cancel",
                                watch_id=str(watch.watch_id),
                                job_id=str(watch.job_id),
                            ),
                            timeout_s=2.0,
                        )
        return len(cancelled)

    async def cancel_all_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        reason: str = "estop",
        host_id: HostId | None = None,
    ) -> int:
        """Cancels all active watches in DB under per-job locks, then sends `xuna/watchCancel`."""
        scope = (owner_id, host_id)
        self._cancelling_scopes.append(scope)
        try:
            now = self._clock()
            matching_job_ids = await self.list_active_watch_job_ids_for_owner(
                owner_id, host_id=host_id
            )
            active_states = (
                AutonomousJobState.QUEUED,
                AutonomousJobState.RUNNING,
                AutonomousJobState.AWAITING_R3_CHECKPOINT,
                AutonomousJobState.WAITING_WAKE,
                AutonomousJobState.WAITING_WATCH,
                AutonomousJobState.PAUSED,
            )
            active_jobs = await self._job_store.list_for_owner(
                owner_id, limit=100, states=active_states
            )
            for j in active_jobs:
                if host_id is None or j.host_id == host_id:
                    matching_job_ids.add(j.job_id)

            pre_cancelled: list[JobWatch] = []
            for jid in sorted(matching_job_ids, key=str):
                async with self._lock_for_job(jid):
                    pre_cancelled.extend(
                        await self.cancel_active_watches_for_job_db_locked(
                            jid,
                            reason=reason,
                            now=now,
                        )
                    )

            return await self.finish_owner_cancel(
                owner_id,
                reason=reason,
                host_id=host_id,
                pre_cancelled=pre_cancelled,
                now=now,
            )
        finally:
            with contextlib.suppress(ValueError):
                self._cancelling_scopes.remove(scope)

    async def _expire_watch_db_locked(
        self,
        watch: JobWatch,
        *,
        now: datetime,
        stop_reason: str,
        message: str,
    ) -> tuple[AutonomousJob | None, bool]:
        expired_watch = replace(
            watch,
            status=WatchStatus.EXPIRED,
            last_event_kind=WatchEventKind.TIMEOUT.value,
            updated_at=now,
        )
        await self._watch_repo.save(expired_watch)
        job = await self._job_store.get(watch.job_id)
        if (
            job is not None
            and not job.is_terminal
            and job.can_transition_to(AutonomousJobState.PAUSED)
            and not self._is_cancel_pending(
                owner_id=job.owner_id, host_id=job.host_id, job_id=job.job_id
            )
        ):
            paused_job = job.with_state(
                AutonomousJobState.PAUSED,
                now=now,
                error_message=message,
                stop_reason=stop_reason,
            )
            await self._job_store.save(paused_job)
            return paused_job, True
        return job, False

    async def _expire_watch_notify_outside_lock(
        self,
        watch: JobWatch,
        *,
        job: AutonomousJob | None,
        message: str,
        stop_reason: str,
        notify_host: bool,
        emit_inbox: bool,
    ) -> None:
        if notify_host and self._hub.is_online(str(watch.host_id)):
            with contextlib.suppress(Exception):
                await self._hub.send_request(
                    str(watch.host_id),
                    "xuna/watchCancel",
                    build_watch_cancel_params(
                        reason="expired",
                        watch_id=str(watch.watch_id),
                        job_id=str(watch.job_id),
                    ),
                    timeout_s=2.0,
                )
        if emit_inbox and self._wake_inbox is not None:
            title_prefix = "Watch timed out" if job is None else f"Timed out: {job.title[:60]}"
            can_extend = (
                watch.wake_checks_used < watch.max_wake_checks
                and stop_reason != "max_wake_checks_exceeded"
            )
            suggested_actions: tuple[WakeSuggestedAction, ...] = (
                (WakeSuggestedAction.EXTEND_WATCH, WakeSuggestedAction.DISMISS)
                if can_extend
                else (WakeSuggestedAction.DISMISS,)
            )
            await self._wake_inbox.emit_item(
                owner_id=watch.owner_id,
                pet_id=job.pet_id if job is not None else "xuna_pet",
                kind=WakeInboxKind.JOB_TIMEOUT,
                title=title_prefix,
                summary=message,
                job_id=watch.job_id,
                host_id=watch.host_id,
                suggested_actions=suggested_actions,
                gesture_cue=WatchGestureCue.LOOK_AROUND,
            )

    async def _send_watch_register(
        self,
        watch: JobWatch,
        *,
        min_wait_seconds: int,
        require_initial_change: bool,
        release_focus: bool,
        max_wait_override: int | None = None,
    ) -> tuple[bool, tuple[str, ...], str | None, str | None]:
        if not self._hub.is_online(str(watch.host_id)):
            return False, (), "host_offline", "Target host is offline"
        target_model = WatchTargetModel(
            pid=watch.target.pid,
            windowId=watch.target.window_id,
            appId=watch.target.app_id,
            workspaceAlias=watch.target.workspace_alias,
            tabId=watch.target.tab_id,
            mcpServerId=watch.target.mcp_server_id,
            protocolTaskId=watch.target.protocol_task_id,
        )
        signals_wire: list[WatchCompositeSignalWire] | None = (
            [s.value for s in watch.signals] if watch.mode == WatchMode.COMPOSITE else None
        )
        mode_wire: WatchModeWire = watch.mode.value
        effective_max_wait = (
            max_wait_override if max_wait_override is not None else watch.max_wait_seconds
        )
        params = build_watch_register_params(
            watch_id=str(watch.watch_id),
            job_id=str(watch.job_id),
            mode=mode_wire,
            signals=signals_wire,
            target=target_model,
            quiescence_seconds=watch.quiescence_seconds,
            min_wait_seconds=min_wait_seconds,
            max_wait_seconds=effective_max_wait,
            require_initial_change=require_initial_change,
            release_focus=release_focus,
        )
        try:
            raw_res = await self._hub.send_request(
                str(watch.host_id),
                "xuna/watchRegister",
                params,
                timeout_s=self._rpc_timeout_s,
            )
            parsed = parse_watch_register_result(raw_res)
        except Exception as exc:
            err_str = str(exc)
            if "rpc_error:-32601" in err_str:
                return False, (), "watch_unsupported", "Host does not support watchRegister"
            return False, (), "watch_unsupported", err_str[:160]

        if not parsed.accepted:
            return False, (), parsed.errorCode or "watch_unsupported", parsed.errorCode
        return True, tuple(parsed.activeSignals or ()), None, None
