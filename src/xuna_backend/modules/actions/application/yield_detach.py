"""`yieldMs` auto-detach service: moves slow interactive waits to `WAITING_WATCH` jobs."""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from xuna_backend.modules.actions.application.ports import ComputerUseSessionStore
from xuna_backend.modules.actions.application.watch_coordinator import WatchCoordinator
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    SubGoal,
    SubGoalStatus,
)
from xuna_backend.modules.actions.domain.ports import AutonomousJobRepository
from xuna_backend.modules.actions.domain.watch import (
    DEFAULT_WATCH_WAIT_SECONDS,
    ContinuationMode,
    TurnDetachedEvent,
)
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.application.ports import HostSessionHub
from xuna_backend.modules.hosts.domain.entities import HostId

DEFAULT_YIELD_MS = 12_000


@dataclass(frozen=True, slots=True)
class YieldDetachRequest:
    """Inputs captured when an interactive turn exceeds `yieldMs`."""

    owner_id: uuid.UUID
    pet_id: str
    host_id: HostId | None
    goal: str
    tool_calls: tuple[ToolCall, ...]
    elapsed_ms: int
    locale: Locale = Locale.VI
    last_action_summary: str | None = None
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class YieldDetachResult:
    """Outcome of attempting to detach an interactive wait into a background job."""

    detached: bool
    event: TurnDetachedEvent | None = None
    notice_text: str = ""
    error_code: str | None = None


class YieldDetachService:
    """Detaches slow interactive tool waits (`> yieldMs`) into `WAITING_WATCH` jobs."""

    def __init__(
        self,
        *,
        job_store: AutonomousJobRepository,
        watch_coordinator: WatchCoordinator,
        hub: HostSessionHub,
        cu_session_store: ComputerUseSessionStore | None = None,
        clock_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._job_store = job_store
        self._watch_coordinator = watch_coordinator
        self._hub = hub
        self._cu_store = cu_session_store
        self._clock = clock_utc

    async def try_detach(self, req: YieldDetachRequest) -> YieldDetachResult:
        """Attempts to create a durable `WAITING_WATCH` job and release the chat lane."""
        host_id = req.host_id
        if host_id is None and self._cu_store is not None:
            active_any = await self._cu_store.get_active_for_owner(req.owner_id)
            if active_any is not None:
                host_id = active_any.host_id
        if host_id is None:
            focus = self._hub.get_focus_state(req.owner_id)
            if focus is not None:
                host_id = focus.focused_host_id
        if host_id is None:
            return YieldDetachResult(detached=False, error_code="no_host")
        if not self._hub.is_online(str(host_id)):
            return YieldDetachResult(detached=False, error_code="host_offline")

        active_cu = (
            await self._cu_store.get_active_for_owner(req.owner_id, host_id=host_id)
            if self._cu_store is not None
            else None
        )
        session_id = active_cu.session_id if active_cu is not None else None

        mode, signals, target, bind_err = await self._watch_coordinator.infer_watch_for_detach(
            owner_id=req.owner_id,
            host_id=host_id,
            session_id=session_id,
        )
        if bind_err is not None or target is None:
            return YieldDetachResult(detached=False, error_code=bind_err or "target_not_found")

        now = self._clock()
        job_id = uuid.uuid4()
        goal_text = req.goal.strip() or "Background task continuation"
        title = goal_text[:64]
        done_criteria = (req.last_action_summary or goal_text)[:240]
        sg = SubGoal(
            id="sg_detached_1",
            title=title[:64],
            description=goal_text[:240],
            status=SubGoalStatus.IN_PROGRESS,
            tool_call_count=0,
            started_at=now,
        )
        cp: dict[str, Any] = {
            "current_sub_goal_idx": 0,
            "step_count": 0,
            "sub_goals": [sg.to_dict()],
            "compacted_history": [],
            "active_turns": [
                {
                    "sub_goal_index": 0,
                    "role": "user",
                    "content": (f"Detached interactive task after {req.elapsed_ms}ms: {goal_text}"),
                }
            ],
            "detached_from_request_id": req.request_id[:64],
        }
        job = AutonomousJob(
            job_id=job_id,
            owner_id=req.owner_id,
            pet_id=req.pet_id[:64] or "xuna_pet",
            host_id=host_id,
            session_id=session_id,
            title=title,
            goal=goal_text,
            state=AutonomousJobState.RUNNING,
            sub_goals=(sg,),
            active_sub_goal_index=0,
            max_steps=50,
            steps_used=0,
            checkpoint_data=cp,
            created_at=now,
            updated_at=now,
            started_at=now,
        )
        reg = await self._watch_coordinator.register_for_detached_job(
            job=job,
            mode=mode,
            signals=signals,
            target=target,
            done_criteria=done_criteria,
            continuation_mode=ContinuationMode.REPORT_AND_ASK,
            max_wait_seconds=DEFAULT_WATCH_WAIT_SECONDS,
        )
        if not reg.accepted or reg.watch is None:
            existing = await self._job_store.get(job_id)
            if existing is not None and not existing.is_terminal:
                fail_code = reg.error_code or "watch_unsupported"
                await self._job_store.save(
                    existing.with_state(
                        AutonomousJobState.FAILED,
                        now=self._clock(),
                        error_message=reg.error_message or fail_code,
                        stop_reason=fail_code[:64],
                    )
                )
            return YieldDetachResult(
                detached=False,
                error_code=reg.error_code or "watch_unsupported",
            )

        notice_text = (
            "Tác vụ này cần thêm thời gian, mình đã chuyển sang chế độ theo dõi nền và "
            "sẽ báo lại bạn khi có kết quả."
            if req.locale == Locale.VI
            else "This task is taking longer, so I moved it to background watch and will "
            "notify your Wake Inbox once it settles."
        )
        event = TurnDetachedEvent(
            job_id=str(job_id),
            watch_id=str(reg.watch.watch_id),
            host_id=str(host_id),
            title=title,
            state=AutonomousJobState.WAITING_WATCH.value,
            reason="yield_ms",
        )
        return YieldDetachResult(
            detached=True,
            event=event,
            notice_text=notice_text,
        )
