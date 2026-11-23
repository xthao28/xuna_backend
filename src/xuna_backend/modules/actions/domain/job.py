"""Domain entity and state models for `AutonomousJob` (Wave D).

Pure domain layer: stdlib-only, framework-free, no external or database dependencies.
"""

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from xuna_backend.modules.actions.domain.watch import (
    WakeInboxKind,
    WakeSuggestedAction,
    WatchGestureCue,
)
from xuna_backend.modules.hosts.domain.entities import HostId


class AutonomousJobState(StrEnum):
    """Lifecycle states of an autonomous job."""

    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_R3_CHECKPOINT = "awaiting_r3_checkpoint"
    WAITING_WAKE = "waiting_wake"
    WAITING_WATCH = "waiting_watch"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubGoalStatus(StrEnum):
    """Lifecycle status of a discrete sub-goal within an execution plan."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


_VALID_TRANSITIONS: dict[AutonomousJobState, set[AutonomousJobState]] = {
    AutonomousJobState.QUEUED: {
        AutonomousJobState.RUNNING,
        AutonomousJobState.CANCELLED,
    },
    AutonomousJobState.RUNNING: {
        AutonomousJobState.AWAITING_R3_CHECKPOINT,
        AutonomousJobState.WAITING_WAKE,
        AutonomousJobState.WAITING_WATCH,
        AutonomousJobState.PAUSED,
        AutonomousJobState.COMPLETED,
        AutonomousJobState.FAILED,
        AutonomousJobState.CANCELLED,
    },
    AutonomousJobState.AWAITING_R3_CHECKPOINT: {
        AutonomousJobState.RUNNING,
        AutonomousJobState.PAUSED,
        AutonomousJobState.CANCELLED,
        AutonomousJobState.FAILED,
    },
    AutonomousJobState.WAITING_WAKE: {
        AutonomousJobState.RUNNING,
        AutonomousJobState.PAUSED,
        AutonomousJobState.CANCELLED,
        AutonomousJobState.FAILED,
    },
    AutonomousJobState.WAITING_WATCH: {
        AutonomousJobState.RUNNING,
        AutonomousJobState.PAUSED,
        AutonomousJobState.CANCELLED,
        AutonomousJobState.FAILED,
    },
    AutonomousJobState.PAUSED: {
        AutonomousJobState.RUNNING,
        AutonomousJobState.WAITING_WATCH,
        AutonomousJobState.CANCELLED,
    },
    AutonomousJobState.COMPLETED: set(),
    AutonomousJobState.FAILED: set(),
    AutonomousJobState.CANCELLED: set(),
}


@dataclass(frozen=True, slots=True)
class SubGoal:
    """A discrete phase or milestone within an autonomous execution plan."""

    id: str
    title: str
    description: str
    status: SubGoalStatus = SubGoalStatus.PENDING
    tool_call_count: int = 0
    result_summary: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "toolCallCount": self.tool_call_count,
            "resultSummary": self.result_summary,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubGoal":
        started_at = (
            datetime.fromisoformat(data["startedAt"])
            if data.get("startedAt")
            else (datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None)
        )
        completed_at = (
            datetime.fromisoformat(data["completedAt"])
            if data.get("completedAt")
            else (
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            )
        )
        status_val = data.get("status", SubGoalStatus.PENDING)
        status = SubGoalStatus(status_val) if isinstance(status_val, str) else status_val
        tool_call_count = int(data.get("toolCallCount", data.get("tool_call_count", 0)))
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            status=status,
            tool_call_count=tool_call_count,
            result_summary=data.get("resultSummary") or data.get("result_summary"),
            started_at=started_at,
            completed_at=completed_at,
        )


@dataclass(frozen=True, slots=True)
class PendingWakeInboxItem:
    """Deferred Wake Inbox item emitted strictly after job state is persisted."""

    owner_id: uuid.UUID
    pet_id: str
    kind: WakeInboxKind
    title: str
    summary: str
    job_id: uuid.UUID | None = None
    host_id: HostId | None = None
    suggested_actions: tuple[WakeSuggestedAction, ...] = ()
    gesture_cue: WatchGestureCue | None = None


@dataclass(frozen=True, slots=True)
class JobStepOutcome:
    """Outcome of executing a single autonomous step."""

    kind: Literal[
        "progress",
        "await_r3",
        "waiting_wake",
        "waiting_watch",
        "paused",
        "completed",
        "failed",
    ]
    sub_goals: tuple[SubGoal, ...] | None = None
    active_sub_goal_index: int | None = None
    checkpoint_data: dict[str, Any] | None = None
    pending_approval_id: uuid.UUID | None = None
    error_message: str | None = None
    stop_reason: str | None = None
    pending_wake_inbox: PendingWakeInboxItem | None = None


@dataclass(frozen=True, slots=True)
class AutonomousJob:
    """An autonomous multi-step job executing outside any single HTTP SSE request."""

    job_id: uuid.UUID
    owner_id: uuid.UUID
    pet_id: str
    host_id: HostId | None
    session_id: str | None
    title: str
    goal: str
    state: AutonomousJobState = AutonomousJobState.QUEUED
    sub_goals: tuple[SubGoal, ...] = ()
    active_sub_goal_index: int = 0
    max_steps: int = 50
    steps_used: int = 0
    pending_approval_id: uuid.UUID | None = None
    checkpoint_data: dict[str, Any] | None = None
    error_message: str | None = None
    stop_reason: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        """True if the job is active or waiting (not in a terminal state)."""
        return self.state in {
            AutonomousJobState.QUEUED,
            AutonomousJobState.RUNNING,
            AutonomousJobState.AWAITING_R3_CHECKPOINT,
            AutonomousJobState.WAITING_WAKE,
            AutonomousJobState.WAITING_WATCH,
            AutonomousJobState.PAUSED,
        }

    @property
    def is_terminal(self) -> bool:
        """True if the job has ended in a terminal state."""
        return self.state in {
            AutonomousJobState.COMPLETED,
            AutonomousJobState.FAILED,
            AutonomousJobState.CANCELLED,
        }

    def can_transition_to(self, target_state: AutonomousJobState) -> bool:
        """Returns True if the transition from current state to target_state is allowed."""
        if self.state == target_state:
            return True
        return target_state in _VALID_TRANSITIONS.get(self.state, set())

    def validate_transition(self, target_state: AutonomousJobState) -> None:
        """Raises ValueError if transition to target_state is invalid."""
        if not self.can_transition_to(target_state):
            raise ValueError(
                f"Invalid job state transition: {self.state.value} -> {target_state.value}"
            )

    def with_state(
        self,
        new_state: AutonomousJobState,
        *,
        now: datetime | None = None,
        pending_approval_id: uuid.UUID | None = None,
        checkpoint_data: dict[str, Any] | None = None,
        error_message: str | None = None,
        stop_reason: str | None = None,
        steps_used: int | None = None,
        active_sub_goal_index: int | None = None,
        sub_goals: tuple[SubGoal, ...] | None = None,
    ) -> "AutonomousJob":
        """Returns a copy of the job with updated state, validating transition."""
        self.validate_transition(new_state)
        now_dt = now or datetime.now(UTC)
        updates: dict[str, Any] = {
            "state": new_state,
            "updated_at": now_dt,
        }
        if new_state == AutonomousJobState.RUNNING and self.started_at is None:
            updates["started_at"] = now_dt
        if new_state in {
            AutonomousJobState.COMPLETED,
            AutonomousJobState.FAILED,
            AutonomousJobState.CANCELLED,
        }:
            updates["completed_at"] = now_dt

        if (
            pending_approval_id is not None
            or new_state != AutonomousJobState.AWAITING_R3_CHECKPOINT
        ):
            updates["pending_approval_id"] = pending_approval_id
        if checkpoint_data is not None:
            updates["checkpoint_data"] = checkpoint_data
        if error_message is not None:
            updates["error_message"] = error_message
        if stop_reason is not None:
            updates["stop_reason"] = stop_reason
        if steps_used is not None:
            updates["steps_used"] = steps_used
        if active_sub_goal_index is not None:
            updates["active_sub_goal_index"] = active_sub_goal_index
        if sub_goals is not None:
            updates["sub_goals"] = sub_goals

        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        """Serializes domain entity to camelCase dictionary for API responses."""
        return {
            "jobId": str(self.job_id),
            "ownerId": str(self.owner_id),
            "petId": self.pet_id,
            "hostId": str(self.host_id) if self.host_id is not None else None,
            "sessionId": self.session_id,
            "title": self.title,
            "goal": self.goal,
            "state": self.state.value,
            "subGoals": [sg.to_dict() for sg in self.sub_goals],
            "activeSubGoalIndex": self.active_sub_goal_index,
            "maxSteps": self.max_steps,
            "stepsUsed": self.steps_used,
            "pendingApprovalId": (
                str(self.pending_approval_id) if self.pending_approval_id is not None else None
            ),
            "checkpointData": self.checkpoint_data,
            "errorMessage": self.error_message,
            "stopReason": self.stop_reason,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }
