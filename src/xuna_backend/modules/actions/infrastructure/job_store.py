"""Durable and in-memory persistence stores for `AutonomousJob` (Wave D M1).

Implements `AutonomousJobRepository`.
"""

import uuid
from collections.abc import Callable

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    SubGoal,
)
from xuna_backend.modules.actions.domain.ports import AutonomousJobRepository
from xuna_backend.modules.actions.infrastructure.models import AutonomousJobModel
from xuna_backend.modules.hosts.domain.entities import HostId

_ACTIVE_STATES = (
    AutonomousJobState.QUEUED.value,
    AutonomousJobState.RUNNING.value,
    AutonomousJobState.AWAITING_R3_CHECKPOINT.value,
    AutonomousJobState.PAUSED.value,
)


def _row_to_entity(row: AutonomousJobModel) -> AutonomousJob:
    sub_goals = tuple(
        SubGoal.from_dict(item) if isinstance(item, dict) else item
        for item in (row.sub_goals or [])
    )
    return AutonomousJob(
        job_id=row.job_id,
        owner_id=row.owner_id,
        pet_id=row.pet_id,
        host_id=HostId(row.host_id) if row.host_id is not None else None,
        session_id=row.session_id,
        title=row.title,
        goal=row.goal,
        state=AutonomousJobState(row.state),
        sub_goals=sub_goals,
        active_sub_goal_index=row.active_sub_goal_index,
        max_steps=row.max_steps,
        steps_used=row.steps_used,
        pending_approval_id=row.pending_approval_id,
        checkpoint_data=row.checkpoint_data,
        error_message=row.error_message,
        stop_reason=row.stop_reason,
        created_at=row.created_at,
        updated_at=row.updated_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
    )


class InMemoryAutonomousJobStore(AutonomousJobRepository):
    """In-memory implementation of `AutonomousJobRepository` for testing."""

    def __init__(self) -> None:
        self._jobs: dict[uuid.UUID, AutonomousJob] = {}

    async def save(self, job: AutonomousJob) -> None:
        self._jobs[job.job_id] = job

    async def get(self, job_id: uuid.UUID) -> AutonomousJob | None:
        return self._jobs.get(job_id)

    async def get_active_for_owner(self, owner_id: uuid.UUID) -> AutonomousJob | None:
        active = [
            j for j in self._jobs.values()
            if j.owner_id == owner_id and j.is_active
        ]
        if not active:
            return None
        active.sort(key=lambda j: j.updated_at, reverse=True)
        return active[0]

    async def list_for_owner(
        self,
        owner_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        states: tuple[AutonomousJobState, ...] | None = None,
    ) -> list[AutonomousJob]:
        matches = [
            j for j in self._jobs.values()
            if j.owner_id == owner_id
            and (states is None or j.state in states)
        ]
        matches.sort(key=lambda j: j.created_at, reverse=True)
        return matches[offset : offset + limit]

    async def count_for_owner(
        self,
        owner_id: uuid.UUID,
        states: tuple[AutonomousJobState, ...] | None = None,
    ) -> int:
        return len([
            j for j in self._jobs.values()
            if j.owner_id == owner_id
            and (states is None or j.state in states)
        ])


class SqlAutonomousJobStore(AutonomousJobRepository):
    """SQLAlchemy / PostgreSQL implementation of `AutonomousJobRepository`."""

    def __init__(self, sessions: Callable[[], AsyncSession]) -> None:
        self._sessions = sessions

    async def save(self, job: AutonomousJob) -> None:
        async with self._sessions() as s:
            existing = await s.get(AutonomousJobModel, job.job_id)
            sub_goals_data = [sg.to_dict() for sg in job.sub_goals]
            host_uuid = job.host_id.value if isinstance(job.host_id, HostId) else job.host_id

            if existing is not None:
                existing.pet_id = job.pet_id
                existing.host_id = host_uuid
                existing.session_id = job.session_id
                existing.title = job.title[:128]
                existing.goal = job.goal
                existing.state = job.state.value
                existing.sub_goals = sub_goals_data
                existing.active_sub_goal_index = job.active_sub_goal_index
                existing.max_steps = job.max_steps
                existing.steps_used = job.steps_used
                existing.pending_approval_id = job.pending_approval_id
                existing.checkpoint_data = job.checkpoint_data
                existing.error_message = job.error_message
                existing.stop_reason = job.stop_reason
                existing.updated_at = job.updated_at
                existing.started_at = job.started_at
                existing.completed_at = job.completed_at
            else:
                s.add(
                    AutonomousJobModel(
                        job_id=job.job_id,
                        owner_id=job.owner_id,
                        pet_id=job.pet_id,
                        host_id=host_uuid,
                        session_id=job.session_id,
                        title=job.title[:128],
                        goal=job.goal,
                        state=job.state.value,
                        sub_goals=sub_goals_data,
                        active_sub_goal_index=job.active_sub_goal_index,
                        max_steps=job.max_steps,
                        steps_used=job.steps_used,
                        pending_approval_id=job.pending_approval_id,
                        checkpoint_data=job.checkpoint_data,
                        error_message=job.error_message,
                        stop_reason=job.stop_reason,
                        created_at=job.created_at,
                        updated_at=job.updated_at,
                        started_at=job.started_at,
                        completed_at=job.completed_at,
                    )
                )
            await s.commit()

    async def get(self, job_id: uuid.UUID) -> AutonomousJob | None:
        async with self._sessions() as s:
            row = await s.get(AutonomousJobModel, job_id)
            return _row_to_entity(row) if row is not None else None

    async def get_active_for_owner(self, owner_id: uuid.UUID) -> AutonomousJob | None:
        async with self._sessions() as s:
            stmt = (
                select(AutonomousJobModel)
                .where(
                    AutonomousJobModel.owner_id == owner_id,
                    AutonomousJobModel.state.in_(_ACTIVE_STATES),
                )
                .order_by(AutonomousJobModel.updated_at.desc())
                .limit(1)
            )
            row = (await s.execute(stmt)).scalars().first()
            return _row_to_entity(row) if row is not None else None

    async def list_for_owner(
        self,
        owner_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        states: tuple[AutonomousJobState, ...] | None = None,
    ) -> list[AutonomousJob]:
        async with self._sessions() as s:
            stmt = select(AutonomousJobModel).where(AutonomousJobModel.owner_id == owner_id)
            if states:
                stmt = stmt.where(AutonomousJobModel.state.in_([st.value for st in states]))
            stmt = stmt.order_by(AutonomousJobModel.created_at.desc()).offset(offset).limit(limit)
            rows = (await s.execute(stmt)).scalars().all()
            return [_row_to_entity(r) for r in rows]

    async def count_for_owner(
        self,
        owner_id: uuid.UUID,
        states: tuple[AutonomousJobState, ...] | None = None,
    ) -> int:
        async with self._sessions() as s:
            stmt = select(sa.func.count(AutonomousJobModel.job_id)).where(
                AutonomousJobModel.owner_id == owner_id
            )
            if states:
                stmt = stmt.where(AutonomousJobModel.state.in_([st.value for st in states]))
            count = (await s.execute(stmt)).scalar_one()
            return int(count)


# Compatibility alias
SqlAlchemyAutonomousJobRepository = SqlAutonomousJobStore
