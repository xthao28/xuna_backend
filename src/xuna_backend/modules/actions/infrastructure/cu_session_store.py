"""PostgreSQL `ComputerUseSessionStore` (UDC-17 S17.6, table `cu_sessions`).

Every call opens its own short session and commits, so a long-lived chat stream never holds
a connection. Counters change with single `UPDATE ... RETURNING` statements, and a partial
unique index keeps one running session per owner across workers.

The last UI tree (`cached_nodes`) and draft text (`typed_since_commit`) are never written to
the database. They live in this process only, attached by session id to every state this
store returns, and are dropped when the session ends here. After a restart or on another
worker they start empty; `typed_count` is stored so the next commit card can say that text
was typed.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.actions.application.computer_use import ComputerUseSessionState
from xuna_backend.modules.actions.infrastructure.models import CuSessionModel
from xuna_backend.modules.hosts.domain.entities import HostId

# Ended rows are kept this long so a repeated stop still answers `alreadyEnded`.
_ENDED_ROW_RETENTION = timedelta(days=1)
# Bounds the process-local scratch if sessions end on another worker.
_MAX_SCRATCH_ENTRIES = 1000


@dataclass
class _Scratch:
    cached_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    typed_since_commit: list[str] = field(default_factory=list)


class SqlComputerUseSessionStore:
    def __init__(self, sessions: Callable[[], AsyncSession]) -> None:
        self._sessions = sessions
        self._scratch: dict[str, _Scratch] = {}

    def _scratch_for(self, session_id: str) -> _Scratch:
        scratch = self._scratch.get(session_id)
        if scratch is None:
            if len(self._scratch) >= _MAX_SCRATCH_ENTRIES:
                self._scratch.pop(next(iter(self._scratch)))
            scratch = _Scratch()
            self._scratch[session_id] = scratch
        return scratch

    def _to_state(self, row: CuSessionModel) -> ComputerUseSessionState:
        if row.active:
            scratch = self._scratch_for(row.session_id)
        else:
            self._scratch.pop(row.session_id, None)
            scratch = _Scratch()
        return ComputerUseSessionState(
            session_id=row.session_id,
            owner_id=row.owner_id,
            host_id=HostId(row.host_id),
            goal=row.goal,
            allowed_apps=tuple(row.apps),
            allowed_domains=tuple(row.domains),
            max_steps=row.max_steps,
            max_minutes=row.max_minutes,
            started_at_utc=row.started_at,
            last_heartbeat_at=row.last_heartbeat_at,
            steps_used=row.steps_used,
            cached_nodes=scratch.cached_nodes,
            typed_since_commit=scratch.typed_since_commit,
            typed_count=row.typed_count,
            owner_prompts=row.owner_prompts,
            active=row.active,
            stop_reason=row.stop_reason,
            target_alias=row.target_alias,
        )

    async def get(self, session_id: str) -> ComputerUseSessionState | None:
        async with self._sessions() as s:
            row = await s.get(CuSessionModel, session_id)
            return self._to_state(row) if row is not None else None

    async def get_active_for_owner(self, owner_id: uuid.UUID) -> ComputerUseSessionState | None:
        async with self._sessions() as s:
            stmt = select(CuSessionModel).where(
                CuSessionModel.owner_id == owner_id, CuSessionModel.active.is_(True)
            )
            row = (await s.execute(stmt)).scalars().first()
            return self._to_state(row) if row is not None else None

    async def start(self, state: ComputerUseSessionState, now: datetime) -> None:
        async with self._sessions() as s:
            replaced = await s.execute(
                update(CuSessionModel)
                .where(CuSessionModel.owner_id == state.owner_id, CuSessionModel.active.is_(True))
                .values(active=False, stop_reason="replaced", ended_at=now, goal="")
                .returning(CuSessionModel.session_id)
            )
            for sid in replaced.scalars().all():
                self._scratch.pop(sid, None)
            await s.execute(
                delete(CuSessionModel).where(
                    CuSessionModel.owner_id == state.owner_id,
                    CuSessionModel.active.is_(False),
                    CuSessionModel.ended_at < now - _ENDED_ROW_RETENTION,
                )
            )
            s.add(
                CuSessionModel(
                    session_id=state.session_id,
                    owner_id=state.owner_id,
                    host_id=state.host_id.value,
                    target_alias=state.target_alias[:32],
                    goal=state.goal[:200],
                    apps=list(state.allowed_apps),
                    domains=list(state.allowed_domains),
                    max_steps=state.max_steps,
                    max_minutes=state.max_minutes,
                    steps_used=state.steps_used,
                    typed_count=state.typed_count,
                    owner_prompts=state.owner_prompts,
                    started_at=state.started_at_utc,
                    last_heartbeat_at=state.last_heartbeat_at,
                    active=True,
                    stop_reason=None,
                    ended_at=None,
                )
            )
            await s.commit()
        scratch = self._scratch_for(state.session_id)
        scratch.cached_nodes = state.cached_nodes
        scratch.typed_since_commit = state.typed_since_commit

    async def mark_ended(
        self, session_id: str, reason: str, now: datetime
    ) -> ComputerUseSessionState | None:
        async with self._sessions() as s:
            await s.execute(
                update(CuSessionModel)
                .where(CuSessionModel.session_id == session_id, CuSessionModel.active.is_(True))
                .values(active=False, stop_reason=reason[:64], ended_at=now, goal="", typed_count=0)
            )
            await s.commit()
            row = await s.get(CuSessionModel, session_id)
            self._scratch.pop(session_id, None)
            return self._to_state(row) if row is not None else None

    async def stop_all_for_owner(self, owner_id: uuid.UUID, reason: str, now: datetime) -> int:
        async with self._sessions() as s:
            result = await s.execute(
                update(CuSessionModel)
                .where(CuSessionModel.owner_id == owner_id, CuSessionModel.active.is_(True))
                .values(active=False, stop_reason=reason[:64], ended_at=now, goal="", typed_count=0)
                .returning(CuSessionModel.session_id)
            )
            ended = list(result.scalars().all())
            await s.commit()
        for sid in ended:
            self._scratch.pop(sid, None)
        return len(ended)

    async def record_step(self, session_id: str) -> int | None:
        async with self._sessions() as s:
            result = await s.execute(
                update(CuSessionModel)
                .where(CuSessionModel.session_id == session_id, CuSessionModel.active.is_(True))
                .values(steps_used=CuSessionModel.steps_used + 1)
                .returning(CuSessionModel.steps_used)
            )
            steps = result.scalars().first()
            await s.commit()
            return steps

    async def set_scope(
        self, session_id: str, apps: tuple[str, ...], domains: tuple[str, ...]
    ) -> None:
        async with self._sessions() as s:
            await s.execute(
                update(CuSessionModel)
                .where(CuSessionModel.session_id == session_id, CuSessionModel.active.is_(True))
                .values(apps=list(apps), domains=list(domains))
            )
            await s.commit()

    async def touch_heartbeat(self, session_id: str, owner_id: uuid.UUID, now: datetime) -> bool:
        async with self._sessions() as s:
            result = await s.execute(
                update(CuSessionModel)
                .where(
                    CuSessionModel.session_id == session_id,
                    CuSessionModel.owner_id == owner_id,
                    CuSessionModel.active.is_(True),
                )
                .values(last_heartbeat_at=now)
                .returning(CuSessionModel.session_id)
            )
            touched = result.scalars().first() is not None
            await s.commit()
            return touched

    async def set_typed_count(self, session_id: str, count: int) -> None:
        async with self._sessions() as s:
            await s.execute(
                update(CuSessionModel)
                .where(CuSessionModel.session_id == session_id, CuSessionModel.active.is_(True))
                .values(typed_count=max(count, 0))
            )
            await s.commit()

    async def record_prompt(self, session_id: str) -> int | None:
        async with self._sessions() as s:
            result = await s.execute(
                update(CuSessionModel)
                .where(CuSessionModel.session_id == session_id, CuSessionModel.active.is_(True))
                .values(owner_prompts=CuSessionModel.owner_prompts + 1)
                .returning(CuSessionModel.owner_prompts)
            )
            prompts = result.scalars().first()
            await s.commit()
            return prompts
