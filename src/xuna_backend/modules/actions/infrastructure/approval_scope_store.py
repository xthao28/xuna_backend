"""PostgreSQL / SQL-backed `ApprovalScopeStore` (Wave A R2, table `approval_scopes`).

Persists 15-minute `conversation_15m` R2 approval scopes so that grants survive
server restarts and work reliably across multiple workers.
"""

import uuid
from collections.abc import Callable
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.actions.domain.entities import CONVERSATION_SCOPE_TTL
from xuna_backend.modules.actions.infrastructure.models import ApprovalScopeModel
from xuna_backend.modules.hosts.domain.entities import HostId


class SqlApprovalScopeStore:
    def __init__(self, sessions: Callable[[], AsyncSession]) -> None:
        self._sessions = sessions

    async def has_valid_scope(
        self,
        owner_id: uuid.UUID,
        pet_id: str,
        host_id: HostId,
        tool: str,
        now: datetime,
    ) -> bool:
        host_uuid = host_id if isinstance(host_id, uuid.UUID) else uuid.UUID(str(host_id))
        owner_uuid = owner_id if isinstance(owner_id, uuid.UUID) else uuid.UUID(str(owner_id))
        async with self._sessions() as s:
            stmt = select(ApprovalScopeModel.expires_at).where(
                ApprovalScopeModel.owner_id == owner_uuid,
                ApprovalScopeModel.pet_id == pet_id,
                ApprovalScopeModel.host_id == host_uuid,
                ApprovalScopeModel.tool == tool,
                ApprovalScopeModel.expires_at > now,
            )
            row = (await s.execute(stmt)).scalar_one_or_none()
            return row is not None

    async def grant_scope(
        self,
        owner_id: uuid.UUID,
        pet_id: str,
        host_id: HostId,
        tool: str,
        now: datetime,
    ) -> datetime:
        host_uuid = host_id if isinstance(host_id, uuid.UUID) else uuid.UUID(str(host_id))
        owner_uuid = owner_id if isinstance(owner_id, uuid.UUID) else uuid.UUID(str(owner_id))
        expires_at = now + CONVERSATION_SCOPE_TTL
        async with self._sessions() as s:
            bind = s.bind
            dialect_name = bind.dialect.name if bind is not None else "postgresql"
            if dialect_name == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                sqlite_stmt = (
                    sqlite_insert(ApprovalScopeModel)
                    .values(
                        owner_id=owner_uuid,
                        pet_id=pet_id,
                        host_id=host_uuid,
                        tool=tool,
                        expires_at=expires_at,
                        created_at=now,
                    )
                    .on_conflict_do_update(
                        index_elements=["owner_id", "pet_id", "host_id", "tool"],
                        set_={"expires_at": expires_at, "created_at": now},
                    )
                )
                await s.execute(sqlite_stmt)
            else:
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                pg_stmt = (
                    pg_insert(ApprovalScopeModel)
                    .values(
                        owner_id=owner_uuid,
                        pet_id=pet_id,
                        host_id=host_uuid,
                        tool=tool,
                        expires_at=expires_at,
                        created_at=now,
                    )
                    .on_conflict_do_update(
                        index_elements=["owner_id", "pet_id", "host_id", "tool"],
                        set_={"expires_at": expires_at, "created_at": now},
                    )
                )
                await s.execute(pg_stmt)
            await s.commit()
        return expires_at

    async def clear_owner(self, owner_id: uuid.UUID, host_id: HostId | None = None) -> int:
        owner_uuid = owner_id if isinstance(owner_id, uuid.UUID) else uuid.UUID(str(owner_id))
        async with self._sessions() as s:
            stmt = delete(ApprovalScopeModel).where(ApprovalScopeModel.owner_id == owner_uuid)
            if host_id is not None:
                host_uuid = host_id if isinstance(host_id, uuid.UUID) else uuid.UUID(str(host_id))
                stmt = stmt.where(ApprovalScopeModel.host_id == host_uuid)
            res = await s.execute(stmt)
            await s.commit()
            return res.rowcount or 0
