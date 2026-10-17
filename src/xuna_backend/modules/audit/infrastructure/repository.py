import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.audit.infrastructure.models import AuditRow


class SqlAuditLog:
    """Insert-only. There is deliberately no update or delete method."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(self, event: AuditEvent) -> None:
        self._session.add(
            AuditRow(
                event_id=event.event_id,
                owner_id=event.owner_id,
                action=event.action.value,
                subject=event.subject,
                request_id=event.request_id,
                metadata_=dict(event.metadata),
                occurred_at=event.occurred_at,
            )
        )
        await self._session.flush()

    async def recent(self, owner_id: uuid.UUID, limit: int) -> list[AuditEvent]:
        rows = await self._session.scalars(
            select(AuditRow)
            .where(AuditRow.owner_id == owner_id)
            .order_by(AuditRow.occurred_at.desc(), AuditRow.event_id)
            .limit(limit)
        )
        return [
            AuditEvent(
                event_id=row.event_id,
                owner_id=row.owner_id,
                action=AuditAction(row.action),
                subject=row.subject,
                request_id=row.request_id,
                metadata=row.metadata_,
                occurred_at=row.occurred_at,
            )
            for row in rows
        ]
