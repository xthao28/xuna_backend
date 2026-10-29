import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.consent.domain.consent import U0, ConsentState
from xuna_backend.modules.consent.infrastructure.models import ConsentRow

POLICY_VERSION = "d23-2026-10-15"


class SqlConsentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def current(self, owner_id: uuid.UUID) -> ConsentState:
        row = await self._session.scalar(
            select(ConsentRow)
            .where(ConsentRow.owner_id == owner_id)
            .order_by(ConsentRow.version.desc())
            .limit(1)
        )
        if row is None:
            return U0
        return ConsentState(
            aggregate_metrics=row.aggregate_metrics,
            experience_events=row.experience_events,
            conversation_text=row.conversation_text,
            conversation_voice=row.conversation_voice,
            owner_documents=row.owner_documents,
            version=row.version,
            updated_at=row.recorded_at,
        )

    async def append(self, owner_id: uuid.UUID, state: ConsentState) -> bool:
        result = await self._session.execute(
            insert(ConsentRow)
            .values(
                owner_id=owner_id,
                version=state.version,
                aggregate_metrics=state.aggregate_metrics,
                experience_events=state.experience_events,
                conversation_text=state.conversation_text,
                conversation_voice=state.conversation_voice,
                owner_documents=state.owner_documents,
                recorded_at=state.updated_at,
                policy=POLICY_VERSION,
            )
            .on_conflict_do_nothing()
            .returning(ConsentRow.version)
        )
        return result.scalar() is not None
