import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.accounts.domain.owner import Owner, OwnerClaims
from xuna_backend.modules.accounts.infrastructure.models import OwnerRow


class SqlOwnerRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(self, claims: OwnerClaims) -> Owner:
        await self._session.execute(
            insert(OwnerRow)
            .values(
                owner_id=uuid.uuid4(),
                issuer=claims.issuer,
                subject=claims.subject,
                created_at=datetime.now(UTC),
            )
            .on_conflict_do_nothing(constraint="uq_owners_issuer_subject")
        )
        row = await self._session.scalar(
            select(OwnerRow).where(
                OwnerRow.issuer == claims.issuer, OwnerRow.subject == claims.subject
            )
        )
        if row is None:
            raise RuntimeError("owner upsert returned no row")
        return Owner(owner_id=row.owner_id, issuer=row.issuer, subject=row.subject)
