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
        owner, _created = await self.get_or_create_with_status(claims)
        return owner

    async def get_or_create_with_status(self, claims: OwnerClaims) -> tuple[Owner, bool]:
        row = await self._session.scalar(
            select(OwnerRow).where(
                OwnerRow.issuer == claims.issuer, OwnerRow.subject == claims.subject
            )
        )
        if row is not None:
            return Owner(owner_id=row.owner_id, issuer=row.issuer, subject=row.subject), False
        inserted = await self._session.scalar(
            insert(OwnerRow)
            .values(
                owner_id=uuid.uuid4(),
                issuer=claims.issuer,
                subject=claims.subject,
                created_at=datetime.now(UTC),
            )
            .on_conflict_do_nothing(constraint="uq_owners_issuer_subject")
            .returning(OwnerRow)
        )
        if inserted is not None:
            return (
                Owner(
                    owner_id=inserted.owner_id,
                    issuer=inserted.issuer,
                    subject=inserted.subject,
                ),
                True,
            )
        row = await self._session.scalar(
            select(OwnerRow).where(
                OwnerRow.issuer == claims.issuer, OwnerRow.subject == claims.subject
            )
        )
        if row is None:
            raise RuntimeError("owner upsert returned no row")
        return Owner(owner_id=row.owner_id, issuer=row.issuer, subject=row.subject), False
