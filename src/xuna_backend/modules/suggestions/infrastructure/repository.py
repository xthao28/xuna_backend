"""SQL implementation of SuggestionRepository (CLD-30, CLD-31, H1-H8)."""

import uuid
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.suggestions.domain.entities import (
    CloudSuggestion,
    SuggestionKind,
    SuggestionProvenance,
    SuggestionStatus,
)
from xuna_backend.modules.suggestions.infrastructure.models import CloudSuggestionRow


class SqlSuggestionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, suggestion: CloudSuggestion) -> None:
        row = CloudSuggestionRow(
            suggestion_id=suggestion.suggestion_id,
            pet_id=suggestion.pet_id.value,
            brain_id=suggestion.brain_id,
            kind=suggestion.kind.value,
            target=suggestion.target,
            delta=suggestion.delta,
            bounds=suggestion.bounds,
            provenance=suggestion.provenance.value,
            evidence_ref=suggestion.evidence_ref,
            status=suggestion.status.value,
            expires_at=suggestion.expires_at,
            created_at=suggestion.created_at,
            applied_at=suggestion.applied_at,
        )
        self._session.add(row)
        await self._session.flush()

    async def get_by_id(self, suggestion_id: uuid.UUID) -> CloudSuggestion | None:
        result = await self._session.execute(
            select(CloudSuggestionRow).where(CloudSuggestionRow.suggestion_id == suggestion_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def list_for_pet(
        self,
        pet_id: PetId,
        status: SuggestionStatus | None = None,
        kind: SuggestionKind | None = None,
    ) -> list[CloudSuggestion]:
        stmt = (
            select(CloudSuggestionRow)
            .where(CloudSuggestionRow.pet_id == pet_id.value)
            .order_by(CloudSuggestionRow.created_at.desc())
        )
        if status is not None:
            stmt = stmt.where(CloudSuggestionRow.status == status.value)
        if kind is not None:
            stmt = stmt.where(CloudSuggestionRow.kind == kind.value)

        result = await self._session.execute(stmt)
        return [self._to_entity(r) for r in result.scalars().all()]

    async def update_status(
        self,
        suggestion_id: uuid.UUID,
        status: SuggestionStatus,
        applied_at: datetime | None = None,
    ) -> CloudSuggestion | None:
        values: dict[str, object] = {"status": status.value}
        if applied_at is not None:
            values["applied_at"] = applied_at

        stmt = (
            update(CloudSuggestionRow)
            .where(CloudSuggestionRow.suggestion_id == suggestion_id)
            .values(**values)
            .returning(CloudSuggestionRow)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def delete(self, suggestion_id: uuid.UUID) -> bool:
        stmt = (
            delete(CloudSuggestionRow)
            .where(CloudSuggestionRow.suggestion_id == suggestion_id)
            .returning(CloudSuggestionRow.suggestion_id)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    def _to_entity(self, row: CloudSuggestionRow) -> CloudSuggestion:
        created_at = (
            row.created_at
            if row.created_at.tzinfo is not None
            else row.created_at.replace(tzinfo=UTC)
        )
        expires_at = (
            row.expires_at
            if row.expires_at.tzinfo is not None
            else row.expires_at.replace(tzinfo=UTC)
        )
        applied_at = None
        if row.applied_at is not None:
            applied_at = (
                row.applied_at
                if row.applied_at.tzinfo is not None
                else row.applied_at.replace(tzinfo=UTC)
            )

        raw_bounds = cast(dict[str, float | int | str], row.bounds)
        bounds = {str(k): float(v) for k, v in raw_bounds.items()}

        return CloudSuggestion(
            suggestion_id=row.suggestion_id,
            pet_id=PetId(row.pet_id),
            brain_id=row.brain_id,
            kind=SuggestionKind(row.kind),
            target=row.target,
            delta=float(row.delta),
            bounds=bounds,
            provenance=SuggestionProvenance(row.provenance),
            evidence_ref=row.evidence_ref,
            status=SuggestionStatus(row.status),
            expires_at=expires_at,
            created_at=created_at,
            applied_at=applied_at,
        )
