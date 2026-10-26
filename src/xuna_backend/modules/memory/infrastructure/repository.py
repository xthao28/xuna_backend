"""SQLAlchemy implementation of MemoryRepository (CLD-14)."""

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
    Projection,
)
from xuna_backend.modules.memory.infrastructure.models import (
    MemoryCandidateRow,
    ProjectionRow,
)


def _to_projection(row: ProjectionRow) -> Projection:
    return Projection(
        projection_id=row.projection_id,
        pet_id=PetId(row.pet_id),
        owner_id=row.owner_id,
        snapshot_generation=row.snapshot_generation,
        schema_version=row.schema_version,
        payload_sha256=row.payload_sha256,
        payload=dict(row.payload),
        device_attested=row.device_attested,
        created_at=row.created_at,
    )


def _to_candidate(row: MemoryCandidateRow) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=row.candidate_id,
        owner_id=row.owner_id,
        pet_id=PetId(row.pet_id),
        source=row.source,
        candidate_type=CandidateType(row.candidate_type),
        content=dict(row.content),
        status=CandidateStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        confirmed_at=row.confirmed_at,
        committed_generation=row.committed_generation,
    )


class SqlMemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_projection(self, projection: Projection) -> None:
        self._session.add(
            ProjectionRow(
                projection_id=projection.projection_id,
                pet_id=projection.pet_id.value,
                owner_id=projection.owner_id,
                snapshot_generation=projection.snapshot_generation,
                schema_version=projection.schema_version,
                payload_sha256=projection.payload_sha256,
                payload=projection.payload,
                device_attested=projection.device_attested,
                created_at=projection.created_at,
            )
        )
        await self._session.flush()

    async def get_projection(self, pet_id: PetId, generation: int) -> Projection | None:
        row = await self._session.scalar(
            select(ProjectionRow).where(
                (ProjectionRow.pet_id == pet_id.value)
                & (ProjectionRow.snapshot_generation == generation)
            )
        )
        return _to_projection(row) if row is not None else None

    async def get_latest_projection(self, pet_id: PetId) -> Projection | None:
        row = await self._session.scalar(
            select(ProjectionRow)
            .where(ProjectionRow.pet_id == pet_id.value)
            .order_by(ProjectionRow.created_at.desc(), ProjectionRow.snapshot_generation.desc())
            .limit(1)
        )
        return _to_projection(row) if row is not None else None

    async def list_projections(self, pet_id: PetId, limit: int = 50) -> list[Projection]:
        result = await self._session.scalars(
            select(ProjectionRow)
            .where(ProjectionRow.pet_id == pet_id.value)
            .order_by(ProjectionRow.created_at.desc(), ProjectionRow.snapshot_generation.desc())
            .limit(limit)
        )
        return [_to_projection(row) for row in result.all()]

    async def add_candidate(self, candidate: MemoryCandidate) -> None:
        self._session.add(
            MemoryCandidateRow(
                candidate_id=candidate.candidate_id,
                owner_id=candidate.owner_id,
                pet_id=candidate.pet_id.value,
                source=candidate.source,
                candidate_type=candidate.candidate_type.value,
                content=candidate.content,
                status=candidate.status.value,
                created_at=candidate.created_at,
                updated_at=candidate.updated_at,
                confirmed_at=candidate.confirmed_at,
                committed_generation=candidate.committed_generation,
            )
        )
        await self._session.flush()

    async def get_candidate(self, candidate_id: uuid.UUID) -> MemoryCandidate | None:
        row = await self._session.scalar(
            select(MemoryCandidateRow).where(MemoryCandidateRow.candidate_id == candidate_id)
        )
        return _to_candidate(row) if row is not None else None

    async def update_candidate(self, candidate: MemoryCandidate) -> None:
        await self._session.execute(
            update(MemoryCandidateRow)
            .where(MemoryCandidateRow.candidate_id == candidate.candidate_id)
            .values(
                status=candidate.status.value,
                content=candidate.content,
                updated_at=candidate.updated_at,
                confirmed_at=candidate.confirmed_at,
                committed_generation=candidate.committed_generation,
            )
        )
        await self._session.flush()

    async def list_candidates(
        self,
        pet_id: PetId,
        status: CandidateStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        query = select(MemoryCandidateRow).where(MemoryCandidateRow.pet_id == pet_id.value)
        if status is not None:
            query = query.where(MemoryCandidateRow.status == status.value)
        query = query.order_by(MemoryCandidateRow.created_at.desc()).limit(limit)
        result = await self._session.scalars(query)
        return [_to_candidate(row) for row in result.all()]
