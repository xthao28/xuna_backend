"""Ports for memory projections and candidate persistence (CLD-14)."""

import uuid
from typing import Protocol

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    MemoryCandidate,
    Projection,
)


class MemoryRepository(Protocol):
    async def add_projection(self, projection: Projection) -> None: ...

    async def get_projection(self, pet_id: PetId, generation: int) -> Projection | None: ...

    async def get_latest_projection(self, pet_id: PetId) -> Projection | None: ...

    async def list_projections(self, pet_id: PetId, limit: int = 50) -> list[Projection]: ...

    async def add_candidate(self, candidate: MemoryCandidate) -> None: ...

    async def get_candidate(self, candidate_id: uuid.UUID) -> MemoryCandidate | None: ...

    async def update_candidate(self, candidate: MemoryCandidate) -> None: ...

    async def list_candidates(
        self,
        pet_id: PetId,
        status: CandidateStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]: ...
