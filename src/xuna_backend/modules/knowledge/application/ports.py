"""Application ports for knowledge module."""

from __future__ import annotations

import uuid
from typing import Protocol

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.domain.entities import (
    KnowledgeDomain,
    KnowledgeSuggestion,
    KnowledgeSummary,
    SuggestionStatus,
)
from xuna_backend.modules.memory.domain.entities import MemoryCandidate


class KnowledgeRepository(Protocol):
    async def save_domain(self, domain: KnowledgeDomain) -> None: ...

    async def get_domain(self, domain_id: uuid.UUID) -> KnowledgeDomain | None: ...

    async def list_domains(
        self, pet_id: PetId, include_deleted: bool = False
    ) -> list[KnowledgeDomain]: ...

    async def count_non_deleted_domains(self, pet_id: PetId) -> int: ...

    async def get_domain_by_slot(self, pet_id: PetId, slot: int) -> KnowledgeDomain | None: ...

    async def update_domain(self, domain: KnowledgeDomain) -> None: ...

    async def save_suggestion(self, suggestion: KnowledgeSuggestion) -> None: ...

    async def get_suggestion(self, suggestion_id: uuid.UUID) -> KnowledgeSuggestion | None: ...

    async def get_pending_suggestion(self, pet_id: PetId) -> KnowledgeSuggestion | None: ...

    async def list_suggestions(
        self, pet_id: PetId, status: SuggestionStatus | None = None
    ) -> list[KnowledgeSuggestion]: ...

    async def update_suggestion(self, suggestion: KnowledgeSuggestion) -> None: ...

    async def save_summary(self, summary: KnowledgeSummary) -> None: ...

    async def get_summary(self, domain_id: uuid.UUID) -> KnowledgeSummary | None: ...


class KnowledgeCandidatePort(Protocol):
    async def save_candidate(self, candidate: MemoryCandidate) -> None: ...

    async def get_candidate(self, candidate_id: uuid.UUID) -> MemoryCandidate | None: ...

    async def list_candidates_by_domain(
        self, domain_id: uuid.UUID, confirmed_only: bool = True
    ) -> list[MemoryCandidate]: ...

    async def list_unsorted_knowledge(
        self, pet_id: PetId, confirmed_only: bool = True
    ) -> list[MemoryCandidate]: ...

    async def update_domain_for_candidates(
        self, candidate_ids: list[uuid.UUID], new_domain_id: uuid.UUID | None
    ) -> None: ...
