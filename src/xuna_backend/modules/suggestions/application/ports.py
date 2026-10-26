"""Application ports for suggestions repository."""

import uuid
from datetime import datetime
from typing import Protocol

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.suggestions.domain.entities import (
    CloudSuggestion,
    SuggestionKind,
    SuggestionStatus,
)


class SuggestionRepository(Protocol):
    async def save(self, suggestion: CloudSuggestion) -> None: ...

    async def get_by_id(self, suggestion_id: uuid.UUID) -> CloudSuggestion | None: ...

    async def list_for_pet(
        self,
        pet_id: PetId,
        status: SuggestionStatus | None = None,
        kind: SuggestionKind | None = None,
    ) -> list[CloudSuggestion]: ...

    async def update_status(
        self,
        suggestion_id: uuid.UUID,
        status: SuggestionStatus,
        applied_at: datetime | None = None,
    ) -> CloudSuggestion | None: ...

    async def delete(self, suggestion_id: uuid.UUID) -> bool: ...
