"""Pydantic schemas for the cloud suggestions API (CLD-30, CLD-31, H1-H8)."""

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CloudSuggestionResponse(_Strict):
    suggestionId: str
    petId: str
    brainId: str
    kind: str
    target: str
    delta: float
    bounds: dict[str, float]
    provenance: str
    evidenceRef: str | None = None
    status: str
    expiresAt: str
    createdAt: str
    appliedAt: str | None = None


class ListSuggestionsResponse(_Strict):
    items: list[CloudSuggestionResponse]


class ConsolidateSuggestionsResponse(_Strict):
    items: list[CloudSuggestionResponse]
    count: int


class PatchSuggestionRequest(_Strict):
    action: str = Field(pattern="^(apply|reject|revert)$")
