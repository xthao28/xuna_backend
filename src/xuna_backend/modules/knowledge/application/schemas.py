"""Application schemas for Knowledge Domains."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreateDomainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1, max_length=40)
    description: str = Field(default="", max_length=160)
    device_slot: int = Field(ge=0, le=4, alias="deviceSlot")


class DomainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    domain_id: uuid.UUID = Field(alias="domainId")
    pet_id: str = Field(alias="petId")
    name: str
    description: str
    device_slot: int | None = Field(default=None, alias="deviceSlot")
    status: str
    origin: str
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")


class UpdateDomainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action: Literal[
        "device_ack",
        "rename",
        "describe",
        "rest",
        "activate",
        "delete",
    ]
    name: str | None = Field(default=None, max_length=40)
    description: str | None = Field(default=None, max_length=160)
    generation: int | None = None
    keep_knowledge: bool = Field(default=True, alias="keepKnowledge")


class TeachKnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str = Field(min_length=1, max_length=280)
    domain_id: uuid.UUID | None = Field(default=None, alias="domainId")


class TeachKnowledgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    candidate_id: uuid.UUID = Field(alias="candidateId")
    fact_key: int = Field(alias="factKey")
    text: str
    status: str
    proposed_domain_id: uuid.UUID | None = Field(default=None, alias="proposedDomainId")
    relation: str
    related_candidate_id: uuid.UUID | None = Field(default=None, alias="relatedCandidateId")


class KnowledgeSuggestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    suggestion_id: uuid.UUID = Field(alias="suggestionId")
    pet_id: str = Field(alias="petId")
    kind: str
    proposed_name: str = Field(alias="proposedName")
    proposed_description: str = Field(alias="proposedDescription")
    member_candidate_ids: list[uuid.UUID] = Field(alias="memberCandidateIds")
    target_domain_id: uuid.UUID | None = Field(default=None, alias="targetDomainId")
    status: str
    created_at: str = Field(alias="createdAt")


class DecideSuggestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action: Literal["accept", "dismiss"]
    name: str | None = Field(default=None, max_length=40)
    description: str | None = Field(default=None, max_length=160)
    device_slot: int | None = Field(default=None, ge=0, le=4, alias="deviceSlot")
