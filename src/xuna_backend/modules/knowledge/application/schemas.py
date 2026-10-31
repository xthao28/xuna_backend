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
    pack_id: str | None = Field(default=None, alias="packId", max_length=32)


class DomainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    domain_id: uuid.UUID = Field(alias="domainId")
    pet_id: str = Field(alias="petId")
    name: str
    description: str
    device_slot: int | None = Field(default=None, alias="deviceSlot")
    status: str
    origin: str
    pack_id: str | None = Field(default=None, alias="packId")
    expert_brief: str = Field(default="", alias="expertBrief")
    safety_tier: str = Field(default="general", alias="safetyTier")
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
        "set_brief",
        "apply_pack",
    ]
    name: str | None = Field(default=None, max_length=40)
    description: str | None = Field(default=None, max_length=160)
    expert_brief: str | None = Field(default=None, alias="expertBrief", max_length=1500)
    pack_id: str | None = Field(default=None, alias="packId", max_length=32)
    generation: int | None = None
    keep_knowledge: bool = Field(default=True, alias="keepKnowledge")


class KnowledgePackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    pack_id: str = Field(alias="packId")
    name: str
    description: str
    safety_tier: str = Field(alias="safetyTier")


class DraftBriefRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = Field(alias="schemaVersion")


class DraftBriefResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    draft: str


class TeachKnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    text: str = Field(min_length=1, max_length=280)
    domain_id: uuid.UUID | None = Field(default=None, alias="domainId", strict=False)
    kind: Literal["convention", "host_default"] = Field(default="convention")
    family: Literal["media", "app", "url", "workflow"] | None = Field(default=None)
    host_alias: str | None = Field(default=None, alias="hostAlias", min_length=1, max_length=32)


class ProposeKnowledgeCaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schemaVersion")
    domain_id: uuid.UUID | None = Field(default=None, alias="domainId", strict=False)
    question: str = Field(min_length=1, max_length=8000)
    answer: str = Field(min_length=1, max_length=12000)


class TeachKnowledgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    candidate_id: uuid.UUID = Field(alias="candidateId")
    fact_key: int = Field(alias="factKey")
    text: str
    kind: str = "convention"
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


class UploadDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schemaVersion")
    title: str = Field(min_length=1, max_length=120)
    domain_id: uuid.UUID | None = Field(default=None, alias="domainId", strict=False)
    media_type: Literal["text/markdown", "text/plain", "application/pdf"] = Field(alias="mediaType")
    data: str = Field(min_length=1, max_length=2_800_000)


class KnowledgeDocumentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    document_id: uuid.UUID = Field(alias="documentId")
    pet_id: str = Field(alias="petId")
    domain_id: uuid.UUID | None = Field(default=None, alias="domainId")
    title: str
    media_type: str = Field(alias="mediaType")
    byte_size: int = Field(alias="byteSize")
    sha256: str
    status: str
    chunk_count: int = Field(alias="chunkCount")
    candidate_id: uuid.UUID | None = Field(default=None, alias="candidateId")
    fact_key: int | None = Field(default=None, alias="factKey")
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")
