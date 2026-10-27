"""Strict application schemas for projections and memory candidates (CLD-14)."""

import base64
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from xuna_backend.modules.identity.domain.device_proof import (
    PUBLIC_KEY_SIZE,
    SIGNATURE_SIZE,
)
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.domain.entities import CandidateType


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DeviceProofDto(_Strict):
    challengeId: str = Field(min_length=36, max_length=36)
    publicKey: str
    signature: str

    def key_bytes(self) -> bytes:
        raw = base64.b64decode(self.publicKey)
        if len(raw) != PUBLIC_KEY_SIZE:
            raise ValueError(f"public key must be {PUBLIC_KEY_SIZE} bytes")
        return raw

    def signature_bytes(self) -> bytes:
        raw = base64.b64decode(self.signature)
        if len(raw) != SIGNATURE_SIZE:
            raise ValueError(f"signature must be {SIGNATURE_SIZE} bytes")
        return raw


class IngestProjectionRequest(_Strict):
    schemaVersion: Literal[1] = 1
    export: dict[str, Any]
    deviceProof: DeviceProofDto | None = None


class CreateCandidateRequest(_Strict):
    schemaVersion: Literal[1] = 1
    petId: str
    source: str = Field(default="conversation", max_length=32)
    candidateType: CandidateType = CandidateType.STATED_PREFERENCE
    content: dict[str, Any]
    domainId: uuid.UUID | None = None
    relation: str | None = None
    relatedCandidateId: uuid.UUID | None = None
    factKey: int | None = None

    def parsed_pet_id(self) -> PetId:
        return PetId(self.petId)


class PatchCandidateRequest(_Strict):
    schemaVersion: Literal[1] = 1
    action: Literal["confirm", "reject", "commit", "delete", "retag"]
    committedGeneration: int | None = Field(default=None, ge=0, le=0xFFFFFFFF)
    domainId: uuid.UUID | None = None
