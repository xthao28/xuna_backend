"""Strict typed view of a verified `xuna.export.v1` payload.

Validation only; nothing here stores device data (CLD-14+ needs gate c).
Unknown fields are rejected so silent schema drift is impossible.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin

Action = Literal["idle", "happy", "sleep", "curious"]
Context = Literal["quiet", "play"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AppInfo(_Strict):
    version: str = Field(max_length=64)


class Capabilities(_Strict):
    statedPreference: bool
    timeSync: bool
    identity: bool


class Identity(_Strict):
    petId: str
    origin: PetOrigin = Field(strict=False)
    snapshotGeneration: int = Field(ge=0, le=0xFFFFFFFF)

    @field_validator("petId")
    @classmethod
    def _pet_id(cls, value: str) -> str:
        return PetId(value).value


class Life(_Strict):
    activeSeconds: int = Field(ge=0, le=0xFFFFFFFF)
    feedbackCount: int = Field(ge=0, le=0xFFFFFFFF)
    decisionId: int = Field(ge=0, le=0xFFFFFFFF)
    context: Context
    action: Action
    quietChoice: Action
    playChoice: Action
    canFeedback: bool
    memoryCount: int = Field(ge=0, le=255)


class Memory(_Strict):
    # "greeting" and "reaction" (owner pet / shoo on the device) since the
    # 2026-10-19 reaction learning (D6 revised); older exports stay valid.
    context: Literal["quiet", "play", "greeting"]
    action: Action
    approved: bool
    source: Literal["button", "text", "reaction"]


Q8 = Field(ge=0, le=0xFFFF)


class Explanation(_Strict):
    interaction: Literal["quiet", "play", "greeting"]
    dayPart: Literal["unknown", "morning", "afternoon", "evening", "night"]
    action: Action
    reason: Literal["unavailable", "highest_estimate", "exploration", "no_evidence", "told"]
    fine: bool
    rated: int = Field(ge=0, le=255)
    probabilityQ8: int = Q8
    effectiveMassQ8: int = Q8
    positiveQ8: int = Q8
    negativeQ8: int = Q8
    statedPositiveQ8: int = Q8
    statedNegativeQ8: int = Q8


class ExportPayloadV1(_Strict):
    exportedAtUtc: str
    app: AppInfo
    capabilities: Capabilities
    identity: Identity
    life: Life
    memories: list[Memory] = Field(max_length=8)
    explanation: Explanation | None
