import base64
import binascii
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1]


def _b64(value: str, size: int) -> bytes:
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("not base64") from error
    if len(raw) != size:
        raise ValueError(f"must decode to {size} bytes")
    return raw


class ChallengeRequest(_Body):
    petId: str

    @field_validator("petId")
    @classmethod
    def _pet(cls, value: str) -> str:
        return PetId(value).value


class DeviceProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    challengeId: str = Field(min_length=36, max_length=36)
    publicKey: str = Field(max_length=200)
    signature: str = Field(max_length=200)

    @field_validator("publicKey")
    @classmethod
    def _key(cls, value: str) -> str:
        _b64(value, 65)
        return value

    @field_validator("signature")
    @classmethod
    def _sig(cls, value: str) -> str:
        _b64(value, 64)
        return value

    def key_bytes(self) -> bytes:
        return base64.b64decode(self.publicKey)

    def signature_bytes(self) -> bytes:
        return base64.b64decode(self.signature)


class BindPetRequest(_Body):
    petId: str
    origin: PetOrigin
    snapshotGeneration: int = Field(ge=0, le=0xFFFFFFFF)
    deviceProof: DeviceProof

    @field_validator("petId")
    @classmethod
    def _pet(cls, value: str) -> str:
        return PetId(value).value
