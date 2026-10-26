"""Pydantic schemas for the backup and tombstone API (CLD-20)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EncryptedDataSchema(_Strict):
    algorithm: str = Field(default="aes-256-gcm", max_length=32)
    keyId: str = Field(max_length=64)
    nonce: str = Field(max_length=64)
    ciphertext: str
    tag: str = Field(max_length=64)


class CreateBackupRequest(_Strict):
    schemaVersion: int = Field(default=1, ge=1, le=1)
    snapshotGeneration: int = Field(ge=0, le=0xFFFFFFFF)
    export: dict[str, Any] | None = None
    encryptedData: EncryptedDataSchema | None = None
    payloadSha256: str | None = Field(default=None, min_length=64, max_length=64)


class BackupManifestResponse(_Strict):
    backupId: str
    petId: str
    brainId: str
    snapshotGeneration: int
    schemaVersion: int
    payloadSha256: str
    encryptedData: EncryptedDataSchema
    createdAt: str


class ListBackupsResponse(_Strict):
    items: list[BackupManifestResponse]


class DeviceProofSchema(_Strict):
    challengeId: str = Field(max_length=64)
    publicKey: str = Field(max_length=128)
    signature: str = Field(max_length=128)


class RestoreBackupRequest(_Strict):
    schemaVersion: int = Field(default=1, ge=1, le=1)
    deviceProof: DeviceProofSchema | None = None


class RestoreBackupResponse(_Strict):
    petId: str
    brainId: str
    deviceId: str
    snapshotGeneration: int
    restoredPayload: dict[str, Any]
    tombstonesApplied: int


class CreateTombstoneRequest(_Strict):
    schemaVersion: int = Field(default=1, ge=1, le=1)
    itemType: str = Field(min_length=1, max_length=32)
    itemId: str = Field(min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=255)


class TombstoneResponse(_Strict):
    tombstoneId: str
    petId: str
    itemType: str
    itemId: str
    deletedAt: str
    reason: str | None = None


class ListTombstonesResponse(_Strict):
    items: list[TombstoneResponse]


class ForgetAllResponse(_Strict):
    petId: str
    newBrainId: str
    tombstoneId: str
