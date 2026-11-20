"""Pydantic request/response schemas for `/v1/hosts` HTTP endpoints (UDC-3 §5.4, UDC-7 S7.1)."""

import base64
import binascii
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from xuna_backend.modules.hosts.domain.crypto import (
    PUBLIC_KEY_SIZE,
    SIGNATURE_SIZE,
    validate_public_key,
)
from xuna_backend.modules.hosts.domain.entities import (
    HostPlatform,
    RiskTier,
    ToolFamily,
)


def _decode_b64_flex(raw: str, expected_len: int) -> bytes:
    padded = raw + "=" * (-len(raw) % 4)
    try:
        if "-" in raw or "_" in raw:
            decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        else:
            decoded = base64.b64decode(padded.encode("ascii"), validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("invalid_base64") from error
    if len(decoded) != expected_len:
        raise ValueError("invalid_key_or_signature_length")
    return decoded


class _StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ClaimHostRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1
    enrollmentCode: str = Field(min_length=20, max_length=28)
    alias: str = Field(min_length=1, max_length=24)
    keyFingerprint: str | None = Field(default=None, min_length=20, max_length=28)


class ConfirmHostRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1


class SelfEnrollHostRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1
    platform: Literal["android", "ios"]
    osVersion: str = Field(min_length=1, max_length=64)
    agentVersion: str = Field(min_length=1, max_length=32)
    publicKey: str = Field(min_length=80, max_length=120)
    alias: str = Field(min_length=1, max_length=24)
    proof: str = Field(min_length=80, max_length=120)
    hostname: str | None = Field(default=None, min_length=1, max_length=32)
    tools: list[dict[str, Any]] | None = Field(default=None, max_length=32)

    def parsed_platform(self) -> HostPlatform:
        return HostPlatform(self.platform)


class UpdateHostManifestRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1
    tools: list[dict[str, Any]] = Field(max_length=32)


class GrantItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal[
        "media",
        "window",
        "app",
        "url",
        "workflow",
        "host",
        "cu",
        "ui",
        "input",
        "timer",
        "clipboard",
        "screen",
        "files",
        "browser",
    ]
    maxRisk: Literal["R0", "R1", "R2", "R3"]
    enabled: bool
    expiresAt: AwareDatetime | None = None
    suspended: bool | None = None

    def parsed_family(self) -> ToolFamily:
        alias_map = {
            "window": ToolFamily.WORKFLOW,
            "timer": ToolFamily.WORKFLOW,
            "clipboard": ToolFamily.APP,
            "browser": ToolFamily.URL,
        }
        if self.family in alias_map:
            return alias_map[self.family]
        return ToolFamily(self.family)

    def parsed_max_risk(self) -> RiskTier:
        return RiskTier(self.maxRisk)


class SetHostGrantsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1] = 1
    grants: list[GrantItemInput] = Field(default_factory=list, max_length=16)
    suspendAll: bool | None = None


class SetHostPolicyRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1
    trustPreset: Literal["strict", "standard", "autopilot"]
    allowApps: list[str] = Field(default_factory=list)
    denyApps: list[str] = Field(default_factory=list)
    allowDomains: list[str] = Field(default_factory=list)
    denyDomains: list[str] = Field(default_factory=list)
    # Read-only field echoed by mobile `HostPolicyModel.toJson()`; ignored on write.
    discoveredApps: list[str] | None = Field(default=None)

    def to_xtp_payload(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schemaVersion,
            "trustPreset": self.trustPreset,
            "allowApps": list(self.allowApps),
            "denyApps": list(self.denyApps),
            "allowDomains": list(self.allowDomains),
            "denyDomains": list(self.denyDomains),
        }


class DevPreflightRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1
    tool: str = Field(min_length=3, max_length=64)
    arguments: dict[str, object] = Field(default_factory=dict)
    sessionId: str | None = Field(default=None, min_length=1, max_length=128)
    deadlineMs: int = Field(default=3_000, ge=100, le=30_000)


class DevInvokeToolRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1
    tool: str = Field(min_length=3, max_length=64)
    arguments: dict[str, object] = Field(default_factory=dict)
    deadlineMs: int = Field(default=10_000, ge=100, le=30_000)
    approve: bool = True
    approvalId: str | None = Field(default=None, min_length=36, max_length=36)
    expiresAt: str | None = Field(default=None, min_length=10, max_length=64)
    scope: Literal["once", "conversation_15m"] | None = None
    signature: str | None = Field(default=None, min_length=80, max_length=120)
    publicKey: str | None = Field(default=None, min_length=80, max_length=120)

    @model_validator(mode="after")
    def _validate_approval_fields(self) -> "DevInvokeToolRequest":
        if self.approvalId is not None:
            try:
                parsed_id = uuid.UUID(self.approvalId)
            except ValueError as error:
                raise ValueError("approvalId must be a valid UUID") from error
            if str(parsed_id) != self.approvalId.lower():
                raise ValueError("approvalId must be a canonical UUID")
        if self.expiresAt is not None:
            try:
                parsed_dt = datetime.fromisoformat(self.expiresAt)
            except ValueError as error:
                raise ValueError("expiresAt must be an ISO-8601 timestamp") from error
            if parsed_dt.tzinfo is None:
                raise ValueError("expiresAt must include a timezone offset")
        if self.signature is not None:
            if self.approvalId is None or self.expiresAt is None:
                raise ValueError("signature requires both approvalId and expiresAt")
            _decode_b64_flex(self.signature, SIGNATURE_SIZE)
        if self.publicKey is not None:
            raw_pub = _decode_b64_flex(self.publicKey, PUBLIC_KEY_SIZE)
            validate_public_key(raw_pub)
        return self
