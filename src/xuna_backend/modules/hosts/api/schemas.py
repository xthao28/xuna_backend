"""Pydantic request/response schemas for `/v1/hosts` HTTP endpoints (UDC-3 §5.4)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from xuna_backend.modules.hosts.domain.entities import RiskTier, ToolFamily


class _StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ClaimHostRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1
    enrollmentCode: str = Field(min_length=20, max_length=28)
    alias: str = Field(min_length=1, max_length=24)
    keyFingerprint: str | None = Field(default=None, min_length=20, max_length=28)


class ConfirmHostRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1


class GrantItemInput(_StrictSchema):
    family: Literal[
        "media",
        "window",
        "app",
        "url",
        "timer",
        "clipboard",
        "screen",
        "files",
        "browser",
    ]
    maxRisk: Literal["R0", "R1", "R2", "R3"]
    enabled: bool

    def parsed_family(self) -> ToolFamily:
        return ToolFamily(self.family)

    def parsed_max_risk(self) -> RiskTier:
        return RiskTier(self.maxRisk)


class SetHostGrantsRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1
    grants: list[GrantItemInput] = Field(min_length=1, max_length=9)


class DevInvokeToolRequest(_StrictSchema):
    schemaVersion: Literal[1] = 1
    tool: str = Field(min_length=3, max_length=64)
    arguments: dict[str, object] = Field(default_factory=dict)
    deadlineMs: int = Field(default=10_000, ge=100, le=30_000)
    approve: bool = True
