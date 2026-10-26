"""Pydantic schemas for Direct Device Transport (CLD-40, CLD-41, E14)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MqttUplinkWebhookRequest(_Strict):
    topic: str
    payload: dict[str, Any] | str
    clientId: str | None = None
    ipAddress: str | None = None


class MqttUplinkWebhookResponse(_Strict):
    status: str
    deviceId: str
    channel: str
    handled: bool


class DeviceTransportStatusResponse(_Strict):
    deviceId: str
    petId: str
    status: str
    lastSeenAt: str
    ipAddress: str | None = None
    clientId: str | None = None


class SendCommandRequest(_Strict):
    action: str = Field(min_length=1, max_length=64, pattern="^[a-zA-Z0-9_.-]+$")
    args: dict[str, Any] = Field(default_factory=dict)


class SendCommandResponse(_Strict):
    commandId: str
    deviceId: str
    topic: str
    action: str
    status: str
    issuedAt: str
