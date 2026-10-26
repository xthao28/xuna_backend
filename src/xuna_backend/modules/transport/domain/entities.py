"""Domain entities for Direct Device Transport (CLD-40, CLD-41, E14, H1-H8).

Rules: stdlib only. No frameworks, no pydantic.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

_TOPIC_PATTERN = re.compile(
    r"^xuna/v1/([0-9a-fA-F-]{36})/(status|(up|down)/(events|state|cmd|suggest))$"
)
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


class TransportChannel(StrEnum):
    STATUS = "status"
    EVENTS = "events"
    STATE = "state"
    CMD = "cmd"
    SUGGEST = "suggest"


class TransportDirection(StrEnum):
    UP = "up"
    DOWN = "down"


class DeviceConnectionStatus(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(frozen=True, slots=True)
class MqttTopic:
    device_id: uuid.UUID
    channel: TransportChannel
    direction: TransportDirection | None = None

    def __post_init__(self) -> None:
        if self.channel == TransportChannel.STATUS:
            if self.direction is not None:
                raise ValueError("STATUS topic must not have a direction")
        else:
            if self.direction is None:
                raise ValueError(f"{self.channel} topic requires direction (up/down)")

    def as_str(self) -> str:
        dev_str = str(self.device_id)
        if self.channel == TransportChannel.STATUS:
            return f"xuna/v1/{dev_str}/status"
        if self.direction is None:
            raise ValueError("direction is required for non-STATUS topic")
        return f"xuna/v1/{dev_str}/{self.direction.value}/{self.channel.value}"

    @classmethod
    def parse(cls, topic: str) -> "MqttTopic":
        match = _TOPIC_PATTERN.match(topic)
        if not match:
            raise ValueError(f"invalid MQTT topic: {topic!r}")

        device_id_str = match.group(1)
        try:
            device_id = uuid.UUID(device_id_str)
        except ValueError as err:
            raise ValueError(f"invalid device_id in topic: {device_id_str!r}") from err

        second_part = match.group(2)
        if second_part == "status":
            return cls(device_id=device_id, channel=TransportChannel.STATUS, direction=None)

        direction_str = match.group(3)
        channel_str = match.group(4)
        return cls(
            device_id=device_id,
            channel=TransportChannel(channel_str),
            direction=TransportDirection(direction_str),
        )


@dataclass(frozen=True, slots=True)
class DeviceStatusRecord:
    device_id: uuid.UUID
    pet_id: str
    status: DeviceConnectionStatus
    last_seen_at: datetime
    ip_address: str | None = None
    client_id: str | None = None

    def __post_init__(self) -> None:
        if self.last_seen_at.tzinfo is None:
            raise ValueError("last_seen_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class DownlinkCommand:
    command_id: uuid.UUID
    device_id: uuid.UUID
    action: str
    args: dict[str, Any]
    issued_at: datetime

    def __post_init__(self) -> None:
        if self.issued_at.tzinfo is None:
            raise ValueError("issued_at must be timezone-aware")
        if not _IDENTIFIER_PATTERN.match(self.action):
            raise ValueError(f"invalid command action: {self.action!r}")
