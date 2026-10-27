"""Transport application ports (CLD-40, CLD-41, E14, H1-H8)."""

import uuid
from collections.abc import Awaitable, Callable
from typing import Protocol

from xuna_backend.modules.transport.domain.entities import (
    DeviceStatusRecord,
)


class MqttPublisher(Protocol):
    async def publish(
        self,
        topic: str,
        payload: str | bytes,
        qos: int = 1,
        retain: bool = False,
    ) -> None: ...


class TransportStatusRepository(Protocol):
    async def save_status(self, record: DeviceStatusRecord) -> None: ...

    async def get_status(self, device_id: uuid.UUID) -> DeviceStatusRecord | None: ...

    async def list_statuses_for_pet(self, pet_id: str) -> list[DeviceStatusRecord]: ...


class DeviceSessionHub(Protocol):
    """Live device-direct sessions, one per pet: the newest verified one wins."""

    async def claim(
        self, pet_id: str, token: object, replace: Callable[[], Awaitable[None]]
    ) -> None:
        """Registers `token` as the pet's session; closes the previous one via its `replace`."""
        ...

    def release(self, pet_id: str, token: object) -> bool:
        """True when `token` was still the pet's session (the caller marks it offline)."""
        ...
