"""Device-issued pet identity (Identity V1, BLE `7a6e000c`).

The device creates the PetId; the cloud only validates and references it
(ADR-0008). Never generate a PetId on the server.
"""

from dataclasses import dataclass
from enum import StrEnum

_HEX = frozenset("0123456789abcdef")


class PetOrigin(StrEnum):
    HATCHED_ON_DEVICE = "hatched_on_device"
    HATCHED_VIA_APP = "hatched_via_app"
    MIGRATED_FROM_V1 = "migrated_from_v1"


@dataclass(frozen=True, slots=True)
class PetId:
    """16-byte UUIDv4 as 32 lowercase hex characters without dashes."""

    value: str

    def __post_init__(self) -> None:
        v = self.value
        if len(v) != 32 or not set(v) <= _HEX:
            raise ValueError("PetId must be 32 lowercase hex characters")
        if v[12] != "4" or v[16] not in "89ab":
            raise ValueError("PetId must be a UUIDv4")

    def __str__(self) -> str:
        return self.value
