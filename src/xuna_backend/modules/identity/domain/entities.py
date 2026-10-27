"""Identity records. Keep UserId (owner_id), PetId, BrainId, DeviceId apart."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin


class BrainStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"  # after a device ForgetAll (threat model §4.2)


@dataclass(frozen=True, slots=True)
class Pet:
    pet_id: PetId
    owner_id: uuid.UUID
    origin: PetOrigin
    active_brain_id: uuid.UUID
    device_id: uuid.UUID
    device_public_key: bytes
    last_generation: int
    bound_at: datetime
    # Set when the owner unbound the board (device key revoked, data kept).
    device_revoked_at: datetime | None = None

    @property
    def device_bound(self) -> bool:
        return self.device_revoked_at is None


@dataclass(frozen=True, slots=True)
class Challenge:
    challenge_id: uuid.UUID
    owner_id: uuid.UUID
    pet_id: PetId
    nonce: bytes
    expires_at: datetime
    consumed: bool
