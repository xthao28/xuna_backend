import uuid
from datetime import datetime
from typing import Protocol

from xuna_backend.modules.identity.domain.entities import Challenge, Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin


class SignatureVerifier(Protocol):
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """ECDSA P-256 over SHA-256(message); raw r||s signature."""
        ...


class IdentityRepository(Protocol):
    async def add_challenge(self, challenge: Challenge) -> None: ...

    async def consume_challenge(self, challenge_id: uuid.UUID) -> Challenge | None:
        """Marks it consumed and returns the prior state, or None if unknown."""
        ...

    async def get_pet(self, pet_id: PetId) -> Pet | None: ...

    async def list_pets(self, owner_id: uuid.UUID) -> list[Pet]: ...

    async def create_pet(
        self,
        *,
        pet_id: PetId,
        owner_id: uuid.UUID,
        origin: PetOrigin,
        public_key: bytes,
        generation: int,
        now: datetime,
    ) -> Pet: ...

    async def update_generation(self, pet_id: PetId, generation: int) -> None: ...
