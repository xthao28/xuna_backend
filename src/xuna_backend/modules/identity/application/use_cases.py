"""CLD-11 identity use cases: device challenge and trust-on-first-use binding."""

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from xuna_backend.modules.identity.application.ports import (
    IdentityRepository,
    SignatureVerifier,
)
from xuna_backend.modules.identity.domain.device_proof import (
    NONCE_SIZE,
    SIGNATURE_SIZE,
    ProofPurpose,
    proof_message,
    validate_public_key,
)
from xuna_backend.modules.identity.domain.entities import Challenge, Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin

CHALLENGE_TTL = timedelta(minutes=5)


class IdentityError(Exception):
    """Business rejection with a stable problem `code`."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class BindPet:
    pet_id: PetId
    origin: PetOrigin
    snapshot_generation: int
    challenge_id: uuid.UUID
    public_key: bytes
    signature: bytes


async def issue_challenge(
    repo: IdentityRepository, owner_id: uuid.UUID, pet_id: PetId, now: datetime
) -> Challenge:
    challenge = Challenge(
        challenge_id=uuid.uuid4(),
        owner_id=owner_id,
        pet_id=pet_id,
        nonce=secrets.token_bytes(NONCE_SIZE),
        expires_at=now + CHALLENGE_TTL,
        consumed=False,
    )
    await repo.add_challenge(challenge)
    return challenge


async def bind_pet(
    repo: IdentityRepository,
    verifier: SignatureVerifier,
    owner_id: uuid.UUID,
    command: BindPet,
    now: datetime,
) -> tuple[Pet, bool]:
    """Returns (pet, created). Challenges are single-use even when rejected."""
    try:
        validate_public_key(command.public_key)
    except ValueError as error:
        raise IdentityError("invalid_public_key") from error
    if len(command.signature) != SIGNATURE_SIZE:
        raise IdentityError("invalid_signature")
    challenge = await repo.consume_challenge(command.challenge_id)
    if (
        challenge is None
        or challenge.consumed
        or challenge.owner_id != owner_id
        or challenge.pet_id != command.pet_id
        or challenge.expires_at <= now
    ):
        raise IdentityError("challenge_invalid")
    message = proof_message(
        challenge.nonce,
        command.pet_id,
        ProofPurpose.BIND,
        command.snapshot_generation,
        command.public_key,
    )
    if not verifier.verify(command.public_key, message, command.signature):
        raise IdentityError("invalid_signature")
    existing = await repo.get_pet(command.pet_id)
    if existing is not None:
        if existing.owner_id != owner_id:
            raise IdentityError("pet_already_bound")
        if existing.device_public_key != command.public_key:
            raise IdentityError("device_key_mismatch")
        return existing, False
    pet = await repo.create_pet(
        pet_id=command.pet_id,
        owner_id=owner_id,
        origin=command.origin,
        public_key=command.public_key,
        generation=command.snapshot_generation,
        now=now,
    )
    return pet, True


async def list_pets(repo: IdentityRepository, owner_id: uuid.UUID) -> list[Pet]:
    return await repo.list_pets(owner_id)


async def get_owned_pet(repo: IdentityRepository, owner_id: uuid.UUID, pet_id: PetId) -> Pet:
    pet = await repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise IdentityError("not_found")  # never reveal another owner's pet
    return pet
