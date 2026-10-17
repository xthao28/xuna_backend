import asyncio
import dataclasses
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from tests.helpers import SimulatedDevice
from xuna_backend.modules.identity.application.use_cases import (
    BindPet,
    IdentityError,
    bind_pet,
    get_owned_pet,
    issue_challenge,
)
from xuna_backend.modules.identity.domain.device_proof import ProofPurpose, proof_message
from xuna_backend.modules.identity.domain.entities import Challenge, Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier

PET = PetId("d945c77026fd4f4a8a899bbff4aeed86")
OWNER = uuid.uuid4()
NOW = datetime(2026, 10, 17, 12, tzinfo=UTC)


class MemoryRepo:
    def __init__(self) -> None:
        self.challenges: dict[uuid.UUID, Challenge] = {}
        self.pets: dict[str, Pet] = {}

    async def add_challenge(self, challenge: Challenge) -> None:
        self.challenges[challenge.challenge_id] = challenge

    async def consume_challenge(self, challenge_id: uuid.UUID) -> Challenge | None:
        prior = self.challenges.get(challenge_id)
        if prior is not None:
            self.challenges[challenge_id] = dataclasses.replace(prior, consumed=True)
        return prior

    async def get_pet(self, pet_id: PetId) -> Pet | None:
        return self.pets.get(pet_id.value)

    async def list_pets(self, owner_id: uuid.UUID) -> list[Pet]:
        return [p for p in self.pets.values() if p.owner_id == owner_id]

    async def create_pet(self, **kw: object) -> Pet:
        pet = Pet(
            pet_id=kw["pet_id"],  # type: ignore[arg-type]
            owner_id=kw["owner_id"],  # type: ignore[arg-type]
            origin=kw["origin"],  # type: ignore[arg-type]
            active_brain_id=uuid.uuid4(),
            device_id=uuid.uuid4(),
            device_public_key=kw["public_key"],  # type: ignore[arg-type]
            last_generation=kw["generation"],  # type: ignore[arg-type]
            bound_at=kw["now"],  # type: ignore[arg-type]
        )
        self.pets[pet.pet_id.value] = pet
        return pet


def command(
    repo: MemoryRepo, device: SimulatedDevice, owner: uuid.UUID = OWNER, now: datetime = NOW
) -> BindPet:
    challenge = asyncio.run(issue_challenge(repo, owner, PET, now))
    message = proof_message(challenge.nonce, PET, ProofPurpose.BIND, 38, device.public_key)
    return BindPet(
        pet_id=PET,
        origin=PetOrigin.MIGRATED_FROM_V1,
        snapshot_generation=38,
        challenge_id=challenge.challenge_id,
        public_key=device.public_key,
        signature=device.sign(message),
    )


def bind(repo: MemoryRepo, cmd: BindPet, owner: uuid.UUID = OWNER, now: datetime = NOW) -> Pet:
    pet, _ = asyncio.run(bind_pet(repo, P256SignatureVerifier(), owner, cmd, now))
    return pet


def rejected(repo: MemoryRepo, cmd: BindPet, code: str, **kw: object) -> None:
    with pytest.raises(IdentityError) as error:
        bind(repo, cmd, **kw)  # type: ignore[arg-type]
    assert error.value.code == code


def test_first_bind_creates_pet_brain_device_then_is_idempotent() -> None:
    repo, device = MemoryRepo(), SimulatedDevice()
    pet = bind(repo, command(repo, device))
    assert pet.last_generation == 38
    again, created = asyncio.run(
        bind_pet(repo, P256SignatureVerifier(), OWNER, command(repo, device), NOW)
    )
    assert not created
    assert again == pet


def test_challenge_is_single_use_and_expires() -> None:
    repo, device = MemoryRepo(), SimulatedDevice()
    cmd = command(repo, device)
    bind(repo, cmd)
    rejected(repo, cmd, "challenge_invalid")
    rejected(repo, command(repo, device), "challenge_invalid", now=NOW + timedelta(minutes=6))


def test_challenge_is_bound_to_owner_and_pet() -> None:
    repo, device = MemoryRepo(), SimulatedDevice()
    rejected(repo, command(repo, device), "challenge_invalid", owner=uuid.uuid4())
    other = dataclasses.replace(
        command(repo, device), pet_id=PetId("000102030405460788090a0b0c0d0e0f")
    )
    rejected(repo, other, "challenge_invalid")


def test_signature_must_cover_this_key_and_nonce() -> None:
    repo, device = MemoryRepo(), SimulatedDevice()
    stolen = command(repo, device)
    rejected(
        repo,
        dataclasses.replace(stolen, public_key=SimulatedDevice().public_key),
        "invalid_signature",
    )
    rejected(
        repo, dataclasses.replace(command(repo, device), signature=bytes(64)), "invalid_signature"
    )
    rejected(
        repo,
        dataclasses.replace(command(repo, device), public_key=b"\x02" + bytes(32)),
        "invalid_public_key",
    )


def test_tofu_pins_owner_and_device_key() -> None:
    repo, device = MemoryRepo(), SimulatedDevice()
    bind(repo, command(repo, device))
    intruder = uuid.uuid4()
    rejected(repo, command(repo, device, owner=intruder), "pet_already_bound", owner=intruder)
    rejected(repo, command(repo, SimulatedDevice()), "device_key_mismatch")


def test_foreign_pet_is_not_found() -> None:
    repo, device = MemoryRepo(), SimulatedDevice()
    bind(repo, command(repo, device))
    with pytest.raises(IdentityError, match="not_found"):
        asyncio.run(get_owned_pet(repo, uuid.uuid4(), PET))
