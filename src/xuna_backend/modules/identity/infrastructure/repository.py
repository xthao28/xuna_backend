import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.identity.domain.entities import BrainStatus, Challenge, Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin
from xuna_backend.modules.identity.infrastructure.models import (
    BrainRow,
    ChallengeRow,
    DeviceRow,
    PetRow,
)


class SqlIdentityRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_challenge(self, challenge: Challenge) -> None:
        self._session.add(
            ChallengeRow(
                challenge_id=challenge.challenge_id,
                owner_id=challenge.owner_id,
                pet_id=challenge.pet_id.value,
                nonce=challenge.nonce,
                expires_at=challenge.expires_at,
                consumed=False,
            )
        )
        await self._session.flush()

    async def consume_challenge(self, challenge_id: uuid.UUID) -> Challenge | None:
        row = await self._session.scalar(
            select(ChallengeRow).where(ChallengeRow.challenge_id == challenge_id).with_for_update()
        )
        if row is None:
            return None
        prior = Challenge(
            challenge_id=row.challenge_id,
            owner_id=row.owner_id,
            pet_id=PetId(row.pet_id),
            nonce=row.nonce,
            expires_at=row.expires_at,
            consumed=row.consumed,
        )
        await self._session.execute(
            update(ChallengeRow)
            .where(ChallengeRow.challenge_id == challenge_id)
            .values(consumed=True)
        )
        return prior

    async def get_pet(self, pet_id: PetId) -> Pet | None:
        rows = await self._pets(PetRow.pet_id == pet_id.value)
        return rows[0] if rows else None

    async def get_pet_by_device(self, device_id: uuid.UUID) -> Pet | None:
        rows = await self._pets(DeviceRow.device_id == device_id)
        return rows[0] if rows else None

    async def list_pets(self, owner_id: uuid.UUID) -> list[Pet]:
        return await self._pets(PetRow.owner_id == owner_id)

    async def create_pet(
        self,
        *,
        pet_id: PetId,
        owner_id: uuid.UUID,
        origin: PetOrigin,
        public_key: bytes,
        generation: int,
        now: datetime,
    ) -> Pet:
        brain_id = uuid.uuid4()
        device_id = uuid.uuid4()
        self._session.add(
            PetRow(
                pet_id=pet_id.value,
                owner_id=owner_id,
                origin=origin.value,
                last_generation=generation,
                bound_at=now,
            )
        )
        await self._session.flush()
        self._session.add_all(
            [
                BrainRow(
                    brain_id=brain_id,
                    pet_id=pet_id.value,
                    status=BrainStatus.ACTIVE.value,
                    started_at=now,
                ),
                DeviceRow(
                    device_id=device_id,
                    pet_id=pet_id.value,
                    public_key=public_key,
                    bound_at=now,
                    revoked_at=None,
                ),
            ]
        )
        await self._session.flush()
        return Pet(
            pet_id=pet_id,
            owner_id=owner_id,
            origin=origin,
            active_brain_id=brain_id,
            device_id=device_id,
            device_public_key=public_key,
            last_generation=generation,
            bound_at=now,
        )

    async def update_generation(self, pet_id: PetId, generation: int) -> None:
        await self._session.execute(
            update(PetRow).where(PetRow.pet_id == pet_id.value).values(last_generation=generation)
        )
        await self._session.flush()

    async def rebind_device(self, pet_id: PetId, public_key: bytes, now: datetime) -> Pet:
        await self._session.execute(
            update(DeviceRow)
            .where((DeviceRow.pet_id == pet_id.value) & (DeviceRow.revoked_at.is_(None)))
            .values(revoked_at=now)
        )
        new_device_id = uuid.uuid4()
        self._session.add(
            DeviceRow(
                device_id=new_device_id,
                pet_id=pet_id.value,
                public_key=public_key,
                bound_at=now,
                revoked_at=None,
            )
        )
        await self._session.flush()
        pet = await self.get_pet(pet_id)
        if pet is None:
            raise RuntimeError(f"pet {pet_id.value} missing after rebind")
        return pet

    async def rotate_brain(self, pet_id: PetId, now: datetime) -> Pet:
        await self._session.execute(
            update(BrainRow)
            .where(
                (BrainRow.pet_id == pet_id.value) & (BrainRow.status == BrainStatus.ACTIVE.value)
            )
            .values(status=BrainStatus.ARCHIVED.value)
        )
        new_brain_id = uuid.uuid4()
        self._session.add(
            BrainRow(
                brain_id=new_brain_id,
                pet_id=pet_id.value,
                status=BrainStatus.ACTIVE.value,
                started_at=now,
            )
        )
        await self._session.flush()
        pet = await self.get_pet(pet_id)
        if pet is None:
            raise RuntimeError(f"pet {pet_id.value} missing after brain rotation")
        return pet

    async def _pets(self, condition: object) -> list[Pet]:
        result = await self._session.execute(
            select(PetRow, BrainRow, DeviceRow)
            .join(BrainRow, (BrainRow.pet_id == PetRow.pet_id) & (BrainRow.status == "active"))
            .join(DeviceRow, (DeviceRow.pet_id == PetRow.pet_id) & (DeviceRow.revoked_at.is_(None)))
            .where(condition)  # type: ignore[arg-type]
            .order_by(PetRow.bound_at)
        )
        return [
            Pet(
                pet_id=PetId(pet.pet_id),
                owner_id=pet.owner_id,
                origin=PetOrigin(pet.origin),
                active_brain_id=brain.brain_id,
                device_id=device.device_id,
                device_public_key=device.public_key,
                last_generation=pet.last_generation,
                bound_at=pet.bound_at,
            )
            for pet, brain, device in result.all()
        ]
