"""Unit tests for backup, restore, and tombstone use cases (CLD-20)."""

import uuid
from datetime import UTC, datetime

import pytest

from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.backups.application.use_cases import (
    CreateBackupInput,
    RestoreDeviceProofInput,
    create_backup,
    create_tombstone,
    forget_all,
    restore_backup,
)
from xuna_backend.modules.backups.domain.entities import (
    BackupManifest,
    DeletionTombstone,
)
from xuna_backend.modules.identity.domain.entities import Challenge, Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
)

PET_ID = PetId("d945c77026fd4f4a8a899bbff4aeed86")
OWNER_ID = uuid.uuid4()
BRAIN_ID = uuid.uuid4()
DEVICE_ID = uuid.uuid4()
SECRET = "test-secret-key-32-chars-long!!"  # noqa: S105


class FakeBackupRepo:
    def __init__(self) -> None:
        self.backups: dict[uuid.UUID, BackupManifest] = {}
        self.tombstones: list[DeletionTombstone] = []

    async def add_backup(self, manifest: BackupManifest) -> None:
        self.backups[manifest.backup_id] = manifest

    async def get_backup(self, backup_id: uuid.UUID) -> BackupManifest | None:
        return self.backups.get(backup_id)

    async def list_backups(self, pet_id: PetId, limit: int = 50) -> list[BackupManifest]:
        return [b for b in self.backups.values() if b.pet_id == pet_id][:limit]

    async def delete_backup(self, backup_id: uuid.UUID) -> None:
        self.backups.pop(backup_id, None)

    async def add_tombstone(self, tombstone: DeletionTombstone) -> None:
        self.tombstones.append(tombstone)

    async def list_tombstones(self, pet_id: PetId, limit: int = 100) -> list[DeletionTombstone]:
        return [t for t in self.tombstones if t.pet_id == pet_id][:limit]

    async def is_tombstoned(self, pet_id: PetId, item_type: str, item_id: str) -> bool:
        return any(
            t.pet_id == pet_id
            and (t.item_type == item_type or t.item_type == "memory")
            and (t.item_id == item_id or t.item_id == "*")
            for t in self.tombstones
        )


class FakeIdentityRepo:
    def __init__(self, pet: Pet) -> None:
        self.pet = pet
        self.challenges: dict[uuid.UUID, Challenge] = {}

    async def get_pet(self, pet_id: PetId) -> Pet | None:
        if self.pet.pet_id == pet_id:
            return self.pet
        return None

    async def list_pets(self, owner_id: uuid.UUID) -> list[Pet]:
        return [self.pet] if self.pet.owner_id == owner_id else []

    async def add_challenge(self, challenge: Challenge) -> None:
        self.challenges[challenge.challenge_id] = challenge

    async def consume_challenge(self, challenge_id: uuid.UUID) -> Challenge | None:
        c = self.challenges.get(challenge_id)
        if c:
            self.challenges[challenge_id] = Challenge(
                challenge_id=c.challenge_id,
                owner_id=c.owner_id,
                pet_id=c.pet_id,
                nonce=c.nonce,
                expires_at=c.expires_at,
                consumed=True,
            )
        return c

    async def create_pet(self, **kwargs: object) -> Pet:
        raise NotImplementedError

    async def update_generation(self, pet_id: PetId, generation: int) -> None:
        pass

    async def rebind_device(self, pet_id: PetId, public_key: bytes, now: datetime) -> Pet:
        self.pet = Pet(
            pet_id=self.pet.pet_id,
            owner_id=self.pet.owner_id,
            origin=self.pet.origin,
            active_brain_id=self.pet.active_brain_id,
            device_id=uuid.uuid4(),
            device_public_key=public_key,
            last_generation=self.pet.last_generation,
            bound_at=now,
        )
        return self.pet

    async def rotate_brain(self, pet_id: PetId, now: datetime) -> Pet:
        self.pet = Pet(
            pet_id=self.pet.pet_id,
            owner_id=self.pet.owner_id,
            origin=self.pet.origin,
            active_brain_id=uuid.uuid4(),
            device_id=self.pet.device_id,
            device_public_key=self.pet.device_public_key,
            last_generation=self.pet.last_generation,
            bound_at=now,
        )
        return self.pet


class FakeMemoryRepo:
    def __init__(self) -> None:
        self.candidates: dict[uuid.UUID, MemoryCandidate] = {}

    async def add_projection(self, projection: object) -> None:
        pass

    async def get_projection(self, pet_id: PetId, generation: int) -> object | None:
        return None

    async def get_latest_projection(self, pet_id: PetId) -> object | None:
        return None

    async def list_projections(self, pet_id: PetId, limit: int = 50) -> list[object]:
        return []

    async def add_candidate(self, candidate: MemoryCandidate) -> None:
        self.candidates[candidate.candidate_id] = candidate

    async def get_candidate(self, candidate_id: uuid.UUID) -> MemoryCandidate | None:
        return self.candidates.get(candidate_id)

    async def update_candidate(self, candidate: MemoryCandidate) -> None:
        self.candidates[candidate.candidate_id] = candidate

    async def list_candidates(
        self, pet_id: PetId, status: CandidateStatus | None = None, limit: int = 50
    ) -> list[MemoryCandidate]:
        items = [c for c in self.candidates.values() if c.pet_id == pet_id]
        if status is not None:
            items = [c for c in items if c.status == status]
        return items[:limit]


class FakeAuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)

    async def recent(self, owner_id: uuid.UUID, limit: int = 50) -> list[AuditEvent]:
        return [e for e in self.events if e.owner_id == owner_id][:limit]


class FakeVerifier:
    def __init__(self, valid: bool = True) -> None:
        self.valid = valid

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        return self.valid


@pytest.fixture
def pet() -> Pet:
    return Pet(
        pet_id=PET_ID,
        owner_id=OWNER_ID,
        origin=PetOrigin.HATCHED_ON_DEVICE,
        active_brain_id=BRAIN_ID,
        device_id=DEVICE_ID,
        device_public_key=b"\x04" + b"\x01" * 64,
        last_generation=10,
        bound_at=datetime.now(UTC),
    )


@pytest.mark.anyio
async def test_create_and_restore_backup_roundtrip(pet: Pet) -> None:
    repo = FakeBackupRepo()
    id_repo = FakeIdentityRepo(pet)
    audit = FakeAuditLog()
    now = datetime.now(UTC)

    export_data = {
        "schemaVersion": 1,
        "payload": {
            "identity": {"petId": PET_ID.value, "snapshotGeneration": 10},
            "life": {"activeSeconds": 1200},
            "memories": [
                {"context": "quiet", "action": "idle", "approved": True},
                {"context": "play", "action": "happy", "approved": True},
            ],
        },
    }

    manifest = await create_backup(
        repo=repo,
        identity_repo=id_repo,
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        command=CreateBackupInput(snapshot_generation=10, export_payload=export_data),
        secret_key=SECRET,
        now=now,
    )
    assert manifest.snapshot_generation == 10
    assert manifest.brain_id == BRAIN_ID
    assert any(e.action == AuditAction.BACKUP_CREATED for e in audit.events)

    # Restore on same hardware
    restore_res = await restore_backup(
        repo=repo,
        identity_repo=id_repo,
        verifier=FakeVerifier(True),
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        backup_id=manifest.backup_id,
        device_proof=None,
        secret_key=SECRET,
        now=now,
    )
    assert restore_res.pet_id == PET_ID
    assert restore_res.brain_id == BRAIN_ID
    assert restore_res.device_id == DEVICE_ID
    assert len(restore_res.restored_payload["payload"]["memories"]) == 2
    assert restore_res.tombstones_applied == 0
    assert any(e.action == AuditAction.BACKUP_RESTORED for e in audit.events)


@pytest.mark.anyio
async def test_tombstone_prevents_memory_resurrection_on_restore(pet: Pet) -> None:
    repo = FakeBackupRepo()
    id_repo = FakeIdentityRepo(pet)
    audit = FakeAuditLog()
    now = datetime.now(UTC)

    export_data = {
        "schemaVersion": 1,
        "payload": {
            "identity": {"petId": PET_ID.value, "snapshotGeneration": 10},
            "memories": [
                {"context": "quiet", "action": "idle", "approved": True},
                {"context": "play", "action": "happy", "approved": True},
            ],
        },
    }

    manifest = await create_backup(
        repo=repo,
        identity_repo=id_repo,
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        command=CreateBackupInput(snapshot_generation=10, export_payload=export_data),
        secret_key=SECRET,
        now=now,
    )

    # Create tombstone for "play" memory context
    await create_tombstone(
        repo=repo,
        identity_repo=id_repo,
        memory_repo=None,
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        item_type="memory",
        item_id="play",
        reason="deleted_by_user",
        now=now,
    )

    # Restore backup: the "play" memory must be stripped out by the tombstone!
    restore_res = await restore_backup(
        repo=repo,
        identity_repo=id_repo,
        verifier=FakeVerifier(True),
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        backup_id=manifest.backup_id,
        device_proof=None,
        secret_key=SECRET,
        now=now,
    )
    assert restore_res.tombstones_applied == 1
    memories = restore_res.restored_payload["payload"]["memories"]
    assert len(memories) == 1
    assert memories[0]["context"] == "quiet"


@pytest.mark.anyio
async def test_restore_to_replacement_device_rebinds_hardware(pet: Pet) -> None:
    repo = FakeBackupRepo()
    id_repo = FakeIdentityRepo(pet)
    audit = FakeAuditLog()
    now = datetime.now(UTC)

    export_data = {"schemaVersion": 1, "payload": {}}
    manifest = await create_backup(
        repo=repo,
        identity_repo=id_repo,
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        command=CreateBackupInput(snapshot_generation=10, export_payload=export_data),
        secret_key=SECRET,
        now=now,
    )

    new_pubkey = b"\x04" + b"\x02" * 64
    challenge_id = uuid.uuid4()
    challenge = Challenge(
        challenge_id=challenge_id,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        nonce=b"\x11" * 32,
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
        consumed=False,
    )
    await id_repo.add_challenge(challenge)

    proof = RestoreDeviceProofInput(
        challenge_id=challenge_id,
        public_key=new_pubkey,
        signature=b"\x99" * 64,
    )

    restore_res = await restore_backup(
        repo=repo,
        identity_repo=id_repo,
        verifier=FakeVerifier(True),
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        backup_id=manifest.backup_id,
        device_proof=proof,
        secret_key=SECRET,
        now=now,
    )

    # PetId and BrainId are preserved; DeviceId is rebound!
    assert restore_res.pet_id == PET_ID
    assert restore_res.brain_id == BRAIN_ID
    assert restore_res.device_id != DEVICE_ID
    assert id_repo.pet.device_public_key == new_pubkey


@pytest.mark.anyio
async def test_forget_all_archives_brain_and_leaves_tombstone(pet: Pet) -> None:
    repo = FakeBackupRepo()
    id_repo = FakeIdentityRepo(pet)
    mem_repo = FakeMemoryRepo()
    audit = FakeAuditLog()
    now = datetime.now(UTC)

    candidate_id = uuid.uuid4()
    await mem_repo.add_candidate(
        MemoryCandidate(
            candidate_id=candidate_id,
            owner_id=OWNER_ID,
            pet_id=PET_ID,
            source="text",
            candidate_type=CandidateType.FACT,
            content={"key": "name", "value": "Thao"},
            status=CandidateStatus.CONFIRMED,
            created_at=now,
            updated_at=now,
        )
    )

    p_id, new_brain_id, tomb_id = await forget_all(
        repo=repo,
        identity_repo=id_repo,
        memory_repo=mem_repo,
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        now=now,
    )

    assert p_id == PET_ID
    assert new_brain_id != BRAIN_ID  # new active brain!
    assert id_repo.pet.active_brain_id == new_brain_id
    assert any(t.item_id == "*" for t in repo.tombstones)
    assert any(e.action == AuditAction.PET_FORGOTTEN for e in audit.events)

    # Candidate was marked DELETED
    cand = await mem_repo.get_candidate(candidate_id)
    assert cand is not None
    assert cand.status == CandidateStatus.DELETED
