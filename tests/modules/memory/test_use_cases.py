"""Use-case tests for memory projections and candidates (CLD-14)."""

import base64
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.helpers import MemoryAuditLog, SimulatedDevice

from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.consent.domain.consent import ConsentState
from xuna_backend.modules.exports.domain.export_document import payload_hash
from xuna_backend.modules.identity.application.use_cases import issue_challenge
from xuna_backend.modules.identity.domain.device_proof import (
    ProofPurpose,
    proof_message,
)
from xuna_backend.modules.identity.domain.entities import Challenge, Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier
from xuna_backend.modules.memory.application.schemas import (
    CreateCandidateRequest,
    DeviceProofDto,
    PatchCandidateRequest,
)
from xuna_backend.modules.memory.application.use_cases import (
    MemoryError,
    create_candidate,
    ingest_projection,
    list_candidates,
    patch_candidate,
)
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
    Projection,
)

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "xuna_export_v1_golden.json"
PET = PetId("d945c77026fd4f4a8a899bbff4aeed86")
OWNER = uuid.uuid4()
NOW = datetime(2026, 10, 26, 10, 0, tzinfo=UTC)


class FakeMemoryRepo:
    def __init__(self) -> None:
        self.projections: dict[tuple[str, int], Projection] = {}
        self.candidates: dict[uuid.UUID, MemoryCandidate] = {}

    async def add_projection(self, projection: Projection) -> None:
        key = (projection.pet_id.value, projection.snapshot_generation)
        self.projections[key] = projection

    async def get_projection(self, pet_id: PetId, generation: int) -> Projection | None:
        return self.projections.get((pet_id.value, generation))

    async def get_latest_projection(self, pet_id: PetId) -> Projection | None:
        pet_projs = [p for p in self.projections.values() if p.pet_id == pet_id]
        if not pet_projs:
            return None
        return sorted(pet_projs, key=lambda p: (p.created_at, p.snapshot_generation))[-1]

    async def list_projections(self, pet_id: PetId, limit: int = 50) -> list[Projection]:
        pet_projs = [p for p in self.projections.values() if p.pet_id == pet_id]
        return sorted(pet_projs, key=lambda p: p.created_at, reverse=True)[:limit]

    async def add_candidate(self, candidate: MemoryCandidate) -> None:
        self.candidates[candidate.candidate_id] = candidate

    async def get_candidate(self, candidate_id: uuid.UUID) -> MemoryCandidate | None:
        return self.candidates.get(candidate_id)

    async def update_candidate(self, candidate: MemoryCandidate) -> None:
        self.candidates[candidate.candidate_id] = candidate

    async def list_candidates(
        self,
        pet_id: PetId,
        status: CandidateStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        cands = [c for c in self.candidates.values() if c.pet_id == pet_id]
        if status is not None:
            cands = [c for c in cands if c.status == status]
        return sorted(cands, key=lambda c: c.created_at, reverse=True)[:limit]


class FakeIdentityRepo:
    def __init__(self) -> None:
        self.challenges: dict[uuid.UUID, Challenge] = {}
        self.pets: dict[str, Pet] = {}

    async def add_challenge(self, challenge: Challenge) -> None:
        self.challenges[challenge.challenge_id] = challenge

    async def consume_challenge(self, challenge_id: uuid.UUID) -> Challenge | None:
        c = self.challenges.get(challenge_id)
        if c is not None:
            self.challenges[challenge_id] = Challenge(
                challenge_id=c.challenge_id,
                owner_id=c.owner_id,
                pet_id=c.pet_id,
                nonce=c.nonce,
                expires_at=c.expires_at,
                consumed=True,
            )
        return c

    async def get_pet(self, pet_id: PetId) -> Pet | None:
        return self.pets.get(pet_id.value)

    async def list_pets(self, owner_id: uuid.UUID) -> list[Pet]:
        return [p for p in self.pets.values() if p.owner_id == owner_id]

    async def create_pet(self, **kw: object) -> Pet:
        raise NotImplementedError

    async def update_generation(self, pet_id: PetId, generation: int) -> None:
        pet = self.pets.get(pet_id.value)
        if pet is not None:
            self.pets[pet_id.value] = Pet(
                pet_id=pet.pet_id,
                owner_id=pet.owner_id,
                origin=pet.origin,
                active_brain_id=pet.active_brain_id,
                device_id=pet.device_id,
                device_public_key=pet.device_public_key,
                last_generation=generation,
                bound_at=pet.bound_at,
            )


def golden_envelope() -> dict[str, object]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def setup_repos(device: SimulatedDevice) -> tuple[FakeMemoryRepo, FakeIdentityRepo]:
    mem_repo = FakeMemoryRepo()
    id_repo = FakeIdentityRepo()
    id_repo.pets[PET.value] = Pet(
        pet_id=PET,
        owner_id=OWNER,
        origin=PetOrigin.MIGRATED_FROM_V1,
        active_brain_id=uuid.uuid4(),
        device_id=uuid.uuid4(),
        device_public_key=device.public_key,
        last_generation=37,
        bound_at=NOW,
    )
    return mem_repo, id_repo


@pytest.mark.anyio
async def test_ingest_projection_requires_u1_consent() -> None:
    device = SimulatedDevice()
    mem_repo, id_repo = setup_repos(device)
    audit = MemoryAuditLog()
    verifier = P256SignatureVerifier()

    with pytest.raises(MemoryError) as exc:
        await ingest_projection(
            repo=mem_repo,
            identity_repo=id_repo,
            verifier=verifier,
            audit=audit,
            owner_id=OWNER,
            pet_id=PET,
            generation=38,
            envelope=golden_envelope(),
            device_proof=None,
            consent=ConsentState(aggregate_metrics=False),  # U0
            now=NOW,
        )
    assert exc.value.code == "consent_required"


@pytest.mark.anyio
async def test_ingest_projection_u1_only_strips_memories() -> None:
    device = SimulatedDevice()
    mem_repo, id_repo = setup_repos(device)
    audit = MemoryAuditLog()
    verifier = P256SignatureVerifier()

    proj, created = await ingest_projection(
        repo=mem_repo,
        identity_repo=id_repo,
        verifier=verifier,
        audit=audit,
        owner_id=OWNER,
        pet_id=PET,
        generation=38,
        envelope=golden_envelope(),
        device_proof=None,
        consent=ConsentState(aggregate_metrics=True, experience_events=False),
        now=NOW,
    )
    assert created is True
    assert proj.snapshot_generation == 38
    # memories must be stripped when U2 is False
    assert proj.payload["memories"] == []
    # Explanation and life remain
    assert proj.payload["life"]["context"] == "quiet"
    assert proj.payload["explanation"]["reason"] == "highest_estimate"
    assert proj.device_attested is False

    # Pet generation updated
    updated_pet = await id_repo.get_pet(PET)
    assert updated_pet is not None and updated_pet.last_generation == 38

    # Audit logged
    assert len(audit.events) == 1
    assert audit.events[0].action == AuditAction.PROJECTION_INGESTED


@pytest.mark.anyio
async def test_ingest_projection_u1_and_u2_retains_memories() -> None:
    device = SimulatedDevice()
    mem_repo, id_repo = setup_repos(device)
    audit = MemoryAuditLog()
    verifier = P256SignatureVerifier()

    proj, created = await ingest_projection(
        repo=mem_repo,
        identity_repo=id_repo,
        verifier=verifier,
        audit=audit,
        owner_id=OWNER,
        pet_id=PET,
        generation=38,
        envelope=golden_envelope(),
        device_proof=None,
        consent=ConsentState(aggregate_metrics=True, experience_events=True),
        now=NOW,
    )
    assert created is True
    assert len(proj.payload["memories"]) == 2


@pytest.mark.anyio
async def test_ingest_projection_idempotency_and_conflict() -> None:
    device = SimulatedDevice()
    mem_repo, id_repo = setup_repos(device)
    audit = MemoryAuditLog()
    verifier = P256SignatureVerifier()
    consent = ConsentState(aggregate_metrics=True, experience_events=True)

    # 1. First ingest
    proj1, created1 = await ingest_projection(
        repo=mem_repo,
        identity_repo=id_repo,
        verifier=verifier,
        audit=audit,
        owner_id=OWNER,
        pet_id=PET,
        generation=38,
        envelope=golden_envelope(),
        device_proof=None,
        consent=consent,
        now=NOW,
    )
    assert created1 is True

    # 2. Re-ingest exact same envelope -> idempotent (200, created=False)
    proj2, created2 = await ingest_projection(
        repo=mem_repo,
        identity_repo=id_repo,
        verifier=verifier,
        audit=audit,
        owner_id=OWNER,
        pet_id=PET,
        generation=38,
        envelope=golden_envelope(),
        device_proof=None,
        consent=consent,
        now=NOW + timedelta(seconds=5),
    )
    assert created2 is False
    assert proj2.projection_id == proj1.projection_id

    # 3. Same generation, different payload -> conflict
    alt_envelope = golden_envelope()
    alt_envelope["sha256"] = "b" * 64
    with pytest.raises(MemoryError) as exc:
        await ingest_projection(
            repo=mem_repo,
            identity_repo=id_repo,
            verifier=verifier,
            audit=audit,
            owner_id=OWNER,
            pet_id=PET,
            generation=38,
            envelope=alt_envelope,
            device_proof=None,
            consent=consent,
            now=NOW,
        )
    assert exc.value.code == "invalid_envelope"


@pytest.mark.anyio
async def test_ingest_projection_stale_generation() -> None:
    device = SimulatedDevice()
    mem_repo, id_repo = setup_repos(device)
    audit = MemoryAuditLog()
    verifier = P256SignatureVerifier()
    consent = ConsentState(aggregate_metrics=True)

    # Ingest generation 38
    await ingest_projection(
        repo=mem_repo,
        identity_repo=id_repo,
        verifier=verifier,
        audit=audit,
        owner_id=OWNER,
        pet_id=PET,
        generation=38,
        envelope=golden_envelope(),
        device_proof=None,
        consent=consent,
        now=NOW,
    )

    # Ingest generation 35 (stale compared to 38)
    stale_envelope = golden_envelope()
    stale_envelope["payload"]["identity"]["snapshotGeneration"] = 35  # type: ignore[index]
    stale_envelope["sha256"] = payload_hash(stale_envelope["payload"])  # type: ignore[arg-type]

    with pytest.raises(MemoryError) as exc:
        await ingest_projection(
            repo=mem_repo,
            identity_repo=id_repo,
            verifier=verifier,
            audit=audit,
            owner_id=OWNER,
            pet_id=PET,
            generation=35,
            envelope=stale_envelope,
            device_proof=None,
            consent=consent,
            now=NOW,
        )
    assert exc.value.code == "conflict_stale_generation"


@pytest.mark.anyio
async def test_ingest_projection_with_device_proof() -> None:
    device = SimulatedDevice()
    mem_repo, id_repo = setup_repos(device)
    audit = MemoryAuditLog()
    verifier = P256SignatureVerifier()
    consent = ConsentState(aggregate_metrics=True)

    # Issue challenge
    challenge = await issue_challenge(id_repo, OWNER, PET, NOW)

    # Build signature over proof message (purpose 2: PROJECTION)
    envelope = golden_envelope()
    compact_payload = json.dumps(
        envelope["payload"], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    message = proof_message(challenge.nonce, PET, ProofPurpose.PROJECTION, 38, compact_payload)
    sig = device.sign(message)

    proof_dto = DeviceProofDto(
        challengeId=str(challenge.challenge_id),
        publicKey=base64.b64encode(device.public_key).decode(),
        signature=base64.b64encode(sig).decode(),
    )

    proj, created = await ingest_projection(
        repo=mem_repo,
        identity_repo=id_repo,
        verifier=verifier,
        audit=audit,
        owner_id=OWNER,
        pet_id=PET,
        generation=38,
        envelope=envelope,
        device_proof=proof_dto,
        consent=consent,
        now=NOW,
    )
    assert created is True
    assert proj.device_attested is True


@pytest.mark.anyio
async def test_memory_candidate_lifecycle() -> None:
    device = SimulatedDevice()
    mem_repo, id_repo = setup_repos(device)
    audit = MemoryAuditLog()

    # 1. Create candidate
    req = CreateCandidateRequest(
        petId=PET.value,
        source="conversation",
        candidateType=CandidateType.STATED_PREFERENCE,
        content={"action": "happy", "context": "play", "polarity": "like"},
    )
    candidate = await create_candidate(mem_repo, id_repo, OWNER, req, NOW)
    assert candidate.status == CandidateStatus.AWAITING_OWNER_CONFIRMATION

    # 2. Confirm candidate
    confirmed = await patch_candidate(
        mem_repo,
        audit,
        OWNER,
        candidate.candidate_id,
        PatchCandidateRequest(action="confirm"),
        NOW,
    )
    assert confirmed.status == CandidateStatus.AWAITING_DEVICE_COMMIT
    assert confirmed.confirmed_at == NOW

    # 3. Commit candidate
    committed = await patch_candidate(
        mem_repo,
        audit,
        OWNER,
        candidate.candidate_id,
        PatchCandidateRequest(action="commit", committedGeneration=39),
        NOW,
    )
    assert committed.status == CandidateStatus.CONFIRMED
    assert committed.committed_generation == 39

    # 4. List candidates
    cands = await list_candidates(mem_repo, id_repo, OWNER, PET)
    assert len(cands) == 1
    assert cands[0].status == CandidateStatus.CONFIRMED

    # 5. Filter candidates by status
    empty = await list_candidates(mem_repo, id_repo, OWNER, PET, status=CandidateStatus.REJECTED)
    assert len(empty) == 0
