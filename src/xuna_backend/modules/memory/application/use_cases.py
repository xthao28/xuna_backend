"""Use cases for projections and memory candidates (CLD-14)."""

import json
import uuid
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.consent.domain.consent import ConsentClass, ConsentState
from xuna_backend.modules.exports.application.schemas import ExportPayloadV1
from xuna_backend.modules.exports.domain.export_document import (
    ExportDocumentError,
    verified_payload,
)
from xuna_backend.modules.identity.application.ports import (
    IdentityRepository,
    SignatureVerifier,
)
from xuna_backend.modules.identity.domain.device_proof import (
    ProofPurpose,
    proof_message,
)
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.identity.domain.projection_key import ProjectionKey
from xuna_backend.modules.memory.application.ports import MemoryRepository
from xuna_backend.modules.memory.application.schemas import (
    CreateCandidateRequest,
    DeviceProofDto,
    PatchCandidateRequest,
)
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    MemoryCandidate,
    Projection,
)


class MemoryError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


async def ingest_projection(
    repo: MemoryRepository,
    identity_repo: IdentityRepository,
    verifier: SignatureVerifier,
    audit: AuditLog,
    owner_id: uuid.UUID,
    pet_id: PetId,
    generation: int,
    envelope: dict[str, Any],
    device_proof: DeviceProofDto | None,
    consent: ConsentState,
    now: datetime,
    request_id: str = "",
) -> tuple[Projection, bool]:
    """Ingests a verified `xuna.export.v1` projection.

    Returns (projection, created). If identical projection already exists, returns
    (existing, False). If generation is stale or conflict occurs, raises MemoryError.
    """
    # 1. Consent check: U1 required to store any aggregate projection
    if not consent.allows(ConsentClass.AGGREGATE_METRICS):
        raise MemoryError("consent_required")

    # 2. Envelope & payload verification
    try:
        payload = verified_payload(json.dumps(envelope))
        ExportPayloadV1.model_validate(payload)
    except (ExportDocumentError, ValidationError) as error:
        raise MemoryError("invalid_envelope") from error

    payload_pet_id = payload.get("identity", {}).get("petId")
    if payload_pet_id != pet_id.value:
        raise MemoryError("pet_id_mismatch")

    payload_generation = payload.get("identity", {}).get("snapshotGeneration")
    if payload_generation != generation:
        raise MemoryError("generation_mismatch")

    # 3. Verify pet exists and is owned by caller
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise MemoryError("not_found")

    # 4. Check for existing projection (idempotency check)
    existing = await repo.get_projection(pet_id, generation)
    incoming_hash = envelope.get("sha256", "")
    if existing is not None:
        if existing.payload_sha256 == incoming_hash:
            return existing, False
        raise MemoryError("projection_conflict")

    # 5. Stale generation check
    latest = await repo.get_latest_projection(pet_id)
    if latest is not None:
        latest_key = ProjectionKey(pet_id, latest.snapshot_generation)
        incoming_key = ProjectionKey(pet_id, generation)
        if not incoming_key.is_newer_than(latest_key):
            raise MemoryError("conflict_stale_generation")

    # 6. Device proof verification (if provided)
    device_attested = False
    if device_proof is not None:
        try:
            challenge_id = uuid.UUID(device_proof.challengeId)
        except ValueError as error:
            raise MemoryError("challenge_invalid") from error

        challenge = await identity_repo.consume_challenge(challenge_id)
        if (
            challenge is None
            or challenge.consumed
            or challenge.owner_id != owner_id
            or challenge.pet_id != pet_id
            or challenge.expires_at < now
        ):
            raise MemoryError("challenge_invalid")

        try:
            pub_key_bytes = device_proof.key_bytes()
            sig_bytes = device_proof.signature_bytes()
        except ValueError as error:
            raise MemoryError("invalid_signature") from error

        if pub_key_bytes != pet.device_public_key:
            raise MemoryError("device_key_mismatch")

        compact_payload = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        msg = proof_message(
            challenge.nonce,
            pet_id,
            ProofPurpose.PROJECTION,
            generation,
            compact_payload,
        )
        if not verifier.verify(pet.device_public_key, msg, sig_bytes):
            raise MemoryError("invalid_signature")
        device_attested = True

    # 7. Privacy filtering: if U2 is not allowed, strip memories
    stored_payload = dict(payload)
    if not consent.allows(ConsentClass.EXPERIENCE_EVENTS):
        stored_payload["memories"] = []

    projection = Projection(
        projection_id=uuid.uuid4(),
        pet_id=pet_id,
        owner_id=owner_id,
        snapshot_generation=generation,
        schema_version=1,
        payload_sha256=incoming_hash,
        payload=stored_payload,
        device_attested=device_attested,
        created_at=now,
    )

    await repo.add_projection(projection)
    await identity_repo.update_generation(pet_id, generation)

    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.PROJECTION_INGESTED,
            occurred_at=now,
            subject=pet_id.value,
            request_id=request_id,
            metadata={
                "generation": generation,
                "attested": device_attested,
            },
        )
    )

    return projection, True


async def get_latest_projection(
    repo: MemoryRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    pet_id: PetId,
) -> Projection:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise MemoryError("not_found")
    proj = await repo.get_latest_projection(pet_id)
    if proj is None:
        raise MemoryError("not_found")
    return proj


async def list_projections(
    repo: MemoryRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    pet_id: PetId,
    limit: int = 50,
) -> list[Projection]:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise MemoryError("not_found")
    return await repo.list_projections(pet_id, limit=limit)


async def create_candidate(
    repo: MemoryRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    body: CreateCandidateRequest,
    now: datetime,
) -> MemoryCandidate:
    pet_id = body.parsed_pet_id()
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise MemoryError("not_found")

    candidate = MemoryCandidate(
        candidate_id=uuid.uuid4(),
        owner_id=owner_id,
        pet_id=pet_id,
        source=body.source,
        candidate_type=body.candidateType,
        content=body.content,
        status=CandidateStatus.AWAITING_OWNER_CONFIRMATION,
        created_at=now,
        updated_at=now,
        domain_id=body.domainId,
        relation=body.relation,
        related_candidate_id=body.relatedCandidateId,
        fact_key=body.factKey,
    )
    await repo.add_candidate(candidate)
    return candidate


async def patch_candidate(
    repo: MemoryRepository,
    audit: AuditLog,
    owner_id: uuid.UUID,
    candidate_id: uuid.UUID,
    body: PatchCandidateRequest,
    now: datetime,
    request_id: str = "",
) -> MemoryCandidate:
    candidate = await repo.get_candidate(candidate_id)
    if candidate is None or candidate.owner_id != owner_id:
        raise MemoryError("not_found")

    match body.action:
        case "confirm":
            if candidate.status != CandidateStatus.AWAITING_OWNER_CONFIRMATION:
                raise MemoryError("invalid_transition")
            domain_id = body.domainId if body.domainId is not None else candidate.domain_id
            updated = MemoryCandidate(
                candidate_id=candidate.candidate_id,
                owner_id=candidate.owner_id,
                pet_id=candidate.pet_id,
                source=candidate.source,
                candidate_type=candidate.candidate_type,
                content=candidate.content,
                status=CandidateStatus.AWAITING_DEVICE_COMMIT,
                created_at=candidate.created_at,
                updated_at=now,
                confirmed_at=now,
                committed_generation=candidate.committed_generation,
                domain_id=domain_id,
                relation=candidate.relation,
                related_candidate_id=candidate.related_candidate_id,
                fact_key=candidate.fact_key,
                embedding=candidate.embedding,
            )
            await repo.update_candidate(updated)
            await audit.record(
                AuditEvent(
                    owner_id=owner_id,
                    action=AuditAction.MEMORY_CANDIDATE_CONFIRMED,
                    occurred_at=now,
                    subject=candidate.pet_id.value,
                    request_id=request_id,
                    metadata={"candidateId": str(candidate_id)[:8]},
                )
            )
            return updated

        case "reject":
            if candidate.status not in (
                CandidateStatus.AWAITING_OWNER_CONFIRMATION,
                CandidateStatus.AWAITING_DEVICE_COMMIT,
            ):
                raise MemoryError("invalid_transition")
            updated = MemoryCandidate(
                candidate_id=candidate.candidate_id,
                owner_id=candidate.owner_id,
                pet_id=candidate.pet_id,
                source=candidate.source,
                candidate_type=candidate.candidate_type,
                content=candidate.content,
                status=CandidateStatus.REJECTED,
                created_at=candidate.created_at,
                updated_at=now,
                confirmed_at=candidate.confirmed_at,
                committed_generation=candidate.committed_generation,
                domain_id=candidate.domain_id,
                relation=candidate.relation,
                related_candidate_id=candidate.related_candidate_id,
                fact_key=candidate.fact_key,
                embedding=candidate.embedding,
            )
            await repo.update_candidate(updated)
            await audit.record(
                AuditEvent(
                    owner_id=owner_id,
                    action=AuditAction.MEMORY_CANDIDATE_REJECTED,
                    occurred_at=now,
                    subject=candidate.pet_id.value,
                    request_id=request_id,
                    metadata={"candidateId": str(candidate_id)[:8]},
                )
            )
            return updated

        case "commit":
            if candidate.status != CandidateStatus.AWAITING_DEVICE_COMMIT:
                raise MemoryError("invalid_transition")
            if body.committedGeneration is None:
                raise MemoryError("generation_required")
            updated = MemoryCandidate(
                candidate_id=candidate.candidate_id,
                owner_id=candidate.owner_id,
                pet_id=candidate.pet_id,
                source=candidate.source,
                candidate_type=candidate.candidate_type,
                content=candidate.content,
                status=CandidateStatus.CONFIRMED,
                created_at=candidate.created_at,
                updated_at=now,
                confirmed_at=candidate.confirmed_at,
                committed_generation=body.committedGeneration,
                domain_id=candidate.domain_id,
                relation=candidate.relation,
                related_candidate_id=candidate.related_candidate_id,
                fact_key=candidate.fact_key,
                embedding=candidate.embedding,
            )
            await repo.update_candidate(updated)
            return updated

        case "retag":
            if candidate.status in (CandidateStatus.REJECTED, CandidateStatus.DELETED):
                raise MemoryError("invalid_transition")
            updated = MemoryCandidate(
                candidate_id=candidate.candidate_id,
                owner_id=candidate.owner_id,
                pet_id=candidate.pet_id,
                source=candidate.source,
                candidate_type=candidate.candidate_type,
                content=candidate.content,
                status=candidate.status,
                created_at=candidate.created_at,
                updated_at=now,
                confirmed_at=candidate.confirmed_at,
                committed_generation=candidate.committed_generation,
                domain_id=body.domainId,
                relation=candidate.relation,
                related_candidate_id=candidate.related_candidate_id,
                fact_key=candidate.fact_key,
                embedding=candidate.embedding,
            )
            await repo.update_candidate(updated)
            return updated

        case "delete":
            updated = MemoryCandidate(
                candidate_id=candidate.candidate_id,
                owner_id=candidate.owner_id,
                pet_id=candidate.pet_id,
                source=candidate.source,
                candidate_type=candidate.candidate_type,
                content=candidate.content,
                status=CandidateStatus.DELETED,
                created_at=candidate.created_at,
                updated_at=now,
                confirmed_at=candidate.confirmed_at,
                committed_generation=candidate.committed_generation,
                domain_id=candidate.domain_id,
                relation=candidate.relation,
                related_candidate_id=candidate.related_candidate_id,
                fact_key=candidate.fact_key,
                embedding=candidate.embedding,
            )
            await repo.update_candidate(updated)
            return updated


async def list_candidates(
    repo: MemoryRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    pet_id: PetId,
    status: CandidateStatus | None = None,
    limit: int = 50,
) -> list[MemoryCandidate]:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise MemoryError("not_found")
    return await repo.list_candidates(pet_id, status=status, limit=limit)
