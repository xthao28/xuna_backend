"""Unit tests for suggestion use cases (CLD-30, CLD-31, H1-H8)."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.consent.domain.consent import ConsentState
from xuna_backend.modules.identity.domain.entities import Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin
from xuna_backend.modules.memory.domain.entities import Projection
from xuna_backend.modules.suggestions.application.use_cases import (
    SuggestionError,
    consolidate_suggestions,
    delete_suggestion,
    get_suggestion,
    list_suggestions,
    update_suggestion_status,
)
from xuna_backend.modules.suggestions.domain.entities import (
    CloudSuggestion,
    SuggestionKind,
    SuggestionProvenance,
    SuggestionStatus,
)

PET_ID = PetId("d945c77026fd4f4a8a899bbff4aeed86")
OWNER_ID = uuid.uuid4()
OTHER_OWNER = uuid.uuid4()
BRAIN_ID = uuid.uuid4()


class MemorySuggestionRepo:
    def __init__(self) -> None:
        self.suggestions: dict[uuid.UUID, CloudSuggestion] = {}

    async def save(self, suggestion: CloudSuggestion) -> None:
        self.suggestions[suggestion.suggestion_id] = suggestion

    async def get_by_id(self, suggestion_id: uuid.UUID) -> CloudSuggestion | None:
        return self.suggestions.get(suggestion_id)

    async def list_for_pet(
        self,
        pet_id: PetId,
        status: SuggestionStatus | None = None,
        kind: SuggestionKind | None = None,
    ) -> list[CloudSuggestion]:
        res = [s for s in self.suggestions.values() if s.pet_id == pet_id]
        if status:
            res = [s for s in res if s.status == status]
        if kind:
            res = [s for s in res if s.kind == kind]
        return res

    async def update_status(
        self,
        suggestion_id: uuid.UUID,
        status: SuggestionStatus,
        applied_at: datetime | None = None,
    ) -> CloudSuggestion | None:
        s = self.suggestions.get(suggestion_id)
        if not s:
            return None
        updated = CloudSuggestion(
            suggestion_id=s.suggestion_id,
            pet_id=s.pet_id,
            brain_id=s.brain_id,
            kind=s.kind,
            target=s.target,
            delta=s.delta,
            bounds=s.bounds,
            provenance=s.provenance,
            evidence_ref=s.evidence_ref,
            status=status,
            expires_at=s.expires_at,
            created_at=s.created_at,
            applied_at=applied_at,
        )
        self.suggestions[suggestion_id] = updated
        return updated

    async def delete(self, suggestion_id: uuid.UUID) -> bool:
        if suggestion_id in self.suggestions:
            del self.suggestions[suggestion_id]
            return True
        return False


class StubIdentityRepo:
    def __init__(self) -> None:
        self.pet = Pet(
            pet_id=PET_ID,
            owner_id=OWNER_ID,
            origin=PetOrigin.HATCHED_ON_DEVICE,
            active_brain_id=BRAIN_ID,
            device_id=uuid.uuid4(),
            device_public_key=b"k" * 64,
            last_generation=1,
            bound_at=datetime.now(UTC),
        )

    async def get_pet(self, pet_id: PetId) -> Pet | None:
        if pet_id == PET_ID:
            return self.pet
        return None

    async def add_challenge(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def consume_challenge(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def list_pets(self, *args: Any, **kwargs: Any) -> list[Pet]:
        return [self.pet]

    async def create_pet(self, *args: Any, **kwargs: Any) -> Pet:
        return self.pet

    async def update_generation(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def rebind_device(self, *args: Any, **kwargs: Any) -> Pet:
        return self.pet

    async def rotate_brain(self, *args: Any, **kwargs: Any) -> Pet:
        return self.pet


class StubMemoryRepo:
    def __init__(self) -> None:
        self.projection: Projection | None = None

    async def get_latest_projection(self, pet_id: PetId) -> Projection | None:
        return self.projection

    async def add_projection(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def get_projection(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def list_projections(self, *args: Any, **kwargs: Any) -> list[Projection]:
        return []

    async def add_candidate(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def get_candidate(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def update_candidate(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def list_candidates(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class StubAuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


@pytest.mark.anyio
async def test_consolidation_requires_consent() -> None:
    repo = MemorySuggestionRepo()
    id_repo = StubIdentityRepo()
    mem_repo = StubMemoryRepo()
    audit = StubAuditLog()

    # U0 (no consent)
    u0 = ConsentState()
    with pytest.raises(SuggestionError) as exc_info:
        await consolidate_suggestions(
            repo=repo,
            identity_repo=id_repo,
            memory_repo=mem_repo,
            audit=audit,
            owner_id=OWNER_ID,
            pet_id=PET_ID,
            consent=u0,
            now=datetime.now(UTC),
        )
    assert exc_info.value.code == "consent_required"


@pytest.mark.anyio
async def test_consolidation_succeeds_with_u1() -> None:
    repo = MemorySuggestionRepo()
    id_repo = StubIdentityRepo()
    mem_repo = StubMemoryRepo()
    audit = StubAuditLog()

    u1 = ConsentState(aggregate_metrics=True)
    now = datetime.now(UTC)
    suggestions = await consolidate_suggestions(
        repo=repo,
        identity_repo=id_repo,
        memory_repo=mem_repo,
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        consent=u1,
        now=now,
    )
    assert len(suggestions) == 3
    assert len(repo.suggestions) == 3
    assert len(audit.events) == 1
    assert audit.events[0].action == AuditAction.SUGGESTION_CONSOLIDATED


@pytest.mark.anyio
async def test_consolidation_unknown_or_unowned_pet() -> None:
    repo = MemorySuggestionRepo()
    id_repo = StubIdentityRepo()
    mem_repo = StubMemoryRepo()
    audit = StubAuditLog()
    u1 = ConsentState(aggregate_metrics=True)

    with pytest.raises(SuggestionError) as exc_info:
        await consolidate_suggestions(
            repo=repo,
            identity_repo=id_repo,
            memory_repo=mem_repo,
            audit=audit,
            owner_id=OTHER_OWNER,
            pet_id=PET_ID,
            consent=u1,
            now=datetime.now(UTC),
        )
    assert exc_info.value.code == "not_found"


@pytest.mark.anyio
async def test_suggestion_lifecycle() -> None:
    repo = MemorySuggestionRepo()
    id_repo = StubIdentityRepo()
    audit = StubAuditLog()
    now = datetime.now(UTC)

    s = CloudSuggestion(
        suggestion_id=uuid.uuid4(),
        pet_id=PET_ID,
        brain_id=BRAIN_ID,
        kind=SuggestionKind.LEARNER_PARAMETER,
        target="learner.gamma",
        delta=0.02,
        bounds={"min": 0.70, "max": 0.95},
        provenance=SuggestionProvenance.CLOUD_INFERRED,
        expires_at=now + timedelta(days=7),
        created_at=now,
    )
    await repo.save(s)

    # 1. Apply suggestion
    applied = await update_suggestion_status(
        repo=repo,
        identity_repo=id_repo,
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        suggestion_id=s.suggestion_id,
        action="apply",
        now=now,
    )
    assert applied.status == SuggestionStatus.APPLIED
    assert applied.applied_at == now
    assert audit.events[-1].action == AuditAction.SUGGESTION_APPLIED

    # 2. Revert applied suggestion
    reverted = await update_suggestion_status(
        repo=repo,
        identity_repo=id_repo,
        audit=audit,
        owner_id=OWNER_ID,
        pet_id=PET_ID,
        suggestion_id=s.suggestion_id,
        action="revert",
        now=now,
    )
    assert reverted.status == SuggestionStatus.REVERTED
    assert audit.events[-1].action == AuditAction.SUGGESTION_REVERTED

    # 3. Invalid transition from REVERTED to APPLIED
    with pytest.raises(SuggestionError) as exc_info:
        await update_suggestion_status(
            repo=repo,
            identity_repo=id_repo,
            audit=audit,
            owner_id=OWNER_ID,
            pet_id=PET_ID,
            suggestion_id=s.suggestion_id,
            action="apply",
            now=now,
        )
    assert exc_info.value.code == "invalid_transition"


@pytest.mark.anyio
async def test_apply_expired_suggestion_fails() -> None:
    repo = MemorySuggestionRepo()
    id_repo = StubIdentityRepo()
    audit = StubAuditLog()
    now = datetime.now(UTC)

    expired_s = CloudSuggestion(
        suggestion_id=uuid.uuid4(),
        pet_id=PET_ID,
        brain_id=BRAIN_ID,
        kind=SuggestionKind.LEARNER_PARAMETER,
        target="learner.gamma",
        delta=0.02,
        bounds={"min": 0.70, "max": 0.95},
        provenance=SuggestionProvenance.CLOUD_INFERRED,
        expires_at=now - timedelta(seconds=1),
        created_at=now - timedelta(days=7),
    )
    await repo.save(expired_s)

    with pytest.raises(SuggestionError) as exc_info:
        await update_suggestion_status(
            repo=repo,
            identity_repo=id_repo,
            audit=audit,
            owner_id=OWNER_ID,
            pet_id=PET_ID,
            suggestion_id=expired_s.suggestion_id,
            action="apply",
            now=now,
        )
    assert exc_info.value.code == "suggestion_expired"


@pytest.mark.anyio
async def test_delete_suggestion() -> None:
    repo = MemorySuggestionRepo()
    id_repo = StubIdentityRepo()
    now = datetime.now(UTC)

    s = CloudSuggestion(
        suggestion_id=uuid.uuid4(),
        pet_id=PET_ID,
        brain_id=BRAIN_ID,
        kind=SuggestionKind.LEARNER_PARAMETER,
        target="learner.gamma",
        delta=0.02,
        bounds={"min": 0.70, "max": 0.95},
        provenance=SuggestionProvenance.CLOUD_INFERRED,
        expires_at=now + timedelta(days=7),
        created_at=now,
    )
    await repo.save(s)

    assert await get_suggestion(repo, id_repo, OWNER_ID, PET_ID, s.suggestion_id) == s
    assert len(await list_suggestions(repo, id_repo, OWNER_ID, PET_ID)) == 1

    deleted = await delete_suggestion(repo, id_repo, OWNER_ID, PET_ID, s.suggestion_id)
    assert deleted is True
    assert len(await list_suggestions(repo, id_repo, OWNER_ID, PET_ID)) == 0
