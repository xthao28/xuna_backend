"""Use-case tests for Knowledge Domains (KD-3)."""

import uuid
from datetime import UTC, datetime

import pytest

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.schemas import (
    CreateDomainRequest,
    DecideSuggestionRequest,
    TeachKnowledgeRequest,
    UpdateDomainRequest,
)
from xuna_backend.modules.knowledge.application.use_cases import (
    CreateDomainUseCase,
    DecideSuggestionUseCase,
    GetDomainUseCase,
    ListDomainsUseCase,
    ListSuggestionsUseCase,
    TeachKnowledgeUseCase,
    UpdateDomainUseCase,
)
from xuna_backend.modules.knowledge.domain.entities import (
    DomainStatus,
    KnowledgeDomain,
    KnowledgeSuggestion,
    KnowledgeSummary,
    RelationKind,
    SuggestionKind,
    SuggestionStatus,
)
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
)

PET = PetId("d945c77026fd4f4a8a899bbff4aeed86")
OWNER = uuid.uuid4()
NOW = datetime(2026, 10, 27, 10, 0, tzinfo=UTC)


class FakeKnowledgeRepo:
    def __init__(self) -> None:
        self.domains: dict[uuid.UUID, KnowledgeDomain] = {}
        self.suggestions: dict[uuid.UUID, KnowledgeSuggestion] = {}
        self.summaries: dict[uuid.UUID, KnowledgeSummary] = {}

    async def save_domain(self, domain: KnowledgeDomain) -> None:
        self.domains[domain.domain_id] = domain

    async def get_domain(self, domain_id: uuid.UUID) -> KnowledgeDomain | None:
        return self.domains.get(domain_id)

    async def list_domains(
        self, pet_id: PetId, include_deleted: bool = False
    ) -> list[KnowledgeDomain]:
        res = [d for d in self.domains.values() if d.pet_id == pet_id]
        if not include_deleted:
            res = [d for d in res if d.status != DomainStatus.DELETED]
        return sorted(
            res,
            key=lambda d: (
                d.device_slot if d.device_slot is not None else 99,
                d.created_at,
            ),
        )

    async def count_non_deleted_domains(self, pet_id: PetId) -> int:
        return len(
            [
                d
                for d in self.domains.values()
                if d.pet_id == pet_id and d.status != DomainStatus.DELETED
            ]
        )

    async def get_domain_by_slot(self, pet_id: PetId, slot: int) -> KnowledgeDomain | None:
        for d in self.domains.values():
            if d.pet_id == pet_id and d.device_slot == slot and d.status != DomainStatus.DELETED:
                return d
        return None

    async def update_domain(self, domain: KnowledgeDomain) -> None:
        self.domains[domain.domain_id] = domain

    async def save_suggestion(self, suggestion: KnowledgeSuggestion) -> None:
        self.suggestions[suggestion.suggestion_id] = suggestion

    async def get_suggestion(self, suggestion_id: uuid.UUID) -> KnowledgeSuggestion | None:
        return self.suggestions.get(suggestion_id)

    async def get_pending_suggestion(self, pet_id: PetId) -> KnowledgeSuggestion | None:
        for s in self.suggestions.values():
            if s.pet_id == pet_id and s.status == SuggestionStatus.PENDING:
                return s
        return None

    async def list_suggestions(
        self, pet_id: PetId, status: SuggestionStatus | None = None
    ) -> list[KnowledgeSuggestion]:
        res = [s for s in self.suggestions.values() if s.pet_id == pet_id]
        if status is not None:
            res = [s for s in res if s.status == status]
        return sorted(res, key=lambda s: s.created_at, reverse=True)

    async def update_suggestion(self, suggestion: KnowledgeSuggestion) -> None:
        self.suggestions[suggestion.suggestion_id] = suggestion

    async def save_summary(self, summary: KnowledgeSummary) -> None:
        self.summaries[summary.domain_id] = summary

    async def get_summary(self, domain_id: uuid.UUID) -> KnowledgeSummary | None:
        return self.summaries.get(domain_id)


class FakeCandidatePort:
    def __init__(self) -> None:
        self.candidates: dict[uuid.UUID, MemoryCandidate] = {}

    async def save_candidate(self, candidate: MemoryCandidate) -> None:
        self.candidates[candidate.candidate_id] = candidate

    async def get_candidate(self, candidate_id: uuid.UUID) -> MemoryCandidate | None:
        return self.candidates.get(candidate_id)

    async def list_candidates_by_domain(
        self, domain_id: uuid.UUID, confirmed_only: bool = True
    ) -> list[MemoryCandidate]:
        res = [c for c in self.candidates.values() if c.domain_id == domain_id]
        if confirmed_only:
            res = [
                c
                for c in res
                if c.status in (CandidateStatus.CONFIRMED, CandidateStatus.AWAITING_DEVICE_COMMIT)
            ]
        return sorted(res, key=lambda c: c.created_at, reverse=True)

    async def list_unsorted_knowledge(
        self, pet_id: PetId, confirmed_only: bool = True
    ) -> list[MemoryCandidate]:
        res = [
            c
            for c in self.candidates.values()
            if c.pet_id == pet_id
            and c.candidate_type == CandidateType.KNOWLEDGE
            and c.domain_id is None
        ]
        if confirmed_only:
            res = [
                c
                for c in res
                if c.status in (CandidateStatus.CONFIRMED, CandidateStatus.AWAITING_DEVICE_COMMIT)
            ]
        return sorted(res, key=lambda c: c.created_at, reverse=True)

    async def update_domain_for_candidates(
        self, candidate_ids: list[uuid.UUID], new_domain_id: uuid.UUID | None
    ) -> None:
        for cid in candidate_ids:
            if cid in self.candidates:
                old = self.candidates[cid]
                self.candidates[cid] = MemoryCandidate(
                    candidate_id=old.candidate_id,
                    owner_id=old.owner_id,
                    pet_id=old.pet_id,
                    source=old.source,
                    candidate_type=old.candidate_type,
                    content=old.content,
                    status=old.status,
                    created_at=old.created_at,
                    updated_at=NOW,
                    confirmed_at=old.confirmed_at,
                    committed_generation=old.committed_generation,
                    domain_id=new_domain_id,
                    relation=old.relation,
                    related_candidate_id=old.related_candidate_id,
                    fact_key=old.fact_key,
                    embedding=old.embedding,
                )


@pytest.mark.anyio
async def test_create_and_list_domains() -> None:
    repo = FakeKnowledgeRepo()
    create_uc = CreateDomainUseCase(repo)
    list_uc = ListDomainsUseCase(repo)

    req = CreateDomainRequest(name="Địa lý", description="Các quốc gia", deviceSlot=0)
    created = await create_uc.execute(PET, OWNER, req, now=NOW)
    assert created.name == "Địa lý"
    assert created.device_slot == 0
    assert created.status == "pending_device"

    # List
    domains = await list_uc.execute(PET)
    assert len(domains) == 1
    assert domains[0].domain_id == created.domain_id

    # Duplicate slot check
    with pytest.raises(ValueError, match="already occupied"):
        await create_uc.execute(
            PET, OWNER, CreateDomainRequest(name="Địa lý 2", deviceSlot=0), now=NOW
        )


@pytest.mark.anyio
async def test_domain_limit_max_5() -> None:
    repo = FakeKnowledgeRepo()
    create_uc = CreateDomainUseCase(repo)

    for i in range(5):
        await create_uc.execute(
            PET, OWNER, CreateDomainRequest(name=f"Domain {i}", deviceSlot=i), now=NOW
        )

    # 6th domain fails
    with pytest.raises(ValueError, match="Maximum 5 non-deleted domains"):
        await create_uc.execute(
            PET, OWNER, CreateDomainRequest(name="Domain 5", deviceSlot=0), now=NOW
        )


@pytest.mark.anyio
async def test_update_domain_lifecycle() -> None:
    repo = FakeKnowledgeRepo()
    candidates = FakeCandidatePort()
    create_uc = CreateDomainUseCase(repo)
    update_uc = UpdateDomainUseCase(repo, candidates)
    get_uc = GetDomainUseCase(repo)

    created = await create_uc.execute(
        PET, OWNER, CreateDomainRequest(name="Khoa học", deviceSlot=1), now=NOW
    )

    # Device ack
    acked = await update_uc.execute(
        created.domain_id, UpdateDomainRequest(action="device_ack"), now=NOW
    )
    assert acked.status == "active"

    # Rename
    renamed = await update_uc.execute(
        created.domain_id,
        UpdateDomainRequest(action="rename", name="Khoa học vũ trụ"),
        now=NOW,
    )
    assert renamed.name == "Khoa học vũ trụ"

    # Rest
    rested = await update_uc.execute(created.domain_id, UpdateDomainRequest(action="rest"), now=NOW)
    assert rested.status == "resting"
    assert rested.device_slot is None

    # Delete
    deleted = await update_uc.execute(
        created.domain_id, UpdateDomainRequest(action="delete"), now=NOW
    )
    assert deleted.status == "deleted"

    # Get domain
    res = await get_uc.execute(created.domain_id)
    assert res is not None
    assert res.status == "deleted"


@pytest.mark.anyio
async def test_teach_knowledge_lexical_assignment_and_relation() -> None:
    repo = FakeKnowledgeRepo()
    candidates = FakeCandidatePort()
    create_uc = CreateDomainUseCase(repo)
    update_uc = UpdateDomainUseCase(repo, candidates)
    teach_uc = TeachKnowledgeUseCase(repo, candidates)

    # 1. Create active domain "Địa lý"
    d = await create_uc.execute(
        PET,
        OWNER,
        CreateDomainRequest(name="Địa lý", description="Thủ đô các nước", deviceSlot=0),
        now=NOW,
    )
    await update_uc.execute(d.domain_id, UpdateDomainRequest(action="device_ack"), now=NOW)

    # 2. Teach a fact matching "Địa lý"
    res1 = await teach_uc.execute(
        PET,
        OWNER,
        TeachKnowledgeRequest(text="Hà Nội là thủ đô của nước Việt Nam"),
        now=NOW,
    )
    assert res1.proposed_domain_id == d.domain_id
    assert res1.relation == RelationKind.NEW.value
    assert res1.fact_key > 0

    # Confirm candidate in fake candidates repo
    cand1 = await candidates.get_candidate(res1.candidate_id)
    assert cand1 is not None
    await candidates.save_candidate(
        MemoryCandidate(
            candidate_id=cand1.candidate_id,
            owner_id=cand1.owner_id,
            pet_id=cand1.pet_id,
            source=cand1.source,
            candidate_type=cand1.candidate_type,
            content=cand1.content,
            status=CandidateStatus.CONFIRMED,
            created_at=cand1.created_at,
            updated_at=NOW,
            domain_id=d.domain_id,
            relation=cand1.relation,
            related_candidate_id=cand1.related_candidate_id,
            fact_key=cand1.fact_key,
        )
    )

    # 3. Teach a refining fact (very similar)
    res2 = await teach_uc.execute(
        PET,
        OWNER,
        TeachKnowledgeRequest(text="Thủ đô của nước Việt Nam là thành phố Hà Nội"),
        now=NOW,
    )
    assert res2.proposed_domain_id == d.domain_id
    assert res2.relation == RelationKind.REFINES.value
    assert res2.related_candidate_id == cand1.candidate_id


@pytest.mark.anyio
async def test_decide_suggestions() -> None:
    repo = FakeKnowledgeRepo()
    candidates = FakeCandidatePort()
    decide_uc = DecideSuggestionUseCase(repo, candidates)
    list_sugg_uc = ListSuggestionsUseCase(repo)

    # Add candidate
    cand_id = uuid.uuid4()
    await candidates.save_candidate(
        MemoryCandidate(
            candidate_id=cand_id,
            owner_id=OWNER,
            pet_id=PET,
            source="conversation",
            candidate_type=CandidateType.KNOWLEDGE,
            content={"text": "Trái Đất quay quanh Mặt Trời"},
            status=CandidateStatus.CONFIRMED,
            created_at=NOW,
            updated_at=NOW,
        )
    )

    # Add suggestion
    sugg_id = uuid.uuid4()
    await repo.save_suggestion(
        KnowledgeSuggestion(
            suggestion_id=sugg_id,
            owner_id=OWNER,
            pet_id=PET,
            kind=SuggestionKind.NEW_DOMAIN,
            proposed_name="Thiên văn",
            proposed_description="Kiến thức về các hành tinh",
            member_candidate_ids=[cand_id],
            status=SuggestionStatus.PENDING,
            created_at=NOW,
        )
    )

    # List pending suggestions
    suggs = await list_sugg_uc.execute(PET)
    assert len(suggs) == 1

    # Accept suggestion
    accepted = await decide_uc.execute(
        sugg_id,
        DecideSuggestionRequest(action="accept", deviceSlot=2),
        now=NOW,
    )
    assert accepted.status == "accepted"

    # Verify domain was created and candidate was assigned
    cand = await candidates.get_candidate(cand_id)
    assert cand is not None
    assert cand.domain_id == accepted.target_domain_id
