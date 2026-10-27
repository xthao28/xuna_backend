"""Tests for Knowledge Suggestion use cases (KD-4, plan §6.3.4)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.suggest import (
    ExpireKnowledgeSuggestionsUseCase,
    GenerateKnowledgeSuggestionUseCase,
)
from xuna_backend.modules.knowledge.domain.entities import (
    DomainOrigin,
    DomainStatus,
    KnowledgeDomain,
    KnowledgeSuggestion,
    KnowledgeSummary,
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
        return sorted(res, key=lambda d: d.created_at)

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
            valid_st = (
                CandidateStatus.CONFIRMED,
                CandidateStatus.AWAITING_DEVICE_COMMIT,
            )
            res = [c for c in res if c.status in valid_st]
        return res

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
            valid_st = (
                CandidateStatus.CONFIRMED,
                CandidateStatus.AWAITING_DEVICE_COMMIT,
            )
            res = [c for c in res if c.status in valid_st]
        return res

    async def update_domain_for_candidates(
        self, candidate_ids: list[uuid.UUID], new_domain_id: uuid.UUID | None
    ) -> None:
        for cid in candidate_ids:
            cand = self.candidates.get(cid)
            if cand is not None:
                self.candidates[cid] = MemoryCandidate(
                    candidate_id=cand.candidate_id,
                    owner_id=cand.owner_id,
                    pet_id=cand.pet_id,
                    source=cand.source,
                    candidate_type=cand.candidate_type,
                    content=cand.content,
                    status=cand.status,
                    created_at=cand.created_at,
                    updated_at=cand.updated_at,
                    domain_id=new_domain_id,
                )


@pytest.mark.anyio
async def test_generate_suggestion_requires_at_least_4_unsorted() -> None:
    repo = FakeKnowledgeRepo()
    cand_port = FakeCandidatePort()
    use_case = GenerateKnowledgeSuggestionUseCase(repo, cand_port)

    # 3 unsorted items (< 4)
    for i in range(3):
        cid = uuid.uuid4()
        cand_port.candidates[cid] = MemoryCandidate(
            candidate_id=cid,
            owner_id=OWNER,
            pet_id=PET,
            source="conversation",
            candidate_type=CandidateType.KNOWLEDGE,
            content={"text": f"Nấu ăn món {i} với nước mắm thơm ngon"},
            status=CandidateStatus.CONFIRMED,
            created_at=NOW,
            updated_at=NOW,
        )

    res = await use_case.execute(PET, OWNER, now=NOW)
    assert res is None


@pytest.mark.anyio
async def test_generate_suggestion_creates_new_domain() -> None:
    repo = FakeKnowledgeRepo()
    cand_port = FakeCandidatePort()
    use_case = GenerateKnowledgeSuggestionUseCase(repo, cand_port)

    # 5 unsorted items on guitar / music
    texts = [
        "Đàn guitar có 6 dây tiêu chuẩn E A D G B E",
        "Hợp âm Đô trưởng C trên guitar gồm nốt Đô Mi Sol",
        "Bấm chặn barre trên đàn guitar cần ngón trỏ thẳng",
        "Dây đàn guitar cần được chỉnh đúng cao độ mỗi ngày",
        "Học tỉa nốt guitar cổ điển giúp ngón tay linh hoạt",
    ]
    for t in texts:
        cid = uuid.uuid4()
        cand_port.candidates[cid] = MemoryCandidate(
            candidate_id=cid,
            owner_id=OWNER,
            pet_id=PET,
            source="conversation",
            candidate_type=CandidateType.KNOWLEDGE,
            content={"text": t},
            status=CandidateStatus.CONFIRMED,
            created_at=NOW,
            updated_at=NOW,
        )

    res = await use_case.execute(PET, OWNER, now=NOW)
    assert res is not None
    assert res.kind == SuggestionKind.NEW_DOMAIN
    assert res.status == SuggestionStatus.PENDING
    assert len(res.member_candidate_ids) >= 4
    assert res.target_domain_id is None
    assert "guitar" in res.proposed_name.lower() or "chủ đề" in res.proposed_name.lower()

    # Second run without deciding returns None because one is already pending
    res2 = await use_case.execute(PET, OWNER, now=NOW)
    assert res2 is None


@pytest.mark.anyio
async def test_generate_suggestion_duplicate_detection_suggests_add_to_domain() -> None:
    repo = FakeKnowledgeRepo()
    cand_port = FakeCandidatePort()
    use_case = GenerateKnowledgeSuggestionUseCase(repo, cand_port)

    # Existing domain: Cooking
    cooking_id = uuid.uuid4()
    domain = KnowledgeDomain(
        domain_id=cooking_id,
        owner_id=OWNER,
        pet_id=PET,
        name="Nấu ăn & Ẩm thực",
        description=(
            "Cách chế biến món ăn ngon, công thức nấu nướng, gia vị "
            "và mẹo nhà bếp: luộc, kho, xào, nướng, chiên rán"
        ),
        device_slot=0,
        status=DomainStatus.ACTIVE,
        origin=DomainOrigin.OWNER,
        created_at=NOW,
        updated_at=NOW,
    )
    await repo.save_domain(domain)

    # 4 unsorted items on cooking
    texts = [
        "Kho thịt ba chỉ với nước dừa giúp thịt mềm bùi",
        "Ướp cá với nước mắm và tiêu trước khi chiên rán",
        "Nấu canh chua cá lóc với cà chua và thơm ngon",
        "Xào rau muống với tỏi trên lửa lớn giữ màu xanh",
    ]
    for t in texts:
        cid = uuid.uuid4()
        cand_port.candidates[cid] = MemoryCandidate(
            candidate_id=cid,
            owner_id=OWNER,
            pet_id=PET,
            source="conversation",
            candidate_type=CandidateType.KNOWLEDGE,
            content={"text": t},
            status=CandidateStatus.CONFIRMED,
            created_at=NOW,
            updated_at=NOW,
        )

    res = await use_case.execute(PET, OWNER, now=NOW)
    assert res is not None
    assert res.kind == SuggestionKind.ADD_TO_DOMAIN
    assert res.target_domain_id == cooking_id


@pytest.mark.anyio
async def test_generate_suggestion_expires_old_pending_after_14_days() -> None:
    repo = FakeKnowledgeRepo()
    expire_use_case = ExpireKnowledgeSuggestionsUseCase(repo)

    # Pending suggestion created 15 days ago
    old_time = NOW - timedelta(days=15)
    sid = uuid.uuid4()
    old_sugg = KnowledgeSuggestion(
        suggestion_id=sid,
        owner_id=OWNER,
        pet_id=PET,
        kind=SuggestionKind.NEW_DOMAIN,
        proposed_name="Chủ đề cũ",
        proposed_description="Mô tả cũ",
        member_candidate_ids=[uuid.uuid4()],
        status=SuggestionStatus.PENDING,
        created_at=old_time,
    )
    await repo.save_suggestion(old_sugg)

    # Expire use case directly
    expired_count = await expire_use_case.execute(PET, now=NOW)
    assert expired_count == 1
    updated = await repo.get_suggestion(sid)
    assert updated is not None
    assert updated.status == SuggestionStatus.EXPIRED
