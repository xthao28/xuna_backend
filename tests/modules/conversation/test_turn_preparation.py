"""Unit tests for prepare_turn (KX-1 §6.7)."""

import uuid
from datetime import UTC, datetime

import pytest

from tests.modules.knowledge.test_use_cases import FakeCandidatePort, FakeKnowledgeRepo
from xuna_backend.modules.conversation.application.chat_turn import TurnContext
from xuna_backend.modules.conversation.application.turn_preparation import prepare_turn
from xuna_backend.modules.conversation.domain.turn_mode import (
    Channel,
    ModeReason,
    RequestedMode,
    TurnMode,
)
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.owner_context import BuildOwnerContextUseCase
from xuna_backend.modules.knowledge.domain.entities import (
    DomainOrigin,
    DomainStatus,
    KnowledgeDomain,
)
from xuna_backend.modules.knowledge.domain.owner_context import DeviceSlotView
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
    Projection,
)

PET = PetId("d945c77026fd4f4a8a899bbff4aeed86")
OWNER = uuid.uuid4()
NOW = datetime(2026, 10, 29, 10, 0, tzinfo=UTC)


class _FakeMemoryRepo:
    def __init__(self, candidates: list[MemoryCandidate], fail: bool = False) -> None:
        self._candidates = candidates
        self._fail = fail

    async def add_projection(self, projection: Projection) -> None: ...
    async def get_projection(self, pet_id: PetId, generation: int) -> Projection | None:
        return None

    async def get_latest_projection(self, pet_id: PetId) -> Projection | None:
        return None

    async def list_projections(self, pet_id: PetId, limit: int = 50) -> list[Projection]:
        return []

    async def add_candidate(self, candidate: MemoryCandidate) -> None: ...
    async def get_candidate(self, candidate_id: uuid.UUID) -> MemoryCandidate | None:
        return None

    async def update_candidate(self, candidate: MemoryCandidate) -> None: ...
    async def list_candidates(
        self,
        pet_id: PetId,
        status: CandidateStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        if self._fail:
            raise RuntimeError("db error")
        return [
            c
            for c in self._candidates
            if c.pet_id == pet_id and (status is None or c.status == status)
        ]


@pytest.mark.anyio
async def test_prepare_turn_combines_facts_owner_context_and_mode() -> None:
    k_repo = FakeKnowledgeRepo()
    c_port = FakeCandidatePort()
    dom = KnowledgeDomain(
        domain_id=uuid.uuid4(),
        owner_id=OWNER,
        pet_id=PET,
        name="Lập trình nhúng",
        description="Mạch vi điều khiển ESP32, xử lý lỗi reset nguồn, FreeRTOS",
        status=DomainStatus.ACTIVE,
        origin=DomainOrigin.OWNER,
        created_at=NOW,
        updated_at=NOW,
        device_slot=0,
    )
    await k_repo.save_domain(dom)
    await c_port.save_candidate(
        MemoryCandidate(
            candidate_id=uuid.uuid4(),
            owner_id=OWNER,
            pet_id=PET,
            source="teach_sheet",
            candidate_type=CandidateType.KNOWLEDGE,
            content={"text": "Mạch ESP32 của mình dùng LDO 3.3V", "kind": "convention"},
            status=CandidateStatus.CONFIRMED,
            created_at=NOW,
            updated_at=NOW,
            domain_id=dom.domain_id,
        )
    )

    fact_cand = MemoryCandidate(
        candidate_id=uuid.uuid4(),
        owner_id=OWNER,
        pet_id=PET,
        source="conversation",
        candidate_type=CandidateType.FACT,
        content={"fact": "owner_name", "value": "Thảo", "summary": "Tên của bạn là Thảo"},
        status=CandidateStatus.CONFIRMED,
        created_at=NOW,
        updated_at=NOW,
    )
    mem_repo = _FakeMemoryRepo([fact_cand])

    prepared = await prepare_turn(
        pet_id=PET,
        text="Tại sao mạch ESP32 của mình bị reset?",
        requested=RequestedMode.AUTO,
        channel=Channel.TEXT,
        base=TurnContext(),
        device_slots=(DeviceSlotView(slot=0, level=2, state="active"),),
        explicit_domain_id=None,
        memory_repo=mem_repo,
        owner_context=BuildOwnerContextUseCase(k_repo, c_port),
        now=NOW,
    )
    assert prepared.decision.mode == TurnMode.EXPERT
    assert prepared.decision.reason == ModeReason.DOMAIN_TASK
    assert prepared.domain_id == dom.domain_id
    assert prepared.context.confirmed_facts == ("Tên của bạn ấy là Thảo",)
    assert prepared.context.owner is not None
    assert prepared.context.owner.domain is not None
    assert prepared.context.owner.domain.level == 2


@pytest.mark.anyio
async def test_prepare_turn_degrades_gracefully_on_repo_errors() -> None:
    k_repo = FakeKnowledgeRepo()
    c_port = FakeCandidatePort()
    mem_repo = _FakeMemoryRepo([], fail=True)

    prepared = await prepare_turn(
        pet_id=PET,
        text="Chào XUNA",
        requested=None,
        channel=Channel.TEXT,
        base=TurnContext(),
        device_slots=None,
        explicit_domain_id=None,
        memory_repo=mem_repo,
        owner_context=BuildOwnerContextUseCase(k_repo, c_port),
        now=NOW,
    )
    assert prepared.decision.mode == TurnMode.CHAT
    assert prepared.decision.reason == ModeReason.LEGACY_CLIENT
    assert prepared.context.confirmed_facts == ()
    assert prepared.domain_id is None
