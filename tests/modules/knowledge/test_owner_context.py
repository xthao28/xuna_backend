"""Unit tests for BuildOwnerContextUseCase (KX-1 §6.6)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from tests.modules.knowledge.test_use_cases import FakeCandidatePort, FakeKnowledgeRepo
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.owner_context import (
    CASE_CHARS,
    CONVENTION_CHARS,
    MAX_CASES,
    MAX_UNSORTED,
    BuildOwnerContextUseCase,
)
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
)

PET = PetId("d945c77026fd4f4a8a899bbff4aeed86")
OWNER = uuid.uuid4()
NOW = datetime(2026, 10, 29, 10, 0, tzinfo=UTC)


def _make_domain(
    name: str,
    description: str,
    *,
    slot: int | None = 0,
    status: DomainStatus = DomainStatus.ACTIVE,
) -> KnowledgeDomain:
    return KnowledgeDomain(
        domain_id=uuid.uuid4(),
        owner_id=OWNER,
        pet_id=PET,
        name=name,
        description=description,
        status=status,
        origin=DomainOrigin.OWNER,
        created_at=NOW,
        updated_at=NOW,
        device_slot=slot,
    )


def _make_candidate(
    text: str,
    *,
    domain_id: uuid.UUID | None = None,
    kind: str | None = "convention",
    age_days: int = 0,
) -> MemoryCandidate:
    content: dict[str, object] = {"text": text, "summary": text}
    if kind is not None:
        content["kind"] = kind
    ts = NOW - timedelta(days=age_days)
    return MemoryCandidate(
        candidate_id=uuid.uuid4(),
        owner_id=OWNER,
        pet_id=PET,
        source="owner",
        candidate_type=CandidateType.KNOWLEDGE,
        content=content,
        status=CandidateStatus.CONFIRMED,
        created_at=ts,
        updated_at=ts,
        confirmed_at=ts,
        domain_id=domain_id,
    )


@pytest.mark.anyio
async def test_explicit_domain_selection_and_level_from_slot() -> None:
    repo = FakeKnowledgeRepo()
    cands = FakeCandidatePort()
    d1 = _make_domain("Lập trình nhúng", "ESP32, FreeRTOS, C++", slot=0)
    d2 = _make_domain("Tài chính", "Kế toán, thuế, đầu tư", slot=1)
    await repo.save_domain(d1)
    await repo.save_domain(d2)
    await cands.save_candidate(_make_candidate("Dự án dùng ESP-IDF 5.2", domain_id=d1.domain_id))

    uc = BuildOwnerContextUseCase(repo, cands)
    slots = (
        DeviceSlotView(slot=0, level=2, state="active"),
        DeviceSlotView(slot=1, level=1, state="active"),
    )
    # Even if text is generic, explicit domain_id selects d1 with level=2
    ctx = await uc.execute(
        PET,
        "Hôm nay mình cần hỏi một việc",
        device_slots=slots,
        explicit_domain_id=d1.domain_id,
        now=NOW,
    )
    assert ctx.domain is not None
    assert ctx.domain.domain_id == d1.domain_id
    assert ctx.domain.level == 2
    assert ctx.domain_explicit is True
    assert ctx.conventions == ("Dự án dùng ESP-IDF 5.2",)


@pytest.mark.anyio
async def test_filters_out_domains_with_resting_or_empty_device_slot() -> None:
    repo = FakeKnowledgeRepo()
    cands = FakeCandidatePort()
    d_resting_on_board = _make_domain("Lập trình nhúng", "Vi điều khiển ESP32, RTOS", slot=0)
    d_active_on_board = _make_domain("Lập trình phần mềm", "Backend Python FastAPI", slot=1)
    await repo.save_domain(d_resting_on_board)
    await repo.save_domain(d_active_on_board)

    uc = BuildOwnerContextUseCase(repo, cands)
    slots = (
        DeviceSlotView(slot=0, level=3, state="resting"),
        DeviceSlotView(slot=1, level=1, state="active"),
    )
    # Query about ESP32 will NOT match slot 0 because slot 0 is resting on device
    ctx = await uc.execute(
        PET,
        "Tại sao mạch vi điều khiển ESP32 của mình bị lỗi?",
        device_slots=slots,
        explicit_domain_id=d_resting_on_board.domain_id,
        now=NOW,
    )
    assert ctx.domain is None
    assert ctx.domain_explicit is False


@pytest.mark.anyio
async def test_caps_conventions_cases_and_unsorted_without_truncating_items() -> None:
    repo = FakeKnowledgeRepo()
    cands = FakeCandidatePort()
    d1 = _make_domain("Lập trình nhúng", "Vi điều khiển ESP32, firmware C++, RTOS", slot=0)
    await repo.save_domain(d1)

    # Add 8 conventions (each 250 chars -> at most 4 fit in 1200 chars!)
    for i in range(8):
        text = f"Quy ước số {i} về ESP32: " + ("x" * 220)
        await cands.save_candidate(_make_candidate(text, domain_id=d1.domain_id, kind="convention"))

    # Add 5 cases (each 250 chars -> at most 3 fit in MAX_CASES=3 and 900 chars)
    for i in range(5):
        text = f"Vấn đề {i} ESP32: " + ("y" * 220)
        await cands.save_candidate(_make_candidate(text, domain_id=d1.domain_id, kind="case"))

    # Document kind is ignored in KX-1 conventions/cases
    await cands.save_candidate(
        _make_candidate("Tài liệu datasheet ESP32", domain_id=d1.domain_id, kind="document")
    )

    uc = BuildOwnerContextUseCase(repo, cands)
    ctx = await uc.execute(
        PET,
        "Kiểm tra quy ước ESP32 giúp mình",
        device_slots=None,
        explicit_domain_id=None,
        now=NOW,
    )
    assert ctx.domain is not None
    assert ctx.domain.level is None
    assert len(ctx.conventions) == 4  # 4 * ~242 = 968 <= 1200; 5th would exceed 1200
    assert sum(len(c) for c in ctx.conventions) <= CONVENTION_CHARS
    assert all(c.endswith("x" * 20) for c in ctx.conventions)  # never cut mid-item
    assert len(ctx.cases) == MAX_CASES
    assert sum(len(c) for c in ctx.cases) <= CASE_CHARS


@pytest.mark.anyio
async def test_unsorted_conventions_added_when_matching_query() -> None:
    repo = FakeKnowledgeRepo()
    cands = FakeCandidatePort()

    for i in range(4):
        await cands.save_candidate(
            _make_candidate(
                f"Dự án nhúng ESP32 dùng chuẩn cấu hình GPIO mục {i}",
                domain_id=None,
                kind=None,  # legacy candidate without kind -> treated as convention
            )
        )
    await cands.save_candidate(
        _make_candidate("Món canh chua cá lóc nấu với me", domain_id=None, kind="convention")
    )

    uc = BuildOwnerContextUseCase(repo, cands)
    ctx = await uc.execute(
        PET,
        "Cho mình hỏi về cấu hình GPIO trong dự án nhúng ESP32",
        device_slots=None,
        explicit_domain_id=None,
        now=NOW,
    )
    assert ctx.domain is None
    assert len(ctx.conventions) == MAX_UNSORTED
    assert all("ESP32" in c for c in ctx.conventions)
