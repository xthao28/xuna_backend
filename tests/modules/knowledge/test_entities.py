"""Unit tests for knowledge entities and hashing (KD-3)."""

import uuid
from datetime import UTC, datetime

import pytest

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.domain.entities import (
    DomainOrigin,
    DomainStatus,
    KnowledgeDomain,
    KnowledgeSuggestion,
    KnowledgeSummary,
    SuggestionKind,
    SuggestionStatus,
    compute_fact_key,
)
from xuna_backend.modules.knowledge.domain.packs import PACKS, SafetyTier, get_pack

PET = PetId("d945c77026fd4f4a8a899bbff4aeed86")
OWNER = uuid.uuid4()
NOW = datetime(2026, 10, 27, 10, 0, tzinfo=UTC)


def test_compute_fact_key() -> None:
    # Deterministic u32 hash
    k1 = compute_fact_key("Hà Nội là thủ đô của Việt Nam")
    k2 = compute_fact_key("Hà Nội là thủ đô của Việt Nam")
    assert k1 == k2
    assert 1 <= k1 <= 0xFFFFFFFF

    # NFC normalization stability
    # composed vs decomposed
    import unicodedata

    text_nfc = unicodedata.normalize("NFC", "Hà Nội")
    text_nfd = unicodedata.normalize("NFD", "Hà Nội")
    assert compute_fact_key(text_nfc) == compute_fact_key(text_nfd)


def test_knowledge_domain_validation() -> None:
    domain_id = uuid.uuid4()
    domain = KnowledgeDomain(
        domain_id=domain_id,
        owner_id=OWNER,
        pet_id=PET,
        name="Địa lý",
        description="Kiến thức về các quốc gia và thủ đô",
        status=DomainStatus.ACTIVE,
        origin=DomainOrigin.OWNER,
        created_at=NOW,
        updated_at=NOW,
        device_slot=0,
    )
    assert domain.name == "Địa lý"
    assert domain.device_slot == 0

    # Name too long
    with pytest.raises(ValueError, match="name must be 1 to 40"):
        KnowledgeDomain(
            domain_id=domain_id,
            owner_id=OWNER,
            pet_id=PET,
            name="a" * 41,
            description="",
            status=DomainStatus.ACTIVE,
            origin=DomainOrigin.OWNER,
            created_at=NOW,
            updated_at=NOW,
        )

    # Invalid device slot
    with pytest.raises(ValueError, match="device_slot must be in 0..4"):
        KnowledgeDomain(
            domain_id=domain_id,
            owner_id=OWNER,
            pet_id=PET,
            name="Test",
            description="",
            status=DomainStatus.ACTIVE,
            origin=DomainOrigin.OWNER,
            created_at=NOW,
            updated_at=NOW,
            device_slot=5,
        )


def test_knowledge_suggestion_validation() -> None:
    suggestion = KnowledgeSuggestion(
        suggestion_id=uuid.uuid4(),
        owner_id=OWNER,
        pet_id=PET,
        kind=SuggestionKind.NEW_DOMAIN,
        proposed_name="Lịch sử",
        proposed_description="Các triều đại Việt Nam",
        member_candidate_ids=[uuid.uuid4()],
        status=SuggestionStatus.PENDING,
        created_at=NOW,
    )
    assert suggestion.proposed_name == "Lịch sử"
    assert suggestion.status == SuggestionStatus.PENDING


def test_knowledge_summary_validation() -> None:
    domain_id = uuid.uuid4()
    summary = KnowledgeSummary(
        domain_id=domain_id,
        text="Hà Nội là thủ đô của Việt Nam.",
        source_candidate_ids=[uuid.uuid4()],
        generated_at=NOW,
    )
    assert summary.domain_id == domain_id
    assert not summary.stale


def test_knowledge_packs_and_domain_brief_validation() -> None:
    expected_ids = ("embedded", "software", "finance", "marketing", "law", "biomed", "language")
    assert tuple(p.pack_id for p in PACKS) == expected_ids
    for p in PACKS:
        assert 1 <= len(p.name) <= 40
        assert 1 <= len(p.description) <= 240
        assert 1 <= len(p.brief) <= 1500
        assert get_pack(p.pack_id) == p

    finance = get_pack("finance")
    law = get_pack("law")
    biomed = get_pack("biomed")
    embedded = get_pack("embedded")
    assert finance is not None and finance.safety_tier == SafetyTier.HIGH_STAKES
    assert law is not None and law.safety_tier == SafetyTier.HIGH_STAKES
    assert biomed is not None and biomed.safety_tier == SafetyTier.HIGH_STAKES
    assert embedded is not None and embedded.safety_tier == SafetyTier.GENERAL
    assert get_pack("unknown") is None

    domain_id = uuid.uuid4()
    with pytest.raises(ValueError, match="expert_brief must be at most 1500"):
        KnowledgeDomain(
            domain_id=domain_id,
            owner_id=OWNER,
            pet_id=PET,
            name="Test",
            description="",
            status=DomainStatus.ACTIVE,
            origin=DomainOrigin.OWNER,
            created_at=NOW,
            updated_at=NOW,
            expert_brief="x" * 1501,
        )

    with pytest.raises(ValueError, match="safety_tier must be"):
        KnowledgeDomain(
            domain_id=domain_id,
            owner_id=OWNER,
            pet_id=PET,
            name="Test",
            description="",
            status=DomainStatus.ACTIVE,
            origin=DomainOrigin.OWNER,
            created_at=NOW,
            updated_at=NOW,
            safety_tier="invalid",
        )
