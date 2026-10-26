"""Unit tests for suggestion domain entities (CLD-30, CLD-31, H1-H8)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.suggestions.domain.entities import (
    CloudSuggestion,
    SuggestionKind,
    SuggestionProvenance,
    SuggestionStatus,
)

PET_ID = PetId("d945c77026fd4f4a8a899bbff4aeed86")
BRAIN_ID = uuid.uuid4()


def test_create_valid_suggestion() -> None:
    now = datetime.now(UTC)
    suggestion = CloudSuggestion(
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
    assert suggestion.target == "learner.gamma"
    assert suggestion.status == SuggestionStatus.PENDING
    assert not suggestion.is_expired(now)
    assert suggestion.is_expired(now + timedelta(days=8))


def test_invalid_target_rejected() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="invalid suggestion target"):
        CloudSuggestion(
            suggestion_id=uuid.uuid4(),
            pet_id=PET_ID,
            brain_id=BRAIN_ID,
            kind=SuggestionKind.LEARNER_PARAMETER,
            target="invalid target with spaces!",
            delta=0.02,
            bounds={"min": 0.70, "max": 0.95},
            provenance=SuggestionProvenance.CLOUD_INFERRED,
            expires_at=now + timedelta(days=7),
            created_at=now,
        )


def test_invalid_bounds_rejected() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="bounds min"):
        CloudSuggestion(
            suggestion_id=uuid.uuid4(),
            pet_id=PET_ID,
            brain_id=BRAIN_ID,
            kind=SuggestionKind.LEARNER_PARAMETER,
            target="learner.gamma",
            delta=0.02,
            bounds={"min": 0.95, "max": 0.70},  # min > max
            provenance=SuggestionProvenance.CLOUD_INFERRED,
            expires_at=now + timedelta(days=7),
            created_at=now,
        )


def test_status_transitions() -> None:
    now = datetime.now(UTC)
    s = CloudSuggestion(
        suggestion_id=uuid.uuid4(),
        pet_id=PET_ID,
        brain_id=BRAIN_ID,
        kind=SuggestionKind.HABIT_PRIOR,
        target="habit.morning",
        delta=0.1,
        bounds={"min": 0.0, "max": 1.0},
        provenance=SuggestionProvenance.CLOUD_INFERRED,
        expires_at=now + timedelta(days=7),
        created_at=now,
    )
    # Valid transitions from pending
    assert s.can_transition_to(SuggestionStatus.APPLIED)
    assert s.can_transition_to(SuggestionStatus.REJECTED)
    assert s.can_transition_to(SuggestionStatus.EXPIRED)
    assert not s.can_transition_to(SuggestionStatus.REVERTED)

    # Valid transitions from applied
    applied = CloudSuggestion(
        suggestion_id=s.suggestion_id,
        pet_id=s.pet_id,
        brain_id=s.brain_id,
        kind=s.kind,
        target=s.target,
        delta=s.delta,
        bounds=s.bounds,
        provenance=s.provenance,
        expires_at=s.expires_at,
        created_at=s.created_at,
        status=SuggestionStatus.APPLIED,
        applied_at=now,
    )
    assert applied.can_transition_to(SuggestionStatus.REVERTED)
    assert not applied.can_transition_to(SuggestionStatus.PENDING)
    assert not applied.can_transition_to(SuggestionStatus.REJECTED)
