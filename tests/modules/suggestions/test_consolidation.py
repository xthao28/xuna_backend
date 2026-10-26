"""Unit tests for sleep learning consolidation heuristics (CLD-30, O1, E15)."""

import uuid
from datetime import UTC, datetime

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.domain.entities import Projection
from xuna_backend.modules.suggestions.application.consolidation import (
    CURIOSITY_BOUNDS,
    GAMMA_BOUNDS,
    HABIT_BOUNDS,
    compute_consolidation_suggestions,
)
from xuna_backend.modules.suggestions.domain.entities import (
    SuggestionKind,
    SuggestionProvenance,
    SuggestionStatus,
)

PET_ID = PetId("d945c77026fd4f4a8a899bbff4aeed86")
BRAIN_ID = uuid.uuid4()


def test_consolidation_without_projection() -> None:
    now = datetime.now(UTC)
    suggestions = compute_consolidation_suggestions(
        pet_id=PET_ID,
        brain_id=BRAIN_ID,
        latest_projection=None,
        now=now,
    )
    assert len(suggestions) == 3
    targets = {s.target: s for s in suggestions}
    assert "learner.gamma" in targets
    assert "learner.curiosity_bonus" in targets
    assert "habit.evening" in targets

    for s in suggestions:
        assert s.status == SuggestionStatus.PENDING
        assert s.provenance == SuggestionProvenance.CLOUD_INFERRED
        assert s.pet_id == PET_ID
        assert s.brain_id == BRAIN_ID
        assert s.created_at == now
        assert s.expires_at > now


def test_consolidation_with_active_projection() -> None:
    now = datetime.now(UTC)
    proj = Projection(
        projection_id=uuid.uuid4(),
        pet_id=PET_ID,
        owner_id=uuid.uuid4(),
        snapshot_generation=12,
        schema_version=1,
        payload_sha256="0" * 64,
        payload={
            "life": {
                "activeSeconds": 600,
                "feedbackCount": 10,
                "context": "play",
            },
            "memories": [{"type": "interaction", "timestamp": 123}],
        },
        device_attested=True,
        created_at=now,
    )
    suggestions = compute_consolidation_suggestions(
        pet_id=PET_ID,
        brain_id=BRAIN_ID,
        latest_projection=proj,
        now=now,
    )
    targets = {s.target: s for s in suggestions}

    gamma_s = targets["learner.gamma"]
    assert gamma_s.kind == SuggestionKind.LEARNER_PARAMETER
    assert gamma_s.delta == 0.02
    assert gamma_s.bounds == GAMMA_BOUNDS
    assert gamma_s.evidence_ref == "projection:gen:12"

    curiosity_s = targets["learner.curiosity_bonus"]
    assert curiosity_s.kind == SuggestionKind.LEARNER_PARAMETER
    assert curiosity_s.delta == 0.05
    assert curiosity_s.bounds == CURIOSITY_BOUNDS

    habit_s = targets["habit.morning"]
    assert habit_s.kind == SuggestionKind.HABIT_PRIOR
    assert habit_s.delta == 0.10
    assert habit_s.bounds == HABIT_BOUNDS
    assert habit_s.evidence_ref == "context:play"
