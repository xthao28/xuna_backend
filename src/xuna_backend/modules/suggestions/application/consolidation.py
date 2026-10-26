"""CLD-30 Sleep learning and batch consolidation heuristics (E15, O1, H1-H8).

Analyzes pet projection and interaction history under U1/U2 consent to generate
bounded, expiring suggestions. Adheres strictly to:
- H2: Device authority.
- H3: Bounded parameters, habit priors, and safe bounds.
- H4: Owner evidence wins; parameters/priors only, never overrides direct feedback.
- H5: Provenance is cloud_inferred.
- H7: Expiring suggestions (default 7 days).
"""

import uuid
from datetime import datetime, timedelta
from typing import Any

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.domain.entities import Projection
from xuna_backend.modules.suggestions.domain.entities import (
    CloudSuggestion,
    SuggestionKind,
    SuggestionProvenance,
    SuggestionStatus,
)

SUGGESTION_TTL = timedelta(days=7)

# Safe bounds for learner parameters (E10, E15, H3)
GAMMA_BOUNDS: dict[str, float] = {"min": 0.70, "max": 0.95}
CURIOSITY_BOUNDS: dict[str, float] = {"min": 0.05, "max": 0.30}
HABIT_BOUNDS: dict[str, float] = {"min": 0.0, "max": 1.0}


def compute_consolidation_suggestions(
    pet_id: PetId,
    brain_id: uuid.UUID,
    latest_projection: Projection | None,
    now: datetime,
) -> list[CloudSuggestion]:
    """Computes bounded suggestions based on history and latest projection."""
    suggestions: list[CloudSuggestion] = []
    expires_at = now + SUGGESTION_TTL

    payload: dict[str, Any] = latest_projection.payload if latest_projection else {}
    life: dict[str, Any] = payload.get("life", {})
    feedback_count = int(life.get("feedbackCount", 0))
    active_seconds = int(life.get("activeSeconds", 0))
    memories: list[Any] = payload.get("memories", [])
    gen = latest_projection.snapshot_generation if latest_projection else 0

    evidence_ref = f"projection:gen:{gen}" if latest_projection else "initial_sleep_learning"

    # 1. Learner Gamma tuning (responsiveness vs stability)
    # If pet has accumulated feedback, slightly tune gamma toward optimal responsiveness
    gamma_delta = 0.02 if feedback_count >= 5 else 0.01
    suggestions.append(
        CloudSuggestion(
            suggestion_id=uuid.uuid4(),
            pet_id=pet_id,
            brain_id=brain_id,
            kind=SuggestionKind.LEARNER_PARAMETER,
            target="learner.gamma",
            delta=gamma_delta,
            bounds=GAMMA_BOUNDS,
            provenance=SuggestionProvenance.CLOUD_INFERRED,
            evidence_ref=evidence_ref,
            status=SuggestionStatus.PENDING,
            expires_at=expires_at,
            created_at=now,
        )
    )

    # 2. Curiosity Bonus tuning (exploration incentive)
    # Higher activity suggests readiness for richer exploration
    curiosity_delta = 0.05 if (active_seconds > 120 or len(memories) > 0) else 0.02
    suggestions.append(
        CloudSuggestion(
            suggestion_id=uuid.uuid4(),
            pet_id=pet_id,
            brain_id=brain_id,
            kind=SuggestionKind.LEARNER_PARAMETER,
            target="learner.curiosity_bonus",
            delta=curiosity_delta,
            bounds=CURIOSITY_BOUNDS,
            provenance=SuggestionProvenance.CLOUD_INFERRED,
            evidence_ref=evidence_ref,
            status=SuggestionStatus.PENDING,
            expires_at=expires_at,
            created_at=now,
        )
    )

    # 3. Habit Prior tuning (daypart anticipation)
    # Target habit slots based on context or general circadian anticipation
    context = str(life.get("context", "quiet"))
    habit_target = "habit.morning" if context == "play" else "habit.evening"
    suggestions.append(
        CloudSuggestion(
            suggestion_id=uuid.uuid4(),
            pet_id=pet_id,
            brain_id=brain_id,
            kind=SuggestionKind.HABIT_PRIOR,
            target=habit_target,
            delta=0.10,
            bounds=HABIT_BOUNDS,
            provenance=SuggestionProvenance.CLOUD_INFERRED,
            evidence_ref=f"context:{context}",
            status=SuggestionStatus.PENDING,
            expires_at=expires_at,
            created_at=now,
        )
    )

    return suggestions
