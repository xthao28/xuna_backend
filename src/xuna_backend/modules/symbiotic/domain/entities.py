"""Domain entities for NREM/REM Dream Consolidation and Co-Regulation (SYM-1 S1.2).

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class CoRegulationMode(StrEnum):
    """Shared bio-rhythm state between the owner and XUNA (SYM-1 S1.2)."""

    STEADY_COMPANION = "steady_companion"
    DROWSY_CARE = "drowsy_care"
    SHARED_BREAKTHROUGH = "shared_breakthrough"
    REUNION_ARC = "reunion_arc"


@dataclass(frozen=True, slots=True)
class DreamVignette:
    """Bio-inspired NREM/REM dream memory consolidated from the day's experiences (SYM-1 S1.2).

    Synthesizes the owner's active project/capsule, coding rhythm, and learned knowledge
    into a first-person ('mình - bạn') vignette for the Evolution Journal plus a
    proactive Morning Curiosity Hook for the next morning.
    """

    dream_id: str
    title_vi: str
    narrative_vi: str
    dream_symbols: tuple[str, ...]
    morning_curiosity_hook_vi: str
    related_project: str
    related_branch: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.dream_id.strip():
            raise ValueError("dream_id must not be empty")
        if not 1 <= len(self.title_vi.strip()) <= 120:
            raise ValueError("title_vi must be 1..120 characters")
        if not 1 <= len(self.narrative_vi.strip()) <= 500:
            raise ValueError("narrative_vi must be 1..500 characters")
        if not 1 <= len(self.morning_curiosity_hook_vi.strip()) <= 240:
            raise ValueError("morning_curiosity_hook_vi must be 1..240 characters")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

    def to_wire(self) -> dict[str, object]:
        return {
            "dreamId": self.dream_id,
            "titleVi": self.title_vi,
            "narrativeVi": self.narrative_vi,
            "dreamSymbols": list(self.dream_symbols),
            "morningCuriosityHookVi": self.morning_curiosity_hook_vi,
            "relatedProject": self.related_project,
            "relatedBranch": self.related_branch,
            "createdAt": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class CoRegulationSnapshot:
    """Evaluated homeostatic & emotional co-regulation response (SYM-1 S1.2)."""

    mode: CoRegulationMode
    pet_mood_hint: str
    gesture_hint: str
    competence_dopamine_boost: float
    speech_vi: str
    suggested_workflow: str | None = None

    def to_wire(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "petMoodHint": self.pet_mood_hint,
            "gestureHint": self.gesture_hint,
            "competenceDopamineBoost": round(self.competence_dopamine_boost, 3),
            "speechVi": self.speech_vi,
            "suggestedWorkflow": self.suggested_workflow,
        }


@dataclass(frozen=True, slots=True)
class SerendipityNote:
    """Off-screen inner life & proactive Serendipity Postcard ('Mảnh giấy nhớ của XUNA',
    SYM-2 S2.2)."""

    note_id: str
    title_vi: str
    spark_vi: str
    connected_topics: tuple[str, ...]
    suggested_action_vi: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.note_id.strip():
            raise ValueError("note_id must not be empty")
        if not 1 <= len(self.title_vi.strip()) <= 120:
            raise ValueError("title_vi must be 1..120 characters")
        if not 1 <= len(self.spark_vi.strip()) <= 500:
            raise ValueError("spark_vi must be 1..500 characters")
        if not 1 <= len(self.suggested_action_vi.strip()) <= 240:
            raise ValueError("suggested_action_vi must be 1..240 characters")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

    def to_wire(self) -> dict[str, object]:
        return {
            "noteId": self.note_id,
            "titleVi": self.title_vi,
            "sparkVi": self.spark_vi,
            "connectedTopics": list(self.connected_topics),
            "suggestedActionVi": self.suggested_action_vi,
            "createdAt": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class HabitMacroProposal:
    """Self-evolving workflow macro mined from repeated owner tool sequences (SYM-2 S2.2)."""

    proposal_id: str
    title_vi: str
    trigger_phrase_vi: str
    steps: tuple[str, ...]
    occurrences: int
    max_risk: str = "R1"

    def __post_init__(self) -> None:
        if not self.proposal_id.strip():
            raise ValueError("proposal_id must not be empty")
        if not 1 <= len(self.steps) <= 5:
            raise ValueError("steps must contain 1..5 tool names")
        if self.occurrences < 1:
            raise ValueError("occurrences must be >= 1")
        if self.max_risk not in ("R0", "R1", "R2"):
            raise ValueError("habit macro max_risk cannot exceed R2")

    def to_wire(self) -> dict[str, object]:
        return {
            "proposalId": self.proposal_id,
            "titleVi": self.title_vi,
            "triggerPhraseVi": self.trigger_phrase_vi,
            "steps": list(self.steps),
            "occurrences": self.occurrences,
            "maxRisk": self.max_risk,
        }

