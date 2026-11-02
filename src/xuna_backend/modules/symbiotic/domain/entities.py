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


@dataclass(frozen=True, slots=True)
class SocialEncounterNote:
    """XUNA-to-XUNA proximity encounter journal entry ('Nhật ký Kết bạn XUNA',
    SYM-4 S4.2). Contains zero owner PII (U0-safe)."""

    encounter_id: str
    peer_short_id: str
    peer_trait: str
    peer_tint: str
    greeting_kind: str
    narrative_vi: str
    shared_trick: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not self.encounter_id.strip():
            raise ValueError("encounter_id must not be empty")
        if not 1 <= len(self.peer_short_id.strip()) <= 16:
            raise ValueError("peer_short_id must be 1..16 characters")
        if not 1 <= len(self.narrative_vi.strip()) <= 400:
            raise ValueError("narrative_vi must be 1..400 characters")
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")

    def to_wire(self) -> dict[str, object]:
        return {
            "encounterId": self.encounter_id,
            "peerShortId": self.peer_short_id,
            "peerTrait": self.peer_trait,
            "peerTint": self.peer_tint,
            "greetingKind": self.greeting_kind,
            "narrativeVi": self.narrative_vi,
            "sharedTrick": self.shared_trick,
            "occurredAt": self.occurred_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class SocialPostcard:
    """Signed skill/trick postcard exchanged between two XUNA pets (SYM-4 S4.2)."""

    postcard_id: str
    from_short_id: str
    to_short_id: str
    shared_skill_id: str
    message_vi: str
    signature_hex: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.postcard_id.strip():
            raise ValueError("postcard_id must not be empty")
        if not 1 <= len(self.from_short_id.strip()) <= 16:
            raise ValueError("from_short_id must be 1..16 characters")
        if not 1 <= len(self.to_short_id.strip()) <= 16:
            raise ValueError("to_short_id must be 1..16 characters")
        if not 1 <= len(self.message_vi.strip()) <= 240:
            raise ValueError("message_vi must be 1..240 characters")
        if len(self.signature_hex.strip()) != 64:
            raise ValueError("signature_hex must be a 64-char SHA-256 hex digest")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

    def to_wire(self) -> dict[str, object]:
        return {
            "postcardId": self.postcard_id,
            "fromShortId": self.from_short_id,
            "toShortId": self.to_short_id,
            "sharedSkillId": self.shared_skill_id,
            "messageVi": self.message_vi,
            "signatureHex": self.signature_hex,
            "createdAt": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ForgedSkillBlueprint:
    """Universal Citizen-Developer Skill Blueprint synthesized by XUNA OmniForge
    (`FORGE-2`). Combines 1-call LLM/deterministic recipe generation with a 1-item
    dry-run preview and zero-token local loop steps."""

    skill_id: str
    title_vi: str
    category: str
    goal: str
    target_folder: str
    file_extensions: tuple[str, ...]
    detected_items_count: int
    dry_run_sample: str
    self_debug_iterations: int
    parameter_fields: tuple[dict[str, str], ...]
    script_steps: tuple[str, ...]
    summary_vi: str
    created_at: datetime
    primitives: tuple[str, ...] = ()
    canvas_surface_kind: str = "data_table"

    def __post_init__(self) -> None:
        if not self.skill_id.strip():
            raise ValueError("skill_id must not be empty")
        if not 1 <= len(self.title_vi.strip()) <= 120:
            raise ValueError("title_vi must be 1..120 characters")
        if not 1 <= len(self.summary_vi.strip()) <= 240:
            raise ValueError("summary_vi must be 1..240 characters")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

    def to_wire(self) -> dict[str, object]:
        return {
            "skillId": self.skill_id,
            "titleVi": self.title_vi,
            "category": self.category,
            "goal": self.goal,
            "targetFolder": self.target_folder,
            "fileExtensions": list(self.file_extensions),
            "detectedItemsCount": self.detected_items_count,
            "dryRunSample": self.dry_run_sample,
            "selfDebugIterations": self.self_debug_iterations,
            "parameterFields": [dict(f) for f in self.parameter_fields],
            "scriptSteps": list(self.script_steps),
            "summaryVi": self.summary_vi,
            "createdAt": self.created_at.isoformat(),
            "primitives": list(self.primitives),
            "canvasSurfaceKind": self.canvas_surface_kind,
        }


@dataclass(frozen=True, slots=True)
class ClawPulseSnapshot:
    """Snapshot produced by `ClawPulseService` (`CLAW-1`-`CLAW-4`):
    combines OpenClaw-compatible `SKILL.md` bridge output, Zero-Token 2-Tier
    Heartbeat status (`HEARTBEAT_OK` / `HEARTBEAT_ALERT`), and A2UI Live Canvas."""

    heartbeat_status: str
    delta_detected: bool
    llm_tokens_used: int
    suppressed_notification: bool
    interval_minutes: int
    skill_markdown: str
    security_verified: bool
    self_heal_generation: int
    correction_rules: tuple[str, ...]
    canvas_title_vi: str
    canvas_columns: tuple[str, ...]
    canvas_rows: tuple[dict[str, object], ...]
    summary_vi: str

    def __post_init__(self) -> None:
        if self.heartbeat_status not in ("HEARTBEAT_OK", "HEARTBEAT_ALERT"):
            raise ValueError("heartbeat_status must be HEARTBEAT_OK or HEARTBEAT_ALERT")
        if self.llm_tokens_used < 0:
            raise ValueError("llm_tokens_used must be >= 0")

    def to_wire(self) -> dict[str, object]:
        return {
            "heartbeatStatus": self.heartbeat_status,
            "deltaDetected": self.delta_detected,
            "llmTokensUsed": self.llm_tokens_used,
            "suppressedNotification": self.suppressed_notification,
            "intervalMinutes": self.interval_minutes,
            "skillMarkdown": self.skill_markdown,
            "securityVerified": self.security_verified,
            "selfHealGeneration": self.self_heal_generation,
            "correctionRules": list(self.correction_rules),
            "canvasTitleVi": self.canvas_title_vi,
            "canvasColumns": list(self.canvas_columns),
            "canvasRows": [dict(r) for r in self.canvas_rows],
            "summaryVi": self.summary_vi,
        }


@dataclass(frozen=True, slots=True)
class SkillChainReceipt:
    """Multi-Skill DAG Chain & Automated 4-Point Self-Test Bench receipt (`WORKSHOP-3`)."""

    chain_id: str
    executed_skills: tuple[str, ...]
    interpolated_goals: tuple[str, ...]
    passed_assertions: int
    total_assertions: int
    self_test_passed: bool
    auto_trigger_on_delta: bool
    compact_index: tuple[dict[str, object], ...]
    undo_snapshot_id: str
    summary_vi: str

    def __post_init__(self) -> None:
        if not self.chain_id.strip():
            raise ValueError("chain_id must not be empty")
        if not 1 <= len(self.executed_skills) <= 5:
            raise ValueError("executed_skills must have 1..5 items")
        if self.passed_assertions < 0 or self.total_assertions <= 0:
            raise ValueError("invalid assertion counts")

    def to_wire(self) -> dict[str, object]:
        return {
            "chainId": self.chain_id,
            "executedSkills": list(self.executed_skills),
            "interpolatedGoals": list(self.interpolated_goals),
            "passedAssertions": self.passed_assertions,
            "totalAssertions": self.total_assertions,
            "selfTestPassed": self.self_test_passed,
            "autoTriggerOnDelta": self.auto_trigger_on_delta,
            "compactIndex": [dict(item) for item in self.compact_index],
            "undoSnapshotId": self.undo_snapshot_id,
            "summaryVi": self.summary_vi,
        }


@dataclass(frozen=True, slots=True)
class ClawMindReceipt:
    """Receipt returned by `ClawPulseService.evaluate_claw_mind` (`MIND-3`, `SWARM-3`)."""

    vault_folder: str
    flushed_facts_count: int
    redacted_secrets_count: int
    soul_markdown: str
    owner_profile_markdown: str
    memory_journal_markdown: str
    aria_snapshot_id: str
    aria_element_count: int
    token_reduction_percent: int
    compact_aria_tree: str
    subagent_worker_id: str
    subagent_tokens_used: int
    subagent_llm_tokens_used: int
    context_saved_percent: int
    critic_passed: bool
    critic_confidence_score: int
    specialist_workers: tuple[dict[str, object], ...]
    subagent_distilled_vi: str
    active_hooks: tuple[dict[str, object], ...]
    summary_vi: str

    def to_wire(self) -> dict[str, object]:
        return {
            "vaultFolder": self.vault_folder,
            "flushedFactsCount": self.flushed_facts_count,
            "redactedSecretsCount": self.redacted_secrets_count,
            "soulMarkdown": self.soul_markdown,
            "ownerProfileMarkdown": self.owner_profile_markdown,
            "memoryJournalMarkdown": self.memory_journal_markdown,
            "ariaSnapshotId": self.aria_snapshot_id,
            "ariaElementCount": self.aria_element_count,
            "tokenReductionPercent": self.token_reduction_percent,
            "compactAriaTree": self.compact_aria_tree,
            "subagentWorkerId": self.subagent_worker_id,
            "subagentTokensUsed": self.subagent_tokens_used,
            "subagentLlmTokensUsed": self.subagent_llm_tokens_used,
            "contextSavedPercent": self.context_saved_percent,
            "criticPassed": self.critic_passed,
            "criticConfidenceScore": self.critic_confidence_score,
            "specialistWorkers": [dict(item) for item in self.specialist_workers],
            "subagentDistilledVi": self.subagent_distilled_vi,
            "activeHooks": [dict(item) for item in self.active_hooks],
            "summaryVi": self.summary_vi,
        }


@dataclass(frozen=True, slots=True)
class Frontier2026Receipt:
    """Receipt returned by `ClawPulseService.evaluate_frontier_2026` (`FRONTIER-1`-`FRONTIER-4`)."""

    overlay_anchor: str
    munch_animation: str
    routed_target: str
    vocal_kind: str
    pcm_byte_count: int
    wav_sha256: str
    detected_owner_tone: str
    empathy_adjustment_vi: str
    active_edges_count: int
    historical_edges_count: int
    temporal_narrative_vi: str
    active_ritual: str
    standby_dock_active: bool
    unlocked_keepsake_vi: str
    playdate_summary_vi: str
    summary_vi: str

    def to_wire(self) -> dict[str, object]:
        return {
            "overlayAnchor": self.overlay_anchor,
            "munchAnimation": self.munch_animation,
            "routedTarget": self.routed_target,
            "vocalKind": self.vocal_kind,
            "pcmByteCount": self.pcm_byte_count,
            "wavSha256": self.wav_sha256,
            "detectedOwnerTone": self.detected_owner_tone,
            "empathyAdjustmentVi": self.empathy_adjustment_vi,
            "activeEdgesCount": self.active_edges_count,
            "historicalEdgesCount": self.historical_edges_count,
            "temporalNarrativeVi": self.temporal_narrative_vi,
            "activeRitual": self.active_ritual,
            "standbyDockActive": self.standby_dock_active,
            "unlockedKeepsakeVi": self.unlocked_keepsake_vi,
            "playdateSummaryVi": self.playdate_summary_vi,
            "summaryVi": self.summary_vi,
        }

