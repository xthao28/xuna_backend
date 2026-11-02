"""Pydantic request and response schemas for Symbiotic Evolution API (SYM-LIVE S-LIVE.3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DreamConsolidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = Field(default=1, ge=1, le=1)
    project: str = Field(default="xuna_pet", min_length=1, max_length=80)
    branch: str = Field(default="main", min_length=1, max_length=80)
    dirtyCount: int = Field(default=0, ge=0, le=10000)
    nextStepHint: str = Field(default="", max_length=240)
    confirmedFacts: list[str] = Field(default_factory=list, max_length=10)
    activePorts: list[int] = Field(default_factory=list, max_length=16)
    continuousWorkMinutes: int = Field(default=0, ge=0, le=1440)
    localHour: int = Field(default=10, ge=0, le=23)
    previousGitDirtyCount: int = Field(default=0, ge=0, le=10000)
    solvedCaseJustNow: bool = False
    hoursSinceLastInteraction: float = Field(default=0.0, ge=0.0, le=8760.0)


class DreamVignetteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dreamId: str
    titleVi: str
    narrativeVi: str
    dreamSymbols: list[str]
    morningCuriosityHookVi: str
    relatedProject: str
    relatedBranch: str
    createdAt: str


class CoRegulationStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: str
    petMoodHint: str
    gestureHint: str
    competenceDopamineBoost: float
    speechVi: str
    suggestedWorkflow: str | None = None


class DreamConsolidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = 1
    dreamVignette: DreamVignetteResponse
    coRegulationState: CoRegulationStateResponse


class UnstuckCoachRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = Field(default=1, ge=1, le=1)
    goal: str = Field(min_length=1, max_length=240)
    project: str = Field(default="xuna_pet", min_length=1, max_length=80)
    toolHistory: list[str] = Field(default_factory=list, max_length=50)


class SerendipityNoteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    noteId: str
    titleVi: str
    sparkVi: str
    connectedTopics: list[str]
    suggestedActionVi: str
    createdAt: str


class HabitMacroProposalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    proposalId: str
    titleVi: str
    triggerPhraseVi: str
    steps: list[str]
    occurrences: int
    maxRisk: str


class UnstuckCoachResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = 1
    goal: str
    project: str
    firstFiveMinuteStep: str
    followUpSteps: list[str]
    encouragementVi: str
    serendipityNote: SerendipityNoteResponse
    habitProposals: list[HabitMacroProposalResponse]


class RecordSocialEncounterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = Field(default=1, ge=1, le=1)
    myShortId: str = Field(min_length=4, max_length=16)
    peerShortId: str = Field(min_length=4, max_length=16)
    peerTrait: str = Field(default="playfulness", min_length=1, max_length=32)
    peerTint: str = Field(default="golden_amber", min_length=1, max_length=32)
    sharedTrick: str = Field(default="spin_dance", min_length=1, max_length=32)


class SocialEncounterNoteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    encounterId: str
    peerShortId: str
    peerTrait: str
    peerTint: str
    greetingKind: str
    narrativeVi: str
    sharedTrick: str
    occurredAt: str


class SocialPostcardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    postcardId: str
    fromShortId: str
    toShortId: str
    sharedSkillId: str
    messageVi: str
    signatureHex: str
    createdAt: str


class RecordSocialEncounterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = 1
    encounter: SocialEncounterNoteResponse
    postcard: SocialPostcardResponse


class ForgeSkillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = Field(default=1, ge=1, le=1)
    goal: str = Field(min_length=3, max_length=240)
    category: str | None = Field(default=None, max_length=40)
    targetFolder: str | None = Field(default=None, max_length=120)
    fileExtensions: list[str] = Field(default_factory=list, max_length=6)
    demonstratedAction: str | None = Field(default=None, max_length=200)


class ForgeParameterFieldResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    key: str
    labelVi: str
    defaultValue: str


class ForgedSkillBlueprintResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    skillId: str
    titleVi: str
    category: str
    goal: str
    targetFolder: str
    fileExtensions: list[str]
    detectedItemsCount: int
    dryRunSample: str
    selfDebugIterations: int
    parameterFields: list[ForgeParameterFieldResponse]
    scriptSteps: list[str]
    summaryVi: str
    createdAt: str
    primitives: list[str] = Field(default_factory=list)
    canvasSurfaceKind: str = "data_table"


class ForgeSkillResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = 1
    skill: ForgedSkillBlueprintResponse


class ClawPulseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = Field(default=1, ge=1, le=1)
    skillTitleVi: str = Field(
        default="Tải Hóa Đơn Hàng Loạt & Lập Bảng Kê", min_length=1, max_length=120
    )
    skillId: str = Field(default="skill-web-harvest-invoices-1", min_length=1, max_length=64)
    targetFolder: str = Field(default="XUNA_HoaDon/Thang_11_2026", min_length=1, max_length=120)
    correctionNotes: list[str] = Field(default_factory=list, max_length=10)
    importedSkillMarkdown: str | None = Field(default=None, max_length=4000)
    intervalMinutes: int = Field(default=30, ge=5, le=720)
    simulateDelta: bool = False


class ClawPulseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = 1
    heartbeatStatus: str
    deltaDetected: bool
    llmTokensUsed: int
    suppressedNotification: bool
    intervalMinutes: int
    skillMarkdown: str
    securityVerified: bool
    selfHealGeneration: int
    correctionRules: list[str]
    canvasTitleVi: str
    canvasColumns: list[str]
    canvasRows: list[dict[str, object]]
    summaryVi: str
