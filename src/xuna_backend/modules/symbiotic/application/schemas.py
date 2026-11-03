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


class WorkshopChainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    chainSkillIds: list[str] = Field(min_length=1, max_length=5)
    runtimeParams: dict[str, str] = Field(default_factory=dict)
    autoTriggerOnDelta: bool = False


class WorkshopChainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = 1
    chainId: str
    executedSkills: list[str]
    interpolatedGoals: list[str]
    passedAssertions: int
    totalAssertions: int
    selfTestPassed: bool
    autoTriggerOnDelta: bool
    compactIndex: list[dict[str, object]]
    undoSnapshotId: str
    summaryVi: str


class ClawMindRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sessionTranscript: str = Field(
        default="Chủ nhân thích báo cáo CSV ngắn gọn và kiểm tra hóa đơn mỗi sáng.",
        min_length=1,
        max_length=4000,
    )
    rawHtml: str = Field(
        default='<button>Tải hóa đơn PDF</button><a href="/export.csv">Xuất CSV</a>',
        min_length=1,
        max_length=4000,
    )
    subagentGoal: str = Field(
        default="Quét sâu 30 CV ứng viên và lọc top 5 ứng viên",
        min_length=1,
        max_length=400,
    )
    tokenBudget: int = Field(default=2048, ge=256, le=8192)
    lifecycleEvent: str = Field(default="on_morning_wake", min_length=1, max_length=64)
    targetSkillId: str = Field(default="skill-web-harvest-invoices-1", min_length=1, max_length=64)


class ClawMindResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = 1
    vaultFolder: str
    flushedFactsCount: int
    redactedSecretsCount: int
    soulMarkdown: str
    ownerProfileMarkdown: str
    memoryJournalMarkdown: str
    ariaSnapshotId: str
    ariaElementCount: int
    tokenReductionPercent: int
    compactAriaTree: str
    subagentWorkerId: str
    subagentTokensUsed: int
    subagentLlmTokensUsed: int = 0
    contextSavedPercent: int = 88
    criticPassed: bool = True
    criticConfidenceScore: int = 98
    specialistWorkers: list[dict[str, object]]
    subagentDistilledVi: str
    activeHooks: list[dict[str, object]]
    summaryVi: str


class Frontier2026Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    overlayAnchor: str = Field(default="mac_notch", min_length=1, max_length=64)
    droppedItemName: str = Field(default="hoa_don_thang_11.csv", min_length=1, max_length=240)
    vocalKind: str = Field(default="happy_chirp", min_length=1, max_length=64)
    speechRateWpm: int = Field(default=110, ge=40, le=320)
    rmsEnergyDb: int = Field(default=-24, ge=-80, le=0)
    subject: str = Field(default="Chủ nhân", min_length=1, max_length=120)
    predicate: str = Field(default="đang_tập_trung_dự_án", min_length=1, max_length=120)
    objectVal: str = Field(default="XUNA Frontier 2026", min_length=1, max_length=240)
    previousObjectVal: str = Field(default="Giao thức BLE M02", min_length=1, max_length=240)
    ritualMode: str = Field(default="start_breathing_478", min_length=1, max_length=64)
    neighborShortId: str = Field(default="XUNA9F2A", min_length=1, max_length=32)
    themeId: str = Field(default="green_edition", min_length=1, max_length=64)


class Frontier2026Response(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = 1
    overlayAnchor: str
    munchAnimation: str
    routedTarget: str
    vocalKind: str
    pcmByteCount: int
    wavSha256: str
    detectedOwnerTone: str
    empathyAdjustmentVi: str
    activeEdgesCount: int
    historicalEdgesCount: int
    temporalNarrativeVi: str
    activeRitual: str
    standbyDockActive: bool
    unlockedKeepsakeVi: str
    playdateSummaryVi: str
    activeColorThemeId: str = "green_edition"
    summaryVi: str


class UiAccessibilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action: str = Field(default="inspect_window", min_length=1, max_length=64)
    windowTitle: str = Field(
        default="Excel - Bảng Kê Hóa Đơn",
        min_length=1,
        max_length=240,
    )
    refId: str = Field(default="ui_3", min_length=1, max_length=64)
    valueText: str = Field(default="", max_length=400)
    maxNodes: int = Field(default=30, ge=1, le=80)


class UiAccessibilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schemaVersion: int = 1
    action: str
    windowTitle: str
    backendDriver: str
    nodeCount: int
    tokenSavingsPercent: int
    screenshotBytesUsed: int = 0
    compactUiTree: str
    invokedRefId: str
    redactedControlsCount: int
    nodes: list[dict[str, object]]
    summaryVi: str


