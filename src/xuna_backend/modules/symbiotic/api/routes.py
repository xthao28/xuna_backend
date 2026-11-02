"""Symbiotic Evolution REST API routes (SYM-LIVE S-LIVE.3)."""

from __future__ import annotations

from datetime import UTC, datetime
from http import HTTPStatus

from fastapi import APIRouter

from xuna_backend.core.errors import AppError
from xuna_backend.modules.symbiotic.application.schemas import (
    CoRegulationStateResponse,
    DreamConsolidateRequest,
    DreamConsolidateResponse,
    DreamVignetteResponse,
    ForgedSkillBlueprintResponse,
    ForgeParameterFieldResponse,
    ForgeSkillRequest,
    ForgeSkillResponse,
    HabitMacroProposalResponse,
    RecordSocialEncounterRequest,
    RecordSocialEncounterResponse,
    SerendipityNoteResponse,
    SocialEncounterNoteResponse,
    SocialPostcardResponse,
    UnstuckCoachRequest,
    UnstuckCoachResponse,
)
from xuna_backend.modules.symbiotic.application.services import (
    CoRegulationEvaluator,
    DreamConsolidatorService,
    HabitWorkflowMiner,
    OmniForgeSynthesizerService,
    SerendipityEngine,
    SocialEcologyService,
)

router = APIRouter(prefix="/v1/symbiotic", tags=["symbiotic"])

_dream_consolidator = DreamConsolidatorService()
_coreg_evaluator = CoRegulationEvaluator()
_serendipity_engine = SerendipityEngine()
_habit_miner = HabitWorkflowMiner()
_social_ecology = SocialEcologyService()
_omniforge = OmniForgeSynthesizerService()


@router.post("/dream-consolidate", response_model=DreamConsolidateResponse)
async def consolidate_dream_and_coregulation(
    body: DreamConsolidateRequest,
) -> DreamConsolidateResponse:
    now = datetime.now(UTC)
    vignette = _dream_consolidator.consolidate_nightly_dream(
        project=body.project,
        branch=body.branch,
        dirty_count=body.dirtyCount,
        next_step_hint=body.nextStepHint,
        confirmed_facts=tuple(body.confirmedFacts),
        active_ports=tuple(body.activePorts),
        now=now,
    )
    coreg = _coreg_evaluator.evaluate(
        continuous_work_minutes=body.continuousWorkMinutes,
        local_hour=body.localHour,
        previous_git_dirty_count=body.previousGitDirtyCount,
        current_git_dirty_count=body.dirtyCount,
        solved_case_just_now=body.solvedCaseJustNow,
        hours_since_last_interaction=body.hoursSinceLastInteraction,
        project_name=body.project,
    )
    return DreamConsolidateResponse(
        dreamVignette=DreamVignetteResponse(
            dreamId=vignette.dream_id,
            titleVi=vignette.title_vi,
            narrativeVi=vignette.narrative_vi,
            dreamSymbols=list(vignette.dream_symbols),
            morningCuriosityHookVi=vignette.morning_curiosity_hook_vi,
            relatedProject=vignette.related_project,
            relatedBranch=vignette.related_branch,
            createdAt=vignette.created_at.astimezone(UTC).isoformat(),
        ),
        coRegulationState=CoRegulationStateResponse(
            mode=coreg.mode.value,
            petMoodHint=coreg.pet_mood_hint,
            gestureHint=coreg.gesture_hint,
            competenceDopamineBoost=coreg.competence_dopamine_boost,
            speechVi=coreg.speech_vi,
            suggestedWorkflow=coreg.suggested_workflow,
        ),
    )


@router.post("/unstuck-coach", response_model=UnstuckCoachResponse)
async def decompose_overwhelm_and_spark(
    body: UnstuckCoachRequest,
) -> UnstuckCoachResponse:
    now = datetime.now(UTC)
    clean_goal = body.goal.strip()
    clean_proj = body.project.strip() or "xuna_pet"
    first_step = (
        f"Mở dự án {clean_proj} và thực hiện đúng 1 mảnh nhỏ nhất của '{clean_goal[:90]}' "
        "trong 5 phút đầu tiên."
    )[:240]
    follow_ups = [
        f"Kiểm tra nhanh kết quả bước đầu tiên trên {clean_proj}.",
        "Đánh dấu hoàn thành mảnh đầu tiên và quyết định bước tiếp theo cùng XUNA.",
    ]
    encouragement_vi = "Chỉ cần 5 phút khởi động thôi bạn ơi, mình ngồi canh giờ ngay cạnh bạn đây!"
    note = _serendipity_engine.generate_note(
        primary_topic=clean_goal,
        secondary_topic=clean_proj,
        first_step_hint=first_step,
        now=now,
    )
    proposals = _habit_miner.mine_from_history(body.toolHistory)
    return UnstuckCoachResponse(
        goal=clean_goal,
        project=clean_proj,
        firstFiveMinuteStep=first_step,
        followUpSteps=follow_ups,
        encouragementVi=encouragement_vi,
        serendipityNote=SerendipityNoteResponse(
            noteId=note.note_id,
            titleVi=note.title_vi,
            sparkVi=note.spark_vi,
            connectedTopics=list(note.connected_topics),
            suggestedActionVi=note.suggested_action_vi,
            createdAt=note.created_at.astimezone(UTC).isoformat(),
        ),
        habitProposals=[
            HabitMacroProposalResponse(
                proposalId=p.proposal_id,
                titleVi=p.title_vi,
                triggerPhraseVi=p.trigger_phrase_vi,
                steps=list(p.steps),
                occurrences=p.occurrences,
                maxRisk=p.max_risk,
            )
            for p in proposals
        ],
    )


@router.post("/social-encounters", response_model=RecordSocialEncounterResponse)
async def record_social_encounter(
    body: RecordSocialEncounterRequest,
) -> RecordSocialEncounterResponse:
    try:
        note, postcard = _social_ecology.record_encounter(
            my_short_id=body.myShortId,
            peer_short_id=body.peerShortId,
            peer_trait=body.peerTrait,
            peer_tint=body.peerTint,
            shared_trick=body.sharedTrick,
        )
    except ValueError as err:
        raise AppError("invalid_argument", HTTPStatus.UNPROCESSABLE_ENTITY) from err

    return RecordSocialEncounterResponse(
        encounter=SocialEncounterNoteResponse(
            encounterId=note.encounter_id,
            peerShortId=note.peer_short_id,
            peerTrait=note.peer_trait,
            peerTint=note.peer_tint,
            greetingKind=note.greeting_kind,
            narrativeVi=note.narrative_vi,
            sharedTrick=note.shared_trick,
            occurredAt=note.occurred_at.astimezone(UTC).isoformat(),
        ),
        postcard=SocialPostcardResponse(
            postcardId=postcard.postcard_id,
            fromShortId=postcard.from_short_id,
            toShortId=postcard.to_short_id,
            sharedSkillId=postcard.shared_skill_id,
            messageVi=postcard.message_vi,
            signatureHex=postcard.signature_hex,
            createdAt=postcard.created_at.astimezone(UTC).isoformat(),
        ),
    )


@router.post("/forge-skill", response_model=ForgeSkillResponse)
async def forge_citizen_skill(
    body: ForgeSkillRequest,
) -> ForgeSkillResponse:
    try:
        blueprint = _omniforge.synthesize_skill(
            goal=body.goal,
            category=body.category,
            target_folder=body.targetFolder,
            file_extensions=tuple(body.fileExtensions),
            demonstrated_action=body.demonstratedAction,
        )
    except ValueError as err:
        raise AppError("invalid_argument", HTTPStatus.UNPROCESSABLE_ENTITY) from err

    return ForgeSkillResponse(
        skill=ForgedSkillBlueprintResponse(
            skillId=blueprint.skill_id,
            titleVi=blueprint.title_vi,
            category=blueprint.category,
            goal=blueprint.goal,
            targetFolder=blueprint.target_folder,
            fileExtensions=list(blueprint.file_extensions),
            detectedItemsCount=blueprint.detected_items_count,
            dryRunSample=blueprint.dry_run_sample,
            selfDebugIterations=blueprint.self_debug_iterations,
            parameterFields=[
                ForgeParameterFieldResponse(
                    key=f["key"],
                    labelVi=f["labelVi"],
                    defaultValue=f["defaultValue"],
                )
                for f in blueprint.parameter_fields
            ],
            scriptSteps=list(blueprint.script_steps),
            summaryVi=blueprint.summary_vi,
            createdAt=blueprint.created_at.astimezone(UTC).isoformat(),
        )
    )
