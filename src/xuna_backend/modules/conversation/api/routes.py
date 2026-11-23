"""`POST /v1/conversations/{petId}/turns` → Server-Sent Events (contract §2), and
phone voice V1: `POST /v1/speech/transcriptions` plus spoken replies (`speak`)."""

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field

from xuna_backend.core.errors import AppError
from xuna_backend.modules.accounts.api.dependencies import Session, state
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.api.routes import (
    build_mesh_pre_router,
    build_stream_orchestrator,
)
from xuna_backend.modules.actions.application.computer_use import session_event_payload
from xuna_backend.modules.actions.application.cu_template import CuTemplatePin
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    EventSink,
    Orchestrator,
)
from xuna_backend.modules.actions.application.yield_detach import (
    DEFAULT_YIELD_MS,
    YieldDetachRequest,
    YieldDetachResult,
    YieldDetachService,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionResultEvent,
    ActionTargetRequiredEvent,
    BoardHidRequestEvent,
    ComputerUseSessionEvent,
    HostWakeRequestEvent,
    LocalActionRequestEvent,
)
from xuna_backend.modules.actions.domain.watch import TurnDetachedEvent
from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    SpeechRecognizer,
    SpeechSynthesizer,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.language_detector import detect_text_language
from xuna_backend.modules.cognition.domain.types import (
    ChatFinished,
    Locale,
    Message,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)
from xuna_backend.modules.consent.api.routes import RequireConsent
from xuna_backend.modules.consent.domain.consent import ConsentClass
from xuna_backend.modules.conversation.api.websocket import router as ws_router
from xuna_backend.modules.conversation.application.chat_turn import (
    MAX_EXPERT_INPUT_CHARS,
    MAX_HISTORY_LINE_CHARS,
    MAX_HISTORY_LINES,
    MAX_INPUT_CHARS,
    CuCleanupCallback,
    DetailDelta,
    GestureChosen,
    HistoryLine,
    PetAction,
    PetMood,
    PetSelf,
    PetTemperament,
    ProviderFailed,
    ReplyBudget,
    ToolCallRunner,
    TurnContext,
    YieldDetachCallback,
    build_request,
    chat_turn,
)
from xuna_backend.modules.conversation.application.mesh_turn_helper import (
    NonZeroHostUuid,
    execute_repair_dispatch,
    prepare_mesh_turn,
    stream_short_circuit_turn,
    stream_with_target_fallback,
    tz_offset_minutes_from_dt,
)
from xuna_backend.modules.conversation.application.spoken_turn import (
    SpeechUnavailable,
    SpokenAudio,
    SpokenFinished,
    SpokenTurnEvent,
    spoken_turn,
)
from xuna_backend.modules.conversation.application.turn_preparation import (
    PreparedTurn,
    prepare_turn,
)
from xuna_backend.modules.conversation.application.voice import (
    MAX_AUDIO_BYTES,
    AudioFormat,
    transcribe_once,
)
from xuna_backend.modules.conversation.domain.candidate_extractor import (
    extract_candidate,
    extract_convention_candidate,
)
from xuna_backend.modules.conversation.domain.signals import conversation_signals
from xuna_backend.modules.conversation.domain.turn_mode import (
    Channel,
    RequestedMode,
    TurnMode,
    decide_mode,
    is_code_or_tool_synthesis_request,
)
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.entities import HostId
from xuna_backend.modules.hosts.domain.twin import HostFocusChangedNotice
from xuna_backend.modules.identity.application.use_cases import IdentityError, get_owned_pet
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.owner_context import BuildOwnerContextUseCase
from xuna_backend.modules.knowledge.application.ports import (
    KnowledgeCandidatePort,
    KnowledgeRepository,
)
from xuna_backend.modules.knowledge.domain.entities import DomainStatus, compute_fact_key
from xuna_backend.modules.knowledge.domain.lexical_matcher import (
    compute_lexical_similarity,
)
from xuna_backend.modules.knowledge.domain.owner_context import DeviceSlotView
from xuna_backend.modules.memory.application.ports import MemoryRepository
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
)

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["conversation"])
router.include_router(ws_router)
U3Owner = Annotated[Owner, RequireConsent(ConsentClass.CONVERSATION_TEXT)]
VoiceOwner = Annotated[Owner, RequireConsent(ConsentClass.CONVERSATION_VOICE)]


class HistoryLineBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["owner", "xuna"]
    text: str = Field(min_length=1, max_length=MAX_HISTORY_LINE_CHARS)


class PetSelfBody(BaseModel):
    """Device-read facts about XUNA (owner 2026-10-23); prompt context only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    activeSeconds: int = Field(default=0, ge=0, lt=2**32)
    learnedEvents: int = Field(default=0, ge=0, lt=2**32)
    current: PetAction | None = None
    quietChoice: PetAction | None = None
    playChoice: PetAction | None = None
    personality: PetTemperament | None = None
    mood: PetMood | None = None


class KnowledgeSlotBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slot: int = Field(ge=0, le=4)
    level: int = Field(ge=0, le=3)
    state: Literal["empty", "active", "resting"]


class BoardHidBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool = False


class CuTemplateBody(BaseModel):
    """UDC-17 S17.5: the scope of a computer-use template saved on the owner's phone, sent
    when the owner taps "Chạy lại". Never stored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=40)
    hostId: str = Field(min_length=1, max_length=64)
    apps: list[str] = Field(min_length=1, max_length=8)
    domains: list[str] = Field(default_factory=list, max_length=12)
    maxSteps: int
    maxMinutes: int

    def to_pin(self) -> CuTemplatePin:
        return CuTemplatePin.create(
            name=self.name,
            host_id=self.hostId,
            apps=self.apps,
            domains=self.domains,
            max_steps=self.maxSteps,
            max_minutes=self.maxMinutes,
        )


def _require_int_schema_version(v: object) -> object:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError("schemaVersion must be an integer")
    return v


class TurnBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Annotated[Literal[1], BeforeValidator(_require_int_schema_version)]
    text: str = Field(min_length=1, max_length=MAX_EXPERT_INPUT_CHARS)
    locale: Locale | None = None
    voice: str | None = None
    # Phone voice V1: also stream the reply as audio, sentence by sentence.
    speak: bool = False
    # KX-1: mode routing and explicit domain selection.
    mode: RequestedMode | None = None
    domainId: uuid.UUID | None = None
    # Turn context (owner 2026-10-23), all optional and never stored: the phone's
    # local time, earlier lines of this conversation, and what the device says
    # about XUNA. Old apps omit them and get the context-free prompt.
    clientTime: AwareDatetime | None = None
    history: list[HistoryLineBody] | None = Field(default=None, max_length=MAX_HISTORY_LINES)
    pet: PetSelfBody | None = None
    # The connected device can play reply gestures (owner 2026-10-27).
    gestures: bool = False

    knowledge: list[KnowledgeSlotBody] | None = Field(default=None, max_length=5)
    extractKnowledge: bool = True

    # UDC-4 (§5.4) & UDC-5 (§5.2): Client capabilities, board HID state, and same-device host ID.
    clientCapabilities: list[str] | None = Field(default=None, max_length=16)
    boardHid: BoardHidBody | None = None
    selfHostId: str | None = Field(default=None, max_length=64)
    # UDC-17 S17.5: rerun of a template saved on the phone (old apps omit it).
    cuTemplate: CuTemplateBody | None = None
    # Spatial Device Mesh (§5.6.6): explicit target override, Follow Me, and subnet proximity.
    targetHostId: NonZeroHostUuid | None = None
    followMe: bool | None = None
    clientSubnetHash: str | None = Field(default=None, max_length=64, pattern=r"^[0-9a-f]{16,64}$")

    def base_context(self) -> TurnContext:
        pet = self.pet
        return TurnContext(
            local_time=self.clientTime,
            history=tuple(HistoryLine(h.role == "owner", h.text) for h in self.history or ()),
            pet=None
            if pet is None
            else PetSelf(
                active_seconds=pet.activeSeconds,
                learned_events=pet.learnedEvents,
                current=pet.current,
                quiet_choice=pet.quietChoice,
                play_choice=pet.playChoice,
                temperament=pet.personality,
                mood=pet.mood,
            ),
            gestures=self.gestures,
        )

    def device_slots(self) -> tuple[DeviceSlotView, ...] | None:
        if self.knowledge is None:
            return None
        return tuple(
            DeviceSlotView(slot=k.slot, level=k.level, state=k.state) for k in self.knowledge
        )


def _effective_context(ctx: TurnContext) -> TurnContext | None:
    has_owner = ctx.owner is not None and (
        ctx.owner.domain is not None
        or bool(ctx.owner.conventions)
        or bool(ctx.owner.cases)
        or bool(ctx.owner.snippets)
    )
    if (
        ctx.local_time is None
        and not ctx.history
        and ctx.pet is None
        and not ctx.gestures
        and not ctx.confirmed_facts
        and not has_owner
    ):
        return None
    return ctx


class TranscriptionBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1]
    audio: str = Field(min_length=4, max_length=(MAX_AUDIO_BYTES * 4) // 3 + 4)  # base64
    format: AudioFormat
    locale: Locale | None = None


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/conversations/{pet_id}/turns")
async def turn(
    pet_id: str, body: TurnBody, request: Request, owner: U3Owner, session: Session
) -> StreamingResponse:
    try:
        repo = state(request, "identity_repository")(session)
        await get_owned_pet(repo, owner.owner_id, PetId(pet_id))
    except (ValueError, IdentityError) as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    stripped_text = body.text.strip()
    if not stripped_text or (
        (body.mode is None or body.mode == RequestedMode.CHAT)
        and len(stripped_text) > MAX_INPUT_CHARS
    ):
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "text")

    try:
        base_ctx = body.base_context()
    except ValueError as error:
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "text") from error

    # CLD-14, KD-4: Extract prospective memory candidates only within MAX_INPUT_CHARS
    extracted = None
    if len(body.text) <= MAX_INPUT_CHARS:
        extracted = extract_candidate(body.text)
        if extracted is None and body.extractKnowledge:
            extracted = extract_convention_candidate(body.text)
    candidate_info: dict[str, Any] | None = None

    now = datetime.now(UTC)
    target_pet_id = PetId(pet_id)
    mem_factory: Callable[[object], MemoryRepository] = state(request, "memory_repository")
    k_factory: Callable[[object], KnowledgeRepository] = state(request, "knowledge_repository")
    cand_factory: Callable[[object], KnowledgeCandidatePort] = state(
        request, "knowledge_candidate_adapter"
    )
    mem_repo = mem_factory(session)
    k_repo = k_factory(session)
    c_port = cand_factory(session)

    try:
        if extracted is not None:
            existing = await mem_repo.list_candidates(
                target_pet_id, status=CandidateStatus.AWAITING_OWNER_CONFIRMATION
            )
            is_dup = any(
                c.candidate_type == extracted.candidate_type
                and (
                    c.content.get("summary") == extracted.content.get("summary")
                    or (
                        c.candidate_type == CandidateType.KNOWLEDGE
                        and c.content.get("text") == extracted.content.get("text")
                    )
                )
                for c in existing
            )
            if not is_dup:
                domain_id: uuid.UUID | None = None
                relation: str | None = None
                fact_key: int | None = None

                if extracted.candidate_type == CandidateType.KNOWLEDGE:
                    text_content = str(extracted.content.get("text", ""))
                    fact_key = compute_fact_key(text_content)
                    try:
                        domains = await k_repo.list_domains(target_pet_id, include_deleted=False)
                        active = [d for d in domains if d.status != DomainStatus.DELETED]
                        best_domain = None
                        best_sim = 0.0
                        for dom in active:
                            sim = compute_lexical_similarity(
                                text_content, [dom.name, dom.description]
                            )
                            if sim > best_sim:
                                best_sim = sim
                                best_domain = dom
                        if best_domain is not None and best_sim >= 0.20:
                            domain_id = best_domain.domain_id
                            domain_cands = await c_port.list_candidates_by_domain(
                                domain_id, confirmed_only=True
                            )
                            for dc in domain_cands:
                                if (
                                    compute_lexical_similarity(
                                        text_content, [str(dc.content.get("text", ""))]
                                    )
                                    >= 0.60
                                ):
                                    relation = "refines"
                                    break
                            if relation is None:
                                relation = "new"
                    except Exception:
                        _logger.debug(
                            "Failed to resolve domain/relation for candidate",
                            exc_info=True,
                        )

                cand = MemoryCandidate(
                    candidate_id=uuid.uuid4(),
                    owner_id=owner.owner_id,
                    pet_id=target_pet_id,
                    source=extracted.source,
                    candidate_type=extracted.candidate_type,
                    content=extracted.content,
                    status=CandidateStatus.AWAITING_OWNER_CONFIRMATION,
                    created_at=now,
                    updated_at=now,
                    domain_id=domain_id,
                    relation=relation,
                    fact_key=fact_key,
                )
                await mem_repo.add_candidate(cand)
                await session.commit()
                candidate_info = {
                    "candidateId": str(cand.candidate_id),
                    "type": cand.candidate_type.value,
                    "summary": cand.content.get("summary") or cand.content.get("text", ""),
                }
    except Exception:  # noqa: S110
        # Candidate extraction is best-effort; must not fail turn
        pass

    client_caps = tuple(body.clientCapabilities or ())
    board_available = bool(body.boardHid.available) if body.boardHid is not None else False
    self_host_id = body.selfHostId.strip() if body.selfHostId else None
    try:
        cu_template = body.cuTemplate.to_pin() if body.cuTemplate is not None else None
    except ValueError as error:
        raise AppError(
            "validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "cuTemplate"
        ) from error
    tz_offset = tz_offset_minutes_from_dt(body.clientTime)
    hub_obj: HostSessionHub | None = getattr(request.app.state, "host_session_hub", None)
    host_repo_factory = getattr(request.app.state, "host_repository", None)
    consent_factory = getattr(request.app.state, "consent_repository", None)
    audit_factory = getattr(request.app.state, "audit_log", None)
    try:
        parsed_target_host_id = HostId(body.targetHostId) if body.targetHostId is not None else None
    except ValueError as error:
        raise AppError(
            "validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "targetHostId"
        ) from error
    mesh_prep = await prepare_mesh_turn(
        owner_id=owner.owner_id,
        pet_id=pet_id,
        text=body.text,
        client_caps=client_caps,
        board_available=board_available,
        self_host_id=self_host_id,
        target_host_id=parsed_target_host_id,
        follow_me=body.followMe,
        tz_offset_minutes=tz_offset,
        include_autonomous_job=getattr(request.app.state, "job_runner", None) is not None,
        consent_repo=consent_factory(session) if consent_factory is not None else None,
        host_repo=host_repo_factory(session) if host_repo_factory is not None else None,
        hub=hub_obj,
        audit=audit_factory(session) if audit_factory is not None else None,
        pre_router=build_mesh_pre_router(request.app.state),
    )
    if mesh_prep.outcome.kind == "not_found":
        raise AppError("not_found", HTTPStatus.NOT_FOUND)
    routing_text = (
        mesh_prep.effective_text if mesh_prep.outcome.kind == "repair_dispatch" else body.text
    )

    try:
        prepared = await prepare_turn(
            pet_id=target_pet_id,
            text=routing_text,
            requested=body.mode,
            channel=Channel.TEXT,
            base=base_ctx,
            device_slots=body.device_slots(),
            explicit_domain_id=body.domainId,
            memory_repo=mem_repo,
            owner_context=BuildOwnerContextUseCase(
                k_repo, c_port, getattr(request.app.state, "embedder", None)
            ),
            now=now,
        )
    except Exception:
        fallback_decision = decide_mode(body.mode, Channel.TEXT, routing_text, domain_matched=False)
        prepared = PreparedTurn(
            context=base_ctx,
            decision=fallback_decision,
            domain_id=None,
        )
    tools = mesh_prep.tools
    host_screen_consent = mesh_prep.host_screen_consent
    pinned_host_id = mesh_prep.pinned_host_id
    host_roster = mesh_prep.host_roster
    template_note: Callable[[Locale], str] | None = None
    if cu_template is not None and tools:
        # The computer's current alias, read for this owner only; the pin itself is by id.
        template_alias: str | None = None
        try:
            host_repo: HostRepository = state(request, "host_repository")(session)
            template_host = await host_repo.get_host(HostId(uuid.UUID(cu_template.host_id)))
            if template_host is not None and template_host.owner_id == owner.owner_id:
                template_alias = str(template_host.alias)
        except Exception:
            _logger.debug("Failed to read the template's computer", exc_info=True)
        pin = cu_template

        def _template_note(locale: Locale) -> str:
            return pin.prompt_note(locale, template_alias)

        template_note = _template_note

    await session.close()  # release the DB connection before a long provider call

    raw_rid = getattr(request.state, "request_id", "")
    request_id_str = str(raw_rid) if isinstance(raw_rid, str) else ""
    stream_orch: Orchestrator | None = (
        build_stream_orchestrator(request.app.state)
        if (client_caps or mesh_prep.outcome.kind == "repair_dispatch")
        else None
    )
    cu_note: str | None = None
    if stream_orch is not None and tools:
        detected = detect_text_language(routing_text)
        eff_locale = detected if detected == Locale.EN else (body.locale or Locale.VI)
        try:
            cu_note = await stream_orch.active_cu_session_prompt_note(
                owner.owner_id,
                host_id=pinned_host_id,
                locale=eff_locale,
                request_id=request_id_str,
            )
        except Exception:
            _logger.debug("Failed to read active CU session prompt note", exc_info=True)
            cu_note = None

    turn_note: Callable[[Locale], str] | None = None
    if template_note is not None and cu_note:
        tpl_fn = template_note
        cu_text = cu_note

        def _combined_note(locale: Locale) -> str:
            return f"{tpl_fn(locale)}{cu_text}"

        turn_note = _combined_note
    elif template_note is not None:
        turn_note = template_note
    elif cu_note:
        cu_only_text = cu_note

        def _cu_only_note(_locale: Locale) -> str:
            return cu_only_text

        turn_note = _cu_only_note

    settings = state(request, "settings")
    chosen_mode = prepared.decision.mode
    code_gen: TextGenerator | None = getattr(
        request.app.state, "code_text_generator", None
    ) or getattr(request.app.state, "expert_text_generator", None)
    expert_gen: TextGenerator | None = (
        getattr(request.app.state, "expert_text_generator", None) or code_gen
    )
    is_code_turn = is_code_or_tool_synthesis_request(routing_text)

    if chosen_mode == TurnMode.EXPERT:
        if is_code_turn and code_gen is not None:
            budget = ReplyBudget(
                max_output_tokens=getattr(
                    settings, "text_code_max_tokens", settings.text_expert_max_tokens
                ),
                timeout_s=getattr(settings, "text_code_timeout_s", settings.text_expert_timeout_s),
            )
            generator: TextGenerator = code_gen
        else:
            budget = ReplyBudget(
                max_output_tokens=settings.text_expert_max_tokens,
                timeout_s=settings.text_expert_timeout_s,
            )
            generator = expert_gen or state(request, "text_generator")
    else:
        budget = ReplyBudget(max_output_tokens=400, timeout_s=settings.text_timeout_s)
        generator = state(request, "text_generator")

    try:
        chat_request = build_request(
            mesh_prep.effective_text,
            body.locale,
            budget.timeout_s,
            _effective_context(prepared.context),
            mode=chosen_mode,
            budget=budget,
            tools=tools,
            turn_note=turn_note,
            host_roster=host_roster if tools else None,
        )
    except ValueError as error:
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "text") from error

    synthesizer: SpeechSynthesizer | None = state(request, "speech_synthesizer")
    # PE-4c: closed tags from the owner's words, sent with `finished` only;
    # the app forwards them to the device. Never stored.
    signals = {"signals": list(conversation_signals(body.text, len(body.history or ())))}
    candidate_payload = {"candidate": candidate_info} if candidate_info is not None else {}
    domain_id_str = str(prepared.domain_id) if prepared.domain_id is not None else None

    tool_runner: ToolCallRunner | None = None
    cu_cleanup: CuCleanupCallback | None = None
    yield_detach: YieldDetachCallback | None = None

    def _make_action_ctx(emit: EventSink) -> ActionContext:
        return ActionContext(
            owner_id=owner.owner_id,
            pet_id=pet_id,
            locale=chat_request.locale,
            client_caps=mesh_prep.effective_client_caps,
            board_available=board_available,
            emit_event=emit,
            request_id=request_id_str,
            self_host_id=self_host_id,
            host_screen_consent=host_screen_consent,
            cu_template=cu_template,
            pinned_host_id=pinned_host_id,
            follow_me=bool(body.followMe),
            client_subnet_hash=body.clientSubnetHash,
            client_tz_offset_minutes=tz_offset,
            habit_recorded_in_preroute=mesh_prep.habit_recorded_in_preroute,
            pre_route_target_required=mesh_prep.pre_route_target_required,
            pinned_target_reason=mesh_prep.pinned_target_reason,
        )

    if stream_orch is not None:
        _sse_orch = stream_orch

        async def _run_tool(call: ToolCall, emit: EventSink) -> Message:
            return await _sse_orch.handle_call(call, _make_action_ctx(emit))

        async def _cleanup_sse_cu(reason: str) -> ComputerUseSessionEvent | None:
            return await _sse_orch.cleanup_dangling_cu_session(
                owner_id=owner.owner_id,
                host_id=pinned_host_id,
                reason=reason,
                request_id=request_id_str,
            )

        tool_runner = _run_tool
        cu_cleanup = _cleanup_sse_cu
        yield_svc: YieldDetachService | None = getattr(
            request.app.state, "yield_detach_service", None
        )
        if yield_svc is not None:
            _sse_yield_svc = yield_svc

            async def _sse_yield_detach(
                calls: tuple[ToolCall, ...],
                elapsed_ms: int,
                last_summary: str | None,
            ) -> YieldDetachResult | None:
                return await _sse_yield_svc.try_detach(
                    YieldDetachRequest(
                        owner_id=owner.owner_id,
                        pet_id=pet_id,
                        host_id=pinned_host_id,
                        goal=mesh_prep.effective_text,
                        tool_calls=calls,
                        elapsed_ms=elapsed_ms,
                        locale=chat_request.locale,
                        last_action_summary=last_summary,
                        request_id=request_id_str,
                    )
                )

            yield_detach = _sse_yield_detach

    async def events() -> AsyncIterator[str]:
        if body.mode is not None:
            yield _sse(
                "mode",
                {
                    "mode": chosen_mode.value,
                    "domainId": domain_id_str,
                    "reason": prepared.decision.reason.value,
                },
            )
        if mesh_prep.outcome.focus_notice is not None:
            yield _sse("host_focus_changed", mesh_prep.outcome.focus_notice.wire_payload())
        code_timeout_s = (
            getattr(settings, "text_code_timeout_s", None) if code_gen is not None else None
        )
        if mesh_prep.short_circuit:
            text_events = stream_short_circuit_turn(mesh_prep, chat_request.locale)
        elif mesh_prep.outcome.kind == "repair_dispatch" and stream_orch is not None:
            text_events = execute_repair_dispatch(
                outcome=mesh_prep.outcome,
                generator=generator,
                base_request=chat_request,
                orchestrator=stream_orch,
                action_ctx_factory=_make_action_ctx,
                sections=(chosen_mode == TurnMode.EXPERT),
                locale_vi=(chat_request.locale == Locale.VI),
                code_generator=code_gen,
                code_timeout_s=code_timeout_s,
                cu_cleanup=cu_cleanup,
                yield_detach=yield_detach,
                yield_ms=DEFAULT_YIELD_MS,
            )
        else:
            raw_text_events = chat_turn(
                generator,
                chat_request,
                sections=(chosen_mode == TurnMode.EXPERT),
                locale_vi=(chat_request.locale == Locale.VI),
                tool_runner=tool_runner,
                code_generator=code_gen,
                code_timeout_s=code_timeout_s,
                cu_cleanup=cu_cleanup,
                yield_detach=yield_detach,
                yield_ms=DEFAULT_YIELD_MS,
            )
            text_events = (
                stream_with_target_fallback(raw_text_events, mesh_prep.target_required_event)
                if mesh_prep.pre_route_target_required is not None
                else raw_text_events
            )
        stream: AsyncIterator[SpokenTurnEvent] = (
            spoken_turn(text_events, synthesizer, body.locale, voice=body.voice)
            if body.speak
            else text_events
        )
        has_detail = False
        stream_completed = False
        event_queue: asyncio.Queue[SpokenTurnEvent | BaseException | None] = asyncio.Queue()

        async def _on_focus_changed(notice: HostFocusChangedNotice) -> None:
            await event_queue.put(notice)

        sub_fn = getattr(hub_obj, "subscribe_focus", None) if hub_obj is not None else None
        unsub_focus = sub_fn(owner.owner_id, _on_focus_changed) if callable(sub_fn) else None

        async def _pump_stream() -> None:
            try:
                async for ev in stream:
                    await event_queue.put(ev)
            except BaseException as exc:
                await event_queue.put(exc)
            finally:
                aclose = getattr(stream, "aclose", None)
                if callable(aclose):
                    with contextlib.suppress(BaseException):
                        await aclose()
                await event_queue.put(None)

        pump_task = asyncio.create_task(_pump_stream())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(event_queue.get(), timeout=4.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    stream_completed = True
                    break
                if isinstance(item, BaseException):
                    raise item
                event = item
                match event:
                    case TextDelta(text=text):
                        yield _sse("delta", {"text": text})
                    case DetailDelta(text=text):
                        has_detail = True
                        yield _sse("detail", {"text": text})
                    case GestureChosen(gesture=gesture):
                        yield _sse("gesture", {"name": gesture.value})
                    case ActionPendingEvent() as pending:
                        yield _sse("action_pending", pending.wire_payload())
                    case ActionTargetRequiredEvent() as target_req:
                        yield _sse("action_target_required", target_req.wire_payload())
                    case HostFocusChangedNotice() as focus_ev:
                        yield _sse("host_focus_changed", focus_ev.wire_payload())
                    case BoardHidRequestEvent() as hid_req:
                        yield _sse(
                            "board_hid_request",
                            {
                                "invocationId": hid_req.invocation_id,
                                "approvalId": hid_req.approval_id,
                                "key": hid_req.key,
                                "usageIndex": hid_req.usage_index,
                            },
                        )
                    case LocalActionRequestEvent() as local_req:
                        local_payload: dict[str, Any] = {
                            "invocationId": local_req.invocation_id,
                            "target": local_req.target,
                            "tool": local_req.tool,
                            "arguments": local_req.arguments,
                        }
                        if local_req.approval_id is not None:
                            local_payload["approvalId"] = local_req.approval_id
                        yield _sse("local_action_request", local_payload)
                    case HostWakeRequestEvent() as wake_req:
                        wake_payload: dict[str, Any] = {
                            "invocationId": wake_req.invocation_id,
                            "hostId": wake_req.host_id,
                            "target": wake_req.target,
                        }
                        if wake_req.mac_address is not None:
                            wake_payload["macAddress"] = wake_req.mac_address
                        yield _sse("host_wake_request", wake_payload)
                    case ActionResultEvent() as act_res:
                        yield _sse("action_result", act_res.wire_payload())
                    case ComputerUseSessionEvent() as cu_ev:
                        yield _sse("cu_session", session_event_payload(cu_ev))
                    case TurnDetachedEvent() as detached:
                        yield _sse("autonomous_job", detached.wire_payload())
                    case ToolCallRequested():
                        continue
                    case ChatFinished() as done:
                        extra = {
                            "mode": chosen_mode.value,
                            "domainId": domain_id_str,
                            "hasDetail": has_detail,
                        }
                        yield _sse(
                            "finished",
                            _finished(done) | signals | candidate_payload | extra,
                        )
                    case SpokenFinished(finished=done, spoken=spoken, first_audio_ms=first_ms):
                        timing = {} if first_ms is None else {"firstAudioMs": first_ms}
                        extra = {
                            "mode": chosen_mode.value,
                            "domainId": domain_id_str,
                            "hasDetail": has_detail,
                        }
                        yield _sse(
                            "finished",
                            _finished(done)
                            | {"spoken": spoken}
                            | timing
                            | signals
                            | candidate_payload
                            | extra,
                        )
                    case SpokenAudio() as audio:
                        yield _sse(
                            "audio",
                            {
                                "seq": audio.seq,
                                "text": audio.text,
                                "format": audio.codec,
                                "sampleRate": audio.sample_rate,
                                "data": base64.b64encode(audio.data).decode("ascii"),
                            },
                        )
                    case SpeechUnavailable(code=code):
                        yield _sse("speech_unavailable", {"code": code})
                    case ProviderFailed(code=code):
                        yield _sse("error", {"code": code})
        finally:
            if callable(unsub_focus):
                with contextlib.suppress(BaseException):
                    unsub_focus()
            if not pump_task.done():
                pump_task.cancel()
                await asyncio.gather(pump_task, return_exceptions=True)
            if stream_orch is not None and not stream_completed:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(
                        stream_orch.cleanup_dangling_cu_session(
                            owner_id=owner.owner_id,
                            host_id=pinned_host_id,
                            reason="turn_disconnected",
                            request_id=request_id_str,
                        )
                    )

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
    )


def _finished(done: ChatFinished) -> dict[str, Any]:
    return {"provider": done.provider, "model": done.model, "latencyMs": done.latency_ms}


@router.post("/speech/transcriptions")
async def transcription(
    body: TranscriptionBody, request: Request, owner: VoiceOwner, session: Session
) -> dict[str, Any]:
    """One push-to-talk utterance → text. The audio goes to the ASR provider for
    this call only (consent U3V) and is never stored or logged."""
    await session.close()  # release the DB connection before the provider call
    recognizer: SpeechRecognizer | None = state(request, "speech_recognizer")
    if recognizer is None:
        raise AppError("voice_unavailable", HTTPStatus.SERVICE_UNAVAILABLE)
    try:
        audio = base64.b64decode(body.audio, validate=True)
    except (binascii.Error, ValueError) as error:
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "audio") from error
    started = time.perf_counter()
    settings = state(request, "settings")
    locale = None if settings.asr_auto_detect else body.locale
    try:
        text = await transcribe_once(recognizer, audio, body.format, locale)
    except ValueError as error:
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "audio") from error
    except ProviderUnavailableError as error:
        raise AppError("provider_unavailable", HTTPStatus.SERVICE_UNAVAILABLE) from error
    return {"text": text, "latencyMs": int((time.perf_counter() - started) * 1000)}


class SynthesisBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1] = 1
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    voice: str | None = None
    locale: Locale | None = None


@router.post("/speech/synthesis")
async def synthesis(
    body: SynthesisBody, request: Request, owner: VoiceOwner, session: Session
) -> dict[str, Any]:
    """Synthesize text into audio for voice preview (requires U3V consent)."""
    await session.close()
    synthesizer: SpeechSynthesizer | None = state(request, "speech_synthesizer")
    if synthesizer is None:
        raise AppError("voice_unavailable", HTTPStatus.SERVICE_UNAVAILABLE)
    audio = bytearray()
    codec = "mp3"
    try:
        async for chunk in synthesizer.synthesize(body.text, body.locale, voice=body.voice):
            audio.extend(chunk.data)
            codec = chunk.codec
    except ProviderUnavailableError as error:
        raise AppError("provider_unavailable", HTTPStatus.SERVICE_UNAVAILABLE) from error
    return {
        "text": body.text,
        "format": codec,
        "audio": base64.b64encode(bytes(audio)).decode("ascii"),
    }
