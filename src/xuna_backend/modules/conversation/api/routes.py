"""`POST /v1/conversations/{petId}/turns` → Server-Sent Events (contract §2), and
phone voice V1: `POST /v1/speech/transcriptions` plus spoken replies (`speak`)."""

import base64
import binascii
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
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from xuna_backend.core.errors import AppError
from xuna_backend.modules.accounts.api.dependencies import Session, state
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.api.routes import build_stream_orchestrator
from xuna_backend.modules.actions.application.catalog import load_tool_catalog_for_owner
from xuna_backend.modules.actions.application.orchestrator import ActionContext, EventSink
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionResultEvent,
    BoardHidRequestEvent,
    HostWakeRequestEvent,
    LocalActionRequestEvent,
)
from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    SpeechRecognizer,
    SpeechSynthesizer,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatFinished,
    Locale,
    Message,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
)
from xuna_backend.modules.consent.api.routes import RequireConsent
from xuna_backend.modules.consent.domain.consent import ConsentClass
from xuna_backend.modules.conversation.api.websocket import router as ws_router
from xuna_backend.modules.conversation.application.chat_turn import (
    MAX_EXPERT_INPUT_CHARS,
    MAX_HISTORY_LINE_CHARS,
    MAX_HISTORY_LINES,
    MAX_INPUT_CHARS,
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
    build_request,
    chat_turn,
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
)
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
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


class TurnBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1]
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

    try:
        prepared = await prepare_turn(
            pet_id=target_pet_id,
            text=body.text,
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
        fallback_decision = decide_mode(body.mode, Channel.TEXT, body.text, domain_matched=False)
        prepared = PreparedTurn(
            context=base_ctx,
            decision=fallback_decision,
            domain_id=None,
        )

    client_caps = tuple(body.clientCapabilities or ())
    board_available = bool(body.boardHid.available) if body.boardHid is not None else False
    self_host_id = body.selfHostId.strip() if body.selfHostId else None
    tools: tuple[ToolSpec, ...] = ()
    if client_caps:
        try:
            host_factory: Callable[[object], HostRepository] = state(request, "host_repository")
            hub: HostSessionHub = state(request, "host_session_hub")
            tools = await load_tool_catalog_for_owner(
                owner_id=owner.owner_id,
                client_caps=client_caps,
                board_available=board_available,
                host_repo=host_factory(session),
                hub=hub,
                self_host_id=self_host_id,
            )
        except Exception:
            _logger.debug("Failed to load tool catalog for turn", exc_info=True)
            tools = ()

    await session.close()  # release the DB connection before a long provider call

    settings = state(request, "settings")
    chosen_mode = prepared.decision.mode
    if chosen_mode == TurnMode.EXPERT:
        budget = ReplyBudget(
            max_output_tokens=settings.text_expert_max_tokens,
            timeout_s=settings.text_expert_timeout_s,
        )
        expert_gen: TextGenerator | None = getattr(request.app.state, "expert_text_generator", None)
        generator: TextGenerator = expert_gen or state(request, "text_generator")
    else:
        budget = ReplyBudget(max_output_tokens=400, timeout_s=settings.text_timeout_s)
        generator = state(request, "text_generator")

    try:
        chat_request = build_request(
            body.text,
            body.locale,
            budget.timeout_s,
            _effective_context(prepared.context),
            mode=chosen_mode,
            budget=budget,
            tools=tools,
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
    if client_caps:
        stream_orch = build_stream_orchestrator(request.app.state)
        raw_rid = getattr(request.state, "request_id", "")
        request_id_str = str(raw_rid) if isinstance(raw_rid, str) else ""

        async def _run_tool(call: ToolCall, emit: EventSink) -> Message:
            return await stream_orch.handle_call(
                call,
                ActionContext(
                    owner_id=owner.owner_id,
                    pet_id=pet_id,
                    locale=chat_request.locale,
                    client_caps=client_caps,
                    board_available=board_available,
                    emit_event=emit,
                    request_id=request_id_str,
                    self_host_id=self_host_id,
                ),
            )

        tool_runner = _run_tool

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
        text_events = chat_turn(
            generator,
            chat_request,
            sections=(chosen_mode == TurnMode.EXPERT),
            locale_vi=(chat_request.locale == Locale.VI),
            tool_runner=tool_runner,
        )
        stream: AsyncIterator[SpokenTurnEvent] = (
            spoken_turn(text_events, synthesizer, body.locale, voice=body.voice)
            if body.speak
            else text_events
        )
        has_detail = False
        async for event in stream:
            match event:
                case TextDelta(text=text):
                    yield _sse("delta", {"text": text})
                case DetailDelta(text=text):
                    has_detail = True
                    yield _sse("detail", {"text": text})
                case GestureChosen(gesture=gesture):
                    yield _sse("gesture", {"name": gesture.value})
                case ActionPendingEvent() as pending:
                    pending_payload: dict[str, Any] = {
                        "approvalId": pending.approval_id,
                        "target": pending.target,
                        "summary": pending.summary,
                        "risk": pending.risk,
                        "expiresIn": pending.expires_in,
                    }
                    if pending.host_id is not None:
                        pending_payload["hostId"] = pending.host_id
                    if pending.args_sha256 is not None:
                        pending_payload["argsSha256"] = pending.args_sha256
                    if pending.expires_at is not None:
                        pending_payload["expiresAt"] = pending.expires_at
                    yield _sse("action_pending", pending_payload)
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
                    act_payload: dict[str, Any] = {
                        "invocationId": act_res.invocation_id,
                        "target": act_res.target,
                        "summary": act_res.summary,
                        "status": act_res.status,
                    }
                    if act_res.gesture is not None:
                        act_payload["gesture"] = act_res.gesture
                    yield _sse("action_result", act_payload)
                case ToolCallRequested():
                    continue
                case ChatFinished() as done:
                    extra = {
                        "mode": chosen_mode.value,
                        "domainId": domain_id_str,
                        "hasDetail": has_detail,
                    }
                    yield _sse("finished", _finished(done) | signals | candidate_payload | extra)
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
