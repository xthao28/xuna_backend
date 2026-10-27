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
from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    SpeechRecognizer,
    SpeechSynthesizer,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatFinished,
    Locale,
    TextDelta,
    ToolCallRequested,
)
from xuna_backend.modules.consent.api.routes import RequireConsent
from xuna_backend.modules.consent.domain.consent import ConsentClass
from xuna_backend.modules.conversation.api.websocket import router as ws_router
from xuna_backend.modules.conversation.application.chat_turn import (
    MAX_HISTORY_LINES,
    MAX_INPUT_CHARS,
    HistoryLine,
    PetAction,
    PetMood,
    PetSelf,
    PetTemperament,
    ProviderFailed,
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
from xuna_backend.modules.conversation.application.voice import (
    MAX_AUDIO_BYTES,
    AudioFormat,
    transcribe_once,
)
from xuna_backend.modules.conversation.domain.candidate_extractor import (
    extract_candidate,
    extract_knowledge_candidate,
)
from xuna_backend.modules.conversation.domain.signals import conversation_signals
from xuna_backend.modules.identity.application.use_cases import IdentityError, get_owned_pet
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.domain.entities import DomainStatus, compute_fact_key
from xuna_backend.modules.knowledge.domain.lexical_matcher import (
    compute_lexical_similarity,
)
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
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


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


class TurnBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1]
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    locale: Locale = Locale.VI
    # Phone voice V1: also stream the reply as audio, sentence by sentence.
    speak: bool = False
    # Turn context (owner 2026-10-23), all optional and never stored: the phone's
    # local time, earlier lines of this conversation, and what the device says
    # about XUNA. Old apps omit them and get the context-free prompt.
    clientTime: AwareDatetime | None = None
    history: list[HistoryLineBody] | None = Field(default=None, max_length=MAX_HISTORY_LINES)
    pet: PetSelfBody | None = None

    knowledge: list[dict[str, Any]] | None = None
    extractKnowledge: bool = True

    def context(
        self,
        confirmed_facts: tuple[str, ...] = (),
        knowledge: tuple[str, ...] = (),
    ) -> TurnContext | None:
        if (
            self.clientTime is None
            and self.history is None
            and self.pet is None
            and not confirmed_facts
            and not knowledge
        ):
            return None
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
            confirmed_facts=confirmed_facts,
            knowledge=knowledge,
        )


class TranscriptionBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1]
    audio: str = Field(min_length=4, max_length=(MAX_AUDIO_BYTES * 4) // 3 + 4)  # base64
    format: AudioFormat
    locale: Locale = Locale.VI


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _format_fact(content: dict[str, Any]) -> str | None:
    """Format a confirmed fact for inclusion in the conversation context."""
    fact_key = content.get("fact")
    value = content.get("value")
    if isinstance(value, str) and value.strip():
        v = value.strip()
        if fact_key == "owner_name":
            return f"Tên của bạn ấy là {v}"
        if fact_key == "favorite_color":
            return f"Màu yêu thích của bạn ấy là {v}"
        if fact_key == "favorite_food":
            return f"Món ăn yêu thích của bạn ấy là {v}"
        if fact_key == "birthday":
            return f"Sinh nhật của bạn ấy là ngày {v}"
        if fact_key == "pet_companion":
            return f"Bạn ấy có nuôi {v}"

    raw_summary = content.get("summary") or content.get("text")
    if raw_summary and isinstance(raw_summary, str) and raw_summary.strip():
        s = str(raw_summary).strip()
        if s.startswith("Tên của bạn là "):
            return f"Tên của bạn ấy là {s[len('Tên của bạn là ') :]}"
        if s.startswith("Màu yêu thích của bạn là "):
            return f"Màu yêu thích của bạn ấy là {s[len('Màu yêu thích của bạn là ') :]}"
        if s.startswith("Món ăn yêu thích của bạn là "):
            return f"Món ăn yêu thích của bạn ấy là {s[len('Món ăn yêu thích của bạn là ') :]}"
        if s.startswith("Sinh nhật của bạn là "):
            return f"Sinh nhật của bạn ấy là {s[len('Sinh nhật của bạn là ') :]}"
        return s
    return None


@router.post("/conversations/{pet_id}/turns")
async def turn(
    pet_id: str, body: TurnBody, request: Request, owner: U3Owner, session: Session
) -> StreamingResponse:
    try:
        repo = state(request, "identity_repository")(session)
        await get_owned_pet(repo, owner.owner_id, PetId(pet_id))
    except (ValueError, IdentityError) as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    # CLD-14, KD-4: Extract prospective memory candidates (stated preferences, facts, knowledge).
    # Extracted candidates are saved in awaiting_owner_confirmation so the owner can review.
    # Strict safety rules apply: credentials/door codes/chit-chat return None and are never saved.
    extracted = extract_candidate(body.text)
    if extracted is None and body.extractKnowledge:
        extracted = extract_knowledge_candidate(body.text)
    candidate_info: dict[str, Any] | None = None
    confirmed_facts: list[str] = []

    try:
        mem_factory: Callable[[object], MemoryRepository] = state(request, "memory_repository")
        mem_repo = mem_factory(session)

        # Layer B: Query confirmed facts for this pet
        confirmed_candidates = await mem_repo.list_candidates(
            PetId(pet_id), status=CandidateStatus.CONFIRMED
        )
        seen_facts: set[str] = set()
        knowledge_items: list[str] = []
        for cand in confirmed_candidates:
            if cand.candidate_type == CandidateType.FACT:
                raw_fact_key = cand.content.get("fact")
                dedup_key = (
                    str(raw_fact_key) if raw_fact_key else str(cand.content.get("summary", ""))
                )
                if dedup_key in seen_facts:
                    continue
                seen_facts.add(dedup_key)
                formatted = _format_fact(cand.content)
                if formatted and len(confirmed_facts) < 5:
                    confirmed_facts.append(formatted)
            elif cand.candidate_type == CandidateType.KNOWLEDGE:
                t = cand.content.get("text")
                if isinstance(t, str) and t.strip():
                    item = t.strip()
                    if item not in knowledge_items and len(knowledge_items) < 5:
                        knowledge_items.append(item)

        if extracted is not None:
            existing = await mem_repo.list_candidates(
                PetId(pet_id), status=CandidateStatus.AWAITING_OWNER_CONFIRMATION
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
                now = datetime.now(UTC)
                domain_id: uuid.UUID | None = None
                relation: str | None = None
                fact_key: int | None = None

                if extracted.candidate_type == CandidateType.KNOWLEDGE:
                    text_content = str(extracted.content.get("text", ""))
                    fact_key = compute_fact_key(text_content)
                    try:
                        k_factory = getattr(request.app.state, "knowledge_repository", None)
                        if k_factory is not None:
                            k_repo = k_factory(session)
                            domains = await k_repo.list_domains(
                                PetId(pet_id), include_deleted=False
                            )
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
                                cand_factory = getattr(
                                    request.app.state, "knowledge_candidate_adapter", None
                                )
                                if cand_factory is not None:
                                    c_port = cand_factory(session)
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
                    pet_id=PetId(pet_id),
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
        # Candidate extraction and fact lookup are best-effort; must not fail turn
        pass

    await session.close()  # release the DB connection before a long provider call
    try:
        chat_request = build_request(
            body.text,
            body.locale,
            state(request, "settings").text_timeout_s,
            body.context(
                confirmed_facts=tuple(confirmed_facts),
                knowledge=tuple(knowledge_items),
            ),
        )
    except ValueError as error:
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "text") from error
    generator: TextGenerator = state(request, "text_generator")
    synthesizer: SpeechSynthesizer | None = state(request, "speech_synthesizer")
    # PE-4c: closed tags from the owner's words, sent with `finished` only;
    # the app forwards them to the device. Never stored.
    signals = {"signals": list(conversation_signals(body.text, len(body.history or ())))}
    candidate_payload = {"candidate": candidate_info} if candidate_info is not None else {}

    async def events() -> AsyncIterator[str]:
        text_events = chat_turn(generator, chat_request)
        stream: AsyncIterator[SpokenTurnEvent] = (
            spoken_turn(text_events, synthesizer, body.locale) if body.speak else text_events
        )
        async for event in stream:
            match event:
                case TextDelta(text=text):
                    yield _sse("delta", {"text": text})
                case ToolCallRequested():
                    continue  # no server tools enabled yet
                case ChatFinished() as done:
                    yield _sse("finished", _finished(done) | signals | candidate_payload)
                case SpokenFinished(finished=done, spoken=spoken, first_audio_ms=first_ms):
                    timing = {} if first_ms is None else {"firstAudioMs": first_ms}
                    yield _sse(
                        "finished",
                        _finished(done) | {"spoken": spoken} | timing | signals | candidate_payload,
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
    try:
        text = await transcribe_once(recognizer, audio, body.format, body.locale)
    except ValueError as error:
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "audio") from error
    except ProviderUnavailableError as error:
        raise AppError("provider_unavailable", HTTPStatus.SERVICE_UNAVAILABLE) from error
    return {"text": text, "latencyMs": int((time.perf_counter() - started) * 1000)}
