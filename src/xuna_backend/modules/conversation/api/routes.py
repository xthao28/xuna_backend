"""`POST /v1/conversations/{petId}/turns` → Server-Sent Events (contract §2), and
phone voice V1: `POST /v1/speech/transcriptions` plus spoken replies (`speak`)."""

import base64
import binascii
import json
import time
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

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
from xuna_backend.modules.conversation.application.chat_turn import (
    MAX_INPUT_CHARS,
    ProviderFailed,
    build_request,
    chat_turn,
)
from xuna_backend.modules.conversation.application.voice import (
    MAX_AUDIO_BYTES,
    AudioFormat,
    SentenceSplitter,
    speakable,
    transcribe_once,
    wav_container,
)
from xuna_backend.modules.identity.application.use_cases import IdentityError, get_owned_pet
from xuna_backend.modules.identity.domain.pet_id import PetId

router = APIRouter(prefix="/v1", tags=["conversation"])
U3Owner = Annotated[Owner, RequireConsent(ConsentClass.CONVERSATION_TEXT)]
VoiceOwner = Annotated[Owner, RequireConsent(ConsentClass.CONVERSATION_VOICE)]


class TurnBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1]
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    locale: Locale = Locale.VI
    # Phone voice V1: also stream the reply as audio, sentence by sentence.
    speak: bool = False


class TranscriptionBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1]
    audio: str = Field(min_length=4, max_length=(MAX_AUDIO_BYTES * 4) // 3 + 4)  # base64
    format: AudioFormat
    locale: Locale = Locale.VI


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
    await session.close()  # release the DB connection before a long provider call
    try:
        chat_request = build_request(
            body.text, body.locale, state(request, "settings").text_timeout_s
        )
    except ValueError as error:
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "text") from error
    generator: TextGenerator = state(request, "text_generator")
    synthesizer: SpeechSynthesizer | None = state(request, "speech_synthesizer")
    voice = _Voice(synthesizer, body.locale) if body.speak else None

    async def events() -> AsyncIterator[str]:
        splitter = SentenceSplitter()
        async for event in chat_turn(generator, chat_request):
            match event:
                case TextDelta(text=text):
                    yield _sse("delta", {"text": text})
                    if voice is not None:
                        for sentence in splitter.feed(text):
                            async for frame in voice.speak(sentence):
                                yield frame
                case ToolCallRequested():
                    continue  # no server tools enabled yet
                case ChatFinished() as done:
                    if voice is not None:
                        for sentence in splitter.flush():
                            async for frame in voice.speak(sentence):
                                yield frame
                    yield _sse(
                        "finished",
                        {
                            "provider": done.provider,
                            "model": done.model,
                            "latencyMs": done.latency_ms,
                        }
                        | ({} if voice is None else {"spoken": voice.spoken}),
                    )
                case ProviderFailed(code=code):
                    yield _sse("error", {"code": code})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
    )


class _Voice:
    """Speaks reply sentences in order. After a TTS failure it stops speaking and
    reports `speech_unavailable` once; the text reply always continues."""

    def __init__(self, synthesizer: SpeechSynthesizer | None, locale: Locale) -> None:
        self._synthesizer = synthesizer
        self._locale = locale
        self._seq = 0
        self.spoken = synthesizer is not None

    async def speak(self, sentence: str) -> AsyncIterator[str]:
        if self._synthesizer is None or not self.spoken:
            if self._synthesizer is None and self._seq == 0:
                self._seq = -1  # report the missing provider once
                yield _sse("speech_unavailable", {"code": "voice_unavailable"})
            return
        audio = b""
        codec, rate = "", 0
        try:
            async for chunk in self._synthesizer.synthesize(speakable(sentence), self._locale):
                audio += chunk.data
                codec, rate = chunk.codec, chunk.sample_rate
        except ProviderUnavailableError:
            self.spoken = False
            yield _sse("speech_unavailable", {"code": "provider_unavailable"})
            return
        if not audio:
            return
        if codec == "pcm16":
            audio, codec = wav_container(audio, rate), "wav"
        self._seq += 1
        yield _sse(
            "audio",
            {
                "seq": self._seq,
                "text": sentence,
                "format": codec,
                "sampleRate": rate,
                "data": base64.b64encode(audio).decode("ascii"),
            },
        )


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
