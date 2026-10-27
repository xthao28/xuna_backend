"""Spoken reply that overlaps with the text stream (phone voice V1 latency, 2026-10-23).

The reply text keeps streaming while finished sentences are synthesised, up to
`max_parallel` at a time, instead of the text waiting for each sentence's TTS.
Audio still reaches the app in sentence order, and the first piece may end at
a comma so the first sound comes sooner. A sentence whose TTS fails is skipped
(reported once) and the later sentences are still spoken: the free Edge voice
fails now and then, and one miss used to silence the rest of the reply
(owner 2026-10-26). Nothing is stored.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import cast

from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    SpeechSynthesizer,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    Locale,
    TextDelta,
)
from xuna_backend.modules.conversation.application.chat_turn import ProviderFailed
from xuna_backend.modules.conversation.application.voice import (
    SentenceSplitter,
    speakable,
    wav_container,
)

# Edge's free service tolerates a couple of parallel calls; more risks 403s.
MAX_PARALLEL_TTS = 2
# Shortest first clause cut at a comma ("Lá cây đổi màu vào mùa thu," = 27).
FIRST_CLAUSE_CHARS = 16


@dataclass(frozen=True, slots=True)
class SpokenAudio:
    """One synthesised sentence, numbered from 1 in speaking order."""

    seq: int
    text: str
    codec: str
    sample_rate: int
    data: bytes


@dataclass(frozen=True, slots=True)
class SpeechUnavailable:
    """Sent at most once: `voice_unavailable` (no TTS configured, nothing is
    spoken) or `provider_unavailable` (a sentence's TTS failed and was skipped;
    later sentences are still spoken); the text reply continues."""

    code: str


@dataclass(frozen=True, slots=True)
class SpokenFinished:
    """Replaces `ChatFinished` on a spoken turn, after the last audio."""

    finished: ChatFinished
    # False when nothing could be spoken or at least one sentence was skipped.
    spoken: bool
    # Turn start → first audio ready on the server; None when nothing was spoken.
    first_audio_ms: int | None


SpokenTurnEvent = ChatEvent | ProviderFailed | SpokenAudio | SpeechUnavailable | SpokenFinished

_DONE = object()
_Job = tuple[str, asyncio.Task[tuple[bytes, str, int]]]


async def spoken_turn(
    events: AsyncIterator[ChatEvent | ProviderFailed],
    synthesizer: SpeechSynthesizer | None,
    locale: Locale | None = None,
    *,
    voice: str | None = None,
    max_parallel: int = MAX_PARALLEL_TTS,
    first_clause_chars: int = FIRST_CLAUSE_CHARS,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[SpokenTurnEvent]:
    """Re-emits the text events with spoken audio merged in as it becomes ready."""
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    started = clock()
    limit = asyncio.Semaphore(max_parallel)
    out: asyncio.Queue[object] = asyncio.Queue()
    jobs: asyncio.Queue[_Job | None] = asyncio.Queue()
    pending: list[asyncio.Task[tuple[bytes, str, int]]] = []
    state = _State()

    async def synthesize(tts: SpeechSynthesizer, text: str) -> tuple[bytes, str, int]:
        async with limit:
            audio = b""
            codec, rate = "", 0
            call = (
                tts.synthesize(text, locale, voice=voice)
                if voice is not None
                else tts.synthesize(text, locale)
            )
            async for chunk in call:
                audio += chunk.data
                codec, rate = chunk.codec, chunk.sample_rate
        if audio and codec == "pcm16":
            return wav_container(audio, rate), "wav", rate
        return audio, codec, rate

    def submit(sentence: str) -> None:
        if state.speech_off:
            return
        if synthesizer is None:
            state.speech_off = True
            out.put_nowait(SpeechUnavailable("voice_unavailable"))
            return
        text = speakable(sentence)
        if not text:
            return
        task = asyncio.create_task(synthesize(synthesizer, text))
        pending.append(task)
        jobs.put_nowait((sentence, task))

    async def produce() -> None:
        splitter = SentenceSplitter(first_clause_chars=first_clause_chars)
        try:
            async for event in events:
                match event:
                    case TextDelta(text=text):
                        out.put_nowait(event)
                        for sentence in splitter.feed(text):
                            submit(sentence)
                    case ChatFinished():
                        for sentence in splitter.flush():
                            submit(sentence)
                        state.finished = event
                    case ProviderFailed():
                        state.failed = True
                        out.put_nowait(event)
                    case _:
                        out.put_nowait(event)
        finally:
            jobs.put_nowait(None)
            out.put_nowait(_DONE)

    async def speak() -> None:
        try:
            while (job := await jobs.get()) is not None:
                sentence, task = job
                if state.speech_off or state.failed:
                    task.cancel()
                    continue
                try:
                    audio, codec, rate = await task
                except ProviderUnavailableError:
                    # Skip this sentence only; report the first miss once.
                    if not state.speech_failed:
                        out.put_nowait(SpeechUnavailable("provider_unavailable"))
                    state.speech_failed = True
                    continue
                if not audio:
                    continue
                state.seq += 1
                if state.first_audio_ms is None:
                    state.first_audio_ms = round((clock() - started) * 1000)
                out.put_nowait(SpokenAudio(state.seq, sentence, codec, rate, audio))
        finally:
            out.put_nowait(_DONE)

    producer = asyncio.create_task(produce())
    speaker = asyncio.create_task(speak())
    try:
        done = 0
        while done < 2:
            item = await out.get()
            if item is _DONE:
                done += 1
            elif isinstance(item, SpokenAudio) and state.failed:
                continue  # the text reply failed: the app shows its local template
            else:
                yield cast(SpokenTurnEvent, item)
        await producer  # re-raise unexpected errors
        await speaker
        if state.finished is not None and not state.failed:
            yield SpokenFinished(
                state.finished,
                spoken=synthesizer is not None and not state.speech_failed,
                first_audio_ms=state.first_audio_ms,
            )
    finally:
        for task in (producer, speaker, *pending):
            task.cancel()
        await asyncio.gather(producer, speaker, *pending, return_exceptions=True)


@dataclass
class _State:
    finished: ChatFinished | None = None
    failed: bool = False
    speech_off: bool = False
    speech_failed: bool = False
    seq: int = 0
    first_audio_ms: int | None = None
