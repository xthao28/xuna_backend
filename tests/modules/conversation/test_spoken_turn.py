"""Spoken replies overlap TTS with the text stream; audio stays in order."""

import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest

from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.domain.types import (
    AudioChunk,
    ChatEvent,
    ChatFinished,
    Locale,
    TextDelta,
)
from xuna_backend.modules.conversation.application.chat_turn import ProviderFailed
from xuna_backend.modules.conversation.application.spoken_turn import (
    SpeechUnavailable,
    SpokenAudio,
    SpokenFinished,
    SpokenTurnEvent,
    spoken_turn,
)

DONE = ChatFinished("fake", "m", 1, 1, 5)


async def reply(
    *parts: str, gate: asyncio.Event | None = None, fail: bool = False
) -> AsyncIterator[ChatEvent | ProviderFailed]:
    for index, part in enumerate(parts):
        if gate is not None and index == 1:
            await gate.wait()
        yield TextDelta(part)
    yield ProviderFailed() if fail else DONE


class Synth:
    """TTS fake: optional per-sentence delay or gate, failure, concurrency count."""

    def __init__(
        self,
        delays: dict[str, float] | None = None,
        gate: asyncio.Event | None = None,
        fail_on: str = "",
    ) -> None:
        self.delays = delays or {}
        self.gate = gate
        self.fail_on = fail_on
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.active = 0
        self.max_active = 0

    async def synthesize(self, text: str, locale: Locale) -> AsyncIterator[AudioChunk]:
        self.started.append(text)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(self.delays.get(text, 0))
            if text == self.fail_on:
                raise ProviderUnavailableError("tts down")
            yield AudioChunk(text.encode(), "mp3", 24000)
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        finally:
            self.active -= 1


def collect(stream: AsyncIterator[SpokenTurnEvent]) -> list[SpokenTurnEvent]:
    async def run() -> list[SpokenTurnEvent]:
        return [event async for event in stream]

    return asyncio.run(asyncio.wait_for(run(), timeout=5))


def kinds(events: list[SpokenTurnEvent]) -> list[str]:
    return [type(event).__name__ for event in events]


def test_text_keeps_streaming_while_the_first_sentence_is_synthesised() -> None:
    async def run() -> list[str]:
        tts_gate = asyncio.Event()
        seen: list[str] = []
        stream = spoken_turn(reply("Một. ", "Hai. ", "Ba."), Synth(gate=tts_gate), Locale.VI)
        async for event in stream:
            seen.append(type(event).__name__)
            if seen.count("TextDelta") == 3:
                tts_gate.set()  # TTS answers only after the whole text arrived
        return seen

    seen = asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert seen[:3] == ["TextDelta"] * 3
    assert seen.count("SpokenAudio") == 3
    assert seen[-1] == "SpokenFinished"


def test_audio_keeps_sentence_order_when_a_later_sentence_is_faster() -> None:
    synth = Synth(delays={"Một câu khá dài.": 0.05})
    events = collect(spoken_turn(reply("Một câu khá dài. ", "Ngắn."), synth, Locale.VI))
    audio = [e for e in events if isinstance(e, SpokenAudio)]
    assert [(a.seq, a.text) for a in audio] == [(1, "Một câu khá dài."), (2, "Ngắn.")]
    assert synth.max_active == 2  # both were synthesised at the same time


def test_at_most_max_parallel_syntheses_run_at_once() -> None:
    synth = Synth(delays={s: 0.01 for s in ("A.", "B.", "C.", "D.", "E.")})
    events = collect(spoken_turn(reply("A. B. C. D. E."), synth, Locale.VI, max_parallel=2))
    assert [e.text for e in events if isinstance(e, SpokenAudio)] == [
        "A.",
        "B.",
        "C.",
        "D.",
        "E.",
    ]
    assert synth.max_active == 2


def test_a_failed_sentence_is_skipped_reported_once_and_later_ones_spoken() -> None:
    synth = Synth(fail_on="Hai.")
    events = collect(spoken_turn(reply("Một. ", "Hai. ", "Ba."), synth, Locale.VI))
    audio = [e for e in events if isinstance(e, SpokenAudio)]
    assert [(a.seq, a.text) for a in audio] == [(1, "Một."), (2, "Ba.")]
    assert [e for e in events if isinstance(e, SpeechUnavailable)] == [
        SpeechUnavailable("provider_unavailable")
    ]
    assert kinds(events).count("TextDelta") == 3
    finished = events[-1]
    assert isinstance(finished, SpokenFinished) and finished.spoken is False


def test_several_failed_sentences_are_still_reported_once() -> None:
    class Flaky(Synth):
        async def synthesize(self, text: str, locale: Locale) -> AsyncIterator[AudioChunk]:
            if text != "Ba.":
                raise ProviderUnavailableError("tts down")
            yield AudioChunk(text.encode(), "mp3", 24000)

    events = collect(spoken_turn(reply("Một. ", "Hai. ", "Ba."), Flaky(), Locale.VI))
    assert [e.text for e in events if isinstance(e, SpokenAudio)] == ["Ba."]
    assert kinds(events).count("SpeechUnavailable") == 1


def test_no_synthesizer_reports_voice_unavailable_once() -> None:
    events = collect(spoken_turn(reply("Một. ", "Hai."), None, Locale.VI))
    assert [e for e in events if isinstance(e, SpeechUnavailable)] == [
        SpeechUnavailable("voice_unavailable")
    ]
    finished = events[-1]
    assert isinstance(finished, SpokenFinished)
    assert (finished.spoken, finished.first_audio_ms, finished.finished) == (False, None, DONE)


def test_a_text_provider_failure_ends_with_error_and_no_audio() -> None:
    synth = Synth(delays={"Một.": 0.05})
    events = collect(spoken_turn(reply("Một. ", fail=True), synth, Locale.VI))
    # The app falls back to its local template, so no audio follows the error.
    assert kinds(events) == ["TextDelta", "ProviderFailed"]
    assert synth.active == 0


def test_first_audio_time_is_measured_from_the_turn_start() -> None:
    ticks: Iterator[float] = iter([10.0, 10.25])
    events = collect(spoken_turn(reply("Chào."), Synth(), Locale.VI, clock=lambda: next(ticks)))
    finished = events[-1]
    assert isinstance(finished, SpokenFinished)
    assert (finished.spoken, finished.first_audio_ms) == (True, 250)


def test_closing_the_stream_cancels_pending_synthesis() -> None:
    async def run(synth: Synth) -> None:
        stream = spoken_turn(reply("Một. ", "Hai. ", "Ba."), synth, Locale.VI)
        async for event in stream:
            if isinstance(event, TextDelta):
                await asyncio.sleep(0.01)  # let the syntheses start (they wait on the gate)
                break  # the app went away
        await stream.aclose()  # type: ignore[attr-defined]

    synth = Synth(gate=asyncio.Event())
    asyncio.run(asyncio.wait_for(run(synth), timeout=5))
    assert synth.active == 0
    assert synth.cancelled


def test_rejects_a_parallel_limit_below_one() -> None:
    with pytest.raises(ValueError):
        collect(spoken_turn(reply("A."), Synth(), Locale.VI, max_parallel=0))
