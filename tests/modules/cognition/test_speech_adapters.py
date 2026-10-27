"""ASR/TTS adapters with fake transports: no real vendor calls."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx
import pytest

from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.domain.types import AudioChunk, Locale, Transcript
from xuna_backend.modules.cognition.infrastructure import edge_tts_adapter
from xuna_backend.modules.cognition.infrastructure.edge_tts_adapter import EdgeSpeechSynthesizer
from xuna_backend.modules.cognition.infrastructure.openai_transcription import (
    OpenAITranscriptionRecognizer,
)


async def _chunks(*chunks: AudioChunk) -> AsyncIterator[AudioChunk]:
    for chunk in chunks:
        yield chunk


def _run(recognizer: OpenAITranscriptionRecognizer, *chunks: AudioChunk) -> list[Transcript]:
    async def go() -> list[Transcript]:
        try:
            return [t async for t in recognizer.transcribe(_chunks(*chunks), Locale.VI)]
        finally:
            await recognizer.aclose()

    return asyncio.run(go())


def test_transcription_sends_one_file_with_language_and_name_hint() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": " Chào XUNA "})

    recognizer = OpenAITranscriptionRecognizer(
        "https://asr.example/v1",
        "whisper-large-v3-turbo",
        "k",
        transport=httpx.MockTransport(handler),
    )
    out = _run(recognizer, AudioChunk(b"OggS1", "ogg_opus", 0), AudioChunk(b"23", "ogg_opus", 0))
    assert out == [Transcript("Chào XUNA", final=True)]
    request = seen[0]
    assert str(request.url) == "https://asr.example/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer k"
    body = request.content
    for part in (
        b'name="language"\r\n\r\nvi',
        b'name="prompt"\r\n\r\nXUNA',
        b"speech.ogg",
        b"OggS123",
    ):
        assert part in body


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(429, json={}),
        httpx.Response(200, json={"nope": 1}),
        httpx.Response(200, text="x"),
    ],
)
def test_transcription_failures_become_provider_unavailable(response: httpx.Response) -> None:
    recognizer = OpenAITranscriptionRecognizer(
        "https://asr.example/v1", "m", transport=httpx.MockTransport(lambda _: response)
    )
    with pytest.raises(ProviderUnavailableError):
        _run(recognizer, AudioChunk(b"OggS", "ogg_opus", 0))


def test_transcription_rejects_unknown_or_mixed_codecs() -> None:
    def never(_: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the provider")

    for chunks in (
        [AudioChunk(b"x", "flac", 0)],
        [AudioChunk(b"x", "wav", 0), AudioChunk(b"y", "ogg_opus", 0)],
    ):
        recognizer = OpenAITranscriptionRecognizer(
            "https://asr.example/v1", "m", transport=httpx.MockTransport(never)
        )
        with pytest.raises(ValueError):
            _run(recognizer, *chunks)


class _FakeCommunicate:
    """One plan per call: each call (attempt) plays the next list."""

    plans: ClassVar[list[list[Any]]] = []
    calls: ClassVar[int] = 0
    last_voice: ClassVar[str] = ""

    def __init__(self, text: str, voice: str) -> None:
        self.voice = voice
        _FakeCommunicate.last_voice = voice
        self.plan = self.plans[min(_FakeCommunicate.calls, len(self.plans) - 1)]
        _FakeCommunicate.calls += 1

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        for item in self.plan:
            if isinstance(item, Exception):
                raise item
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            yield item


def _speak(
    monkeypatch: pytest.MonkeyPatch, *plans: list[Any], timeout_s: float = 10.0
) -> list[AudioChunk]:
    _FakeCommunicate.plans = list(plans)
    _FakeCommunicate.calls = 0
    monkeypatch.setattr(edge_tts_adapter.edge_tts, "Communicate", _FakeCommunicate)

    async def go() -> list[AudioChunk]:
        synth = EdgeSpeechSynthesizer(timeout_s=timeout_s)
        return [c async for c in synth.synthesize("Chào", Locale.VI)]

    return asyncio.run(go())


def test_edge_yields_mp3_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _speak(
        monkeypatch,
        [
            {"type": "WordBoundary"},
            {"type": "audio", "data": b"ID3a"},
            {"type": "audio", "data": b"b"},
        ],
    )
    assert [(c.data, c.codec, c.sample_rate) for c in chunks] == [
        (b"ID3a", "mp3", 24000),
        (b"b", "mp3", 24000),
    ]
    assert _FakeCommunicate.calls == 1


def test_edge_retries_no_audio_and_a_stream_cut_mid_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = _speak(
        monkeypatch,
        [RuntimeError("No audio was received")],
        [{"type": "audio", "data": b"half"}, RuntimeError("cut")],
        [{"type": "audio", "data": b"whole"}],
    )
    # The cut attempt's half sentence is never yielded.
    assert [c.data for c in chunks] == [b"whole"]
    assert _FakeCommunicate.calls == 3


def test_edge_is_unavailable_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ProviderUnavailableError):
        _speak(monkeypatch, [RuntimeError("403")])
    assert _FakeCommunicate.calls == edge_tts_adapter.MAX_ATTEMPTS
    with pytest.raises(ProviderUnavailableError):
        _speak(monkeypatch, [{"type": "WordBoundary"}])  # never any audio
    assert _FakeCommunicate.calls == edge_tts_adapter.MAX_ATTEMPTS


def test_edge_attempts_share_one_time_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ProviderUnavailableError, match="TimeoutError"):
        _speak(monkeypatch, [0.2, {"type": "audio", "data": b"late"}], timeout_s=0.05)
    assert _FakeCommunicate.calls == 1


def test_edge_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError):
        EdgeSpeechSynthesizer(max_attempts=0)


def test_transcription_auto_detect_omits_language() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "Hello XUNA"})

    recognizer = OpenAITranscriptionRecognizer(
        "https://asr.example/v1",
        "whisper-large-v3-turbo",
        transport=httpx.MockTransport(handler),
    )

    async def go() -> list[Transcript]:
        try:
            return [
                t
                async for t in recognizer.transcribe(
                    _chunks(AudioChunk(b"OggS", "ogg_opus", 0)), None
                )
            ]
        finally:
            await recognizer.aclose()

    out = asyncio.run(go())
    assert out == [Transcript("Hello XUNA", final=True)]
    request = seen[0]
    assert b'name="language"' not in request.content


def test_edge_auto_selects_english_voice_for_english_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeCommunicate.plans = [[{"type": "audio", "data": b"voice_audio"}]]
    _FakeCommunicate.calls = 0
    _FakeCommunicate.last_voice = ""
    monkeypatch.setattr(edge_tts_adapter.edge_tts, "Communicate", _FakeCommunicate)

    async def go() -> list[AudioChunk]:
        synth = EdgeSpeechSynthesizer(voice="vi-VN-HoaiMyNeural")
        return [c async for c in synth.synthesize("Hello, nice to meet you!")]

    chunks = asyncio.run(go())
    assert len(chunks) == 1
    # Even though default voice is vi-VN-HoaiMyNeural, English text triggers en-US-AriaNeural
    assert _FakeCommunicate.last_voice == "en-US-AriaNeural"

