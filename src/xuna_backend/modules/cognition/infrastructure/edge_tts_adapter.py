"""TTS adapter for Microsoft Edge's free online voices (`edge-tts`, LGPL-3.0).

Vietnamese voices `vi-VN-HoaiMyNeural` / `vi-VN-NamMinhNeural`; first audio
0.47-0.63 s in the V0 benchmark, but the free service is flaky: on the owner's
Mac (2026-10-23) 2 of 12 sentences answered "no audio" after ~2.7 s and others
took up to 5.4 s, while the same text worked on the next call. Each sentence is
therefore retried within one time budget, and audio is only yielded once a
whole attempt succeeded, so a stream cut mid-sentence never plays half a
sentence. Callers treat `ProviderUnavailableError` as "show the text only".
Unofficial service: fine for development, not a production commitment.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

import edge_tts

from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.language_detector import detect_text_language
from xuna_backend.modules.cognition.domain.types import AudioChunk, Capability, Locale

NAME = "edge"
_DEFAULT_VOICES = {Locale.VI: "vi-VN-HoaiMyNeural", Locale.EN: "en-US-AriaNeural"}
SAMPLE_RATE = 24000  # edge-tts default output: 24 kHz mono MP3
MAX_ATTEMPTS = 3


class EdgeSpeechSynthesizer:
    def __init__(
        self, voice: str = "", timeout_s: float = 10.0, max_attempts: int = MAX_ATTEMPTS
    ) -> None:
        if not 0 < timeout_s <= 60:
            raise ValueError("timeout_s out of range")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._voice = voice
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts

    async def synthesize(
        self,
        text: str,
        locale: Locale | None = None,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        target_locale = detect_text_language(text, default=locale or Locale.VI)
        chosen = voice or self._voice
        if target_locale == Locale.EN:
            active_voice = (
                chosen
                if (chosen and chosen.startswith("en-"))
                else _DEFAULT_VOICES[Locale.EN]
            )
        else:
            active_voice = (
                chosen
                if (chosen and not chosen.startswith("en-"))
                else _DEFAULT_VOICES[Locale.VI]
            )
        audio: list[bytes] = []
        last: Exception | None = None
        try:
            # One budget for every attempt of this sentence.
            async with asyncio.timeout(self._timeout_s):
                for _ in range(self._max_attempts):
                    try:
                        audio = await _attempt(text, active_voice)
                    # EdgeTTS "no audio", dropped streams and aiohttp handshake
                    # errors (403) share no base class.
                    except Exception as error:
                        last = error
                        continue
                    if audio:
                        break
                    last = RuntimeError("empty audio")
        except TimeoutError as error:
            last = error
            audio = []
        if not audio:
            raise ProviderUnavailableError(type(last).__name__) from last
        for data in audio:
            yield AudioChunk(data, "mp3", SAMPLE_RATE)


async def _attempt(text: str, voice: str) -> list[bytes]:
    return [
        bytes(chunk["data"])
        async for chunk in edge_tts.Communicate(text, voice).stream()
        if chunk["type"] == "audio" and chunk.get("data")
    ]


def register(registry: ProviderRegistry) -> None:
    def synthesizer(options: Mapping[str, Any]) -> object:
        return EdgeSpeechSynthesizer(
            voice=str(options.get("voice", "")),
            timeout_s=float(options.get("timeout_s", 10.0)),
        )

    registry.register(Capability.SPEECH_SYNTHESIS, NAME, synthesizer)
