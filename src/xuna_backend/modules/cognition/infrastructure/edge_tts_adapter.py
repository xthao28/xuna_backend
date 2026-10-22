"""TTS adapter for Microsoft Edge's free online voices (`edge-tts`, LGPL-3.0).

Vietnamese voices `vi-VN-HoaiMyNeural` / `vi-VN-NamMinhNeural`; first audio
0.47-0.63 s in the V0 benchmark, but the free service sometimes answers 403 or
"no audio": callers treat `ProviderUnavailableError` as "show the text only".
Unofficial service: fine for development, not a production commitment.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

import edge_tts

from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.types import AudioChunk, Capability, Locale

NAME = "edge"
_DEFAULT_VOICES = {Locale.VI: "vi-VN-HoaiMyNeural", Locale.EN: "en-US-AriaNeural"}
SAMPLE_RATE = 24000  # edge-tts default output: 24 kHz mono MP3


class EdgeSpeechSynthesizer:
    def __init__(self, voice: str = "", timeout_s: float = 10.0) -> None:
        if not 0 < timeout_s <= 60:
            raise ValueError("timeout_s out of range")
        self._voice = voice
        self._timeout_s = timeout_s

    async def synthesize(self, text: str, locale: Locale) -> AsyncIterator[AudioChunk]:
        voice = self._voice or _DEFAULT_VOICES[locale]
        sent = False
        try:
            async with asyncio.timeout(self._timeout_s):
                async for chunk in edge_tts.Communicate(text, voice).stream():
                    if chunk["type"] == "audio" and chunk.get("data"):
                        sent = True
                        yield AudioChunk(bytes(chunk["data"]), "mp3", SAMPLE_RATE)
        # Timeouts, EdgeTTS "no audio", and aiohttp handshake errors (403) share no
        # base class.
        except Exception as error:
            if sent:
                return  # keep what was already streamed; the next sentence may still work
            raise ProviderUnavailableError(type(error).__name__) from error


def register(registry: ProviderRegistry) -> None:
    def synthesizer(options: Mapping[str, Any]) -> object:
        return EdgeSpeechSynthesizer(
            voice=str(options.get("voice", "")),
            timeout_s=float(options.get("timeout_s", 10.0)),
        )

    registry.register(Capability.SPEECH_SYNTHESIS, NAME, synthesizer)
