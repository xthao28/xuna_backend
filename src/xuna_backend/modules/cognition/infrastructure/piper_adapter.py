"""Local TTS adapter: Piper (`piper-tts`, GPL-3.0) with an ONNX voice file.

Optional extra (`pip install .[piper]`), never a default dependency: it keeps
GPL code out of the default install. V0 benchmark (2026-10-22): voice
`vi_VN-vais1000-medium` (dataset CC BY 4.0) gave first audio ≈ 130 ms on a 2014
laptop CPU, offline, ASR round-trip WER 4.3 %. Synthesis runs in a worker thread
so the event loop stays free.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.types import AudioChunk, Capability, Locale

NAME = "piper"


class PiperSpeechSynthesizer:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ValueError("piper model file not found")
        try:
            # Optional GPL extra, imported on demand (absent in the default install).
            from piper import PiperVoice  # type: ignore[import-not-found, unused-ignore]
        except ImportError as error:
            raise ValueError("piper-tts is not installed (pip install .[piper])") from error
        self._voice = PiperVoice.load(str(path))

    def _render(self, text: str) -> list[tuple[bytes, int]]:
        return [(c.audio_int16_bytes, c.sample_rate) for c in self._voice.synthesize(text)]

    async def synthesize(
        self,
        text: str,
        locale: Locale | None = None,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        try:
            chunks = await asyncio.to_thread(self._render, text)
        except Exception as error:  # onnxruntime errors have no common base class
            raise ProviderUnavailableError(type(error).__name__) from error
        for data, rate in chunks:
            yield AudioChunk(data, "pcm16", rate)


def register(registry: ProviderRegistry) -> None:
    def synthesizer(options: Mapping[str, Any]) -> object:
        return PiperSpeechSynthesizer(model_path=str(options.get("model_path", "")))

    registry.register(Capability.SPEECH_SYNTHESIS, NAME, synthesizer)
