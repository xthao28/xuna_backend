"""ASR adapter for any OpenAI-style `POST {base}/audio/transcriptions` server.

Selected for phone voice V1 by the V0 benchmark (2026-10-22): Groq
`whisper-large-v3-turbo` reached WER 2.2 % clean / 3.6 % with 10 dB noise on the
synthetic Vietnamese corpus, p50 1.4 s with a kept-alive connection. Also fits a
self-hosted faster-whisper server later. The audio is sent for this call only and
never stored or logged; transcripts are untrusted (owner confirmation still
gates anything that teaches XUNA).
"""

from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.types import (
    AudioChunk,
    Capability,
    Locale,
    Transcript,
)

NAME = "openai_transcription"
# Container → (file name, MIME type) for the multipart upload.
_UPLOADS = {
    "ogg_opus": ("speech.ogg", "audio/ogg"),
    "m4a_aac": ("speech.m4a", "audio/mp4"),
    "wav": ("speech.wav", "audio/wav"),
}
# Whisper hears the pet's name as "Equenna"/"x quên" without a hint (V0).
_PROMPT = "XUNA"


class OpenAITranscriptionRecognizer:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")) or not model:
            raise ValueError("base_url (http/https) and model are required")
        if not 0 < timeout_s <= 60:
            raise ValueError("timeout_s out of range")
        self._url = base_url.rstrip("/") + "/audio/transcriptions"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # One client for the process: a new TLS handshake per call cost 0.25-0.7 s.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)), transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def transcribe(
        self, audio: AsyncIterator[AudioChunk], locale: Locale | None = None
    ) -> AsyncIterator[Transcript]:
        data = b""
        codec = ""
        async for chunk in audio:
            if codec and chunk.codec != codec:
                raise ValueError("mixed codecs in one upload")
            codec = chunk.codec
            data += chunk.data
        upload = _UPLOADS.get(codec)
        if upload is None or not data:
            raise ValueError("unsupported or empty audio")
        request_data: dict[str, str] = {
            "model": self._model,
            "prompt": _PROMPT,
            "temperature": "0",
        }
        if locale is not None:
            request_data["language"] = locale.value
        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                files={"file": (upload[0], data, upload[1])},
                data=request_data,
            )
        except httpx.HTTPError as error:
            raise ProviderUnavailableError(type(error).__name__) from error
        if response.status_code != 200:
            raise ProviderUnavailableError(f"status {response.status_code}")
        try:
            text = str(response.json()["text"]).strip()
        except (ValueError, KeyError, TypeError) as error:
            raise ProviderUnavailableError("malformed response") from error
        yield Transcript(text=text, final=True)


def register(registry: ProviderRegistry) -> None:
    def recognizer(options: Mapping[str, Any]) -> object:
        return OpenAITranscriptionRecognizer(
            base_url=str(options.get("base_url", "")),
            model=str(options.get("model", "")),
            api_key=str(options.get("api_key", "")),
            timeout_s=float(options.get("timeout_s", 10.0)),
        )

    registry.register(Capability.SPEECH_RECOGNITION, NAME, recognizer)
