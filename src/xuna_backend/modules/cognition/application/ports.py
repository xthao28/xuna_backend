"""Ports for cognitive providers. Adapters live in `infrastructure/`.

Vendors (LLM/ASR/TTS) are chosen by the CLD-13 benchmark (D11a open) and
plugged in through `ProviderRegistry`. Never import a vendor SDK outside its
adapter module. Every adapter enforces its own timeout and raises
`ProviderUnavailableError` so callers can fall back to local templates.
"""

from collections.abc import AsyncIterator
from typing import Protocol

from xuna_backend.modules.cognition.domain.types import (
    AudioChunk,
    ChatEvent,
    ChatRequest,
    EmbeddingRequest,
    EmbeddingResult,
    Locale,
    ToolCall,
    ToolSpec,
    Transcript,
)


class ProviderUnavailableError(Exception):
    """Outage, timeout or quota. Not for invalid requests (raise ValueError)."""


class TextGenerator(Protocol):
    """LLM chat. Streams deltas, may request tools, ends with ChatFinished."""

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]: ...


class SpeechRecognizer(Protocol):
    """Streaming ASR: audio in, partial then final transcripts out."""

    def transcribe(
        self, audio: AsyncIterator[AudioChunk], locale: Locale | None = None
    ) -> AsyncIterator[Transcript]: ...


class SpeechSynthesizer(Protocol):
    """Streaming TTS: text in, audio frames out."""

    def synthesize(
        self,
        text: str,
        locale: Locale | None = None,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]: ...


class Tool(Protocol):
    """A server-side tool. Device actions are never tools: they go through
    the device ActionGate as validated suggestions (ADR-0008)."""

    @property
    def spec(self) -> ToolSpec: ...

    async def invoke(self, call: ToolCall) -> str: ...


class Embedder(Protocol):
    """Embedding provider. Takes batch of texts, returns vector representations."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult: ...
