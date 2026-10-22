"""Provider-independent cognition types (Cloud plan §4, ADR-0008).

Capability families follow the proven split of voice-assistant servers such
as xiaozhi-esp32-server (VAD → ASR → LLM ↔ tools → TTS, plus memory/intent),
but with typed async streams and one rule they lack: provider output is
*untrusted*. It is display text or a candidate, never a confirmed fact,
preference or growth change; promotion needs owner confirmation and a durable
device commit (CLD-14).
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Capability(StrEnum):
    """One adapter is selected per capability by configuration."""

    TEXT = "text"  # LLM chat with optional tool calls
    VISION = "vision"  # multimodal LLM (gated)
    SPEECH_DETECTION = "speech_detection"  # VAD (VOICE-gated)
    SPEECH_RECOGNITION = "speech_recognition"  # ASR (VOICE-gated)
    SPEECH_SYNTHESIS = "speech_synthesis"  # TTS (VOICE-gated)


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Locale(StrEnum):
    VI = "vi"
    EN = "en"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A callable tool offered to the model (function calling / MCP)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    locale: Locale = Locale.VI
    max_output_tokens: int = 512
    timeout_s: float = 15.0

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("messages must not be empty")
        if not 1 <= self.max_output_tokens <= 8192:
            raise ValueError("max_output_tokens out of range")
        if not 0 < self.timeout_s <= 120:
            raise ValueError("timeout_s out of range")


@dataclass(frozen=True, slots=True)
class TextDelta:
    """Streamed text fragment; render it, never persist it as fact."""

    text: str
    untrusted: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class ToolCallRequested:
    """The model asks for a tool; arguments are untrusted input."""

    call: ToolCall


@dataclass(frozen=True, slots=True)
class ChatFinished:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


ChatEvent = TextDelta | ToolCallRequested | ChatFinished


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """Opus or PCM16 frame. Raw audio is never stored (D23)."""

    data: bytes
    # Frames: "opus" | "pcm16" | "mp3". Whole uploads (ASR input): "ogg_opus" |
    # "m4a_aac" | "wav" (container formats a phone records directly).
    codec: str
    sample_rate: int  # 0 when the container carries it


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    final: bool
    untrusted: bool = field(default=True, init=False)
