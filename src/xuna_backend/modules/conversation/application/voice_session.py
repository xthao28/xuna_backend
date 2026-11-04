"""Voice WebSocket session state machine (VOICE-WS, 2026-10-23).

Protocol:
Client -> Server: hello, listen.start, binary audio frames, listen.stop, turn.commit, abort, bye.
Server -> Client: hello.ok, stt.final, gesture, reply.delta, tts.start, binary audio,
                  tts.end, reply.finished, error.
"""

import math
import struct
import time
import uuid
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Literal

from xuna_backend.modules.cognition.application.ports import (
    SpeechRecognizer,
    SpeechSynthesizer,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.types import Locale, ToolSpec
from xuna_backend.modules.conversation.application.chat_turn import (
    ReplyBudget,
    ToolCallRunner,
    TurnContext,
    build_request,
    chat_turn,
)
from xuna_backend.modules.conversation.application.spoken_turn import (
    SpokenTurnEvent,
    spoken_turn,
)
from xuna_backend.modules.conversation.application.voice import (
    MAX_AUDIO_BYTES,
    AudioFormat,
    transcribe_once,
    wav_container,
)
from xuna_backend.modules.conversation.domain.turn_mode import TurnMode
from xuna_backend.modules.identity.domain.pet_id import PetId


class SessionState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    AWAITING_COMMIT = "awaiting_commit"
    REPLYING = "replying"


class VoiceSessionError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


VadEvent = Literal["vad.start", "vad.end"]


def compute_pcm16_rms(chunk: bytes) -> float:
    """Computes RMS amplitude of little-endian 16-bit PCM audio samples (WS-3 VAD)."""
    sample_count = len(chunk) // 2
    if sample_count <= 0:
        return 0.0
    samples = struct.unpack(f"<{sample_count}h", chunk[: sample_count * 2])
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / sample_count)


class VoiceSession:
    def __init__(
        self,
        session_id: str,
        pet_id: PetId,
        owner_id: uuid.UUID,
        recognizer: SpeechRecognizer | None,
        generator: TextGenerator,
        synthesizer: SpeechSynthesizer | None,
        locale: Locale | None = None,
        voice: str | None = None,
        speak: bool = True,
        sample_rate: int = 16000,
        vad_enabled: bool = False,
        vad_speech_rms_threshold: float = 500.0,
        vad_silence_rms_threshold: float = 250.0,
        vad_speech_frames_required: int = 2,
        vad_silence_frames_required: int = 6,
    ) -> None:
        self.session_id = session_id
        self.pet_id = pet_id
        self.owner_id = owner_id
        self.recognizer = recognizer
        self.generator = generator
        self.synthesizer = synthesizer
        self.locale = locale
        self.voice = voice
        self.speak = speak
        self.sample_rate = sample_rate
        self.vad_enabled = vad_enabled
        self.vad_speech_rms_threshold = vad_speech_rms_threshold
        self.vad_silence_rms_threshold = vad_silence_rms_threshold
        self.vad_speech_frames_required = vad_speech_frames_required
        self.vad_silence_frames_required = vad_silence_frames_required

        self.state = SessionState.IDLE
        self._audio_buffer = bytearray()
        self._listen_started_at: float = 0.0
        self._vad_speech_active: bool = False
        self._vad_speech_frames: int = 0
        self._vad_silence_frames: int = 0

    def start_listening(self) -> None:
        # Can start listening from IDLE, AWAITING_COMMIT, or barge-in during REPLYING
        self._audio_buffer.clear()
        self.state = SessionState.LISTENING
        self._listen_started_at = time.monotonic()
        self._vad_speech_active = False
        self._vad_speech_frames = 0
        self._vad_silence_frames = 0

    def append_audio(self, chunk: bytes) -> VadEvent | None:
        if self.state != SessionState.LISTENING:
            return None
        if len(self._audio_buffer) + len(chunk) > MAX_AUDIO_BYTES:
            raise VoiceSessionError("audio_too_large")
        self._audio_buffer.extend(chunk)
        if not self.vad_enabled:
            return None
        rms = compute_pcm16_rms(chunk)
        if not self._vad_speech_active:
            if rms >= self.vad_speech_rms_threshold:
                self._vad_speech_frames += 1
                if self._vad_speech_frames >= self.vad_speech_frames_required:
                    self._vad_speech_active = True
                    self._vad_silence_frames = 0
                    return "vad.start"
            else:
                self._vad_speech_frames = 0
        else:
            if rms < self.vad_silence_rms_threshold:
                self._vad_silence_frames += 1
                if self._vad_silence_frames >= self.vad_silence_frames_required:
                    self._vad_speech_active = False
                    return "vad.end"
            else:
                self._vad_silence_frames = 0
        return None

    async def stop_listening(self) -> tuple[str, int]:
        if self.state != SessionState.LISTENING:
            raise VoiceSessionError("invalid_state")
        self.state = SessionState.TRANSCRIBING

        if not self._audio_buffer:
            self.state = SessionState.AWAITING_COMMIT
            return "", 0

        if self.recognizer is None:
            self.state = SessionState.AWAITING_COMMIT
            raise VoiceSessionError("voice_unavailable")

        start = time.monotonic()
        wav_data = wav_container(bytes(self._audio_buffer), self.sample_rate)
        transcript = await transcribe_once(self.recognizer, wav_data, AudioFormat.WAV, self.locale)
        asr_ms = int((time.monotonic() - start) * 1000)

        self.state = SessionState.AWAITING_COMMIT
        self._audio_buffer.clear()
        return transcript, asr_ms

    async def stream_turn_reply(
        self,
        text: str,
        context: TurnContext,
        voice: str | None = None,
        *,
        mode: TurnMode = TurnMode.CHAT,
        budget: ReplyBudget | None = None,
        generator: TextGenerator | None = None,
        code_generator: TextGenerator | None = None,
        tools: tuple[ToolSpec, ...] = (),
        tool_runner: ToolCallRunner | None = None,
    ) -> AsyncIterator[SpokenTurnEvent]:
        if not text.strip():
            self.state = SessionState.IDLE
            return

        self.state = SessionState.REPLYING
        try:
            req = build_request(
                text,
                context=context,
                locale=self.locale,
                mode=mode,
                budget=budget,
                tools=tools,
            )
            chosen_gen = generator or self.generator
            text_events = chat_turn(
                chosen_gen,
                req,
                sections=(mode == TurnMode.EXPERT),
                locale_vi=(req.locale == Locale.VI),
                tool_runner=tool_runner,
                code_generator=code_generator,
            )
            chosen_voice = voice or self.voice
            if self.speak and self.synthesizer is not None:
                async for event in spoken_turn(
                    text_events, self.synthesizer, self.locale, voice=chosen_voice
                ):
                    yield event
            else:
                async for event in text_events:
                    yield event
        finally:
            self.state = SessionState.IDLE

    def abort(self, reason: str = "cancel") -> None:
        self._audio_buffer.clear()
        self.state = SessionState.IDLE
