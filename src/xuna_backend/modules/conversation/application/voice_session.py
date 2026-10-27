"""Voice WebSocket session state machine (VOICE-WS, 2026-10-23).

Protocol:
Client -> Server: hello, listen.start, binary audio frames, listen.stop, turn.commit, abort, bye.
Server -> Client: hello.ok, stt.final, gesture, reply.delta, tts.start, binary audio,
                  tts.end, reply.finished, error.
"""

import time
import uuid
from collections.abc import AsyncIterator
from enum import StrEnum

from xuna_backend.modules.cognition.application.ports import (
    SpeechRecognizer,
    SpeechSynthesizer,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.types import Locale
from xuna_backend.modules.conversation.application.chat_turn import (
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

        self.state = SessionState.IDLE
        self._audio_buffer = bytearray()
        self._listen_started_at: float = 0.0

    def start_listening(self) -> None:
        # Can start listening from IDLE, AWAITING_COMMIT, or barge-in during REPLYING
        self._audio_buffer.clear()
        self.state = SessionState.LISTENING
        self._listen_started_at = time.monotonic()

    def append_audio(self, chunk: bytes) -> None:
        if self.state != SessionState.LISTENING:
            return
        if len(self._audio_buffer) + len(chunk) > MAX_AUDIO_BYTES:
            raise VoiceSessionError("audio_too_large")
        self._audio_buffer.extend(chunk)

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
    ) -> AsyncIterator[SpokenTurnEvent]:
        if not text.strip():
            self.state = SessionState.IDLE
            return

        self.state = SessionState.REPLYING
        try:
            req = build_request(text, context=context, locale=self.locale)
            text_events = chat_turn(self.generator, req)
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
