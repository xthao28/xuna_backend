"""Unit & API tests for SYM-LIVE REST routes, VOICE-WS (WS-3 VAD), and PE-4 parity."""

from __future__ import annotations

import asyncio
import struct
import uuid
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.cognition.application.ports import TextGenerator
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Locale,
    TextDelta,
)
from xuna_backend.modules.conversation.application.chat_turn import (
    PetAction,
    PetMood,
    PetSelf,
    PetTemperament,
    TurnContext,
    build_request,
    chat_turn,
)
from xuna_backend.modules.conversation.application.voice_session import (
    VoiceSession,
    compute_pcm16_rms,
)
from xuna_backend.modules.identity.domain.pet_id import PetId


class _CapturingGenerator(TextGenerator):
    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.captured_requests: list[ChatRequest] = []

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.captured_requests.append(request)
        return self._events()

    async def _events(self) -> AsyncIterator[ChatEvent]:
        yield TextDelta(text=self.reply_text)
        yield ChatFinished(
            provider="fake",
            model="fake-model",
            input_tokens=12,
            output_tokens=6,
            latency_ms=15,
        )


def _client() -> TestClient:
    settings = Settings(
        environment=Environment.TEST,
        database_url="postgresql+psycopg://postgres:postgres@localhost:5432/xuna_test",
        text_provider="echo",
    )
    return TestClient(create_app(settings))


def test_sym_live_api_dream_consolidate_unstuck_coach_and_social_encounter() -> None:
    client = _client()

    # 1. POST /v1/symbiotic/dream-consolidate
    dream_res = client.post(
        "/v1/symbiotic/dream-consolidate",
        json={
            "schemaVersion": 1,
            "project": "xuna_pet",
            "branch": "dev",
            "dirtyCount": 2,
            "nextStepHint": "Hoàn thiện SYM-LIVE",
            "confirmedFacts": ["Bạn thích trà xanh buổi sáng"],
            "activePorts": [8000],
            "continuousWorkMinutes": 140,
            "localHour": 23,
        },
    )
    assert dream_res.status_code == 200
    dream_body = dream_res.json()
    assert dream_body["dreamVignette"]["relatedProject"] == "xuna_pet"
    assert "xuna_pet" in dream_body["dreamVignette"]["narrativeVi"]
    assert dream_body["coRegulationState"]["mode"] == "drowsy_care"
    assert dream_body["coRegulationState"]["suggestedWorkflow"] == "freeze_for_tonight"

    # 2. POST /v1/symbiotic/unstuck-coach
    coach_res = client.post(
        "/v1/symbiotic/unstuck-coach",
        json={
            "schemaVersion": 1,
            "goal": "Tối ưu hóa toàn bộ hệ thống Cộng sinh",
            "project": "xuna_pet",
            "toolHistory": [
                "workspace.git_summary",
                "dev.ports",
                "workspace.git_summary",
                "dev.ports",
            ],
        },
    )
    assert coach_res.status_code == 200
    coach_body = coach_res.json()
    assert "5 phút" in coach_body["firstFiveMinuteStep"]
    assert len(coach_body["followUpSteps"]) == 2
    assert coach_body["serendipityNote"]["noteId"].startswith("srp-")
    assert len(coach_body["habitProposals"]) == 1
    assert coach_body["habitProposals"][0]["steps"] == ["workspace.git_summary", "dev.ports"]

    # 3. POST /v1/symbiotic/social-encounters
    social_res = client.post(
        "/v1/symbiotic/social-encounters",
        json={
            "schemaVersion": 1,
            "myShortId": "D945E8F0",
            "peerShortId": "A1B2C3D4",
            "peerTrait": "curiosity",
            "peerTint": "golden_amber",
            "sharedTrick": "spin_dance",
        },
    )
    assert social_res.status_code == 200
    social_body = social_res.json()
    assert social_body["encounter"]["greetingKind"] == "harmonious_spark"
    assert len(social_body["postcard"]["signatureHex"]) == 64

    # Self-encounter returns 422 invalid_argument
    bad_social = client.post(
        "/v1/symbiotic/social-encounters",
        json={
            "schemaVersion": 1,
            "myShortId": "A1B2C3D4",
            "peerShortId": "a1b2c3d4",
        },
    )
    assert bad_social.status_code == 422
    assert bad_social.json()["code"] == "invalid_argument"


def test_voice_ws3_pcm16_rms_vad_emits_start_and_end_only_when_enabled() -> None:
    loud_frame = struct.pack("<160h", *([1200] * 160))
    quiet_frame = struct.pack("<160h", *([40] * 160))

    assert compute_pcm16_rms(loud_frame) == pytest.approx(1200.0)
    assert compute_pcm16_rms(quiet_frame) == pytest.approx(40.0)

    pet_id = PetId(uuid.uuid4().hex)
    owner_id = uuid.uuid4()
    gen = _CapturingGenerator("Chào bạn!")

    # 1. Push-to-talk mode (vad_enabled=False): never emits VAD events
    ptt_session = VoiceSession(
        session_id="s-ptt",
        pet_id=pet_id,
        owner_id=owner_id,
        recognizer=None,
        generator=gen,
        synthesizer=None,
        vad_enabled=False,
    )
    ptt_session.start_listening()
    for _ in range(5):
        assert ptt_session.append_audio(loud_frame) is None

    # 2. VAD mode (vad_enabled=True): emits vad.start on speech and vad.end after silence
    vad_session = VoiceSession(
        session_id="s-vad",
        pet_id=pet_id,
        owner_id=owner_id,
        recognizer=None,
        generator=gen,
        synthesizer=None,
        vad_enabled=True,
        vad_speech_frames_required=2,
        vad_silence_frames_required=3,
    )
    vad_session.start_listening()
    assert vad_session.append_audio(quiet_frame) is None
    assert vad_session.append_audio(loud_frame) is None
    assert vad_session.append_audio(loud_frame) == "vad.start"
    # Continued speech does not re-emit vad.start
    assert vad_session.append_audio(loud_frame) is None
    # 3 consecutive quiet frames emit vad.end
    assert vad_session.append_audio(quiet_frame) is None
    assert vad_session.append_audio(quiet_frame) is None
    assert vad_session.append_audio(quiet_frame) == "vad.end"


def test_pe4_switching_ai_providers_produces_identical_personality_context() -> None:
    async def _run() -> None:
        context = TurnContext(
            pet=PetSelf(
                active_seconds=7200,
                learned_events=14,
                current=PetAction.CURIOUS,
                quiet_choice=PetAction.IDLE,
                play_choice=PetAction.HAPPY,
                temperament=PetTemperament.CURIOUS,
                mood=PetMood.JOYFUL,
            ),
            confirmed_facts=("Bạn thích cà phê sữa đá",),
        )

        provider_a = _CapturingGenerator("Mình nhớ bạn thích cà phê sữa đá!")
        provider_b = _CapturingGenerator("Chào bạn, hôm nay mình rất vui!")

        req_a = build_request("Hôm nay thế nào?", context=context, locale=Locale.VI)
        req_b = build_request("Hôm nay thế nào?", context=context, locale=Locale.VI)

        _ = [ev async for ev in chat_turn(provider_a, req_a)]
        _ = [ev async for ev in chat_turn(provider_b, req_b)]

        assert len(provider_a.captured_requests) == 1
        assert len(provider_b.captured_requests) == 1
        assert provider_a.captured_requests[0].messages == provider_b.captured_requests[0].messages
        system_prompt = provider_a.captured_requests[0].messages[0].content
        assert "tò mò" in system_prompt.lower()
        assert "rất vui" in system_prompt.lower()

    asyncio.run(_run())
