"""Voice WebSocket integration tests (VOICE-WS WS-1).

Protocol tests:
- Handshake timeout / invalid hello / missing token
- Pet ownership check
- U3V consent check
- Push-to-talk stream: listen.start -> binary chunks -> listen.stop -> stt.final
- Streaming reply: turn.commit -> reply.delta -> tts.start -> binary audio
  -> tts.end -> reply.finished
- Abort & Bye lifecycle
"""

import json

import pytest

from tests.helpers import LocalIssuer
from tests.modules.conversation.test_conversation_api import (
    PET,
    bind,
    headers,
)
from tests.modules.conversation.test_voice_api import (
    FakeRecognizer,
    FakeSynthesizer,
    consents,
    make,
)

pytestmark = pytest.mark.postgres


@pytest.fixture
def issuer() -> LocalIssuer:
    return LocalIssuer()


def test_ws_missing_or_invalid_hello(database_url: str, issuer: LocalIssuer) -> None:
    with (
        make(database_url, issuer, FakeRecognizer(), FakeSynthesizer()) as client,
        client.websocket_connect("/v1/voice") as ws,
    ):
        ws.send_json({"type": "not_hello"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "unsupported_protocol"


def test_ws_unauthorized_token(database_url: str, issuer: LocalIssuer) -> None:
    with (
        make(database_url, issuer, FakeRecognizer(), FakeSynthesizer()) as client,
        client.websocket_connect("/v1/voice") as ws,
    ):
        ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": "invalid_jwt_token"},
                "petId": PET,
            }
        )
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "unauthorized"


def test_ws_consent_required_when_voice_consent_denied(
    database_url: str, issuer: LocalIssuer
) -> None:
    with make(database_url, issuer, FakeRecognizer(), FakeSynthesizer()) as client:
        h = headers(issuer)
        # Bind pet so pet exists and is owned by owner
        bind(client, h)
        # Consent for text is on, but voice is off
        consents(client, h, text=True, voice=False)

        token = h["Authorization"].removeprefix("Bearer ")
        with client.websocket_connect("/v1/voice") as ws:
            ws.send_json(
                {
                    "type": "hello",
                    "schemaVersion": 1,
                    "auth": {"token": token},
                    "petId": PET,
                }
            )
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "consent_required"


def test_ws_full_push_to_talk_and_reply_flow(database_url: str, issuer: LocalIssuer) -> None:
    recognizer = FakeRecognizer(text="XUNA giỏi quá, thương bé ghê")
    synthesizer = FakeSynthesizer(pcm=False)
    with make(database_url, issuer, recognizer, synthesizer) as client:
        h = headers(issuer)
        bind(client, h)
        consents(client, h, text=True, voice=True)

        token = h["Authorization"].removeprefix("Bearer ")
        with client.websocket_connect("/v1/voice") as ws:
            # 1. Hello handshake
            ws.send_json(
                {
                    "type": "hello",
                    "schemaVersion": 1,
                    "auth": {"token": token},
                    "petId": PET,
                    "speak": True,
                }
            )
            hello_ok = ws.receive_json()
            assert hello_ok["type"] == "hello.ok"
            assert "sessionId" in hello_ok
            assert hello_ok["downlink"]["format"] == "mp3"

            # 2. Push-to-talk: listen.start
            ws.send_json({"type": "listen.start"})

            # Send binary PCM16 audio chunks (e.g. 16000Hz mono 16bit = 32000 bytes/sec)
            ws.send_bytes(b"\x00\x00" * 320)  # 20ms chunk
            ws.send_bytes(b"\x00\x00" * 320)

            # listen.stop
            ws.send_json({"type": "listen.stop"})
            stt = ws.receive_json()
            assert stt["type"] == "stt.final"
            assert stt["text"] == "XUNA giỏi quá, thương bé ghê"
            assert isinstance(stt["asrMs"], int)

            # 3. Commit turn with context
            ws.send_json(
                {
                    "type": "turn.commit",
                    "text": stt["text"],
                    "context": {
                        "pet": {
                            "activeSeconds": 3600,
                            "learnedEvents": 12,
                            "current": "idle",
                            "mood": "calm",
                        }
                    },
                }
            )

            # Receive streaming reply events
            events = []
            audio_bytes_received = []
            while True:
                msg = ws.receive()
                if msg.get("bytes"):
                    audio_bytes_received.append(msg["bytes"])
                elif msg.get("text"):
                    import json

                    parsed = json.loads(msg["text"])
                    events.append(parsed)
                    if parsed["type"] == "error":
                        raise AssertionError(f"Unexpected websocket error: {parsed}")
                    if parsed["type"] == "reply.finished":
                        break

            event_types = [e["type"] for e in events]
            assert "reply.delta" in event_types
            assert "tts.start" in event_types
            assert "tts.end" in event_types
            assert "reply.finished" in event_types
            assert len(audio_bytes_received) > 0
            # Owner 2026-10-27: no gesture unless the app says the device can play one.
            assert "gesture" not in event_types
            # PE-4c: the WebSocket turn carries the same tags as the HTTP turn.
            assert events[-1]["signals"] == ["praise", "affection"]

            # 4. Clean bye
            ws.send_json({"type": "bye"})


def test_ws_abort_barge_in(database_url: str, issuer: LocalIssuer) -> None:
    recognizer = FakeRecognizer()
    synthesizer = FakeSynthesizer()
    with make(database_url, issuer, recognizer, synthesizer) as client:
        h = headers(issuer)
        bind(client, h)
        consents(client, h, text=True, voice=True)

        token = h["Authorization"].removeprefix("Bearer ")
        with client.websocket_connect("/v1/voice") as ws:
            ws.send_json(
                {
                    "type": "hello",
                    "schemaVersion": 1,
                    "auth": {"token": token},
                    "petId": PET,
                }
            )
            assert ws.receive_json()["type"] == "hello.ok"

            # Start listening then abort
            ws.send_json({"type": "listen.start"})
            ws.send_bytes(b"\x00\x00" * 320)
            ws.send_json({"type": "abort", "reason": "user_cancelled"})

            # After abort, listen.stop should return invalid_state since listening was aborted
            ws.send_json({"type": "listen.stop"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "invalid_state"

            ws.send_json({"type": "bye"})


def test_ws_reply_gesture_comes_before_the_text(database_url: str, issuer: LocalIssuer) -> None:
    # Owner 2026-10-27: the echo generator repeats the tag; it is sent as a
    # `gesture` event once and never reaches the reply text or the speech.
    with make(database_url, issuer, FakeRecognizer(), FakeSynthesizer(pcm=False)) as client:
        h = headers(issuer)
        bind(client, h)
        consents(client, h, text=True, voice=True)

        token = h["Authorization"].removeprefix("Bearer ")
        with client.websocket_connect("/v1/voice") as ws:
            ws.send_json(
                {"type": "hello", "schemaVersion": 1, "auth": {"token": token}, "petId": PET}
            )
            assert ws.receive_json()["type"] == "hello.ok"
            ws.send_json(
                {
                    "type": "turn.commit",
                    "text": "<g:bounce_play> chơi nào",
                    "context": {"gestures": True},
                }
            )
            events = []
            while True:
                msg = ws.receive()
                if msg.get("text"):
                    parsed = json.loads(msg["text"])
                    events.append(parsed)
                    if parsed["type"] in ("reply.finished", "error"):
                        break

            assert events[0] == {"type": "gesture", "name": "bounce_play"}
            replies = [e for e in events if e["type"] in ("reply.delta", "tts.start")]
            assert replies and not any("<g:" in e["text"] for e in replies)
            assert events[-1]["type"] == "reply.finished"
            ws.send_json({"type": "bye"})
