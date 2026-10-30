"""Voice WebSocket (`/v1/voice`) tool-calling integration tests (UDC-4 S4.4).

Uses fake ASR (`SpeechRecognizer`), fake LLM (`TextGenerator`), and fake TTS
(`SpeechSynthesizer`) with in-memory repositories so tests run in every environment.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient
from tests.helpers import LocalIssuer, MemoryAuditLog
from tests.modules.actions.test_actions_api_and_turn import (
    FAKE_PUBKEY,
    NOW,
    PET_HEX,
    _DummyDatabase,
    _FakeConsentRepo,
    _FakeHostRepo,
    _FakeIdentityRepo,
    _FakeKnowledgeCandidatePort,
    _FakeKnowledgeRepo,
    _FakeMemoryRepo,
    _FakeOwnerRepo,
    _ScriptedLLM,
)

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.infrastructure.repository import (
    InMemoryApprovalRepository,
)
from xuna_backend.modules.cognition.domain.types import (
    AudioChunk,
    ChatFinished,
    Locale,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    Transcript,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)


class _FakeRecognizer:
    def __init__(self, transcript: str = "Tạm dừng nhạc trên Mac") -> None:
        self.transcript = transcript

    async def transcribe(
        self, audio: AsyncIterator[AudioChunk], locale: Locale | None = None
    ) -> AsyncIterator[Transcript]:
        _ = [chunk async for chunk in audio]
        yield Transcript(self.transcript, final=True)


class _FakeSynthesizer:
    async def synthesize(
        self,
        text: str,
        locale: Locale | None = None,
        *,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=b"FAKE_MP3_BYTES", codec="mp3", sample_rate=24000)


def _done(latency_ms: int = 10) -> ChatFinished:
    return ChatFinished(
        provider="fake",
        model="m1",
        input_tokens=12,
        output_tokens=6,
        latency_ms=latency_ms,
    )


def _build_voice_harness(
    llm: _ScriptedLLM,
    *,
    transcript: str = "Tạm dừng nhạc trên Mac",
) -> tuple[
    TestClient,
    LocalIssuer,
    Owner,
    _FakeHostRepo,
    InMemoryApprovalRepository,
    MemoryAuditLog,
]:
    issuer = LocalIssuer()
    owner = Owner(
        owner_id=uuid.uuid4(),
        issuer="https://auth.test.xuna",
        subject="owner-1",
    )
    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
        ),
        verifier=issuer.verifier(),
        generator=llm,
        recognizer=_FakeRecognizer(transcript),
        synthesizer=_FakeSynthesizer(),
    )
    app.state.database = _DummyDatabase()
    owner_repo = _FakeOwnerRepo(owner)
    identity_repo = _FakeIdentityRepo(owner.owner_id, PET_HEX)
    consent_repo = _FakeConsentRepo()
    mem_repo = _FakeMemoryRepo()
    k_repo = _FakeKnowledgeRepo()
    c_port = _FakeKnowledgeCandidatePort()
    host_repo = _FakeHostRepo()
    approval_repo = InMemoryApprovalRepository()
    audit_log = MemoryAuditLog()

    app.state.owner_repository = lambda _s: owner_repo
    app.state.identity_repository = lambda _s: identity_repo
    app.state.consent_repository = lambda _s: consent_repo
    app.state.memory_repository = lambda _s: mem_repo
    app.state.knowledge_repository = lambda _s: k_repo
    app.state.knowledge_candidate_adapter = lambda _s: c_port
    app.state.host_repository = lambda _s: host_repo
    app.state.approval_repository = lambda _s: approval_repo
    app.state.audit_log = lambda _s: audit_log

    return TestClient(app), issuer, owner, host_repo, approval_repo, audit_log


def test_voice_websocket_push_to_talk_r1_tool_call_and_tts_reply() -> None:
    llm = _ScriptedLLM(
        [
            [
                ToolCallRequested(
                    ToolCall(
                        id="ws-call-1",
                        name="media__play_pause",
                        arguments={"target": "Mac"},
                    )
                ),
                _done(10),
            ],
            [
                TextDelta("<g:bounce_play>Mình đã tạm dừng nhạc trên Mac rồi nhé."),
                _done(14),
            ],
        ]
    )
    client, issuer, owner, host_repo, _, _ = _build_voice_harness(
        llm, transcript="Tạm dừng nhạc trên Mac"
    )

    host = Host(
        host_id=HostId.generate(),
        owner_id=owner.owner_id,
        alias=HostAlias("Mac"),
        platform=HostPlatform.MACOS,
        hostname="mac-host",
        os_version="15.0",
        agent_version="0.1.0",
        public_key=FAKE_PUBKEY,
        status=HostStatus.ACTIVE,
        created_at=NOW,
    )
    asyncio.run(host_repo.save_host(host))
    asyncio.run(
        host_repo.save_manifest(
            Manifest(
                host_id=host.host_id,
                tools=(
                    ToolDescriptor(
                        name="media.play_pause",
                        description="Toggle media playback.",
                        input_schema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                        family=ToolFamily.MEDIA,
                        risk=RiskTier.R1,
                    ),
                ),
                updated_at=NOW,
            )
        )
    )
    asyncio.run(
        host_repo.set_grants(
            host.host_id,
            [
                Grant(
                    host_id=host.host_id,
                    family=ToolFamily.MEDIA,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )
    )

    hub = client.app.state.host_session_hub  # type: ignore[union-attr]

    async def fake_send_text(raw: str) -> None:
        frame = json.loads(raw)
        if frame.get("method") == "tools/call":
            hub.on_response(
                str(host.host_id),
                frame["id"],
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {"state": "paused"},
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 15},
                },
            )

    async def noop() -> None:
        pass

    asyncio.run(hub.open_session(str(host.host_id), object(), fake_send_text, noop, noop))

    token = issuer.token("owner-1")
    with client.websocket_connect("/v1/voice") as ws:
        ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": token},
                "petId": PET_HEX,
                "speak": True,
                "clientCapabilities": ["host_actions_v1"],
            }
        )
        hello_ok = ws.receive_json()
        assert hello_ok["type"] == "hello.ok"

        # Push-to-talk ASR
        ws.send_json({"type": "listen.start"})
        ws.send_bytes(b"\x00\x00" * 320)
        ws.send_json({"type": "listen.stop"})
        stt = ws.receive_json()
        assert stt["type"] == "stt.final"
        assert stt["text"] == "Tạm dừng nhạc trên Mac"

        # Commit turn
        ws.send_json(
            {
                "type": "turn.commit",
                "text": stt["text"],
                "context": {"gestures": True},
            }
        )

        events: list[dict[str, Any]] = []
        audio_chunks: list[bytes] = []
        while True:
            msg = ws.receive()
            if msg.get("bytes"):
                audio_chunks.append(msg["bytes"])
            elif msg.get("text"):
                parsed = json.loads(msg["text"])
                events.append(parsed)
                if parsed["type"] in ("reply.finished", "error"):
                    break

        event_types = [e["type"] for e in events]
        assert "action_result" in event_types
        assert "gesture" in event_types
        assert "reply.delta" in event_types
        assert "tts.start" in event_types
        assert "tts.end" in event_types
        assert event_types[-1] == "reply.finished"
        assert len(audio_chunks) >= 1

        act_res = next(e for e in events if e["type"] == "action_result")
        assert act_res["status"] == "ok"
        assert act_res["target"] == "Mac"
        assert act_res["gesture"] == "bounce_play"

        ws.send_json({"type": "bye"})


def test_voice_websocket_r2_approval_and_board_hid_flow() -> None:
    llm = _ScriptedLLM(
        [
            [
                ToolCallRequested(
                    ToolCall(
                        id="ws-board-1",
                        name="board__media_key",
                        arguments={"key": "play_pause"},
                    )
                ),
                _done(8),
            ],
            [
                TextDelta("Mình đã bấm phát nhạc trên máy đang ghép rồi nhé."),
                _done(10),
            ],
        ]
    )
    client, issuer, _, _, _, _ = _build_voice_harness(llm, transcript="Bấm phát nhạc giúp mình")
    token = issuer.token("owner-1")

    with client.websocket_connect("/v1/voice") as ws:
        ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": token},
                "petId": PET_HEX,
                "speak": False,
                "clientCapabilities": ["board_hid_v1"],
                "boardHid": {"available": True},
            }
        )
        assert ws.receive_json()["type"] == "hello.ok"

        ws.send_json(
            {
                "type": "turn.commit",
                "text": "Bấm phát nhạc giúp mình",
                "context": {
                    "clientTime": datetime.now(UTC).isoformat(),
                },
            }
        )

        events: list[dict[str, Any]] = []
        while True:
            msg = ws.receive()
            if msg.get("text"):
                parsed = json.loads(msg["text"])
                events.append(parsed)
                if parsed["type"] == "action_pending":
                    # Owner approves over WebSocket (or REST)
                    ws.send_json(
                        {
                            "type": "action.decide",
                            "approvalId": parsed["approvalId"],
                            "decision": "approve",
                        }
                    )
                elif parsed["type"] == "board_hid_request":
                    assert parsed["key"] == "play_pause"
                    assert parsed["usageIndex"] == 1
                    ws.send_json(
                        {
                            "type": "action.result",
                            "invocationId": parsed["invocationId"],
                            "status": "ok",
                        }
                    )
                elif parsed["type"] in ("reply.finished", "error"):
                    break

        event_types = [e["type"] for e in events]
        assert event_types[:3] == [
            "action_pending",
            "board_hid_request",
            "action_result",
        ]
        assert "reply.delta" in event_types
        assert event_types[-1] == "reply.finished"
        assert events[2]["status"] == "ok"
        assert events[2]["gesture"] == "bounce_play"

        ws.send_json({"type": "bye"})
