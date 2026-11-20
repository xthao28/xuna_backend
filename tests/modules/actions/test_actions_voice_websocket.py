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
from xuna_backend.modules.actions.application.computer_use import InMemoryComputerUseSessionStore
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
    # No database here: the durable session store gets an in-memory stand-in.
    app.state.cu_session_store = InMemoryComputerUseSessionStore()
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
                    assert parsed["risk"] == parsed["risk"].lower()
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


def test_voice_session_stream_turn_reply_turn_note_and_ws_cu_lifecycle() -> None:
    """Covers VoiceSession turn_note, WS follow-up note, barge-in cleanup, reply_failed, and bye."""
    from datetime import timedelta

    from xuna_backend.modules.actions.application.computer_use import ComputerUseSessionState
    from xuna_backend.modules.cognition.domain.types import Role, ToolSpec
    from xuna_backend.modules.conversation.application.chat_turn import TurnContext
    from xuna_backend.modules.conversation.application.voice_session import VoiceSession
    from xuna_backend.modules.identity.domain.pet_id import PetId

    # 1. Direct VoiceSession.stream_turn_reply accepts str, Callable, and None turn_note
    async def run_direct() -> None:
        llm_direct = _ScriptedLLM([[TextDelta("Ok"), _done(5)]] * 3)
        vs = VoiceSession(
            session_id="vs-1",
            pet_id=PetId(PET_HEX),
            owner_id=uuid.uuid4(),
            recognizer=None,
            generator=llm_direct,
            synthesizer=None,
            speak=False,
        )
        tool = ToolSpec(
            name="ui__inspect",
            description="Inspect",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )
        ctx = TurnContext(local_time=NOW)
        _ = [
            ev
            async for ev in vs.stream_turn_reply(
                "Tiếp tục", ctx, tools=(tool,), turn_note="\n- NOTE_STR"
            )
        ]
        assert llm_direct.requests[-1].messages[0].content.endswith("\n- NOTE_STR")

        _ = [
            ev
            async for ev in vs.stream_turn_reply(
                "Please continue doing this for me",
                ctx,
                tools=(tool,),
                turn_note=lambda loc: f"\n- NOTE_{loc.value}",
            )
        ]
        assert llm_direct.requests[-1].messages[0].content.endswith("\n- NOTE_en")

    asyncio.run(run_direct())

    def _seed_ws_cu(
        client: TestClient, host_repo: _FakeHostRepo, owner_id: uuid.UUID, sid: str
    ) -> Host:
        host = Host(
            host_id=HostId.generate(),
            owner_id=owner_id,
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
                [Grant(host.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW)],
            )
        )
        hub = client.app.state.host_session_hub  # type: ignore[union-attr]

        async def noop(*_: object) -> None:
            return None

        asyncio.run(hub.open_session(str(host.host_id), object(), noop, noop, noop))
        now_utc = datetime.now(UTC)
        asyncio.run(
            client.app.state.cu_session_store.start(  # type: ignore[union-attr]
                ComputerUseSessionState(
                    session_id=sid,
                    owner_id=owner_id,
                    host_id=host.host_id,
                    goal="Work",
                    allowed_apps=("notes",),
                    allowed_domains=(),
                    max_steps=20,
                    max_minutes=10,
                    started_at_utc=now_utc - timedelta(seconds=1),
                    last_heartbeat_at=now_utc,
                    target_alias="Mac",
                    last_window={"appId": "notes", "windowTitle": "Voice Note"},
                ),
                now_utc,
            )
        )
        return host

    # 2. WS follow-up injects prompt note + abort mid-turn emits cu_session(ended)
    turn1_started = asyncio.Event()

    async def ws_stream(_req: Any, idx: int) -> AsyncIterator[Any]:
        if idx == 0:
            yield TextDelta("Sẵn sàng.")
            yield _done(5)
        else:
            yield TextDelta("Đang làm...")
            turn1_started.set()
            await asyncio.sleep(5.0)
            yield _done(5)

    llm_ws = _ScriptedLLM(ws_stream)
    c_ws, i_ws, o_ws, hr_ws, _, _ = _build_voice_harness(llm_ws)
    _seed_ws_cu(c_ws, hr_ws, o_ws.owner_id, "cu-ws-barge")
    token = i_ws.token("owner-1")

    with c_ws.websocket_connect("/v1/voice") as ws:
        ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": token},
                "petId": PET_HEX,
                "speak": False,
                "clientCapabilities": ["host_actions_v1"],
            }
        )
        assert ws.receive_json()["type"] == "hello.ok"

        # Turn 1 completes normally -> follow-up prompt note injected & CU session kept alive
        ws.send_json({"type": "turn.commit", "text": "Tiếp tục ghi chú"})
        while True:
            ev = ws.receive_json()
            if ev["type"] == "reply.finished":
                break
        assert llm_ws.requests[0].messages[0].role == Role.SYSTEM
        assert "cu-ws-barge" in llm_ws.requests[0].messages[0].content
        assert (
            asyncio.run(c_ws.app.state.cu_session_store.get_active_for_owner(o_ws.owner_id))  # type: ignore[union-attr]
            is not None
        )

        # Turn 2 aborted mid-flight -> emits cu_session(ended) and ends store session
        ws.send_json({"type": "turn.commit", "text": "Làm tiếp đi"})
        first_delta = ws.receive_json()
        assert first_delta["type"] == "reply.delta"
        ws.send_json({"type": "abort", "reason": "cancel"})
        ended_msg = ws.receive_json()
        assert ended_msg["type"] == "cu_session"
        assert ended_msg["status"] == "ended"
        assert ended_msg["endReason"] == "turn_cancelled"
        assert (
            asyncio.run(c_ws.app.state.cu_session_store.get_active_for_owner(o_ws.owner_id))  # type: ignore[union-attr]
            is None
        )
        ws.send_json({"type": "bye"})

    # 3. WS reply_failed emits cu_session(ended) BEFORE error(reply_failed)
    async def crash_stream(_req: Any, _idx: int) -> AsyncIterator[Any]:
        if False:
            yield _done()
        raise RuntimeError("boom")

    c_err, i_err, o_err, hr_err, _, _ = _build_voice_harness(_ScriptedLLM(crash_stream))
    _seed_ws_cu(c_err, hr_err, o_err.owner_id, "cu-ws-crash")
    with c_err.websocket_connect("/v1/voice") as ws:
        ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": i_err.token("owner-1")},
                "petId": PET_HEX,
                "speak": False,
                "clientCapabilities": ["host_actions_v1"],
            }
        )
        assert ws.receive_json()["type"] == "hello.ok"
        ws.send_json({"type": "turn.commit", "text": "Chạy"})
        ev1 = ws.receive_json()
        ev2 = ws.receive_json()
        assert ev1["type"] == "cu_session" and ev1["status"] == "ended"
        assert ev1["endReason"] == "turn_failed"
        assert ev2 == {"type": "error", "code": "reply_failed"}
        ws.send_json({"type": "bye"})

    # 4. WS bye mid-turn cleans up CU session without crashing
    async def hang_stream(_req: Any, _idx: int) -> AsyncIterator[Any]:
        yield TextDelta("Đợi")
        await asyncio.sleep(5.0)
        yield _done()

    c_bye, i_bye, o_bye, hr_bye, _, _ = _build_voice_harness(_ScriptedLLM(hang_stream))
    _seed_ws_cu(c_bye, hr_bye, o_bye.owner_id, "cu-ws-bye")
    with c_bye.websocket_connect("/v1/voice") as ws:
        ws.send_json(
            {
                "type": "hello",
                "schemaVersion": 1,
                "auth": {"token": i_bye.token("owner-1")},
                "petId": PET_HEX,
                "speak": False,
                "clientCapabilities": ["host_actions_v1"],
            }
        )
        assert ws.receive_json()["type"] == "hello.ok"
        ws.send_json({"type": "turn.commit", "text": "Chạy"})
        assert ws.receive_json()["type"] == "reply.delta"
        ws.send_json({"type": "bye"})
    assert (
        asyncio.run(c_bye.app.state.cu_session_store.get_active_for_owner(o_bye.owner_id))  # type: ignore[union-attr]
        is None
    )
