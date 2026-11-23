"""Tests for Consent U5 redaction, yieldMs detach, REST watch routes, and WS non-intrusive push."""

import asyncio
import base64
import gc
import json
import statistics
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from starlette.websockets import WebSocket
from tests.helpers import ISSUER, LocalIssuer, MemoryAuditLog, SimulatedDevice, b64, idem
from tests.modules.actions.test_actions_api_and_turn import (
    PET_HEX,
    _FakeConsentRepo,
    _FakeIdentityRepo,
    _FakeKnowledgeCandidatePort,
    _FakeKnowledgeRepo,
    _FakeMemoryRepo,
)
from tests.modules.hosts.test_hosts_use_cases import (
    FakeEnrollmentRepository,
    FakeHostRepository,
)

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.application.compactor import SubGoalCompactor
from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    InMemoryComputerUseSessionStore,
)
from xuna_backend.modules.actions.application.job_runner import (
    AutonomousJobRunner,
    OrchestratorJobStepExecutor,
)
from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.application.planner_executor import JobPlannerExecutor
from xuna_backend.modules.actions.application.wake_inbox import WakeInboxService
from xuna_backend.modules.actions.application.watch_coordinator import WatchCoordinator
from xuna_backend.modules.actions.application.yield_detach import (
    YieldDetachRequest,
    YieldDetachService,
)
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    SubGoal,
    SubGoalStatus,
)
from xuna_backend.modules.actions.domain.redaction import (
    redact_data_images_linear,
    redact_sensitive_text,
    sanitize_wake_text,
)
from xuna_backend.modules.actions.domain.watch import (
    JobWatch,
    TurnDetachedEvent,
    WakeInboxKind,
    WakeSuggestedAction,
    WatchMode,
    WatchStatus,
    WatchTarget,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.job_store import InMemoryAutonomousJobStore
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.actions.infrastructure.watch_repository import (
    InMemoryJobWatchRepository,
    InMemoryWakeInboxRepository,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Locale,
    Message,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
)
from xuna_backend.modules.conversation.application.chat_turn import TurnEvent, chat_turn
from xuna_backend.modules.hosts.application.xtp_frames import WatchEventParamsModel
from xuna_backend.modules.hosts.domain.crypto import HostProofPurpose, host_proof_message
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
    ToolPermissionState,
)
from xuna_backend.modules.hosts.domain.twin import HostPresence, HostTwin


class _DummyDbSession:
    async def commit(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def flush(self) -> None:
        pass


class _DummyDatabase:
    @asynccontextmanager
    async def sessions(self) -> AsyncIterator[_DummyDbSession]:
        yield _DummyDbSession()

    async def dispose(self) -> None:
        pass


class _FakeOwnerRepo:
    def __init__(self, owners: list[Owner]) -> None:
        self._owners = {o.subject: o for o in owners}

    async def get_or_create(self, claims: Any) -> Owner:
        sub = getattr(claims, "subject", None) or str(claims)
        if sub not in self._owners:
            self._owners[sub] = Owner(
                owner_id=uuid.uuid4(),
                issuer=getattr(claims, "issuer", "https://auth.test.xuna"),
                subject=sub,
            )
        return self._owners[sub]

    async def get_by_subject(self, issuer: str, subject: str) -> Owner | None:
        return self._owners.get(subject)

    async def get(self, owner_id: uuid.UUID) -> Owner | None:
        for o in self._owners.values():
            if o.owner_id == owner_id:
                return o
        return None


class _FakeHub:
    def __init__(self, host_id: HostId, owner_id: uuid.UUID) -> None:
        self.host_id = host_id
        self.owner_id = owner_id
        self.online = True
        self.twin = HostTwin(host_id=str(host_id), owner_id=owner_id)
        self.sent: list[tuple[str, str, dict[str, object]]] = []

    def is_online(self, host_id: str) -> bool:
        return self.online and host_id == str(self.host_id)

    def get_twin(self, host_id: str) -> HostTwin | None:
        return self.twin if host_id == str(self.host_id) and self.online else None

    def get_twins_for_owner(self, owner_id: uuid.UUID) -> list[HostTwin]:
        return [self.twin] if self.online and owner_id == self.owner_id else []

    def get_focus_state(self, owner_id: uuid.UUID) -> Any:
        return None

    async def send_request(
        self, host_id: str, method: str, params: dict[str, object], timeout_s: float
    ) -> dict[str, object]:
        del timeout_s
        if not self.is_online(host_id):
            raise RuntimeError("host_offline")
        self.sent.append((host_id, method, params))
        if method == "xuna/watchRegister":
            signals = params.get("signals")
            active = (
                list(signals) if isinstance(signals, list) else [str(params.get("mode", "timer"))]
            )
            return {
                "schemaVersion": 1,
                "accepted": True,
                "watchId": str(params["watchId"]),
                "activeSignals": active,
                "eventDriven": True,
                "fallbackUsed": False,
            }
        if method == "xuna/watchCancel":
            return {"schemaVersion": 1, "cancelledCount": 1}
        return {}


def test_consent_u5_sanitize_wake_text_strips_images_base64_paths_and_secrets() -> None:
    raw_blob = "A" * 160
    raw = (
        "Summary: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA "
        f"blob={raw_blob} "
        "token=sk-1234567890abcdef123456 "
        "path=/Users/alice/secret/report.txt "
        "win=C:\\Users\\alice\\secret.txt\x00"
    )
    cleaned = sanitize_wake_text(raw, max_chars=2000)
    assert "data:image/" not in cleaned
    assert raw_blob not in cleaned
    assert "sk-1234567890" not in cleaned
    assert "/Users/alice" not in cleaned
    assert "C:\\Users\\alice" not in cleaned
    assert "\x00" not in cleaned
    assert "[IMAGE_REDACTED]" in cleaned
    assert "[BASE64_REDACTED]" in cleaned
    assert "[REDACTED]" in cleaned
    assert "[PATH]" in cleaned


def test_chat_turn_yield_ms_detaches_to_waiting_watch_and_preserves_cu() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        hub.twin = replace(
            hub.twin,
            presence=HostPresence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                updated_at=now,
                foreground_app_id="editor",
            ),
        )
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        cu_store = InMemoryComputerUseSessionStore()
        cu = ComputerUseSessionState(
            session_id="cu_detach",
            owner_id=owner_id,
            host_id=host_id,
            goal="Slow build",
            allowed_apps=("editor",),
            allowed_domains=(),
            max_steps=20,
            max_minutes=15,
            started_at_utc=now,
            last_heartbeat_at=now,
        )
        await cu_store.start(cu, now)
        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            cu_session_store=cu_store,
        )
        yield_svc = YieldDetachService(
            job_store=job_store,
            watch_coordinator=coord,
            hub=hub,  # type: ignore[arg-type]
            cu_session_store=cu_store,
        )

        class _SingleRoundToolGen:
            async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
                del request
                yield ToolCallRequested(
                    call=ToolCall(id="call_slow", name="ui__click", arguments={"refId": "b1"})
                )
                yield ChatFinished(
                    provider="fake", model="m", latency_ms=5, input_tokens=5, output_tokens=5
                )

        async def _slow_runner(call: ToolCall, emit: Any) -> Message:
            del emit
            await asyncio.sleep(5.0)
            return Message(role=Role.TOOL, content='{"status":"ok"}', tool_call_id=call.id)

        async def _yield_cb(
            calls: tuple[ToolCall, ...], elapsed_ms: int, last_summary: str | None
        ) -> Any:
            return await yield_svc.try_detach(
                YieldDetachRequest(
                    owner_id=owner_id,
                    pet_id="pet_1",
                    host_id=host_id,
                    goal="Run long build",
                    tool_calls=calls,
                    elapsed_ms=elapsed_ms,
                    locale=Locale.EN,
                    last_action_summary=last_summary,
                    request_id="req_1",
                )
            )

        events: list[TurnEvent] = []
        req = ChatRequest(
            messages=(Message(role=Role.USER, content="Run long build"),),
            tools=(
                ToolSpec(
                    name="ui__click",
                    description="Click",
                    parameters={"type": "object", "properties": {}, "additionalProperties": False},
                ),
            ),
            locale=Locale.EN,
        )
        async for ev in chat_turn(
            _SingleRoundToolGen(),
            req,
            locale_vi=False,
            tool_runner=_slow_runner,
            yield_detach=_yield_cb,
            yield_ms=50,
        ):
            events.append(ev)

        detached = [e for e in events if isinstance(e, TurnDetachedEvent)]
        assert len(detached) == 1
        assert detached[0].state == AutonomousJobState.WAITING_WATCH.value
        assert any(isinstance(e, ChatFinished) for e in events)
        active_job = await job_store.get_active_for_owner(owner_id)
        assert active_job is not None
        assert active_job.state == AutonomousJobState.WAITING_WATCH
        assert active_job.session_id == "cu_detach"
        active_cu = await cu_store.get("cu_detach")
        assert active_cu is not None and active_cu.active is True

    asyncio.run(_run())


def test_watch_rest_routes_owner_isolation_idempotency_and_extend_wake() -> None:
    issuer = LocalIssuer()
    owner1 = Owner(owner_id=uuid.uuid4(), issuer=ISSUER, subject="owner-1")
    owner2 = Owner(owner_id=uuid.uuid4(), issuer=ISSUER, subject="owner-2")
    host_id = HostId.generate()
    hub = _FakeHub(host_id, owner1.owner_id)
    job_store = InMemoryAutonomousJobStore()
    watch_repo = InMemoryJobWatchRepository()
    inbox_repo = InMemoryWakeInboxRepository()
    wake_inbox = WakeInboxService(repo=inbox_repo)

    now = datetime.now(UTC)
    paused_job = AutonomousJob(
        job_id=uuid.uuid4(),
        owner_id=owner1.owner_id,
        pet_id="pet_1",
        host_id=host_id,
        session_id=None,
        title="Build task",
        goal="Run tests",
        state=AutonomousJobState.PAUSED,
        stop_reason="watch_timeout",
        created_at=now,
        updated_at=now,
    )
    watch = JobWatch(
        watch_id=uuid.uuid4(),
        job_id=paused_job.job_id,
        owner_id=owner1.owner_id,
        host_id=host_id,
        mode=WatchMode.TIMER,
        signals=(),
        target=WatchTarget(),
        done_criteria="Wait for build",
        status=WatchStatus.EXPIRED,
        expires_at=now - timedelta(seconds=5),
        created_at=now - timedelta(seconds=60),
        updated_at=now,
    )

    async def _seed() -> Any:
        await job_store.save(paused_job)
        await watch_repo.save(watch)
        return await wake_inbox.emit_item(
            owner_id=owner1.owner_id,
            pet_id="pet_1",
            kind=WakeInboxKind.JOB_TIMEOUT,
            title="Timed out",
            summary="Still running",
            job_id=paused_job.job_id,
            host_id=host_id,
            suggested_actions=(WakeSuggestedAction.EXTEND_WATCH, WakeSuggestedAction.DISMISS),
        )

    item = asyncio.run(_seed())

    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
        ),
        verifier=issuer.verifier(),
    )
    app.state.database = _DummyDatabase()
    app.state.owner_repository = lambda _s: _FakeOwnerRepo([owner1, owner2])
    app.state.audit_log = lambda _s: MemoryAuditLog()
    app.state.host_session_hub = hub
    app.state.job_store = job_store
    app.state.watch_repository = watch_repo
    app.state.wake_inbox_repository = inbox_repo
    app.state.wake_inbox_service = wake_inbox
    coord = WatchCoordinator(
        watch_repo=watch_repo,
        job_store=job_store,
        hub=hub,  # type: ignore[arg-type]
        wake_inbox=wake_inbox,
    )
    app.state.watch_coordinator = coord

    h1 = {"Authorization": f"Bearer {issuer.token('owner-1')}"}
    h2 = {"Authorization": f"Bearer {issuer.token('owner-2')}"}

    with TestClient(app) as client:
        app.state.database = _DummyDatabase()
        # 1. GET wake-inbox
        r_list = client.get("/v1/actions/wake-inbox", headers=h1)
        assert r_list.status_code == 200
        body = r_list.json()
        assert body["unreadCount"] == 1
        assert body["items"][0]["itemId"] == str(item.item_id)

        # Foreign owner cannot see or ack owner1's item -> 404
        assert client.get("/v1/actions/wake-inbox", headers=h2).json()["items"] == []
        r_foreign_ack = client.post(
            f"/v1/actions/wake-inbox/{item.item_id}/ack",
            headers={**h2, **idem()},
            json={"schemaVersion": 1, "dismiss": False},
        )
        assert r_foreign_ack.status_code == 404

        # 2. Idempotency-Key required and enforced on ack
        r_no_idem = client.post(f"/v1/actions/wake-inbox/{item.item_id}/ack", headers=h1)
        assert r_no_idem.status_code == 400
        assert r_no_idem.json()["code"] == "idempotency_key_required"

        ack_headers = {**h1, **idem()}
        r_ack1 = client.post(
            f"/v1/actions/wake-inbox/{item.item_id}/ack",
            headers=ack_headers,
            json={"schemaVersion": 1, "dismiss": False},
        )
        assert r_ack1.status_code == 200
        assert r_ack1.json()["readAt"] is not None
        r_ack_replay = client.post(
            f"/v1/actions/wake-inbox/{item.item_id}/ack",
            headers=ack_headers,
            json={"schemaVersion": 1, "dismiss": False},
        )
        assert r_ack_replay.status_code == 200
        r_ack_mismatch = client.post(
            f"/v1/actions/wake-inbox/{item.item_id}/ack",
            headers=ack_headers,
            json={"schemaVersion": 1, "dismiss": True},
        )
        assert r_ack_mismatch.status_code == 409
        assert r_ack_mismatch.json()["code"] == "idempotency_key_reused"

        # 3. GET /v1/actions/jobs/{job_id}/watch
        r_watch = client.get(f"/v1/actions/jobs/{paused_job.job_id}/watch", headers=h1)
        assert r_watch.status_code == 200
        assert r_watch.json()["mode"] == "timer"
        assert (
            client.get(f"/v1/actions/jobs/{paused_job.job_id}/watch", headers=h2).status_code == 404
        )

        # 4. POST extend-watch (> 900 rejected 422, valid resumes to WAITING_WATCH)
        r_bad_ext = client.post(
            f"/v1/actions/jobs/{paused_job.job_id}/extend-watch",
            headers={**h1, **idem()},
            json={"schemaVersion": 1, "additionalSeconds": 901},
        )
        assert r_bad_ext.status_code == 422
        r_ext = client.post(
            f"/v1/actions/jobs/{paused_job.job_id}/extend-watch",
            headers={**h1, **idem()},
            json={"schemaVersion": 1, "additionalSeconds": 300},
        )
        assert r_ext.status_code == 200
        assert r_ext.json()["status"] == "active"

        # 5. POST wake-now
        r_wake = client.post(
            f"/v1/actions/jobs/{paused_job.job_id}/wake-now",
            headers={**h1, **idem()},
            json={"schemaVersion": 1},
        )
        assert r_wake.status_code == 200
        assert r_wake.json()["state"] == "running"


def test_voice_websocket_non_intrusive_wake_inbox_push() -> None:
    issuer = LocalIssuer()
    owner = Owner(owner_id=uuid.uuid4(), issuer=ISSUER, subject="ws-owner")
    inbox_repo = InMemoryWakeInboxRepository()
    wake_inbox = WakeInboxService(repo=inbox_repo)
    turn_gate = asyncio.Event()

    class _SlowTextGenerator:
        async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
            del request
            yield TextDelta(text="Speaking first part. ")
            await asyncio.sleep(0.15)
            yield TextDelta(text="Done speaking.")
            yield ChatFinished(
                provider="fake", model="m", latency_ms=10, input_tokens=5, output_tokens=5
            )
            turn_gate.set()

    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
        ),
        verifier=issuer.verifier(),
        generator=_SlowTextGenerator(),
    )
    app.state.database = _DummyDatabase()
    app.state.owner_repository = lambda _s: _FakeOwnerRepo([owner])
    app.state.identity_repository = lambda _s: _FakeIdentityRepo(owner.owner_id, PET_HEX)
    app.state.consent_repository = lambda _s: _FakeConsentRepo()
    app.state.memory_repository = lambda _s: _FakeMemoryRepo()
    app.state.knowledge_repository = lambda _s: _FakeKnowledgeRepo()
    app.state.knowledge_candidate_adapter = lambda _s: _FakeKnowledgeCandidatePort()
    app.state.audit_log = lambda _s: MemoryAuditLog()
    app.state.wake_inbox_repository = inbox_repo
    app.state.wake_inbox_service = wake_inbox

    with TestClient(app) as client:
        app.state.database = _DummyDatabase()
        with client.websocket_connect("/v1/voice") as ws:
            ws.send_json(
                {
                    "type": "hello",
                    "schemaVersion": 1,
                    "auth": {"token": issuer.token("ws-owner")},
                    "petId": PET_HEX,
                    "speak": False,
                    "locale": "en-US",
                }
            )
            hello_ok = ws.receive_json()
            assert hello_ok["type"] == "hello.ok"

            ws.send_json(
                {
                    "type": "turn.commit",
                    "text": "Tell me a short note",
                    "context": {"gestures": False},
                }
            )
            first_delta = ws.receive_json()
            assert first_delta["type"] == "reply.delta"

            # Emit wake inbox item while turn is actively replying
            asyncio.run(
                wake_inbox.emit_item(
                    owner_id=owner.owner_id,
                    pet_id="pet_1",
                    kind=WakeInboxKind.JOB_REPORT,
                    title="Background done",
                    summary="Build finished cleanly.",
                )
            )

            seen_types: list[str] = [first_delta["type"]]
            for _ in range(10):
                msg = ws.receive_json()
                seen_types.append(msg["type"])
                if msg["type"] == "wake_inbox_item":
                    break

            assert "reply.finished" in seen_types
            assert "wake_inbox_item" in seen_types
            assert seen_types.index("reply.finished") < seen_types.index("wake_inbox_item")
            ws.send_json({"type": "bye"})


def test_host_socket_dispatches_watch_event_asynchronously_without_breaking_handshake() -> None:
    agent_key = SimulatedDevice()
    owner_id = uuid.uuid4()
    host_id = HostId.generate()
    now = datetime.now(UTC)
    host = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("WatchHost"),
        platform=HostPlatform.MACOS,
        hostname="watchhost",
        os_version="14.0",
        agent_version="0.2.0",
        public_key=agent_key.public_key,
        status=HostStatus.ACTIVE,
        created_at=now,
    )
    host_repo = FakeHostRepository()
    host_repo.hosts[host_id.value] = host

    received_events: list[tuple[uuid.UUID, HostId, Any]] = []
    event_done = asyncio.Event()

    async def _on_watch_event(oid: uuid.UUID, hid: HostId, params: Any) -> None:
        received_events.append((oid, hid, params))
        event_done.set()

    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
        )
    )
    app.state.database = _DummyDatabase()
    app.state.host_repository = lambda _s: host_repo
    app.state.host_enrollment_repository = lambda _s: FakeEnrollmentRepository()
    app.state.audit_log = lambda _s: MemoryAuditLog()
    app.state.on_host_watch_event = _on_watch_event

    watch_id = uuid.uuid4()
    job_id = uuid.uuid4()
    with TestClient(app) as client:
        app.state.database = _DummyDatabase()
        with client.websocket_connect("/v1/hosts/connect", subprotocols=["xtp/1"]) as ws:
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "xuna/hello",
                        "params": {
                            "protocols": ["xtp/1"],
                            "hostId": str(host_id),
                            "platform": "macos",
                            "osVersion": "14.0",
                            "agentVersion": "0.2.0",
                        },
                    }
                )
            )
            assert json.loads(ws.receive_text())["result"] == {"protocol": "xtp/1"}
            challenge = json.loads(ws.receive_text())
            nonce = base64.b64decode(challenge["params"]["nonce"])
            proof = host_proof_message(
                nonce=nonce,
                purpose=HostProofPurpose.SESSION,
                payload=agent_key.public_key,
                host_id=host_id,
            )
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "xuna/proof",
                        "params": {"purpose": 2, "signature": b64(agent_key.sign(proof))},
                    }
                )
            )
            assert json.loads(ws.receive_text())["result"] == {"verified": True}
            assert json.loads(ws.receive_text())["method"] == "xuna/ready"
            tools_req = json.loads(ws.receive_text())
            assert tools_req["method"] == "tools/list"
            ws.send_text(
                json.dumps({"jsonrpc": "2.0", "id": tools_req["id"], "result": {"tools": []}})
            )

            # Send xuna/watchEvent notification and follow with ping
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "xuna/watchEvent",
                        "params": {
                            "schemaVersion": 1,
                            "watchId": str(watch_id),
                            "jobId": str(job_id),
                            "kind": "settled",
                            "observedSignals": ["ui_quiescence"],
                            "elapsedSeconds": 15,
                            "changeCount": 3,
                            "digestChanged": True,
                            "occurredAt": now.isoformat(),
                        },
                    }
                )
            )
            ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping", "params": {}}))
            assert json.loads(ws.receive_text()) == {"jsonrpc": "2.0", "id": 99, "result": {}}

    assert len(received_events) == 1
    assert received_events[0][0] == owner_id
    assert received_events[0][1] == host_id
    assert received_events[0][2].watchId == str(watch_id)
    assert received_events[0][2].kind == "settled"


def test_consent_u5_sanitize_wake_text_rejects_concatenated_bearer_mime_svg_b64url_and_quoted_space_paths() -> (  # noqa: E501
    None
):
    concat_bearer = "x" * 130 + "Bearer secret123456"
    cleaned_concat = sanitize_wake_text(concat_bearer)
    assert "secret123456" not in cleaned_concat
    assert "[REDACTED]" in cleaned_concat

    long_prefix_bearer = "a" * 1992 + "Bearer supersecrettokenvalue1234"
    cleaned_long = sanitize_wake_text(long_prefix_bearer, max_chars=2000)
    assert "supersecrettokenvalue1234" not in cleaned_long
    assert "[REDACTED]" in cleaned_long

    svg_b64 = (
        "Icon data:image/svg+xml;charset=utf-8;base64,"
        "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg== ok"
    )
    cleaned_svg_b64 = sanitize_wake_text(svg_b64)
    assert "data:image/" not in cleaned_svg_b64
    assert "PHN2Zy" not in cleaned_svg_b64
    assert cleaned_svg_b64 == "Icon [IMAGE_REDACTED] ok"

    svg_raw = "Icon data:image/svg+xml,%3Csvg%3E ok"
    cleaned_svg_raw = sanitize_wake_text(svg_raw)
    assert "data:image/" not in cleaned_svg_raw
    assert "%3Csvg%3E" not in cleaned_svg_raw
    assert cleaned_svg_raw == "Icon [IMAGE_REDACTED] ok"

    b64url_blob = ("AbCdEf0123_-" * 22)[:256]
    cleaned_b64url = sanitize_wake_text(f"Payload: {b64url_blob} done")
    assert b64url_blob not in cleaned_b64url
    assert cleaned_b64url == "Payload: [BASE64_REDACTED] done"

    paths_text = (
        'Opened "/Users/alice/my folder/secret file.txt" and '
        "read '/var/tmp/sub dir/secret.txt' then "
        'saved "C:\\Users\\alice\\My Docs\\secret file.txt" and '
        "wrote 'D:\\Temp Work\\secret.txt'"
    )
    cleaned_paths = sanitize_wake_text(paths_text)
    assert "secret file.txt" not in cleaned_paths
    assert "secret.txt" not in cleaned_paths
    assert 'Opened "[PATH]"' in cleaned_paths
    assert "read '[PATH]'" in cleaned_paths
    assert 'saved "[PATH]"' in cleaned_paths
    assert "wrote '[PATH]'" in cleaned_paths


class _DetachRunnerHub(_FakeHub):
    async def send_request(
        self, host_id: str, method: str, params: dict[str, object], timeout_s: float
    ) -> dict[str, object]:
        if method == "xuna/get_policy":
            if not self.is_online(host_id):
                raise RuntimeError("host_offline")
            self.sent.append((host_id, method, params))
            return {"trustPreset": "autopilot"}
        if method in {"xuna/watchRegister", "xuna/watchCancel"}:
            return await super().send_request(host_id, method, params, timeout_s)
        if not self.is_online(host_id):
            raise RuntimeError("host_offline")
        self.sent.append((host_id, method, params))
        return {
            "structuredContent": {"status": "ok", "observed": True},
            "_meta": {"xuna/status": "ok"},
        }

    async def notify(self, host_id: str, method: str, params: dict[str, object]) -> None:
        self.sent.append((host_id, method, params))

    async def stop_host(self, host_id: str, reason: str = "owner_app") -> bool:
        self.sent.append((host_id, "xuna/stop", {"reason": reason}))
        return True


class _LocalFakeHostRepo:
    def __init__(self, host: Host, manifest: Manifest, grants: list[Grant]) -> None:
        self._host = host
        self._manifest = manifest
        self._grants = grants

    async def get_host(self, host_id: HostId) -> Host | None:
        return self._host if host_id == self._host.host_id else None

    async def save_host(self, host: Host) -> None:
        self._host = host

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        return [self._host] if self._host.owner_id == owner_id else []

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self._grants) if host_id == self._host.host_id else []

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self._grants = list(grants)

    async def suspend_r2_plus_grants_for_owner(self, owner_id: uuid.UUID, now: datetime) -> int:
        return 0

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self._manifest if host_id == self._host.host_id else None

    async def save_manifest(self, manifest: Manifest) -> None:
        self._manifest = manifest


class _LocalScriptedGenerator:
    def __init__(self, turns: list[tuple[str, list[ToolCall]]]) -> None:
        self._turns = list(turns)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        del request
        if not self._turns:
            text, tool_calls = "All verified.", []
        else:
            text, tool_calls = self._turns.pop(0)
        if text:
            yield TextDelta(text=text)
        for tc in tool_calls:
            yield ToolCallRequested(call=tc)
        yield ChatFinished(
            provider="fake",
            model="fake-1",
            latency_ms=5,
            input_tokens=10,
            output_tokens=5,
        )


def _make_detach_host_tools(
    owner_id: uuid.UUID, host_id: HostId
) -> tuple[Host, Manifest, list[Grant]]:
    now = datetime.now(UTC)
    host = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("Workstation"),
        platform=HostPlatform.MACOS,
        hostname="workstation",
        os_version="14.0",
        agent_version="0.2.0",
        public_key=b"\x04" + bytes(range(1, 65)),
        status=HostStatus.ACTIVE,
        created_at=now,
    )
    inspect_desc = ToolDescriptor(
        name="ui.inspect",
        description="Inspect current window.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=ToolFamily.UI,
        risk=RiskTier.R0,
        observes=True,
        reversible=True,
        permission=ToolPermissionState.GRANTED,
    )
    manifest = Manifest(host_id=host_id, tools=(inspect_desc,), updated_at=now)
    grants = [
        Grant(
            host_id=host_id,
            family=ToolFamily.UI,
            max_risk=RiskTier.R0,
            enabled=True,
            updated_at=now,
        )
    ]
    return host, manifest, grants


def test_yield_ms_detached_job_enforces_rule_h1_post_wake_verification() -> None:
    async def _run_case(*, with_verify_tool: bool) -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        host, manifest, grants = _make_detach_host_tools(owner_id, host_id)
        host_repo = _LocalFakeHostRepo(host, manifest, grants)
        hub = _DetachRunnerHub(host_id, owner_id)
        hub.twin = replace(
            hub.twin,
            presence=HostPresence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                updated_at=now,
                foreground_app_id="editor",
            ),
        )
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        wake_inbox = WakeInboxService(repo=InMemoryWakeInboxRepository())
        cu_store = InMemoryComputerUseSessionStore()
        cu = ComputerUseSessionState(
            session_id="cu_h1",
            owner_id=owner_id,
            host_id=host_id,
            goal="Verify after yield",
            allowed_apps=("editor",),
            allowed_domains=(),
            max_steps=20,
            max_minutes=15,
            started_at_utc=now,
            last_heartbeat_at=now,
        )
        await cu_store.start(cu, now)

        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            cu_session_store=cu_store,
            wake_inbox=wake_inbox,
        )
        yield_svc = YieldDetachService(
            job_store=job_store,
            watch_coordinator=coord,
            hub=hub,  # type: ignore[arg-type]
            cu_session_store=cu_store,
        )

        class _SingleSlowClickGen:
            async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
                del request
                yield ToolCallRequested(
                    call=ToolCall(id="slow_1", name="ui__click", arguments={"refId": "b1"})
                )
                yield ChatFinished(
                    provider="fake", model="m", latency_ms=5, input_tokens=5, output_tokens=5
                )

        async def _slow_runner(call: ToolCall, emit: Any) -> Message:
            del emit
            await asyncio.sleep(5.0)
            return Message(role=Role.TOOL, content='{"status":"ok"}', tool_call_id=call.id)

        async def _yield_cb(
            calls: tuple[ToolCall, ...], elapsed_ms: int, last_summary: str | None
        ) -> Any:
            return await yield_svc.try_detach(
                YieldDetachRequest(
                    owner_id=owner_id,
                    pet_id="pet_1",
                    host_id=host_id,
                    goal="Run long build",
                    tool_calls=calls,
                    elapsed_ms=elapsed_ms,
                    locale=Locale.EN,
                    last_action_summary=last_summary,
                    request_id="req_h1",
                )
            )

        events: list[TurnEvent] = []
        req = ChatRequest(
            messages=(Message(role=Role.USER, content="Run long build"),),
            tools=(
                ToolSpec(
                    name="ui__click",
                    description="Click",
                    parameters={"type": "object", "properties": {}, "additionalProperties": False},
                ),
            ),
            locale=Locale.EN,
        )
        async for ev in chat_turn(
            _SingleSlowClickGen(),
            req,
            locale_vi=False,
            tool_runner=_slow_runner,
            yield_detach=_yield_cb,
            yield_ms=50,
        ):
            events.append(ev)

        assert any(isinstance(e, TurnDetachedEvent) for e in events)
        detached_job = await job_store.get_active_for_owner(owner_id)
        assert detached_job is not None
        assert detached_job.state == AutonomousJobState.WAITING_WATCH
        assert detached_job.sub_goals[0].tool_call_count == 0

        post_wake_turns = (
            [
                ("", [ToolCall(id="verify_1", name="ui__inspect", arguments={})]),
                ("Verified UI settled.", []),
            ]
            if with_verify_tool
            else [("I think it is done!", [])]
        )
        post_wake_gen = _LocalScriptedGenerator(post_wake_turns)
        compactor = SubGoalCompactor()
        planner = JobPlannerExecutor(compactor=compactor)

        def _make_orch() -> Orchestrator:
            return Orchestrator(
                host_repo=host_repo,  # type: ignore[arg-type]
                hub=hub,  # type: ignore[arg-type]
                approval_repo=InMemoryApprovalRepository(),
                approval_broker=InMemoryApprovalBroker(),
                local_result_broker=InMemoryLocalResultBroker(),
                last_target_store=InMemoryLastTargetStore(),
                audit=MemoryAuditLog(),
                scope_store=InMemoryApprovalScopeStore(),
                cu_session_store=cu_store,
            )

        runner = AutonomousJobRunner(
            job_store=job_store,
            cu_session_store=cu_store,
            planner=planner,
            compactor=compactor,
            watch_coordinator=coord,
        )
        coord.set_job_runner(runner)
        runner._step_executor = OrchestratorJobStepExecutor(
            orchestrator_factory=_make_orch,
            host_repo_factory=lambda: host_repo,  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            audit_factory=MemoryAuditLog,
            generator_factory=lambda: post_wake_gen,
            planner=planner,
            compactor=compactor,
            watch_coordinator=coord,
            wake_inbox=wake_inbox,
        )

        w1 = await watch_repo.get_active_for_job(detached_job.job_id)
        assert w1 is not None
        await coord.on_watch_event(
            owner_id,
            host_id,
            WatchEventParamsModel(
                schemaVersion=1,
                watchId=str(w1.watch_id),
                jobId=str(detached_job.job_id),
                kind="settled",
                observedSignals=["ui_quiescence"],
                elapsedSeconds=12,
                changeCount=1,
                digestChanged=True,
                occurredAt=datetime.now(UTC).isoformat(),
            ),
        )

        expected_state = (
            AutonomousJobState.COMPLETED if with_verify_tool else AutonomousJobState.FAILED
        )
        for _ in range(100):
            cur = await job_store.get(detached_job.job_id)
            if cur is not None and cur.state == expected_state:
                break
            await asyncio.sleep(0.01)

        final_job = await job_store.get(detached_job.job_id)
        assert final_job is not None
        inbox_items = await wake_inbox.list_items(owner_id)
        if not with_verify_tool:
            assert final_job.state == AutonomousJobState.FAILED
            assert final_job.stop_reason == "no_tool_executed"
            assert final_job.sub_goals[0].status == SubGoalStatus.FAILED
            assert final_job.sub_goals[0].tool_call_count == 0
            assert any(i.kind == WakeInboxKind.JOB_FAILED for i in inbox_items)
            assert not any(i.kind == WakeInboxKind.JOB_REPORT for i in inbox_items)
        else:
            assert final_job.state == AutonomousJobState.COMPLETED
            assert final_job.sub_goals[0].status == SubGoalStatus.COMPLETED
            assert final_job.sub_goals[0].tool_call_count == 1
            assert any(i.kind == WakeInboxKind.JOB_REPORT for i in inbox_items)
            assert not any(i.kind == WakeInboxKind.JOB_FAILED for i in inbox_items)

        await runner.shutdown()

    asyncio.run(_run_case(with_verify_tool=False))
    asyncio.run(_run_case(with_verify_tool=True))


def test_yield_detach_host_offline_during_handoff_leaves_no_orphaned_running_job() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        now = datetime.now(UTC)
        hub = _FakeHub(host_id, owner_id)
        hub.twin = replace(
            hub.twin,
            presence=HostPresence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                updated_at=now,
                foreground_app_id="editor",
            ),
        )
        job_store = InMemoryAutonomousJobStore()
        watch_repo = InMemoryJobWatchRepository()
        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
        )
        orig_infer = coord.infer_watch_for_detach

        async def _infer_then_drop(**kwargs: Any) -> Any:
            out = await orig_infer(**kwargs)
            hub.online = False
            return out

        coord.infer_watch_for_detach = _infer_then_drop  # type: ignore[method-assign]
        yield_svc = YieldDetachService(
            job_store=job_store,
            watch_coordinator=coord,
            hub=hub,  # type: ignore[arg-type]
        )
        res = await yield_svc.try_detach(
            YieldDetachRequest(
                owner_id=owner_id,
                pet_id="pet_1",
                host_id=host_id,
                goal="Slow build",
                tool_calls=(ToolCall(id="c1", name="ui__click", arguments={}),),
                elapsed_ms=12500,
                locale=Locale.EN,
            )
        )
        assert res.detached is False
        assert res.error_code == "host_offline"
        assert await job_store.get_active_for_owner(owner_id) is None
        assert await job_store.list_for_owner(owner_id) == []

        # Direct register_for_detached_job fails any pre-saved non-terminal job
        pre_job = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=owner_id,
            pet_id="pet_1",
            host_id=host_id,
            session_id=None,
            title="Pre-saved running",
            goal="Goal",
            state=AutonomousJobState.RUNNING,
            created_at=now,
            updated_at=now,
        )
        await job_store.save(pre_job)
        reg = await coord.register_for_detached_job(
            job=pre_job,
            mode=WatchMode.TIMER,
            signals=(),
            target=WatchTarget(),
            done_criteria="Wait",
        )
        assert reg.accepted is False
        assert reg.error_code == "host_offline"
        saved_pre = await job_store.get(pre_job.job_id)
        assert saved_pre is not None
        assert saved_pre.state == AutonomousJobState.FAILED
        assert saved_pre.stop_reason == "host_offline"
        assert await job_store.get_active_for_owner(owner_id) is None

    asyncio.run(_run())


def test_host_socket_multiple_concurrent_watch_events_with_slow_handler() -> None:
    agent_key = SimulatedDevice()
    owner_id = uuid.uuid4()
    host_id = HostId.generate()
    now = datetime.now(UTC)
    host = Host(
        host_id=host_id,
        owner_id=owner_id,
        alias=HostAlias("MultiWatchHost"),
        platform=HostPlatform.MACOS,
        hostname="multiwatchhost",
        os_version="14.0",
        agent_version="0.2.0",
        public_key=agent_key.public_key,
        status=HostStatus.ACTIVE,
        created_at=now,
    )
    host_repo = FakeHostRepository()
    host_repo.hosts[host_id.value] = host

    received_watch_ids: list[str] = []
    watch_ids = [str(uuid.uuid4()) for _ in range(3)]
    all_done = threading.Event()

    async def _slow_on_watch_event(oid: uuid.UUID, hid: HostId, params: Any) -> None:
        del oid, hid
        await asyncio.sleep(0.05)
        received_watch_ids.append(params.watchId)
        if len(received_watch_ids) >= len(watch_ids):
            all_done.set()

    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
        )
    )
    app.state.database = _DummyDatabase()
    app.state.host_repository = lambda _s: host_repo
    app.state.host_enrollment_repository = lambda _s: FakeEnrollmentRepository()
    app.state.audit_log = lambda _s: MemoryAuditLog()
    app.state.on_host_watch_event = _slow_on_watch_event

    job_id = str(uuid.uuid4())
    with TestClient(app) as client:
        app.state.database = _DummyDatabase()
        with client.websocket_connect("/v1/hosts/connect", subprotocols=["xtp/1"]) as ws:
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "xuna/hello",
                        "params": {
                            "protocols": ["xtp/1"],
                            "hostId": str(host_id),
                            "platform": "macos",
                            "osVersion": "14.0",
                            "agentVersion": "0.2.0",
                        },
                    }
                )
            )
            assert json.loads(ws.receive_text())["result"] == {"protocol": "xtp/1"}
            challenge = json.loads(ws.receive_text())
            nonce = base64.b64decode(challenge["params"]["nonce"])
            proof = host_proof_message(
                nonce=nonce,
                purpose=HostProofPurpose.SESSION,
                payload=agent_key.public_key,
                host_id=host_id,
            )
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "xuna/proof",
                        "params": {"purpose": 2, "signature": b64(agent_key.sign(proof))},
                    }
                )
            )
            assert json.loads(ws.receive_text())["result"] == {"verified": True}
            assert json.loads(ws.receive_text())["method"] == "xuna/ready"
            tools_req = json.loads(ws.receive_text())
            ws.send_text(
                json.dumps({"jsonrpc": "2.0", "id": tools_req["id"], "result": {"tools": []}})
            )

            for wid in watch_ids:
                ws.send_text(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": "xuna/watchEvent",
                            "params": {
                                "schemaVersion": 1,
                                "watchId": wid,
                                "jobId": job_id,
                                "kind": "settled",
                                "observedSignals": ["ui_quiescence"],
                                "elapsedSeconds": 10,
                                "changeCount": 1,
                                "digestChanged": True,
                                "occurredAt": now.isoformat(),
                            },
                        }
                    )
                )
                gc.collect()

            ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping", "params": {}}))
            assert json.loads(ws.receive_text()) == {"jsonrpc": "2.0", "id": 99, "result": {}}
            assert all_done.wait(timeout=2.0)

    assert sorted(received_watch_ids) == sorted(watch_ids)


def test_voice_websocket_wake_inbox_fifo_order_when_arriving_mid_flush(
    monkeypatch: Any,
) -> None:
    issuer = LocalIssuer()
    owner = Owner(owner_id=uuid.uuid4(), issuer=ISSUER, subject="ws-fifo-owner")
    inbox_repo = InMemoryWakeInboxRepository()
    wake_inbox = WakeInboxService(repo=inbox_repo)

    class _SlowTextGenerator:
        async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
            del request
            yield TextDelta(text="Part 1 ")
            await asyncio.sleep(0.15)
            yield TextDelta(text="Part 2.")
            yield ChatFinished(
                provider="fake", model="m", latency_ms=10, input_tokens=5, output_tokens=5
            )

    orig_send_json: Callable[..., Any] = WebSocket.send_json
    item3_emitted = False
    bg_tasks: list[asyncio.Task[Any]] = []

    async def _slow_first_wake_send(self: WebSocket, data: Any, mode: str = "text") -> None:
        nonlocal item3_emitted
        if (
            not item3_emitted
            and isinstance(data, dict)
            and data.get("type") == "wake_inbox_item"
            and data.get("title") == "Item 1"
        ):
            item3_emitted = True
            t3 = asyncio.create_task(
                wake_inbox.emit_item(
                    owner_id=owner.owner_id,
                    pet_id="pet_1",
                    kind=WakeInboxKind.JOB_REPORT,
                    title="Item 3",
                    summary="Third item emitted mid-flush.",
                )
            )
            bg_tasks.append(t3)
            await asyncio.sleep(0.03)
        await orig_send_json(self, data, mode=mode)

    monkeypatch.setattr(WebSocket, "send_json", _slow_first_wake_send)

    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+psycopg://xuna:unused@127.0.0.1:1/none",
        ),
        verifier=issuer.verifier(),
        generator=_SlowTextGenerator(),
    )
    app.state.database = _DummyDatabase()
    app.state.owner_repository = lambda _s: _FakeOwnerRepo([owner])
    app.state.identity_repository = lambda _s: _FakeIdentityRepo(owner.owner_id, PET_HEX)
    app.state.consent_repository = lambda _s: _FakeConsentRepo()
    app.state.memory_repository = lambda _s: _FakeMemoryRepo()
    app.state.knowledge_repository = lambda _s: _FakeKnowledgeRepo()
    app.state.knowledge_candidate_adapter = lambda _s: _FakeKnowledgeCandidatePort()
    app.state.audit_log = lambda _s: MemoryAuditLog()
    app.state.wake_inbox_repository = inbox_repo
    app.state.wake_inbox_service = wake_inbox

    with TestClient(app) as client:
        app.state.database = _DummyDatabase()
        with client.websocket_connect("/v1/voice") as ws:
            ws.send_json(
                {
                    "type": "hello",
                    "schemaVersion": 1,
                    "auth": {"token": issuer.token("ws-fifo-owner")},
                    "petId": PET_HEX,
                    "speak": False,
                    "locale": "en-US",
                }
            )
            assert ws.receive_json()["type"] == "hello.ok"

            ws.send_json(
                {
                    "type": "turn.commit",
                    "text": "Short reply",
                    "context": {"gestures": False},
                }
            )
            first_delta = ws.receive_json()
            assert first_delta["type"] == "reply.delta"

            asyncio.run(
                wake_inbox.emit_item(
                    owner_id=owner.owner_id,
                    pet_id="pet_1",
                    kind=WakeInboxKind.JOB_REPORT,
                    title="Item 1",
                    summary="First queued item.",
                )
            )
            asyncio.run(
                wake_inbox.emit_item(
                    owner_id=owner.owner_id,
                    pet_id="pet_1",
                    kind=WakeInboxKind.JOB_REPORT,
                    title="Item 2",
                    summary="Second queued item.",
                )
            )

            wake_titles: list[str] = []
            for _ in range(15):
                msg = ws.receive_json()
                if msg["type"] == "wake_inbox_item":
                    wake_titles.append(msg["title"])
                    if len(wake_titles) == 3:
                        break

            assert wake_titles == ["Item 1", "Item 2", "Item 3"]
            ws.send_json({"type": "bye"})


def test_consent_u5_sanitize_wake_text_unc_utf8_colon_svg_spaces_and_unquoted_win_spaces() -> None:
    unc_text = r"Copied to \\fileserver01\finance\secret_payroll.xlsx cleanly"
    cleaned_unc = sanitize_wake_text(unc_text)
    assert "fileserver01" not in cleaned_unc
    assert "secret_payroll" not in cleaned_unc
    assert cleaned_unc == "Copied to [PATH] cleanly"

    quoted_unc = 'Opened "\\\\fileserver01\\finance\\My Docs\\secret payroll.xlsx" ok'
    cleaned_quoted_unc = sanitize_wake_text(quoted_unc)
    assert "fileserver01" not in cleaned_quoted_unc
    assert cleaned_quoted_unc == 'Opened "[PATH]" ok'

    vi_posix = "Mở tệp /Users/thảo/tài_liệu/bí_mật.txt và /thảo/bí_mật/file.txt thành công"
    cleaned_vi_posix = sanitize_wake_text(vi_posix)
    assert "tài_liệu" not in cleaned_vi_posix
    assert "bí_mật" not in cleaned_vi_posix
    assert cleaned_vi_posix == "Mở tệp [PATH] và [PATH] thành công"

    colon_posix = "Đường dẫn:/Users/alice/secret/key.pem đã xử lý"
    cleaned_colon = sanitize_wake_text(colon_posix)
    assert "/Users/alice" not in cleaned_colon
    assert "key.pem" not in cleaned_colon
    assert cleaned_colon == "Đường dẫn:[PATH] đã xử lý"

    svg_spaces = (
        'Icon data:image/svg+xml;charset=utf-8,<svg xmlns="http://www.w3.org/2000/svg">'
        '<circle r="5"/></svg> done'
    )
    cleaned_svg_spaces = sanitize_wake_text(svg_spaces)
    assert "<svg" not in cleaned_svg_spaces
    assert "xmlns" not in cleaned_svg_spaces
    assert cleaned_svg_spaces == "Icon [IMAGE_REDACTED] done"

    svg_self_close = (
        'Icon data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="10"/> done'
    )
    cleaned_svg_sc = sanitize_wake_text(svg_self_close)
    assert "<svg" not in cleaned_svg_sc
    assert cleaned_svg_sc == "Icon [IMAGE_REDACTED] done"

    win_spaces = r"Saved C:\Users\Alice Smith\My Documents\secret.pdf ok"
    cleaned_win_spaces = sanitize_wake_text(win_spaces)
    assert "Alice Smith" not in cleaned_win_spaces
    assert "secret.pdf" not in cleaned_win_spaces
    assert cleaned_win_spaces == "Saved [PATH] ok"

    unc_spaces_nested = r"Copied to \\fileserver01\finance\My Secret\secret_payroll.xlsx cleanly"
    assert sanitize_wake_text(unc_spaces_nested) == "Copied to [PATH] cleanly"

    unc_spaces_share = r"Copied to \\fileserver01\Shared Docs\secret_payroll.xlsx cleanly"
    assert sanitize_wake_text(unc_spaces_share) == "Copied to [PATH] cleanly"

    cross_prose_drive_and_unc = (
        r"Saved C:\Users\Alice Smith\My Documents\secret.pdf thanh cong and "
        r"\\fileserver01\Shared Docs\secret_payroll.xlsx ok"
    )
    assert sanitize_wake_text(cross_prose_drive_and_unc) == "Saved [PATH] thanh cong and [PATH] ok"

    cross_prose_escape = (
        r"Saved C:\Users\Alice Smith\My Documents\secret.pdf ok, escape seq a\b done"
    )
    assert sanitize_wake_text(cross_prose_escape) == r"Saved [PATH] ok, escape seq a\b done"

    svg_comma_space = (
        'Icon data:image/svg+xml;charset=utf-8, <svg xmlns="http://www.w3.org/2000/svg">'
        '<circle r="5"/></svg> done'
    )
    assert sanitize_wake_text(svg_comma_space) == "Icon [IMAGE_REDACTED] done"

    file_uri_posix = "Opened file:///Users/alice/secret/key.pem ok"
    assert sanitize_wake_text(file_uri_posix) == "Opened [PATH] ok"

    file_uri_win = "Opened file:///C:/Users/alice/secret/key.pem ok"
    assert sanitize_wake_text(file_uri_win) == "Opened [PATH] ok"

    # Non-regression: URLs, dates, Rule H1 messages, and clean prose remain intact
    normal = "Hoàn tất ngày 22/11/2026 tại https://example.com/api/v1/status rồi."
    assert sanitize_wake_text(normal) == normal
    rule_h1 = "Rule H1 check failed: no real tool executed in sub-goal"
    assert sanitize_wake_text(rule_h1) == rule_h1


def test_planner_executor_process_tool_result_sanitizes_failure_and_complete_summaries() -> None:
    now = datetime.now(UTC)
    planner = JobPlannerExecutor(clock_utc=lambda: now)
    sg = SubGoal(id="sg1", title="Inspect", description="Inspect host")
    job = AutonomousJob(
        job_id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        pet_id="pet_1",
        host_id=HostId.generate(),
        session_id=None,
        title="Sanitize tool results",
        goal="Inspect",
        state=AutonomousJobState.RUNNING,
        sub_goals=(sg,),
        checkpoint_data={"active_turns": [], "compacted_history": []},
        created_at=now,
        updated_at=now,
    )

    raw_payload = (
        "Failed reading /Users/thảo/tài_liệu/bí_mật.txt and "
        r"\\fileserver01\finance\payroll.xlsx with Bearer sample_tok_999 and "
        'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg"><circle r="1"/></svg>'
    )
    fail_outcome = planner.process_tool_result(
        job=job,
        tool_name="fs.read",
        tool_result={"status": "failed", "error": raw_payload},
        mark_sub_goal_complete=False,
        now=now,
    )
    assert fail_outcome.kind == "failed"
    assert fail_outcome.sub_goals is not None
    fail_sg_summary = fail_outcome.sub_goals[0].result_summary or ""
    fail_cp_text = json.dumps(fail_outcome.checkpoint_data)
    for forbidden in ("tài_liệu", "fileserver01", "sample_tok_999", "<svg"):
        assert forbidden not in fail_sg_summary
        assert forbidden not in fail_cp_text
        assert forbidden not in (fail_outcome.error_message or "")

    ok_outcome = planner.process_tool_result(
        job=job,
        tool_name="ui.inspect",
        tool_result={"status": "ok"},
        mark_sub_goal_complete=True,
        outcome_summary=raw_payload,
        now=now,
    )
    assert ok_outcome.kind == "completed"
    assert ok_outcome.sub_goals is not None
    ok_sg_summary = ok_outcome.sub_goals[0].result_summary or ""
    ok_cp_text = json.dumps(ok_outcome.checkpoint_data)
    for forbidden in ("tài_liệu", "fileserver01", "sample_tok_999", "<svg"):
        assert forbidden not in ok_sg_summary
        assert forbidden not in ok_cp_text


def _paired_median_cpu_elapsed(
    fn: Callable[[str], str],
    small_payload: str,
    large_payload: str,
    runs: int = 5,
) -> tuple[float, float, str]:
    fn(small_payload)
    out_large = fn(large_payload)
    small_samples: list[float] = []
    large_samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(runs):
            t0 = time.process_time()
            fn(small_payload)
            small_samples.append(time.process_time() - t0)
            t1 = time.process_time()
            out_large = fn(large_payload)
            large_samples.append(time.process_time() - t1)
    finally:
        if gc_was_enabled:
            gc.enable()
    return statistics.median(small_samples), statistics.median(large_samples), out_large


def test_consent_u5_linear_redos_guards_for_paths_svg_and_pem() -> None:
    # 1. Windows backslash drive ReDoS guard (n=16000)
    t0 = time.perf_counter()
    drive_bs = sanitize_wake_text("Saved C:\\" + ("a\\" * 16000) + "secret.pdf ok")
    assert time.perf_counter() - t0 < 0.15
    assert drive_bs == "Saved [PATH] ok"

    # 2. Windows UNC ReDoS guard (n=16000)
    t0 = time.perf_counter()
    unc_bs = sanitize_wake_text(r"Copied \\srv\share" + ("\\a" * 16000) + r"\secret.xlsx ok")
    assert time.perf_counter() - t0 < 0.15
    assert unc_bs == "Copied [PATH] ok"

    # 3. Forward-slash Windows drive ReDoS guard (n=16000, not mangled to [BASE64_REDACTED])
    t0 = time.perf_counter()
    drive_fs = sanitize_wake_text("Saved D:/" + ("a/" * 16000) + "secret.pdf ok")
    assert time.perf_counter() - t0 < 0.15
    assert drive_fs == "Saved [PATH] ok"

    # 4. Pathological unclosed/dotted/colon/param inputs (< 0.15s each) + output invariants
    for payload, expected_prefix in (
        ("Saved C:\\" + ("one two three four five " * 200), "Saved [PATH] two three four five "),
        ("Saved C:\\" + ("a.b " * 1000), "Saved [PATH] a.b "),
        ("C:" + ("a:b:" * 2000), "C:a:b:a:b:"),
        ("data:image/png" + (";k=v" * 2500), "data:image/png;k=v;k=v;k=v"),
    ):
        t0 = time.perf_counter()
        out = sanitize_wake_text(payload)
        assert time.perf_counter() - t0 < 0.15
        assert out.startswith(expected_prefix)
        assert len(out) == 2000

    # 4b. Trailing ')' / ']' and alternating punct O(n) leaf stripping
    for trailing_ch in (")", "]"):
        trailing_32k = "C:\\file" + (trailing_ch * 32000)
        trailing_64k = "C:\\file" + (trailing_ch * 64000)
        dt_trailing_32k, dt_trailing_64k, trailing_out = _paired_median_cpu_elapsed(
            sanitize_wake_text, trailing_32k, trailing_64k
        )
        assert dt_trailing_64k < 0.20
        assert dt_trailing_64k < max(dt_trailing_32k * 2.6, 0.02)
        assert trailing_out.startswith("[PATH]")
        assert "file" not in trailing_out

    for alt_pat in (")]" * 32000, ")])." * 16000, ".)" * 32000, ".]" * 32000):
        alt_32k = "C:\\file" + alt_pat[:32000]
        alt_64k = "C:\\file" + alt_pat
        dt_alt_32k, dt_alt_64k, alt_out = _paired_median_cpu_elapsed(
            sanitize_wake_text, alt_32k, alt_64k
        )
        assert dt_alt_64k < 0.20
        assert dt_alt_64k < max(dt_alt_32k * 2.6, 0.02)
        assert alt_out.startswith("[PATH]")
        assert "file" not in alt_out

    # 4c. Linear scaling for 400 KB unclosed <svg repetition (< 0.25s, ~2x per 2x)
    svg_200k = "data:image/svg+xml, <svg " * 8000
    svg_400k = "data:image/svg+xml, <svg " * 16000
    dt_200k, dt_400k, svg_400k_out = _paired_median_cpu_elapsed(
        redact_data_images_linear, svg_200k, svg_400k
    )
    svg_200k_out = redact_data_images_linear(svg_200k)
    assert "<svg" not in svg_200k_out
    assert "<svg" not in svg_400k_out
    assert dt_400k < 0.25
    assert dt_400k < max(dt_200k * 2.6, 0.02)

    # 4d. Linear scaling for unclosed <svg> with repeated inner <text> tags (100 KB -> 200 KB)
    svg_inner_100k = "data:image/svg+xml,<svg>" + ("<text>a</text>" * 7140)
    svg_inner_200k = "data:image/svg+xml,<svg>" + ("<text>a</text>" * 14280)
    dt_inner_100k, dt_inner_200k, svg_inner_200k_out = _paired_median_cpu_elapsed(
        redact_data_images_linear, svg_inner_100k, svg_inner_200k
    )
    svg_inner_100k_out = redact_data_images_linear(svg_inner_100k)
    assert svg_inner_100k_out == "[IMAGE_REDACTED]"
    assert svg_inner_200k_out == "[IMAGE_REDACTED]"
    assert dt_inner_200k < 0.25
    assert dt_inner_200k < max(dt_inner_100k * 2.6, 0.02)

    t0 = time.perf_counter()
    svg_inner_wake = sanitize_wake_text("data:image/svg+xml,<svg>" + ("<text>a</text>" * 20000))
    assert time.perf_counter() - t0 < 0.25
    assert svg_inner_wake == "[IMAGE_REDACTED]"

    t0 = time.perf_counter()
    long_unclosed_svg = sanitize_wake_text(
        'data:image/svg+xml, <svg attr="' + ("x" * 10000) + '"> done'
    )
    assert time.perf_counter() - t0 < 0.15
    assert long_unclosed_svg == "[IMAGE_REDACTED] done"

    # 5. 100 KB unclosed <svg repetition & 112 KB unclosed PEM repetition
    svg_100k = "data:image/svg+xml, <svg " * 4000
    t0 = time.perf_counter()
    svg_100k_out = redact_data_images_linear(svg_100k)
    assert time.perf_counter() - t0 < 0.25
    assert "<svg" not in svg_100k_out

    t0 = time.perf_counter()
    sanitize_wake_text(svg_100k)
    assert time.perf_counter() - t0 < 0.15

    pem_112k = "-----BEGIN PRIVATE KEY----- " * 4000
    t0 = time.perf_counter()
    pem_res = redact_sensitive_text(pem_112k)
    assert time.perf_counter() - t0 < 0.25
    assert pem_res.redacted_count == 0

    t0 = time.perf_counter()
    sanitize_wake_text(pem_112k)
    assert time.perf_counter() - t0 < 0.15

    # 6. Self-closing <svg/> must not swallow prose before a later </svg>
    sc_before_later = 'Icon data:image/svg+xml, <svg width="10"/> middle <svg><circle/></svg> done'
    assert (
        sanitize_wake_text(sc_before_later)
        == "Icon [IMAGE_REDACTED] middle <svg><circle/></svg> done"
    )

    # 7. Repeated unclosed KV secrets, JSON-key KV, and quoted XML comments stay strictly O(n)
    dt_kv_20k, dt_kv_40k, _ = _paired_median_cpu_elapsed(
        lambda s: redact_sensitive_text(s).text,
        'password=\\"a' * 20_000,
        'password=\\"a' * 40_000,
    )
    assert dt_kv_40k < 0.20
    assert dt_kv_40k < max(dt_kv_20k * 2.6, 0.02)

    dt_jkv_20k, dt_jkv_40k, _ = _paired_median_cpu_elapsed(
        lambda s: redact_sensitive_text(s).text,
        '{"password": "a' * 20_000,
        '{"password": "a' * 40_000,
    )
    assert dt_jkv_40k < 0.20
    assert dt_jkv_40k < max(dt_jkv_20k * 2.6, 0.02)

    dt_cmt_20k, dt_cmt_40k, _ = _paired_median_cpu_elapsed(
        redact_data_images_linear,
        'data:image/svg+xml,"<!-- ' * 8_000,
        'data:image/svg+xml,"<!-- ' * 16_000,
    )
    assert dt_cmt_40k < 0.25
    assert dt_cmt_40k < max(dt_cmt_20k * 2.6, 0.02)

    dt_svg_jnl_10k, dt_svg_jnl_20k, _ = _paired_median_cpu_elapsed(
        redact_data_images_linear,
        "data:image/svg+xml, \\n" * 10_000,
        "data:image/svg+xml, \\n" * 20_000,
    )
    assert dt_svg_jnl_20k < 0.25
    assert dt_svg_jnl_20k < max(dt_svg_jnl_10k * 2.6, 0.02)


def test_consent_u5_boundaries_json_urls_dates_base64_truncation_and_idempotency() -> None:
    # Sentence-ending period after extension before unrelated a\b escape
    sentence_trail = r"Saved C:\Users\Alice Smith\My Documents\secret.pdf. Next file a\b done"
    cleaned_sentence = sanitize_wake_text(sentence_trail)
    assert "Alice Smith" not in cleaned_sentence
    assert "secret.pdf" not in cleaned_sentence
    assert cleaned_sentence == r"Saved [PATH]. Next file a\b done"

    # Single-line JSON payload with mixed paths, secrets, and HTTPS URL
    json_payload = json.dumps(
        {
            "path": r"C:\Users\alice\My Docs\secret.txt",
            "fwd": "D:/Secret Docs/token.txt",
            "posix": "/Users/alice/my folder/key.pem",
            "url": "https://example.com/api/v1/status",
            "auth": "Bearer tok_secret_12345",
        }
    )
    cleaned_json = sanitize_wake_text(json_payload)
    for secret_part in (
        "alice",
        "secret.txt",
        "Secret Docs",
        "token.txt",
        "key.pem",
        "tok_secret_12345",
    ):
        assert secret_part not in cleaned_json
    assert "https://example.com/api/v1/status" in cleaned_json

    # URLs preserved vs file:// URIs redacted
    for safe_url in (
        "https://example.com/api/v1/status?q=1",
        "http://127.0.0.1:8080/v1/health",
        "wss://relay.example.com/v1/stream",
        "my_file://host/a/b",
    ):
        assert sanitize_wake_text(safe_url) == safe_url

    assert sanitize_wake_text("file://localhost/Users/alice/secret/key.pem") == "[PATH]"
    assert sanitize_wake_text("file:///C:/Users/Alice/secret.txt?x=1#y") == "[PATH]"
    assert sanitize_wake_text('"file:///Users/alice/my dir/key.pem"') == '"[PATH]"'
    assert sanitize_wake_text("'file:///D:/My Dir/key.pem'") == "'[PATH]'"
    assert sanitize_wake_text("(file:///Users/alice/secret.txt)") == "([PATH])"
    assert sanitize_wake_text("Path:file:///Users/alice/secret.txt") == "Path:[PATH]"

    # Dates, ratios, versions, and clean Vietnamese prose preserved intact
    vi_dates = "Ngày 22/11/2026 (tỷ lệ 10/20, phiên bản v1/v2) đã hoàn tất kiểm tra 100%."
    assert sanitize_wake_text(vi_dates) == vi_dates
    assert (
        sanitize_wake_text("Saved D:/Users/alice/file.txt on 22/11/2026")
        == "Saved [PATH] on 22/11/2026"
    )
    assert (
        sanitize_wake_text(
            "Đường dẫn:D:/Tài liệu/bí mật.txt và C:\\Users\\thảo\\tài liệu\\bí mật.txt xong"
        )
        == "Đường dẫn:[PATH] và [PATH] xong"
    )
    assert sanitize_wake_text("mục:/Users/alice/secret.txt xong") == "mục:[PATH] xong"

    # Raw base64 blob containing '/' redacted as [BASE64_REDACTED], NOT [PATH]
    jpeg_b64 = "/9j/4AAQSkZJRgABAQEASABIAAD/" + ("A1B2c3D4/" * 15)
    assert sanitize_wake_text(f"Raw {jpeg_b64} end") == "Raw [BASE64_REDACTED] end"

    # Title max_chars=120 boundary safety & idempotency
    bounded_110 = sanitize_wake_text("a" * 110 + "Bearer supersecrettoken99", max_chars=120)
    assert "supersecrettoken99" not in bounded_110
    assert len(bounded_110) <= 120
    assert "[REDACTED]" in bounded_110

    bounded_112 = sanitize_wake_text("a" * 112 + "Bearer supersecrettoken99", max_chars=120)
    assert "supersecrettoken99" not in bounded_112
    assert "Bearer" not in bounded_112
    assert len(bounded_112) <= 120

    for sample in (
        "Saved [PATH]\\ and [PATH]/ plus [REDACTED].",
        "Icon [IMAGE_REDACTED] and [BASE64_REDACTED]!",
    ):
        assert sanitize_wake_text(sample) == sample

    # Plain-English planner status parity
    for plain in (
        "Node verified ok",
        "Host channel dropped mid-call",
        "Rule H1 check failed: no real tool executed in sub-goal",
        "Sub-goal 1 finished at step 6",
    ):
        assert sanitize_wake_text(plain) == plain


def test_planner_executor_and_wake_inbox_sanitize_forward_slash_and_32kb_payloads() -> None:
    now = datetime.now(UTC)
    planner = JobPlannerExecutor(clock_utc=lambda: now)
    sg = SubGoal(id="sg1", title="Inspect", description="Inspect host")
    job = AutonomousJob(
        job_id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        pet_id="pet_1",
        host_id=HostId.generate(),
        session_id=None,
        title="Sanitize long forward-slash tool error",
        goal="Inspect",
        state=AutonomousJobState.RUNNING,
        sub_goals=(sg,),
        checkpoint_data={
            "active_turns": [{"thought_signature": "sig_keep_1", "text": "step 1"}],
            "compacted_history": [],
        },
        created_at=now,
        updated_at=now,
    )

    large_error = "Saved D:/Secret Docs/token.txt " + ("a\\" * 16000) + "secret.pdf"
    t0 = time.perf_counter()
    outcome = planner.process_tool_result(
        job=job,
        tool_name="fs.read",
        tool_result={"status": "failed", "error": large_error},
        mark_sub_goal_complete=False,
        now=now,
    )
    assert time.perf_counter() - t0 < 0.2
    assert outcome.kind == "failed"
    assert outcome.sub_goals is not None
    sg_summary = outcome.sub_goals[0].result_summary or ""
    cp_text = json.dumps(outcome.checkpoint_data)
    for forbidden in ("Secret Docs", "token.txt", "secret.pdf"):
        assert forbidden not in sg_summary
        assert forbidden not in cp_text
        assert forbidden not in (outcome.error_message or "")
    assert outcome.checkpoint_data is not None
    assert (
        outcome.checkpoint_data["compacted_history"][0]["outcome"]
        == f"Failed: {sanitize_wake_text(large_error)}"
    )

    ok_outcome = planner.process_tool_result(
        job=job,
        tool_name="ui.inspect",
        tool_result={"status": "ok"},
        mark_sub_goal_complete=False,
        now=now,
    )
    assert ok_outcome.checkpoint_data is not None
    assert ok_outcome.checkpoint_data["active_turns"][0]["thought_signature"] == "sig_keep_1"

    async def _run_inbox() -> None:
        inbox = WakeInboxService(
            repo=InMemoryWakeInboxRepository(),
            clock_utc=lambda: now,
        )
        t_inbox = time.perf_counter()
        item = await inbox.emit_item(
            owner_id=job.owner_id,
            pet_id=job.pet_id,
            job_id=job.job_id,
            host_id=job.host_id,
            kind=WakeInboxKind.JOB_FAILED,
            title="Opened D:/Secret Docs/token.txt",
            summary=("Prefix " + ("word " * 14000) + "Bearer secret_tail_999"),
        )
        assert time.perf_counter() - t_inbox < 0.2
        assert "Secret Docs" not in item.title
        assert "token.txt" not in item.title
        assert "secret_tail_999" not in item.summary

    asyncio.run(_run_inbox())
