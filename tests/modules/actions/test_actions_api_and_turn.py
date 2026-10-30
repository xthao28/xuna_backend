"""API and SSE turn-loop tests for `actions` and `/v1/conversations/{pet_id}/turns` (UDC-4 S4.3).

Covers:
1. Back-compat snapshot: old clients omitting `clientCapabilities` receive identical
   `ChatRequest` (no tools, no tool system prompt) and SSE events.
2. Capable client (`clientCapabilities=["host_actions_v1"]`) with a fake LLM emitting
   an R1 tool call (`media__play_pause`) -> `action_result` SSE + model follow-up reply.
3. Capable client with an R2 tool call (`url__open`) -> `action_pending` SSE ->
   `POST /v1/actions/approvals/{approvalId}` (owner-scoped + idempotent) ->
   `action_result` SSE -> model follow-up reply.
4. Capable client with `board__media_key` (`clientCapabilities=["board_hid_v1"]`,
   `boardHid={"available": True}`) -> `action_pending` -> approve ->
   `board_hid_request` -> `POST /v1/actions/{invocationId}/result` -> `action_result`.
5. Multi-round cap (`MAX_TOOL_ROUNDS = 3`) and per-round cap (`MAX_CALLS_PER_ROUND = 3`).
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from tests.helpers import LocalIssuer, MemoryAuditLog

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionResultEvent,
    ApprovalRequest,
    BoardHidRequestEvent,
)
from xuna_backend.modules.actions.infrastructure.repository import (
    InMemoryApprovalRepository,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Message,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
)
from xuna_backend.modules.consent.domain.consent import ConsentState
from xuna_backend.modules.conversation.application.chat_turn import (
    PERSONA_VI,
    build_request,
    chat_turn,
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
from xuna_backend.modules.identity.domain.entities import Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin

NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)
PET_HEX = "d945c77026fd4f4a8a899bbff4aeed86"
FAKE_PUBKEY = b"\x04" + bytes(range(1, 65))


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
    def __init__(self, owner: Owner) -> None:
        self._owner = owner

    async def get_or_create(self, claims: Any) -> Owner:
        return self._owner


class _FakeIdentityRepo:
    def __init__(self, owner_id: uuid.UUID, pet_hex: str) -> None:
        self._pet = Pet(
            pet_id=PetId(pet_hex),
            owner_id=owner_id,
            origin=PetOrigin.MIGRATED_FROM_V1,
            active_brain_id=uuid.uuid4(),
            device_id=uuid.uuid4(),
            device_public_key=FAKE_PUBKEY,
            last_generation=1,
            bound_at=NOW,
        )

    async def get_pet(self, pet_id: PetId) -> Pet | None:
        if pet_id == self._pet.pet_id:
            return self._pet
        return None


class _FakeConsentRepo:
    def __init__(self) -> None:
        self._state = ConsentState(
            aggregate_metrics=False,
            experience_events=False,
            conversation_text=True,
            conversation_voice=True,
            owner_documents=False,
            version=1,
            updated_at=NOW,
        )

    async def current(self, owner_id: uuid.UUID) -> ConsentState:
        return self._state


class _FakeMemoryRepo:
    async def list_candidates(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def add_candidate(self, *args: Any, **kwargs: Any) -> None:
        pass


class _FakeKnowledgeRepo:
    async def list_domains(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class _FakeKnowledgeCandidatePort:
    async def list_candidates_by_domain(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class _FakeHostRepo:
    def __init__(self) -> None:
        self.hosts: dict[HostId, Host] = {}
        self.manifests: dict[HostId, Manifest] = {}
        self.grants: dict[HostId, list[Grant]] = {}

    async def get_host(self, host_id: HostId) -> Host | None:
        return self.hosts.get(host_id)

    async def save_host(self, host: Host) -> None:
        self.hosts[host.host_id] = host

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        return [h for h in self.hosts.values() if h.owner_id == owner_id]

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self.grants.get(host_id, []))

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self.grants[host_id] = list(grants)

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifests.get(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        self.manifests[manifest.host_id] = manifest


class _ScriptedLLM:
    """Fake `TextGenerator` whose rounds are scripted by a callback or list."""

    def __init__(
        self,
        rounds: list[list[ChatEvent]] | Callable[[ChatRequest, int], AsyncIterator[ChatEvent]],
    ) -> None:
        self.requests: list[ChatRequest] = []
        self._rounds = rounds

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        idx = len(self.requests)
        self.requests.append(request)
        if callable(self._rounds):
            async for ev in self._rounds(request, idx):
                yield ev
            return
        events = self._rounds[min(idx, len(self._rounds) - 1)]
        for ev in events:
            yield ev


def _parse_sse(raw_text: str) -> list[tuple[str, dict[str, Any]]]:
    parsed: list[tuple[str, dict[str, Any]]] = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        ev_name = ""
        data_dict: dict[str, Any] = {}
        for line in lines:
            if line.startswith("event: "):
                ev_name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                data_dict = json.loads(line[len("data: ") :])
        if ev_name:
            parsed.append((ev_name, data_dict))
    return parsed


def _build_test_harness(
    llm: _ScriptedLLM,
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


# ---------------------------------------------------------------------------
# 1. Back-compatibility snapshot test for old clients (no `clientCapabilities`)
# ---------------------------------------------------------------------------


def _done(latency_ms: int = 10) -> ChatFinished:
    return ChatFinished(
        provider="fake",
        model="m1",
        input_tokens=10,
        output_tokens=5,
        latency_ms=latency_ms,
    )


def test_old_client_without_capabilities_is_unchanged() -> None:
    llm = _ScriptedLLM(
        [
            [
                TextDelta("Chào bạn, mình là XUNA!"),
                _done(10),
            ]
        ]
    )
    client, issuer, owner, host_repo, _, _ = _build_test_harness(llm)

    # Even if the owner has an online host, omitting `clientCapabilities` offers 0 tools
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

    headers = {"Authorization": f"Bearer {issuer.token('owner-1')}"}
    resp = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        json={"schemaVersion": 1, "text": "Chào XUNA"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert len(llm.requests) == 1
    req = llm.requests[0]
    assert req.tools == ()
    assert req.messages[0].content == PERSONA_VI

    events = _parse_sse(resp.text)
    assert [name for name, _ in events] == ["delta", "finished"]
    assert events[0][1] == {"text": "Chào bạn, mình là XUNA!"}


# ---------------------------------------------------------------------------
# 2. Capable client R1 tool call (`media__play_pause`) over `/turns` SSE
# ---------------------------------------------------------------------------


def test_turn_sse_r1_tool_call_and_followup_reply() -> None:
    llm = _ScriptedLLM(
        [
            [
                ToolCallRequested(
                    ToolCall(
                        id="call-1",
                        name="media__play_pause",
                        arguments={"target": "Mac"},
                    )
                ),
                _done(12),
            ],
            [
                TextDelta("<g:bounce_play>Mình đã tạm dừng nhạc trên Mac rồi nhé!"),
                _done(15),
            ],
        ]
    )
    client, issuer, owner, host_repo, _, _ = _build_test_harness(llm)

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

    # Register a live session on `host_session_hub` that replies `ok` to `tools/call`
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
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 18},
                },
            )

    async def noop() -> None:
        pass

    asyncio.run(hub.open_session(str(host.host_id), object(), fake_send_text, noop, noop))

    headers = {"Authorization": f"Bearer {issuer.token('owner-1')}"}
    resp = client.post(
        f"/v1/conversations/{PET_HEX}/turns",
        json={
            "schemaVersion": 1,
            "text": "Tạm dừng nhạc trên Mac",
            "clientCapabilities": ["host_actions_v1"],
        },
        headers=headers,
    )
    assert resp.status_code == 200
    assert len(llm.requests) == 2
    assert [t.name for t in llm.requests[0].tools] == ["media__play_pause"]
    # Round 1 request contains the assistant tool_calls message + Role.TOOL result message
    round1_roles = [m.role for m in llm.requests[1].messages]
    assert round1_roles[-2:] == [Role.ASSISTANT, Role.TOOL]
    assert llm.requests[1].messages[-2].tool_calls[0].name == "media__play_pause"

    events = _parse_sse(resp.text)
    assert [name for name, _ in events] == [
        "action_result",
        "gesture",
        "delta",
        "finished",
    ]
    assert events[0][1]["status"] == "ok"
    assert events[0][1]["target"] == "Mac"
    assert events[0][1]["gesture"] == "bounce_play"
    assert events[1][1] == {"name": "bounce_play"}
    assert events[2][1] == {"text": "Mình đã tạm dừng nhạc trên Mac rồi nhé!"}


# ---------------------------------------------------------------------------
# 3. R2 approval route (`POST /v1/actions/approvals/{approvalId}`) & Board HID route
# ---------------------------------------------------------------------------


def test_action_approval_and_board_result_endpoints_owner_scoping_and_idempotency() -> None:
    llm = _ScriptedLLM(
        [
            [
                ToolCallRequested(
                    ToolCall(
                        id="call-hid-1",
                        name="board__media_key",
                        arguments={"key": "next"},
                    )
                ),
                _done(10),
            ],
            [
                TextDelta("Mình đã bấm chuyển bài tiếp theo rồi nhé!"),
                _done(11),
            ],
        ]
    )
    client, issuer, owner, _, _, _ = _build_test_harness(llm)
    owner_headers = {"Authorization": f"Bearer {issuer.token('owner-1')}"}

    # Unknown approvalId returns 404
    missing_resp = client.post(
        f"/v1/actions/approvals/{uuid.uuid4()}",
        json={"decision": "approve"},
        headers=owner_headers,
    )
    assert missing_resp.status_code == 404

    # Unknown invocationId returns 404
    missing_res = client.post(
        "/v1/actions/nonexistent-inv/result",
        json={"status": "ok"},
        headers=owner_headers,
    )
    assert missing_res.status_code == 404

    # Pre-register a pending approval and test `POST /v1/actions/approvals/{approvalId}`
    live_now = datetime.now(UTC)
    approval_id = uuid.uuid4()
    req = ApprovalRequest(
        approval_id=approval_id,
        owner_id=owner.owner_id,
        host_id=None,
        target_alias="Máy đang ghép",
        tool="board.media_key",
        args_sha256="b" * 64,
        risk=RiskTier.R2,
        summary="XUNA sẽ bấm Bài tiếp (⏭) trên máy đang ghép",
        created_at=live_now,
        expires_at=live_now + timedelta(seconds=60),
    )
    approval_repo: InMemoryApprovalRepository = client.app.state.approval_repository(None)  # type: ignore[union-attr]
    approval_broker = client.app.state.approval_broker  # type: ignore[union-attr]
    asyncio.run(approval_repo.save(req))
    asyncio.run(approval_broker.create(req))

    # First call approves
    resp1 = client.post(
        f"/v1/actions/approvals/{approval_id}",
        json={"decision": "approve"},
        headers=owner_headers,
    )
    assert resp1.status_code == 200
    assert resp1.json()["state"] == "approved"

    # Second call is idempotent
    resp2 = client.post(
        f"/v1/actions/approvals/{approval_id}",
        json={"decision": "approve"},
        headers=owner_headers,
    )
    assert resp2.status_code == 200
    assert resp2.json()["state"] == "approved"

    # Register a local board invocation and report its result via `POST /v1/actions/{id}/result`
    local_broker = client.app.state.local_result_broker  # type: ignore[union-attr]
    local_broker.register("inv-123", owner.owner_id)
    res_ok = client.post(
        "/v1/actions/inv-123/result",
        json={"status": "ok"},
        headers=owner_headers,
    )
    assert res_ok.status_code == 200
    assert res_ok.json() == {"invocationId": "inv-123", "status": "ok"}


# ---------------------------------------------------------------------------
# 4. Multi-round cap (3 rounds) and per-round cap (3 calls) in `chat_turn`
# ---------------------------------------------------------------------------


def test_chat_turn_enforces_three_rounds_and_three_calls_per_round() -> None:
    async def scripted_stream(request: ChatRequest, round_idx: int) -> AsyncIterator[ChatEvent]:
        if request.tools:
            # Emit 4 tool calls in round 0 to test MAX_CALLS_PER_ROUND = 3
            count = 4 if round_idx == 0 else 1
            for i in range(count):
                yield ToolCallRequested(
                    ToolCall(
                        id=f"r{round_idx}-c{i}",
                        name="media__play_pause",
                        arguments={},
                    )
                )
            yield _done(5)
        else:
            # Final round has `tools == ()` after 3 tool rounds
            yield TextDelta("Đã hoàn tất chuỗi hành động.")
            yield _done(5)

    llm = _ScriptedLLM(scripted_stream)
    base_req = build_request(
        "Thử chuỗi công cụ",
        tools=(
            ToolSpec(
                name="media__play_pause",
                description="Play/pause",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        ),
    )

    executed_ids: list[str] = []

    async def runner(call: ToolCall, emit: Any) -> Message:
        executed_ids.append(call.id)
        await emit(
            ActionResultEvent(
                invocation_id=f"inv-{call.id}",
                target="Mac",
                summary="✓ Đã thực hiện",
                status="ok",
                gesture="bounce_play",
            )
        )
        return Message(
            role=Role.TOOL,
            content='{"status":"ok"}',
            tool_call_id=call.id,
        )

    async def collect() -> list[Any]:
        return [ev async for ev in chat_turn(llm, base_req, tool_runner=runner)]

    events = asyncio.run(collect())
    # Round 0 had 4 calls -> only 3 executed, 4th got `too_many_calls`.
    # Rounds 1 and 2 had 1 call each -> total 5 executed calls across 3 rounds.
    assert executed_ids == [
        "r0-c0",
        "r0-c1",
        "r0-c2",
        "r1-c0",
        "r2-c0",
    ]
    # Round 3 (4th LLM call) had `tools == ()`
    assert len(llm.requests) == 4
    assert llm.requests[3].tools == ()
    # Check that r0-c3 received `{"status":"too_many_calls"}`
    r1_tool_msgs = [m for m in llm.requests[1].messages if m.role == Role.TOOL]
    assert len(r1_tool_msgs) == 4
    assert json.loads(r1_tool_msgs[3].content) == {"status": "too_many_calls"}
    assert any(
        isinstance(e, TextDelta) and e.text == "Đã hoàn tất chuỗi hành động." for e in events
    )
    assert (
        sum(
            1
            for e in events
            if isinstance(e, ActionPendingEvent | BoardHidRequestEvent | ActionResultEvent)
        )
        == 5
    )
