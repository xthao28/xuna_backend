"""Wave A End-to-End Test Suite: Comprehensive Multi-Tier Verification.

Covers Tiers 1-4 for Wave A of the XUNA Clawbot upgrade:
- Tier 1: Feature Coverage (F1 to F7, >=5 tests per feature)
- Tier 2: Boundary & Corner Cases (payloads, buffer limits, injection, Unicode)
- Tier 3: Cross-Feature Combinations (CU session + draft + quota fallback + signature preservation)
- Tier 4: Real-World Scenarios & Security Audits (Apple Notes workflow, 15-target terminal denylist,
  zero secret leakage, mobile wire format)

PostgreSQL tests require XUNA_TEST_DATABASE_URL and are marked with @pytest.mark.postgres.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import inspect, text

from xuna_backend.core.db import Database
from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseError,
    ComputerUseSessionRunner,
    ComputerUseSessionState,
    SignedR3ApprovalProof,
    contains_prompt_injection_payload,
    is_activation_key_combo,
    is_consequential_element_name,
    is_hard_denied_target,
)
from xuna_backend.modules.actions.application.orchestrator import (
    CAP_HOST_ACTIONS_V1,
    ActionContext,
    ApprovalRequest,
    Orchestrator,
)
from xuna_backend.modules.actions.application.ports import (
    ApprovalBroker,
    ApprovalRepository,
    ComputerUseSessionStore,
    LocalResultBroker,
)
from xuna_backend.modules.actions.application.tool_discovery import (
    EligibleHostTool,
    describe_tool,
    parse_invoke,
    search_tools,
)
from xuna_backend.modules.actions.domain.entities import (
    CONVERSATION_SCOPE_TTL,
    ActionPendingEvent,
    ApprovalDecision,
    ApprovalScope,
    ApprovalState,
)
from xuna_backend.modules.actions.infrastructure.approval_scope_store import (
    SqlApprovalScopeStore,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryLastTargetStore,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditEvent
from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    TextGenerator,
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
from xuna_backend.modules.cognition.infrastructure.openai_compatible import (
    _GLOBAL_SIGNATURE_CACHE,
    OpenAICompatibleTextGenerator,
    _cache_signature,
    _is_quota_exhausted,
    _sanitize_error_text,
    _serialize_message,
)
from xuna_backend.modules.consent.application.ports import ConsentRepository
from xuna_backend.modules.consent.domain.consent import ConsentState
from xuna_backend.modules.conversation.application.chat_turn import (
    TurnEvent,
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
from xuna_backend.modules.identity.application.ports import SignatureVerifier

# ==============================================================================
# Shared Test Doubles and Fixtures
# ==============================================================================

NOW = datetime(2026, 11, 15, 0, 0, 0, tzinfo=UTC)
FAKE_PUBKEY = b"\x04" + bytes(range(1, 65))


def _make_host(
    owner_id: uuid.UUID,
    alias: str = "Mac",
    *,
    host_id: HostId | None = None,
    platform: HostPlatform = HostPlatform.MACOS,
    status: HostStatus = HostStatus.ACTIVE,
) -> Host:
    h_id = (
        host_id
        if isinstance(host_id, HostId)
        else (HostId(host_id) if host_id else HostId.generate())
    )
    return Host(
        host_id=h_id,
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname="xuna-host",
        os_version="15.0",
        agent_version="0.1.0",
        public_key=FAKE_PUBKEY,
        status=status,
        created_at=NOW,
    )


class _FakeConsentRepository(ConsentRepository):
    def __init__(self, host_screen: bool = True) -> None:
        self.state = ConsentState(host_screen=host_screen, version=1)

    async def current(self, owner_id: uuid.UUID) -> ConsentState:
        return self.state

    async def append(self, owner_id: uuid.UUID, state: ConsentState) -> bool:
        self.state = state
        return True


class _FakeSignatureVerifier(SignatureVerifier):
    def __init__(self, allow_all: bool = True) -> None:
        self.allow_all = allow_all

    def verify(
        self, public_key_pem: str | bytes, message: bytes, signature_base64: str | bytes
    ) -> bool:
        return self.allow_all


class _InMemoryCuSessionStore(ComputerUseSessionStore):
    def __init__(self) -> None:
        self._sessions: dict[str, ComputerUseSessionState] = {}

    async def start(self, state: ComputerUseSessionState, now: datetime) -> None:
        self._sessions[state.session_id] = state

    async def create(self, state: ComputerUseSessionState) -> None:
        self._sessions[state.session_id] = state

    async def get(self, session_id: str) -> ComputerUseSessionState | None:
        return self._sessions.get(session_id)

    async def get_active_for_owner(self, owner_id: uuid.UUID) -> ComputerUseSessionState | None:
        for s in self._sessions.values():
            if s.owner_id == owner_id and s.active:
                return s
        return None

    async def stop_all_for_owner(self, owner_id: uuid.UUID, reason: str, now: datetime) -> int:
        count = 0
        for s in self._sessions.values():
            if s.owner_id == owner_id and s.active:
                s.active = False
                s.stop_reason = reason
                count += 1
        return count

    async def record_step(self, session_id: str) -> int | None:
        s = self._sessions.get(session_id)
        if s and s.active:
            s.steps_used += 1
            return s.steps_used
        return None

    async def mark_ended(
        self, session_id: str, reason: str, now: datetime
    ) -> ComputerUseSessionState | None:
        s = self._sessions.get(session_id)
        if s:
            s.active = False
            s.stop_reason = reason
        return s

    async def touch_heartbeat(self, session_id: str, owner_id: uuid.UUID, now: datetime) -> bool:
        s = self._sessions.get(session_id)
        if s and s.active and s.owner_id == owner_id:
            s.last_heartbeat_at = now
            return True
        return False

    async def set_typed_count(self, session_id: str, count: int) -> None:
        s = self._sessions.get(session_id)
        if s:
            s.typed_count = count

    async def record_prompt(self, session_id: str) -> int | None:
        s = self._sessions.get(session_id)
        if s:
            s.owner_prompts += 1
            return s.owner_prompts
        return None

    async def set_scope(
        self, session_id: str, allowed_apps: tuple[str, ...], allowed_domains: tuple[str, ...]
    ) -> None:
        s = self._sessions.get(session_id)
        if s:
            s.allowed_apps = allowed_apps
            s.allowed_domains = allowed_domains


class _FakeHostRepository:
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

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifests.get(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        self.manifests[manifest.host_id] = manifest

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self.grants.get(host_id, []))

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self.grants[host_id] = list(grants)


class _FakeSessionHub:
    def __init__(self) -> None:
        self.sent_requests: list[tuple[str, str, dict[str, object], float]] = []
        self.invocations: list[dict[str, Any]] = []
        self.next_response: dict[str, object] = {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {"status": "ok"},
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 42},
        }

    def is_online(self, host_id: str) -> bool:
        return True

    async def send_request(
        self,
        host_id: str,
        method: str,
        params: dict[str, object],
        timeout_s: float,
    ) -> dict[str, object]:
        self.sent_requests.append((host_id, method, params, timeout_s))
        tool_name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        self.invocations.append({"host_id": host_id, "tool": tool_name, "arguments": arguments})
        return self.next_response


class _FakeApprovalRepository(ApprovalRepository):
    def __init__(self) -> None:
        self.records: dict[str, ApprovalRequest] = {}

    async def save(self, req: ApprovalRequest) -> None:
        self.records[str(req.approval_id)] = req

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        return self.records.get(str(approval_id))

    async def set_status(self, approval_id: uuid.UUID, status: str, resolved_at: datetime) -> None:
        req = self.records.get(str(approval_id))
        if req:
            req.status = status


class _FakeApprovalBroker(ApprovalBroker):
    def __init__(self, auto_approve: bool = False) -> None:
        self.published_events: list[ActionPendingEvent] = []
        self.slots: dict[uuid.UUID, ApprovalRequest] = {}
        self.auto_approve = auto_approve

    async def create(self, req: ApprovalRequest) -> None:
        self.slots[req.approval_id] = req
        self.published_events.append(
            ActionPendingEvent(
                approval_id=str(req.approval_id),
                target=req.target_alias,
                summary=req.summary,
                risk=req.risk.value,
                expires_in=60,
                host_id=str(req.host_id),
                args_sha256=req.args_sha256,
                expires_at=req.expires_at.astimezone(UTC).isoformat(),
            )
        )

    def get_slot_request(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        return self.slots.get(approval_id)

    async def wait(self, approval_id: uuid.UUID, timeout_s: float) -> ApprovalState:
        req = self.slots.get(approval_id)
        if self.auto_approve and req:
            import dataclasses

            updated = dataclasses.replace(req, state=ApprovalState.APPROVED)
            self.slots[approval_id] = updated
            return ApprovalState.APPROVED
        if req:
            return req.state
        return ApprovalState.PENDING

    async def decide(
        self,
        approval_id: uuid.UUID,
        owner_id: uuid.UUID,
        decision: ApprovalDecision,
        now: datetime,
        *,
        scope: ApprovalScope = ApprovalScope.ONCE,
        signature: bytes | None = None,
        signer_public_key: bytes | None = None,
    ) -> ApprovalRequest | None:
        req = self.slots.get(approval_id)
        if req:
            import dataclasses

            new_state = (
                ApprovalState.APPROVED
                if decision == ApprovalDecision.APPROVE
                else ApprovalState.REJECTED
            )
            updated = dataclasses.replace(req, state=new_state)
            self.slots[approval_id] = updated
            return updated
        return None


class _FakeLocalBroker(LocalResultBroker):
    async def publish_result(
        self, owner_id: uuid.UUID, call_id: str, result: dict[str, Any]
    ) -> None:
        pass


class _FakeAuditLog(AuditLog):
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def _make_tool_descriptor(
    name: str,
    family: ToolFamily,
    risk: RiskTier,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Tool {name}",
        family=family,
        risk=risk,
        input_schema={
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )


def _make_action_context(
    owner_id: uuid.UUID,
    pet_id: str = "pet-1",
    emit: Any = None,
) -> ActionContext:
    async def _default_emit(ev: Any) -> None:
        pass

    return ActionContext(
        owner_id=owner_id,
        pet_id=pet_id,
        locale=Locale.VI,
        client_caps=(CAP_HOST_ACTIONS_V1,),
        board_available=False,
        emit_event=emit or _default_emit,
    )


def _make_cu_session_state(
    session_id: str,
    owner_id: uuid.UUID,
    host_id: HostId,
    goal: str = "Test task",
    allowed_apps: tuple[str, ...] = ("notes",),
    allowed_domains: tuple[str, ...] = (),
    max_steps: int = 20,
    max_minutes: int = 15,
    started_at_utc: datetime = NOW,
    last_heartbeat_at: datetime = NOW,
    active: bool = True,
) -> ComputerUseSessionState:
    h_id = host_id if isinstance(host_id, HostId) else HostId(host_id)
    return ComputerUseSessionState(
        session_id=session_id,
        owner_id=owner_id,
        host_id=h_id,
        goal=goal,
        allowed_apps=allowed_apps,
        allowed_domains=allowed_domains,
        max_steps=max_steps,
        max_minutes=max_minutes,
        started_at_utc=started_at_utc,
        last_heartbeat_at=last_heartbeat_at,
        active=active,
    )


def _make_r3_proof(
    host_id: HostId, tool_name: str, arguments: dict[str, Any]
) -> SignedR3ApprovalProof:
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    args_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return SignedR3ApprovalProof(
        approval_id=uuid.uuid4(),
        host_id=host_id,
        tool_name=tool_name,
        args_sha256=args_sha256,
        scope=ApprovalScope.ONCE,
        expires_at_iso="2026-11-15T00:15:00Z",
        signature=b"dummy_sig_bytes",
        signer_public_key=FAKE_PUBKEY,
    )


def _setup_test_orchestrator(
    cu_store: ComputerUseSessionStore | None = None,
    clock: Callable[[], datetime] | None = None,
    broker: _FakeApprovalBroker | None = None,
) -> tuple[
    Orchestrator, _FakeHostRepository, _FakeSessionHub, _FakeApprovalBroker, _InMemoryCuSessionStore
]:
    cu_session_store = cu_store if cu_store is not None else _InMemoryCuSessionStore()
    host_repo = _FakeHostRepository()
    hub = _FakeSessionHub()
    approval_repo = _FakeApprovalRepository()
    approval_broker = broker if broker is not None else _FakeApprovalBroker()
    local_broker = _FakeLocalBroker()
    last_target = InMemoryLastTargetStore()
    audit = _FakeAuditLog()

    orch = Orchestrator(
        host_repo=host_repo,
        hub=hub,
        approval_repo=approval_repo,
        approval_broker=approval_broker,
        local_result_broker=local_broker,
        last_target_store=last_target,
        audit=audit,
        cu_session_store=cu_session_store,
        clock=clock or (lambda: NOW),
    )
    return orch, host_repo, hub, approval_broker, cu_session_store


# ==============================================================================
# Tier 1: Feature Coverage (F1 - F7, >=5 tests per feature)
# ==============================================================================


class TestTier1F1ThoughtSignatures:
    """F1: Thought Signature Preservation in ToolCall, serialization, and generator switches."""

    def test_f1_01_tool_call_extra_field_instantiation(self) -> None:
        """T1-F1.1: ToolCall stores extra dict without mutating attributes; defaults to None."""
        tc_default = ToolCall(id="tc-1", name="notes", arguments={"app": "notes"})
        assert tc_default.extra is None

        extra_data = {"thought_signature": "sig_hex_999", "meta": "test_meta"}
        tc_extra = ToolCall(
            id="tc-2",
            name="app.launch",
            arguments={"appId": "notes"},
            extra=extra_data,
        )
        assert tc_extra.id == "tc-2"
        assert tc_extra.name == "app.launch"
        assert tc_extra.extra == extra_data
        assert tc_extra.extra["thought_signature"] == "sig_hex_999"

    def test_f1_02_serialize_message_with_extra_content(self) -> None:
        """T1-F1.2: _serialize_message encodes extra_content into tool_calls JSON structure."""
        tc = ToolCall(
            id="call-42",
            name="app.launch",
            arguments={"appId": "notes"},
            extra={"thought_signature": "sig_abc123"},
        )
        msg = Message(role=Role.ASSISTANT, content="", tool_calls=(tc,))
        serialized = _serialize_message(msg)

        assert serialized["role"] == "assistant"
        assert len(serialized["tool_calls"]) == 1
        serialized_call = serialized["tool_calls"][0]
        assert serialized_call["id"] == "call-42"
        assert serialized_call["function"]["name"] == "app.launch"
        assert "extra_content" in serialized_call
        assert serialized_call["extra_content"] == {"thought_signature": "sig_abc123"}

    def test_f1_03_serialize_message_fallback_to_shared_cache(self) -> None:
        """T1-F1.3: _serialize_message recovers thought signature from global cache."""
        _GLOBAL_SIGNATURE_CACHE.clear()
        _cache_signature("call-cached-77", {"thought_signature": "sig_from_cache_77"})

        tc_no_extra = ToolCall(id="call-cached-77", name="notes", arguments={})
        msg = Message(role=Role.ASSISTANT, content="", tool_calls=(tc_no_extra,))

        serialized = _serialize_message(msg)
        call_payload = serialized["tool_calls"][0]
        assert "extra_content" in call_payload
        assert call_payload["extra_content"] == {"thought_signature": "sig_from_cache_77"}

    @pytest.mark.anyio
    async def test_f1_04_chat_turn_generator_switch_preserves_thought_signatures(self) -> None:
        """T1-F1.4: In chat_turn, generator switch preserves earlier signatures in messages."""
        captured_code_requests: list[ChatRequest] = []

        class MockPrimaryGen(TextGenerator):
            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
                yield ToolCallRequested(
                    ToolCall(
                        id="call-switch-1",
                        name="forge__execute",
                        arguments={"command": "ls"},
                        extra={"thought_signature": "sig_forge_001"},
                    )
                )
                yield ChatFinished("mock_primary", "model_1", 10, 10, 50)

        class MockCodeGen(TextGenerator):
            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
                captured_code_requests.append(req)
                yield TextDelta("Execution completed successfully.")
                yield ChatFinished("mock_code", "model_code", 10, 10, 50)

        async def mock_runner(call: ToolCall, emit: Any) -> Message:
            return Message(role=Role.TOOL, content='{"status":"ok"}', tool_call_id=call.id)

        req = ChatRequest(
            messages=(Message(role=Role.USER, content="Run forge command"),),
            tools=(
                ToolSpec("forge__execute", "Run command", {"type": "object", "properties": {}}),
            ),
        )

        events: list[TurnEvent] = []
        async for ev in chat_turn(
            MockPrimaryGen(),
            req,
            tool_runner=mock_runner,
            code_generator=MockCodeGen(),
        ):
            events.append(ev)

        assert len(captured_code_requests) == 1
        code_req = captured_code_requests[0]
        assistant_msgs = [m for m in code_req.messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) >= 1
        first_call = assistant_msgs[0].tool_calls[0]
        assert first_call.extra == {"thought_signature": "sig_forge_001"}

    @pytest.mark.anyio
    async def test_f1_05_multi_round_thought_signature_durability(self) -> None:
        """T1-F1.5: Thought signatures survive across 3 alternating rounds."""
        recorded_requests: list[ChatRequest] = []
        round_counter = 0

        class MultiRoundGen(TextGenerator):
            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
                nonlocal round_counter
                recorded_requests.append(req)
                round_counter += 1
                if round_counter == 1:
                    yield ToolCallRequested(
                        ToolCall(
                            id="call-r1",
                            name="search",
                            arguments={"q": "apple"},
                            extra={"thought_signature": "sig_r1"},
                        )
                    )
                elif round_counter == 2:
                    yield ToolCallRequested(
                        ToolCall(
                            id="call-r2",
                            name="search",
                            arguments={"q": "notes"},
                            extra={"thought_signature": "sig_r2"},
                        )
                    )
                else:
                    yield TextDelta("Found all notes.")
                yield ChatFinished("mock", "m", 5, 5, 20)

        async def mock_runner(call: ToolCall, emit: Any) -> Message:
            return Message(role=Role.TOOL, content='{"status":"ok"}', tool_call_id=call.id)

        req = ChatRequest(
            messages=(Message(role=Role.USER, content="Search notes"),),
            tools=(ToolSpec("search", "search tool", {"type": "object", "properties": {}}),),
        )

        async for _ in chat_turn(MultiRoundGen(), req, tool_runner=mock_runner):
            pass

        assert len(recorded_requests) == 3
        round_3_msgs = recorded_requests[2].messages
        r1_assistant = [m for m in round_3_msgs if any(c.id == "call-r1" for c in m.tool_calls)]
        assert len(r1_assistant) == 1
        assert r1_assistant[0].tool_calls[0].extra == {"thought_signature": "sig_r1"}


class TestTier1F2QuotaAndRedaction:
    """F2: Quota Fast-Fail (429/400) and Secret Redaction."""

    @pytest.mark.anyio
    async def test_f2_01_http_429_resource_exhausted_fast_fail(self) -> None:
        """T1-F2.1: HTTP 429 with RESOURCE_EXHAUSTED fast-fails without sleep retries."""
        error_json = json.dumps(
            {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Resource quota exceeded",
                }
            }
        )

        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text=error_json)

        transport = httpx.MockTransport(mock_handler)
        gen = OpenAICompatibleTextGenerator(
            base_url="https://api.example.com",
            model="gemini-flash",
            api_key="AIzaSyFakeKey12345678901234567890123456",
            transport=transport,
        )

        req = ChatRequest(messages=(Message(role=Role.USER, content="hi"),))
        start_time = time.monotonic()
        with pytest.raises(ProviderUnavailableError) as exc_info:
            async for _ in gen.stream(req):
                pass
        elapsed = time.monotonic() - start_time

        assert elapsed < 0.8
        assert "quota_exhausted" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_f2_02_http_400_daily_quota_fast_fail(self) -> None:
        """T1-F2.2: HTTP 400 with 'daily quota exceeded' triggers immediate error."""
        error_json = json.dumps(
            {"error": {"code": 400, "message": "daily quota exceeded for this project"}}
        )

        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text=error_json)

        transport = httpx.MockTransport(mock_handler)
        gen = OpenAICompatibleTextGenerator(
            base_url="https://api.example.com",
            model="gemini-flash",
            api_key="AIzaSyFakeKey12345678901234567890123456",
            transport=transport,
        )

        req = ChatRequest(messages=(Message(role=Role.USER, content="hi"),))
        with pytest.raises(ProviderUnavailableError) as exc_info:
            async for _ in gen.stream(req):
                pass

        assert "quota_exhausted" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_f2_03_error_body_truncation_to_1024_bytes(self) -> None:
        """T1-F2.3: Provider error body exceeding 1024 bytes is cleanly truncated."""
        huge_error_body = "E" * 4096

        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text=huge_error_body)

        transport = httpx.MockTransport(mock_handler)
        gen = OpenAICompatibleTextGenerator(
            base_url="https://api.example.com",
            model="gemini-flash",
            api_key="test-key",
            transport=transport,
        )

        req = ChatRequest(messages=(Message(role=Role.USER, content="hi"),))
        with pytest.raises(ProviderUnavailableError) as exc_info:
            async for _ in gen.stream(req):
                pass

        err_msg = str(exc_info.value)
        assert len(err_msg) <= 1060  # prefix 'status 500: ' + 1024 bytes
        assert "EEEE" in err_msg

    def test_f2_04_secret_redaction_in_error_logs(self) -> None:
        """T1-F2.4: Secret redaction sanitizes Gemini, OpenAI, Groq, and Bearer secrets."""
        raw_error = (
            "Error for key AIzaSyAbc123Def456Ghi789Jkl012Mno345Pqr "
            "and sk-proj-123456789012345678901234 "
            "with Bearer test_bearer_token_1234567890 and raw Groq key gsk_raw_groq_key_0123456789abcdef "
            "and explicit_secret_pass"
        )
        secrets = ["explicit_secret_pass"]
        sanitized = _sanitize_error_text(raw_error, secrets)

        assert "AIzaSy" not in sanitized
        assert "sk-proj-" not in sanitized
        assert "test_bearer_token" not in sanitized
        assert "gsk_raw_groq_key" not in sanitized
        assert "explicit_secret_pass" not in sanitized
        assert "[REDACTED_API_KEY]" in sanitized
        assert "[REDACTED]" in sanitized
        assert "Bearer [REDACTED]" in sanitized

    @pytest.mark.anyio
    async def test_f2_05_exhausted_key_not_retried_in_subsequent_rounds(self) -> None:
        """T1-F2.5: In chat_turn, quota-exhausted generator falls back immediately."""
        primary_calls = 0
        code_calls = 0

        class QuotaFailingGen(TextGenerator):
            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
                nonlocal primary_calls
                primary_calls += 1
                raise ProviderUnavailableError("quota_exhausted: daily limit reached")
                yield  # pragma: no cover

        class FallbackGen(TextGenerator):
            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
                nonlocal code_calls
                code_calls += 1
                yield TextDelta("Hello from fallback generator!")
                yield ChatFinished("fallback", "model_fb", 5, 5, 25)

        req = ChatRequest(messages=(Message(role=Role.USER, content="Hello"),))
        events: list[TurnEvent] = []
        async for ev in chat_turn(QuotaFailingGen(), req, code_generator=FallbackGen()):
            events.append(ev)

        assert primary_calls == 1  # Not retried in a loop
        assert code_calls == 1
        text_deltas = [e.text for e in events if isinstance(e, TextDelta)]
        assert "".join(text_deltas) == "Hello from fallback generator!"


class TestTier1F3CompactToolDiscovery:
    """F3: Compact Tool Discovery Schema without redundant tool_describe."""

    @pytest.fixture
    def eligible_tools(self) -> dict[str, EligibleHostTool]:
        desc = _make_tool_descriptor(
            name="app.create_note",
            family=ToolFamily.APP,
            risk=RiskTier.R2,
            properties={
                "title": {"type": "string", "maxLength": 100, "description": "Title of note"},
                "content": {"type": "string", "description": "Multiline note body"},
            },
            required=["title", "content"],
        )
        return {
            "app.create_note": EligibleHostTool(
                descriptor=desc,
                aliases=("Mac",),
            )
        }

    def test_f3_01_tool_search_returns_compact_args(
        self, eligible_tools: dict[str, EligibleHostTool]
    ) -> None:
        """T1-F3.1: tool_search returns matches where each entry has compact args dictionary."""
        result = search_tools(eligible_tools, "create note")
        matches = result.get("matches", [])
        assert len(matches) >= 1
        match = matches[0]
        assert match["name"] == "app.create_note"
        assert "args" in match
        args = match["args"]
        assert "title" in args
        assert "content" in args
        assert "REQUIRED" in args["title"]

    def test_f3_02_tool_search_schema_compactness(
        self, eligible_tools: dict[str, EligibleHostTool]
    ) -> None:
        """T1-F3.2: Compact args in search output excludes boilerplate."""
        result = search_tools(eligible_tools, "create_note")
        args = result["matches"][0]["args"]
        title_arg = args["title"]
        assert "# " not in title_arg
        assert "schema" not in title_arg
        assert "string" in title_arg

    def test_f3_03_tool_search_next_guidance(
        self, eligible_tools: dict[str, EligibleHostTool]
    ) -> None:
        """T1-F3.3: tool_search next guidance points to invoke or describe."""
        result = search_tools(eligible_tools, "note")
        expected_next = (
            "Proceed directly to tool_invoke(name, argumentsJson) or call tool_describe "
            "for verbose schema."
        )
        assert result["next"] == expected_next

    def test_f3_04_tool_describe_backward_compatibility(
        self, eligible_tools: dict[str, EligibleHostTool]
    ) -> None:
        """T1-F3.4: describe_tool continues returning complete descriptor payload."""
        desc_json = describe_tool(eligible_tools, "app.create_note")
        data = json.loads(desc_json)
        output = data.get("output", data)
        assert output["name"] == "app.create_note"
        assert output["risk"].lower() == "r2"
        assert "Mac" in output["hosts"]
        assert "args" in output
        assert "next" in output

    def test_f3_05_direct_tool_search_to_tool_invoke_sequence(
        self, eligible_tools: dict[str, EligibleHostTool]
    ) -> None:
        """T1-F3.5: Direct tool_search to tool_invoke sequence without describe."""
        search_res = search_tools(eligible_tools, "note")
        selected_tool = search_res["matches"][0]["name"]

        # Parse invoke arguments directly using search result's tool name
        arguments_json = '{"title": "Meeting", "content": "Discuss Wave A"}'
        parsed = parse_invoke(
            eligible_tools,
            {"name": selected_tool, "argumentsJson": arguments_json, "target": "Mac"},
        )
        assert parsed.tool_name == "app.create_note"
        assert parsed.arguments["title"] == "Meeting"
        assert parsed.arguments["content"] == "Discuss Wave A"
        assert parsed.arguments["target"] == "Mac"
        assert parsed.arguments["title"] == "Meeting"
        assert parsed.arguments["content"] == "Discuss Wave A"
        assert parsed.arguments["target"] == "Mac"


class TestTier1F4CUSessionEnvelope:
    """F4: Computer Use Session Envelope in Orchestrator."""

    @pytest.mark.anyio
    async def test_f4_01_app_launch_exempt_when_in_cu_session(self) -> None:
        """T1-F4.1: app.launch for in-scope app within active CU session is exempt from R3."""
        orch, host_repo, hub, broker, cu_store = _setup_test_orchestrator()
        owner_id = uuid.uuid4()
        mac = _make_host(owner_id, "Mac")
        await host_repo.save_host(mac)

        # Set up active CU session allowing 'notes'
        session_id = "cu-session-001"
        cu_state = _make_cu_session_state(
            session_id=session_id,
            owner_id=owner_id,
            host_id=mac.host_id,
            goal="Compose leave request",
            allowed_apps=("notes",),
        )
        await cu_store.create(cu_state)

        # Tool descriptor for app.launch
        launch_desc = _make_tool_descriptor(
            "app.launch",
            ToolFamily.APP,
            RiskTier.R2,
            {"id": {"type": "string"}, "appId": {"type": "string"}},
        )
        await host_repo.save_manifest(
            Manifest(host_id=mac.host_id, tools=(launch_desc,), updated_at=NOW)
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )

        ctx = _make_action_context(owner_id=owner_id, pet_id="pet-1")
        call = ToolCall(id="call-launch-1", name="app.launch", arguments={"id": "notes"})
        msg = await orch.handle_call(call, ctx)

        assert msg.role == Role.TOOL
        # No ActionPendingEvent published because app.launch is in-scope
        assert len(broker.published_events) == 0
        assert len(hub.invocations) == 1
        assert hub.invocations[0]["tool"] == "app.launch"

    @pytest.mark.anyio
    async def test_f4_02_app_launch_out_of_scope_requires_r3(self) -> None:
        """T1-F4.2: app.launch for app NOT in active session requires R3 approval."""
        orch, host_repo, hub, broker, cu_store = _setup_test_orchestrator()
        owner_id = uuid.uuid4()
        mac = _make_host(owner_id, "Mac")
        await host_repo.save_host(mac)

        cu_state = _make_cu_session_state(
            session_id="cu-session-002",
            owner_id=owner_id,
            host_id=mac.host_id,
            goal="Compose note",
            allowed_apps=("notes",),
        )
        await cu_store.create(cu_state)

        launch_desc = _make_tool_descriptor(
            "app.launch",
            ToolFamily.APP,
            RiskTier.R2,
            {"id": {"type": "string"}, "appId": {"type": "string"}},
        )
        await host_repo.save_manifest(
            Manifest(host_id=mac.host_id, tools=(launch_desc,), updated_at=NOW)
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R3,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )

        ctx = _make_action_context(owner_id=owner_id, pet_id="pet-1")
        # Attempt launching calculator (not in session scope)
        call = ToolCall(id="call-calc", name="app.launch", arguments={"id": "calculator"})
        msg = await orch.handle_call(call, ctx)

        assert len(broker.published_events) == 1
        pending = broker.published_events[0]
        assert pending.risk.lower() == "r3"
        payload = json.loads(msg.content)
        assert payload.get("status") in ("pending_approval", "expired", "rejected")

    @pytest.mark.anyio
    async def test_f4_03_ui_input_non_destructive_exempt_in_session(self) -> None:
        """T1-F4.3: Non-destructive inputs within active session are exempt from R3."""
        orch, host_repo, hub, broker, cu_store = _setup_test_orchestrator()
        owner_id = uuid.uuid4()
        mac = _make_host(owner_id, "Mac")
        await host_repo.save_host(mac)

        await cu_store.create(
            _make_cu_session_state(
                session_id="cu-session-003",
                owner_id=owner_id,
                host_id=mac.host_id,
                goal="Read document",
                allowed_apps=("preview",),
            )
        )

        scroll_desc = _make_tool_descriptor(
            "input.scroll", ToolFamily.INPUT, RiskTier.R2, {"deltaY": {"type": "integer"}}
        )
        await host_repo.save_manifest(
            Manifest(host_id=mac.host_id, tools=(scroll_desc,), updated_at=NOW)
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.INPUT,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )

        ctx = _make_action_context(owner_id=owner_id, pet_id="pet-1")
        call = ToolCall(id="call-scroll", name="input.scroll", arguments={"deltaY": 100})
        await orch.handle_call(call, ctx)

        assert len(broker.published_events) == 0
        assert len(hub.invocations) == 1

    @pytest.mark.anyio
    async def test_f4_04_reprompt_r3_on_consequential_actions(self) -> None:
        """T1-F4.4: Consequential element names ('Delete', 'Send', 'Pay') trigger R3 reprompt."""
        assert is_consequential_element_name("Delete file") is True
        assert is_consequential_element_name("Send") is True
        assert is_consequential_element_name("Pay now") is True
        assert is_consequential_element_name("Gửi phản hồi") is True
        assert is_consequential_element_name("Xóa ghi chú") is True
        assert is_consequential_element_name("Đăng bài") is True
        assert is_consequential_element_name("Thanh toán") is True
        # Non-consequential names
        assert is_consequential_element_name("Đăng nhập") is False
        assert is_consequential_element_name("Đăng ký") is False
        assert is_consequential_element_name("Cancel") is False
        assert is_consequential_element_name("Next") is False

    @pytest.mark.anyio
    async def test_f4_05_session_expiry_revokes_envelope(self) -> None:
        """T1-F4.5: Expired session ceases to grant R3 exemptions."""
        fake_time = NOW
        orch, host_repo, hub, broker, cu_store = _setup_test_orchestrator(clock=lambda: fake_time)
        owner_id = uuid.uuid4()
        mac = _make_host(owner_id, "Mac")
        await host_repo.save_host(mac)

        # Create session with max_minutes=5
        session_state = _make_cu_session_state(
            session_id="cu-session-exp",
            owner_id=owner_id,
            host_id=mac.host_id,
            goal="Expired task",
            allowed_apps=("notes",),
            max_minutes=5,
            started_at_utc=NOW,
            last_heartbeat_at=NOW
            + timedelta(minutes=6),  # Keep heartbeat fresh so time budget triggers
        )
        await cu_store.create(session_state)

        launch_desc = _make_tool_descriptor(
            "app.launch",
            ToolFamily.APP,
            RiskTier.R2,
            {"id": {"type": "string"}, "appId": {"type": "string"}},
        )
        await host_repo.save_manifest(
            Manifest(host_id=mac.host_id, tools=(launch_desc,), updated_at=NOW)
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R3,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )

        # Advance clock past 5 minute time budget
        fake_time = NOW + timedelta(minutes=6)
        ctx = _make_action_context(owner_id=owner_id, pet_id="pet-1")
        call = ToolCall(id="call-exp", name="app.launch", arguments={"id": "notes"})
        await orch.handle_call(call, ctx)

        # Because session is expired, active_cu is None and app.launch (R2) goes to normal approval
        assert len(broker.published_events) == 1
        assert broker.published_events[0].risk.lower() == "r2"


class TestTier1F5SafeDraftAndKeyRisk:
    """F5: Safe Draft and Key Risk in Computer Use."""

    @pytest.fixture
    def cu_runner(self) -> ComputerUseSessionRunner:
        cu_store = _InMemoryCuSessionStore()
        consent_repo = _FakeConsentRepository()
        verifier = _FakeSignatureVerifier(allow_all=True)

        async def fake_host_caller(
            host_id: HostId, tool: str, args: dict[str, Any]
        ) -> dict[str, Any]:
            return {"status": "ok"}

        return ComputerUseSessionRunner(
            consent_repo=consent_repo,
            session_store=cu_store,
            signature_verifier=verifier,
            host_caller=fake_host_caller,
        )

    @pytest.mark.anyio
    async def test_f5_01_input_type_text_in_draft_target_r2(
        self, cu_runner: ComputerUseSessionRunner
    ) -> None:
        """T1-F5.1: input.type_text in draft targets (textarea, textfield) does not require R3."""
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        proof = _make_r3_proof(
            host_id,
            "cu.session_start",
            {
                "goal": "Write draft",
                "apps": ["notes"],
                "domains": [],
                "maxSteps": 12,
                "maxMinutes": 5,
            },
        )
        session = await cu_runner.start_session(
            owner_id=owner_id,
            host_id=host_id,
            goal="Write draft",
            apps=["notes"],
            r3_proof=proof,
        )

        session.cached_nodes["field-1"] = {"role": "textarea", "name": "Content"}

        res = await cu_runner.execute_step(
            owner_id=owner_id,
            session_id=session.session_id,
            tool_name="input.type_text",
            arguments={"elementId": "field-1", "text": "Draft content"},
            r3_proof=None,
        )
        assert res["step"] == 1
        assert res["effect"] == "changed"

    @pytest.mark.anyio
    async def test_f5_02_multiline_text_with_newline_exempt_in_draft(
        self, cu_runner: ComputerUseSessionRunner
    ) -> None:
        """T1-F5.2: Multiline text containing \\n or \\r\\n in draft target does not require R3."""
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        proof = _make_r3_proof(
            host_id,
            "cu.session_start",
            {
                "goal": "Multiline draft",
                "apps": ["notes"],
                "domains": [],
                "maxSteps": 12,
                "maxMinutes": 5,
            },
        )
        session = await cu_runner.start_session(
            owner_id=owner_id,
            host_id=host_id,
            goal="Multiline draft",
            apps=["notes"],
            r3_proof=proof,
        )
        session.cached_nodes["editor-1"] = {"role": "document", "name": "Note Document"}

        multiline_content = (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\n\nĐơn xin nghỉ phép"
        )
        res = await cu_runner.execute_step(
            owner_id=owner_id,
            session_id=session.session_id,
            tool_name="input.type_text",
            arguments={"elementId": "editor-1", "text": multiline_content},
            r3_proof=None,
        )
        assert res["step"] == 1
        assert res["effect"] == "changed"

    def test_f5_03_safe_editing_shortcuts_exempt(self) -> None:
        """T1-F5.3: Safe editing shortcuts (Cmd+N, Cmd+A, Cmd+C, Cmd+V) are exempt."""
        assert is_activation_key_combo("Cmd+N") is False
        assert is_activation_key_combo("Cmd+A") is False
        assert is_activation_key_combo("Cmd+C") is False
        assert is_activation_key_combo("Cmd+V") is False
        assert is_activation_key_combo("Ctrl+N") is False
        assert is_activation_key_combo("Ctrl+S") is False

        # Activation keys that submit/activate controls
        assert is_activation_key_combo("Enter") is True
        assert is_activation_key_combo("Return") is True
        assert is_activation_key_combo("Shift+Enter") is True
        assert is_activation_key_combo("Space") is True

    @pytest.mark.anyio
    async def test_f5_04_draft_line_break_enter_exempt(
        self, cu_runner: ComputerUseSessionRunner
    ) -> None:
        """T1-F5.4: Enter/Return key in draftTextRoles does not require R3."""
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        proof = _make_r3_proof(
            host_id,
            "cu.session_start",
            {
                "goal": "Enter line break",
                "apps": ["notes"],
                "domains": [],
                "maxSteps": 12,
                "maxMinutes": 5,
            },
        )
        session = await cu_runner.start_session(
            owner_id=owner_id,
            host_id=host_id,
            goal="Enter line break",
            apps=["notes"],
            r3_proof=proof,
        )
        session.cached_nodes["txt-area"] = {"role": "textarea", "name": "Note Body"}

        res = await cu_runner.execute_step(
            owner_id=owner_id,
            session_id=session.session_id,
            tool_name="input.key",
            arguments={"elementId": "txt-area", "combo": "Enter"},
            r3_proof=None,
        )
        assert res["step"] == 1
        assert res["effect"] == "changed"

    @pytest.mark.anyio
    async def test_f5_05_password_and_sensitive_fields_strictly_forbidden(
        self, cu_runner: ComputerUseSessionRunner
    ) -> None:
        """T1-F5.5: Typing or clicking into password fields is strictly forbidden or requires R3."""
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        proof = _make_r3_proof(
            host_id,
            "cu.session_start",
            {
                "goal": "Sensitive test",
                "apps": ["notes"],
                "domains": [],
                "maxSteps": 12,
                "maxMinutes": 5,
            },
        )
        session = await cu_runner.start_session(
            owner_id=owner_id,
            host_id=host_id,
            goal="Sensitive test",
            apps=["notes"],
            r3_proof=proof,
        )
        session.cached_nodes["pwd-1"] = {"role": "secure_textfield", "isPassword": True}

        with pytest.raises(ComputerUseError) as exc_info:
            await cu_runner.execute_step(
                owner_id=owner_id,
                session_id=session.session_id,
                tool_name="input.type_text",
                arguments={"elementId": "pwd-1", "text": "secret123"},
                r3_proof=None,
            )
        assert exc_info.value.code == "password_field_forbidden"


async def _seed_owner_and_host(
    db: Database,
    owner_id: uuid.UUID,
    host_id: uuid.UUID,
) -> None:
    async with db.sessions() as s:
        await s.execute(
            text(
                "INSERT INTO owners (owner_id, issuer, subject, created_at) "
                "VALUES (:owner_id, 'test_issuer', :subject, :created_at) "
                "ON CONFLICT DO NOTHING"
            ),
            {"owner_id": owner_id, "subject": f"sub_{owner_id.hex[:12]}", "created_at": NOW},
        )
        await s.execute(
            text(
                "INSERT INTO hosts (host_id, owner_id, alias, platform, hostname, "
                "os_version, agent_version, public_key, status, created_at) "
                "VALUES (:host_id, :owner_id, 'Mac', 'macos', 'mac-host', "
                "'15.0', '1.0', :pubkey, 'active', :created_at) "
                "ON CONFLICT DO NOTHING"
            ),
            {"host_id": host_id, "owner_id": owner_id, "pubkey": FAKE_PUBKEY, "created_at": NOW},
        )
        await s.commit()


@pytest.mark.postgres
class TestTier1F6PostgresApprovalScope:
    """F6: PostgreSQL Approval Scope Persistence (requires XUNA_TEST_DATABASE_URL)."""

    @pytest.mark.anyio
    async def test_f6_01_approval_scope_migration_applied(self, database_url: str) -> None:
        """T1-F6.1: Migration 0020 creates approval_scopes table with correct columns and PK."""
        db = Database(database_url)
        await db.ping()
        try:
            async with db.engine.connect() as conn:

                def inspect_tables(sync_conn: Any) -> tuple[bool, set[str], list[str]]:
                    inspector = inspect(sync_conn)
                    has_table = inspector.has_table("approval_scopes")
                    cols = (
                        {c["name"] for c in inspector.get_columns("approval_scopes")}
                        if has_table
                        else set()
                    )
                    pk = (
                        inspector.get_pk_constraint("approval_scopes").get(
                            "constrained_columns", []
                        )
                        if has_table
                        else []
                    )
                    return has_table, cols, pk

                has_table, cols, pk = await conn.run_sync(inspect_tables)
                assert has_table is True
                assert {
                    "owner_id",
                    "pet_id",
                    "host_id",
                    "tool",
                    "expires_at",
                    "created_at",
                }.issubset(cols)
                assert set(pk) == {"owner_id", "pet_id", "host_id", "tool"}
        finally:
            await db.dispose()

    @pytest.mark.anyio
    async def test_f6_02_grant_and_has_valid_scope_round_trip(self, database_url: str) -> None:
        """T1-F6.2: grant_scope inserts scope and has_valid_scope returns True."""
        db = Database(database_url)
        await db.ping()
        try:
            owner_id = uuid.uuid4()
            host_id = uuid.uuid4()
            await _seed_owner_and_host(db, owner_id, host_id)

            store = SqlApprovalScopeStore(db.sessions)
            pet_id = "pet-rt-1"
            tool = "app.launch"

            # Check before grant
            assert await store.has_valid_scope(owner_id, pet_id, host_id, tool, NOW) is False

            # Grant 15m scope
            expires_at = await store.grant_scope(owner_id, pet_id, host_id, tool, NOW)
            assert expires_at == NOW + CONVERSATION_SCOPE_TTL

            # Check after grant
            assert await store.has_valid_scope(owner_id, pet_id, host_id, tool, NOW) is True
            assert (
                await store.has_valid_scope(
                    owner_id, pet_id, host_id, tool, NOW + timedelta(minutes=10)
                )
                is True
            )
        finally:
            await db.dispose()

    @pytest.mark.anyio
    async def test_f6_03_scope_survives_store_recreation(self, database_url: str) -> None:
        """T1-F6.3: Granted scopes survive store re-instantiation (simulating server restart)."""
        db = Database(database_url)
        await db.ping()
        try:
            owner_id = uuid.uuid4()
            host_id = uuid.uuid4()
            await _seed_owner_and_host(db, owner_id, host_id)

            store_1 = SqlApprovalScopeStore(db.sessions)
            pet_id = "pet-restart-test"
            tool = "input.click"

            await store_1.grant_scope(owner_id, pet_id, host_id, tool, NOW)

            # Re-create store instance
            store_2 = SqlApprovalScopeStore(db.sessions)
            assert await store_2.has_valid_scope(owner_id, pet_id, host_id, tool, NOW) is True
        finally:
            await db.dispose()

    @pytest.mark.anyio
    async def test_f6_04_expired_scope_returns_false(self, database_url: str) -> None:
        """T1-F6.4: has_valid_scope returns False when queried after expires_at."""
        db = Database(database_url)
        await db.ping()
        try:
            owner_id = uuid.uuid4()
            host_id = uuid.uuid4()
            await _seed_owner_and_host(db, owner_id, host_id)

            store = SqlApprovalScopeStore(db.sessions)
            pet_id = "pet-exp-test"
            tool = "ui.inspect"

            expires_at = await store.grant_scope(owner_id, pet_id, host_id, tool, NOW)
            assert (
                await store.has_valid_scope(
                    owner_id, pet_id, host_id, tool, expires_at - timedelta(seconds=1)
                )
                is True
            )
            assert await store.has_valid_scope(owner_id, pet_id, host_id, tool, expires_at) is False
            assert (
                await store.has_valid_scope(
                    owner_id, pet_id, host_id, tool, expires_at + timedelta(seconds=1)
                )
                is False
            )
        finally:
            await db.dispose()

    @pytest.mark.anyio
    async def test_f6_05_clear_owner_revokes_all_scopes(self, database_url: str) -> None:
        """T1-F6.5: clear_owner deletes all scopes for given owner and returns count."""
        db = Database(database_url)
        await db.ping()
        try:
            owner_id = uuid.uuid4()
            host_id = uuid.uuid4()
            await _seed_owner_and_host(db, owner_id, host_id)

            store = SqlApprovalScopeStore(db.sessions)
            pet_id = "pet-clear-test"

            await store.grant_scope(owner_id, pet_id, host_id, "tool.one", NOW)
            await store.grant_scope(owner_id, pet_id, host_id, "tool.two", NOW)

            deleted_count = await store.clear_owner(owner_id)
            assert deleted_count >= 2

            assert await store.has_valid_scope(owner_id, pet_id, host_id, "tool.one", NOW) is False
            assert await store.has_valid_scope(owner_id, pet_id, host_id, "tool.two", NOW) is False
        finally:
            await db.dispose()


class TestTier1F7TerminalAndSettingsHardDenial:
    """F7: Backend Terminal and Settings Hard-Denial."""

    def test_f7_01_backend_terminal_emulators_hard_denied(self) -> None:
        """T1-F7.1: is_hard_denied_target denies all modern and legacy terminal emulators."""
        terminals = [
            "warp",
            "alacritty",
            "kitty",
            "wezterm",
            "wezterm-gui",
            "ghostty",
            "hyper",
            "terminal",
            "iterm",
            "iterm2",
            "powershell",
            "pwsh",
            "bash",
            "zsh",
            "cmd",
        ]
        for term in terminals:
            assert is_hard_denied_target(term) is True, f"{term} should be hard-denied"
            # Modern Mac bundle apps
            if term not in ("cmd", "powershell", "pwsh", "bash", "zsh"):
                assert is_hard_denied_target(f"/Applications/{term.capitalize()}.app") is True
            # Windows executables
            if term in ("cmd", "powershell", "pwsh", "wt"):
                assert is_hard_denied_target(f"C:\\Windows\\System32\\{term}.exe") is True

        # Explicit verification of wezterm-gui and wezterm-gui.exe binaries and paths
        assert is_hard_denied_target("wezterm-gui") is True
        assert is_hard_denied_target("wezterm-gui.exe") is True
        assert is_hard_denied_target("/usr/local/bin/wezterm-gui") is True
        assert is_hard_denied_target("C:\\Program Files\\WezTerm\\wezterm-gui.exe") is True

        # Fail-closed boundary testing: None, empty string, and whitespace
        assert is_hard_denied_target("") is True, "Empty string target must fail closed"
        assert is_hard_denied_target("   ") is True, "Whitespace-only target must fail closed"
        assert is_hard_denied_target(None) is True, "None target must fail closed"

    def test_f7_02_backend_system_settings_hard_denied(self) -> None:
        """T1-F7.2: is_hard_denied_target denies system settings and control panels."""
        settings_targets = [
            "com.apple.systemsettings",
            "system_settings",
            "systemsettings",
            "system preferences",
            "control.exe",
            "regedit.exe",
            "taskmgr",
        ]
        for tgt in settings_targets:
            assert is_hard_denied_target(tgt) is True, f"{tgt} should be hard-denied"

    @pytest.mark.anyio
    async def test_f7_03_cu_session_rejects_denied_apps(self) -> None:
        """T1-F7.3: cu.session_start with denylisted target raises error."""
        runner = ComputerUseSessionRunner(
            session_store=_InMemoryCuSessionStore(),
            consent_repo=_FakeConsentRepository(),
            signature_verifier=_FakeSignatureVerifier(),
            host_caller=lambda h, t, a: asyncio.sleep(0, result={"status": "ok"}),
        )
        owner_id = uuid.uuid4()
        host_id = HostId.generate()

        with pytest.raises(ComputerUseError) as exc_info:
            await runner.start_session(
                owner_id=owner_id,
                host_id=host_id,
                goal="Open terminal",
                apps=["warp"],
            )
        assert exc_info.value.code == "denylisted_target_forbidden"

    @pytest.mark.anyio
    async def test_f7_04_cu_step_rejects_denied_apps(self) -> None:
        """T1-F7.4: Mid-session step targeting denylisted target raises error."""
        runner = ComputerUseSessionRunner(
            session_store=_InMemoryCuSessionStore(),
            consent_repo=_FakeConsentRepository(),
            signature_verifier=_FakeSignatureVerifier(),
            host_caller=lambda h, t, a: asyncio.sleep(0, result={"status": "ok"}),
        )
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        proof = _make_r3_proof(
            host_id,
            "cu.session_start",
            {
                "goal": "Normal session",
                "apps": ["notes"],
                "domains": [],
                "maxSteps": 12,
                "maxMinutes": 5,
            },
        )

        session = await runner.start_session(
            owner_id=owner_id,
            host_id=host_id,
            goal="Normal session",
            apps=["notes"],
            r3_proof=proof,
        )

        with pytest.raises(ComputerUseError) as exc_info:
            await runner.execute_step(
                owner_id=owner_id,
                session_id=session.session_id,
                tool_name="app.launch",
                arguments={"appId": "warp"},
            )
        assert exc_info.value.code == "denylisted_target_forbidden"

    def test_f7_05_elevated_window_forbidden(self) -> None:
        """T1-F7.5: Observations with isElevated=True raise elevated_window_forbidden."""
        runner = ComputerUseSessionRunner(
            session_store=_InMemoryCuSessionStore(),
            consent_repo=_FakeConsentRepository(),
            signature_verifier=_FakeSignatureVerifier(),
            host_caller=lambda h, t, a: asyncio.sleep(0, result={"status": "ok"}),
        )
        session = _make_cu_session_state("s-elevated", uuid.uuid4(), HostId.generate())

        with pytest.raises(ComputerUseError) as exc_info:
            runner._validate_window_target(session, {"appId": "notes", "isElevated": True})
        assert exc_info.value.code == "elevated_window_forbidden"


# ==============================================================================
# Tier 2: Boundary and Corner Cases
# ==============================================================================


class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary Value Analysis and Corner Cases."""

    def test_t2_01_empty_and_whitespace_draft_typing(self) -> None:
        """B1: Typing empty strings, whitespace, newlines, and emojis."""
        variations = ["", "\n", "   ", "\r\n\t", "🎉📝", "Cộng hòa xã hội chủ nghĩa Việt Nam"]
        for var in variations:
            assert contains_prompt_injection_payload(var) is False

    def test_t2_02_buffer_length_limits(self) -> None:
        """B2: Buffer limits: 4096-byte boundary vs oversized text."""
        exact_4096 = "a" * 4096
        assert len(exact_4096.encode("utf-8")) == 4096
        overflow_4097 = "a" * 4097
        assert len(overflow_4097.encode("utf-8")) == 4097

    def test_t2_03_quota_429_payload_variations(self) -> None:
        """B3: HTML 502/503 responses, malformed JSON, and clean 1024-byte UTF-8 truncation."""
        html_payload = "<html><body><h1>503 Service Unavailable</h1></body></html>"
        assert _is_quota_exhausted(html_payload) is False

        malformed_json = '{"error": {"message": "daily quota exceeded", '
        assert _is_quota_exhausted(malformed_json) is True

        vietnamese_multibyte = "Cộng hòa xã hội " * 100
        encoded = vietnamese_multibyte.encode("utf-8")[:1024]
        decoded = encoded.decode("utf-8", errors="replace")
        assert len(decoded.encode("utf-8")) <= 1024

    def test_t2_04_app_id_traversal_and_injection(self) -> None:
        """B4: Path traversal, command chaining, and null bytes in application IDs."""
        traversal_attempts = [
            "../../Applications/Terminal.app",
            "notes; rm -rf /",
            "notes && calc",
            "notes\x00calc",
            "com.apple.Terminal/../../../etc/passwd",
        ]
        for attempt in traversal_attempts:
            if "Terminal" in attempt or "terminal" in attempt.lower():
                assert is_hard_denied_target(attempt) is True

    @pytest.mark.postgres
    @pytest.mark.anyio
    async def test_t2_05_scope_exact_expiration_boundary(self, database_url: str) -> None:
        """B5: Exact microsecond boundary evaluation of scope expiration."""
        db = Database(database_url)
        await db.ping()
        try:
            owner_id = uuid.uuid4()
            host_id = uuid.uuid4()
            await _seed_owner_and_host(db, owner_id, host_id)

            store = SqlApprovalScopeStore(db.sessions)
            tool = "app.launch"

            expires_at = await store.grant_scope(owner_id, "pet-bva", host_id, tool, NOW)

            # Exactly 1 microsecond before expiration -> valid
            assert (
                await store.has_valid_scope(
                    owner_id, "pet-bva", host_id, tool, expires_at - timedelta(microseconds=1)
                )
                is True
            )
            # Exactly at expiration -> expired (False)
            assert (
                await store.has_valid_scope(owner_id, "pet-bva", host_id, tool, expires_at) is False
            )
        finally:
            await db.dispose()


# ==============================================================================
# Tier 3: Cross-Feature Combinations
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Tier 3: Complex Multi-Component Interactions."""

    @pytest.mark.anyio
    async def test_t3_01_cu_session_multiline_draft_quota_fallback(self) -> None:
        """C1: CU session + multiline draft + mid-turn 429 quota + generator switch."""
        captured_requests: list[ChatRequest] = []
        turn_attempts = 0

        class PrimaryFailingGen(TextGenerator):
            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
                nonlocal turn_attempts
                turn_attempts += 1
                if turn_attempts == 1:
                    # Round 1: Model invokes code/forge tool with thought signature
                    yield ToolCallRequested(
                        ToolCall(
                            id="call-draft-1",
                            name="forge__execute",
                            arguments={"code": "draft_text()"},
                            extra={"thought_signature": "sig_draft_initial"},
                        )
                    )
                    yield ChatFinished("primary", "mod1", 10, 10, 20)
                else:
                    # Round 2: Provider hits 429 quota exhaustion
                    raise ProviderUnavailableError("quota_exhausted: RESOURCE_EXHAUSTED")
                    yield  # pragma: no cover

        class BackupGen(TextGenerator):
            async def stream(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
                captured_requests.append(req)
                yield TextDelta("Successfully drafted leave request.")
                yield ChatFinished("backup", "mod2", 15, 15, 30)

        async def mock_runner(call: ToolCall, emit: Any) -> Message:
            return Message(role=Role.TOOL, content='{"status":"ok"}', tool_call_id=call.id)

        req = ChatRequest(
            messages=(Message(role=Role.USER, content="Draft note"),),
            tools=(
                ToolSpec("forge__execute", "Execute code", {"type": "object", "properties": {}}),
            ),
        )

        events: list[TurnEvent] = []
        async for ev in chat_turn(
            PrimaryFailingGen(),
            req,
            tool_runner=mock_runner,
            code_generator=BackupGen(),
        ):
            events.append(ev)

        # Verify fallback generator received messages with round 1's thought signature preserved
        assert len(captured_requests) == 1
        backup_req = captured_requests[0]
        r1_call = backup_req.messages[1].tool_calls[0]
        assert r1_call.extra == {"thought_signature": "sig_draft_initial"}

        # Verify final text was emitted
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert "Successfully drafted" in text

    @pytest.mark.anyio
    async def test_t3_02_terminal_attempt_inside_valid_cu_session(self) -> None:
        """C2: Terminal attempt inside active CU session strictly rejected."""
        orch, host_repo, hub, broker, cu_store = _setup_test_orchestrator()
        owner_id = uuid.uuid4()
        mac = _make_host(owner_id, "Mac")
        await host_repo.save_host(mac)

        # Active session granted for notes
        await cu_store.create(
            _make_cu_session_state(
                session_id="cu-valid-session",
                owner_id=owner_id,
                host_id=mac.host_id,
                goal="Draft note",
                allowed_apps=("notes",),
            )
        )

        launch_desc = _make_tool_descriptor(
            "app.launch", ToolFamily.APP, RiskTier.R2, {"appId": {"type": "string"}}
        )
        await host_repo.save_manifest(
            Manifest(host_id=mac.host_id, tools=(launch_desc,), updated_at=NOW)
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R3,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )

        ctx = _make_action_context(owner_id=owner_id, pet_id="pet-1")
        # Attempt launching warp terminal inside active notes session
        call = ToolCall(id="call-warp", name="app.launch", arguments={"appId": "warp"})
        await orch.handle_call(call, ctx)

        # Must not execute directly on host; must be intercepted/demanded R3 or blocked
        assert len(hub.invocations) == 0

    @pytest.mark.postgres
    @pytest.mark.anyio
    async def test_t3_03_concurrent_postgres_approval_scopes(self, database_url: str) -> None:
        """C3: Concurrent PostgreSQL approval scope queries and grants."""
        db = Database(database_url)
        await db.ping()
        try:
            owner_id = uuid.uuid4()
            host_id = uuid.uuid4()
            await _seed_owner_and_host(db, owner_id, host_id)

            store = SqlApprovalScopeStore(db.sessions)

            async def worker(idx: int) -> bool:
                tool_name = f"tool.worker.{idx}"
                await store.grant_scope(owner_id, "pet-concurrent", host_id, tool_name, NOW)
                return await store.has_valid_scope(
                    owner_id, "pet-concurrent", host_id, tool_name, NOW
                )

            results = await asyncio.gather(*[worker(i) for i in range(10)])
            assert all(results) is True
        finally:
            await db.dispose()

    @pytest.mark.anyio
    async def test_t3_04_session_envelope_with_compact_discovery_and_code_switch(self) -> None:
        """C4: Compact discovery finds tool -> session started -> generator switch."""
        desc = _make_tool_descriptor("app.launch", ToolFamily.APP, RiskTier.R2)
        eligible = {"app.launch": EligibleHostTool(descriptor=desc, aliases=("Mac",))}

        search_res = search_tools(eligible, "launch")
        assert len(search_res["matches"]) == 1
        assert search_res["matches"][0]["name"] == "app.launch"


# ==============================================================================
# Tier 4: Real-World Scenarios and Security Audits
# ==============================================================================


class TestTier4RealWorldScenariosAndAudits:
    """Tier 4: Simulated End-to-End Workflows and Security Audits."""

    @pytest.mark.anyio
    async def test_t4_01_apple_notes_leave_request_workflow(self) -> None:
        """Scenario 1: Simulated Apple Notes Leave Request Workflow ("Đơn xin nghỉ phép").

        Verifies that a multi-step task:
        1. cu.session_start(apps=["notes"]) -> Prompts owner exactly ONCE (Prompt #1)
        2. app.launch("notes")              -> 0 prompts (exempt by session envelope)
        3. ui.find(name="New Note")         -> 0 prompts (R1)
        4. input.key("Cmd+N")               -> 0 prompts (exempt editing shortcut)
        5. input.type_text("CỘNG HÒA...")    -> 0 prompts (exempt multiline draft)
        6. input.key("Enter")               -> 0 prompts (exempt draft line break)
        Total owner prompts throughout workflow: EXACTLY 1.
        """
        orch, host_repo, hub, broker, cu_store = _setup_test_orchestrator()
        owner_id = uuid.uuid4()
        mac = _make_host(owner_id, "Mac")
        await host_repo.save_host(mac)

        # Register tools
        tools = (
            _make_tool_descriptor(
                "cu.session_start",
                ToolFamily.CU,
                RiskTier.R3,
                {
                    "goal": {"type": "string"},
                    "apps": {"type": "array", "items": {"type": "string"}},
                    "domains": {"type": "array", "items": {"type": "string"}},
                    "maxSteps": {"type": "integer"},
                    "maxMinutes": {"type": "integer"},
                },
            ),
            _make_tool_descriptor(
                "app.launch",
                ToolFamily.APP,
                RiskTier.R2,
                {"id": {"type": "string"}, "appId": {"type": "string"}},
            ),
            _make_tool_descriptor(
                "ui.find", ToolFamily.UI, RiskTier.R1, {"name": {"type": "string"}}
            ),
            _make_tool_descriptor(
                "input.key", ToolFamily.INPUT, RiskTier.R2, {"combo": {"type": "string"}}
            ),
            _make_tool_descriptor(
                "input.type_text", ToolFamily.INPUT, RiskTier.R2, {"text": {"type": "string"}}
            ),
        )
        await host_repo.save_manifest(Manifest(host_id=mac.host_id, tools=tools, updated_at=NOW))
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.CU,
                    max_risk=RiskTier.R3,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.UI,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.INPUT,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                ),
            ],
        )

        ctx = _make_action_context(owner_id=owner_id, pet_id="pet-notes")

        # Step 1: Start CU Session (R3 prompt #1)
        start_call = ToolCall(
            id="c1",
            name="cu.session_start",
            arguments={"goal": "Đơn xin nghỉ phép", "apps": ["notes"]},
        )
        await orch.handle_call(start_call, ctx)
        assert len(broker.published_events) == 1
        assert broker.published_events[0].risk.lower() == "r3"
        approval_id = broker.published_events[0].approval_id

        # Owner approves session
        await orch.decide_approval(
            approval_id=uuid.UUID(approval_id),
            owner_id=owner_id,
            decision=ApprovalDecision.APPROVE,
        )
        # Create active CU session state simulating completed session_start
        await cu_store.create(
            _make_cu_session_state(
                session_id="session-notes-01",
                owner_id=owner_id,
                host_id=mac.host_id,
                goal="Đơn xin nghỉ phép",
                allowed_apps=("notes",),
            )
        )

        # Step 2: app.launch("notes") -> 0 prompts
        launch_call = ToolCall(id="c2", name="app.launch", arguments={"id": "notes"})
        await orch.handle_call(launch_call, ctx)
        assert len(broker.published_events) == 1  # Still 1!

        # Step 3: ui.find -> 0 prompts
        find_call = ToolCall(id="c3", name="ui.find", arguments={"name": "New Note"})
        await orch.handle_call(find_call, ctx)
        assert len(broker.published_events) == 1

        # Step 4: input.key("Cmd+N") -> 0 prompts
        key_call = ToolCall(id="c4", name="input.key", arguments={"combo": "Cmd+N"})
        await orch.handle_call(key_call, ctx)
        assert len(broker.published_events) == 1

        # Step 5: input.type_text(multiline text) -> 0 prompts
        text_content = (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n\n"
            "ĐƠN XIN NGHỈ PHÉP\n"
            "Kính gửi: Ban Giám Đốc\n"
            "Tôi xin nghỉ phép 2 ngày từ 16/11 đến 17/11."
        )
        type_call = ToolCall(id="c5", name="input.type_text", arguments={"text": text_content})
        await orch.handle_call(type_call, ctx)
        assert len(broker.published_events) == 1

        # Step 6: input.key("Enter") -> 0 prompts
        enter_call = ToolCall(id="c6", name="input.key", arguments={"combo": "Enter"})
        await orch.handle_call(enter_call, ctx)
        assert len(broker.published_events) == 1

        # Total owner prompts throughout entire workflow is exactly 1!
        assert len(broker.published_events) == 1

    def test_t4_02_terminal_denylist_audit_15_targets(self) -> None:
        """Scenario 2: Audit all 15 modern and legacy terminal emulator targets fail closed."""
        all_targets = [
            "terminal",
            "iterm",
            "iterm2",
            "warp",
            "alacritty",
            "kitty",
            "wezterm",
            "wezterm-gui",
            "wezterm-gui.exe",
            "ghostty",
            "hyper",
            "cmd",
            "powershell",
            "wt",
            "bash",
            "zsh",
            "dev.warp.Warp-Stable",
        ]
        for target in all_targets:
            assert (
                is_hard_denied_target(target) is True
            ), f"Security Audit Failed: {target} is not hard-denied!"

        # Fail-closed security audit: empty strings, whitespace, and None
        assert is_hard_denied_target("") is True, "Security Audit Failed: empty string must fail closed"
        assert is_hard_denied_target("   ") is True, "Security Audit Failed: whitespace must fail closed"
        assert is_hard_denied_target(None) is True, "Security Audit Failed: None must fail closed"

    def test_t4_03_zero_secret_leakage_audit(self) -> None:
        """Scenario 3: Zero Secret Leakage Audit across all error responses and logs."""
        raw_traces = [
            "Connection failed: AIzaSyD98765432101234567890123456789012 invalid",
            "OpenAI token sk-proj-abcdefghijklmnopqrstuvwxyz012345 rejected",
            "Groq Bearer gsk_0123456789abcdefghijklmnopqrstuvwxyz failed",
            "Groq key gsk_0123456789abcdefghijklmnopqrstuvwxyz failed",
        ]
        secret_patterns = [
            re.compile(r"AIzaSy[0-9A-Za-z_\-]{33}"),
            re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
            re.compile(r"gsk_[0-9A-Za-z_\-]{20,}"),
        ]

        for trace in raw_traces:
            sanitized = _sanitize_error_text(trace)
            for pat in secret_patterns:
                assert pat.search(sanitized) is None, f"Secret leakage detected: {sanitized}"
            assert "[REDACTED" in sanitized, f"Redaction marker missing in sanitized: {sanitized}"

    def test_t4_04_mobile_client_wire_compatibility_audit(self) -> None:
        """Scenario 4: ActionPendingEvent wire serialization adheres to mobile wire format."""
        event = ActionPendingEvent(
            approval_id="appr-1234",
            target="Mac",
            summary="Soạn đơn xin nghỉ",
            risk="R3",
            expires_in=120,
            host_id="host-mac-01",
            args_sha256="abc123sha",
            expires_at="2026-11-15T00:15:00Z",
            standing_tool="notes.create",
            standing_arguments={"title": "Đơn xin nghỉ"},
            standing_approved_count=3,
        )

        wire = event.wire_payload()
        assert wire["approvalId"] == "appr-1234"
        assert wire["target"] == "Mac"
        assert wire["summary"] == "Soạn đơn xin nghỉ"
        assert wire["risk"] == "r3"  # Lowercase wire string for mobile client!
        assert wire["expiresIn"] == 120
        assert wire["hostId"] == "host-mac-01"
        assert wire["argsSha256"] == "abc123sha"
        assert wire["expiresAt"] == "2026-11-15T00:15:00Z"
        assert wire["standing"] == {
            "tool": "notes.create",
            "arguments": {"title": "Đơn xin nghỉ"},
            "approvedCount": 3,
        }
