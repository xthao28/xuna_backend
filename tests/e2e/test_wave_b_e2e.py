"""Wave B End-to-End Test Suite: Native OS App Bridge & Browser Engine Integration.

Covers Tiers 1-4 for Wave B of the XUNA Clawbot upgrade:
- Tier 1: Feature Coverage (F1 to F16 across Native App Bridge & Browser Engine)
- Tier 2: Boundary, Error & Security Attack Cases (Anti-SSRF Matrix, R3, ARIA)
- Tier 3: Cross-Feature Combinations & State Interactions (Chains, 15m Scopes, CU Envelopes)
- Tier 4: Real-World Scenarios S1-S3, Mobile Wire Schema Compatibility, Rule H5 Generic Layer Scan
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from xuna_backend.modules.actions.application.orchestrator import (
    CAP_HOST_ACTIONS_V1,
    ActionContext,
    Orchestrator,
)
from xuna_backend.modules.actions.application.ports import (
    ComputerUseSessionStore,
)
from xuna_backend.modules.actions.domain.entities import (
    MAX_TOOL_OUTPUT_BYTES,
    ActionPendingEvent,
    ActionResultEvent,
    ApprovalDecision,
    ApprovalScope,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import (
    InMemoryApprovalRepository,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditEvent
from xuna_backend.modules.cognition.domain.types import Locale, Role, ToolCall
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
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

# ==============================================================================
# Shared Constants and Test Doubles
# ==============================================================================

NOW = datetime(2026, 11, 15, 0, 0, 0, tzinfo=UTC)
FAKE_PUBKEY = b"\x04" + bytes(range(1, 65))


def _make_host(
    owner_id: uuid.UUID,
    alias: str = "Mac Studio",
    *,
    host_id: HostId | None = None,
    platform: HostPlatform = HostPlatform.MACOS,
    status: HostStatus = HostStatus.ACTIVE,
) -> Host:
    h_id = host_id if host_id is not None else HostId.generate()
    return Host(
        host_id=h_id,
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname="xuna-studio-host",
        os_version="15.0",
        agent_version="0.2.0",
        public_key=FAKE_PUBKEY,
        status=status,
        created_at=NOW,
    )


def _make_descriptor(
    name: str,
    family: ToolFamily,
    risk: RiskTier,
    *,
    properties: dict[str, object] | None = None,
    required: list[str] | None = None,
) -> ToolDescriptor:
    props: dict[str, object] = dict(properties) if properties is not None else {}
    req: list[str] = list(required) if required is not None else []
    return ToolDescriptor(
        name=name,
        description=f"Tool {name}",
        input_schema={
            "type": "object",
            "properties": props,
            "required": req,
            "additionalProperties": False,
        },
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=family,
        risk=risk,
        permission=ToolPermissionState.GRANTED,
        policy_version=1,
    )


def _build_all_wave_b_descriptors() -> list[ToolDescriptor]:
    return [
        _make_descriptor(
            "notes.create",
            ToolFamily.APP,
            RiskTier.R2,
            properties={
                "title": {"type": "string"},
                "body": {"type": "string"},
                "folder": {"type": "string"},
            },
            required=["title", "body"],
        ),
        _make_descriptor(
            "notes.append",
            ToolFamily.APP,
            RiskTier.R2,
            properties={
                "id": {"type": "string"},
                "title": {"type": "string"},
                "text": {"type": "string"},
            },
            required=["text"],
        ),
        _make_descriptor(
            "notes.search",
            ToolFamily.APP,
            RiskTier.R1,
            properties={"query": {"type": "string"}, "maxResults": {"type": "integer"}},
            required=["query"],
        ),
        _make_descriptor(
            "reminders.create",
            ToolFamily.APP,
            RiskTier.R2,
            properties={
                "title": {"type": "string"},
                "list": {"type": "string"},
                "dueDate": {"type": "string"},
                "notes": {"type": "string"},
                "priority": {"type": "integer"},
            },
            required=["title"],
        ),
        _make_descriptor(
            "calendar.list_events",
            ToolFamily.APP,
            RiskTier.R1,
            properties={
                "timeMin": {"type": "string"},
                "timeMax": {"type": "string"},
                "calendarName": {"type": "string"},
                "maxResults": {"type": "integer"},
            },
            required=["timeMin", "timeMax"],
        ),
        _make_descriptor(
            "calendar.create_event",
            ToolFamily.APP,
            RiskTier.R2,
            properties={
                "title": {"type": "string"},
                "startTime": {"type": "string"},
                "endTime": {"type": "string"},
                "location": {"type": "string"},
                "notes": {"type": "string"},
                "calendarName": {"type": "string"},
            },
            required=["title", "startTime", "endTime"],
        ),
        _make_descriptor(
            "finder.reveal",
            ToolFamily.FILES,
            RiskTier.R1,
            properties={"path": {"type": "string"}},
            required=["path"],
        ),
        _make_descriptor(
            "shortcuts.run",
            ToolFamily.WORKFLOW,
            RiskTier.R2,
            properties={"shortcutName": {"type": "string"}, "input": {"type": "string"}},
            required=["shortcutName"],
        ),
        _make_descriptor(
            "mail.create_draft",
            ToolFamily.APP,
            RiskTier.R2,
            properties={
                "to": {"type": "array", "items": {"type": "string"}},
                "cc": {"type": "array", "items": {"type": "string"}},
                "bcc": {"type": "array", "items": {"type": "string"}},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            required=["to", "subject", "body"],
        ),
        _make_descriptor(
            "mail.send",
            ToolFamily.APP,
            RiskTier.R3,
            properties={
                "to": {"type": "array", "items": {"type": "string"}},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            required=["to", "subject", "body"],
        ),
        _make_descriptor(
            "web.navigate",
            ToolFamily.URL,
            RiskTier.R2,
            properties={"url": {"type": "string"}, "timeoutMs": {"type": "integer"}},
            required=["url"],
        ),
        _make_descriptor(
            "web.snapshot",
            ToolFamily.URL,
            RiskTier.R1,
            properties={"compact": {"type": "boolean"}, "maxBytes": {"type": "integer"}},
        ),
        _make_descriptor(
            "web.click",
            ToolFamily.URL,
            RiskTier.R2,
            properties={
                "ref": {"type": "string"},
                "selector": {"type": "string"},
                "targetText": {"type": "string"},
                "elementName": {"type": "string"},
            },
            required=["ref"],
        ),
        _make_descriptor(
            "web.fill",
            ToolFamily.URL,
            RiskTier.R2,
            properties={
                "ref": {"type": "string"},
                "value": {"type": "string"},
                "selector": {"type": "string"},
                "targetText": {"type": "string"},
            },
            required=["ref", "value"],
        ),
        _make_descriptor(
            "web.select",
            ToolFamily.URL,
            RiskTier.R2,
            properties={
                "ref": {"type": "string"},
                "value": {"type": "string"},
                "selector": {"type": "string"},
            },
            required=["ref", "value"],
        ),
        _make_descriptor(
            "web.extract",
            ToolFamily.URL,
            RiskTier.R1,
            properties={
                "ref": {"type": "string"},
                "selector": {"type": "string"},
                "attribute": {"type": "string"},
            },
        ),
        _make_descriptor(
            "web.wait_for",
            ToolFamily.URL,
            RiskTier.R1,
            properties={
                "selector": {"type": "string"},
                "state": {"type": "string"},
                "timeoutMs": {"type": "integer"},
            },
        ),
    ]


class _FakeHostRepository(HostRepository):
    """Complete implementation of HostRepository protocol."""

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

    async def suspend_r2_plus_grants_for_owner(self, owner_id: uuid.UUID, now: datetime) -> int:
        count = 0
        for host in self.hosts.values():
            if host.owner_id == owner_id:
                old = self.grants.get(host.host_id, [])
                kept = [g for g in old if g.max_risk == RiskTier.R1]
                count += len(old) - len(kept)
                self.grants[host.host_id] = kept
        return count


class _FakeSessionHub(HostSessionHub):
    """Complete implementation of HostSessionHub protocol."""

    def __init__(self) -> None:
        self.sent_requests: list[tuple[str, str, dict[str, object], float]] = []
        self.invocations: list[dict[str, object]] = []
        self.custom_responses: dict[str, dict[str, object]] = {}
        self.online_hosts: set[str] = set()

    def register_pending(
        self,
        pending_id: str,
        on_enrolled: Callable[[HostId, HostAlias], Awaitable[None]],
    ) -> None:
        pass

    def unregister_pending(self, pending_id: str) -> None:
        pass

    async def notify_enrolled(self, pending_id: str, host_id: HostId, alias: HostAlias) -> None:
        pass

    async def open_session(
        self,
        host_id: str,
        token: object,
        send_text: Callable[[str], Awaitable[None]],
        close_duplicate: Callable[[], Awaitable[None]],
        close_revoked: Callable[[], Awaitable[None]],
    ) -> bool:
        self.online_hosts.add(host_id)
        return True

    def close_session(self, host_id: str, token: object) -> bool:
        self.online_hosts.discard(host_id)
        return True

    def on_response(self, host_id: str, request_id: int | str, result: dict[str, object]) -> None:
        pass

    def on_error_response(
        self, host_id: str, request_id: int | str, code: int, message: str
    ) -> None:
        pass

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
        arguments = params.get("arguments")
        args_dict: dict[str, object] = dict(arguments) if isinstance(arguments, dict) else {}
        self.invocations.append({"host_id": host_id, "tool": tool_name, "arguments": args_dict})

        if tool_name in self.custom_responses:
            return self.custom_responses[tool_name]

        # Generate genuine structured content based on tool
        sc: dict[str, object] = {"status": "ok"}
        if tool_name == "notes.create":
            sc = {
                "id": "note-101",
                "title": str(args_dict.get("title", "")),
                "folder": str(args_dict.get("folder", "Notes")),
            }
        elif tool_name == "notes.append":
            sc = {
                "id": str(args_dict.get("id", "note-101")),
                "title": str(args_dict.get("title", "")),
                "appended": True,
            }
        elif tool_name == "notes.search":
            sc = {
                "notes": [{"id": "note-101", "title": "Sprint Plan", "snippet": "Deliver Wave B"}],
                "total": 1,
            }
        elif tool_name == "reminders.create":
            sc = {
                "id": "rem-202",
                "title": str(args_dict.get("title", "")),
                "list": str(args_dict.get("list", "Reminders")),
                "dueDate": str(args_dict.get("dueDate", "")),
            }
        elif tool_name == "calendar.list_events":
            sc = {
                "events": [
                    {
                        "id": "evt-303",
                        "title": "Wave B Sync",
                        "startTime": str(args_dict.get("timeMin", "")),
                        "endTime": str(args_dict.get("timeMax", "")),
                    }
                ],
                "total": 1,
            }
        elif tool_name == "calendar.create_event":
            sc = {
                "id": "evt-303",
                "title": str(args_dict.get("title", "")),
                "startTime": str(args_dict.get("startTime", "")),
                "endTime": str(args_dict.get("endTime", "")),
            }
        elif tool_name == "finder.reveal":
            sc = {"path": str(args_dict.get("path", "")), "revealed": True}
        elif tool_name == "shortcuts.run":
            sc = {
                "name": str(args_dict.get("shortcutName", "")),
                "status": "ok",
                "output": "Shortcut executed",
            }
        elif tool_name == "mail.create_draft":
            sc = {
                "id": "draft-505",
                "subject": str(args_dict.get("subject", "")),
                "to": args_dict.get("to", []),
                "isDraft": True,
            }
        elif tool_name == "mail.send":
            sc = {"sent": True, "subject": str(args_dict.get("subject", ""))}
        elif tool_name == "web.navigate":
            sc = {
                "url": str(args_dict.get("url", "")),
                "title": "XUNA Docs",
                "status": 200,
                "navigated": True,
                "loaded": True,
            }
        elif tool_name == "web.snapshot":
            sc = {
                "url": "https://docs.xuna.ai",
                "snapshotId": "snap-606",
                "compactAriaTree": "- button 'Submit' [ref=e1]\n- textbox 'Query' [ref=e2]",
                "elementCount": 2,
                "byteCount": 60,
                "truncated": False,
            }
        elif tool_name == "web.click":
            sc = {"ref": str(args_dict.get("ref", "e1")), "clicked": True, "action": "click"}
        elif tool_name == "web.fill":
            sc = {
                "ref": str(args_dict.get("ref", "e1")),
                "filled": True,
                "valueLength": len(str(args_dict.get("value", ""))),
            }
        elif tool_name == "web.select":
            sc = {
                "ref": str(args_dict.get("ref", "e1")),
                "selected": True,
                "value": str(args_dict.get("value", "")),
            }
        elif tool_name == "web.extract":
            sc = {
                "ref": str(args_dict.get("ref", "")),
                "content": "Extracted document body",
                "extracted": True,
            }
        elif tool_name == "web.wait_for":
            sc = {
                "selector": str(args_dict.get("selector", "")),
                "waited": True,
                "satisfied": True,
                "elapsedMs": 40,
            }

        return {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": sc,
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 25},
        }

    async def notify(self, host_id: str, method: str, params: dict[str, object]) -> None:
        pass

    async def stop_host(self, host_id: str, reason: str = "owner_app") -> bool:
        return True

    async def disconnect_host(self, host_id: str, reason: str = "revoked") -> None:
        self.online_hosts.discard(host_id)


class _FakeAuditLog(AuditLog):
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)

    async def recent(self, owner_id: uuid.UUID, limit: int) -> list[AuditEvent]:
        return [e for e in self.events if e.owner_id == owner_id][-limit:]


class EventCollector:
    """Helper to collect emitted events and optionally auto-decide approvals."""

    def __init__(
        self,
        orch: Orchestrator,
        owner_id: uuid.UUID,
        *,
        auto_decision: ApprovalDecision | None = ApprovalDecision.APPROVE,
        scope: ApprovalScope = ApprovalScope.ONCE,
    ) -> None:
        self.orch = orch
        self.owner_id = owner_id
        self.auto_decision = auto_decision
        self.scope = scope
        self.pending_events: list[ActionPendingEvent] = []
        self.result_events: list[ActionResultEvent] = []
        self.all_events: list[object] = []

    async def emit(self, ev: object) -> None:
        self.all_events.append(ev)
        if isinstance(ev, ActionPendingEvent):
            self.pending_events.append(ev)
            if self.auto_decision is not None:
                await self.orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=self.owner_id,
                    decision=self.auto_decision,
                    scope=self.scope,
                )
        elif isinstance(ev, ActionResultEvent):
            self.result_events.append(ev)


def _setup_test_orchestrator(
    owner_id: uuid.UUID,
    *,
    clock_dt: datetime | None = None,
    cu_store: ComputerUseSessionStore | None = None,
) -> tuple[
    Orchestrator,
    _FakeHostRepository,
    _FakeSessionHub,
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    _FakeAuditLog,
]:
    current_now = clock_dt if clock_dt is not None else NOW
    host_repo = _FakeHostRepository()
    hub = _FakeSessionHub()
    approval_repo = InMemoryApprovalRepository()
    approval_broker = InMemoryApprovalBroker()
    scope_store = InMemoryApprovalScopeStore()
    last_target = InMemoryLastTargetStore()
    audit = _FakeAuditLog()

    host = _make_host(owner_id)
    descriptors = _build_all_wave_b_descriptors()
    manifest = Manifest(
        host_id=host.host_id,
        tools=tuple(descriptors),
        updated_at=current_now,
        schema_version=1,
    )
    grants = [
        Grant(
            host_id=host.host_id,
            family=ToolFamily.APP,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=current_now,
        ),
        Grant(
            host_id=host.host_id,
            family=ToolFamily.FILES,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=current_now,
        ),
        Grant(
            host_id=host.host_id,
            family=ToolFamily.WORKFLOW,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=current_now,
        ),
        Grant(
            host_id=host.host_id,
            family=ToolFamily.URL,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=current_now,
        ),
        Grant(
            host_id=host.host_id,
            family=ToolFamily.CU,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=current_now,
        ),
        Grant(
            host_id=host.host_id,
            family=ToolFamily.SCREEN,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=current_now,
        ),
        Grant(
            host_id=host.host_id,
            family=ToolFamily.UI,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=current_now,
        ),
        Grant(
            host_id=host.host_id,
            family=ToolFamily.INPUT,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=current_now,
        ),
    ]

    # Pre-populate host
    host_repo.hosts[host.host_id] = host
    host_repo.manifests[host.host_id] = manifest
    host_repo.grants[host.host_id] = list(grants)

    orch = Orchestrator(
        host_repo=host_repo,
        hub=hub,
        approval_repo=approval_repo,
        approval_broker=approval_broker,
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=last_target,
        audit=audit,
        scope_store=scope_store,
        cu_session_store=cu_store,
        clock=lambda: current_now,
    )
    return orch, host_repo, hub, approval_broker, scope_store, audit


def _make_action_context(
    owner_id: uuid.UUID,
    collector: EventCollector,
    *,
    pet_id: str = "pet-1",
    locale: Locale = Locale.VI,
) -> ActionContext:
    return ActionContext(
        owner_id=owner_id,
        pet_id=pet_id,
        locale=locale,
        client_caps=(CAP_HOST_ACTIONS_V1,),
        board_available=False,
        emit_event=collector.emit,
    )


# ==============================================================================
# Tier 1: Feature Coverage (F1 to F16 across Native App Bridge & Browser Engine)
# ==============================================================================


class TestTier1WaveBNativeFeatures:
    """F1 - F9: Native OS App Bridge E2E Dispatch & Event Serialization."""

    @pytest.mark.anyio
    async def test_f1_notes_create_approval_and_dispatch(self) -> None:
        """F1: notes.create prompts at R2, owner approves, dispatches over XTP."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c1",
            name="notes__create",
            arguments={"title": "Sprint 42", "body": "Finish M3 tests"},
        )
        msg = await orch.handle_call(call, ctx)

        assert msg.role == Role.TOOL
        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["id"] == "note-101"
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"
        assert "Tạo ghi chú 'Sprint 42'" in collector.pending_events[0].summary
        assert len(collector.result_events) == 1
        assert collector.result_events[0].status == "ok"
        assert len(hub.invocations) == 1
        assert hub.invocations[0]["tool"] == "notes.create"

    @pytest.mark.anyio
    async def test_f2_notes_append_approval_and_dispatch(self) -> None:
        """F2: notes.append prompts at R2, owner approves, appends to existing note."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c2",
            name="notes__append",
            arguments={"id": "note-101", "text": "Appended verification steps"},
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["appended"] is True
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"
        assert len(hub.invocations) == 1
        assert hub.invocations[0]["tool"] == "notes.append"

    @pytest.mark.anyio
    async def test_f3_notes_search_read_lane_zero_prompt(self) -> None:
        """F3: notes.search is R1 (Read-Lane), dispatches immediately with ZERO owner prompts."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(id="c3", name="notes__search", arguments={"query": "Sprint"})
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["total"] == 1
        assert len(collector.pending_events) == 0  # Read-lane: 0 prompts!
        assert len(hub.invocations) == 1
        assert hub.invocations[0]["tool"] == "notes.search"

    @pytest.mark.anyio
    async def test_f4_reminders_create_approval_and_dispatch(self) -> None:
        """F4: reminders.create prompts at R2, owner approves, creates reminder."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c4",
            name="reminders__create",
            arguments={"title": "Review PR", "dueDate": "2026-11-15T18:00:00Z", "priority": 1},
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["id"] == "rem-202"
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"
        assert len(hub.invocations) == 1
        assert hub.invocations[0]["tool"] == "reminders.create"

    @pytest.mark.anyio
    async def test_f5_calendar_list_events_read_lane(self) -> None:
        """F5: calendar.list_events is R1, dispatches with ZERO owner prompts."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c5",
            name="calendar__list_events",
            arguments={"timeMin": "2026-11-15T00:00:00Z", "timeMax": "2026-11-15T23:59:59Z"},
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["total"] == 1
        assert len(collector.pending_events) == 0  # Zero prompts!
        assert len(hub.invocations) == 1
        assert hub.invocations[0]["tool"] == "calendar.list_events"

    @pytest.mark.anyio
    async def test_f6_calendar_create_event_approval_and_dispatch(self) -> None:
        """F6: calendar.create_event prompts at R2, owner approves, creates calendar event."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c6",
            name="calendar__create_event",
            arguments={
                "title": "Sprint Review",
                "startTime": "2026-11-15T14:00:00Z",
                "endTime": "2026-11-15T15:00:00Z",
            },
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["id"] == "evt-303"
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"
        assert len(hub.invocations) == 1
        assert hub.invocations[0]["tool"] == "calendar.create_event"

    @pytest.mark.anyio
    async def test_f7_finder_reveal_read_lane(self) -> None:
        """F7: finder.reveal is R1, reveals file path with ZERO owner prompts."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c7", name="finder__reveal", arguments={"path": "/Users/thao/Documents/report.pdf"}
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["revealed"] is True
        assert len(collector.pending_events) == 0  # Zero prompts!
        assert len(hub.invocations) == 1
        assert hub.invocations[0]["tool"] == "finder.reveal"

    @pytest.mark.anyio
    async def test_f8_shortcuts_run_approval_and_dispatch(self) -> None:
        """F8: shortcuts.run prompts at R2, owner approves, executes shortcut."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c8", name="shortcuts__run", arguments={"shortcutName": "Daily Standup Brief"}
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["status"] == "ok"
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"
        assert len(hub.invocations) == 1
        assert hub.invocations[0]["tool"] == "shortcuts.run"

    @pytest.mark.anyio
    async def test_f9_mail_create_draft_r2_vs_mail_send_r3(self) -> None:
        """F9: mail.create_draft is R2 (draft-only); mail.send strictly forces R3 risk floor."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # 1. mail.create_draft -> R2
        draft_call = ToolCall(
            id="c9_draft",
            name="mail__create_draft",
            arguments={"to": ["team@xuna.ai"], "subject": "Wave B Report", "body": "Draft content"},
        )
        msg_draft = await orch.handle_call(draft_call, ctx)
        data_draft = json.loads(msg_draft.content)
        assert data_draft["status"] == "ok"
        assert data_draft["output"]["isDraft"] is True
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"

        # 2. mail.send -> Strictly R3 floor
        send_call = ToolCall(
            id="c9_send",
            name="mail__send",
            arguments={"to": ["team@xuna.ai"], "subject": "Final Send", "body": "Outbound email"},
        )
        msg_send = await orch.handle_call(send_call, ctx)
        data_send = json.loads(msg_send.content)
        assert data_send["status"] == "ok"
        assert len(collector.pending_events) == 2
        assert collector.pending_events[1].risk.lower() == "r3"


class TestTier1WaveBBrowserFeatures:
    """F10 - F16: Browser Engine (web.*) E2E Dispatch & Validation."""

    @pytest.mark.anyio
    async def test_f10_web_navigate_valid_https_public_url(self) -> None:
        """F10: web.navigate validates public HTTPS URL, prompts at R2, navigates."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c10", name="web__navigate", arguments={"url": "https://docs.xuna.ai/guide"}
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["navigated"] is True
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"
        assert "docs.xuna.ai" in collector.pending_events[0].summary
        assert len(hub.invocations) == 1
        assert hub.invocations[0]["tool"] == "web.navigate"

    @pytest.mark.anyio
    async def test_f11_web_snapshot_compact_aria_limiting(self) -> None:
        """F11: web.snapshot is R1 read-lane, returns compact ARIA bounded strictly <= 6144 B."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(id="c11", name="web__snapshot", arguments={"compact": True})
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert "compactAriaTree" in data["output"]
        assert len(collector.pending_events) == 0  # Read-lane: zero prompts
        assert len(msg.content.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES

    @pytest.mark.anyio
    async def test_f12_web_click_benign_r2(self) -> None:
        """F12: web.click on benign element remains R2, executes safely."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c12", name="web__click", arguments={"ref": "e1", "targetText": "Next Page"}
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["clicked"] is True
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"

    @pytest.mark.anyio
    async def test_f13_web_fill_benign_r2(self) -> None:
        """F13: web.fill with benign query remains R2."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c13",
            name="web__fill",
            arguments={"ref": "search-input", "value": "quantum physics"},
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["filled"] is True
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"

    @pytest.mark.anyio
    async def test_f14_web_select_dropdown_r2(self) -> None:
        """F14: web.select prompts at R2, selects dropdown option."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c14", name="web__select", arguments={"ref": "lang-picker", "value": "vi"}
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["selected"] is True
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"

    @pytest.mark.anyio
    async def test_f15_web_extract_read_lane(self) -> None:
        """F15: web.extract is R1 read-lane, extracts text with zero prompts."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(id="c15", name="web__extract", arguments={"selector": "article.main"})
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["extracted"] is True
        assert len(collector.pending_events) == 0  # Read-lane: zero prompts

    @pytest.mark.anyio
    async def test_f16_web_wait_for_selector(self) -> None:
        """F16: web.wait_for is R1 read-lane, waits for selector with zero prompts."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c16", name="web__wait_for", arguments={"selector": "#content", "state": "visible"}
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["satisfied"] is True
        assert len(collector.pending_events) == 0  # Read-lane: zero prompts


# ==============================================================================
# Tier 2: Boundary, Error & Security Attack Cases
# ==============================================================================


class TestTier2BoundaryAndSecurityCases:
    """Tier 2: SSRF Attack Matrix, Consequential Boundaries, ARIA Compaction, Schema Validation."""

    @pytest.mark.anyio
    async def test_t2_01_anti_ssrf_comprehensive_matrix(self) -> None:
        """T2.1: Comprehensive Anti-SSRF attack matrix blocks all private/metadata IPs."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        malicious_urls = [
            "http://example.com",  # Non-HTTPS
            "ftp://example.com/file",  # Non-HTTPS
            "file:///etc/passwd",  # File scheme
            "https://127.0.0.1/admin",  # IPv4 loopback
            "https://10.0.0.1/internal",  # 10.0.0.0/8 private
            "https://192.168.1.1/router",  # 192.168.0.0/16 private
            "https://172.16.0.1/status",  # 172.16.0.0/12 private
            "https://169.254.169.254/latest/meta-data/",  # AWS metadata
            "https://[::1]/api",  # IPv6 loopback
            "https://127.0.0.0x1/debug",  # Hex obfuscation
            "https://127.0.0.0x01",  # Hex obfuscation
            "https://0x7f.0.0.0x1",  # Full hex IP
            "https://10.0.0.0x1",  # Hex LAN
            "https://192.168.1.0x1",  # Hex LAN
            "https://localhost:8080",  # Localhost
            "https://app.localhost",  # Localhost subdomain
            "https://service.local",  # mDNS .local
            "https://instance.internal",  # Cloud intranet .internal
            "https://router.lan",  # Router .lan
            "https://home.arpa",  # .arpa
            "https://device.localdomain",  # .localdomain
            "https://admin:pass@example.com",  # Userinfo credentials
            "https://example.com:0",  # Invalid port 0
            "https://example.com:65536",  # Port overflow
            "https://127.0.0.1%00/",  # Percent-encoded null byte loopback
            "https://127.0.0.1%2e/",  # Percent-encoded dot loopback
            "https://127.0.0.1%23/",  # Percent-encoded hash loopback
            "https://127.0.0.1%2f/",  # Percent-encoded slash loopback
            "https://169.254.169.254\x00/",  # Raw null byte metadata
            "https://10.0.0.1\x00/",  # Raw null byte private IP
        ]

        for idx, bad_url in enumerate(malicious_urls):
            call = ToolCall(id=f"ssrf_{idx}", name="web__navigate", arguments={"url": bad_url})
            msg = await orch.handle_call(call, ctx)

            assert msg.role == Role.TOOL
            data = json.loads(msg.content)
            assert data["status"] == "invalid_args", f"Failed to reject SSRF url: {bad_url}"
            assert data["output"]["errorCode"] == "url_invalid"

        # Hub must NEVER have received any of these malicious calls
        assert len(hub.invocations) == 0
        assert len(collector.pending_events) == 0

    @pytest.mark.anyio
    async def test_t2_02_consequential_web_click_elevation_matrix(self) -> None:
        """T2.2: Consequential targets (checkout, delete, pay) dynamically elevate to R3."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        consequential_targets = [
            {"ref": "btn-checkout"},
            {"ref": "e1", "selector": "#btn-pay-now"},
            {"ref": "e2", "targetText": "Delete Account"},
            {"ref": "e3", "elementName": "Submit Order"},
            {"ref": "payNow"},
            {"ref": "buyNow"},
            {"ref": "submitOrder"},
            {"ref": "deleteItem"},
            {"ref": "DELETE_ITEM"},
            {"ref": "checkOut"},
            {"ref": "checkout"},
            {"ref": "check_out"},
            {"ref": "placeOrder"},
            {"ref": "wire_transfer", "targetText": "wire money"},
            {"ref": "e4", "targetText": "Xác nhận thanh toán"},
            {"ref": "e5", "targetText": "Xóa tài khoản"},
            {"ref": "btn_chuyen_khoan"},
            {"ref": "btn_dat_hang", "targetText": "Đặt hàng ngay"},
            {"ref": "btn_huy", "targetText": "hủy"},
            {"ref": "btn_huy_tk", "targetText": "hủy tài khoản"},
            {"ref": "btn_nop", "targetText": "nộp bài"},
            {"ref": "btn_go_bo", "targetText": "gỡ bỏ tiện ích"},
        ]

        for idx, target_arg in enumerate(consequential_targets):
            collector.pending_events.clear()
            call = ToolCall(id=f"click_cq_{idx}", name="web__click", arguments=target_arg)
            msg = await orch.handle_call(call, ctx)
            data = json.loads(msg.content)
            assert data["status"] == "ok"
            assert len(collector.pending_events) == 1, f"Expected R3 prompt for target {target_arg}"
            assert (
                collector.pending_events[0].risk.lower() == "r3"
            ), f"Expected R3 elevation for target {target_arg}"

    @pytest.mark.anyio
    async def test_t2_03_false_positive_prevention_on_benign_words(self) -> None:
        """T2.3: Benign tokens ('cardigan', 'spin') remain R2 without false elevation."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        benign_targets = [
            {"ref": "link_home", "targetText": "Trang chủ"},
            {"ref": "cardigan", "targetText": "wool cardigan"},
            {"ref": "item_spin", "targetText": "spin the wheel"},
            {"ref": "opinion", "targetText": "share your opinion"},
            {"ref": "wildcard", "targetText": "wildcard search"},
            {"ref": "guide_link", "targetText": "view the guide"},
            {"ref": "theme_button", "targetText": "Dark Theme"},
        ]

        for idx, target_arg in enumerate(benign_targets):
            collector.pending_events.clear()
            call = ToolCall(id=f"click_benign_{idx}", name="web__click", arguments=target_arg)
            msg = await orch.handle_call(call, ctx)
            data = json.loads(msg.content)
            assert data["status"] == "ok"
            assert len(collector.pending_events) == 1
            assert (
                collector.pending_events[0].risk.lower() == "r2"
            ), f"False elevation to R3 on benign token: {target_arg}"

    @pytest.mark.anyio
    async def test_t2_04_sensitive_web_fill_password_and_card_elevation(self) -> None:
        """T2.4: Sensitive fields (password, cvv, card) dynamically elevate web.fill to R3."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        sensitive_fills = [
            {"ref": "input_password", "value": "super_secret"},
            {"ref": "cvv_box", "selector": "input#cvv", "value": "123"},
            {"ref": "card_number", "value": "4111222233334444"},
            {"ref": "e1", "targetText": "Mật khẩu đăng nhập", "value": "user_pass"},
            {"ref": "input_box", "value": "payNow"},
            {"ref": "input_box", "value": "DELETE_ITEM"},
        ]

        for idx, fill_arg in enumerate(sensitive_fills):
            collector.pending_events.clear()
            call = ToolCall(id=f"fill_sens_{idx}", name="web__fill", arguments=fill_arg)
            msg = await orch.handle_call(call, ctx)
            data = json.loads(msg.content)
            assert data["status"] == "ok"
            assert len(collector.pending_events) == 1
            assert (
                collector.pending_events[0].risk.lower() == "r3"
            ), f"Expected R3 elevation for fill {fill_arg}"

    @pytest.mark.anyio
    async def test_t2_05_aria_compact_tree_shrink_progression(self) -> None:
        """T2.5: Progressive shrinking bounds giant 40KB ARIA tree to strictly <= 6144 B."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # Generate large 40KB ARIA tree
        large_aria = "# ARIA Tree — Huge Web Application\n" + (
            "- [ref=e{i}] button 'Perform Action #{i} in panel' state=visible\n" * 400
        ).format(i=1)
        assert len(large_aria.encode("utf-8")) > 20000

        hub.custom_responses["web.snapshot"] = {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {
                "url": "https://example.com",
                "snapshotId": "giant-snap",
                "compactAriaTree": large_aria,
                "nodes": [{"id": i} for i in range(250)],
            },
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 80},
        }

        call = ToolCall(id="snap_huge", name="web__snapshot", arguments={"compact": True})
        msg = await orch.handle_call(call, ctx)

        encoded_bytes = msg.content.encode("utf-8")
        assert len(encoded_bytes) <= MAX_TOOL_OUTPUT_BYTES
        assert len(encoded_bytes) <= 6144

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert "compactAriaTree" in data["output"]

    @pytest.mark.anyio
    async def test_t2_06_malformed_arguments_and_schema_rejection(self) -> None:
        """T2.6: Schema violation (unexpected extra or missing fields) returns invalid_args."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # 1. notes.create missing required 'body'
        call_missing = ToolCall(
            id="err1", name="notes__create", arguments={"title": "Missing body"}
        )
        msg1 = await orch.handle_call(call_missing, ctx)
        data1 = json.loads(msg1.content)
        assert data1["status"] == "invalid_args"

        # 2. finder.reveal with additionalProperties violation
        call_extra = ToolCall(
            id="err2",
            name="finder__reveal",
            arguments={"path": "/var/data/doc.txt", "hackFlag": True},
        )
        msg2 = await orch.handle_call(call_extra, ctx)
        data2 = json.loads(msg2.content)
        assert data2["status"] == "invalid_args"

        # Zero XTP calls sent
        assert len(hub.invocations) == 0

    @pytest.mark.anyio
    async def test_t2_07_remote_host_error_and_timeout_propagation(self) -> None:
        """T2.7: Remote host errorCode and timeouts are preserved faithfully (Rule H4)."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # 1. Host error code preservation
        hub.custom_responses["notes.append"] = {
            "content": [{"type": "text", "text": "target_not_found"}],
            "structuredContent": {"errorCode": "target_not_found", "status": "failed"},
            "isError": True,
            "_meta": {
                "xuna/status": "failed",
                "xuna/errorCode": "target_not_found",
                "xuna/durationMs": 15,
            },
        }
        call = ToolCall(
            id="err_target", name="notes__append", arguments={"id": "ghost-note", "text": "append"}
        )
        msg = await orch.handle_call(call, ctx)
        data = json.loads(msg.content)
        assert data["status"] == "failed"
        assert data["output"]["errorCode"] == "target_not_found"
        assert len(collector.result_events) == 1
        assert collector.result_events[0].error_code == "target_not_found"

        # 2. Timeout error propagation
        class _TimingOutHub(_FakeSessionHub):
            async def send_request(
                self, host_id: str, method: str, params: dict[str, object], timeout_s: float
            ) -> dict[str, object]:
                raise TimeoutError("Deadline exceeded")

        orch._hub = _TimingOutHub()
        collector.result_events.clear()
        call_timeout = ToolCall(
            id="timeout_call", name="notes__create", arguments={"title": "T", "body": "B"}
        )
        msg_to = await orch.handle_call(call_timeout, ctx)
        data_to = json.loads(msg_to.content)
        assert data_to["status"] == "timeout"
        assert collector.result_events[0].status == "timeout"


# ==============================================================================
# Tier 3: Cross-Feature Combinations & State Interactions
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Tier 3: Multi-Tool Chains, 15m Scope Re-use, Standing Approvals, CU Session Envelopes."""

    @pytest.mark.anyio
    async def test_t3_01_native_notes_and_browser_research_chain(self) -> None:
        """T3.1: Multi-tool chain: navigate -> snapshot -> extract -> notes.create/append."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # Step 1: web.navigate
        call1 = ToolCall(
            id="step1", name="web__navigate", arguments={"url": "https://docs.xuna.ai/specs"}
        )
        msg1 = await orch.handle_call(call1, ctx)
        assert json.loads(msg1.content)["status"] == "ok"

        # Step 2: web.snapshot (R1 read-lane)
        call2 = ToolCall(id="step2", name="web__snapshot", arguments={"compact": True})
        msg2 = await orch.handle_call(call2, ctx)
        assert json.loads(msg2.content)["status"] == "ok"

        # Step 3: web.extract (R1 read-lane)
        call3 = ToolCall(id="step3", name="web__extract", arguments={"selector": "div.content"})
        msg3 = await orch.handle_call(call3, ctx)
        assert json.loads(msg3.content)["status"] == "ok"

        # Step 4: notes.create
        call4 = ToolCall(
            id="step4",
            name="notes__create",
            arguments={"title": "Research Dossier", "body": "Initial findings"},
        )
        msg4 = await orch.handle_call(call4, ctx)
        assert json.loads(msg4.content)["status"] == "ok"

        # Step 5: notes.append
        call5 = ToolCall(
            id="step5",
            name="notes__append",
            arguments={"id": "note-101", "text": "Appended browser extracts"},
        )
        msg5 = await orch.handle_call(call5, ctx)
        assert json.loads(msg5.content)["status"] == "ok"

        # Verify exact chain execution
        assert len(hub.invocations) == 5
        assert [inv["tool"] for inv in hub.invocations] == [
            "web.navigate",
            "web.snapshot",
            "web.extract",
            "notes.create",
            "notes.append",
        ]

    @pytest.mark.anyio
    async def test_t3_02_15m_conversation_scope_for_r2_native_tools(self) -> None:
        """T3.2: 15-minute conversation scope grants auto-approval for subsequent R2 calls."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, scope_store, _ = _setup_test_orchestrator(owner_id)

        # First call: approved with CONVERSATION_15M scope
        collector1 = EventCollector(orch, owner_id, scope=ApprovalScope.CONVERSATION_15M)
        ctx1 = _make_action_context(owner_id, collector1)
        call1 = ToolCall(
            id="c1", name="notes__create", arguments={"title": "Note 1", "body": "Body 1"}
        )
        await orch.handle_call(call1, ctx1)
        assert len(collector1.pending_events) == 1

        # Second call within 15m: auto-approved without prompting!
        collector2 = EventCollector(orch, owner_id, auto_decision=None)  # No decider!
        ctx2 = _make_action_context(owner_id, collector2)
        call2 = ToolCall(
            id="c2", name="notes__create", arguments={"title": "Note 2", "body": "Body 2"}
        )
        msg2 = await orch.handle_call(call2, ctx2)

        data2 = json.loads(msg2.content)
        assert data2["status"] == "ok"
        # Zero new owner prompts!
        assert len(collector2.pending_events) == 0
        assert len(hub.invocations) == 2

    @pytest.mark.anyio
    async def test_t3_03_15m_scope_does_not_exempt_r3_actions(self) -> None:
        """T3.3: 15-minute conversation scope NEVER exempts R3 actions."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)

        # Grant 15m scope on benign web.click
        collector1 = EventCollector(orch, owner_id, scope=ApprovalScope.CONVERSATION_15M)
        ctx1 = _make_action_context(owner_id, collector1)
        call_benign = ToolCall(
            id="c_benign", name="web__click", arguments={"ref": "link_home", "targetText": "Home"}
        )
        await orch.handle_call(call_benign, ctx1)
        assert len(collector1.pending_events) == 1
        assert collector1.pending_events[0].risk.lower() == "r2"

        # Consequential web.click -> Must prompt at R3 despite active 15m scope!
        collector2 = EventCollector(orch, owner_id, auto_decision=ApprovalDecision.APPROVE)
        ctx2 = _make_action_context(owner_id, collector2)
        call_cq = ToolCall(id="c_cq", name="web__click", arguments={"ref": "btn-checkout"})
        await orch.handle_call(call_cq, ctx2)

        assert len(collector2.pending_events) == 1
        assert collector2.pending_events[0].risk.lower() == "r3"

    @pytest.mark.anyio
    async def test_t3_04_cu_session_envelope_does_not_exempt_unscoped_tools(self) -> None:
        """T3.4: Active CU session envelope does NOT exempt unscoped tools (web.* or notes.*)."""
        owner_id = uuid.uuid4()
        from xuna_backend.modules.actions.application.computer_use import (
            ComputerUseSessionState,
            InMemoryComputerUseSessionStore,
        )

        cu_store = InMemoryComputerUseSessionStore()
        orch, host_repo, hub, _, _, _ = _setup_test_orchestrator(owner_id, cu_store=cu_store)
        host = (await host_repo.list_hosts(owner_id))[0]

        # Start active CU session
        session = ComputerUseSessionState(
            session_id="cu-session-1",
            owner_id=owner_id,
            host_id=host.host_id,
            goal="Automate tasks",
            allowed_apps=("notes",),
            allowed_domains=(),
            max_steps=20,
            max_minutes=15,
            started_at_utc=NOW,
            last_heartbeat_at=NOW,
            active=True,
        )
        await cu_store.start(session, NOW)

        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # web.navigate (ToolFamily.URL) is NOT exempt by CU session envelope -> prompts at R2!
        call_nav = ToolCall(
            id="nav_call", name="web__navigate", arguments={"url": "https://docs.xuna.ai"}
        )
        await orch.handle_call(call_nav, ctx)
        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r2"

        # notes.create (ToolFamily.APP) is NOT exempt by CU session envelope -> prompts at R2!
        call_notes = ToolCall(
            id="notes_call", name="notes__create", arguments={"title": "T", "body": "B"}
        )
        await orch.handle_call(call_notes, ctx)
        assert len(collector.pending_events) == 2
        assert collector.pending_events[1].risk.lower() == "r2"

    @pytest.mark.anyio
    async def test_t3_05_rejection_and_expiration_flows(self) -> None:
        """T3.5: Owner rejection terminates call with status='rejected'; zero XTP request sent."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)

        # Explicit owner rejection
        collector = EventCollector(orch, owner_id, auto_decision=ApprovalDecision.REJECT)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="rej_call",
            name="notes__create",
            arguments={"title": "Rejected Note", "body": "Content"},
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "rejected"
        assert len(collector.pending_events) == 1
        assert len(collector.result_events) == 1
        assert collector.result_events[0].status == "rejected"
        assert len(hub.invocations) == 0  # Zero requests dispatched


# ==============================================================================
# Tier 4: Comprehensive Real-World Workloads & Audits
# ==============================================================================


class TestTier4RealWorldScenariosAndAudits:
    """Tier 4: Production Scenarios S1, S2, S3, Mobile Wire Compatibility, Rule H5 Audit."""

    @pytest.mark.anyio
    async def test_t4_01_scenario_s1_multistep_note_and_file_reveal(self) -> None:
        """S1 Scenario: notes.create -> notes.append -> notes.search -> finder.reveal."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # 1. notes.create
        msg1 = await orch.handle_call(
            ToolCall(
                id="s1_1",
                name="notes__create",
                arguments={"title": "Q4 Architecture Spec", "body": "Deliverables"},
            ),
            ctx,
        )
        assert json.loads(msg1.content)["status"] == "ok"

        # 2. notes.append
        msg2 = await orch.handle_call(
            ToolCall(
                id="s1_2",
                name="notes__append",
                arguments={"title": "Q4 Architecture Spec", "text": "Addendum"},
            ),
            ctx,
        )
        assert json.loads(msg2.content)["status"] == "ok"

        # 3. notes.search (read-lane)
        msg3 = await orch.handle_call(
            ToolCall(id="s1_3", name="notes__search", arguments={"query": "Q4 Architecture"}),
            ctx,
        )
        assert json.loads(msg3.content)["status"] == "ok"

        # 4. finder.reveal (read-lane)
        msg4 = await orch.handle_call(
            ToolCall(
                id="s1_4",
                name="finder__reveal",
                arguments={"path": "/Users/thao/Documents/specs/m3_plan.pdf"},
            ),
            ctx,
        )
        assert json.loads(msg4.content)["status"] == "ok"

        # Total owner prompts: exactly 2 (notes.create and notes.append; read-lane had 0 prompts)
        assert len(collector.pending_events) == 2
        assert len(hub.invocations) == 4

    @pytest.mark.anyio
    async def test_t4_02_scenario_s2_meeting_prep_and_draft_communication(self) -> None:
        """S2 Scenario: calendar events -> mail.create_draft (draft-only R2)."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # 1. calendar.list_events (read-lane)
        msg1 = await orch.handle_call(
            ToolCall(
                id="s2_1",
                name="calendar__list_events",
                arguments={"timeMin": "2026-11-15T09:00:00Z", "timeMax": "2026-11-15T18:00:00Z"},
            ),
            ctx,
        )
        assert json.loads(msg1.content)["status"] == "ok"

        # 2. calendar.create_event
        msg2 = await orch.handle_call(
            ToolCall(
                id="s2_2",
                name="calendar__create_event",
                arguments={
                    "title": "Wave B Review",
                    "startTime": "2026-11-15T16:00:00Z",
                    "endTime": "2026-11-15T17:00:00Z",
                },
            ),
            ctx,
        )
        assert json.loads(msg2.content)["status"] == "ok"

        # 3. mail.create_draft (STRICTLY DRAFT-ONLY)
        msg3 = await orch.handle_call(
            ToolCall(
                id="s2_3",
                name="mail__create_draft",
                arguments={
                    "to": ["thao@xuna.ai"],
                    "subject": "Review Agenda",
                    "body": "Attached agenda",
                },
            ),
            ctx,
        )
        data3 = json.loads(msg3.content)
        assert data3["status"] == "ok"
        assert data3["output"]["isDraft"] is True

        # Total owner prompts: exactly 2 (calendar.create_event and mail.create_draft)
        assert len(collector.pending_events) == 2
        assert collector.pending_events[0].risk.lower() == "r2"
        assert collector.pending_events[1].risk.lower() == "r2"

    @pytest.mark.anyio
    async def test_t4_03_scenario_s3_secure_browser_research_and_checkout(self) -> None:
        """S3 Scenario: browser research -> web.click R3 halt without approval."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _ = _setup_test_orchestrator(owner_id)

        # 1. SSRF prevention
        collector_ssrf = EventCollector(orch, owner_id)
        ctx_ssrf = _make_action_context(owner_id, collector_ssrf)
        ssrf_call = ToolCall(
            id="s3_0", name="web__navigate", arguments={"url": "https://127.0.0.1/admin"}
        )
        msg0 = await orch.handle_call(ssrf_call, ctx_ssrf)
        assert json.loads(msg0.content)["output"]["errorCode"] == "url_invalid"

        # 2. Legitimate navigate
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)
        msg1 = await orch.handle_call(
            ToolCall(
                id="s3_1",
                name="web__navigate",
                arguments={"url": "https://shop.vietnam.vn/products"},
            ),
            ctx,
        )
        assert json.loads(msg1.content)["status"] == "ok"

        # 3. web.snapshot (read-lane)
        msg2 = await orch.handle_call(
            ToolCall(id="s3_2", name="web__snapshot", arguments={"compact": True}), ctx
        )
        assert json.loads(msg2.content)["status"] == "ok"

        # 4. web.fill benign query
        msg3 = await orch.handle_call(
            ToolCall(
                id="s3_3", name="web__fill", arguments={"ref": "search-input", "value": "pixel pet"}
            ),
            ctx,
        )
        assert json.loads(msg3.content)["status"] == "ok"

        # 5. web.click consequential checkout -> without R3 approval, halts closed!
        collector_reject = EventCollector(orch, owner_id, auto_decision=ApprovalDecision.REJECT)
        ctx_reject = _make_action_context(owner_id, collector_reject)
        msg4 = await orch.handle_call(
            ToolCall(
                id="s3_4",
                name="web__click",
                arguments={"ref": "btn-checkout", "targetText": "Xác nhận thanh toán"},
            ),
            ctx_reject,
        )
        assert json.loads(msg4.content)["status"] == "rejected"
        assert collector_reject.pending_events[0].risk.lower() == "r3"

    def test_t4_04_mobile_wire_format_compatibility(self) -> None:
        """T4.4: ActionPendingEvent wire_payload emits lowercase 'r1'/'r2'/'r3' matching mobile."""
        pending_r1 = ActionPendingEvent(
            approval_id=str(uuid.uuid4()),
            target="Mac Studio",
            summary="Test R1",
            risk="R1",
            expires_in=60,
        )
        payload_r1 = pending_r1.wire_payload()
        assert payload_r1["risk"] == "r1"

        pending_r2 = ActionPendingEvent(
            approval_id=str(uuid.uuid4()),
            target="Mac Studio",
            summary="Test R2",
            risk="r2",
            expires_in=60,
        )
        payload_r2 = pending_r2.wire_payload()
        assert payload_r2["risk"] == "r2"

        pending_r3 = ActionPendingEvent(
            approval_id=str(uuid.uuid4()),
            target="Mac Studio",
            summary="Test R3",
            risk="R3",
            expires_in=60,
        )
        payload_r3 = pending_r3.wire_payload()
        assert payload_r3["risk"] == "r3"

        # Verify ActionResultEvent serialization fields
        res_ev = ActionResultEvent(
            invocation_id="inv-1",
            target="Mac Studio",
            summary="Finished",
            status="ok",
            gesture="nod",
            error_code=None,
            host_id="h-1",
        )
        assert res_ev.invocation_id == "inv-1"
        assert res_ev.status == "ok"

    def test_t4_05_rule_h5_generic_layer_scan(self) -> None:
        """T4.5: Rule H5 Generic Layer Scan: 0 hardcoded app names in generic layers."""
        base_dir = Path(__file__).resolve().parents[2] / "src" / "xuna_backend" / "modules"
        target_files = [
            base_dir / "actions" / "application" / "prompts.py",
            base_dir / "actions" / "application" / "catalog.py",
        ]
        conversation_dir = base_dir / "conversation" / "application"
        if conversation_dir.exists():
            target_files.extend(conversation_dir.glob("*.py"))

        forbidden_regex = re.compile(r"\b(spotify|chrome|safari|zalo)\b", re.IGNORECASE)

        for file_path in target_files:
            if not file_path.exists():
                continue
            text = file_path.read_text(encoding="utf-8")
            for line_idx, line in enumerate(text.splitlines(), start=1):
                clean_line = line.strip()
                if clean_line.startswith("#"):
                    continue
                match = forbidden_regex.search(clean_line)
                assert match is None, (
                    f"Rule H5 violation: Found forbidden hardcoded app '{match.group(0)}' "
                    f"in {file_path.name}:{line_idx} -> '{clean_line}'"
                )
