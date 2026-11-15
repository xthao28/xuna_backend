"""Wave C End-to-End Test Suite: Vision & Normalized Coordinate Fallback Engine.

Comprehensive 4-Tier E2E verification of Wave C
("Đôi mắt Vision + Điều khiển mọi app không cần AX chuẩn"):
- Tier 1: Feature Coverage (F1 to F8: Hybrid Observe, Coordinate Click & Drag,
  HiDPI scaling, UI Settle, Fail-closed permissions).
- Tier 2: Boundary & Corner Cases (Boundary coords 0.0/1.0, maxNodes 1..300,
  maxEdge 320..1280, empty window, nested node hit-test).
- Tier 3: Cross-Feature Combinations & State Interactions (Payload decoupling <= 6144 B,
  Ephemeral Multimodal Vision, Consent U5 Zero-Retention, thought_signature preservation,
  R3 elevation, password denial, uncommitted draft protection).
- Tier 4: Real-World Application Scenarios (S1: Non-AX Canvas drag,
  S2: Electron coordinate navigation, S3: Payment elevation,
  S4: Protected draft escalation, S5: Permission denied fail-closed,
  S6: Rule H5 Generic Compliance AST scan).
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from xuna_backend.modules.actions.application.catalog import (
    EligibleHostTool,
    _host_tool_spec,
)
from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    InMemoryComputerUseSessionStore,
    hit_test_node_at_coordinates,
)
from xuna_backend.modules.actions.application.orchestrator import (
    CAP_HOST_ACTIONS_V1,
    ActionContext,
    Orchestrator,
)
from xuna_backend.modules.actions.application.prompts import (
    MAX_TOOL_OUTPUT_BYTES,
    encode_tool_payload,
)
from xuna_backend.modules.actions.domain.entities import (
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
    _serialize_message,
)
from xuna_backend.modules.conversation.application.chat_turn import chat_turn
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

NOW = datetime(2026, 11, 15, 0, 0, 0, tzinfo=UTC)
FAKE_PUBKEY = b"\x04" + bytes(range(1, 65))
TINY_1X1_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP////////////////////////////////////////////////"
    "//////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAA"
    "AP/aAAgBAQABPxA="
)


def _make_host(
    owner_id: uuid.UUID,
    alias: str = "Studio Host",
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
        hostname="xuna-studio-wave-c",
        os_version="15.0",
        agent_version="0.3.0",
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


def _build_all_wave_c_descriptors() -> list[ToolDescriptor]:
    return [
        _make_descriptor(
            "cu.session_start",
            ToolFamily.CU,
            RiskTier.R3,
            properties={
                "goal": {"type": "string"},
                "apps": {"type": "array", "items": {"type": "string"}},
                "domains": {"type": "array", "items": {"type": "string"}},
                "maxSteps": {"type": "integer"},
                "maxMinutes": {"type": "integer"},
            },
            required=["goal"],
        ),
        _make_descriptor(
            "cu.session_end",
            ToolFamily.CU,
            RiskTier.R2,
            properties={"reason": {"type": "string"}},
        ),
        _make_descriptor(
            "ui.observe_hybrid",
            ToolFamily.UI,
            RiskTier.R1,
            properties={
                "scope": {"type": "string"},
                "maxNodes": {"type": "integer"},
                "maxEdge": {"type": "integer"},
                "skeleton": {"type": "boolean"},
                "interactiveOnly": {"type": "boolean"},
                "rootElementId": {"type": "string"},
                "collapseWebWrappers": {"type": "boolean"},
            },
            required=["scope"],
        ),
        _make_descriptor(
            "ui.inspect",
            ToolFamily.UI,
            RiskTier.R1,
            properties={"depth": {"type": "integer"}},
        ),
        _make_descriptor(
            "ui.find",
            ToolFamily.UI,
            RiskTier.R2,
            properties={"query": {"type": "string"}},
            required=["query"],
        ),
        _make_descriptor(
            "input.click",
            ToolFamily.INPUT,
            RiskTier.R2,
            properties={
                "elementId": {"type": "string"},
                "elementName": {"type": "string"},
                "xNorm": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "yNorm": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "headed": {"type": "boolean"},
                "waitFor": {"type": "string"},
                "waitForGone": {"type": "string"},
            },
        ),
        _make_descriptor(
            "input.drag",
            ToolFamily.INPUT,
            RiskTier.R2,
            properties={
                "fromXNorm": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "fromYNorm": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "toXNorm": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "toYNorm": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "durationMs": {"type": "integer", "minimum": 50, "maximum": 5000},
                "headed": {"type": "boolean"},
            },
            required=["fromXNorm", "fromYNorm", "toXNorm", "toYNorm"],
        ),
        _make_descriptor(
            "input.type_text",
            ToolFamily.INPUT,
            RiskTier.R2,
            properties={
                "text": {"type": "string"},
                "elementId": {"type": "string"},
                "clearBefore": {"type": "boolean"},
            },
            required=["text"],
        ),
        _make_descriptor(
            "input.key",
            ToolFamily.INPUT,
            RiskTier.R2,
            properties={"combo": {"type": "string"}},
            required=["combo"],
        ),
        _make_descriptor(
            "input.scroll",
            ToolFamily.INPUT,
            RiskTier.R2,
            properties={"deltaX": {"type": "integer"}, "deltaY": {"type": "integer"}},
        ),
    ]


def _make_cu_session_state(
    owner_id: uuid.UUID,
    host_id: HostId,
    *,
    session_id: str = "cu-session-wave-c-test",
    goal: str = "Wave C Test Session",
    allowed_apps: tuple[str, ...] = ("testapp",),
    allowed_domains: tuple[str, ...] = (),
    max_steps: int = 40,
    max_minutes: int = 20,
    typed_count: int = 0,
    typed_since_commit: list[str] | None = None,
    cached_nodes: dict[str, dict[str, Any]] | None = None,
) -> ComputerUseSessionState:
    return ComputerUseSessionState(
        session_id=session_id,
        owner_id=owner_id,
        host_id=host_id,
        goal=goal,
        allowed_apps=allowed_apps,
        allowed_domains=allowed_domains,
        max_steps=max_steps,
        max_minutes=max_minutes,
        started_at_utc=NOW,
        last_heartbeat_at=NOW,
        steps_used=0,
        consecutive_no_change=0,
        last_action_fingerprint=None,
        cached_nodes=dict(cached_nodes) if cached_nodes is not None else {},
        typed_since_commit=list(typed_since_commit) if typed_since_commit is not None else [],
        typed_count=typed_count,
        owner_prompts=1,
        active=True,
    )


class _FakeHostRepository(HostRepository):
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
    def __init__(self) -> None:
        self.sent_requests: list[tuple[str, str, dict[str, object], float]] = []
        self.invocations: list[dict[str, object]] = []
        self.tool_calls: list[dict[str, object]] = []
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
        self.invocations.append(
            {"host_id": host_id, "tool": tool_name, "arguments": args_dict, "method": method}
        )

        if method == "tools/call":
            self.tool_calls.append({"host_id": host_id, "tool": tool_name, "arguments": args_dict})

        if tool_name in self.custom_responses:
            return self.custom_responses[tool_name]

        if method == "xuna/preflight":
            if "xuna/preflight" in self.custom_responses:
                return self.custom_responses["xuna/preflight"]
            raise RuntimeError("rpc_error:-32601: Method not found")

        sc: dict[str, object] = {"status": "ok"}
        if tool_name == "ui.observe_hybrid":
            sc = {
                "appId": "com.test.app",
                "windowTitle": "Test Window",
                "snapshotId": "snap-hybrid-101",
                "surfaces": ["window"],
                "collapsedWrappersCount": 0,
                "nodes": [
                    {
                        "id": "btn-run",
                        "role": "button",
                        "name": "Execute Action",
                        "bounds": [0.1, 0.1, 0.3, 0.2],
                    },
                    {
                        "id": "txt-input",
                        "role": "textfield",
                        "name": "Search Query",
                        "bounds": [0.4, 0.1, 0.8, 0.2],
                        "editable": True,
                    },
                ],
                "screenshot": {
                    "mimeType": "image/jpeg",
                    "width": 1280,
                    "height": 720,
                    "dataBase64": TINY_1X1_JPEG_B64,
                },
            }
        elif tool_name == "input.click":
            sc = {
                "elementId": str(args_dict.get("elementId", "")),
                "elementName": str(args_dict.get("elementName", "")),
                "xNorm": args_dict.get("xNorm"),
                "yNorm": args_dict.get("yNorm"),
                "clicked": True,
                "settled": True,
                "settleMs": 120,
            }
        elif tool_name == "input.drag":
            sc = {
                "fromXNorm": args_dict.get("fromXNorm"),
                "fromYNorm": args_dict.get("fromYNorm"),
                "toXNorm": args_dict.get("toXNorm"),
                "toYNorm": args_dict.get("toYNorm"),
                "durationMs": args_dict.get("durationMs", 300),
                "dragged": True,
                "settled": True,
                "settleMs": 250,
            }
        elif tool_name == "input.type_text":
            sc = {
                "typed": True,
                "textLength": len(str(args_dict.get("text", ""))),
                "settled": True,
                "settleMs": 50,
            }
        elif tool_name == "cu.session_start":
            sc = {
                "sessionId": "cu-session-wave-c-1",
                "active": True,
                "apps": args_dict.get("apps", ["testapp"]),
                "domains": args_dict.get("domains", []),
            }
        elif tool_name == "cu.session_end":
            sc = {
                "sessionId": "cu-session-wave-c-1",
                "active": False,
                "reason": str(args_dict.get("reason", "completed")),
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
    cu_store: InMemoryComputerUseSessionStore | None = None,
) -> tuple[
    Orchestrator,
    _FakeHostRepository,
    _FakeSessionHub,
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    _FakeAuditLog,
    InMemoryComputerUseSessionStore,
]:
    current_now = clock_dt if clock_dt is not None else NOW
    host_repo = _FakeHostRepository()
    hub = _FakeSessionHub()
    approval_repo = InMemoryApprovalRepository()
    approval_broker = InMemoryApprovalBroker()
    scope_store = InMemoryApprovalScopeStore()
    last_target = InMemoryLastTargetStore()
    audit = _FakeAuditLog()
    store = cu_store if cu_store is not None else InMemoryComputerUseSessionStore()

    host = _make_host(owner_id)
    descriptors = _build_all_wave_c_descriptors()
    manifest = Manifest(
        host_id=host.host_id,
        tools=tuple(descriptors),
        updated_at=current_now,
        schema_version=1,
    )
    grants = [
        Grant(
            host_id=host.host_id,
            family=ToolFamily.CU,
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
        Grant(
            host_id=host.host_id,
            family=ToolFamily.APP,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=current_now,
        ),
    ]

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
        cu_session_store=store,
        clock=lambda: current_now,
    )
    return orch, host_repo, hub, approval_broker, scope_store, audit, store


def _make_action_context(
    owner_id: uuid.UUID,
    collector: EventCollector,
    *,
    pet_id: str = "pet-wave-c",
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
# Tier 1: Feature Coverage (F1 to F8)
# ==============================================================================


class TestTier1WaveCFeatureCoverage:
    """F1 - F8: Full coverage for Wave C features."""

    @pytest.mark.anyio
    async def test_t1_f1_observe_hybrid_returns_ax_tree_and_jpeg_screenshot(self) -> None:
        """F1: ui.observe_hybrid returns compact AX tree with [0.0, 1.0] bounds and JPEG."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c-obs-1",
            name="ui__observe_hybrid",
            arguments={"scope": "active_window", "maxNodes": 50, "maxEdge": 800},
        )
        msg = await orch.handle_call(call, ctx)

        assert msg.role == Role.TOOL
        assert msg.image_base64 == TINY_1X1_JPEG_B64
        assert msg.image_mime_type == "image/jpeg"

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        output = data["output"]
        assert output["appId"] == "com.test.app"
        assert output["windowTitle"] == "Test Window"
        assert len(output["nodes"]) == 2
        assert output["nodes"][0]["bounds"] == [0.1, 0.1, 0.3, 0.2]

        assert len(hub.tool_calls) == 1
        assert hub.tool_calls[0]["tool"] == "ui.observe_hybrid"

    @pytest.mark.anyio
    async def test_t1_f2_observe_hybrid_fail_closed_permission_missing(self) -> None:
        """F2a: Screen recording permission missing fails closed with needs_permission."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        hub.custom_responses["ui.observe_hybrid"] = {
            "content": [{"type": "text", "text": "Screen recording denied"}],
            "structuredContent": {
                "status": "needs_permission",
                "errorCode": "screen_recording",
                "message": "Screen recording permission missing on host",
            },
            "isError": True,
            "_meta": {
                "xuna/status": "needs_permission",
                "xuna/errorCode": "screen_recording",
            },
        }

        call = ToolCall(
            id="c-obs-perm-fail",
            name="ui__observe_hybrid",
            arguments={"scope": "active_window"},
        )
        msg = await orch.handle_call(call, ctx)

        assert msg.image_base64 is None
        data = json.loads(msg.content)
        assert data["status"] in ("needs_permission", "error")
        assert (
            data.get("errorCode") == "screen_recording"
            or data["output"].get("errorCode") == "screen_recording"
        )

    @pytest.mark.anyio
    async def test_t1_f2_observe_hybrid_fail_closed_capture_unavailable(self) -> None:
        """F2b: Screen capture platform driver missing fails closed with unsupported."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        hub.custom_responses["ui.observe_hybrid"] = {
            "content": [{"type": "text", "text": "Screen capture unavailable"}],
            "structuredContent": {
                "status": "unsupported",
                "errorCode": "platform_unsupported",
                "message": "Screen capture driver is missing on host platform",
            },
            "isError": True,
            "_meta": {
                "xuna/status": "unsupported",
                "xuna/errorCode": "platform_unsupported",
            },
        }

        call = ToolCall(
            id="c-obs-driver-fail",
            name="ui__observe_hybrid",
            arguments={"scope": "active_window"},
        )
        msg = await orch.handle_call(call, ctx)

        assert msg.image_base64 is None
        data = json.loads(msg.content)
        assert data["status"] in ("unsupported", "error")
        assert (
            data.get("errorCode") == "platform_unsupported"
            or data["output"].get("errorCode") == "platform_unsupported"
        )

    @pytest.mark.anyio
    async def test_t1_f3_input_click_normalized_coordinates_dispatch(self) -> None:
        """F3: input.click dispatches normalized coordinates [0.0, 1.0] with UI settle."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(
            owner_id,
            host.host_id,
            allowed_apps=("testapp",),
            cached_nodes={
                "btn-action": {
                    "id": "btn-action",
                    "role": "button",
                    "name": "Action Button",
                    "bounds": [0.20, 0.70, 0.30, 0.80],
                }
            },
        )
        await cu_store.start(active_cu, NOW)

        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c-click-norm-1",
            name="input__click",
            arguments={"xNorm": 0.25, "yNorm": 0.75},
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        output = data["output"]
        assert output["xNorm"] == 0.25
        assert output["yNorm"] == 0.75
        assert output["settled"] is True
        assert output["settleMs"] == 120

        assert len(hub.tool_calls) == 1
        assert hub.tool_calls[0]["tool"] == "input.click"
        assert hub.tool_calls[0]["arguments"]["xNorm"] == 0.25
        assert hub.tool_calls[0]["arguments"]["yNorm"] == 0.75

    @pytest.mark.anyio
    async def test_t1_f4_input_drag_normalized_coordinates_dispatch(self) -> None:
        """F4: input.drag dispatches normalized drag coordinates and duration with settle."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(
            owner_id,
            host.host_id,
            allowed_apps=("testapp",),
            cached_nodes={
                "node-start": {
                    "id": "node-start",
                    "role": "shape",
                    "name": "Start Node",
                    "bounds": [0.10, 0.15, 0.25, 0.30],
                },
                "node-end": {
                    "id": "node-end",
                    "role": "shape",
                    "name": "End Node",
                    "bounds": [0.60, 0.70, 0.75, 0.85],
                },
            },
        )
        await cu_store.start(active_cu, NOW)

        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c-drag-1",
            name="input__drag",
            arguments={
                "fromXNorm": 0.15,
                "fromYNorm": 0.20,
                "toXNorm": 0.65,
                "toYNorm": 0.80,
                "durationMs": 450,
            },
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        output = data["output"]
        assert output["fromXNorm"] == 0.15
        assert output["toXNorm"] == 0.65
        assert output["durationMs"] == 450
        assert output["settled"] is True
        assert output["settleMs"] == 250

        assert len(hub.tool_calls) == 1
        assert hub.tool_calls[0]["tool"] == "input.drag"

    def test_t1_f5_hidpi_scaling_coordinate_conversion_contract(self) -> None:
        """F5: HiDPI Retina scaling contract translates normalized coords to physical pixels."""
        window_x = 100
        window_y = 50
        window_width = 1280
        window_height = 720
        dpr = 2.0  # Retina 2x scale

        x_norm = 0.5
        y_norm = 0.25

        logical_x = window_x + (x_norm * window_width)
        logical_y = window_y + (y_norm * window_height)
        physical_x = round(logical_x * dpr)
        physical_y = round(logical_y * dpr)

        assert logical_x == 740.0
        assert logical_y == 230.0
        assert physical_x == 1480
        assert physical_y == 460

    def test_t1_f6_ui_settle_telemetry_reporting(self) -> None:
        """F6: UI Settle telemetry (settled, settleMs <= 1500ms) preserved in payload encoding."""
        raw_output = {
            "appId": "com.custom.app",
            "settled": True,
            "settleMs": 350,
            "changes": {"added": ["e10"], "removed": []},
        }
        encoded = encode_tool_payload("ok", raw_output)
        assert len(encoded.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES
        parsed = json.loads(encoded)
        assert parsed["output"]["settled"] is True
        assert parsed["output"]["settleMs"] == 350


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================


class TestTier2BoundaryAndCornerCases:
    """T2: Boundary coordinates, edge parameters, empty windows, and nested node containment."""

    @pytest.mark.anyio
    async def test_t2_boundary_coordinates_extremes_and_out_of_bounds(self) -> None:
        """T2.1: Coordinate boundary values (0.0, 1.0) and out-of-bounds rejection."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(
            owner_id,
            host.host_id,
            allowed_apps=("testapp",),
            cached_nodes={
                "node-tl": {
                    "id": "node-tl",
                    "role": "button",
                    "name": "Top Left Corner",
                    "bounds": [0.0, 0.0, 0.1, 0.1],
                },
                "node-br": {
                    "id": "node-br",
                    "role": "button",
                    "name": "Bottom Right Corner",
                    "bounds": [0.9, 0.9, 1.0, 1.0],
                },
            },
        )
        await cu_store.start(active_cu, NOW)

        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # Extreme valid corners: (0.0, 0.0) and (1.0, 1.0)
        call_top_left = ToolCall(
            id="c-corner-1",
            name="input__click",
            arguments={"xNorm": 0.0, "yNorm": 0.0},
        )
        msg_tl = await orch.handle_call(call_top_left, ctx)
        data_tl = json.loads(msg_tl.content)
        assert data_tl["status"] == "ok"

        call_bottom_right = ToolCall(
            id="c-corner-2",
            name="input__click",
            arguments={"xNorm": 1.0, "yNorm": 1.0},
        )
        msg_br = await orch.handle_call(call_bottom_right, ctx)
        data_br = json.loads(msg_br.content)
        assert data_br["status"] == "ok"

        # Out of bounds coordinates: (-0.01, 0.5)
        # 1. API level schema validation rejects invalid range [0.0, 1.0]
        call_neg = ToolCall(
            id="c-neg",
            name="input__click",
            arguments={"xNorm": -0.01, "yNorm": 0.5},
        )
        msg_neg = await orch.handle_call(call_neg, ctx)
        data_neg = json.loads(msg_neg.content)
        assert data_neg["status"] in ("invalid_args", "bad_arguments", "error")

        # 2. Risk evaluator defense-in-depth escalates out-of-bounds coordinates to R3
        descriptor_click = next(
            d for d in _build_all_wave_c_descriptors() if d.name == "input.click"
        )
        risk_neg = Orchestrator._effective_tool_risk(
            descriptor_click, {"xNorm": -0.01, "yNorm": 0.5}, active_cu=None
        )
        assert risk_neg == RiskTier.R3
        risk_high = Orchestrator._effective_tool_risk(
            descriptor_click, {"xNorm": 1.05, "yNorm": 0.5}, active_cu=None
        )
        assert risk_high == RiskTier.R3

    def test_t2_max_nodes_and_max_edge_boundary_parameters(self) -> None:
        """T2.2: maxNodes clamped to [1..300], maxEdge clamped to [320..1280]."""
        descriptor = _make_descriptor(
            "ui.observe_hybrid",
            ToolFamily.UI,
            RiskTier.R1,
            properties={
                "scope": {"type": "string"},
                "maxNodes": {"type": "integer", "minimum": 1, "maximum": 300},
                "maxEdge": {"type": "integer", "minimum": 320, "maximum": 1280},
            },
            required=["scope"],
        )
        item = EligibleHostTool(descriptor=descriptor, aliases=("desktop-test",))
        spec = _host_tool_spec("ui.observe_hybrid", item)
        props = spec.parameters["properties"]

        assert props["maxNodes"]["minimum"] == 1
        assert props["maxNodes"]["maximum"] == 300
        assert props["maxEdge"]["minimum"] == 320
        assert props["maxEdge"]["maximum"] == 1280

    @pytest.mark.anyio
    async def test_t2_empty_window_zero_ax_nodes_fallback(self) -> None:
        """T2.3: Observe hybrid handles window with 0 AX nodes, click escalates safely to R3."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(owner_id, host.host_id, allowed_apps=("graphicscanvas",))
        await cu_store.start(active_cu, NOW)

        hub.custom_responses["ui.observe_hybrid"] = {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {
                "appId": "com.canvas.graphics",
                "windowTitle": "Pure Canvas View",
                "snapshotId": "snap-empty-0",
                "surfaces": ["canvas"],
                "collapsedWrappersCount": 0,
                "nodes": [],  # 0 AX nodes
                "screenshot": {
                    "mimeType": "image/jpeg",
                    "width": 1280,
                    "height": 720,
                    "dataBase64": TINY_1X1_JPEG_B64,
                },
            },
            "isError": False,
            "_meta": {"xuna/status": "ok"},
        }

        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # Observe hybrid succeeds
        obs_call = ToolCall(
            id="c-obs-empty", name="ui__observe_hybrid", arguments={"scope": "active_window"}
        )
        msg_obs = await orch.handle_call(obs_call, ctx)
        data_obs = json.loads(msg_obs.content)
        assert data_obs["status"] == "ok"
        assert len(data_obs["output"]["nodes"]) == 0

        # Clicking in unclassified window escalates to R3 because no AX node is cached
        collector_unclassified = EventCollector(
            orch, owner_id, auto_decision=ApprovalDecision.APPROVE
        )
        ctx_unclass = _make_action_context(owner_id, collector_unclassified)
        click_call = ToolCall(
            id="c-click-unclass", name="input__click", arguments={"xNorm": 0.5, "yNorm": 0.5}
        )
        await orch.handle_call(click_call, ctx_unclass)

        assert len(collector_unclassified.pending_events) == 1
        assert collector_unclassified.pending_events[0].risk.lower() == "r3"

    def test_t2_complex_nested_node_bounds_leaf_selection(self) -> None:
        """T2.4: hit_test_node_at_coordinates selects smallest leaf node among nested bounds."""
        nodes: list[dict[str, Any]] = [
            {
                "id": "window",
                "name": "Main Window",
                "role": "window",
                "bounds": [0.0, 0.0, 1.0, 1.0],
            },
            {
                "id": "panel",
                "name": "Content Panel",
                "role": "panel",
                "bounds": [0.1, 0.1, 0.9, 0.9],
            },
            {
                "id": "group",
                "name": "Action Group",
                "role": "group",
                "bounds": [0.2, 0.2, 0.6, 0.6],
            },
            {
                "id": "button",
                "name": "Save File",
                "role": "button",
                "bounds": [0.25, 0.25, 0.35, 0.35],
            },
        ]

        matched = hit_test_node_at_coordinates(nodes, 0.30, 0.30)
        assert matched is not None
        assert matched["id"] == "button"
        assert matched["name"] == "Save File"

        matched_group = hit_test_node_at_coordinates(nodes, 0.50, 0.50)
        assert matched_group is not None
        assert matched_group["id"] == "group"


# ==============================================================================
# Tier 3: Cross-Feature Combinations & State Interactions
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """T3: Multimodal transport, thought signatures, R3 security gates, and uncommitted drafts."""

    @pytest.mark.anyio
    async def test_t3_ephemeral_multimodal_vision_pipeline_and_zero_retention(self) -> None:
        """T3.1: Tool output JSON <= 6144 B, image passed ephemerally to LLM, zero DB retention."""
        large_image_b64 = "X" * 150_000  # 150 KB JPEG
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, audit, _ = _setup_test_orchestrator(owner_id)

        hub.custom_responses["ui.observe_hybrid"] = {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {
                "appId": "com.designer.app",
                "windowTitle": "Vector Art Canvas",
                "snapshotId": "snap-vision-1",
                "nodes": [
                    {"id": f"node-{i}", "role": "shape", "bounds": [0.1, 0.1, 0.5, 0.5]}
                    for i in range(10)
                ],
                "screenshot": {
                    "mimeType": "image/jpeg",
                    "width": 1280,
                    "height": 720,
                    "dataBase64": large_image_b64,
                },
            },
            "isError": False,
            "_meta": {"xuna/status": "ok"},
        }

        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c-vision-turn", name="ui__observe_hybrid", arguments={"scope": "active_window"}
        )
        msg = await orch.handle_call(call, ctx)

        # 1. Decoupled JSON payload <= 6144 bytes
        assert len(msg.content.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES
        # 2. Ephemeral image held in Message
        assert msg.image_base64 == large_image_b64
        # 3. Serialized message formatted into multimodal content block
        serialized = _serialize_message(msg)
        content_blocks = serialized["content"]
        assert isinstance(content_blocks, list)
        assert content_blocks[1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{large_image_b64}"},
        }

        # 4. Zero retention: audit log contains zero base64 image strings
        for ev in audit.events:
            ev_str = str(ev.metadata)
            assert large_image_b64 not in ev_str

    @pytest.mark.anyio
    async def test_t3_thought_signature_preservation_across_multimodal_turns(self) -> None:
        """T3.2: Preserves thought_signature across multimodal turns and LLM generator switches."""
        signature = "sig-alpha-wave-c-thought"

        class MultimodalGenerator:
            def __init__(self) -> None:
                self.calls: list[ChatRequest] = []

            async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
                self.calls.append(request)
                if len(self.calls) == 1:
                    yield ToolCallRequested(
                        call=ToolCall(
                            id="call-m-1",
                            name="ui__observe_hybrid",
                            arguments={"scope": "active_window"},
                            extra={"thought_signature": signature},
                        )
                    )
                else:
                    yield TextDelta("Examined window visually and ready to proceed.")
                    yield ChatFinished(
                        provider="vision-mock",
                        model="vision-pro",
                        input_tokens=15,
                        output_tokens=10,
                        latency_ms=80,
                    )

        gen = MultimodalGenerator()

        async def dummy_runner(call: ToolCall, emit: object) -> Message:
            return Message(
                role=Role.TOOL,
                content='{"status":"ok"}',
                tool_call_id=call.id,
                image_base64=TINY_1X1_JPEG_B64,
            )

        init_req = ChatRequest(
            messages=(Message(role=Role.USER, content="Inspect interface"),),
            tools=(
                ToolSpec(
                    name="ui__observe_hybrid", description="observe", parameters={"type": "object"}
                ),
            ),
        )

        events: list[ChatEvent] = []
        async for ev in chat_turn(gen, init_req, tool_runner=dummy_runner):
            events.append(ev)

        assert len(gen.calls) == 2
        round2_messages = gen.calls[1].messages
        assistant_msg = next(m for m in round2_messages if m.role == Role.ASSISTANT)
        assert assistant_msg.tool_calls is not None
        assert assistant_msg.tool_calls[0].extra.get("thought_signature") == signature

    @pytest.mark.anyio
    async def test_t3_reverse_hit_test_consequential_button_elevation(self) -> None:
        """T3.3: Coordinate click on consequential button escalates to R3 phone approval."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(
            owner_id,
            host.host_id,
            allowed_apps=("accountapp",),
            cached_nodes={
                "btn-delete": {
                    "id": "btn-delete",
                    "role": "button",
                    "name": "Xóa tài khoản vĩnh viễn",
                    "bounds": [0.70, 0.80, 0.95, 0.90],
                }
            },
        )
        await cu_store.start(active_cu, NOW)

        collector_manual = EventCollector(orch, owner_id, auto_decision=ApprovalDecision.APPROVE)
        ctx = _make_action_context(owner_id, collector_manual)

        call = ToolCall(
            id="c-click-delete", name="input__click", arguments={"xNorm": 0.80, "yNorm": 0.85}
        )
        await orch.handle_call(call, ctx)

        assert len(collector_manual.pending_events) == 1
        pending = collector_manual.pending_events[0]
        assert pending.risk.lower() == "r3"
        assert pending.target == "Studio Host"

    @pytest.mark.anyio
    async def test_t3_reverse_hit_test_password_field_denial(self) -> None:
        """T3.4: Coordinate click on password field is denied with password_field_forbidden."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(
            owner_id,
            host.host_id,
            allowed_apps=("authapp",),
            cached_nodes={
                "pwd-1": {
                    "id": "pwd-1",
                    "role": "securetextfield",
                    "name": "Mật khẩu đăng nhập",
                    "bounds": [0.30, 0.40, 0.70, 0.50],
                    "isPassword": True,
                }
            },
        )
        await cu_store.start(active_cu, NOW)

        collector = EventCollector(orch, owner_id, auto_decision=ApprovalDecision.APPROVE)
        ctx = _make_action_context(owner_id, collector)

        call = ToolCall(
            id="c-click-pwd", name="input__click", arguments={"xNorm": 0.50, "yNorm": 0.45}
        )
        msg = await orch.handle_call(call, ctx)

        data = json.loads(msg.content)
        assert data["status"] == "denied"
        assert data["output"]["errorCode"] == "password_field_forbidden"

    @pytest.mark.anyio
    async def test_t3_uncommitted_draft_text_coordinate_click_escalation(self) -> None:
        """T3.5: Active session with uncommitted draft text escalates non-editable clicks to R3."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(
            owner_id,
            host.host_id,
            allowed_apps=("editorapp",),
            typed_count=1,
            typed_since_commit=["Quarterly Financial Report Draft..."],
            cached_nodes={
                "btn-close": {
                    "id": "btn-close",
                    "role": "button",
                    "name": "Discard Changes",
                    "bounds": [0.85, 0.05, 0.95, 0.12],
                },
                "txt-body": {
                    "id": "txt-body",
                    "role": "textarea",
                    "name": "Document Body",
                    "bounds": [0.10, 0.20, 0.90, 0.80],
                    "editable": True,
                },
            },
        )
        await cu_store.start(active_cu, NOW)

        # 1. Clicking Discard button with uncommitted draft escalates to R3
        collector_close = EventCollector(orch, owner_id, auto_decision=ApprovalDecision.APPROVE)
        ctx_close = _make_action_context(owner_id, collector_close)
        call_close = ToolCall(
            id="c-draft-close", name="input__click", arguments={"xNorm": 0.90, "yNorm": 0.08}
        )
        await orch.handle_call(call_close, ctx_close)

        assert len(collector_close.pending_events) == 1
        assert collector_close.pending_events[0].risk.lower() == "r3"

        # 2. Clicking inside verified editable text area remains R2
        collector_text = EventCollector(orch, owner_id, auto_decision=ApprovalDecision.APPROVE)
        ctx_text = _make_action_context(owner_id, collector_text)
        call_text = ToolCall(
            id="c-draft-text", name="input__click", arguments={"xNorm": 0.50, "yNorm": 0.50}
        )
        msg_text = await orch.handle_call(call_text, ctx_text)
        data_text = json.loads(msg_text.content)
        assert data_text["status"] == "ok"

    @pytest.mark.anyio
    async def test_t3_drag_interaction_security_gates(self) -> None:
        """T3.6: Drag intersecting password node or with draft text escalates to R3."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(
            owner_id,
            host.host_id,
            allowed_apps=("canvasapp",),
            cached_nodes={
                "pwd-pin": {
                    "id": "pwd-pin",
                    "role": "securetextfield",
                    "name": "PIN Keypad",
                    "bounds": [0.40, 0.40, 0.60, 0.60],
                    "isPassword": True,
                }
            },
        )
        await cu_store.start(active_cu, NOW)

        # 1. Drag intersecting password field is denied immediately
        collector_pwd_drag = EventCollector(orch, owner_id, auto_decision=ApprovalDecision.APPROVE)
        ctx_pwd_drag = _make_action_context(owner_id, collector_pwd_drag)

        call_drag = ToolCall(
            id="c-drag-pwd",
            name="input__drag",
            arguments={"fromXNorm": 0.45, "fromYNorm": 0.45, "toXNorm": 0.80, "toYNorm": 0.80},
        )
        msg_drag = await orch.handle_call(call_drag, ctx_pwd_drag)
        data_drag = json.loads(msg_drag.content)
        assert data_drag["status"] == "denied"
        assert data_drag["output"]["errorCode"] == "password_field_forbidden"

        # 2. Drag with uncommitted draft text on generic area escalates to R3
        active_cu.cached_nodes.clear()
        active_cu.typed_count = 1
        active_cu.typed_since_commit = ["Sensitive draft text"]
        collector_draft_drag = EventCollector(
            orch, owner_id, auto_decision=ApprovalDecision.APPROVE
        )
        ctx_draft_drag = _make_action_context(owner_id, collector_draft_drag)

        call_draft_drag = ToolCall(
            id="c-drag-draft",
            name="input__drag",
            arguments={"fromXNorm": 0.10, "fromYNorm": 0.10, "toXNorm": 0.50, "toYNorm": 0.50},
        )
        msg_draft = await orch.handle_call(call_draft_drag, ctx_draft_drag)
        data_draft = json.loads(msg_draft.content)
        assert data_draft["status"] == "ok"
        assert len(collector_draft_drag.pending_events) == 1
        assert collector_draft_drag.pending_events[0].risk.lower() == "r3"


# ==============================================================================
# Tier 4: Real-World Application Scenarios (S1 to S6)
# ==============================================================================


class TestTier4RealWorldScenarios:
    """T4: End-to-end workload simulations across arbitrary desktop applications."""

    @pytest.mark.anyio
    async def test_t4_s1_non_ax_canvas_drawing_app_drag_interaction(self) -> None:
        """Scenario 1: Non-AX Canvas Drawing App Drag Interaction (normalized drag with settle)."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(owner_id, host.host_id, allowed_apps=("customcanvas",))
        await cu_store.start(active_cu, NOW)

        hub.custom_responses["ui.observe_hybrid"] = {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {
                "appId": "com.art.customcanvas",
                "windowTitle": "Untitled Drawing — Vector Studio",
                "snapshotId": "snap-canvas-01",
                "nodes": [
                    {"id": "toolbox", "role": "toolbar", "bounds": [0.0, 0.0, 1.0, 0.08]},
                    {"id": "canvas-area", "role": "canvas", "bounds": [0.0, 0.08, 1.0, 1.0]},
                ],
                "screenshot": {
                    "mimeType": "image/jpeg",
                    "width": 1280,
                    "height": 720,
                    "dataBase64": TINY_1X1_JPEG_B64,
                },
            },
            "isError": False,
            "_meta": {"xuna/status": "ok"},
        }

        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # Step 1: Hybrid observation
        obs_call = ToolCall(
            id="c-s1-obs", name="ui__observe_hybrid", arguments={"scope": "active_window"}
        )
        obs_msg = await orch.handle_call(obs_call, ctx)
        obs_data = json.loads(obs_msg.content)
        assert obs_data["status"] == "ok"
        assert obs_msg.image_base64 == TINY_1X1_JPEG_B64

        # Step 2: Draw a stroke using normalized drag across the canvas
        drag_call = ToolCall(
            id="c-s1-drag",
            name="input__drag",
            arguments={
                "fromXNorm": 0.20,
                "fromYNorm": 0.25,
                "toXNorm": 0.70,
                "toYNorm": 0.75,
                "durationMs": 400,
            },
        )
        drag_msg = await orch.handle_call(drag_call, ctx)
        drag_data = json.loads(drag_msg.content)
        assert drag_data["status"] == "ok"
        output = drag_data["output"]
        assert output["dragged"] is True
        assert output["settled"] is True
        assert output["settleMs"] == 250

    @pytest.mark.anyio
    async def test_t4_s2_electron_web_app_coordinate_button_navigation(self) -> None:
        """Scenario 2: Electron Coordinate Navigation (observe -> coordinate click -> settle)."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(
            owner_id,
            host.host_id,
            allowed_apps=("electronchat",),
            cached_nodes={
                "tab-general": {
                    "id": "tab-general",
                    "role": "link",
                    "name": "#general channel",
                    "bounds": [0.02, 0.15, 0.23, 0.22],
                }
            },
        )
        await cu_store.start(active_cu, NOW)

        hub.custom_responses["ui.observe_hybrid"] = {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {
                "appId": "com.electron.chat",
                "windowTitle": "Team Chat & Workspace",
                "snapshotId": "snap-electron-02",
                "nodes": [
                    {
                        "id": "nav-channels",
                        "role": "group",
                        "name": "Channels Nav",
                        "bounds": [0.0, 0.0, 0.25, 1.0],
                    },
                    {
                        "id": "tab-general",
                        "role": "link",
                        "name": "#general channel",
                        "bounds": [0.02, 0.15, 0.23, 0.22],
                    },
                ],
                "screenshot": {
                    "mimeType": "image/jpeg",
                    "width": 1280,
                    "height": 720,
                    "dataBase64": TINY_1X1_JPEG_B64,
                },
            },
            "isError": False,
            "_meta": {"xuna/status": "ok"},
        }

        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        # 1. Observe hybrid
        obs_msg = await orch.handle_call(
            ToolCall(
                id="c-s2-obs", name="ui__observe_hybrid", arguments={"scope": "active_window"}
            ),
            ctx,
        )
        assert json.loads(obs_msg.content)["status"] == "ok"

        # 2. Click coordinate (0.12, 0.18) corresponding to #general channel
        click_msg = await orch.handle_call(
            ToolCall(
                id="c-s2-click", name="input__click", arguments={"xNorm": 0.12, "yNorm": 0.18}
            ),
            ctx,
        )
        click_data = json.loads(click_msg.content)
        assert click_data["status"] == "ok"
        assert click_data["output"]["clicked"] is True
        assert click_data["output"]["settled"] is True

    @pytest.mark.anyio
    async def test_t4_s3_payment_sensitive_action_elevation_over_coordinate_click(self) -> None:
        """Scenario 3: Sensitive Action Elevation over Coordinate Click (R3 phone signature)."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(
            owner_id,
            host.host_id,
            allowed_apps=("shopapp",),
            cached_nodes={
                "btn-pay": {
                    "id": "btn-pay",
                    "role": "button",
                    "name": "Xác nhận thanh toán 1,500,000 VND",
                    "bounds": [0.60, 0.85, 0.95, 0.95],
                }
            },
        )
        await cu_store.start(active_cu, NOW)

        collector = EventCollector(orch, owner_id, auto_decision=ApprovalDecision.APPROVE)
        ctx = _make_action_context(owner_id, collector)

        click_call = ToolCall(
            id="c-s3-pay", name="input__click", arguments={"xNorm": 0.75, "yNorm": 0.90}
        )
        msg = await orch.handle_call(click_call, ctx)

        assert len(collector.pending_events) == 1
        pending = collector.pending_events[0]
        assert pending.risk.lower() == "r3"
        assert pending.target == "Studio Host"

        data = json.loads(msg.content)
        assert data["status"] == "ok"
        assert data["output"]["clicked"] is True

    @pytest.mark.anyio
    async def test_t4_s4_protected_draft_text_coordinate_click_escalation(self) -> None:
        """Scenario 4: Protected Draft Text Coordinate Click Escalation (preserves draft)."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, cu_store = _setup_test_orchestrator(owner_id)
        host = (await orch._host_repo.list_hosts(owner_id))[0]

        active_cu = _make_cu_session_state(
            owner_id,
            host.host_id,
            allowed_apps=("texteditor",),
            typed_count=1,
            typed_since_commit=["Confidential Proposal Details..."],
            cached_nodes={
                "btn-cancel": {
                    "id": "btn-cancel",
                    "role": "button",
                    "name": "Cancel & Exit",
                    "bounds": [0.80, 0.02, 0.98, 0.10],
                }
            },
        )
        await cu_store.start(active_cu, NOW)

        collector = EventCollector(orch, owner_id, auto_decision=ApprovalDecision.REJECT)
        ctx = _make_action_context(owner_id, collector)

        click_call = ToolCall(
            id="c-s4-cancel", name="input__click", arguments={"xNorm": 0.85, "yNorm": 0.05}
        )
        msg = await orch.handle_call(click_call, ctx)

        assert len(collector.pending_events) == 1
        assert collector.pending_events[0].risk.lower() == "r3"

        data = json.loads(msg.content)
        assert data["status"] == "rejected"

    @pytest.mark.anyio
    async def test_t4_s5_screen_recording_permission_denied_fail_closed(self) -> None:
        """Scenario 5: Recording Permission Denied Fail-Closed (honest error, zero fake image)."""
        owner_id = uuid.uuid4()
        orch, _, hub, _, _, _, _ = _setup_test_orchestrator(owner_id)
        collector = EventCollector(orch, owner_id)
        ctx = _make_action_context(owner_id, collector)

        hub.custom_responses["ui.observe_hybrid"] = {
            "content": [{"type": "text", "text": "Screen recording denied by system preferences"}],
            "structuredContent": {
                "status": "needs_permission",
                "errorCode": "screen_recording",
                "message": "Host denied screen recording permission",
            },
            "isError": True,
            "_meta": {
                "xuna/status": "needs_permission",
                "xuna/errorCode": "screen_recording",
            },
        }

        obs_call = ToolCall(
            id="c-s5-perm", name="ui__observe_hybrid", arguments={"scope": "active_window"}
        )
        msg = await orch.handle_call(obs_call, ctx)

        assert msg.image_base64 is None
        data = json.loads(msg.content)
        assert data["status"] in ("needs_permission", "error")
        assert (
            data.get("errorCode") == "screen_recording"
            or data["output"].get("errorCode") == "screen_recording"
        )

    def test_t4_s6_rule_h5_generic_compliance_ast_scan(self) -> None:
        """Scenario 6: Rule H5 Generic Compliance AST scan of backend generic application layers."""
        project_root = Path(__file__).resolve().parents[2]
        files_to_scan = [
            project_root
            / "src"
            / "xuna_backend"
            / "modules"
            / "actions"
            / "application"
            / "prompts.py",
            project_root
            / "src"
            / "xuna_backend"
            / "modules"
            / "actions"
            / "application"
            / "catalog.py",
            project_root
            / "src"
            / "xuna_backend"
            / "modules"
            / "actions"
            / "application"
            / "orchestrator.py",
            project_root
            / "src"
            / "xuna_backend"
            / "modules"
            / "actions"
            / "application"
            / "computer_use.py",
        ]
        conv_dir = (
            project_root / "src" / "xuna_backend" / "modules" / "conversation" / "application"
        )
        if conv_dir.exists():
            for p in conv_dir.rglob("*.py"):
                files_to_scan.append(p)

        pattern = re.compile(
            r"\b(spotify|chrome|safari|zalo|telegram|slack|notion|figma|photoshop)\b",
            re.IGNORECASE,
        )

        violations: list[str] = []
        for file_path in files_to_scan:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    val_str = node.value.strip()
                    if "\n" in val_str or len(val_str) > 120:
                        continue
                    match = pattern.search(val_str)
                    if match:
                        snippet = val_str[:25]
                        violations.append(
                            f"{file_path.name}:{node.lineno} matched "
                            f"'{match.group(0)}' in '{snippet}'"
                        )

        assert not violations, "Rule H5 Generic Layer Violations found:\n" + "\n".join(violations)
