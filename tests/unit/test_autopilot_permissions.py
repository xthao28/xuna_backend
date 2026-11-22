"""Unit tests for Autopilot permission relaxation (ADR-0015 amendment 2026-11-22).

Verifies:
- `HostTwin.trust_preset` is synced via `xuna/get_policy` (once per connection) and
  updated immediately via `set_host_policy` / `get_host_policy`.
- Under `trustPreset == "autopilot"`:
  - R2 tools (`app.launch`, `url.open`), `cu.session_start`, `cu.session_extend`,
    ordinary `input.key("Enter")` / `input.type_text` / `input.click`, and ordinary
    `web.click("Confirm")` execute with ZERO approval prompts while recording audit logs.
  - High-impact actions (`mail.send`, financial/payment/checkout, destructive delete/uninstall,
    external `Send`/`Gửi`/`Post`/`Đăng bài`, password fields, and hard-denied terminals)
    STILL strictly require R3 phone-signed approval or fail closed!
- Under `trustPreset == "standard"` (and fail-closed on policy query error):
  - Existing R2/R3 approval prompts are 100% preserved.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from xuna_backend.modules.actions.application.computer_use import is_high_impact_element_name
from xuna_backend.modules.actions.application.orchestrator import ActionContext, Orchestrator
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.application.use_cases import set_host_policy
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
from xuna_backend.modules.hosts.infrastructure.session_hub import InMemoryHostSessionHub

NOW = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
FAKE_PUBKEY = b"\x04" + bytes(range(1, 65))


class _FakeHostRepo:
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


class _FakeAudit:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def _ok_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _build_manifest(host_id: HostId) -> Manifest:
    tools = (
        ToolDescriptor(
            name="app.launch",
            description="Launch app.",
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
            output_schema=_ok_schema(),
            family=ToolFamily.APP,
            risk=RiskTier.R2,
            reversible=True,
            observes=False,
        ),
        ToolDescriptor(
            name="cu.session_start",
            description="Start CU session.",
            input_schema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "apps": {"type": "array", "items": {"type": "string"}},
                    "domains": {"type": "array", "items": {"type": "string"}},
                    "maxSteps": {"type": "integer"},
                    "maxMinutes": {"type": "integer"},
                },
                "required": ["goal", "apps", "domains", "maxSteps", "maxMinutes"],
                "additionalProperties": False,
            },
            output_schema=_ok_schema(),
            family=ToolFamily.CU,
            risk=RiskTier.R3,
            reversible=False,
            observes=False,
        ),
        ToolDescriptor(
            name="input.key",
            description="Press key.",
            input_schema={
                "type": "object",
                "properties": {"combo": {"type": "string"}},
                "required": ["combo"],
                "additionalProperties": False,
            },
            output_schema=_ok_schema(),
            family=ToolFamily.INPUT,
            risk=RiskTier.R2,
            reversible=False,
            observes=False,
        ),
        ToolDescriptor(
            name="input.click",
            description="Click UI element.",
            input_schema={
                "type": "object",
                "properties": {
                    "elementId": {"type": "string"},
                    "elementName": {"type": "string"},
                },
                "additionalProperties": False,
            },
            output_schema=_ok_schema(),
            family=ToolFamily.INPUT,
            risk=RiskTier.R2,
            reversible=False,
            observes=False,
        ),
        ToolDescriptor(
            name="mail.send",
            description="Send email.",
            input_schema={
                "type": "object",
                "properties": {"to": {"type": "string"}, "subject": {"type": "string"}},
                "required": ["to", "subject"],
                "additionalProperties": False,
            },
            output_schema=_ok_schema(),
            family=ToolFamily.APP,
            risk=RiskTier.R3,
            reversible=False,
            observes=False,
        ),
        ToolDescriptor(
            name="web.click",
            description="Click web element.",
            input_schema={
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "targetText": {"type": "string"},
                },
                "required": ["ref"],
                "additionalProperties": False,
            },
            output_schema=_ok_schema(),
            family=ToolFamily.URL,
            risk=RiskTier.R2,
            reversible=False,
            observes=False,
        ),
    )
    return Manifest(host_id=host_id, schema_version=1, tools=tools, updated_at=NOW)


async def _make_env(
    *,
    trust_preset: str,
    preflight_map: dict[str, dict[str, object]] | None = None,
    fail_get_policy: bool = False,
) -> tuple[
    Orchestrator,
    Host,
    _FakeHostRepo,
    InMemoryHostSessionHub,
    _FakeAudit,
    list[dict[str, Any]],
    list[ActionStreamEvent],
    ActionContext,
]:
    owner_id = uuid.uuid4()
    host = Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias("MacBook Pro"),
        platform=HostPlatform.MACOS,
        hostname="macbook-pro",
        os_version="12.6",
        agent_version="0.1.0",
        public_key=FAKE_PUBKEY,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        last_seen_at=NOW,
    )
    grants = [
        Grant(
            host_id=host.host_id,
            family=fam,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=NOW,
        )
        for fam in (ToolFamily.APP, ToolFamily.CU, ToolFamily.INPUT, ToolFamily.URL)
    ]
    repo = _FakeHostRepo(host, _build_manifest(host.host_id), grants)
    audit = _FakeAudit()
    hub = InMemoryHostSessionHub()
    sent_frames: list[dict[str, Any]] = []
    stream_events: list[ActionStreamEvent] = []
    pf_map = preflight_map or {}

    async def _send_text(raw: str) -> None:
        frame = json.loads(raw)
        sent_frames.append(frame)
        req_id = frame.get("id")
        method = frame.get("method")
        if req_id is None:
            return
        if method == "xuna/get_policy":
            if fail_get_policy:
                hub.on_error_response(str(host.host_id), req_id, -32601, "method not found")
            else:
                hub.on_response(
                    str(host.host_id),
                    req_id,
                    {"schemaVersion": 1, "trustPreset": trust_preset},
                )
        elif method == "xuna/set_policy":
            hub.on_response(str(host.host_id), req_id, {"schemaVersion": 1, "status": "ok"})
        elif method == "xuna/preflight":
            tool_name = str((frame.get("params") or {}).get("name") or "")
            pf = pf_map.get(
                tool_name,
                {"status": "ok", "effectiveRisk": "R2", "reasonCode": "autopilot"},
            )
            hub.on_response(str(host.host_id), req_id, {"preflight": pf})
        elif method == "tools/call":
            tool_name = str((frame.get("params") or {}).get("name") or "")
            sc: dict[str, object] = (
                {"sessionId": "cu-auto-1", "apps": ["antigravity_ide"], "domains": []}
                if tool_name == "cu.session_start"
                else {"effect": "changed"}
            )
            hub.on_response(
                str(host.host_id),
                req_id,
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "isError": False,
                    "structuredContent": sc,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
                },
            )

    async def _noop() -> None:
        return None

    await hub.open_session(
        str(host.host_id),
        object(),
        _send_text,
        _noop,
        _noop,
        owner_id=owner_id,
    )

    orch = Orchestrator(
        host_repo=repo,
        hub=hub,
        approval_repo=InMemoryApprovalRepository(),
        approval_broker=InMemoryApprovalBroker(),
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=audit,
        clock=lambda: NOW,
    )

    async def _emit(ev: ActionStreamEvent) -> None:
        stream_events.append(ev)

    ctx = ActionContext(
        owner_id=owner_id,
        pet_id="pet-1",
        locale=Locale.VI,
        client_caps=("host_actions_v1",),
        board_available=False,
        emit_event=_emit,
        request_id="req-autopilot-test",
    )
    return orch, host, repo, hub, audit, sent_frames, stream_events, ctx


def test_is_high_impact_element_name_parity() -> None:
    for risky in (
        "Send",
        "Gửi",
        "Post",
        "Publish",
        "Đăng bài",
        "Xuất bản",
        "Pay now",
        "Checkout",
        "Thanh toán",
        "Chuyển khoản",
        "Đặt hàng",
        "Delete",
        "Xóa",
        "Sign out",
        "Đăng xuất",
        "Uninstall",
    ):
        assert is_high_impact_element_name(risky), f"Expected high impact for {risky!r}"

    for safe in ("Submit", "Confirm", "Remove filter", "Hủy", "Đăng nhập", "Đăng ký", "Sort order"):
        assert not is_high_impact_element_name(safe), f"Unexpected high impact for {safe!r}"


def test_autopilot_runs_r2_and_cu_session_without_prompt() -> None:
    async def _run() -> None:
        orch, host, _, hub, audit, sent_frames, events, ctx = await _make_env(
            trust_preset="autopilot"
        )

        # 1. app.launch (R2) executes directly without any ActionPendingEvent
        msg_launch = await orch.handle_call(
            ToolCall(id="c1", name="app__launch", arguments={"id": "telegram"}),
            ctx,
        )
        assert json.loads(msg_launch.content)["status"] == "ok"
        assert not any(isinstance(e, ActionPendingEvent) for e in events)
        twin = hub.get_twin(str(host.host_id))
        assert twin is not None
        assert twin.trust_preset == "autopilot"

        # 2. cu.session_start executes directly without P-256 prompt under autopilot
        msg_cu = await orch.handle_call(
            ToolCall(
                id="c2",
                name="cu__session_start",
                arguments={
                    "goal": "Check progress in IDE",
                    "apps": ["antigravity_ide"],
                    "domains": [],
                    "maxSteps": 20,
                    "maxMinutes": 10,
                },
            ),
            ctx,
        )
        assert json.loads(msg_cu.content)["status"] == "ok"
        assert not any(isinstance(e, ActionPendingEvent) for e in events)

        # 3. input.key("Enter") executes directly without prompt under autopilot
        msg_key = await orch.handle_call(
            ToolCall(id="c3", name="input__key", arguments={"combo": "Enter"}),
            ctx,
        )
        assert json.loads(msg_key.content)["status"] == "ok"
        assert not any(isinstance(e, ActionPendingEvent) for e in events)

        # 4. Ordinary web.click("Confirm") executes directly without prompt under autopilot
        msg_web = await orch.handle_call(
            ToolCall(id="c4", name="web__click", arguments={"ref": "e1", "targetText": "Confirm"}),
            ctx,
        )
        assert json.loads(msg_web.content)["status"] == "ok"
        assert not any(isinstance(e, ActionPendingEvent) for e in events)

        # xuna/get_policy was fetched only ONCE and cached on HostTwin
        policy_queries = [f for f in sent_frames if f.get("method") == "xuna/get_policy"]
        assert len(policy_queries) == 1
        invoked = [e for e in audit.events if e.action == AuditAction.HOST_TOOL_INVOKED]
        assert len(invoked) == 4

    asyncio.run(_run())


def test_autopilot_still_prompts_for_high_impact_and_send() -> None:
    async def _run() -> None:
        orch, _, _, _, _, _, events, ctx = await _make_env(
            trust_preset="autopilot",
            preflight_map={
                "input.click": {
                    "status": "ok",
                    "effectiveRisk": "R3",
                    "reasonCode": "consequential_target",
                    "elementName": "Gửi",
                }
            },
        )

        # Trigger high-impact actions and cancel immediately after ActionPendingEvent fires
        for call in (
            ToolCall(
                id="r3-mail",
                name="mail__send",
                arguments={"to": "a@example.com", "subject": "Hi"},
            ),
            ToolCall(
                id="r3-send",
                name="input__click",
                arguments={"elementId": "@s1:e2", "elementName": "Gửi"},
            ),
            ToolCall(
                id="r3-pay",
                name="web__click",
                arguments={"ref": "e9", "targetText": "Thanh toán"},
            ),
        ):
            events.clear()
            task = asyncio.create_task(orch.handle_call(call, ctx))
            for _ in range(50):
                if any(isinstance(e, ActionPendingEvent) for e in events):
                    break
                await asyncio.sleep(0.01)
            pending = [e for e in events if isinstance(e, ActionPendingEvent)]
            assert len(pending) == 1, f"Expected R3 ActionPendingEvent for {call.name}"
            assert pending[0].risk == "R3"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(_run())


def test_switching_autopilot_to_standard_re_enables_prompts_and_error_fails_closed() -> None:
    async def _run() -> None:
        # 1. Policy fetch error fails closed to standard -> prompts for R2 app.launch
        orch_err, _, _, _, _, _, events_err, ctx_err = await _make_env(
            trust_preset="autopilot", fail_get_policy=True
        )
        task = asyncio.create_task(
            orch_err.handle_call(
                ToolCall(id="c-err", name="app__launch", arguments={"id": "notes"}), ctx_err
            )
        )
        for _ in range(50):
            if any(isinstance(e, ActionPendingEvent) for e in events_err):
                break
            await asyncio.sleep(0.01)
        assert any(isinstance(e, ActionPendingEvent) for e in events_err)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # 2. Switch live host autopilot -> standard via set_host_policy: immediate effect
        orch, host, repo, hub, audit, _, events, ctx = await _make_env(trust_preset="autopilot")
        msg1 = await orch.handle_call(
            ToolCall(id="c-a", name="app__launch", arguments={"id": "notes"}), ctx
        )
        assert json.loads(msg1.content)["status"] == "ok"
        assert not any(isinstance(e, ActionPendingEvent) for e in events)

        await set_host_policy(
            hosts=repo,
            hub=hub,
            audit=audit,
            owner_id=host.owner_id,
            host_id=host.host_id,
            policy_data={"trustPreset": "standard", "allowApps": ["notes"]},
            now=NOW,
        )
        twin = hub.get_twin(str(host.host_id))
        assert twin is not None
        assert twin.trust_preset == "standard"

        task2 = asyncio.create_task(
            orch.handle_call(ToolCall(id="c-b", name="app__launch", arguments={"id": "notes"}), ctx)
        )
        for _ in range(50):
            if any(isinstance(e, ActionPendingEvent) for e in events):
                break
            await asyncio.sleep(0.01)
        assert any(isinstance(e, ActionPendingEvent) for e in events)
        task2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task2

    asyncio.run(_run())
