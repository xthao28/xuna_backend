"""UDC-17 S17.1: the host decides the risk of an input step (`xuna/preflight`).

The backend asks the host before any approval, only ever raises the host's answer, shows the
live element label on the approval card, falls back to its legacy estimate only for hosts that
predate preflight, and treats every other failure as R3.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.helpers import SimulatedDevice
from tests.modules.hosts.test_hosts_use_cases import FakeHostRepository

from xuna_backend.modules.actions.application import orchestrator as orchestrator_module
from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    InMemoryComputerUseSessionStore,
    is_consequential_element_name,
)
from xuna_backend.modules.actions.application.orchestrator import ActionContext, Orchestrator
from xuna_backend.modules.actions.domain.action_summary import ActionSummary
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionResultEvent,
    ActionStreamEvent,
    ApprovalDecision,
    ApprovalScope,
    ComputerUseSessionEvent,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.application.xtp_frames import parse_preflight_result
from xuna_backend.modules.hosts.domain.crypto import approval_proof_message
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
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier

NOW = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
_EMPTY: dict[str, object] = {"type": "object", "properties": {}, "additionalProperties": False}

# What the fake host answers to `xuna/preflight`, per call.
Answer = dict[str, Any] | str  # a `preflight` body, or "method_not_found" / "silent" / "malformed"


class _AuditLog:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def record(self, event: Any) -> None:
        self.events.append(event)

    async def recent(self, owner_id: uuid.UUID, limit: int) -> list[Any]:
        return self.events[:limit]


@dataclass
class _World:
    orch: Orchestrator
    ctx: ActionContext
    owner_id: uuid.UUID
    host: Host
    cu_store: InMemoryComputerUseSessionStore
    audit: _AuditLog
    answers: list[Answer]
    preflights: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pending: list[ActionPendingEvent] = field(default_factory=list)
    results: list[ActionResultEvent] = field(default_factory=list)
    sessions: list[ComputerUseSessionEvent] = field(default_factory=list)
    # What the fake host returns for `cu.session_extend` (its full scope after extending).
    host_scope: tuple[list[str], list[str]] = (["explorer"], [])
    host_calls: list[dict[str, Any]] = field(default_factory=list)
    approvals: InMemoryApprovalRepository | None = None

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        msg = await self.orch.handle_call(
            ToolCall(id=f"c-{uuid.uuid4().hex[:6]}", name=name, arguments=arguments), self.ctx
        )
        body: dict[str, Any] = json.loads(msg.content)
        return body


def _descriptor(
    name: str,
    risk: RiskTier,
    properties: dict[str, object],
    family: ToolFamily = ToolFamily.INPUT,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        family=family,
        risk=risk,
        description=name,
        input_schema={"type": "object", "properties": properties, "additionalProperties": False},
        output_schema=_EMPTY,
    )


async def _build(
    answers: list[Answer],
    *,
    platform: HostPlatform = HostPlatform.WINDOWS,
    input_ceiling: RiskTier = RiskTier.R3,
    clock: Callable[[], datetime] | None = None,
    extra_tools: tuple[ToolDescriptor, ...] = (),
    structured_for: Callable[[str, dict[str, Any]], dict[str, Any] | None] | None = None,
) -> _World:
    owner_id = uuid.uuid4()
    repo = FakeHostRepository()
    hub = InMemoryHostSessionHub()
    phone = SimulatedDevice()
    host = Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias("PC Thảo"),
        platform=platform,
        hostname="pc-thao",
        os_version="10.0",
        agent_version="0.3.0",
        public_key=SimulatedDevice().public_key,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        last_seen_at=NOW,
    )
    phone_host = Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias("Điện thoại"),
        platform=HostPlatform.IOS,
        hostname="phone",
        os_version="18.0",
        agent_version="0.3.0",
        public_key=phone.public_key,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        last_seen_at=NOW,
    )
    await repo.save_host(host)
    await repo.save_host(phone_host)
    element_props: dict[str, object] = {
        "elementId": {"type": "string"},
        "elementName": {"type": "string"},
    }
    await repo.save_manifest(
        Manifest(
            host_id=host.host_id,
            tools=(
                _descriptor("input.click", RiskTier.R2, element_props),
                _descriptor(
                    "input.type_text",
                    # A desktop agent with ADR-0015 D1 declares R2 and raises per call.
                    RiskTier.R2,
                    {**element_props, "text": {"type": "string"}},
                ),
                _descriptor(
                    "input.key",
                    RiskTier.R2,
                    {"combo": {"type": "string"}, "inForm": {"type": "boolean"}},
                ),
                _descriptor(
                    "cu.session_extend",
                    RiskTier.R3,
                    {
                        "sessionId": {"type": "string"},
                        "apps": {"type": "array", "items": {"type": "string"}},
                        "domains": {"type": "array", "items": {"type": "string"}},
                    },
                    ToolFamily.CU,
                ),
                _descriptor(
                    "cu.session_end",
                    RiskTier.R0,
                    {"sessionId": {"type": "string"}, "outcome": {"type": "string"}},
                    ToolFamily.CU,
                ),
                *extra_tools,
            ),
            updated_at=NOW,
        )
    )
    await repo.set_grants(
        host.host_id,
        [
            Grant(host.host_id, ToolFamily.INPUT, input_ceiling, True, NOW),
            Grant(host.host_id, ToolFamily.CU, RiskTier.R3, True, NOW),
            Grant(host.host_id, ToolFamily.UI, RiskTier.R2, True, NOW),
        ],
    )

    cu_store = InMemoryComputerUseSessionStore()
    approvals = InMemoryApprovalRepository()
    state = ComputerUseSessionState(
        session_id="cu-1",
        owner_id=owner_id,
        host_id=host.host_id,
        goal="Gửi báo cáo",
        allowed_apps=("explorer",),
        allowed_domains=(),
        max_steps=20,
        max_minutes=10,
        started_at_utc=NOW,
        last_heartbeat_at=NOW,
    )
    # What the backend cached from the last ui.inspect: the legacy estimate reads this.
    state.cached_nodes["@s1:e3"] = {"id": "3", "role": "button", "name": "Xóa ghi chú"}
    await cu_store.start(state, NOW)

    audit = _AuditLog()
    world_ref: dict[str, _World] = {}

    async def _send_text(raw: str) -> None:
        frame = json.loads(raw)
        world = world_ref["w"]
        method = frame.get("method")
        if method == "xuna/preflight":
            world.preflights.append(frame["params"])
            answer = world.answers.pop(0) if world.answers else "method_not_found"
            if answer == "method_not_found":
                hub.on_error_response(str(host.host_id), frame["id"], -32601, "Method not found")
            elif answer == "silent":
                return
            elif answer == "malformed":
                hub.on_response(
                    str(host.host_id),
                    frame["id"],
                    {"preflight": {"status": "ok", "effectiveRisk": "R2", "note": "trust me"}},
                )
            else:
                hub.on_response(str(host.host_id), frame["id"], {"preflight": answer})
            return
        if method == "tools/call":
            world.tool_calls.append(frame["params"])
            name = frame["params"]["name"]
            structured: dict[str, Any] = {"effect": "changed"}
            if name == "cu.session_extend":
                structured = {
                    "sessionId": frame["params"]["arguments"]["sessionId"],
                    "apps": world.host_scope[0],
                    "domains": world.host_scope[1],
                }
            elif name == "cu.session_end":
                structured = {"ended": True}
            if structured_for is not None:
                scripted = structured_for(name, frame["params"]["arguments"])
                if scripted is not None:
                    structured = scripted
            hub.on_response(
                str(host.host_id),
                frame["id"],
                {
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 3},
                    "structuredContent": structured,
                },
            )

    async def _noop() -> None:
        return None

    await hub.open_session(str(host.host_id), object(), _send_text, _noop, _noop)

    orch = Orchestrator(
        host_repo=repo,
        hub=hub,
        approval_repo=approvals,
        approval_broker=InMemoryApprovalBroker(),
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=audit,
        scope_store=InMemoryApprovalScopeStore(),
        verifier=P256SignatureVerifier(),
        cu_session_store=cu_store,
        clock=clock or (lambda: NOW),
    )

    async def _sink(event: ActionStreamEvent) -> None:
        if isinstance(event, ComputerUseSessionEvent):
            world_ref["w"].sessions.append(event)
            return
        if isinstance(event, ActionResultEvent):
            world_ref["w"].results.append(event)
            return
        if not isinstance(event, ActionPendingEvent):
            return
        world_ref["w"].pending.append(event)
        signature = None
        signer = None
        if event.risk == "R3":
            assert event.host_id is not None
            signature = phone.sign(
                approval_proof_message(
                    event.approval_id, event.args_sha256, event.host_id, event.expires_at
                )
            )
            signer = phone.public_key
        await orch.decide_approval(
            approval_id=uuid.UUID(event.approval_id),
            owner_id=owner_id,
            decision=ApprovalDecision.APPROVE,
            scope=ApprovalScope.ONCE,
            signature=signature,
            signer_public_key=signer,
        )

    ctx = ActionContext(
        owner_id=owner_id,
        pet_id="pet-1",
        locale=Locale.VI,
        client_caps={"host_actions_v1"},
        board_available=False,
        emit_event=_sink,
        host_screen_consent=True,
    )
    world = _World(
        orch=orch,
        ctx=ctx,
        owner_id=owner_id,
        host=host,
        cu_store=cu_store,
        audit=audit,
        answers=list(answers),
        approvals=approvals,
    )
    world_ref["w"] = world
    return world


def _ok(risk: str, reason: str, name: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"status": "ok", "effectiveRisk": risk, "reasonCode": reason}
    if name is not None:
        body["element"] = {"role": "button", "name": name}
    return body


def _run(coro_factory: Callable[[], Any]) -> None:
    asyncio.run(coro_factory())


def test_host_raises_a_benign_looking_click_to_r3_and_the_card_shows_the_live_label() -> None:
    async def scenario() -> None:
        w = await _build([_ok("R3", "consequential_target", "Gửi báo cáo")])
        body = await w.call("input__click", {"elementId": "@s1:e2", "elementName": "Mở"})
        assert body["status"] == "ok"
        assert w.preflights == [
            {"name": "input.click", "arguments": {"elementId": "@s1:e2", "elementName": "Mở"}}
        ]
        assert len(w.pending) == 1
        assert w.pending[0].risk == "R3"
        assert "Gửi báo cáo" in w.pending[0].summary
        assert "'Mở'" not in w.pending[0].summary
        # The chat chip after the step names the same element the owner signed for.
        assert "Gửi báo cáo" in w.results[-1].summary
        assert "'Mở'" not in w.results[-1].summary
        # The approval is bound to the model's real arguments, not to the display label.
        assert w.tool_calls[-1]["arguments"] == {"elementId": "@s1:e2", "elementName": "Mở"}
        assert "signature" in w.tool_calls[-1]["_meta"]["xuna/approval"]
        # Screen content never reaches the audit trail.
        assert all("Gửi báo cáo" not in json.dumps(e.metadata) for e in w.audit.events)

    _run(scenario)


def test_host_answer_replaces_the_backend_cache_guess() -> None:
    async def scenario() -> None:
        # The backend cache says "Xóa ghi chú"; the host reads the live tree and says R2.
        w = await _build([_ok("R2", "navigation_target", "Mở thư mục")])
        body = await w.call("input__click", {"elementId": "@s1:e3"})
        assert body["status"] == "ok"
        assert w.pending == []
        assert "xuna/approval" not in w.tool_calls[-1]["_meta"]

    _run(scenario)


def test_declared_name_can_raise_but_never_lower_the_host_answer() -> None:
    async def scenario() -> None:
        w = await _build(
            [
                _ok("R2", "navigation_target", "Mở"),
                _ok("R2", "search_submit"),
                _ok("R3", "activation_key"),
            ],
        )
        await w.call("input__click", {"elementId": "@s1:e1", "elementName": "Gửi"})
        assert [p.risk for p in w.pending] == ["R3"]
        # ADR-0015 D2: the host saw focus in a search field, so Enter stays R2 ...
        await w.call("input__key", {"combo": "Enter"})
        assert [p.risk for p in w.pending] == ["R3"]
        # ... and the model's `inForm: false` cannot lower the host's R3.
        await w.call("input__key", {"combo": "Enter", "inForm": False})
        assert [p.risk for p in w.pending] == ["R3", "R3"]

    _run(scenario)


def test_unsigned_drafts_appear_on_the_next_commit_card() -> None:
    async def scenario() -> None:
        w = await _build(
            [
                _ok("R2", "draft_text", "Người nhận"),
                _ok("R2", "draft_text", "Soạn tin"),
                _ok("R3", "unclassified_target", "Trang web"),
                _ok("R3", "consequential_target", "Gửi"),
                _ok("R3", "consequential_target", "Gửi"),
            ]
        )
        assert (await w.call("input__type_text", {"elementId": "@s1:e1", "text": "Lan"}))[
            "status"
        ] == "ok"
        await w.call("input__type_text", {"elementId": "@s1:e2", "text": "Báo cáo xong rồi"})
        # Drafts ran without a card.
        assert w.pending == []
        # Typing the host could not classify is signed, and its text is on its own card,
        # without the earlier drafts (a typing card is not a commit).
        await w.call("input__type_text", {"elementId": "@s1:e5", "text": "ghi chú"})
        assert len(w.pending) == 1
        assert "ghi chú" in w.pending[0].summary
        assert "Lan" not in w.pending[0].summary
        # The commit card shows everything typed since the last commit.
        await w.call("input__click", {"elementId": "@s1:e3"})
        commit = w.pending[-1].summary
        assert "Gửi" in commit
        assert "Đã gõ: 'Lan / Báo cáo xong rồi / ghi chú'" in commit
        # After the signed commit the list starts again.
        await w.call("input__click", {"elementId": "@s1:e3"})
        assert "Đã gõ" not in w.pending[-1].summary
        # Draft text never reaches the audit trail.
        assert all("Báo cáo xong" not in json.dumps(e.metadata) for e in w.audit.events)

    _run(scenario)


def test_drafts_are_dropped_when_the_session_ends() -> None:
    async def scenario() -> None:
        w = await _build([_ok("R2", "draft_text", "Soạn tin")])
        await w.call("input__type_text", {"elementId": "@s1:e2", "text": "nháp"})
        state = await w.cu_store.get_active_for_owner(w.owner_id)
        assert state is not None and state.typed_since_commit == ["nháp"]
        await w.orch.emergency_stop(owner_id=w.owner_id)
        assert state.typed_since_commit == []

    _run(scenario)


def test_with_unsigned_drafts_fits_the_summary_limit() -> None:
    base = "Nhấp vào 'Gửi' trên PC Thảo"
    long = ActionSummary.with_typed_text(base, ["x" * 500, "y" * 10], Locale.VI)
    assert len(long) <= 240
    assert long.startswith(base + " · Đã gõ: 'xxx")
    assert long.endswith("…'")
    assert ActionSummary.with_typed_text(base, [], Locale.VI) == base
    assert "Typed: 'a b'" in ActionSummary.with_typed_text(base, ["a\n b"], Locale.EN)


def test_host_refusal_reaches_the_model_without_asking_the_owner() -> None:
    async def scenario() -> None:
        w = await _build(
            [
                {"status": "denied", "errorCode": "password_field_forbidden"},
                {"status": "failed", "errorCode": "stale_ref"},
                {"status": "denied", "errorCode": "cu_session_required"},
            ]
        )
        pwd = await w.call("input__type_text", {"elementId": "@s1:e4", "text": "secret"})
        assert pwd == {"status": "denied", "output": {"errorCode": "password_field_forbidden"}}
        stale = await w.call("input__click", {"elementId": "@s9:e1"})
        assert stale["status"] == "failed"
        assert stale["output"]["errorCode"] == "stale_ref"
        assert await w.cu_store.get_active_for_owner(w.owner_id) is not None
        gone = await w.call("input__click", {"elementId": "@s1:e1"})
        assert gone["output"]["errorCode"] == "cu_session_required"
        # The host has no session: the backend stops exempting R2 steps too.
        assert await w.cu_store.get_active_for_owner(w.owner_id) is None
        assert w.pending == []
        assert w.tool_calls == []

    _run(scenario)


def test_host_r3_against_an_r2_ceiling_is_denied_not_prompted() -> None:
    async def scenario() -> None:
        w = await _build([_ok("R3", "consequential_target", "Gửi")], input_ceiling=RiskTier.R2)
        body = await w.call("input__click", {"elementId": "@s1:e2"})
        assert body["status"] == "denied"
        assert w.pending == []
        assert w.tool_calls == []

    _run(scenario)


def test_malformed_or_missing_answers_are_r3_without_a_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator_module, "PREFLIGHT_TIMEOUT_SECONDS", 0.05)

    async def scenario() -> None:
        w = await _build(["malformed", "silent"])
        for idx in range(2):
            body = await w.call("input__click", {"elementId": f"@s1:e{idx + 1}"})
            assert body["status"] == "ok"
        assert [p.risk for p in w.pending] == ["R3", "R3"]

    _run(scenario)


def test_hosts_without_preflight_keep_the_legacy_estimate() -> None:
    async def scenario() -> None:
        w = await _build(["method_not_found"] * 3)
        state = await w.cu_store.get_active_for_owner(w.owner_id)
        assert state is not None
        state.cached_nodes["@s1:e1"] = {"id": "1", "role": "button", "name": "Mở thư mục"}
        # Legacy: the cached "Xóa ghi chú" escalates; a cached plain button stays R2.
        await w.call("input__click", {"elementId": "@s1:e3"})
        await w.call("input__click", {"elementId": "@s1:e1"})
        # S17.6: a ref the backend never read (no inspect yet, or a restart emptied the
        # cache) has an unknown label, so it may be a commit: R3.
        await w.call("input__click", {"elementId": "@s1:e9"})
        assert [p.risk for p in w.pending] == ["R3", "R3"]
        assert len(w.tool_calls) == 3

    _run(scenario)


def test_hosts_of_other_kinds_are_not_asked() -> None:
    async def scenario() -> None:
        w = await _build([], platform=HostPlatform.ANDROID)
        await w.call("input__click", {"elementId": "@s1:e1"})
        assert w.preflights == []

    _run(scenario)


@pytest.mark.parametrize(
    "body",
    [
        {"status": "ok"},
        {"status": "ok", "effectiveRisk": "R9"},
        {"status": "denied"},
        {"status": "denied", "errorCode": "stale_ref", "effectiveRisk": "R2"},
        {"status": "ok", "effectiveRisk": "R2", "reasonCode": "Looks safe"},
        {"status": "ok", "effectiveRisk": "R2", "element": {"role": "button", "name": "x" * 81}},
        {"status": "ok", "effectiveRisk": "R2", "element": {"role": "button", "name": "a\nb"}},
        {"status": "ok", "effectiveRisk": "R2", "extra": True},
    ],
)
def test_parse_preflight_result_rejects_contract_violations(body: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_preflight_result({"preflight": body})


def test_parse_preflight_result_reads_a_valid_answer() -> None:
    parsed = parse_preflight_result({"preflight": _ok("R3", "consequential_target", "Gửi báo cáo")})
    assert parsed.effective_risk == RiskTier.R3
    assert parsed.element_name == "Gửi báo cáo"
    with pytest.raises(ValueError):
        parse_preflight_result({"preflight": _ok("R2", "x"), "extra": 1})


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Đăng", True),
        ("Đăng bài", True),
        ("Đăng xuất", True),
        ("Đăng nhập", False),
        ("Đăng ký tài khoản", False),
        ("Mở", False),
    ],
)
def test_post_button_reads_as_a_commit_but_sign_in_does_not(label: str, expected: bool) -> None:
    assert is_consequential_element_name(label) is expected
