"""UDC-17 S17.6: the phone heartbeat (dead-man switch) on the orchestrator path, and draft
text that never reaches durable storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from tests.unit.test_udc17_preflight import NOW, _build, _ok, _run

from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    is_password_ui_node,
    session_expiry_reason,
)
from xuna_backend.modules.actions.application.prompts import state_driven_hint
from xuna_backend.modules.actions.domain.action_summary import ActionSummary
from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.cognition.domain.types import Locale
from xuna_backend.modules.hosts.domain.entities import HostId


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


def _state(**changes: object) -> ComputerUseSessionState:
    state = ComputerUseSessionState(
        session_id="s",
        owner_id=uuid.uuid4(),
        host_id=HostId.generate(),
        goal="g",
        allowed_apps=("notes",),
        allowed_domains=(),
        max_steps=10,
        max_minutes=5,
        started_at_utc=NOW,
        last_heartbeat_at=NOW,
    )
    for key, value in changes.items():
        setattr(state, key, value)
    return state


def test_expiry_rules() -> None:
    assert session_expiry_reason(_state(), NOW + timedelta(seconds=15)) is None
    assert (
        session_expiry_reason(_state(), NOW + timedelta(seconds=15, milliseconds=1))
        == "heartbeat_timeout"
    )
    fresh = NOW + timedelta(minutes=5, seconds=1)
    assert session_expiry_reason(_state(last_heartbeat_at=fresh), fresh) == "time_budget_exceeded"
    assert session_expiry_reason(_state(steps_used=10), NOW) == "step_budget_exceeded"
    assert session_expiry_reason(_state(steps_used=9), NOW) is None


def test_a_lost_heartbeat_ends_the_session_here_and_on_the_host() -> None:
    async def scenario() -> None:
        clock = _Clock()
        w = await _build([_ok("R2", "navigation_target", "Mở")], clock=clock)
        clock.now = NOW + timedelta(seconds=10)
        assert await w.orch.record_cu_heartbeat(owner_id=w.owner_id, session_id="cu-1")
        clock.now = NOW + timedelta(seconds=24)  # 14 s after the last heartbeat
        assert await w.orch.active_cu_session_event(w.owner_id) is not None
        assert w.tool_calls == []

        clock.now = NOW + timedelta(seconds=26)  # 16 s: three heartbeats missed
        assert await w.orch.active_cu_session_event(w.owner_id) is None
        assert w.tool_calls[-1]["name"] == "cu.session_end"
        assert w.tool_calls[-1]["arguments"] == {"sessionId": "cu-1", "outcome": "stopped"}
        stops = [e for e in w.audit.events if e.action == AuditAction.HOST_CU_SESSION_STOPPED]
        assert [e.metadata for e in stops] == [
            {"hostConfirmed": True, "reason": "heartbeat_timeout"}
        ]
        ended = await w.cu_store.get("cu-1")
        assert ended is not None and not ended.active
        assert ended.stop_reason == "heartbeat_timeout"
        assert ended.goal == ""

        # A heartbeat that arrives late does not revive the session.
        assert await w.orch.record_cu_heartbeat(owner_id=w.owner_id, session_id="cu-1") is None
        assert await w.cu_store.get_active_for_owner(w.owner_id) is None

        # Navigation is no longer exempt: the owner is asked again.
        await w.call("input__click", {"elementId": "@s1:e1"})
        assert [p.risk for p in w.pending] == ["R2"]

    _run(scenario)


def test_a_tool_call_after_a_lost_heartbeat_is_not_exempt() -> None:
    async def scenario() -> None:
        clock = _Clock()
        w = await _build([_ok("R2", "navigation_target", "Mở")], clock=clock)
        clock.now = NOW + timedelta(seconds=16)
        await w.call("input__click", {"elementId": "@s1:e1"})
        assert [p.risk for p in w.pending] == ["R2"]
        assert w.tool_calls[0]["name"] == "cu.session_end"

    _run(scenario)


def test_heartbeats_keep_a_session_alive() -> None:
    async def scenario() -> None:
        clock = _Clock()
        w = await _build([_ok("R2", "navigation_target", "Mở")], clock=clock)
        for second in range(5, 65, 5):
            clock.now = NOW + timedelta(seconds=second)
            assert await w.orch.record_cu_heartbeat(owner_id=w.owner_id, session_id="cu-1")
        await w.call("input__click", {"elementId": "@s1:e1"})
        assert w.pending == []
        assert [c["name"] for c in w.tool_calls] == ["input.click"]

    _run(scenario)


def test_a_heartbeat_for_someone_else_or_another_session_is_refused() -> None:
    async def scenario() -> None:
        w = await _build([])
        assert await w.orch.record_cu_heartbeat(owner_id=uuid.uuid4(), session_id="cu-1") is None
        assert await w.orch.record_cu_heartbeat(owner_id=w.owner_id, session_id="cu-x") is None
        assert await w.orch.record_cu_heartbeat(owner_id=w.owner_id, session_id="cu-1")

    _run(scenario)


def test_draft_text_reaches_the_card_but_never_the_stored_approval() -> None:
    async def scenario() -> None:
        w = await _build(
            [_ok("R2", "draft_text", "Soạn"), _ok("R3", "consequential_target", "Gửi")]
        )
        await w.call("input__type_text", {"elementId": "@s1:e2", "text": "mã mở cửa 4821"})
        await w.call("input__click", {"elementId": "@s1:e3"})
        card = w.pending[-1]
        assert card.risk == "R3"
        assert "mã mở cửa 4821" in card.summary
        assert w.approvals is not None
        stored = await w.approvals.get(uuid.UUID(card.approval_id))
        assert stored is not None
        assert "4821" not in stored.summary
        state = await w.cu_store.get("cu-1")
        assert state is not None and state.typed_count == 0

    _run(scenario)


def test_after_a_restart_the_commit_card_says_text_was_typed() -> None:
    async def scenario() -> None:
        w = await _build(
            [_ok("R2", "draft_text", "Soạn"), _ok("R3", "consequential_target", "Gửi")]
        )
        await w.call("input__type_text", {"elementId": "@s1:e2", "text": "Chào Lan"})
        state = await w.cu_store.get("cu-1")
        assert state is not None and state.typed_count == 1
        # A restart keeps the durable count but loses the process-local draft text.
        state.typed_since_commit.clear()
        await w.call("input__click", {"elementId": "@s1:e3"})
        summary = w.pending[-1].summary
        assert "Chào Lan" not in summary
        assert "Đã gõ 1 đoạn trước đó, không hiện lại được nội dung" in summary
        assert state.typed_count == 0

    _run(scenario)


def test_unseen_drafts_notice_fits_the_card() -> None:
    base = "x" * 240
    vi = ActionSummary.with_typed_text(base, [], Locale.VI, typed_count=3)
    assert len(vi) == 240
    assert vi.endswith("Đã gõ 3 đoạn trước đó, không hiện lại được nội dung")
    en = ActionSummary.with_typed_text("Click 'Send'", [], Locale.EN, typed_count=2)
    assert en == "Click 'Send' · Typed 2 part(s) earlier; the text can't be shown again"
    assert ActionSummary.with_typed_text("Click", [], Locale.VI, typed_count=0) == "Click"


def test_password_fields_are_recognised_by_what_they_are() -> None:
    # Real password fields, by flag, role, subrole or input type.
    assert is_password_ui_node({"role": "textfield", "secure": True})
    assert is_password_ui_node({"role": "AXTextField", "subrole": "AXSecureTextField"})
    assert is_password_ui_node({"role": "textbox", "inputType": "password"})
    # Editable fields named like one, and unknown roles (fail closed).
    assert is_password_ui_node({"role": "edit", "name": "Nhập mật khẩu"})
    assert is_password_ui_node({"role": "textarea", "name": "Mã PIN"})
    assert is_password_ui_node({"name": "Password"})
    # Buttons, links and labels that only mention a password.
    assert not is_password_ui_node({"role": "button", "name": "Quên mật khẩu?"})
    assert not is_password_ui_node({"role": "link", "name": "Forgot password?"})
    assert not is_password_ui_node({"role": "AXStaticText", "name": "Mật khẩu"})
    assert not is_password_ui_node({"role": "text", "name": "Mật khẩu"})


def test_a_forgot_password_button_can_be_clicked_but_typing_a_password_cannot() -> None:
    async def scenario() -> None:
        w = await _build([_ok("R2", "navigation_target", "Quên mật khẩu?")])
        click = await w.call(
            "input__click", {"elementId": "@s1:e1", "elementName": "Quên mật khẩu?"}
        )
        assert click["status"] == "ok"
        assert [c["name"] for c in w.tool_calls] == ["input.click"]

        typed = await w.call(
            "input__type_text",
            {"elementId": "@s1:e2", "elementName": "Mật khẩu", "text": "hunter2"},
        )
        assert typed == {"status": "denied", "output": {"errorCode": "password_field_forbidden"}}
        assert [c["name"] for c in w.tool_calls] == ["input.click"]

    _run(scenario)


def test_url_refusals_from_the_host_carry_a_hint() -> None:
    for code in ("url_not_public", "url_unresolvable"):
        hint = state_driven_hint({"errorCode": code})
        assert hint is not None and "retry" in hint


def _ended_audits(w: object) -> list[dict[str, object]]:
    events = w.audit.events  # type: ignore[attr-defined]
    return [e.metadata for e in events if e.action == AuditAction.HOST_CU_SESSION_ENDED]


def test_a_session_end_is_audited_with_counts_only() -> None:
    async def scenario() -> None:
        w = await _build(
            [_ok("R2", "draft_text", "Soạn"), _ok("R3", "consequential_target", "Gửi")]
        )
        await w.call("input__type_text", {"elementId": "@s1:e2", "text": "Chào Lan"})
        await w.call("input__click", {"elementId": "@s1:e3"})
        await w.call("cu__session_end", {"sessionId": "cu-1", "outcome": "completed"})
        # One signature to start the session, one for the commit; the draft asked nothing.
        assert _ended_audits(w) == [{"reason": "completed", "steps": 2, "prompts": 2, "minutes": 0}]
        assert all("Chào Lan" not in str(e.metadata) for e in w.audit.events)

    _run(scenario)


def test_emergency_stop_and_a_lost_heartbeat_are_audited_once() -> None:
    async def scenario() -> None:
        w = await _build([])
        await w.orch.emergency_stop(owner_id=w.owner_id)
        await w.orch.emergency_stop(owner_id=w.owner_id)
        assert _ended_audits(w) == [{"reason": "owner_app", "steps": 0, "prompts": 1, "minutes": 0}]

        clock = _Clock()
        h = await _build([], clock=clock)
        clock.now = NOW + timedelta(minutes=1)
        assert await h.orch.active_cu_session_event(h.owner_id) is None
        assert _ended_audits(h) == [
            {"reason": "heartbeat_timeout", "steps": 0, "prompts": 1, "minutes": 1}
        ]

    _run(scenario)
