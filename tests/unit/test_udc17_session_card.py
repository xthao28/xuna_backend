"""UDC-17 S17.3: the phone's live session card, stopping one session, extending a session's
scope, and denials that tell the app where the owner can fix them."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta

from tests.unit.test_udc17_preflight import NOW, _build, _ok, _run

from xuna_backend.modules.actions.application.computer_use import (
    build_session_event,
    session_event_payload,
)
from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.hosts.domain.entities import RiskTier


def test_every_step_updates_the_session_card() -> None:
    async def scenario() -> None:
        w = await _build([_ok("R2", "navigation_target", "Mở"), _ok("R2", "draft_text", "Soạn")])
        await w.call("input__click", {"elementId": "@s1:e1"})
        await w.call("input__type_text", {"elementId": "@s1:e2", "text": "Chào Lan"})
        assert [s.step for s in w.sessions] == [1, 2]
        card = w.sessions[-1]
        assert card.status == "running"
        assert card.session_id == "cu-1"
        assert card.max_steps == 20
        assert card.seconds_left == 600
        assert card.typed == ("Chào Lan",)
        assert "Chào Lan" in card.summary
        # The card is streamed only: nothing of it reaches the audit trail.
        assert all("Chào Lan" not in json.dumps(e.metadata) for e in w.audit.events)

    _run(scenario)


def test_the_card_sees_the_session_end() -> None:
    async def scenario() -> None:
        w = await _build([])
        await w.call("cu__session_end", {"sessionId": "cu-1", "outcome": "completed"})
        assert w.sessions[-1].status == "ended"
        assert w.sessions[-1].end_reason == "completed"
        assert await w.orch.active_cu_session_event(w.owner_id) is None

    _run(scenario)


def test_snapshot_applies_the_time_budget() -> None:
    async def scenario() -> None:
        w = await _build([])
        snap = await w.orch.active_cu_session_event(w.owner_id)
        assert snap is not None and snap.status == "running"
        state = await w.cu_store.get_active_for_owner(w.owner_id)
        assert state is not None and state.started_at_utc is not None
        state.started_at_utc = NOW - timedelta(minutes=state.max_minutes, seconds=1)
        assert await w.orch.active_cu_session_event(w.owner_id) is None

    _run(scenario)


def test_stopping_one_session_ends_it_here_and_on_the_host() -> None:
    async def scenario() -> None:
        w = await _build([_ok("R2", "draft_text", "Soạn")])
        await w.call("input__type_text", {"elementId": "@s1:e2", "text": "nháp"})
        state = await w.cu_store.get_active_for_owner(w.owner_id)
        assert state is not None

        stranger = await w.orch.stop_cu_session(owner_id=uuid.uuid4(), session_id="cu-1")
        assert stranger is None
        assert state.active

        result = await w.orch.stop_cu_session(owner_id=w.owner_id, session_id="cu-1")
        assert result is not None and result.stopped and result.host_confirmed
        assert not state.active
        assert state.typed_since_commit == []
        end_call = w.tool_calls[-1]
        assert end_call["name"] == "cu.session_end"
        assert end_call["arguments"] == {"sessionId": "cu-1", "outcome": "stopped"}
        stops = [e for e in w.audit.events if e.action == AuditAction.HOST_CU_SESSION_STOPPED]
        assert len(stops) == 1
        assert stops[0].metadata == {"hostConfirmed": True, "reason": "owner_stopped"}

        again = await w.orch.stop_cu_session(owner_id=w.owner_id, session_id="cu-1")
        assert again is not None and again.already_ended
        assert len([c for c in w.tool_calls if c["name"] == "cu.session_end"]) == 1

        # The next input step is no longer exempt: the owner is asked again.
        await w.call("input__type_text", {"elementId": "@s1:e2", "text": "nữa"})
        assert w.pending and w.pending[-1].risk == "R2"

    _run(scenario)


def test_extend_is_signed_lists_every_item_and_takes_the_hosts_scope() -> None:
    async def scenario() -> None:
        w = await _build([])
        w.host_scope = (["explorer", "calendar"], ["lich.example.com"])
        body = await w.call(
            "cu__session_extend",
            {"sessionId": "cu-1", "apps": ["calendar"], "domains": ["lich.example.com"]},
        )
        assert body["status"] == "ok"
        assert w.pending[-1].risk == "R3"
        assert "calendar" in w.pending[-1].summary
        assert "lich.example.com" in w.pending[-1].summary
        state = await w.cu_store.get_active_for_owner(w.owner_id)
        assert state is not None
        assert state.allowed_apps == ("explorer", "calendar")
        assert state.allowed_domains == ("lich.example.com",)
        assert w.sessions[-1].apps == ("explorer", "calendar")

    _run(scenario)


def test_extend_card_shows_long_items_in_full_or_refuses() -> None:
    async def scenario() -> None:
        w = await _build([])
        # Three 40-char domains fit: each appears whole on the card.
        fits = [f"lich-{i}-{'a' * 27}.example.com" for i in range(3)]
        w.host_scope = (["explorer"], fits)
        body = await w.call(
            "cu__session_extend", {"sessionId": "cu-1", "apps": [], "domains": fits}
        )
        assert body["status"] == "ok"
        assert all(d in w.pending[-1].summary for d in fits)

        asked = len(w.pending)
        calls = len(w.tool_calls)
        # A domain whose tail would be cut (it would read like a Google page) is refused,
        # nothing is asked and nothing reaches the host.
        long_domain = "accounts.google.com.secure-login-verify.example-attacker.net"
        assert len(long_domain) <= 60
        too_long = [long_domain + "x" * 10]
        refused = await w.call(
            "cu__session_extend", {"sessionId": "cu-1", "apps": [], "domains": too_long}
        )
        assert refused["status"] == "denied"
        assert refused["output"] == {"errorCode": "extend_scope_too_long"}
        assert "fewer or shorter" in refused["hint"]
        # Items that each fit but together overflow the card are refused too.
        apps = [f"app_{c * 36}" for c in "abc"]
        domains = [f"{c * 28}.example.com" for c in "def"]
        overflow = await w.call(
            "cu__session_extend", {"sessionId": "cu-1", "apps": apps, "domains": domains}
        )
        assert overflow["output"] == {"errorCode": "extend_scope_too_long"}
        assert len(w.pending) == asked
        assert len(w.tool_calls) == calls
        denials = [
            e
            for e in w.audit.events
            if e.action == AuditAction.HOST_TOOL_DENIED
            and e.metadata.get("reason") == "extend_scope_too_long"
        ]
        assert len(denials) == 2

    _run(scenario)


def test_denials_name_the_reason_and_the_host() -> None:
    async def scenario() -> None:
        w = await _build(
            [
                _ok("R3", "consequential_target", "Gửi"),
                {"status": "denied", "errorCode": "password_field_forbidden"},
            ],
            input_ceiling=RiskTier.R2,
        )
        ceiling = await w.call("input__click", {"elementId": "@s1:e3"})
        assert ceiling["status"] == "denied"
        assert ceiling["output"] == {"errorCode": "risk_above_grant"}
        assert "Do not retry" in ceiling["hint"]
        assert w.results[-1].error_code == "risk_above_grant"
        assert w.results[-1].host_id == str(w.host.host_id)

        await w.call("input__type_text", {"elementId": "@s1:e4", "text": "x"})
        assert w.results[-1].error_code == "password_field_forbidden"

    _run(scenario)


def test_successful_results_carry_no_error_code() -> None:
    async def scenario() -> None:
        w = await _build([_ok("R2", "navigation_target", "Mở")])
        await w.call("input__click", {"elementId": "@s1:e1"})
        assert w.results[-1].status == "ok"
        assert w.results[-1].error_code is None
        assert w.results[-1].host_id == str(w.host.host_id)

    _run(scenario)


def test_session_payload_is_bounded() -> None:
    async def scenario() -> None:
        w = await _build([])
        state = await w.cu_store.get_active_for_owner(w.owner_id)
        assert state is not None
        state.typed_since_commit.append("x" * 2000)
        payload = session_event_payload(build_session_event(state, NOW, "s"))
        assert len(payload["typed"][0]) == 500
        assert payload["status"] == "running"
        assert "endReason" not in payload

    _run(scenario)
