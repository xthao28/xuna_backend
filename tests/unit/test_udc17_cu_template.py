"""UDC-17 S17.5: rerunning a computer-use template saved on the owner's phone.

For the turn that carries `cuTemplate`, `cu.session_start` must run on the template's computer
with the same apps and domains and budgets no larger than the template's. A mismatch is refused
before the owner is asked; a match still needs the phone-signed R3 approval, and the card names
the template only when the whole card still fits.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from typing import Any

import pytest
from pydantic import ValidationError
from tests.unit.test_udc17_preflight import _build, _descriptor

from xuna_backend.modules.actions.application.cu_template import CuTemplatePin
from xuna_backend.modules.actions.application.prompts import state_driven_hint
from xuna_backend.modules.actions.domain.action_summary import ActionSummary
from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.cognition.domain.types import Locale
from xuna_backend.modules.conversation.api.routes import CuTemplateBody
from xuna_backend.modules.hosts.domain.entities import RiskTier, ToolFamily

_START = _descriptor(
    "cu.session_start",
    RiskTier.R3,
    {
        "goal": {"type": "string"},
        "apps": {"type": "array", "items": {"type": "string"}},
        "domains": {"type": "array", "items": {"type": "string"}},
        "maxSteps": {"type": "integer"},
        "maxMinutes": {"type": "integer"},
    },
    ToolFamily.CU,
)

_ARGS: dict[str, Any] = {
    "goal": "Gửi báo cáo tuần cho Lan",
    "apps": ["notes", "mail"],
    "domains": ["docs.example.com"],
    "maxSteps": 20,
    "maxMinutes": 10,
}


def _host(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    if name == "cu.session_start":
        return {"sessionId": "cu-2"}
    return None


def _pin(default_host_id: str, **overrides: Any) -> CuTemplatePin:
    values: dict[str, Any] = {
        "name": "Báo cáo tuần",
        "host_id": default_host_id,
        "apps": ["mail", "NOTES"],
        "domains": ["Docs.Example.com."],
        "max_steps": 20,
        "max_minutes": 10,
    }
    values.update(overrides)
    return CuTemplatePin.create(**values)


async def _world_with(pin_for: Any) -> Any:
    world = await _build([], extra_tools=(_START,), structured_for=_host)
    world.ctx = dataclasses.replace(world.ctx, cu_template=pin_for(str(world.host.host_id)))
    return world


def test_a_matching_start_still_asks_the_owner_and_the_card_names_the_template() -> None:
    async def scenario() -> None:
        world = await _world_with(lambda host_id: _pin(host_id))
        body = await world.call("cu__session_start", dict(_ARGS))
        assert body["status"] == "ok"
        assert len(world.pending) == 1
        card = world.pending[0]
        assert card.risk == "R3"
        assert card.summary.startswith("Chạy lại mẫu 'Báo cáo tuần': ")
        # The stored approval row does not carry the template name.
        assert world.approvals is not None
        row = await world.approvals.get(uuid.UUID(card.approval_id))
        assert row is not None and "Báo cáo tuần" not in row.summary
        assert [c["name"] for c in world.tool_calls] == ["cu.session_start"]

    asyncio.run(scenario())


def test_smaller_budgets_match() -> None:
    async def scenario() -> None:
        world = await _world_with(lambda host_id: _pin(host_id))
        body = await world.call("cu__session_start", {**_ARGS, "maxSteps": 5, "maxMinutes": 3})
        assert body["status"] == "ok"
        assert len(world.pending) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("args", "pin_overrides", "reason"),
    [
        ({**_ARGS, "apps": ["notes"]}, {}, "apps"),
        ({**_ARGS, "apps": ["notes", "mail", "terminal"]}, {}, "apps"),
        ({**_ARGS, "domains": ["docs.example.com", "evil.example.net"]}, {}, "domains"),
        ({**_ARGS, "domains": []}, {}, "domains"),
        ({**_ARGS, "maxSteps": 21}, {}, "budget"),
        ({**_ARGS, "maxMinutes": 11}, {}, "budget"),
        (dict(_ARGS), {"host_id": str(uuid.uuid4())}, "host"),
    ],
)
def test_a_mismatch_is_refused_before_anyone_is_asked(
    args: dict[str, Any], pin_overrides: dict[str, Any], reason: str
) -> None:
    async def scenario() -> None:
        world = await _world_with(lambda host_id: _pin(host_id, **pin_overrides))
        body = await world.call("cu__session_start", args)
        assert body["status"] == "denied"
        assert body["output"]["errorCode"] == "template_mismatch"
        assert body["output"]["mismatch"] == reason
        assert body["output"]["expected"]["apps"] == ["mail", "notes"]
        assert "hint" in body
        assert world.pending == []
        assert world.tool_calls == []
        denied = [e for e in world.audit.events if e.action == AuditAction.HOST_TOOL_DENIED]
        assert denied[-1].metadata["reason"] == "template_mismatch"
        assert "Báo cáo" not in str(denied[-1].metadata)

    asyncio.run(scenario())


def test_without_a_template_the_start_is_unchanged() -> None:
    async def scenario() -> None:
        world = await _build([], extra_tools=(_START,), structured_for=_host)
        body = await world.call("cu__session_start", {**_ARGS, "apps": ["terminal"]})
        assert body["status"] == "ok"
        assert not world.pending[0].summary.startswith("Chạy lại mẫu")

    asyncio.run(scenario())


def test_the_template_name_never_pushes_the_card_past_its_limit() -> None:
    long_summary = "x" * 220
    assert ActionSummary.with_template_name(long_summary, "Báo cáo tuần") == long_summary
    assert ActionSummary.with_template_name("Bắt đầu", "A", Locale.EN) == (
        "Rerun template 'A': Bắt đầu"
    )


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"name": "  "}, "template_name_invalid"),
        ({"name": "a\nb"}, "template_name_invalid"),
        ({"name": "x" * 41}, "template_name_invalid"),
        ({"host_id": "pc-1"}, "template_host_invalid"),
        ({"apps": []}, "template_apps_invalid"),
        ({"apps": ["Not An Id!"]}, "template_apps_invalid"),
        ({"apps": [f"app{i}" for i in range(9)]}, "template_apps_invalid"),
        ({"domains": ["localhost"]}, "template_domains_invalid"),
        ({"domains": ["https://a.example.com"]}, "template_domains_invalid"),
        ({"domains": [f"d{i}.example.com" for i in range(13)]}, "template_domains_invalid"),
        ({"max_steps": 41}, "template_budget_invalid"),
        ({"max_minutes": 0}, "template_budget_invalid"),
    ],
)
def test_the_template_is_validated_like_a_session_start(
    overrides: dict[str, Any], code: str
) -> None:
    with pytest.raises(ValueError, match=code):
        _pin(str(uuid.uuid4()), **overrides)


def test_the_turn_body_rejects_unknown_or_missing_template_fields() -> None:
    good = {
        "name": "Báo cáo tuần",
        "hostId": str(uuid.uuid4()),
        "apps": ["notes"],
        "maxSteps": 10,
        "maxMinutes": 5,
    }
    assert CuTemplateBody.model_validate(good).to_pin().domains == frozenset()
    with pytest.raises(ValidationError):
        CuTemplateBody.model_validate({**good, "goal": "x"})
    with pytest.raises(ValidationError):
        CuTemplateBody.model_validate({k: v for k, v in good.items() if k != "apps"})


def test_mismatch_hint_tells_the_model_what_to_do() -> None:
    hint = state_driven_hint({"errorCode": "template_mismatch"})
    assert hint is not None and "cu__session_start" in hint


def test_ending_with_completed_streams_the_ended_card_the_phone_offers_to_save() -> None:
    # The phone offers "Lưu thành mẫu" only on this event; without it the offer never shows.
    async def scenario() -> None:
        world = await _build([])
        body = await world.call("cu__session_end", {"sessionId": "cu-1", "outcome": "completed"})
        assert body["status"] == "ok"
        last = world.sessions[-1]
        assert last.session_id == "cu-1"
        assert last.status == "ended"
        assert last.end_reason == "completed"
        assert last.apps == ("explorer",)
        assert last.max_minutes == 10

    asyncio.run(scenario())
