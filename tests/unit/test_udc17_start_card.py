"""UDC-17 S17.5 follow-up: the `cu.session_start` approval card shows the whole scope.

The owner signs the apps, domains, budgets and computer on this card, so every app and every
domain is listed in full (VI and EN). A scope the 240-character card cannot show is refused
before anyone is asked (`start_scope_too_long`), as `cu.session_extend` is since S17.3. The
S17.5 template prefix is added only when the whole card still fits.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest
from tests.unit.test_udc17_cu_template import _START, _host
from tests.unit.test_udc17_preflight import _build

from xuna_backend.modules.actions.application.computer_use import session_event_payload
from xuna_backend.modules.actions.application.cu_template import CuTemplatePin
from xuna_backend.modules.actions.domain.action_summary import (
    ActionSummary,
    _session_start_text,
)
from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.cognition.domain.types import Locale

_ALIAS = "PC Thảo"  # the harness host's alias

# Six apps (more than the old card's four) and three domains.
_WIDE: dict[str, Any] = {
    "goal": "Soạn báo cáo",
    "apps": ["notes", "mail", "excel", "word", "teams", "zalo"],
    "domains": ["docs.example.com", "lich.example.com", "drive.example.org"],
    "maxSteps": 20,
    "maxMinutes": 10,
}


def _card(args: dict[str, Any], locale: Locale = Locale.VI) -> str:
    return ActionSummary.format_pending("cu.session_start", args, _ALIAS, locale)


def _uncut(args: dict[str, Any]) -> str:
    # `format_pending` cuts at 240 as a backstop, so lengths are measured before that cut.
    return _session_start_text(args, _ALIAS, Locale.VI)


def _args_with_card_length(target: int) -> dict[str, Any]:
    """Start arguments whose uncut VI card is exactly `target` characters, every item <= 60."""
    args: dict[str, Any] = {
        "goal": "Đối chiếu lịch",
        "apps": ["calendar"],
        "domains": [],
        "maxSteps": 20,
        "maxMinutes": 10,
    }
    index = 0
    for _ in range(50):
        missing = target - len(_uncut(args))
        if missing == 0:
            return args
        if args["domains"] and len(args["domains"][-1]) < 60:
            label = args["domains"][-1].removesuffix(".example.com")
            grow = min(missing, 60 - len(args["domains"][-1]))
            args["domains"][-1] = f"{label}{'a' * grow}.example.com"
        else:
            # A new domain adds its own length plus ", " (or replaces "không có").
            index += 1
            args["domains"].append(f"d{index}.example.com")
            if len(_uncut(args)) > target:
                args["domains"].pop()
                # Not enough room for another domain: grow the goal instead.
                args["goal"] += "x" * missing
                assert len(args["goal"]) <= 60
        assert len(_uncut(args)) <= target
    raise AssertionError(f"could not build a {target}-character card: {_uncut(args)!r}")


@pytest.mark.parametrize("locale", [Locale.VI, Locale.EN])
def test_the_card_lists_every_app_and_domain(locale: Locale) -> None:
    card = _card(_WIDE, locale)
    for item in [*_WIDE["apps"], *_WIDE["domains"]]:
        assert item in card
    assert _ALIAS in card
    assert ("20 bước/10 phút" if locale == Locale.VI else "20 steps/10 min") in card
    assert ActionSummary.start_scope_fits_card(_WIDE, _ALIAS, locale)


@pytest.mark.parametrize(("locale", "none"), [(Locale.VI, "không có"), (Locale.EN, "none")])
def test_no_domains_is_said_not_left_out(locale: Locale, none: str) -> None:
    card = _card({**_WIDE, "domains": []}, locale)
    assert ("tên miền " if locale == Locale.VI else "domains ") + none in card


def test_only_the_goal_is_shortened_and_it_says_so() -> None:
    card = _card({**_WIDE, "goal": "g" * 80})
    assert "'" + "g" * 59 + "…'" in card
    assert all(item in card for item in [*_WIDE["apps"], *_WIDE["domains"]])


def test_the_card_limit_is_inclusive() -> None:
    exact = _args_with_card_length(240)
    assert len(_uncut(exact)) == 240
    assert ActionSummary.start_scope_fits_card(exact, _ALIAS)
    over = _args_with_card_length(241)
    assert len(_uncut(over)) == 241
    assert not ActionSummary.start_scope_fits_card(over, _ALIAS)


def test_the_orchestrator_card_names_every_app_and_domain() -> None:
    async def scenario() -> None:
        world = await _build([], extra_tools=(_START,), structured_for=_host)
        body = await world.call("cu__session_start", dict(_WIDE))
        assert body["status"] == "ok"
        assert len(world.pending) == 1
        card = world.pending[0]
        assert card.risk == "R3"
        for item in [*_WIDE["apps"], *_WIDE["domains"]]:
            assert item in card.summary
        # A card of exactly 240 characters is still asked, not refused.
        exact = _args_with_card_length(240)
        body = await world.call("cu__session_start", exact)
        assert body["status"] == "ok"
        assert len(world.pending[-1].summary) == 240

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "args",
    [
        # One domain longer than 60 characters: its tail would read like another site.
        {**_WIDE, "domains": ["accounts.google.com.secure-login-verify.example-attacker.net.x"]},
        # Every item fits, but together they overflow the card.
        {
            **_WIDE,
            "apps": [f"app_{c * 30}" for c in "abcd"],
            "domains": [f"{c * 30}.example.com" for c in "efg"],
        },
    ],
)
def test_a_scope_the_card_cannot_show_is_refused_before_anyone_is_asked(
    args: dict[str, Any],
) -> None:
    async def scenario() -> None:
        world = await _build([], extra_tools=(_START,), structured_for=_host)
        assert not ActionSummary.start_scope_fits_card(args, _ALIAS)
        body = await world.call("cu__session_start", args)
        assert body["status"] == "denied"
        assert body["output"] == {"errorCode": "start_scope_too_long"}
        assert "cu__session_extend" in body["hint"]
        assert world.pending == []
        assert world.tool_calls == []
        denials = [
            e
            for e in world.audit.events
            if e.action == AuditAction.HOST_TOOL_DENIED
            and e.metadata.get("reason") == "start_scope_too_long"
        ]
        assert len(denials) == 1
        assert denials[0].metadata["tool"] == "cu.session_start"

    asyncio.run(scenario())


def _pin_for(host_id: str, args: dict[str, Any], name: str = "Báo cáo tuần") -> CuTemplatePin:
    return CuTemplatePin.create(
        name=name,
        host_id=host_id,
        apps=list(args["apps"]),
        domains=list(args["domains"]),
        max_steps=int(args["maxSteps"]),
        max_minutes=int(args["maxMinutes"]),
    )


def test_the_template_prefix_is_dropped_when_it_would_not_fit() -> None:
    async def scenario() -> None:
        world = await _build([], extra_tools=(_START,), structured_for=_host)
        args = _args_with_card_length(230)
        world.ctx = dataclasses.replace(
            world.ctx, cu_template=_pin_for(str(world.host.host_id), args)
        )
        body = await world.call("cu__session_start", args)
        assert body["status"] == "ok"
        summary = world.pending[-1].summary
        assert not summary.startswith("Chạy lại mẫu")
        assert summary == _card(args)

    asyncio.run(scenario())


def test_a_template_whose_scope_cannot_fit_is_refused_without_a_retry_loop() -> None:
    async def scenario() -> None:
        world = await _build([], extra_tools=(_START,), structured_for=_host)
        args: dict[str, Any] = {
            "goal": "Báo cáo tuần",
            "apps": [f"app_{c * 30}" for c in "abcd"],
            "domains": [f"{c * 30}.example.com" for c in "efg"],
            "maxSteps": 20,
            "maxMinutes": 10,
        }
        world.ctx = dataclasses.replace(
            world.ctx, cu_template=_pin_for(str(world.host.host_id), args)
        )
        body = await world.call("cu__session_start", args)
        assert body["status"] == "denied"
        assert body["output"] == {"errorCode": "start_scope_too_long", "templatePinned": True}
        # The model is not told to trim the scope (that would only hit `template_mismatch`).
        assert "Do not retry" in body["hint"]
        assert "cu__session_extend" not in body["hint"]
        assert world.pending == []
        assert world.tool_calls == []

    asyncio.run(scenario())


_SMALL_SCOPE = (("notes", "mail"), ("docs.example.com",))
# A scope reached by extending (8 apps, 12 domains): no start card can show it.
_LARGE_SCOPE = (
    tuple(f"app{i}" for i in range(8)),
    tuple(f"site{i}.example.com" for i in range(12)),
)


@pytest.mark.parametrize(("scope", "fits"), [(_SMALL_SCOPE, True), (_LARGE_SCOPE, False)])
def test_the_session_card_tells_the_phone_whether_a_rerun_would_fit(
    scope: tuple[tuple[str, ...], tuple[str, ...]], fits: bool
) -> None:
    async def scenario() -> None:
        world = await _build([])
        state = await world.cu_store.get_active_for_owner(world.owner_id)
        assert state is not None
        state.allowed_apps, state.allowed_domains = scope
        body = await world.call("cu__session_end", {"sessionId": "cu-1", "outcome": "completed"})
        assert body["status"] == "ok"
        ended = world.sessions[-1]
        assert ended.status == "ended"
        assert ended.rerun_fits_card is fits
        assert session_event_payload(ended)["rerunFitsCard"] is fits

        # The flag agrees with what a rerun of that scope then meets, with the longest goal.
        rerun = await _build([], extra_tools=(_START,), structured_for=_host)
        args: dict[str, Any] = {
            "goal": "g" * 80,
            "apps": list(scope[0]),
            "domains": list(scope[1]),
            "maxSteps": state.max_steps,
            "maxMinutes": state.max_minutes,
        }
        rerun.ctx = dataclasses.replace(
            rerun.ctx, cu_template=_pin_for(str(rerun.host.host_id), args)
        )
        for locale in (Locale.VI, Locale.EN):
            rerun.ctx = dataclasses.replace(rerun.ctx, locale=locale)
            result = await rerun.call("cu__session_start", args)
            assert (result["status"] == "ok") is fits

    asyncio.run(scenario())


def test_a_rerun_must_fit_with_the_longest_goal_the_card_shows() -> None:
    # Fits with this session's short goal, but a rerun's goal is the model's restatement and
    # may take the card's full 60 characters.
    args = _args_with_card_length(225)
    assert ActionSummary.start_scope_fits_card(args, _ALIAS)
    assert not ActionSummary.rerun_scope_fits_card(
        tuple(args["apps"]), tuple(args["domains"]), 20, 10, _ALIAS
    )
    short = _args_with_card_length(150)
    assert ActionSummary.rerun_scope_fits_card(
        tuple(short["apps"]), tuple(short["domains"]), 20, 10, _ALIAS
    )
