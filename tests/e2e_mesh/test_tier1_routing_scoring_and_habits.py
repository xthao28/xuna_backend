"""Tier 1 Backend E2E Mesh Tests: Pre-Router, Scorer, Habit Learner, Prober & Meta-Tools.

Opaque-box tests exercising:
- Feature 5: Deterministic Pre-Router (`targetHostId`, `/may`, `/follow`, `@alias`/`@place`,
  `PendingIntent` 1-word repair, `job_runner` hard `pinned_host_id`, Scoped Tool Catalog)
- Feature 6: Evidence Confidence Scorer (`R0`-`R3`), Habit Learner (`gamma=0.85`, dual-bucket,
  negative correction), Vietnamese fuzzy/phonetic matching, `unique_capable_host`
- Feature 7: Parallel R0/R1 Scatter-Gather Prober (`MeshProber`) & `mesh__status` /
  `mesh__find_resource` meta-tools.
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import psycopg
import pytest

from tests.e2e_mesh.conftest import pg_dsn
from tests.e2e_mesh.xtp_sim import (
    ScriptedTextGenerator,
    SimulatedOwnerClient,
    SimulatedXtpHost,
    done_event,
    make_standard_mesh_tools,
    make_tool_descriptor,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatRequest,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)

pytestmark = [pytest.mark.postgres, pytest.mark.anyio]


# =============================================================================
# Feature 5 — Deterministic Pre-Router, Hard Pinning & Scoped Catalog
# =============================================================================


async def test_t1_f5_01_explicit_target_host_id_pins_host_before_llm(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F5-01: TurnBody.targetHostId pins host before LLM and scopes tool catalog."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="p1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Paused."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()

        mac_tools = make_standard_mesh_tools(
            platform="macos",
            extra_tools=[make_tool_descriptor("window.tile_mac", family="app", risk="R1")],
        )
        win_tools = make_standard_mesh_tools(
            platform="windows",
            extra_tools=[make_tool_descriptor("window.snap_win", family="app", risk="R1")],
        )
        host_mac = SimulatedXtpHost(platform="macos", tools=mac_tools)
        host_win = SimulatedXtpHost(platform="windows", tools=win_tools)
        try:
            await host_mac.connect_enroll(srv.ws_url, owner, "Mac")
            id_win = await host_win.connect_enroll(srv.ws_url, owner, "Win")
            await owner.grant_all_standard(str(host_mac.host_id), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")

            outcome = await owner.stream_turn(
                pet_id,
                "pause music",
                target_host_id=str(id_win),
            )
            res = outcome.first("action_result")
            assert res is not None
            assert res["status"] == "ok"
            assert res["hostId"] == str(id_win)
            assert res.get("targetPlatform") == "windows"
            assert res.get("targetReason")

            assert len(host_win.received_tool_calls) == 1
            assert len(host_mac.received_tool_calls) == 0

            tool_names_offered = {t.name for t in llm.requests[0].tools}
            assert "window__snap_win" in tool_names_offered
            assert "window__tile_mac" not in tool_names_offered
        finally:
            await host_mac.disconnect()
            await host_win.disconnect()
            await owner.aclose()


async def test_t1_f5_02_slash_may_and_follow_commands_emit_host_focus_changed(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F5-02: /may <alias>, /follow on|off, /may auto emit host_focus_changed."""
    llm = ScriptedTextGenerator()
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_mac = SimulatedXtpHost(platform="macos")
        host_win = SimulatedXtpHost(platform="windows")
        try:
            await host_mac.connect_enroll(srv.ws_url, owner, "Mac")
            id_win = await host_win.connect_enroll(srv.ws_url, owner, "Win")

            t1 = await owner.stream_turn(pet_id, "/may Win")
            ev1 = t1.first("host_focus_changed")
            assert ev1 is not None, f"Expected host_focus_changed, got {t1.events}"
            assert ev1["hostId"] == str(id_win)
            assert ev1["source"] == "slash_cmd"

            hosts1 = {h["alias"]: h for h in await owner.list_hosts()}
            assert hosts1["Win"]["focused"] is True
            assert hosts1["Mac"]["focused"] is False

            t2 = await owner.stream_turn(pet_id, "/follow on")
            ev2 = t2.first("host_focus_changed")
            assert ev2 is not None
            assert ev2["followMe"] is True
            assert ev2["source"] == "slash_cmd"

            t3 = await owner.stream_turn(pet_id, "/may auto")
            ev3 = t3.first("host_focus_changed")
            assert ev3 is not None
            assert ev3["hostId"] is None
            assert ev3["source"] == "slash_cmd"

            hosts3 = {h["alias"]: h for h in await owner.list_hosts()}
            assert hosts3["Win"]["focused"] is False
            assert hosts3["Mac"]["focused"] is False
        finally:
            await host_mac.disconnect()
            await host_win.disconnect()
            await owner.aclose()


async def test_t1_f5_03_inline_at_alias_and_quoted_nickname_resolution(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F5-03: Inline @alias and @"quoted nickname" pin matching host before LLM."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(
                    ToolCall(id="a1", name="app__launch", arguments={"id": "browser_x"})
                ),
                done_event(),
            ],
            [TextDelta("Opened."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_mac = SimulatedXtpHost(platform="macos")
        host_win = SimulatedXtpHost(platform="windows")
        try:
            await host_mac.connect_enroll(srv.ws_url, owner, "AlphaMac")
            id_win = await host_win.connect_enroll(srv.ws_url, owner, "BetaWin")
            await owner.grant_all_standard(str(host_mac.host_id), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")

            patch = await owner.patch_host(
                str(id_win),
                {"schemaVersion": 1, "place": "office", "nicknames": ["Desk PC"]},
            )
            assert patch.status_code == 200

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    r = await owner.decide_approval(data["approvalId"], "approve")
                    assert r.status_code == 200

            outcome = await owner.stream_turn(
                pet_id,
                'open app @"Desk PC"',
                on_event=_approve,
            )
            res = outcome.first("action_result")
            assert res is not None and res["status"] == "ok"
            assert res["hostId"] == str(id_win)
            assert len(host_win.received_tool_calls) == 1
            assert len(host_mac.received_tool_calls) == 0
        finally:
            await host_mac.disconnect()
            await host_win.disconnect()
            await owner.aclose()


async def test_t1_f5_04_pending_intent_1_word_repair_within_ttl(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F5-04: 1-word host reply repairs PendingIntent & updates habit."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="r1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Which host?"), done_event()],
            [TextDelta("Paused on AlphaMac."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_mac = SimulatedXtpHost(platform="macos")
        host_win = SimulatedXtpHost(platform="windows")
        try:
            id_mac = await host_mac.connect_enroll(srv.ws_url, owner, "AlphaMac")
            id_win = await host_win.connect_enroll(srv.ws_url, owner, "BetaWin")
            await owner.grant_all_standard(str(id_mac), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")

            subnet = "11223344556677889900aabbccddeeff"
            for h in (host_mac, host_win):
                await h.send_presence(
                    idle_seconds=10,
                    screen_locked=False,
                    media_playing=False,
                    subnet_hash=subnet,
                )

            turn1 = await owner.stream_turn(pet_id, "pause media", client_subnet_hash=subnet)
            req = turn1.first("action_target_required")
            assert req is not None and req["reasonCode"] == "ask_target"
            assert len(host_mac.received_tool_calls) == 0
            assert len(host_win.received_tool_calls) == 0

            # Turn 2: 1-word repair without repeating tool intent
            turn2 = await owner.stream_turn(pet_id, "AlphaMac", client_subnet_hash=subnet)
            res2 = turn2.first("action_result")
            assert (
                res2 is not None
            ), f"Expected action_result after 1-word repair, got {turn2.events}"
            assert res2["status"] == "ok"
            assert res2["hostId"] == str(id_mac)
            assert len(host_mac.received_tool_calls) == 1

            habits = await owner.list_habits(pet_id=pet_id)
            assert habits.status_code == 200
            mac_habits = [h for h in habits.json()["items"] if h["hostId"] == str(id_mac)]
            assert len(mac_habits) >= 1
            assert mac_habits[0]["positiveScore"] >= 1.0
        finally:
            await host_mac.disconnect()
            await host_win.disconnect()
            await owner.aclose()


async def test_t1_f5_05_job_runner_hard_pinned_host_id_and_scoped_catalog(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F5-05: AutonomousJob pins job.host_id even when another host is focused."""
    step_called = False

    def _responder(req: ChatRequest, _idx: int) -> list[Any]:
        nonlocal step_called
        if req.tools and not step_called:
            step_called = True
            return [
                ToolCallRequested(ToolCall(id="job-1", name="media__play_pause", arguments={})),
                done_event(),
            ]
        return [TextDelta("Sub-goal step completed."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()

        mac_tools = make_standard_mesh_tools(
            platform="macos",
            extra_tools=[
                make_tool_descriptor(
                    "window.mac_only",
                    family="app",
                    risk="R1",
                    platform="macos",
                )
            ],
        )
        win_tools = make_standard_mesh_tools(
            platform="windows",
            extra_tools=[
                make_tool_descriptor(
                    "window.win_only",
                    family="app",
                    risk="R1",
                    platform="windows",
                )
            ],
        )
        host_mac = SimulatedXtpHost(platform="macos", tools=mac_tools)
        host_win = SimulatedXtpHost(platform="windows", tools=win_tools)
        try:
            id_mac = await host_mac.connect_enroll(srv.ws_url, owner, "MacFocus")
            id_win = await host_win.connect_enroll(srv.ws_url, owner, "WinPinned")
            await owner.grant_all_standard(str(id_mac), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")

            # Focus Mac, then create AutonomousJob pinned strictly to Win
            focus_resp = await owner.set_focus(host_id=str(id_mac), follow_me=False)
            assert focus_resp.status_code == 200

            job_resp = await owner.create_job(
                pet_id=pet_id,
                goal="Step 1: pause media\nStep 2: verify\nStep 3: finish",
                host_id=str(id_win),
                max_steps=4,
            )
            assert job_resp.status_code == 201, job_resp.text
            job_id = job_resp.json()["jobId"]

            for _ in range(40):
                if host_win.received_tool_calls:
                    break
                await asyncio.sleep(0.05)

            assert len(host_win.received_tool_calls) >= 1
            assert len(host_mac.received_tool_calls) == 0

            tool_req = next((r for r in llm.requests if r.tools), None)
            assert tool_req is not None, "Expected LLM request with scoped tools"
            offered = {t.name for t in tool_req.tools}
            assert "window__win_only" in offered
            assert "window__mac_only" not in offered

            await owner.http.post(
                f"/v1/actions/jobs/{job_id}/cancel",
                headers=owner.headers,
                json={"schemaVersion": 1, "reason": "test_done"},
            )
        finally:
            await host_mac.disconnect()
            await host_win.disconnect()
            await owner.aclose()


# =============================================================================
# Feature 6 — Evidence Confidence Scorer & Contextual Habit Learner
# =============================================================================


async def test_t1_f6_01_scorer_resource_and_presence_auto_routes_r2(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F6-01: Resource match + unlocked active presence auto-routes R2 above floor."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(
                    ToolCall(id="s1", name="app__launch", arguments={"id": "code_editor_x"})
                ),
                done_event(),
            ],
            [TextDelta("Launched."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_a = SimulatedXtpHost(platform="macos")
        host_b = SimulatedXtpHost(platform="windows")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "ReadyNode")
            id_b = await host_b.connect_enroll(srv.ws_url, owner, "LockedNode")
            await owner.grant_all_standard(str(id_a), max_risk="R2")
            await owner.grant_all_standard(str(id_b), max_risk="R2")

            subnet = "abcdef0123456789abcdef0123456789"
            await host_a.send_presence(
                idle_seconds=5,
                screen_locked=False,
                media_playing=False,
                subnet_hash=subnet,
            )
            await host_a.send_resources(installed_apps=["code_editor_x"], workspaces=[])

            await host_b.send_presence(
                idle_seconds=600,
                screen_locked=True,
                media_playing=False,
                subnet_hash="ffffffffffffffffffffffffffffffff",
            )
            await host_b.send_resources(installed_apps=["other_tool_y"], workspaces=[])

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            outcome = await owner.stream_turn(
                pet_id,
                "launch code_editor_x",
                client_subnet_hash=subnet,
                on_event=_approve,
            )
            assert outcome.first("action_target_required") is None
            pending = outcome.first("action_pending")
            res = outcome.first("action_result")
            assert pending is not None and pending.get("hostId") == str(id_a)
            assert res is not None and res["status"] == "ok" and res["hostId"] == str(id_a)
            assert len(host_a.received_tool_calls) == 1
            assert len(host_b.received_tool_calls) == 0
        finally:
            await host_a.disconnect()
            await host_b.disconnect()
            await owner.aclose()


async def test_t1_f6_02_media_family_uses_media_playing_presence_weighting(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F6-02: media family routes to host with mediaPlaying=True over quieter idle."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="m1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Paused."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_playing = SimulatedXtpHost(platform="macos")
        host_silent = SimulatedXtpHost(platform="windows")
        try:
            id_play = await host_playing.connect_enroll(srv.ws_url, owner, "PlayingNode")
            id_silent = await host_silent.connect_enroll(srv.ws_url, owner, "SilentNode")
            await owner.grant_all_standard(str(id_play), max_risk="R1")
            await owner.grant_all_standard(str(id_silent), max_risk="R1")

            subnet = "aabbccddeeff00112233445566778899"
            await host_playing.send_presence(
                idle_seconds=200,
                screen_locked=False,
                media_playing=True,
                subnet_hash=subnet,
            )
            await host_silent.send_presence(
                idle_seconds=10,
                screen_locked=False,
                media_playing=False,
                subnet_hash=subnet,
            )

            outcome = await owner.stream_turn(
                pet_id,
                "pause music",
                client_subnet_hash=subnet,
            )
            assert outcome.first("action_target_required") is None
            res = outcome.first("action_result")
            assert res is not None and res["status"] == "ok"
            assert res["hostId"] == str(id_play)
            assert len(host_playing.received_tool_calls) == 1
            assert len(host_silent.received_tool_calls) == 0
        finally:
            await host_playing.disconnect()
            await host_silent.disconnect()
            await owner.aclose()


async def test_t1_f6_03_habit_learner_gamma_085_positive_and_dual_bucket(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F6-03: 3 positive picks yield P=2.5725 (gamma=0.85) in specific & dash buckets."""
    llm = ScriptedTextGenerator(
        responder=lambda req, idx: (
            [
                ToolCallRequested(
                    ToolCall(id=f"c{idx}", name="app__launch", arguments={"id": "audio_player_x"})
                ),
                done_event(),
            ]
            if not any(m.role == Role.TOOL for m in req.messages)
            else [TextDelta("OK."), done_event()]
        )
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_a = SimulatedXtpHost(platform="macos")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "HabitMac")
            await owner.grant_all_standard(str(id_a), max_risk="R2")

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            for _ in range(3):
                res = await owner.stream_turn(
                    pet_id,
                    "launch audio_player_x",
                    target_host_id=str(id_a),
                    on_event=_approve,
                )
                assert res.require("action_result")["status"] == "ok"

            habits_resp = await owner.list_habits(pet_id=pet_id)
            assert habits_resp.status_code == 200
            items = habits_resp.json()["items"]
            expected_p = 1.0 + 0.85 + 0.85**2  # 2.5725

            specific = [h for h in items if ":audio_player_x:" in h["contextKey"]]
            wildcard_dash = [h for h in items if ":-:" in h["contextKey"]]
            assert len(specific) == 1
            assert len(wildcard_dash) == 1
            for row in (specific[0], wildcard_dash[0]):
                assert row["positiveScore"] == pytest.approx(expected_p, rel=1e-4)
                assert row["negativeScore"] == pytest.approx(0.0)
                assert row["sampleCount"] == 3
                assert row["gamma"] == pytest.approx(0.85)
        finally:
            await host_a.disconnect()
            await owner.aclose()


async def test_t1_f6_04_negative_correction_lowers_habit_and_overrides_preference(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F6-04: Picking host B after ask_target records redirect penalty on host A."""
    llm = ScriptedTextGenerator(
        responder=lambda req, idx: (
            [
                ToolCallRequested(ToolCall(id=f"n{idx}", name="media__play_pause", arguments={})),
                done_event(),
            ]
            if not any(m.role == Role.TOOL for m in req.messages)
            else [TextDelta("Done."), done_event()]
        )
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_a = SimulatedXtpHost(platform="macos")
        host_b = SimulatedXtpHost(platform="windows")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "NodeA")
            id_b = await host_b.connect_enroll(srv.ws_url, owner, "NodeB")
            await owner.grant_all_standard(str(id_a), max_risk="R1")
            await owner.grant_all_standard(str(id_b), max_risk="R1")

            subnet = "22334455667788990011aabbccddeeff"
            for h in (host_a, host_b):
                await h.send_presence(
                    idle_seconds=15, screen_locked=False, media_playing=False, subnet_hash=subnet
                )

            # Seed positive habit for NodeA via 2 explicit turns
            for _ in range(2):
                await owner.stream_turn(
                    pet_id, "pause", target_host_id=str(id_a), client_subnet_hash=subnet
                )

            # Give NodeB slight presence tie so ask_target fires, then pick NodeB via PendingIntent
            await host_a.send_presence(
                idle_seconds=60, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await host_b.send_presence(
                idle_seconds=10, screen_locked=False, media_playing=False, subnet_hash=subnet
            )

            t_ask = await owner.stream_turn(pet_id, "pause", client_subnet_hash=subnet)
            if t_ask.first("action_target_required") is not None:
                t_repair = await owner.stream_turn(pet_id, "NodeB", client_subnet_hash=subnet)
                assert t_repair.require("action_result")["hostId"] == str(id_b)

                with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                    neg_row = conn.execute(
                        """
                        SELECT negative_score, last_reason_code FROM host_routing_habits
                        WHERE host_id = %s ORDER BY updated_at DESC LIMIT 1
                        """,
                        (id_a,),
                    ).fetchone()
                    assert neg_row is not None
                    assert neg_row[0] >= 1.0
                    assert neg_row[1] == "redirect_penalty"
            else:
                # Negative correction candidate must trigger ask_target first
                raise AssertionError("Expected ask_target or negative correction candidate")
        finally:
            await host_a.disconnect()
            await host_b.disconnect()
            await owner.aclose()


async def test_t1_f6_05_safe_unique_only_vietnamese_fuzzy_phonetic_matching(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F6-05: Phonetic Vietnamese @tag (@"fong lam viec") matches "Phòng Làm Việc"."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="f1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Done."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_office = SimulatedXtpHost(platform="macos")
        host_other = SimulatedXtpHost(platform="windows")
        try:
            id_office = await host_office.connect_enroll(srv.ws_url, owner, "Phòng Làm Việc")
            await host_other.connect_enroll(srv.ws_url, owner, "LivingStation")
            await owner.grant_all_standard(str(id_office), max_risk="R1")
            await owner.grant_all_standard(str(host_other.host_id), max_risk="R1")

            outcome = await owner.stream_turn(pet_id, 'pause @"fong lam viec"')
            res = outcome.first("action_result")
            assert res is not None and res["status"] == "ok"
            assert res["hostId"] == str(id_office)
            assert len(host_office.received_tool_calls) == 1
            assert len(host_other.received_tool_calls) == 0
        finally:
            await host_office.disconnect()
            await host_other.disconnect()
            await owner.aclose()


async def test_t1_f6_06_single_unlocked_capable_host_auto_resolves_unique_capable_host(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F6-06: Single unlocked online capable R2 host resolves directly to approval."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(
                    ToolCall(id="u1", name="app__launch", arguments={"id": "code_editor_x"})
                ),
                done_event(),
            ],
            [TextDelta("Launched."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_a = SimulatedXtpHost(platform="macos")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "OnlyHost")
            await owner.grant_all_standard(str(id_a), max_risk="R2")
            await host_a.send_presence(idle_seconds=20, screen_locked=False, media_playing=False)

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            outcome = await owner.stream_turn(
                pet_id,
                "launch app",
                client_subnet_hash=None,
                on_event=_approve,
            )
            assert outcome.first("action_target_required") is None
            pending = outcome.first("action_pending")
            res = outcome.first("action_result")
            assert pending is not None and pending["hostId"] == str(id_a)
            assert res is not None and res["status"] == "ok"
        finally:
            await host_a.disconnect()
            await owner.aclose()


# =============================================================================
# Feature 7 — Parallel Prober & Mesh Meta-Tools
# =============================================================================


async def test_t1_f7_01_mesh_status_runs_concurrent_r0_r1_probes(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F7-01: mesh__status runs concurrent R0/R1 probes & reports offline host."""
    tool_outputs: list[str] = []

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        for m in req.messages:
            if m.role == Role.TOOL:
                tool_outputs.append(m.content)
        if idx == 0:
            return [
                ToolCallRequested(
                    ToolCall(id="ms1", name="mesh__status", arguments={"includeVitals": True})
                ),
                done_event(),
            ]
        return [TextDelta("Mesh status checked."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents(host_screen=True)
        h1 = SimulatedXtpHost(platform="macos")
        h2 = SimulatedXtpHost(platform="windows")
        h3 = SimulatedXtpHost(platform="linux")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "NodeOne")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "NodeTwo")
            id3 = await h3.connect_enroll(srv.ws_url, owner, "NodeThree")
            await owner.grant_all_standard(str(id1), max_risk="R2")
            await owner.grant_all_standard(str(id2), max_risk="R2")
            await h1.send_presence(idle_seconds=5, screen_locked=False, media_playing=True)
            await h2.send_presence(idle_seconds=10, screen_locked=False, media_playing=False)
            await h3.disconnect()

            await owner.stream_turn(pet_id, "check mesh status")
            assert len(tool_outputs) == 1
            parsed = json.loads(tool_outputs[0])
            hosts_map = {item["hostId"]: item for item in parsed["hosts"]}
            assert set(hosts_map.keys()) == {str(id1), str(id2), str(id3)}
            assert hosts_map[str(id1)]["online"] is True
            assert hosts_map[str(id2)]["online"] is True
            assert hosts_map[str(id3)]["online"] is False
            assert len(h1.received_tool_calls) >= 1
            assert len(h2.received_tool_calls) >= 1
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await h3.disconnect()
            await owner.aclose()


async def test_t1_f7_02_mesh_find_resource_matches_host_twin_resources(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F7-02: mesh__find_resource searches HostTwin.resources without hardcoding."""
    tool_outputs: list[str] = []

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        for m in req.messages:
            if m.role == Role.TOOL:
                tool_outputs.append(m.content)
        if idx == 0:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="fr1",
                        name="mesh__find_resource",
                        arguments={"query": "Apollo_Repo", "kind": "workspace"},
                    )
                ),
                done_event(),
            ]
        return [TextDelta("Found workspace."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1 = SimulatedXtpHost(platform="macos")
        h2 = SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "NodeA")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "NodeB")
            await h1.send_resources(installed_apps=["custom_cad_x"], workspaces=["Other_Repo"])
            await h2.send_resources(installed_apps=[], workspaces=["Apollo_Repo"])

            await owner.stream_turn(pet_id, "find Apollo_Repo")
            assert len(tool_outputs) == 1
            parsed = json.loads(tool_outputs[0])
            match_ids = {m["hostId"] for m in parsed["matches"]}
            assert match_ids == {str(id2)}
            assert str(id1) not in match_ids
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t1_f7_03_mesh_find_resource_falls_back_to_r0_app_list_allowed(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F7-03: mesh__find_resource probes R0 app.list_allowed if resources is empty."""
    tool_outputs: list[str] = []

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        for m in req.messages:
            if m.role == Role.TOOL:
                tool_outputs.append(m.content)
        if idx == 0:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="fr2",
                        name="mesh__find_resource",
                        arguments={"query": "synth_studio_x", "kind": "app"},
                    )
                ),
                done_event(),
            ]
        return [TextDelta("Found app."), done_event()]

    def _handler(name: str, _args: dict[str, Any], _meta: dict[str, Any]) -> dict[str, Any]:
        if name == "app.list_allowed":
            return {"status": "ok", "structuredContent": {"apps": [{"id": "synth_studio_x"}]}}
        return {"status": "ok", "structuredContent": {"ok": True}}

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1 = SimulatedXtpHost(platform="macos", tool_handler=_handler)
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "EmptyResourcesHost")
            await owner.grant_all_standard(str(id1), max_risk="R1")
            # Intentionally do NOT call send_resources so HostTwin.resources stays None

            await owner.stream_turn(pet_id, "find synth_studio_x")
            assert len(tool_outputs) == 1
            parsed = json.loads(tool_outputs[0])
            match_ids = {m["hostId"] for m in parsed["matches"]}
            assert str(id1) in match_ids
            assert any(c["name"] == "app.list_allowed" for c in h1.received_tool_calls)
        finally:
            await h1.disconnect()
            await owner.aclose()


async def test_t1_f7_04_mesh_prober_partial_timeout_isolation_2000ms(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F7-04: Hanging host times out cleanly at ~2000ms without failing healthy host."""
    tool_outputs: list[str] = []

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        for m in req.messages:
            if m.role == Role.TOOL:
                tool_outputs.append(m.content)
        if idx == 0:
            return [
                ToolCallRequested(
                    ToolCall(id="ms2", name="mesh__status", arguments={"includeVitals": False})
                ),
                done_event(),
            ]
        return [TextDelta("Status ready."), done_event()]

    async def _slow_handler(
        _name: str, _args: dict[str, Any], _meta: dict[str, Any]
    ) -> dict[str, Any]:
        await asyncio.sleep(3.0)
        return {"status": "ok", "structuredContent": {"online": True}}

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_fast = SimulatedXtpHost(platform="macos")
        h_slow = SimulatedXtpHost(platform="windows", tool_handler=_slow_handler)
        try:
            id_fast = await h_fast.connect_enroll(srv.ws_url, owner, "FastNode")
            id_slow = await h_slow.connect_enroll(srv.ws_url, owner, "SlowNode")
            await owner.grant_all_standard(str(id_fast), max_risk="R1")
            await owner.grant_all_standard(str(id_slow), max_risk="R1")

            await owner.stream_turn(pet_id, "mesh status")
            assert len(tool_outputs) == 1
            parsed = json.loads(tool_outputs[0])
            hosts_map = {item["hostId"]: item for item in parsed["hosts"]}
            assert hosts_map[str(id_fast)]["probeStatus"] == "ok"
            assert hosts_map[str(id_slow)]["probeStatus"] == "timeout"
        finally:
            await h_fast.disconnect()
            await h_slow.disconnect()
            await owner.aclose()


async def test_t1_f7_05_mesh_prober_enforces_host_grant_and_u5_for_r1_probes(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F7-05: MeshProber executes R1 system.vitals only on host with host.R1 + U5."""
    tool_outputs: list[str] = []

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        for m in req.messages:
            if m.role == Role.TOOL:
                tool_outputs.append(m.content)
        if idx == 0:
            return [
                ToolCallRequested(
                    ToolCall(id="ms3", name="mesh__status", arguments={"includeVitals": True})
                ),
                done_event(),
            ]
        return [TextDelta("Vitals checked."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents(host_screen=True)
        h_granted = SimulatedXtpHost(platform="macos")
        h_r0_only = SimulatedXtpHost(platform="windows")
        try:
            id_g = await h_granted.connect_enroll(srv.ws_url, owner, "GrantedNode")
            id_r0 = await h_r0_only.connect_enroll(srv.ws_url, owner, "R0OnlyNode")
            await owner.set_grants(
                str(id_g),
                [{"family": "host", "maxRisk": "R1", "enabled": True}],
            )
            await owner.set_grants(
                str(id_r0),
                [{"family": "host", "maxRisk": "R0", "enabled": True}],
            )

            await owner.stream_turn(pet_id, "check vitals")
            assert len(tool_outputs) == 1
            parsed = json.loads(tool_outputs[0])
            hosts_map = {item["hostId"]: item for item in parsed["hosts"]}
            assert any(c["name"] == "system.vitals" for c in h_granted.received_tool_calls)
            assert not any(c["name"] == "system.vitals" for c in h_r0_only.received_tool_calls)
            assert hosts_map[str(id_r0)]["vitalsStatus"] == "denied"
            assert hosts_map[str(id_g)]["vitalsStatus"] == "ok"
        finally:
            await h_granted.disconnect()
            await h_r0_only.disconnect()
            await owner.aclose()
