"""Tier 3 Backend E2E Mesh Tests: Pairwise Cross-Feature Combinations & Voice WS (F3-F12).

Opaque-box pairwise tests covering all 14 Tier 3 combinations (`T3-01`..`T3-14`):
- Follow Me + Scorer re-weighting
- Negative Correction Habit Learning -> Subsequent Turn Re-Routing
- Multi-Host Parallel CU + 2-Tier E-Stop (HTTP & Voice WS)
- XTP Presence + MeshProber + Battery Advisor
- Tab/Workspace Teleport + MissionEnvelope + HTTPS Guard + Tier-1 E-Stop
- Hard Pin + Scoped Tool Catalog cross-OS schema collision prevention
- AutonomousJob Hard Pin + Offline WoL Deferral
- Desktop xuna/focusRequest -> HostTwin -> Voice WS focus visibility
- xuna/resources Digest -> mesh__find_resource & Scorer M_res
- Voice WS ask_target -> 1-word PendingIntent Repair -> Habit Update
- PATCH Spatial Metadata + 409 Guard + @place Routing
- Linux Server Registration + Zero-Hardcode Routing
- Universal Clipboard + Dual-End Secret Redaction + 1200-char Cap
- Smart Fan-out (smart_pause_media, fleet_vitals, fleet_git_status).
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import psycopg
import pytest

from tests.e2e_mesh.conftest import pg_dsn
from tests.e2e_mesh.xtp_sim import (
    FakeSpeechRecognizer,
    FakeSpeechSynthesizer,
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


def _require_ws_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    match = next((e for e in events if e.get("type") == event_type), None)
    assert match is not None, f"Expected WS event {event_type!r}, got {events}"
    return match


async def test_t3_01_follow_me_reweights_presence_over_continuity(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T3-01: follow_me=True reweights M_pres=0.40, M_cont=0.05 so active host wins."""
    llm = ScriptedTextGenerator(
        responder=lambda req, idx: (
            [
                ToolCallRequested(ToolCall(id=f"fm{idx}", name="host__status", arguments={})),
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
        h1, h2 = SimulatedXtpHost(platform="macos"), SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "ActiveDesk")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "StaleDesk")
            await owner.grant_all_standard(str(id1), max_risk="R1")
            await owner.grant_all_standard(str(id2), max_risk="R1")

            subnet = "102030405060708090a0b0c0d0e0f000"
            # Seed last_target continuity on StaleDesk first
            await h2.send_presence(
                idle_seconds=5, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await owner.stream_turn(
                pet_id, "check status", target_host_id=str(id2), client_subnet_hash=subnet
            )

            # ActiveDesk is fresh (idle=3s) while StaleDesk is idle (700s):
            # follow_me=False keeps StaleDesk leading on continuity+habit (ask_target),
            # while follow_me=True flips and auto-routes to ActiveDesk.
            await h1.send_presence(
                idle_seconds=3, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await h2.send_presence(
                idle_seconds=700, screen_locked=False, media_playing=False, subnet_hash=subnet
            )

            turn_no_fm = await owner.stream_turn(
                pet_id,
                "check host status",
                follow_me=False,
                client_subnet_hash=subnet,
            )
            req_no_fm = turn_no_fm.first("action_target_required")
            assert req_no_fm is not None and req_no_fm["reasonCode"] == "ask_target"
            assert req_no_fm["candidates"][0]["hostId"] == str(id2)

            turn = await owner.stream_turn(
                pet_id,
                "check host status",
                follow_me=True,
                client_subnet_hash=subnet,
            )
            res = turn.first("action_result")
            assert res is not None and res["status"] == "ok"
            assert res["hostId"] == str(id1)
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t3_02_habit_negative_correction_reroutes_subsequent_turn(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T3-02: Repeated redirects away from Host1 flip subsequent unpinned turn to Host2."""
    llm = ScriptedTextGenerator(
        responder=lambda req, idx: (
            [
                ToolCallRequested(ToolCall(id=f"nc{idx}", name="media__play_pause", arguments={})),
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
        h1, h2 = SimulatedXtpHost(platform="macos"), SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "AlphaHost")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "BetaHost")
            await owner.grant_all_standard(str(id1), max_risk="R1")
            await owner.grant_all_standard(str(id2), max_risk="R1")

            subnet = "a0b0c0d0e0f011223344556677889900"
            for h in (h1, h2):
                await h.send_presence(
                    idle_seconds=10, screen_locked=False, media_playing=False, subnet_hash=subnet
                )

            # Build initial positive habit on AlphaHost
            for _ in range(2):
                await owner.stream_turn(
                    pet_id, "play", target_host_id=str(id1), client_subnet_hash=subnet
                )

            # Apply 4 negative redirects toward BetaHost via ask_target -> 1-word repair
            # (never positively executing AlphaHost during negative-correction rounds)
            for idx in range(4):
                if idx == 0:
                    await h1.send_presence(
                        idle_seconds=45,
                        screen_locked=False,
                        media_playing=False,
                        subnet_hash=subnet,
                    )
                else:
                    await h1.send_presence(
                        idle_seconds=45,
                        screen_locked=False,
                        media_playing=True,
                        subnet_hash=subnet,
                    )
                await h2.send_presence(
                    idle_seconds=5,
                    screen_locked=False,
                    media_playing=False,
                    subnet_hash=subnet,
                )
                t = await owner.stream_turn(pet_id, "play", client_subnet_hash=subnet)
                req = t.first("action_target_required")
                assert req is not None and req["reasonCode"] == "ask_target", t.events
                assert req["candidates"][0]["hostId"] == str(id1)
                t_rep = await owner.stream_turn(pet_id, "BetaHost", client_subnet_hash=subnet)
                rep_res = t_rep.first("action_result")
                assert rep_res is not None and rep_res["hostId"] == str(id2)

            # Equal presence: BetaHost should now win unpinned turn
            for h in (h1, h2):
                await h.send_presence(
                    idle_seconds=10, screen_locked=False, media_playing=False, subnet_hash=subnet
                )
            t_final = await owner.stream_turn(pet_id, "play", client_subnet_hash=subnet)
            res = t_final.first("action_result")
            assert res is not None and res["hostId"] == str(id2)
            habits = (await owner.list_habits(pet_id=pet_id)).json()["items"]
            alpha_habits = [h for h in habits if h["hostId"] == str(id1)]
            beta_habits = [h for h in habits if h["hostId"] == str(id2)]
            assert alpha_habits and all(
                h["negativeScore"] > 0 and h["netScore"] < 0.50 for h in alpha_habits
            )
            assert any(h["netScore"] > 0.50 for h in beta_habits)
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t3_03_multi_host_parallel_cu_and_tier1_estop_http_and_voice_ws(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T3-03: Parallel CU lanes + Tier-1 HTTP/WS E-Stop isolates target host."""
    llm = ScriptedTextGenerator(responder=lambda req, idx: [TextDelta("OK."), done_event()])
    async with mesh_server_factory(
        generator=llm,
        recognizer=FakeSpeechRecognizer(),
        synthesizer=FakeSpeechSynthesizer(),
    ) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(
            platform="macos", tools=make_standard_mesh_tools(platform="macos", include_cu=True)
        )
        h_win = SimulatedXtpHost(
            platform="windows", tools=make_standard_mesh_tools(platform="windows", include_cu=True)
        )
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "MacCU")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "WinCU")
            await owner.grant_all_standard(str(id_mac), max_risk="R3")
            await owner.grant_all_standard(str(id_win), max_risk="R3")

            for hid in (id_mac, id_win):
                r = await owner.invoke_tool_dev(
                    str(hid), "cu.session_start", {"goal": f"Run on {hid}"}
                )
                assert r.status_code == 200, r.text

            # Tier-1 E-Stop on Mac via Voice WS action.stop {hostId: MacCU}
            voice = await owner.connect_voice_ws(pet_id)
            try:
                await voice.send_stop(host_id=str(id_mac))
                ev = await voice.recv_event(wait_s=5.0)
                assert ev["type"] == "action_stopped"
                assert ev.get("hostId") == str(id_mac)
            finally:
                await voice.close()

            await asyncio.sleep(0.05)
            assert len(h_mac.received_stops) == 1
            assert len(h_win.received_stops) == 0

            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                lanes = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT host_id FROM cu_sessions WHERE active = true"
                    ).fetchall()
                }
            assert lanes == {str(id_win)}

            # Tier-2 global E-Stop stops remaining Win lane
            stop_all = await owner.emergency_stop(host_id=None)
            assert stop_all.status_code == 200
            assert stop_all.json()["hostId"] is None
            await asyncio.sleep(0.05)
            assert len(h_win.received_stops) == 1
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t3_04_xtp_presence_prober_and_battery_advisor_no_fabrication(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M4] T3-04: Low-battery laptop (10%) triggers advisory recommending battery-less PC."""
    tool_outputs: list[str] = []

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        for m in req.messages:
            if m.role == Role.TOOL:
                tool_outputs.append(m.content)
        if idx == 0:
            return [
                ToolCallRequested(
                    ToolCall(id="bat1", name="mesh__status", arguments={"includeVitals": True})
                ),
                done_event(),
            ]
        return [TextDelta("Checked battery and status."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(platform="macos")
        h_win = SimulatedXtpHost(platform="windows")
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "MacBookAir")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "DesktopTower")
            await owner.grant_all_standard(str(id_mac), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")

            await h_mac.send_presence(
                idle_seconds=4,
                screen_locked=False,
                media_playing=False,
                battery_percent=10,
                is_charging=False,
            )
            await h_win.send_presence(
                idle_seconds=8,
                screen_locked=False,
                media_playing=False,
                include_battery=False,
            )

            await owner.stream_turn(pet_id, "check mesh battery status")
            assert len(tool_outputs) == 1
            parsed = json.loads(tool_outputs[0])
            by_id = {h["hostId"]: h for h in parsed["hosts"]}
            assert by_id[str(id_mac)]["presence"]["batteryPercent"] == 10
            assert by_id[str(id_win)]["presence"]["batteryPercent"] is None
            assert by_id[str(id_mac)].get("lowBattery") is True
            assert by_id[str(id_win)].get("lowBattery") is not True
            advisories = parsed.get("batteryAdvisories") or []
            assert advisories and any("MacBookAir" in a and "DesktopTower" in a for a in advisories)
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t3_05_teleport_pipeline_mission_envelope_https_guard_and_tier1_estop(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M4] T3-05: Teleport validates HTTPS, emits MissionEnvelope, supports Tier-1 E-Stop."""
    urls_to_return = ["https://example.org/workspace", "file:///etc/passwd"]
    call_idx = 0

    def _tool_handler(name: str, args: dict[str, Any], _meta: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_idx
        if name == "browser.current_url":
            u = urls_to_return[min(call_idx, len(urls_to_return) - 1)]
            call_idx += 1
            return {"status": "ok", "structuredContent": {"url": u}}
        if name == "url.open":
            return {"status": "ok", "structuredContent": {"opened": True}}
        return {"status": "ok", "structuredContent": {"ok": True}}

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        tool_names = {t.name for t in req.tools}
        if not any(m.role == Role.TOOL for m in req.messages):
            if "mesh__teleport" in tool_names:
                return [
                    ToolCallRequested(
                        ToolCall(
                            id=f"tp{idx}",
                            name="mesh__teleport",
                            arguments={"sourceHost": "MacBook", "targetHost": "WinTower"},
                        )
                    ),
                    done_event(),
                ]
            return [
                ToolCallRequested(
                    ToolCall(id=f"br{idx}", name="browser__current_url", arguments={})
                ),
                done_event(),
            ]
        return [TextDelta("Teleport handled."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(
            platform="macos",
            tools=make_standard_mesh_tools(platform="macos", include_teleport=True),
            tool_handler=_tool_handler,
        )
        h_win = SimulatedXtpHost(
            platform="windows",
            tools=make_standard_mesh_tools(platform="windows", include_teleport=True),
            tool_handler=_tool_handler,
        )
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "MacBook")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "WinTower")
            await owner.grant_all_standard(str(id_mac), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            # Turn 1: Valid HTTPS teleport emits MissionEnvelope with steps across both hosts
            t1 = await owner.stream_turn(
                pet_id, "teleport tab from MacBook to WinTower", on_event=_approve
            )
            pending = t1.first("action_pending")
            res = t1.first("action_result")
            assert pending is not None
            env = pending.get("missionEnvelope")
            assert env is not None and env.get("goal") and env.get("missionId") and env.get("steps")
            assert res is not None and res["status"] == "ok"
            assert res.get("missionEnvelope") is not None
            assert any(
                c["name"] == "url.open"
                and c["arguments"].get("url") == "https://example.org/workspace"
                for c in h_win.received_tool_calls
            )

            # Turn 2: Non-HTTPS URL (file:///etc/passwd) MUST fail closed on WinTower
            before_count = len(h_win.received_tool_calls)
            t2 = await owner.stream_turn(
                pet_id, "teleport tab from MacBook to WinTower", on_event=_approve
            )
            res2 = t2.first("action_result")
            assert res2 is not None and res2["status"] != "ok"
            assert len(h_win.received_tool_calls) == before_count

            # Tier-1 E-Stop on WinTower isolates WinTower while keeping MacBook operational
            stop_win = await owner.emergency_stop(host_id=str(id_win))
            assert stop_win.status_code == 200, stop_win.text
            assert stop_win.json()["hostId"] == str(id_win)
            await asyncio.sleep(0.05)
            assert len(h_win.received_stops) == 1
            assert len(h_mac.received_stops) == 0

            mac_ok = await owner.invoke_tool_dev(str(id_mac), "app.launch", {"id": "editor_x"})
            assert mac_ok.status_code == 200, mac_ok.text
            assert mac_ok.json()["status"] == "ok"
            win_blocked = await owner.invoke_tool_dev(
                str(id_win), "url.open", {"url": "https://example.org/after_stop"}
            )
            assert win_blocked.status_code != 200 or win_blocked.json()["status"] != "ok"
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t3_06_hard_pin_scoped_catalog_prevents_cross_os_schema_collision(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T3-06: Pinned catalog exposes ONLY target OS tool inputSchema properties."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(
                    ToolCall(id="sc1", name="input__click", arguments={"macOption": True})
                ),
                done_event(),
            ],
            [TextDelta("Done."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()

        mac_click = make_tool_descriptor(
            "input.click",
            family="input",
            risk="R1",
            platform="macos",
            input_schema={
                "type": "object",
                "properties": {"macOption": {"type": "boolean"}},
                "additionalProperties": False,
            },
        )
        win_click = make_tool_descriptor(
            "input.click",
            family="input",
            risk="R1",
            platform="windows",
            input_schema={
                "type": "object",
                "properties": {"winCtrl": {"type": "boolean"}},
                "additionalProperties": False,
            },
        )
        h_mac = SimulatedXtpHost(platform="macos", tools=[mac_click])
        h_win = SimulatedXtpHost(platform="windows", tools=[win_click])
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "Mac")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "Win")
            await owner.grant_all_standard(str(id_mac), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")

            await owner.stream_turn(pet_id, "@Mac click")
            offered = {t.name: t for t in llm.requests[0].tools}
            click_spec = offered["input__click"]
            props = click_spec.parameters.get("properties", {})
            assert "macOption" in props
            assert "winCtrl" not in props
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t3_07_autonomous_job_hard_pin_and_offline_wol_deferral(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M4] T3-07: AutonomousJob pinned to offline host with wol_mac does not drift & defers."""
    llm = ScriptedTextGenerator(
        responder=lambda _req, _idx: [
            ToolCallRequested(ToolCall(id="jw1", name="app__launch", arguments={"id": "app_x"})),
            done_event(),
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(platform="macos")
        h_win = SimulatedXtpHost(platform="windows")
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "OnlineMac")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "WakeWin")
            await owner.grant_all_standard(str(id_mac), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")
            patch_r = await owner.patch_host(
                str(id_win), {"schemaVersion": 1, "wolMac": "AA:BB:CC:DD:EE:11"}
            )
            assert patch_r.status_code == 200, patch_r.text
            await h_win.disconnect()

            job_r = await owner.create_job(
                pet_id=pet_id,
                goal="Step 1: open app\nStep 2: check\nStep 3: done",
                host_id=str(id_win),
                max_steps=4,
            )
            assert job_r.status_code == 201
            job_id = job_r.json()["jobId"]
            await asyncio.sleep(0.2)

            # Must never drift onto OnlineMac and must preserve hard pin + emit wake deferral
            assert len(h_mac.received_tool_calls) == 0
            job_before = await owner.get_job(job_id)
            assert job_before.status_code == 200, job_before.text
            job_payload = job_before.json()
            assert job_payload["hostId"] == str(id_win)
            assert job_payload["state"] in (
                "queued",
                "running",
                "paused",
                "waiting_wake",
                "deferred_wake",
            )

            # Explicit turn targeting WakeWin while offline emits host_wake_request & defers
            t_wake = await owner.stream_turn(pet_id, "launch app_x @WakeWin")
            req_wake = t_wake.first("action_target_required")
            assert req_wake is not None and req_wake["reasonCode"] == "host_offline"
            assert req_wake["candidates"][0]["wolAvailable"] is True
            wake_ev = t_wake.first("host_wake_request")
            assert wake_ev is not None, f"Expected host_wake_request, got {t_wake.events}"
            assert wake_ev["hostId"] == str(id_win)
            assert wake_ev["macAddress"] == "AA:BB:CC:DD:EE:11"

            # Reconnect WakeWin cleanly; R2 app.launch MUST NOT run silently before approval
            await h_win.connect_reconnect(srv.ws_url, host_id=id_win)
            await asyncio.sleep(0.1)
            assert not any(c["name"] == "app.launch" for c in h_win.received_tool_calls)
            await owner.http.post(
                f"/v1/actions/jobs/{job_id}/cancel",
                headers=owner.headers,
                json={"schemaVersion": 1, "reason": "test_done"},
            )
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t3_08_xtp_focus_request_updates_host_twin_and_voice_ws(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T3-08: Desktop xuna/focusRequest updates owner focus visible to Voice WS."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="vf1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Paused."), done_event()],
        ]
    )
    async with mesh_server_factory(
        generator=llm,
        recognizer=FakeSpeechRecognizer(),
        synthesizer=FakeSpeechSynthesizer(),
    ) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(platform="macos")
        h_win = SimulatedXtpHost(platform="windows")
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "Mac")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "Win")
            await owner.grant_all_standard(str(id_mac), max_risk="R1")
            await owner.grant_all_standard(str(id_win), max_risk="R1")

            await h_mac.send_focus_request(source="hotkey", follow_me=False)
            voice = await owner.connect_voice_ws(pet_id)
            try:
                await voice.send_turn_commit("pause music")
                events = await voice.receive_until_turn_done()
                res = _require_ws_event(events, "action_result")
                assert res["hostId"] == str(id_mac)
            finally:
                await voice.close()
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t3_09_xtp_resources_digest_feeds_find_resource_and_scorer_m_res(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T3-09: xuna/resources populates mesh__find_resource and drives M_res=1.00 routing."""
    tool_outputs: list[str] = []

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        for m in req.messages:
            if m.role == Role.TOOL:
                tool_outputs.append(m.content)
        if idx == 0:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="f3d",
                        name="mesh__find_resource",
                        arguments={"query": "blender_3d", "kind": "app"},
                    )
                ),
                done_event(),
            ]
        if idx == 1:
            return [
                ToolCallRequested(
                    ToolCall(id="l3d", name="app__launch", arguments={"id": "blender_3d"})
                ),
                done_event(),
            ]
        return [TextDelta("Launched blender_3d."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(platform="macos")
        h_win = SimulatedXtpHost(platform="windows")
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "Mac3D")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "WinPlain")
            await owner.grant_all_standard(str(id_mac), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")

            subnet = "556677889900112233445566778899aa"
            await h_mac.send_presence(
                idle_seconds=5, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await h_win.send_presence(
                idle_seconds=5, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await h_mac.send_resources(installed_apps=["blender_3d"], workspaces=[])
            await h_win.send_resources(installed_apps=["plain_tool_x"], workspaces=[])

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            turn = await owner.stream_turn(
                pet_id,
                "find and launch blender_3d",
                client_subnet_hash=subnet,
                on_event=_approve,
            )
            find_out = json.loads(tool_outputs[0])
            assert any(m["hostId"] == str(id_mac) for m in find_out["matches"])
            res = turn.first("action_result")
            assert res is not None and res["hostId"] == str(id_mac)
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t3_10_voice_ws_ask_target_1_word_repair_updates_habit(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T3-10: Voice WS turn emits ask_target; 1-word turn.commit repairs & updates habit."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="vr1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Which device?"), done_event()],
            [TextDelta("Paused on WinNode."), done_event()],
        ]
    )
    async with mesh_server_factory(
        generator=llm,
        recognizer=FakeSpeechRecognizer(),
        synthesizer=FakeSpeechSynthesizer(),
    ) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(platform="macos")
        h_win = SimulatedXtpHost(platform="windows")
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "MacNode")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "WinNode")
            await owner.grant_all_standard(str(id_mac), max_risk="R1")
            await owner.grant_all_standard(str(id_win), max_risk="R1")

            subnet = "0102030405060708090a0b0c0d0e0f10"
            for h in (h_mac, h_win):
                await h.send_presence(
                    idle_seconds=10, screen_locked=False, media_playing=False, subnet_hash=subnet
                )

            voice = await owner.connect_voice_ws(pet_id, client_subnet_hash=subnet)
            try:
                await voice.send_turn_commit("pause music", client_subnet_hash=subnet)
                evs1 = await voice.receive_until_turn_done()
                req = next((e for e in evs1 if e["type"] == "action_target_required"), None)
                assert req is not None and req["reasonCode"] == "ask_target"

                await voice.send_turn_commit("WinNode", client_subnet_hash=subnet)
                evs2 = await voice.receive_until_turn_done()
                res = next((e for e in evs2 if e["type"] == "action_result"), None)
                assert res is not None and res["hostId"] == str(id_win)
                assert res.get("targetReason")
                assert res.get("targetPlatform") == "windows"
            finally:
                await voice.close()

            habits = (await owner.list_habits(pet_id=pet_id)).json()["items"]
            assert any(h["hostId"] == str(id_win) and h["positiveScore"] >= 1.0 for h in habits)
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t3_11_patch_host_place_and_nicknames_enables_at_place_routing(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T3-11: PATCH place/nicknames + 409 guard + @studio deterministic routing."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="st1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Done."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1, h2 = SimulatedXtpHost(platform="macos"), SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "MacEdit")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "WinOther")
            await owner.grant_all_standard(str(id1), max_risk="R1")
            await owner.grant_all_standard(str(id2), max_risk="R1")

            r_patch = await owner.patch_host(
                str(id1),
                {"schemaVersion": 1, "place": "studio", "nicknames": ["máy dựng"]},
            )
            assert r_patch.status_code == 200
            r_dup = await owner.patch_host(str(id2), {"schemaVersion": 1, "alias": "macedit"})
            assert r_dup.status_code == 409

            turn = await owner.stream_turn(pet_id, "@studio pause")
            res = turn.first("action_result")
            assert res is not None and res["hostId"] == str(id1)
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t3_12_linux_server_registration_zero_hardcode_routing(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T3-12: linux server workflow.run routes R2 without battery/lock penalty."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="lx1", name="workflow__run", arguments={})),
                done_event(),
            ],
            [TextDelta("Workflow ran."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        srv_tools = [
            make_tool_descriptor("host.status", family="host", risk="R0", platform="linux"),
            make_tool_descriptor("workflow.run", family="workflow", risk="R2", platform="linux"),
        ]
        h_srv = SimulatedXtpHost(platform="linux", tools=srv_tools)
        try:
            id_srv = await h_srv.connect_enroll(srv.ws_url, owner, "LinuxServer")
            await owner.patch_host(str(id_srv), {"schemaVersion": 1, "hostKind": "server"})
            await owner.grant_all_standard(str(id_srv), max_risk="R2")
            await h_srv.send_presence(
                idle_seconds=0, screen_locked=False, media_playing=False, include_battery=False
            )

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            turn = await owner.stream_turn(pet_id, "run workflow", on_event=_approve)
            res = turn.first("action_result")
            assert res is not None and res["status"] == "ok"
            assert res["hostId"] == str(id_srv)
            assert res.get("targetPlatform") == "linux"
        finally:
            await h_srv.disconnect()
            await owner.aclose()


async def test_t3_13_universal_clipboard_dual_redaction_1200_cap_and_fail_closed(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M4] T3-13: Universal Clipboard caps at 1200 chars, redacts secrets, enforces R2 approval."""
    synthetic_key = "sk-" + "secretToken1234567890"
    long_clip = f"Report header {synthetic_key} " + ("x" * 1450)

    def _clip_handler(name: str, args: dict[str, Any], _meta: dict[str, Any]) -> dict[str, Any]:
        if name == "selection.read":
            # Intentionally return unredacted raw secret + >1200 chars so backend
            # redaction and 1200-char capping on clipboard.write are genuinely exercised.
            return {
                "status": "ok",
                "structuredContent": {
                    "text": long_clip,
                    "truncated": False,
                    "redactedCount": 0,
                },
            }
        if name == "clipboard.write":
            return {"status": "ok", "structuredContent": {"written": True}}
        return {"status": "ok", "structuredContent": {"ok": True}}

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        tool_names = {t.name for t in req.tools}
        if not any(m.role == Role.TOOL for m in req.messages):
            if "mesh__clipboard_transfer" in tool_names:
                return [
                    ToolCallRequested(
                        ToolCall(
                            id="cb1",
                            name="mesh__clipboard_transfer",
                            arguments={"sourceHost": "MacClip", "targetHost": "WinClip"},
                        )
                    ),
                    done_event(),
                ]
            return [
                ToolCallRequested(
                    ToolCall(id="cb_r", name="selection__read", arguments={"maxChars": 1200})
                ),
                done_event(),
            ]
        return [TextDelta("Clipboard synced."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(
            platform="macos",
            tools=make_standard_mesh_tools(platform="macos", include_clipboard=True),
            tool_handler=_clip_handler,
        )
        h_win = SimulatedXtpHost(
            platform="windows",
            tools=make_standard_mesh_tools(platform="windows", include_clipboard=True),
            tool_handler=_clip_handler,
        )
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "MacClip")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "WinClip")
            await owner.grant_all_standard(str(id_mac), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            turn = await owner.stream_turn(
                pet_id,
                "copy selection from MacClip to WinClip",
                on_event=_approve,
            )
            pending = turn.first("action_pending")
            assert pending is not None and pending.get("targetReason")
            write_calls = [c for c in h_win.received_tool_calls if c["name"] == "clipboard.write"]
            assert len(write_calls) == 1
            written_text = write_calls[0]["arguments"]["text"]
            assert len(written_text) <= 1200
            assert synthetic_key not in written_text
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t3_14_smart_fanout_pause_media_idempotent_false_and_fleet_grants(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M4] T3-14: smart_pause_media pauses only mediaPlaying=True hosts; checks grants."""

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        tool_names = {t.name for t in req.tools}
        if not any(m.role == Role.TOOL for m in req.messages):
            if "mesh__smart_pause_media" in tool_names:
                return [
                    ToolCallRequested(
                        ToolCall(id="sp1", name="mesh__smart_pause_media", arguments={})
                    ),
                    done_event(),
                ]
            return [
                ToolCallRequested(
                    ToolCall(id="sp_fallback", name="media__play_pause", arguments={})
                ),
                done_event(),
            ]
        return [TextDelta("Paused playing hosts."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_play_1 = SimulatedXtpHost(platform="macos")
        h_paused = SimulatedXtpHost(platform="windows")
        h_play_2 = SimulatedXtpHost(platform="linux")
        try:
            id1 = await h_play_1.connect_enroll(srv.ws_url, owner, "PlayingMac")
            id2 = await h_paused.connect_enroll(srv.ws_url, owner, "PausedWin")
            id3 = await h_play_2.connect_enroll(srv.ws_url, owner, "PlayingLinux")
            for hid in (id1, id2, id3):
                await owner.grant_all_standard(str(hid), max_risk="R1")

            await h_play_1.send_presence(idle_seconds=5, screen_locked=False, media_playing=True)
            await h_paused.send_presence(idle_seconds=5, screen_locked=False, media_playing=False)
            await h_play_2.send_presence(idle_seconds=5, screen_locked=False, media_playing=True)

            await owner.stream_turn(pet_id, "pause media on all hosts")
            assert len(h_play_1.received_tool_calls) == 1
            assert len(h_play_2.received_tool_calls) == 1
            assert len(h_paused.received_tool_calls) == 0
        finally:
            await h_play_1.disconnect()
            await h_paused.disconnect()
            await h_play_2.disconnect()
            await owner.aclose()
