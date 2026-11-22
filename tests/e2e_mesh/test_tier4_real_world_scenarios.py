"""Tier 4 Backend E2E Mesh Tests: Real-World Multi-Host Scenarios & Privacy/Honest-Code Audits.

End-to-end realistic multi-host workloads (`T4-01`..`T4-08`) plus H1-H4 privacy &
honest-code verification across real PostgreSQL, real XTP v1 WebSockets, real SSE,
and real Voice WebSocket sessions.
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import psycopg
import pytest

from tests.e2e_mesh.conftest import fetch_scalar, pg_dsn
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


async def test_t4_01_mac_windows_phone_workspace_teleport_with_follow_me_hotkey(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2 + M4] T4-01: Mac + Win + Phone workspace teleport triggered after Follow Me hotkey."""

    def _tp_handler(name: str, _args: dict[str, Any], _meta: dict[str, Any]) -> dict[str, Any]:
        if name == "browser.current_url":
            return {
                "status": "ok",
                "structuredContent": {"url": "https://example.org/project_alpha"},
            }
        if name == "workspace.capture_capsule":
            return {"status": "ok", "structuredContent": {"project": "workspace_alpha"}}
        return {"status": "ok", "structuredContent": {"opened": True, "launched": True}}

    def _responder(req: ChatRequest, _idx: int) -> list[Any]:
        tool_names = {t.name for t in req.tools}
        if not any(m.role == Role.TOOL for m in req.messages):
            if "mesh__teleport" in tool_names:
                return [
                    ToolCallRequested(
                        ToolCall(
                            id="t401_tp",
                            name="mesh__teleport",
                            arguments={"sourceHost": "MacDesk", "targetHost": "WinDesk"},
                        )
                    ),
                    done_event(),
                ]
            return [
                ToolCallRequested(
                    ToolCall(id="t401_br", name="browser__current_url", arguments={})
                ),
                done_event(),
            ]
        return [TextDelta("Workspace teleported."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(
            platform="macos",
            tools=make_standard_mesh_tools(platform="macos", include_teleport=True),
            tool_handler=_tp_handler,
        )
        h_win = SimulatedXtpHost(
            platform="windows",
            tools=make_standard_mesh_tools(platform="windows", include_teleport=True),
            tool_handler=_tp_handler,
        )
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "MacDesk")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "WinDesk")
            await owner.grant_all_standard(str(id_mac), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")

            subnet = "11112222333344445555666677778888"
            await h_mac.send_presence(
                idle_seconds=20,
                screen_locked=False,
                media_playing=False,
                subnet_hash=subnet,
                foreground_app_id="code_editor_x",
            )
            await h_mac.send_resources(
                installed_apps=["code_editor_x"], workspaces=["workspace_alpha"]
            )

            # Owner moves to WinDesk and triggers focus hotkey + Follow Me
            await h_win.send_presence(
                idle_seconds=1, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await h_win.send_focus_request(source="hotkey", follow_me=True)

            hosts_map = {h["hostId"]: h for h in await owner.list_hosts()}
            assert hosts_map[str(id_win)]["focused"] is True
            assert hosts_map[str(id_mac)]["focused"] is False

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            turn = await owner.stream_turn(
                pet_id,
                "teleport tab and workspace from MacDesk here",
                client_subnet_hash=subnet,
                on_event=_approve,
            )
            pending = turn.first("action_pending")
            res = turn.first("action_result")
            assert pending is not None
            env = pending.get("missionEnvelope")
            assert env is not None and len(env.get("steps", [])) >= 2
            assert res is not None and res["status"] == "ok"
            assert res.get("missionEnvelope") is not None
            assert "/Users/" not in json.dumps(turn.events)
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t4_02_offline_desktop_wol_wake_and_run_deferred_job(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2 + M4] T4-02: Offline Win with wol_mac emits host_wake_request & gates R2 on resume.

    Note: Path B (180s TTL expiry) is not fast-forwarded over HTTP because public Settings exposes
    no TTL override.
    """
    llm = ScriptedTextGenerator(
        responder=lambda req, idx: (
            [
                ToolCallRequested(
                    ToolCall(id=f"wol{idx}", name="app__launch", arguments={"id": "project_app_x"})
                ),
                done_event(),
            ]
            if not any(m.role == Role.TOOL for m in req.messages)
            else [TextDelta("Resumed after wake."), done_event()]
        )
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(platform="macos")
        h_win = SimulatedXtpHost(platform="windows")
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "MacOnline")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "WinSleep")
            await owner.grant_all_standard(str(id_mac), max_risk="R2")
            await owner.grant_all_standard(str(id_win), max_risk="R2")
            await owner.patch_host(str(id_win), {"schemaVersion": 1, "wolMac": "AA:BB:CC:DD:EE:FF"})
            await h_win.disconnect()

            # Explicit @WinSleep must NOT fall back to MacOnline; exposes wolAvailable=True
            t1 = await owner.stream_turn(pet_id, "launch project_app_x @WinSleep")
            req1 = t1.first("action_target_required")
            wake1 = t1.first("host_wake_request")
            assert req1 is not None and req1["reasonCode"] == "host_offline"
            assert req1["candidates"][0]["wolAvailable"] is True
            assert len(h_mac.received_tool_calls) == 0

            # Path A: host_wake_request emitted unconditionally; reconnect within TTL requires
            # explicit owner approval (never silent R2 execution on xuna/ready)
            assert wake1 is not None, f"Expected host_wake_request, got {t1.events}"
            assert wake1["hostId"] == str(id_win)
            assert wake1["macAddress"] == "AA:BB:CC:DD:EE:FF"
            await h_win.connect_reconnect(srv.ws_url, host_id=id_win)
            await asyncio.sleep(0.1)
            assert len(h_win.received_tool_calls) == 0

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            t_resume = await owner.stream_turn(pet_id, "WinSleep", on_event=_approve)
            assert t_resume.first("action_pending") is not None
            res_resume = t_resume.first("action_result")
            assert res_resume is not None and res_resume["status"] == "ok"
            assert res_resume["hostId"] == str(id_win)
            assert any(c["name"] == "app.launch" for c in h_win.received_tool_calls)

            # Clearing wolMac via PATCH disables wolAvailable and emits zero host_wake_request
            await h_win.disconnect()
            await owner.patch_host(str(id_win), {"schemaVersion": 1, "wolMac": ""})
            t2 = await owner.stream_turn(pet_id, "launch project_app_x @WinSleep")
            req2 = t2.first("action_target_required")
            assert req2 is not None and req2["candidates"][0]["wolAvailable"] is False
            assert t2.first("host_wake_request") is None
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t4_03_voice_ws_ambiguous_repair_negative_correction_and_habits_crud(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T4-03: Voice WS ask_target -> 1-word repair -> 4 redirects -> habits CRUD."""
    llm = ScriptedTextGenerator(
        responder=lambda req, idx: (
            [
                ToolCallRequested(ToolCall(id=f"v4_{idx}", name="media__play_pause", arguments={})),
                done_event(),
            ]
            if not any(m.role == Role.TOOL for m in req.messages)
            else [TextDelta("Done."), done_event()]
        )
    )
    async with mesh_server_factory(
        generator=llm,
        recognizer=FakeSpeechRecognizer("play music"),
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

            subnet = "f0e0d0c0b0a090807060504030201000"
            for h in (h_mac, h_win):
                await h.send_presence(
                    idle_seconds=10, screen_locked=False, media_playing=False, subnet_hash=subnet
                )

            voice = await owner.connect_voice_ws(pet_id, client_subnet_hash=subnet)
            try:
                await voice.send_turn_commit("play music", client_subnet_hash=subnet)
                evs1 = await voice.receive_until_turn_done()
                assert any(e["type"] == "action_target_required" for e in evs1)

                await voice.send_turn_commit("Mac", client_subnet_hash=subnet)
                evs2 = await voice.receive_until_turn_done()
                res2 = _require_ws_event(evs2, "action_result")
                assert res2["hostId"] == str(id_mac)

                # 4 redirects toward Win via ask_target -> "Win" (never positively executing Mac)
                for idx in range(4):
                    if idx == 0:
                        await h_mac.send_presence(
                            idle_seconds=45,
                            screen_locked=False,
                            media_playing=False,
                            subnet_hash=subnet,
                        )
                    else:
                        await h_mac.send_presence(
                            idle_seconds=45,
                            screen_locked=False,
                            media_playing=True,
                            subnet_hash=subnet,
                        )
                    await h_win.send_presence(
                        idle_seconds=5,
                        screen_locked=False,
                        media_playing=False,
                        subnet_hash=subnet,
                    )
                    await voice.send_turn_commit("play music", client_subnet_hash=subnet)
                    ev_a = await voice.receive_until_turn_done()
                    req_a = _require_ws_event(ev_a, "action_target_required")
                    assert req_a["reasonCode"] == "ask_target"
                    assert req_a["candidates"][0]["hostId"] == str(id_mac)
                    await voice.send_turn_commit("Win", client_subnet_hash=subnet)
                    ev_rep = await voice.receive_until_turn_done()
                    rep_res = _require_ws_event(ev_rep, "action_result")
                    assert rep_res["hostId"] == str(id_win)

                # Equalize presence: unpinned turn now routes to Win
                for h in (h_mac, h_win):
                    await h.send_presence(
                        idle_seconds=10,
                        screen_locked=False,
                        media_playing=False,
                        subnet_hash=subnet,
                    )
                await voice.send_turn_commit("play music", client_subnet_hash=subnet)
                evs_final = await voice.receive_until_turn_done()
                res_final = _require_ws_event(evs_final, "action_result")
                assert res_final["hostId"] == str(id_win)
            finally:
                await voice.close()

            habits_r = await owner.list_habits(pet_id=pet_id)
            assert habits_r.status_code == 200
            items = habits_r.json()["items"]
            assert len(items) >= 1
            assert all(item["why"] and "sampleCount" in item for item in items)
            mac_habits = [item for item in items if item["hostId"] == str(id_mac)]
            assert mac_habits and all(
                h["negativeScore"] > 0 and h["netScore"] < 0.50 for h in mac_habits
            )

            del_r = await owner.delete_habit(items[0]["habitId"])
            assert del_r.status_code == 200
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t4_04_smart_pause_media_across_fleet_keeping_focus_pin(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2 + M4] T4-04: smart_pause_media pauses only playing hosts and preserves focus pin."""

    def _responder(req: ChatRequest, _idx: int) -> list[Any]:
        tool_names = {t.name for t in req.tools}
        if not any(m.role == Role.TOOL for m in req.messages):
            if "mesh__smart_pause_media" in tool_names:
                return [
                    ToolCallRequested(
                        ToolCall(id="t404", name="mesh__smart_pause_media", arguments={})
                    ),
                    done_event(),
                ]
            return [
                ToolCallRequested(ToolCall(id="t404_fb", name="media__play_pause", arguments={})),
                done_event(),
            ]
        return [TextDelta("Paused active media hosts."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(platform="macos")
        h_win = SimulatedXtpHost(platform="windows")
        h_lin = SimulatedXtpHost(platform="linux")
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "MacPlay")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "WinIdle")
            id_lin = await h_lin.connect_enroll(srv.ws_url, owner, "LinuxPlay")
            for hid in (id_mac, id_win, id_lin):
                await owner.grant_all_standard(str(hid), max_risk="R1")

            await h_mac.send_presence(idle_seconds=5, screen_locked=False, media_playing=True)
            await h_win.send_presence(idle_seconds=5, screen_locked=False, media_playing=False)
            await h_lin.send_presence(idle_seconds=5, screen_locked=False, media_playing=True)

            await owner.set_focus(host_id=str(id_mac), follow_me=False)
            turn = await owner.stream_turn(pet_id, "pause media on all hosts")
            res = turn.first("action_result")
            assert res is not None and res["status"] == "ok"
            assert res.get("missionEnvelope") is not None
            assert len(h_mac.received_tool_calls) == 1
            assert len(h_lin.received_tool_calls) == 1
            assert len(h_win.received_tool_calls) == 0

            hosts = {h["hostId"]: h for h in await owner.list_hosts()}
            assert hosts[str(id_mac)]["focused"] is True
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await h_lin.disconnect()
            await owner.aclose()


async def test_t4_05_scoped_tier1_estop_on_runaway_host_preserves_sibling_cu_lane(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T4-05: Tier-1 E-Stop ends runaway Mac CU lane & grants without interrupting Win."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        await owner.enable_consents()
        h_mac = SimulatedXtpHost(
            platform="macos", tools=make_standard_mesh_tools(platform="macos", include_cu=True)
        )
        h_win = SimulatedXtpHost(
            platform="windows", tools=make_standard_mesh_tools(platform="windows", include_cu=True)
        )
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "RunawayMac")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "HealthyWin")
            await owner.grant_all_standard(str(id_mac), max_risk="R3")
            await owner.grant_all_standard(str(id_win), max_risk="R3")

            for hid in (id_mac, id_win):
                r = await owner.invoke_tool_dev(str(hid), "cu.session_start", {"goal": "CU work"})
                assert r.status_code == 200, r.text

            stop_mac = await owner.emergency_stop(host_id=str(id_mac))
            assert stop_mac.status_code == 200
            await asyncio.sleep(0.05)
            assert len(h_mac.received_stops) == 1
            assert len(h_win.received_stops) == 0

            # Sibling Win CU lane remains active & Win can still execute R2/R3 tools
            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                active = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT host_id FROM cu_sessions WHERE active = true"
                    ).fetchall()
                }
            assert active == {str(id_win)}

            win_app = await owner.invoke_tool_dev(
                str(id_win), "app.launch", {"id": "code_editor_x"}
            )
            assert win_app.status_code == 200, win_app.text
            assert win_app.json()["status"] == "ok"

            # Tier-2 Stop All closes Win lane too
            stop_all = await owner.emergency_stop(host_id=None)
            assert stop_all.status_code == 200
            await asyncio.sleep(0.05)
            assert len(h_win.received_stops) == 1
            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                active_after = fetch_scalar(
                    conn, "SELECT COUNT(*) FROM cu_sessions WHERE active = true"
                )
            assert active_after == 0
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t4_06_linux_server_and_bridge_dynamic_discovery_and_routing(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T4-06: Enroll linux server & bridge, discover tools dynamically, route via @place."""
    tool_outputs: list[str] = []

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        for m in req.messages:
            if m.role == Role.TOOL:
                tool_outputs.append(m.content)
        if idx == 0:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="fr_tv",
                        name="mesh__find_resource",
                        arguments={"query": "tv", "kind": "tool"},
                    )
                ),
                done_event(),
            ]
        if idx == 1:
            return [TextDelta("Found bridge tool."), done_event()]
        if idx == 2:
            return [
                ToolCallRequested(ToolCall(id="wf_run", name="workflow__run", arguments={})),
                done_event(),
            ]
        return [TextDelta("Backup workflow ran."), done_event()]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()

        srv_tools = [
            make_tool_descriptor("host.status", family="host", risk="R0", platform="linux"),
            make_tool_descriptor("workflow.run", family="workflow", risk="R2", platform="linux"),
        ]
        bridge_tools = [
            make_tool_descriptor("host.status", family="host", risk="R0", platform="linux"),
            make_tool_descriptor("tv.control", family="media", risk="R1", platform="linux"),
        ]
        h_srv = SimulatedXtpHost(platform="linux", tools=srv_tools)
        h_bridge = SimulatedXtpHost(platform="linux", tools=bridge_tools)
        try:
            id_srv = await h_srv.connect_enroll(srv.ws_url, owner, "LabServer")
            id_bridge = await h_bridge.connect_enroll(srv.ws_url, owner, "LivingBridge")
            await owner.patch_host(
                str(id_srv), {"schemaVersion": 1, "hostKind": "server", "place": "rack"}
            )
            await owner.patch_host(
                str(id_bridge), {"schemaVersion": 1, "hostKind": "bridge", "place": "living_room"}
            )
            await owner.grant_all_standard(str(id_srv), max_risk="R2")
            await owner.grant_all_standard(str(id_bridge), max_risk="R2")

            await h_srv.send_presence(
                idle_seconds=0, screen_locked=False, media_playing=False, include_battery=False
            )
            await h_bridge.send_presence(
                idle_seconds=0, screen_locked=False, media_playing=False, include_battery=False
            )

            await owner.stream_turn(pet_id, "find tv control tool")
            found = json.loads(tool_outputs[0])
            assert any(m["hostId"] == str(id_bridge) for m in found["matches"])

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            t_wf = await owner.stream_turn(pet_id, "@rack run backup workflow", on_event=_approve)
            res = t_wf.first("action_result")
            assert res is not None and res["status"] == "ok" and res["hostId"] == str(id_srv)
        finally:
            await h_srv.disconnect()
            await h_bridge.disconnect()
            await owner.aclose()


async def test_t4_07_universal_clipboard_redaction_and_low_battery_advisor(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M4] T4-07: Clipboard redacts secret + caps 1200 chars; Advisor flags 11% battery."""
    synthetic_secret = "sk-" + "test1234567890"
    clip_payload = f"OPENAI_API_KEY={synthetic_secret} " + ("z" * 1380)

    def _handler(name: str, _args: dict[str, Any], _meta: dict[str, Any]) -> dict[str, Any]:
        if name == "selection.read":
            # Return raw unredacted >1200-char payload so backend redaction & capping are verified.
            return {
                "status": "ok",
                "structuredContent": {"text": clip_payload, "truncated": False, "redactedCount": 0},
            }
        return {"status": "ok", "structuredContent": {"written": True}}

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        if any(m.role == Role.TOOL for m in req.messages):
            return [TextDelta("Clipboard transferred."), done_event()]
        last = req.messages[-1].content
        if "battery status" in last:
            return [
                ToolCallRequested(
                    ToolCall(
                        id=f"t407_bat_{idx}",
                        name="mesh__status",
                        arguments={"includeVitals": True},
                    )
                ),
                done_event(),
            ]
        if "heavy cu" in last:
            return [
                ToolCallRequested(
                    ToolCall(
                        id=f"t407_cu_{idx}",
                        name="cu__session_start",
                        arguments={"goal": "Long desktop workflow"},
                    )
                ),
                done_event(),
            ]
        tool_names = {t.name for t in req.tools}
        if "mesh__clipboard_transfer" in tool_names:
            return [
                ToolCallRequested(
                    ToolCall(
                        id=f"t407_{idx}",
                        name="mesh__clipboard_transfer",
                        arguments={"sourceHost": "MacLaptop", "targetHost": "WinDesktop"},
                    )
                ),
                done_event(),
            ]
        return [
            ToolCallRequested(
                ToolCall(id=f"t407_r_{idx}", name="selection__read", arguments={"maxChars": 1200})
            ),
            done_event(),
        ]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        phone_resp = await owner.self_enroll_mobile(alias="OwnerPhone", platform="ios")
        assert phone_resp.status_code == 200, phone_resp.text
        h_mac = SimulatedXtpHost(
            platform="macos",
            tools=make_standard_mesh_tools(
                platform="macos", include_clipboard=True, include_cu=True
            ),
            tool_handler=_handler,
        )
        h_win = SimulatedXtpHost(
            platform="windows",
            tools=make_standard_mesh_tools(
                platform="windows", include_clipboard=True, include_cu=True
            ),
            tool_handler=_handler,
        )
        try:
            id_mac = await h_mac.connect_enroll(srv.ws_url, owner, "MacLaptop")
            id_win = await h_win.connect_enroll(srv.ws_url, owner, "WinDesktop")
            await owner.grant_all_standard(str(id_mac), max_risk="R3")
            await owner.grant_all_standard(str(id_win), max_risk="R3")

            await h_mac.send_presence(
                idle_seconds=4,
                screen_locked=False,
                media_playing=False,
                battery_percent=11,
                is_charging=False,
            )
            await h_win.send_presence(
                idle_seconds=6,
                screen_locked=False,
                media_playing=False,
                include_battery=False,
            )

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    if data.get("risk") == "r3":
                        await owner.decide_approval(
                            data["approvalId"],
                            "approve",
                            host_id=data["hostId"],
                            args_sha256_hex=data["argsSha256"],
                            expires_at=data["expiresAt"],
                            sign_r3=True,
                        )
                    else:
                        await owner.decide_approval(data["approvalId"], "approve")

            await owner.stream_turn(
                pet_id, "paste selection from MacLaptop to WinDesktop", on_event=_approve
            )
            writes = [c for c in h_win.received_tool_calls if c["name"] == "clipboard.write"]
            assert len(writes) == 1
            assert len(writes[0]["arguments"]["text"]) <= 1200
            assert synthetic_secret not in writes[0]["arguments"]["text"]

            # Battery Advisor: mesh__status & heavy CU turn flag 11% unplugged MacLaptop
            # and suggest WinDesktop without fabricating a % for WinDesktop.
            await owner.stream_turn(pet_id, "check battery status")
            tool_msgs = [m.content for r in llm.requests for m in r.messages if m.role == Role.TOOL]
            status_payload = json.loads(tool_msgs[-1])
            by_id = {h["hostId"]: h for h in status_payload["hosts"]}
            assert by_id[str(id_mac)]["presence"]["batteryPercent"] == 11
            assert by_id[str(id_mac)].get("lowBattery") is True
            assert by_id[str(id_win)]["presence"]["batteryPercent"] is None
            assert by_id[str(id_win)].get("lowBattery") is not True
            advisories = status_payload.get("batteryAdvisories") or []
            assert advisories and any("MacLaptop" in a and "WinDesktop" in a for a in advisories)

            cu_turn = await owner.stream_turn(
                pet_id, "heavy cu job", target_host_id=str(id_mac), on_event=_approve
            )
            cu_pending = cu_turn.first("action_pending")
            assert cu_pending is not None
            cu_json = json.dumps(cu_turn.events)
            assert "MacLaptop" in cu_json and "WinDesktop" in cu_json
        finally:
            await h_mac.disconnect()
            await h_win.disconnect()
            await owner.aclose()


async def test_t4_08_metadata_patch_409_at_place_ambiguity_and_r3_p256_signature(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T4-08: PATCH 409 guard + @studio ambiguity filtered + R3 phone P-256 signature."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(
                    ToolCall(
                        id="r3_08",
                        name="input__click",
                        arguments={"elementId": "send_btn"},
                    )
                ),
                done_event(),
            ],
            [TextDelta("Which studio host?"), done_event()],
            [TextDelta("Clicked after R3 signature."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        # Enroll phone so P-256 phone key is trusted for R3 approval verification
        phone_resp = await owner.self_enroll_mobile(alias="OwnerPhone", platform="ios")
        assert phone_resp.status_code == 200

        h1 = SimulatedXtpHost(
            platform="macos",
            tools=make_standard_mesh_tools(platform="macos", include_cu=True),
        )
        h2 = SimulatedXtpHost(
            platform="windows",
            tools=make_standard_mesh_tools(platform="windows", include_cu=True),
        )
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "Mac1")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "Mac2")
            await owner.grant_all_standard(str(id1), max_risk="R3")
            await owner.grant_all_standard(str(id2), max_risk="R3")

            assert (
                await owner.patch_host(
                    str(id1), {"schemaVersion": 1, "place": "studio", "nicknames": ["mac dựng"]}
                )
            ).status_code == 200
            assert (
                await owner.patch_host(str(id2), {"schemaVersion": 1, "place": "studio"})
            ).status_code == 200

            # Duplicate active folded alias blocked with 409
            r_409 = await owner.patch_host(str(id2), {"schemaVersion": 1, "alias": "mac1"})
            assert r_409.status_code == 409

            # Turn 1: @studio is ambiguous between Mac1 and Mac2
            t1 = await owner.stream_turn(pet_id, "@studio submit report")
            req1 = t1.first("action_target_required")
            assert req1 is not None and req1["reasonCode"] == "ask_target"
            assert {c["hostId"] for c in req1["candidates"]} == {str(id1), str(id2)}

            # Turn 2: 1-word repair "Mac1" -> R3 action_pending -> phone P-256 signature -> ok
            async def _sign_r3(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    assert data["risk"] == "r3"
                    assert data.get("targetReason")
                    r = await owner.decide_approval(
                        data["approvalId"],
                        "approve",
                        host_id=data["hostId"],
                        args_sha256_hex=data["argsSha256"],
                        expires_at=data["expiresAt"],
                        sign_r3=True,
                    )
                    assert r.status_code == 200, r.text

            t2 = await owner.stream_turn(pet_id, "Mac1", on_event=_sign_r3)
            res2 = t2.first("action_result")
            assert res2 is not None and res2["status"] == "ok"
            assert res2["hostId"] == str(id1)
            assert any(
                c["name"] == "input.click"
                and (c["meta"].get("xuna/approval") or {}).get("signature")
                for c in h1.received_tool_calls
            )
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t4_privacy_and_honest_code_end_to_end_audit(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2 + M4] H1-H4 End-to-End Privacy & Honest-Code Postgres/API Verification."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="pr1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Done."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1 = SimulatedXtpHost(platform="macos")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "PrivacyMac")
            await owner.grant_all_standard(str(id1), max_risk="R1")
            await h1.send_presence(
                idle_seconds=2,
                screen_locked=False,
                media_playing=True,
                foreground_app_id="synthetic_editor",
            )
            await h1.send_resources(
                installed_apps=["synthetic_editor"],
                workspaces=["synthetic_workspace"],
            )
            await owner.stream_turn(pet_id, "pause", target_host_id=str(id1))
            await h1.disconnect()

            # Offline host MUST immediately show presence == None (H2)
            hosts_after = await owner.list_hosts()
            assert hosts_after[0]["online"] is False
            assert hosts_after[0]["presence"] is None

            # Inspect PostgreSQL tables for zero HostTwin / path / window-title leakage
            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                host_row = fetch_scalar(
                    conn,
                    "SELECT row_to_json(hosts) FROM hosts WHERE host_id = %s",
                    (id1,),
                )
                habits_rows = conn.execute(
                    "SELECT row_to_json(host_routing_habits) FROM host_routing_habits "
                    "WHERE host_id = %s",
                    (id1,),
                ).fetchall()
                audits = await owner.audit_events(limit=100)

            dumped = json.dumps({"host": host_row, "habits": habits_rows, "audits": audits})
            assert "synthetic_editor" not in dumped
            assert "synthetic_workspace" not in dumped
            assert "/Users/" not in dumped
            assert "C:\\\\" not in dumped
        finally:
            await h1.disconnect()
            await owner.aclose()
