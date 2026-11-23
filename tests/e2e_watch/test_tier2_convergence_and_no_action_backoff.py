"""Tier 2 Watch & Wake E2E: Multi-Signal Composite, Silent `NO_ACTION` Backoff, and Wire Parity."""

import asyncio
import json
import uuid
from collections.abc import Callable
from typing import Any

import psycopg
import pytest

from tests.e2e_watch.conftest import DESKTOP_FIXTURES_DIR, FIXTURES_DIR, fetch_scalar, pg_dsn
from tests.e2e_watch.xtp_sim import (
    ScriptedTextGenerator,
    SimulatedWatchOwnerClient,
    SimulatedWatchXtpHost,
    done_event,
    load_watch_frames_fixture,
    wait_until,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatRequest,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)
from xuna_backend.modules.hosts.application.xtp_frames import (
    WatchCancelParamsModel,
    WatchEventParamsModel,
    WatchRegisterParamsModel,
)

pytestmark = [pytest.mark.postgres, pytest.mark.anyio]


async def test_t2_01_composite_watch_and_silent_no_action_backoff(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
) -> None:
    """Composite watch + 2x `NO_ACTION` backoff (`8 -> 12 -> 18`), 0 WakeInbox on `NO_ACTION`."""
    dsn = pg_dsn(clean_watch_db)
    wake_phase = {"wakes_seen": 0, "inspected_this_wake": False}

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        watch_turns = sum(1 for m in req.messages if "[Host watch 'settled']" in str(m.content))
        if watch_turns > wake_phase["wakes_seen"]:
            wake_phase["wakes_seen"] = watch_turns
            wake_phase["inspected_this_wake"] = False

        if watch_turns == 0:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="wait_0",
                        name="job__wait",
                        arguments={
                            "mode": "composite",
                            "signals": ["ui_quiescence", "process_idle"],
                            "quiescenceSeconds": 8,
                            "maxWaitSeconds": 300,
                            "doneCriteria": "Task finishes completely",
                            "continuationMode": "report_and_ask",
                        },
                    )
                ),
                done_event(),
            ]

        if not wake_phase["inspected_this_wake"]:
            wake_phase["inspected_this_wake"] = True
            return [
                ToolCallRequested(
                    ToolCall(id=f"insp_{watch_turns}", name="ui__inspect", arguments={})
                ),
                done_event(),
            ]

        # Wakes 1 and 2 decide task is still running -> NO_ACTION re-watch with same raw arg 8
        if watch_turns < 3:
            return [
                ToolCallRequested(
                    ToolCall(
                        id=f"wait_again_{watch_turns}",
                        name="job__wait",
                        arguments={
                            "mode": "composite",
                            "signals": ["ui_quiescence", "process_idle"],
                            "quiescenceSeconds": 8,
                            "maxWaitSeconds": 300,
                            "doneCriteria": "Task finishes completely",
                            "continuationMode": "report_and_ask",
                        },
                    )
                ),
                done_event(),
            ]

        return [TextDelta("Verified complete on 3rd wake."), done_event()]

    llm = ScriptedTextGenerator(responder=_respond)
    async with watch_server_factory(generator=llm) as srv:
        owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer
        )
        host = SimulatedWatchXtpHost()
        voice_ws = None
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            host_id = await host.connect_enroll(srv.ws_url, owner, "BackoffMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")
            await host.send_presence(foreground_app_id="editor")
            voice_ws = await owner.connect_voice_ws(pet_id, target_host_id=str(host_id))

            job_id = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="Wait through NO_ACTION cycles",
                    host_id=str(host_id),
                    max_steps=20,
                    sub_goals=[{"id": "sg1", "title": "Monitor", "description": "Wait"}],
                )
            ).json()["jobId"]

            # Watch #1: quiescenceSeconds == 8
            await wait_until(
                lambda: len(host.received_watch_registers) >= 1,
                description="initial composite watchRegister",
            )
            w1 = host.received_watch_registers[0]
            assert w1["quiescenceSeconds"] == 8
            assert w1["signals"] == ["ui_quiescence", "process_idle"]

            # Wake #1 -> NO_ACTION -> Watch #2: quiescenceSeconds == 12 (8 * 1.5)
            await host.send_watch_event(
                watch_id=w1["watchId"],
                job_id=job_id,
                kind="settled",
                observed_signals=["ui_quiescence", "process_idle"],
                elapsed_seconds=9,
            )
            await wait_until(
                lambda: len(host.received_watch_registers) >= 2,
                description="2nd watchRegister after NO_ACTION",
            )
            w2 = host.received_watch_registers[1]
            assert w2["quiescenceSeconds"] == 12

            with psycopg.connect(dsn) as conn:
                inbox_cnt_1 = fetch_scalar(
                    conn,
                    "SELECT COUNT(*) FROM wake_inbox_items WHERE job_id = %s",
                    (uuid.UUID(job_id),),
                )
                checks_1 = fetch_scalar(
                    conn,
                    "SELECT wake_checks_used FROM job_watches WHERE watch_id = %s",
                    (uuid.UUID(w2["watchId"]),),
                )
            assert inbox_cnt_1 == 0
            assert checks_1 == 1
            assert (await owner.list_wake_inbox()).json()["items"] == []

            # Wake #2 -> NO_ACTION -> Watch #3: quiescenceSeconds == 18 (12 * 1.5)
            await host.send_watch_event(
                watch_id=w2["watchId"],
                job_id=job_id,
                kind="settled",
                observed_signals=["ui_quiescence", "process_idle"],
                elapsed_seconds=13,
            )
            await wait_until(
                lambda: len(host.received_watch_registers) >= 3,
                description="3rd watchRegister after NO_ACTION",
            )
            w3 = host.received_watch_registers[2]
            assert w3["quiescenceSeconds"] == 18

            with psycopg.connect(dsn) as conn:
                inbox_cnt_2 = fetch_scalar(
                    conn,
                    "SELECT COUNT(*) FROM wake_inbox_items WHERE job_id = %s",
                    (uuid.UUID(job_id),),
                )
                checks_2 = fetch_scalar(
                    conn,
                    "SELECT wake_checks_used FROM job_watches WHERE watch_id = %s",
                    (uuid.UUID(w3["watchId"]),),
                )
            assert inbox_cnt_2 == 0
            assert checks_2 == 2
            # Confirm ZERO wake_inbox_item arrived on Voice WS during NO_ACTION cycles
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(voice_ws.ws.recv(), timeout=0.1)

            # Wake #3 -> real completion -> 1 WakeInboxItem emitted & pushed!
            await host.send_watch_event(
                watch_id=w3["watchId"],
                job_id=job_id,
                kind="settled",
                observed_signals=["ui_quiescence", "process_idle"],
                elapsed_seconds=19,
            )
            wake_ev, _ = await voice_ws.recv_until_wake_inbox(wait_s=5.0)
            assert wake_ev["kind"] == "job_report"
            assert wake_ev["jobId"] == job_id

            async def _is_completed() -> bool:
                return (await owner.get_job(job_id)).json()["state"] == "completed"

            await wait_until(_is_completed, description="job COMPLETED after 3rd wake")
            with psycopg.connect(dsn) as conn:
                inbox_cnt_final = fetch_scalar(
                    conn,
                    "SELECT COUNT(*) FROM wake_inbox_items WHERE job_id = %s",
                    (uuid.UUID(job_id),),
                )
            assert inbox_cnt_final == 1
        finally:
            if voice_ws is not None:
                await voice_ws.close()
            await host.disconnect()
            await owner.aclose()


async def test_t2_02_max_wake_checks_exceeded_fails_closed(
    watch_server_factory: Callable[..., Any],
) -> None:
    """Exceeding `max_wake_checks` (8) fails closed (`FAILED`, never `COMPLETED`)."""
    state = {"wakes_seen": 0, "inspected": False}

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        watch_turns = sum(1 for m in req.messages if "[Host watch 'settled']" in str(m.content))
        if watch_turns > state["wakes_seen"]:
            state["wakes_seen"] = watch_turns
            state["inspected"] = False

        if watch_turns > 0 and not state["inspected"]:
            state["inspected"] = True
            return [
                ToolCallRequested(
                    ToolCall(id=f"obs_{watch_turns}", name="ui__inspect", arguments={})
                ),
                done_event(),
            ]
        # Always attempt another wait (never claim done!)
        return [
            ToolCallRequested(
                ToolCall(
                    id=f"wait_{watch_turns}",
                    name="job__wait",
                    arguments={
                        "mode": "timer",
                        "quiescenceSeconds": 8,
                        "maxWaitSeconds": 120,
                        "doneCriteria": "Never finishes",
                        "continuationMode": "report_and_ask",
                    },
                )
            ),
            done_event(),
        ]

    llm = ScriptedTextGenerator(responder=_respond)
    async with watch_server_factory(generator=llm) as srv:
        owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer
        )
        host = SimulatedWatchXtpHost()
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            host_id = await host.connect_enroll(srv.ws_url, owner, "LimitMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")

            job_id = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="Exhaust wake checks",
                    host_id=str(host_id),
                    max_steps=40,
                    sub_goals=[{"id": "sg1", "title": "Loop", "description": "Wait"}],
                )
            ).json()["jobId"]

            # Drive 8 settled wakes (`DEFAULT_MAX_WAKE_CHECKS = 8`); 9th job__wait trips fail-closed
            for idx in range(8):
                await wait_until(
                    lambda expected=idx + 1: len(host.received_watch_registers) >= expected,
                    description=f"watchRegister #{idx + 1}",
                )
                w = host.received_watch_registers[idx]
                if idx >= 1:
                    # Verify backoff capped at 60s
                    assert w["quiescenceSeconds"] <= 60
                await host.send_watch_event(
                    watch_id=w["watchId"],
                    job_id=job_id,
                    kind="settled",
                    observed_signals=["timer"],
                    elapsed_seconds=5,
                )

            async def _is_failed() -> bool:
                return (await owner.get_job(job_id)).json()["state"] == "failed"

            await wait_until(_is_failed, timeout_s=8.0, description="job FAILED on max_wake_checks")
            job_data = (await owner.get_job(job_id)).json()
            assert job_data["state"] == "failed"
            assert job_data["stopReason"] == "max_wake_checks_exceeded"

            async def _has_failed_item() -> bool:
                items = (await owner.list_wake_inbox()).json()["items"]
                return any(
                    i["jobId"] == job_id
                    and i["kind"] == "job_failed"
                    and "extend_watch" not in i["suggestedActions"]
                    for i in items
                )

            await wait_until(_has_failed_item, description="JOB_FAILED in wake_inbox")
            # Confirm backoff reached 60s cap (8 -> 12 -> 18 -> 27 -> 40 -> 60 -> 60...)
            q_seq = [r["quiescenceSeconds"] for r in host.received_watch_registers]
            assert q_seq[:6] == [8, 12, 18, 27, 40, 60]
        finally:
            await host.disconnect()
            await owner.aclose()


async def test_t2_03_cross_layer_watch_frames_parity_and_live_ws_validation(
    watch_server_factory: Callable[..., Any],
) -> None:
    """Byte-identical `watch_frames.json` across backend/desktop + live XTP wire parity."""
    backend_fixture = FIXTURES_DIR / "watch_frames.json"
    desktop_fixture = DESKTOP_FIXTURES_DIR / "watch_frames.json"
    assert backend_fixture.exists() and desktop_fixture.exists()
    assert backend_fixture.read_bytes() == desktop_fixture.read_bytes()

    fixture = load_watch_frames_fixture()
    frames = fixture["frames"]

    # Validate valid watchEvent fixture frames against xtp_frames.py models
    event_keys = [k for k in frames if k.startswith("watchEvent")]
    assert len(event_keys) >= 4
    for key in event_keys:
        WatchEventParamsModel.model_validate(frames[key]["params"])

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        if any(m.role == Role.TOOL for m in req.messages):
            return [TextDelta("Checked."), done_event()]
        return [
            ToolCallRequested(
                ToolCall(
                    id="w_parity",
                    name="job__wait",
                    arguments={
                        "mode": "timer",
                        "quiescenceSeconds": 4,
                        "maxWaitSeconds": 60,
                        "doneCriteria": "Parity check",
                    },
                )
            ),
            done_event(),
        ]

    llm = ScriptedTextGenerator(responder=_respond)
    async with watch_server_factory(generator=llm) as srv:
        owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer
        )
        host = SimulatedWatchXtpHost()
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            host_id = await host.connect_enroll(srv.ws_url, owner, "ParityMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")

            job_id = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="Parity test job",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "Parity", "description": "Wait"}],
                )
            ).json()["jobId"]

            await wait_until(
                lambda: len(host.received_watch_registers) >= 1,
                description="live xuna/watchRegister frame",
            )
            live_reg = host.received_watch_registers[0]
            WatchRegisterParamsModel.model_validate(live_reg)

            # Send a valid watchEvent fixture shape (`watchEventError` pattern) bound to live IDs
            err_fixture_params = dict(frames["watchEventErrorNotification"]["params"])
            err_fixture_params["watchId"] = live_reg["watchId"]
            err_fixture_params["jobId"] = job_id
            WatchEventParamsModel.model_validate(err_fixture_params)
            assert host._ws is not None
            await host._ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "xuna/watchEvent",
                        "params": err_fixture_params,
                    }
                )
            )
            await host.ping()

            async def _is_failed() -> bool:
                return (await owner.get_job(job_id)).json()["state"] == "failed"

            await wait_until(_is_failed, description="job FAILED from fixture error watchEvent")
            job_data = (await owner.get_job(job_id)).json()
            assert job_data["stopReason"] == "needs_permission"
            # Cancel any remaining or test E-Stop watchCancel wire shape
            await owner.emergency_stop(reason="owner_app")
            await wait_until(
                lambda: len(host.received_watch_cancels) >= 1,
                description="live xuna/watchCancel frame",
            )
            WatchCancelParamsModel.model_validate(host.received_watch_cancels[0])
        finally:
            await host.disconnect()
            await owner.aclose()
