"""Tier 4 Watch & Wake E2E: Autopilot `auto_continue` vs R3 High-Impact Checkpoint & E-Stop."""

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC
from typing import Any

import psycopg
import pytest

from tests.e2e_watch.conftest import fetch_scalar, pg_dsn
from tests.e2e_watch.xtp_sim import (
    ScriptedTextGenerator,
    SimulatedWatchOwnerClient,
    SimulatedWatchXtpHost,
    args_sha256,
    done_event,
    wait_until,
)
from xuna_backend.modules.actions.application import orchestrator as orch_mod
from xuna_backend.modules.cognition.domain.types import (
    ChatRequest,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)

pytestmark = [pytest.mark.postgres, pytest.mark.anyio]


async def test_t4_01_autopilot_auto_continue_r0_r2_then_stops_at_r3_timeout_and_live_approval(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autopilot auto-resumes R0-R2 (`ui.inspect`, `app.launch`) but gates `mail.send` at R3."""
    dsn = pg_dsn(clean_watch_db)

    def _build_responder() -> Callable[[ChatRequest, int], list[Any]]:
        def _respond(req: ChatRequest, _idx: int) -> list[Any]:
            msgs = [str(m.content) for m in req.messages]
            last_user = next(
                (str(m.content) for m in reversed(req.messages) if m.role.value == "user"), ""
            )
            joined = "\n".join(msgs)
            if "Sub-Goal 1" in last_user and "[Host watch" not in joined:
                return [
                    ToolCallRequested(
                        ToolCall(
                            id="w_auto",
                            name="job__wait",
                            arguments={
                                "mode": "timer",
                                "quiescenceSeconds": 4,
                                "maxWaitSeconds": 180,
                                "doneCriteria": "Build completes",
                                "continuationMode": "auto_continue",
                            },
                        )
                    ),
                    done_event(),
                ]
            if "[Host watch 'settled']" in last_user and '"snapshotId"' not in joined:
                return [
                    ToolCallRequested(ToolCall(id="obs_1", name="ui__inspect", arguments={})),
                    done_event(),
                ]
            if "Sub-Goal 1" in joined and "Sub-Goal 2" not in joined:
                return [TextDelta("Build complete, advancing to Sub-Goal 2."), done_event()]

            if "Sub-Goal 2" in last_user and '"launched"' not in joined:
                return [
                    ToolCallRequested(
                        ToolCall(id="launch_1", name="app__launch", arguments={"id": "mail_client"})
                    ),
                    done_event(),
                ]
            if '"launched"' in joined and '"sent"' not in joined:
                return [
                    ToolCallRequested(
                        ToolCall(
                            id="mail_1",
                            name="mail__send",
                            arguments={
                                "to": ["boss@example.com"],
                                "subject": "Build Report",
                                "body": "All checks passed.",
                            },
                        )
                    ),
                    done_event(),
                ]
            return [TextDelta("Report email sent."), done_event()]

        return _respond

    # Subcase A: Timeout with locked phone -> AWAITING_R3_CHECKPOINT + WakeInbox(JOB_NEEDS_DECISION)
    monkeypatch.setattr(orch_mod, "APPROVAL_TTL_SECONDS", 0.15)
    llm_a = ScriptedTextGenerator(responder=_build_responder())
    async with watch_server_factory(generator=llm_a) as srv:
        owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer
        )
        host = SimulatedWatchXtpHost()
        host.trust_preset = "autopilot"
        voice_ws = None
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            await owner.self_enroll_mobile()
            host_id = await host.connect_enroll(srv.ws_url, owner, "AutoMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")
            voice_ws = await owner.connect_voice_ws(pet_id, target_host_id=str(host_id))

            job_id = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="Wait for build then send email",
                    host_id=str(host_id),
                    max_steps=12,
                    sub_goals=[
                        {"id": "sg1", "title": "Sub-Goal 1", "description": "Wait for build"},
                        {"id": "sg2", "title": "Sub-Goal 2", "description": "Open mail & send"},
                    ],
                )
            ).json()["jobId"]

            await wait_until(
                lambda: len(host.received_watch_registers) >= 1,
                description="autopilot watchRegister",
            )
            w_id = host.received_watch_registers[0]["watchId"]
            await host.send_watch_event(
                watch_id=w_id,
                job_id=job_id,
                kind="settled",
                observed_signals=["timer"],
                elapsed_seconds=5,
            )

            async def _awaiting_r3() -> bool:
                return (await owner.get_job(job_id)).json()["state"] == "awaiting_r3_checkpoint"

            await wait_until(_awaiting_r3, timeout_s=5.0, description="AWAITING_R3_CHECKPOINT")

            # Verify R2 app.launch ran automatically with 0 prompts, mail.send did NOT execute!
            called_names = [c["name"] for c in host.received_tool_calls]
            assert "ui.inspect" in called_names
            assert "app.launch" in called_names
            assert "mail.send" not in called_names

            wake_ev, _ = await voice_ws.recv_until_wake_inbox(wait_s=5.0)
            assert wake_ev["kind"] == "job_needs_decision"
            assert wake_ev["jobId"] == job_id
            assert set(wake_ev["suggestedActions"]) == {"open_approval", "dismiss"}

            with psycopg.connect(dsn) as conn:
                inbox_kind = fetch_scalar(
                    conn,
                    "SELECT kind FROM wake_inbox_items WHERE job_id = %s",
                    (uuid.UUID(job_id),),
                )
            assert inbox_kind == "job_needs_decision"
        finally:
            if voice_ws is not None:
                await voice_ws.close()
            await host.disconnect()
            await owner.aclose()

    # Subcase B: Live P-256 phone approval executes R3 mail.send and finishes COMPLETED
    monkeypatch.setattr(orch_mod, "APPROVAL_TTL_SECONDS", 10.0)
    llm_b = ScriptedTextGenerator(responder=_build_responder())
    async with watch_server_factory(generator=llm_b) as srv:
        owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer
        )
        host = SimulatedWatchXtpHost()
        host.trust_preset = "autopilot"
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            await owner.self_enroll_mobile()
            host_id = await host.connect_enroll(srv.ws_url, owner, "AutoLiveMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")

            job_id = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="Wait for build then send email",
                    host_id=str(host_id),
                    max_steps=12,
                    sub_goals=[
                        {"id": "sg1", "title": "Sub-Goal 1", "description": "Wait for build"},
                        {"id": "sg2", "title": "Sub-Goal 2", "description": "Open mail & send"},
                    ],
                )
            ).json()["jobId"]

            await wait_until(
                lambda: len(host.received_watch_registers) >= 1,
                description="live approval watchRegister",
            )
            await host.send_watch_event(
                watch_id=host.received_watch_registers[0]["watchId"],
                job_id=job_id,
                kind="settled",
                observed_signals=["timer"],
                elapsed_seconds=5,
            )

            async def _pending_r3_id() -> str | None:
                j = (await owner.get_job(job_id)).json()
                if j["state"] == "awaiting_r3_checkpoint" and j.get("pendingApprovalId"):
                    return str(j["pendingApprovalId"])
                return None

            approval_id = await wait_until(
                _pending_r3_id, timeout_s=5.0, description="pendingApprovalId"
            )
            assert "mail.send" not in [c["name"] for c in host.received_tool_calls]

            with psycopg.connect(dsn) as conn:
                row = conn.execute(
                    "SELECT host_id, args_sha256, expires_at "
                    "FROM action_approvals WHERE approval_id = %s",
                    (uuid.UUID(approval_id),),
                ).fetchone()
            assert row is not None
            appr_host_id, appr_hash, appr_exp = str(row[0]), str(row[1]), row[2]
            expected_args = {
                "to": ["boss@example.com"],
                "subject": "Build Report",
                "body": "All checks passed.",
            }
            assert appr_hash == args_sha256(expected_args)

            approve_resp = await owner.decide_approval(
                approval_id,
                "approve",
                host_id=appr_host_id,
                args_sha256_hex=appr_hash,
                expires_at=appr_exp.astimezone(UTC).isoformat(),
                sign_r3=True,
            )
            assert approve_resp.status_code == 200, approve_resp.text

            async def _is_completed() -> bool:
                return (await owner.get_job(job_id)).json()["state"] == "completed"

            await wait_until(
                _is_completed, timeout_s=5.0, description="job COMPLETED after P-256 sign"
            )
            assert any(c["name"] == "mail.send" for c in host.received_tool_calls)
        finally:
            await host.disconnect()
            await owner.aclose()


async def test_t4_02_autopilot_high_impact_web_and_input_targets_stop_at_r3(
    watch_server_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0015 (2026-11-22): ordinary targets auto-run; publish/pay/delete stay R3."""
    monkeypatch.setattr(orch_mod, "APPROVAL_TTL_SECONDS", 0.15)

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        joined = "\n".join(str(m.content) for m in req.messages)
        if "[Active Sub-Goal 1/2]" in joined:
            if "[Host watch" not in joined:
                return [
                    ToolCallRequested(
                        ToolCall(
                            id="w_hi",
                            name="job__wait",
                            arguments={
                                "mode": "timer",
                                "quiescenceSeconds": 4,
                                "maxWaitSeconds": 120,
                                "doneCriteria": "Ready to act",
                                "continuationMode": "auto_continue",
                            },
                        )
                    ),
                    done_event(),
                ]
            if '"snapshotId"' not in joined:
                return [
                    ToolCallRequested(ToolCall(id="obs", name="ui__inspect", arguments={})),
                    done_event(),
                ]
            return [TextDelta("Wait settled and inspected."), done_event()]

        if '"clicked"' in joined:
            return [TextDelta("Ordinary click executed under autopilot."), done_event()]
        if "SAFE_WEB" in joined:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="web_safe",
                        name="web__click",
                        arguments={"ref": "btn_login", "targetText": "Đăng nhập"},
                    )
                ),
                done_event(),
            ]
        if "SAFE_INPUT" in joined:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="inp_safe",
                        name="input__click",
                        arguments={"elementId": "btn_draft", "elementName": "Save Draft"},
                    )
                ),
                done_event(),
            ]
        if "WEB_PAY" in joined:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="web_pay",
                        name="web__click",
                        arguments={"ref": "btn_1", "targetText": "Confirm Payment"},
                    )
                ),
                done_event(),
            ]
        return [
            ToolCallRequested(
                ToolCall(
                    id="inp_pub",
                    name="input__click",
                    arguments={"elementId": "pub_1", "elementName": "Publish Release"},
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
        host.trust_preset = "autopilot"
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            host_id = await host.connect_enroll(srv.ws_url, owner, "HiImpactMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")

            # 1) Contrast: ordinary web.click("Đăng nhập") & input.click("Save Draft") auto-run
            for idx_case, (tag, expected_tool) in enumerate(
                (("SAFE_WEB", "web.click"), ("SAFE_INPUT", "input.click")), start=1
            ):
                calls_before = len(host.received_tool_calls)
                job_id = (
                    await owner.create_job(
                        pet_id=pet_id,
                        goal=f"{tag} ordinary autopilot check",
                        host_id=str(host_id),
                        sub_goals=[
                            {"id": "sg1", "title": "Wait", "description": "Wait"},
                            {"id": "sg2", "title": "Act", "description": f"{tag} act"},
                        ],
                    )
                ).json()["jobId"]

                await wait_until(
                    lambda expected=idx_case: len(host.received_watch_registers) >= expected,
                    description=f"watchRegister for {tag}",
                )
                w_id = host.received_watch_registers[-1]["watchId"]
                await host.send_watch_event(
                    watch_id=w_id,
                    job_id=job_id,
                    kind="settled",
                    observed_signals=["timer"],
                    elapsed_seconds=4,
                )

                async def _completed(jid: str = job_id) -> bool:
                    return (await owner.get_job(jid)).json()["state"] == "completed"

                await wait_until(
                    _completed, timeout_s=5.0, description=f"{tag} COMPLETED without R3 prompt"
                )
                new_calls = [c["name"] for c in host.received_tool_calls[calls_before:]]
                assert expected_tool in new_calls

            # 2) High-impact: web.click("Confirm Payment") & input.click("Publish Release") -> R3
            for idx_case, (tag, blocked_tool) in enumerate(
                (("WEB_PAY", "web.click"), ("INPUT_PUB", "input.click")), start=3
            ):
                calls_before = len(host.received_tool_calls)
                job_id = (
                    await owner.create_job(
                        pet_id=pet_id,
                        goal=f"{tag} high impact check",
                        host_id=str(host_id),
                        sub_goals=[
                            {"id": "sg1", "title": "Wait", "description": "Wait"},
                            {"id": "sg2", "title": "Commit", "description": f"{tag} commit"},
                        ],
                    )
                ).json()["jobId"]

                await wait_until(
                    lambda expected=idx_case: len(host.received_watch_registers) >= expected,
                    description=f"watchRegister for {tag}",
                )
                w_id = host.received_watch_registers[-1]["watchId"]
                await host.send_watch_event(
                    watch_id=w_id,
                    job_id=job_id,
                    kind="settled",
                    observed_signals=["timer"],
                    elapsed_seconds=4,
                )

                async def _awaiting_r3(jid: str = job_id) -> bool:
                    return (await owner.get_job(jid)).json()["state"] == "awaiting_r3_checkpoint"

                await wait_until(
                    _awaiting_r3, timeout_s=5.0, description=f"{tag} AWAITING_R3_CHECKPOINT"
                )
                new_calls = [c["name"] for c in host.received_tool_calls[calls_before:]]
                assert blocked_tool not in new_calls
                await owner.cancel_job(job_id, reason="done_subtest")
        finally:
            await host.disconnect()
            await owner.aclose()


async def test_t4_03_estop_http_and_voice_ws_cancel_active_watches_and_ignore_late_events(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
) -> None:
    """HTTP `POST /v1/actions/stop` & Voice WS `action.stop` emit `watchCancel(reason='estop')`."""
    dsn = pg_dsn(clean_watch_db)

    def _respond(_req: ChatRequest, _idx: int) -> list[Any]:
        return [
            ToolCallRequested(
                ToolCall(
                    id="w_estop",
                    name="job__wait",
                    arguments={
                        "mode": "composite",
                        "signals": ["ui_quiescence", "process_idle"],
                        "quiescenceSeconds": 8,
                        "maxWaitSeconds": 300,
                        "doneCriteria": "Should be stopped by E-Stop",
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
        voice_ws = None
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            host_id = await host.connect_enroll(srv.ws_url, owner, "StopMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")
            await host.send_presence(foreground_app_id="editor")
            voice_ws = await owner.connect_voice_ws(pet_id, target_host_id=str(host_id))

            # 1) HTTP E-Stop
            j1 = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="E-Stop HTTP watch test",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "Wait", "description": "Wait"}],
                )
            ).json()["jobId"]
            await wait_until(
                lambda: len(host.received_watch_registers) >= 1,
                description="HTTP E-Stop watchRegister",
            )
            w1 = host.received_watch_registers[0]["watchId"]
            assert w1 in host.active_watches

            stop_resp = await owner.emergency_stop(reason="owner_app")
            assert stop_resp.status_code == 200, stop_resp.text

            await wait_until(
                lambda: len(host.received_watch_cancels) >= 1,
                description="HTTP E-Stop xuna/watchCancel",
            )
            c1 = host.received_watch_cancels[0]
            assert c1 == {"schemaVersion": 1, "reason": "estop"}
            assert "watchId" not in c1 and "jobId" not in c1
            assert host.active_watches == {}

            j1_after = (await owner.get_job(j1)).json()
            assert j1_after["state"] == "cancelled"
            with psycopg.connect(dsn) as conn:
                w1_status = fetch_scalar(
                    conn, "SELECT status FROM job_watches WHERE watch_id = %s", (uuid.UUID(w1),)
                )
            assert w1_status == "cancelled"

            # Late xuna/watchEvent MUST be ignored and never change cancelled -> completed
            await host.send_watch_event(
                watch_id=w1,
                job_id=j1,
                kind="settled",
                observed_signals=["ui_quiescence"],
                elapsed_seconds=10,
            )
            await asyncio.sleep(0.05)
            assert (await owner.get_job(j1)).json()["state"] == "cancelled"

            # 2) Voice WS `action.stop` (re-grant standard tools after HTTP E-Stop suspended R2+)
            await owner.grant_all_standard(str(host_id), max_risk="R3")
            j2 = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="E-Stop Voice WS watch test",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "Wait", "description": "Wait"}],
                )
            ).json()["jobId"]
            await wait_until(
                lambda: len(host.received_watch_registers) >= 2,
                description="WS E-Stop watchRegister",
            )
            w2 = host.received_watch_registers[1]["watchId"]

            await voice_ws.send_stop(reason="owner_ws")
            stop_ack = await voice_ws.recv_until_event(
                lambda ev: ev.get("type") == "action_stopped", wait_s=5.0
            )
            assert stop_ack[-1]["type"] == "action_stopped"
            assert stop_ack[-1]["stopped"] is True
            await wait_until(
                lambda: len(host.received_watch_cancels) >= 2,
                description="WS E-Stop xuna/watchCancel",
            )
            c2 = host.received_watch_cancels[1]
            assert c2 == {"schemaVersion": 1, "reason": "estop"}
            assert (await owner.get_job(j2)).json()["state"] == "cancelled"
            with psycopg.connect(dsn) as conn:
                w2_status = fetch_scalar(
                    conn, "SELECT status FROM job_watches WHERE watch_id = %s", (uuid.UUID(w2),)
                )
            assert w2_status == "cancelled"
        finally:
            if voice_ws is not None:
                await voice_ws.close()
            await host.disconnect()
            await owner.aclose()
