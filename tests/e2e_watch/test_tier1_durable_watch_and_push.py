"""Tier 1 Watch & Wake E2E: Migration 0023, Durable `WAITING_WATCH`, <1s Push, Controls & Ack."""

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import Any

import psycopg
import pytest

from tests.e2e_watch.conftest import fetch_scalar, pg_dsn, run_alembic
from tests.e2e_watch.xtp_sim import (
    ScriptedTextGenerator,
    SimulatedWatchOwnerClient,
    SimulatedWatchXtpHost,
    done_event,
    wait_until,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatRequest,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)

pytestmark = [pytest.mark.postgres, pytest.mark.anyio]


async def test_t1_01_migration_0023_schema_indexes_and_roundtrip(clean_watch_db: str) -> None:
    """Migration 0023 head, `job_watches` & `wake_inbox_items` schema, 0022->0023 roundtrip."""
    dsn = pg_dsn(clean_watch_db)
    try:
        with psycopg.connect(dsn) as conn:
            rev = fetch_scalar(conn, "SELECT version_num FROM alembic_version")
            assert rev == "0023", f"Expected Alembic head '0023', got {rev!r}"

            tables = {
                r[0]
                for r in conn.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    """
                ).fetchall()
            }
            assert {"job_watches", "wake_inbox_items"} <= tables

            watch_cols = {
                r[0]
                for r in conn.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'job_watches'
                    """
                ).fetchall()
            }
            assert {
                "watch_id",
                "job_id",
                "owner_id",
                "host_id",
                "mode",
                "signals",
                "target_spec",
                "done_criteria",
                "continuation_mode",
                "quiescence_seconds",
                "max_wait_seconds",
                "max_wake_checks",
                "wake_checks_used",
                "status",
                "last_event_kind",
                "expires_at",
                "created_at",
                "updated_at",
            } <= watch_cols

            inbox_cols = {
                r[0]
                for r in conn.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'wake_inbox_items'
                    """
                ).fetchall()
            }
            assert {
                "item_id",
                "owner_id",
                "pet_id",
                "job_id",
                "host_id",
                "kind",
                "title",
                "summary",
                "suggested_actions",
                "gesture_cue",
                "read_at",
                "dismissed_at",
                "created_at",
            } <= inbox_cols

            indexes = {
                r[0]
                for r in conn.execute(
                    """
                    SELECT indexname FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND tablename IN ('job_watches', 'wake_inbox_items')
                    """
                ).fetchall()
            }
            assert {
                "ix_job_watches_job_id",
                "ix_job_watches_owner_id",
                "ix_job_watches_host_id",
                "ix_job_watches_status",
                "ix_job_watches_expires_at",
                "ix_wake_inbox_items_owner_id",
                "ix_wake_inbox_items_read_at",
                "ix_wake_inbox_items_created_at",
            } <= indexes

        run_alembic(clean_watch_db, "downgrade", "0022")
        run_alembic(clean_watch_db, "upgrade", "head")
        with psycopg.connect(dsn) as conn:
            assert fetch_scalar(conn, "SELECT version_num FROM alembic_version") == "0023"
    finally:
        run_alembic(clean_watch_db, "upgrade", "head")


async def test_t1_02_durable_watch_zero_tokens_during_sleep_and_subsecond_push(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
) -> None:
    """`job__wait` -> `WAITING_WATCH` -> 0 LLM calls during sleep -> <1s push wake -> WakeInbox."""
    dsn = pg_dsn(clean_watch_db)

    def _respond(req: ChatRequest, idx: int) -> list[Any]:
        user_texts = [str(m.content) for m in req.messages]
        # Interactive CU start turn
        if any("Start CU session" in t for t in user_texts) and idx == 0:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="cu_start_1",
                        name="cu__session_start",
                        arguments={"goal": "Monitor build", "apps": ["editor"]},
                    )
                ),
                done_event(),
            ]
        if idx == 1:
            return [TextDelta("CU session active."), done_event()]
        # Autonomous job step 1: invoke job__wait
        if any("Watch the build" in t for t in user_texts) and not any(
            "[Host watch 'settled']" in t for t in user_texts
        ):
            return [
                ToolCallRequested(
                    ToolCall(
                        id="wait_1",
                        name="job__wait",
                        arguments={
                            "mode": "composite",
                            "signals": ["ui_quiescence", "process_idle"],
                            "quiescenceSeconds": 8,
                            "maxWaitSeconds": 300,
                            "doneCriteria": "Build output reports finished",
                            "continuationMode": "report_and_ask",
                        },
                    )
                ),
                done_event(),
            ]
        # Post-wake observation tool (Rule H1)
        if any("[Host watch 'settled']" in t for t in user_texts) and not any(
            '"snapshotId"' in t for t in user_texts
        ):
            return [
                ToolCallRequested(ToolCall(id="obs_1", name="ui__inspect", arguments={})),
                done_event(),
            ]
        return [TextDelta("Build finished and verified via ui.inspect."), done_event()]

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
            await owner.self_enroll_mobile()
            host_id = await host.connect_enroll(srv.ws_url, owner, "WatchMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")

            # Start a real CU session so `cu_sessions.last_window` populates pid & windowId
            async def _approve_r3(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    r = await owner.decide_approval(
                        data["approvalId"],
                        "approve",
                        host_id=data["hostId"],
                        args_sha256_hex=data["argsSha256"],
                        expires_at=data["expiresAt"],
                        sign_r3=True,
                    )
                    assert r.status_code == 200, r.text

            cu_turn = await owner.stream_turn(
                pet_id,
                "Start CU session on @WatchMac",
                target_host_id=str(host_id),
                on_event=_approve_r3,
            )
            assert cu_turn.first("finished") is not None

            voice_ws = await owner.connect_voice_ws(pet_id, target_host_id=str(host_id))

            job_resp = await owner.create_job(
                pet_id=pet_id,
                goal="Watch the build until done",
                host_id=str(host_id),
                max_steps=8,
                sub_goals=[
                    {"id": "sg1", "title": "Wait & verify", "description": "Watch the build"}
                ],
            )
            assert job_resp.status_code == 201, job_resp.text
            job_id = job_resp.json()["jobId"]

            await wait_until(
                lambda: len(host.received_watch_registers) >= 1,
                description="xuna/watchRegister RPC",
            )
            reg = host.received_watch_registers[0]
            assert reg["mode"] == "composite"
            assert reg["signals"] == ["ui_quiescence", "process_idle"]
            assert reg["releaseFocus"] is True
            assert reg["target"]["pid"] == 48210
            assert reg["target"]["windowId"] == 1042
            assert reg["target"]["appId"] == "editor"

            # Verify durable PostgreSQL state
            with psycopg.connect(dsn) as conn:
                job_state = fetch_scalar(
                    conn,
                    "SELECT state FROM autonomous_jobs WHERE job_id = %s",
                    (uuid.UUID(job_id),),
                )
                watch_status = fetch_scalar(
                    conn,
                    "SELECT status FROM job_watches WHERE watch_id = %s",
                    (uuid.UUID(reg["watchId"]),),
                )
            assert job_state == "waiting_watch"
            assert watch_status == "active"

            # Verify GET /v1/actions/jobs/{job_id}/watch summary
            summary_resp = await owner.get_job_watch(job_id)
            assert summary_resp.status_code == 200, summary_resp.text
            summary = summary_resp.json()
            assert summary["watchId"] == reg["watchId"]
            assert summary["status"] == "active"
            assert summary["mode"] == "composite"
            assert summary["signals"] == ["ui_quiescence", "process_idle"]
            assert summary["wakeChecksUsed"] == 0

            # 0 LLM calls during wait
            llm_calls_while_sleeping = len(llm.requests)
            await asyncio.sleep(0.25)
            assert len(llm.requests) == llm_calls_while_sleeping

            # Push xuna/watchEvent(settled) over real WebSocket and measure wake latency (< 1s)
            t0 = time.monotonic()
            await host.send_watch_event(
                watch_id=reg["watchId"],
                job_id=job_id,
                kind="settled",
                observed_signals=["ui_quiescence", "process_idle"],
                elapsed_seconds=12,
                change_count=3,
                digest_changed=True,
            )

            wake_ws_ev, _ = await voice_ws.recv_until_wake_inbox(wait_s=5.0)
            wake_latency_s = time.monotonic() - t0
            assert wake_latency_s < 1.0, f"Wake took {wake_latency_s:.3f}s (>= 1.0s)"
            assert wake_ws_ev["kind"] == "job_report"
            assert wake_ws_ev["jobId"] == job_id

            final_job = (await owner.get_job(job_id)).json()
            assert final_job["state"] == "completed"
            assert any(c["name"] == "ui.inspect" for c in host.received_tool_calls)

            inbox_resp = await owner.list_wake_inbox(unread_only=True)
            assert inbox_resp.status_code == 200
            inbox_data = inbox_resp.json()
            assert inbox_data["unreadCount"] == 1
            assert inbox_data["items"][0]["itemId"] == wake_ws_ev["itemId"]

            summary_after = (await owner.get_job_watch(job_id)).json()
            assert summary_after["status"] == "triggered"
            assert summary_after["wakeChecksUsed"] == 1
            assert summary_after["lastEventKind"] == "settled"
        finally:
            if voice_ws is not None:
                await voice_ws.close()
            await host.disconnect()
            await owner.aclose()


async def test_t1_03_report_modes_wake_now_and_ack_idempotency(
    watch_server_factory: Callable[..., Any],
) -> None:
    """`report_only`/`report_and_ask` pause, `wake-now`, and `ack` idempotency."""

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        msgs = [str(m.content) for m in req.messages]
        last_user = next(
            (str(m.content) for m in reversed(req.messages) if m.role.value == "user"), ""
        )
        if "Sub-Goal 1" in last_user and not any("[Host watch" in m for m in msgs):
            cont = "report_only" if "REPORT_ONLY" in " ".join(msgs) else "report_and_ask"
            return [
                ToolCallRequested(
                    ToolCall(
                        id=f"w_{cont}",
                        name="job__wait",
                        arguments={
                            "mode": "timer",
                            "quiescenceSeconds": 4,
                            "maxWaitSeconds": 120,
                            "doneCriteria": "Step 1 finished",
                            "continuationMode": cont,
                        },
                    )
                ),
                done_event(),
            ]
        if (
            "[Host watch" in last_user or "[Owner requested immediate check]" in last_user
        ) and not any('"snapshotId"' in m for m in msgs):
            return [
                ToolCallRequested(ToolCall(id="insp", name="ui__inspect", arguments={})),
                done_event(),
            ]
        if "Sub-Goal 2" in last_user and not any(
            m for m in msgs if '"launched": true' in m or '"launched":true' in m
        ):
            return [
                ToolCallRequested(ToolCall(id="obs2", name="ui__inspect", arguments={})),
                done_event(),
            ]
        return [TextDelta("Sub-goal verified."), done_event()]

    llm = ScriptedTextGenerator(responder=_respond)
    async with watch_server_factory(generator=llm) as srv:
        owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer
        )
        other_owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer, subject="other-owner"
        )
        host = SimulatedWatchXtpHost()
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            host_id = await host.connect_enroll(srv.ws_url, owner, "CtrlMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")

            # 1) report_only pauses after sg1 with JOB_REPORT
            j1 = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="REPORT_ONLY multi-step job",
                    host_id=str(host_id),
                    sub_goals=[
                        {"id": "sg1", "title": "Sub-Goal 1", "description": "Wait 1"},
                        {"id": "sg2", "title": "Sub-Goal 2", "description": "Follow up"},
                    ],
                )
            ).json()["jobId"]
            await wait_until(
                lambda: len(host.received_watch_registers) >= 1,
                description="report_only watchRegister",
            )
            w1_id = host.received_watch_registers[0]["watchId"]
            # Use wake-now to trigger immediate observation while WAITING_WATCH!
            wn_resp = await owner.wake_now(j1)
            assert wn_resp.status_code == 200, wn_resp.text
            await wait_until(
                lambda: any(
                    c["reason"] == "owner_cancel" and c.get("watchId") == w1_id
                    for c in host.received_watch_cancels
                ),
                description="wake-now watchCancel",
            )

            async def _is_paused(jid: str) -> bool:
                return (await owner.get_job(jid)).json()["state"] == "paused"

            await wait_until(lambda: _is_paused(j1), description="j1 paused after wake-now")
            j1_data = (await owner.get_job(j1)).json()
            assert j1_data["stopReason"] == "report_only"

            async def _j1_inbox() -> list[dict[str, Any]]:
                items = (await owner.list_wake_inbox()).json()["items"]
                return (
                    items
                    if any(i["jobId"] == j1 and i["kind"] == "job_report" for i in items)
                    else []
                )

            inbox_items = await wait_until(_j1_inbox, description="j1 JOB_REPORT in wake_inbox")
            assert any(i["jobId"] == j1 and i["kind"] == "job_report" for i in inbox_items)

            # 2) report_and_ask pauses after sg1 with JOB_NEEDS_DECISION + CONTINUE_JOB
            j2 = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="REPORT_AND_ASK multi-step job",
                    host_id=str(host_id),
                    sub_goals=[
                        {"id": "sg1", "title": "Sub-Goal 1", "description": "Wait 1"},
                        {"id": "sg2", "title": "Sub-Goal 2", "description": "Follow up"},
                    ],
                )
            ).json()["jobId"]
            await wait_until(
                lambda: len(host.received_watch_registers) >= 2,
                description="report_and_ask watchRegister",
            )
            w2_id = host.received_watch_registers[1]["watchId"]
            await host.send_watch_event(
                watch_id=w2_id,
                job_id=j2,
                kind="settled",
                observed_signals=["timer"],
                elapsed_seconds=4,
            )
            await wait_until(lambda: _is_paused(j2), description="j2 paused on report_and_ask")
            j2_data = (await owner.get_job(j2)).json()
            assert j2_data["stopReason"] == "awaiting_owner_decision"

            async def _j2_decision() -> dict[str, Any] | None:
                items = (await owner.list_wake_inbox()).json()["items"]
                return next(
                    (i for i in items if i["jobId"] == j2 and i["kind"] == "job_needs_decision"),
                    None,
                )

            decision_item = await wait_until(
                _j2_decision, description="j2 JOB_NEEDS_DECISION in wake_inbox"
            )
            assert "continue_job" in decision_item["suggestedActions"]

            # 3) Ack idempotency matrix (400 missing key, 200 replay, 409 mismatch, 404 cross-owner)
            item_id = decision_item["itemId"]
            missing_key = await owner.ack_wake_inbox(item_id, include_idem=False)
            assert missing_key.status_code == 400

            key1 = str(uuid.uuid4())
            ok1 = await owner.ack_wake_inbox(item_id, dismiss=False, idempotency_key=key1)
            assert ok1.status_code == 200
            assert ok1.json()["readAt"] is not None

            replay = await owner.ack_wake_inbox(item_id, dismiss=False, idempotency_key=key1)
            assert replay.status_code == 200
            assert replay.json() == ok1.json()

            conflict = await owner.ack_wake_inbox(item_id, dismiss=True, idempotency_key=key1)
            assert conflict.status_code == 409

            await other_owner.bind_pet()
            cross = await other_owner.ack_wake_inbox(item_id, dismiss=False)
            assert cross.status_code == 404
        finally:
            await host.disconnect()
            await owner.aclose()
            await other_owner.aclose()
