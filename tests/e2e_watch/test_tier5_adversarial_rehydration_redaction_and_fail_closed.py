"""Tier 5 Watch & Wake E2E: Rehydration, Fail-Closed Modes, U5 Redaction, Sensitive Paths."""

import asyncio
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from tests.e2e_watch.conftest import REPO_ROOT, fetch_scalar, pg_dsn, start_watch_server
from tests.e2e_watch.xtp_sim import (
    ScriptedTextGenerator,
    SimulatedWatchOwnerClient,
    SimulatedWatchXtpHost,
    done_event,
    wait_until,
)
from xuna_backend.modules.actions.application.computer_use import ComputerUseSessionState
from xuna_backend.modules.cognition.domain.types import (
    ChatRequest,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)
from xuna_backend.modules.hosts.domain.entities import HostId

pytestmark = [pytest.mark.postgres, pytest.mark.anyio]


async def test_t5_01_backend_restart_db_rehydration_and_completion(
    clean_watch_db: str,
) -> None:
    """Real `srv1` stop -> `srv2` boot on same DB -> reconnect -> watchEvent completes."""
    dsn = pg_dsn(clean_watch_db)

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        joined = "\n".join(str(m.content) for m in req.messages)
        if "[Host watch 'settled']" not in joined:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="w_rehydrate",
                        name="job__wait",
                        arguments={
                            "mode": "composite",
                            "signals": ["ui_quiescence", "process_idle"],
                            "quiescenceSeconds": 8,
                            "maxWaitSeconds": 300,
                            "doneCriteria": "Build survives server restart",
                        },
                    )
                ),
                done_event(),
            ]
        if '"snapshotId"' not in joined:
            return [
                ToolCallRequested(ToolCall(id="obs_reh", name="ui__inspect", arguments={})),
                done_event(),
            ]
        return [TextDelta("Resumed after server restart and verified."), done_event()]

    llm1 = ScriptedTextGenerator(responder=_respond)
    host = SimulatedWatchXtpHost()
    saved_issuer = None
    owner_subject = f"rehydrate-{uuid.uuid4().hex[:8]}"

    async with start_watch_server(clean_watch_db, generator=llm1) as srv1:
        saved_issuer = srv1.issuer
        owner1 = SimulatedWatchOwnerClient(
            base_url=srv1.base_url, ws_url=srv1.ws_url, issuer=srv1.issuer, subject=owner_subject
        )
        try:
            pet_id = await owner1.bind_pet()
            await owner1.enable_consents()
            host_id = await host.connect_enroll(srv1.ws_url, owner1, "RestartMac")
            await owner1.grant_all_standard(str(host_id), max_risk="R3")
            await host.send_presence(foreground_app_id="editor")

            job_id = (
                await owner1.create_job(
                    pet_id=pet_id,
                    goal="Survive backend restart",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "Build", "description": "Wait"}],
                )
            ).json()["jobId"]

            await wait_until(
                lambda: len(host.received_watch_registers) >= 1,
                description="srv1 watchRegister",
            )
            w1 = host.received_watch_registers[0]
            watch_id = w1["watchId"]

            with psycopg.connect(dsn) as conn:
                assert (
                    fetch_scalar(
                        conn,
                        "SELECT state FROM autonomous_jobs WHERE job_id = %s",
                        (uuid.UUID(job_id),),
                    )
                    == "waiting_watch"
                )
                assert (
                    fetch_scalar(
                        conn,
                        "SELECT status FROM job_watches WHERE watch_id = %s",
                        (uuid.UUID(watch_id),),
                    )
                    == "active"
                )
        finally:
            await host.disconnect()
            await owner1.aclose()

    # srv1 has fully stopped & disposed DB pool. Boot brand new srv2 against SAME clean_watch_db.
    host.received_watch_registers.clear()
    host.active_watches.clear()
    llm2 = ScriptedTextGenerator(responder=_respond)
    async with start_watch_server(clean_watch_db, generator=llm2, issuer=saved_issuer) as srv2:
        owner2 = SimulatedWatchOwnerClient(
            base_url=srv2.base_url, ws_url=srv2.ws_url, issuer=srv2.issuer, subject=owner_subject
        )
        try:
            await host.connect_reconnect(srv2.ws_url, host_id=host_id)
            await wait_until(
                lambda: len(host.received_watch_registers) >= 1,
                timeout_s=5.0,
                description="srv2 on_host_ready rehydration watchRegister",
            )
            rehydrated = host.received_watch_registers[0]
            assert rehydrated["watchId"] == watch_id
            assert rehydrated["jobId"] == job_id
            assert 5 <= rehydrated["maxWaitSeconds"] <= 300

            await host.send_watch_event(
                watch_id=watch_id,
                job_id=job_id,
                kind="settled",
                observed_signals=["ui_quiescence", "process_idle"],
                elapsed_seconds=15,
            )

            async def _is_completed() -> bool:
                return (await owner2.get_job(job_id)).json()["state"] == "completed"

            await wait_until(_is_completed, timeout_s=5.0, description="srv2 job COMPLETED")

            async def _has_report_item() -> bool:
                inbox = (await owner2.list_wake_inbox()).json()["items"]
                return any(i["jobId"] == job_id and i["kind"] == "job_report" for i in inbox)

            await wait_until(_has_report_item, description="srv2 JOB_REPORT in wake_inbox")
        finally:
            await host.disconnect()
            await owner2.aclose()


async def test_t5_02_reconnect_rejection_and_disconnect_expiry_sweep_plus_extend(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
) -> None:
    """Rehydrate `target_not_found` -> FAILED; disconnect + `sweep_once` -> PAUSED -> extend."""
    dsn = pg_dsn(clean_watch_db)

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        joined = "\n".join(str(m.content) for m in req.messages)
        if "[Host watch" not in joined:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="w_rec",
                        name="job__wait",
                        arguments={
                            "mode": "timer",
                            "quiescenceSeconds": 4,
                            "maxWaitSeconds": 120,
                            "doneCriteria": "Reconnect / expiry check",
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
        return [TextDelta("Recovered via extend-watch."), done_event()]

    llm = ScriptedTextGenerator(responder=_respond)
    async with watch_server_factory(generator=llm) as srv:
        owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer
        )
        host = SimulatedWatchXtpHost()
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            host_id = await host.connect_enroll(srv.ws_url, owner, "SweepMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")

            # Subcase A: Reconnect rehydration failure (`target_not_found`)
            j_fail = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="Rehydrate failure job",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "FailRehydrate", "description": "Wait"}],
                )
            ).json()["jobId"]
            await wait_until(lambda: len(host.received_watch_registers) >= 1, description="reg A")
            await host.disconnect()

            host.reject_modes["timer"] = "target_not_found"
            await host.connect_reconnect(srv.ws_url, host_id=host_id)

            async def _j_fail_failed() -> bool:
                return (await owner.get_job(j_fail)).json()["state"] == "failed"

            await wait_until(_j_fail_failed, description="rehydrate rejection FAILED")
            j_fail_data = (await owner.get_job(j_fail)).json()
            assert j_fail_data["stopReason"] == "target_not_found"

            async def _has_failed_inbox() -> bool:
                items = (await owner.list_wake_inbox()).json()["items"]
                return any(i["jobId"] == j_fail and i["kind"] == "job_failed" for i in items)

            await wait_until(_has_failed_inbox, description="rehydrate rejection WakeInbox")

            # Subcase B: Disconnect + expiry sweep (`watch_timeout`) -> no rehydrate -> extend
            host.reject_modes.pop("timer", None)
            host.received_watch_registers.clear()
            host.active_watches.clear()

            j_exp = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="Expiry and extend job",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "Expire", "description": "Wait"}],
                )
            ).json()["jobId"]
            await wait_until(lambda: len(host.received_watch_registers) >= 1, description="reg B")
            w_exp_id = host.received_watch_registers[0]["watchId"]
            await host.disconnect()

            # Backdate expires_at in DB and run deterministic sweep_once()
            with psycopg.connect(dsn) as conn:
                conn.execute(
                    "UPDATE job_watches SET expires_at = %s WHERE watch_id = %s",
                    (datetime.now(UTC) - timedelta(seconds=30), uuid.UUID(w_exp_id)),
                )
                conn.commit()
            expired_cnt = await srv.app.state.watch_coordinator.sweep_once()
            assert expired_cnt >= 1

            j_exp_paused = (await owner.get_job(j_exp)).json()
            assert j_exp_paused["state"] == "paused"
            assert j_exp_paused["stopReason"] == "watch_timeout"
            timeout_item = next(
                i
                for i in (await owner.list_wake_inbox()).json()["items"]
                if i["jobId"] == j_exp and i["kind"] == "job_timeout"
            )
            assert set(timeout_item["suggestedActions"]) == {"extend_watch", "dismiss"}

            # Reconnect host -> expired watch MUST NOT rehydrate
            host.received_watch_registers.clear()
            await host.connect_reconnect(srv.ws_url, host_id=host_id)
            await asyncio.sleep(0.1)
            assert host.received_watch_registers == []

            # Owner clicks extend-watch -> resumes WAITING_WATCH & registers new watch
            ext_resp = await owner.extend_watch(j_exp, additional_seconds=120)
            assert ext_resp.status_code == 200, ext_resp.text
            assert ext_resp.json()["status"] == "active"
            assert len(host.received_watch_registers) == 1
            new_w_id = host.received_watch_registers[0]["watchId"]

            await host.send_watch_event(
                watch_id=new_w_id,
                job_id=j_exp,
                kind="settled",
                observed_signals=["timer"],
                elapsed_seconds=5,
            )

            async def _j_exp_done() -> bool:
                return (await owner.get_job(j_exp)).json()["state"] == "completed"

            await wait_until(_j_exp_done, description="extended job COMPLETED")
        finally:
            await host.disconnect()
            await owner.aclose()


async def test_t5_03_fail_closed_negative_modes_and_protocol_async_and_fs_sensitive_paths(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
) -> None:
    """Fail-closed errors (`target_lost`, `needs_permission`, `watch_limit`, `protocol_async`)."""
    dsn = pg_dsn(clean_watch_db)

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        joined = "\n".join(str(m.content) for m in req.messages)
        if "FS_MISSING" in joined:
            args: dict[str, Any] = {"mode": "fs_quiescence", "doneCriteria": "FS missing alias"}
        elif "FS_TRAVERSAL" in joined:
            args = {
                "mode": "fs_quiescence",
                "workspaceAlias": "../.ssh",
                "doneCriteria": "FS traversal",
            }
        elif "FS_UNKNOWN" in joined:
            args = {
                "mode": "fs_quiescence",
                "workspaceAlias": "unknown_ws",
                "doneCriteria": "FS unknown",
            }
        elif "FS_POLICY_BLOCKED" in joined:
            args = {
                "mode": "fs_quiescence",
                "workspaceAlias": "repo_a",
                "doneCriteria": "FS blocked by host",
            }
        elif "PROTO_ASYNC" in joined:
            args = {"mode": "protocol_async", "doneCriteria": "Wait for MCP task"}
        else:
            args = {
                "mode": "ui_quiescence",
                "quiescenceSeconds": 4,
                "maxWaitSeconds": 120,
                "doneCriteria": "Wait UI",
            }
        return [
            ToolCallRequested(ToolCall(id="w_neg", name="job__wait", arguments=args)),
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
            host_id = await host.connect_enroll(srv.ws_url, owner, "NegMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")
            await host.send_presence(foreground_app_id="editor")
            await host.send_resources(installed_apps=["editor"], workspaces=["repo_a"])

            async def _wait_failed(jid: str) -> dict[str, Any]:
                async def _is_failed() -> bool:
                    return (await owner.get_job(jid)).json()["state"] == "failed"

                await wait_until(_is_failed, description=f"job {jid} FAILED")
                return dict((await owner.get_job(jid)).json())

            # 1) target_lost watchEvent -> FAILED
            j_lost = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="UI target lost",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "Lost", "description": "Wait"}],
                )
            ).json()["jobId"]
            await wait_until(
                lambda: len(host.received_watch_registers) >= 1, description="reg lost"
            )
            await host.send_watch_event(
                watch_id=host.received_watch_registers[-1]["watchId"],
                job_id=j_lost,
                kind="target_lost",
                observed_signals=[],
                elapsed_seconds=5,
            )
            assert (await _wait_failed(j_lost))["stopReason"] == "target_lost"

            # 2) error(needs_permission) watchEvent -> FAILED
            j_perm = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="UI needs permission",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "Perm", "description": "Wait"}],
                )
            ).json()["jobId"]
            await wait_until(
                lambda: len(host.received_watch_registers) >= 2, description="reg perm"
            )
            await host.send_watch_event(
                watch_id=host.received_watch_registers[-1]["watchId"],
                job_id=j_perm,
                kind="error",
                observed_signals=[],
                elapsed_seconds=2,
                error_code="needs_permission",
            )
            assert (await _wait_failed(j_perm))["stopReason"] == "needs_permission"

            # 3) fs_quiescence fail-closed (missing alias, traversal, unknown, host policy_blocked)
            reg_before_fs = len(host.received_watch_registers)
            for tag, expected_reason in (
                ("FS_MISSING", "workspace_not_found"),
                ("FS_TRAVERSAL", "invalid_args"),
                ("FS_UNKNOWN", "workspace_not_found"),
            ):
                jid = (
                    await owner.create_job(
                        pet_id=pet_id,
                        goal=f"{tag} test",
                        host_id=str(host_id),
                        sub_goals=[{"id": "sg1", "title": tag, "description": tag}],
                    )
                ).json()["jobId"]
                assert (await _wait_failed(jid))["stopReason"] == expected_reason
            # None of the invalid fs_quiescence requests should even reach the host
            assert len(host.received_watch_registers) == reg_before_fs

            host.reject_modes["fs_quiescence"] = "policy_blocked"
            j_fs_blk = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="FS_POLICY_BLOCKED test",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "FSBlk", "description": "Wait"}],
                )
            ).json()["jobId"]
            assert (await _wait_failed(j_fs_blk))["stopReason"] == "policy_blocked"
            host.reject_modes.pop("fs_quiescence", None)

            # 4) Wave 1 protocol_async fail-closed:
            # 4a) Missing mcpServerId/protocolTaskId -> watch_unsupported before host RPC
            reg_before_proto = len(host.received_watch_registers)
            j_proto_no_handle = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="PROTO_ASYNC missing handle",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "Proto1", "description": "Wait"}],
                )
            ).json()["jobId"]
            assert (await _wait_failed(j_proto_no_handle))["stopReason"] == "watch_unsupported"
            assert len(host.received_watch_registers) == reg_before_proto

            # 4b) Populated mcpServerId/protocolTaskId in cu_session -> host rejects unsupported
            with psycopg.connect(dsn) as conn:
                owner_uuid = fetch_scalar(
                    conn, "SELECT owner_id FROM hosts WHERE host_id = %s", (host_id,)
                )
            assert owner_uuid is not None
            now = datetime.now(UTC)
            cu_state = ComputerUseSessionState(
                session_id=f"cu_proto_{uuid.uuid4().hex[:8]}",
                owner_id=owner_uuid,
                host_id=HostId(host_id),
                goal="Proto async test",
                allowed_apps=("editor",),
                allowed_domains=(),
                max_steps=20,
                max_minutes=15,
                started_at_utc=now,
                last_heartbeat_at=now,
                target_alias="NegMac",
                last_window={
                    "appId": "editor",
                    "mcpServerId": "build_mcp",
                    "protocolTaskId": "task_01",
                },
            )
            await srv.app.state.cu_session_store.start(cu_state, now)

            j_proto_host_rej = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="PROTO_ASYNC host reject",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "Proto2", "description": "Wait"}],
                )
            ).json()["jobId"]
            assert (await _wait_failed(j_proto_host_rej))["stopReason"] == "watch_unsupported"
            assert len(host.received_watch_registers) == reg_before_proto + 1
            assert host.received_watch_registers[-1]["mode"] == "protocol_async"

            # 5) watch_limit_reached (5th concurrent watch rejected by host max=4)
            active_jobs: list[str] = []
            for idx in range(4):
                jid = (
                    await owner.create_job(
                        pet_id=pet_id,
                        goal=f"Concurrent watch {idx}",
                        host_id=str(host_id),
                        sub_goals=[{"id": "sg1", "title": "Concurrent", "description": "Wait"}],
                    )
                ).json()["jobId"]
                active_jobs.append(jid)
            await wait_until(lambda: len(host.active_watches) == 4, description="4 active watches")

            j_5th = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="Concurrent watch 5th limit",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "Limit5", "description": "Wait"}],
                )
            ).json()["jobId"]
            assert (await _wait_failed(j_5th))["stopReason"] == "watch_limit_reached"

            for jid in active_jobs:
                await owner.cancel_job(jid, reason="cleanup")
        finally:
            await host.disconnect()
            await owner.aclose()


async def test_t5_04_consent_u5_zero_retention_and_sensitive_path_denylist_parity(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
) -> None:
    """Consent U5 zero-retention across DB/API + cross-layer `fs_quiescence` denylist parity."""
    dsn = pg_dsn(clean_watch_db)
    b64_blob = (
        "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODkw" * 3
    )
    img_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    bearer_fixture = "Bearer supersecrettoken98765"
    sk_fixture = "sk-abcdef1234567890secretkey"
    posix_path = "/Users/alice/secret/key.pem"
    win_path = r"C:\Users\alice\doc.txt"

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        joined = "\n".join(str(m.content) for m in req.messages)
        if "[Host watch" not in joined:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="w_u5",
                        name="job__wait",
                        arguments={
                            "mode": "timer",
                            "quiescenceSeconds": 4,
                            "maxWaitSeconds": 60,
                            "doneCriteria": "U5 redaction test",
                        },
                    )
                ),
                done_event(),
            ]
        if '"snapshotId"' not in joined:
            return [
                ToolCallRequested(ToolCall(id="obs_u5", name="ui__observe_hybrid", arguments={})),
                done_event(),
            ]
        return [
            TextDelta(
                f"Summary with {img_url} {b64_blob} {bearer_fixture} {sk_fixture} "
                f"{posix_path} {win_path}"
            ),
            done_event(),
        ]

    async def _u5_tool_handler(
        tool_name: str, _args: dict[str, Any], _meta: dict[str, Any]
    ) -> dict[str, Any]:
        if tool_name == "ui.observe_hybrid":
            return {
                "status": "ok",
                "structuredContent": {
                    "appId": "editor",
                    "pid": 48210,
                    "windowId": 1042,
                    "snapshotId": "snap_u5",
                    "nodes": [],
                    "screenshot": {"dataBase64": b64_blob, "mimeType": "image/jpeg"},
                },
            }
        return {"status": "ok", "structuredContent": {"ok": True}}

    llm = ScriptedTextGenerator(responder=_respond)
    async with watch_server_factory(generator=llm) as srv:
        owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer
        )
        host = SimulatedWatchXtpHost(tool_handler=_u5_tool_handler)
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            host_id = await host.connect_enroll(srv.ws_url, owner, "U5Mac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")

            job_id = (
                await owner.create_job(
                    pet_id=pet_id,
                    goal="Verify U5 zero-retention",
                    host_id=str(host_id),
                    sub_goals=[{"id": "sg1", "title": "U5", "description": "Observe & summarize"}],
                )
            ).json()["jobId"]

            await wait_until(lambda: len(host.received_watch_registers) >= 1, description="U5 reg")
            await host.send_watch_event(
                watch_id=host.received_watch_registers[0]["watchId"],
                job_id=job_id,
                kind="settled",
                observed_signals=["timer"],
                elapsed_seconds=5,
            )

            async def _done() -> bool:
                return (await owner.get_job(job_id)).json()["state"] == "completed"

            await wait_until(_done, description="U5 job COMPLETED")

            # Verify ephemeral vision image was delivered in-memory to LLM, NOT persisted in DB
            assert any(m.image_base64 == b64_blob for req in llm.requests for m in req.messages)

            async def _u5_inbox_item() -> dict[str, Any] | None:
                inbox_items = (await owner.list_wake_inbox()).json()["items"]
                return next((i for i in inbox_items if i["jobId"] == job_id), None)

            item = await wait_until(_u5_inbox_item, description="U5 wake_inbox_item")
            summary_str = item["summary"]
            assert "[IMAGE_REDACTED]" in summary_str
            assert "[BASE64_REDACTED]" in summary_str
            assert "[REDACTED]" in summary_str
            assert "[PATH]" in summary_str

            forbidden_literals = (
                b64_blob,
                "iVBORw0KGgo",
                "supersecrettoken98765",
                "sk-abcdef1234567890secretkey",
                posix_path,
                win_path,
            )
            with psycopg.connect(dsn) as conn:
                cp_row = fetch_scalar(
                    conn,
                    "SELECT checkpoint_data::text FROM autonomous_jobs WHERE job_id = %s",
                    (uuid.UUID(job_id),),
                )
                sg_row = fetch_scalar(
                    conn,
                    "SELECT sub_goals::text FROM autonomous_jobs WHERE job_id = %s",
                    (uuid.UUID(job_id),),
                )
                watch_row = fetch_scalar(
                    conn,
                    "SELECT target_spec::text FROM job_watches WHERE job_id = %s",
                    (uuid.UUID(job_id),),
                )
                inbox_row = fetch_scalar(
                    conn,
                    "SELECT title || ' ' || summary FROM wake_inbox_items WHERE job_id = %s",
                    (uuid.UUID(job_id),),
                )
                audit_rows = conn.execute(
                    "SELECT metadata::text FROM audit_events ORDER BY occurred_at DESC LIMIT 20"
                ).fetchall()

            haystack = "\n".join(
                [str(cp_row or ""), str(sg_row or ""), str(watch_row or ""), str(inbox_row or "")]
                + [str(r[0]) for r in audit_rows]
                + [json.dumps(item)]
            )
            for forbidden in forbidden_literals:
                assert forbidden not in haystack, f"U5 leak detected: {forbidden!r}"
        finally:
            await host.disconnect()
            await owner.aclose()

    # Cross-layer sensitive path denylist parity (Dart & macOS Swift)
    dart_driver = (
        REPO_ROOT / "desktop" / "lib" / "features" / "watch" / "native_watch_signal_driver.dart"
    ).read_text(encoding="utf-8")
    swift_plugin = (
        REPO_ROOT / "desktop" / "macos" / "Runner" / "XunaHostPlugin+Watch.swift"
    ).read_text(encoding="utf-8")
    for token in (".ssh", ".aws", ".gnupg", ".kube", "library", ".env", ".git"):
        assert token in dart_driver.lower(), f"Missing {token!r} in Dart denylist"
        assert token in swift_plugin.lower(), f"Missing {token!r} in Swift denylist"
    assert "objects" in dart_driver.lower() and ".git/objects" in swift_plugin.lower()
