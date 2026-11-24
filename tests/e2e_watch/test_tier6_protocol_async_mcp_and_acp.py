"""Tier 6 Watch & Wake E2E: Wave 2 `protocol_async` MCP, ACP, Autopilot/R3 & E-Stop."""

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
    make_watch_tools,
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


def _tc(cid: str, name: str, **args: Any) -> list[Any]:
    return [ToolCallRequested(ToolCall(id=cid, name=name, arguments=args)), done_event()]


def _ok(sc: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "structuredContent": sc}


def _mcp_handler(
    *, task_id: str = "task_cp_1", server_override: str | None = None, summary: str = "Checks ok."
) -> Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]:
    def _h(tool: str, args: dict[str, Any], _meta: dict[str, Any]) -> dict[str, Any]:
        if tool == "mcp.call":
            srv = server_override or str(args["server"])
            return _ok(
                {
                    "server": args["server"],
                    "tool": args["tool"],
                    "summary": "Detached.",
                    "status": "running",
                    "mcpServerId": srv,
                    "protocolTaskId": task_id,
                }
            )
        if tool == "mcp.task_result":
            return _ok(
                {
                    "server": args["server"],
                    "protocolTaskId": args["protocolTaskId"],
                    "status": "completed",
                    "progressPercent": 100,
                    "summary": summary,
                }
            )
        if tool == "app.launch":
            return _ok({"launched": True})
        return _ok({"sent": True}) if tool == "mail.send" else _ok({"ok": True})

    return _h


def _host(handler: Callable[..., Any] | None = None) -> SimulatedWatchXtpHost:
    h = SimulatedWatchXtpHost(
        tools=make_watch_tools(include_mcp_async=True), reject_modes={}, tool_handler=handler
    )
    h.trust_preset = "autopilot"
    return h


async def _enroll(
    srv: Any, host: SimulatedWatchXtpHost, alias: str, *, mobile: bool = False
) -> tuple[SimulatedWatchOwnerClient, str, uuid.UUID]:
    owner = SimulatedWatchOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
    pet_id = await owner.bind_pet()
    await owner.enable_consents()
    if mobile:
        await owner.self_enroll_mobile()
    host_id = await host.connect_enroll(srv.ws_url, owner, alias)
    await owner.grant_all_standard(str(host_id), max_risk="R3")
    return owner, pet_id, host_id


async def _job(
    owner: SimulatedWatchOwnerClient,
    pet_id: str,
    host_id: uuid.UUID,
    goal: str,
    sub_goals: list[dict[str, Any]],
    *,
    max_steps: int = 8,
) -> str:
    r = await owner.create_job(
        pet_id=pet_id, goal=goal, host_id=str(host_id), max_steps=max_steps, sub_goals=sub_goals
    )
    return str(r.json()["jobId"])


async def _wait_state(
    owner: SimulatedWatchOwnerClient, job_id: str, target_state: str
) -> dict[str, Any]:
    async def _match() -> dict[str, Any] | None:
        j = (await owner.get_job(job_id)).json()
        return dict(j) if j["state"] == target_state else None

    return await wait_until(_match, timeout_s=5.0, description=f"job {job_id} -> {target_state}")


async def _push_proto(
    host: SimulatedWatchXtpHost, watch_id: str, job_id: str, kind: str = "settled", elapsed: int = 4
) -> None:
    sigs = ["protocol_async"] if kind in ("settled", "changed") else []
    await host.send_watch_event(
        watch_id=watch_id, job_id=job_id, kind=kind, observed_signals=sigs, elapsed_seconds=elapsed
    )


async def test_t6_01_mcp_async_autobind_override_zero_sleep_tokens_h1_and_u5(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
) -> None:
    """MCP detach -> `last_window` auto-bind & explicit override -> `mcp__task_result` + U5."""
    dsn = pg_dsn(clean_watch_db)
    bearer_fx = "Bearer topsecret_wave2_fx_98765"
    posix_fx = "/Users/alice/private/build.key"

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        joined = "\n".join(str(m.content) for m in req.messages)
        ovr = "OVERRIDE_CASE" in joined
        srv_id, tid = ("override_mcp", "task_override_2") if ovr else ("build_mcp", "task_cp_1")
        if "[Host watch 'settled']" in joined:
            if '"progressPercent"' not in joined:
                assert "mcp__task_result" in joined
                return _tc(f"r_{tid}", "mcp__task_result", server=srv_id, protocolTaskId=tid)
            return [TextDelta(f"Verified {srv_id} ({tid}): {bearer_fx} {posix_fx}"), done_event()]
        if "protocolTaskId" not in joined:
            return _tc(
                f"c_{tid}", "mcp__call", server="build_mcp", tool="run_suite", asyncMode=True
            )
        w: dict[str, Any] = {
            "mode": "protocol_async",
            "doneCriteria": "Suite completes",
            "continuationMode": "report_only" if ovr else "report_and_ask",
        }
        if ovr:
            w.update({"mcpServerId": "override_mcp", "protocolTaskId": "task_override_2"})
        return _tc(f"w_{tid}", "job__wait", **w)

    handler = _mcp_handler(summary=f"Build ok; artifact at {posix_fx}; {bearer_fx}")
    llm = ScriptedTextGenerator(responder=_respond)
    async with watch_server_factory(generator=llm) as srv:
        host = _host(handler)
        owner, pet_id, host_id = await _enroll(srv, host, "ProtoMac")
        voice_ws = await owner.connect_voice_ws(pet_id, target_host_id=str(host_id))
        try:
            # Subcase A: auto-bind from checkpoint_data["last_window"] (job__wait omits handle)
            sg = [{"id": "sg1", "title": "Sub-Goal 1", "description": "Suite"}]
            job_a = await _job(owner, pet_id, host_id, "Run MCP suite", sg)
            await wait_until(lambda: len(host.received_watch_registers) >= 1, description="reg A")
            reg_a = host.received_watch_registers[0]
            assert reg_a["mode"] == "protocol_async" and reg_a["requireInitialChange"] is False
            assert reg_a["target"] == {"mcpServerId": "build_mcp", "protocolTaskId": "task_cp_1"}

            with psycopg.connect(dsn) as conn:
                row = conn.execute(
                    "SELECT j.state, j.checkpoint_data->'last_window', w.status "
                    "FROM autonomous_jobs j JOIN job_watches w ON w.job_id = j.job_id "
                    "WHERE j.job_id = %s",
                    (uuid.UUID(job_a),),
                ).fetchone()
            assert row == (
                "waiting_watch",
                {"mcpServerId": "build_mcp", "protocolTaskId": "task_cp_1"},
                "active",
            )

            sleep_reqs = len(llm.requests)
            await asyncio.sleep(0.25)
            assert len(llm.requests) == sleep_reqs

            await _push_proto(host, reg_a["watchId"], job_a)
            wake_a, _ = await voice_ws.recv_until_wake_inbox(wait_s=5.0)
            assert wake_a["jobId"] == job_a and wake_a["kind"] == "job_report"
            assert (await owner.get_job(job_a)).json()["state"] == "completed"
            assert any(c["name"] == "mcp.task_result" for c in host.received_tool_calls)
            assert "[REDACTED]" in wake_a["summary"] and "[PATH]" in wake_a["summary"]
            assert "topsecret_wave2_fx_98765" not in wake_a["summary"]
            assert posix_fx not in wake_a["summary"]
            with psycopg.connect(dsn) as conn:
                db_sum = fetch_scalar(
                    conn,
                    "SELECT summary FROM wake_inbox_items WHERE job_id = %s",
                    (uuid.UUID(job_a),),
                )
            assert isinstance(db_sum, str)
            assert "topsecret_wave2_fx_98765" not in db_sum and posix_fx not in db_sum

            # Subcase B: explicit job__wait args override checkpoint_data["last_window"]
            sg_b = [{"id": "sg1", "title": "Sub-Goal 1", "description": "OVERRIDE_CASE"}]
            job_b = await _job(owner, pet_id, host_id, "OVERRIDE_CASE", sg_b)
            await wait_until(lambda: len(host.received_watch_registers) >= 2, description="reg B")
            reg_b = host.received_watch_registers[1]
            assert reg_b["target"] == {
                "mcpServerId": "override_mcp",
                "protocolTaskId": "task_override_2",
            }
            await _push_proto(host, reg_b["watchId"], job_b, elapsed=3)
            await _wait_state(owner, job_b, "completed")
            assert any(
                c["name"] == "mcp.task_result"
                and c["arguments"]["protocolTaskId"] == "task_override_2"
                for c in host.received_tool_calls
            )
        finally:
            await voice_ws.close()
            await host.disconnect()
            await owner.aclose()


async def test_t6_02_autopilot_protocol_async_auto_continue_r2_then_r3_gate(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-wake `mcp.task_result` (R1) + `app.launch` (R2) auto-run; `mail.send` (R3) gates."""
    dsn = pg_dsn(clean_watch_db)
    monkeypatch.setattr(orch_mod, "APPROVAL_TTL_SECONDS", 10.0)
    mail_args = {"to": ["lead@example.com"], "subject": "Wave 2 Ready", "body": "Verified."}

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        joined = "\n".join(str(m.content) for m in req.messages)
        if "[Active Sub-Goal 1/2]" in joined:
            if "[Host watch" not in joined:
                return _tc(
                    "w_auto",
                    "job__wait",
                    mode="protocol_async",
                    mcpServerId="build_mcp",
                    protocolTaskId="task_auto_1",
                    doneCriteria="Build finished",
                    continuationMode="auto_continue",
                )
            if '"completed"' not in joined:
                return _tc(
                    "tr", "mcp__task_result", server="build_mcp", protocolTaskId="task_auto_1"
                )
            return [TextDelta("Verified, advancing."), done_event()]
        if '"launched"' not in joined:
            return _tc("l_r2", "app__launch", id="mail_client")
        if '"sent"' not in joined:
            return _tc("m_r3", "mail__send", **mail_args)
        return [TextDelta("Email dispatched."), done_event()]

    llm = ScriptedTextGenerator(responder=_respond)
    async with watch_server_factory(generator=llm) as srv:
        host = _host(_mcp_handler())
        owner, pet_id, host_id = await _enroll(srv, host, "ProtoAutoMac", mobile=True)
        try:
            sgs = [
                {"id": "sg1", "title": "Wait", "description": "Wait build"},
                {"id": "sg2", "title": "Notify", "description": "Launch & email"},
            ]
            job_id = await _job(owner, pet_id, host_id, "Auto R2 then R3", sgs, max_steps=12)
            await wait_until(lambda: len(host.received_watch_registers) >= 1, description="reg")
            await _push_proto(host, host.received_watch_registers[0]["watchId"], job_id, elapsed=5)
            r3_job = await _wait_state(owner, job_id, "awaiting_r3_checkpoint")
            approval_id = str(r3_job["pendingApprovalId"])
            called = [c["name"] for c in host.received_tool_calls]
            assert "mcp.task_result" in called and "app.launch" in called
            assert "mail.send" not in called

            with psycopg.connect(dsn) as conn:
                row = conn.execute(
                    "SELECT host_id, args_sha256, expires_at FROM action_approvals "
                    "WHERE approval_id = %s",
                    (uuid.UUID(approval_id),),
                ).fetchone()
            assert row is not None and str(row[1]) == args_sha256(mail_args)
            approve = await owner.decide_approval(
                approval_id,
                "approve",
                host_id=str(row[0]),
                args_sha256_hex=str(row[1]),
                expires_at=row[2].astimezone(UTC).isoformat(),
                sign_r3=True,
            )
            assert approve.status_code == 200, approve.text
            await _wait_state(owner, job_id, "completed")
            assert any(c["name"] == "mail.send" for c in host.received_tool_calls)
        finally:
            await host.disconnect()
            await owner.aclose()


async def test_t6_03_acp_harness_detach_and_target_lost_fail_closed(
    watch_server_factory: Callable[..., Any],
) -> None:
    """ACP harness async handle + `target_lost` watchEvent -> `FAILED` (`target_lost`)."""

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        joined = "\n".join(str(m.content) for m in req.messages)
        if "protocolTaskId" not in joined:
            return _tc(
                "acp_1",
                "mcp__call",
                server="acp_agent",
                tool="solve",
                payloadJson='{"prompt":"Run harness"}',
                asyncMode=True,
            )
        return _tc("w_acp", "job__wait", mode="protocol_async", doneCriteria="ACP completes")

    llm = ScriptedTextGenerator(responder=_respond)
    async with watch_server_factory(generator=llm) as srv:
        host = _host(_mcp_handler(task_id="acp_task_7", server_override="acp_agent"))
        owner, pet_id, host_id = await _enroll(srv, host, "AcpMac")
        try:
            sg = [{"id": "sg1", "title": "ACP", "description": "Run ACP"}]
            job_id = await _job(owner, pet_id, host_id, "Run ACP harness async", sg)
            await wait_until(lambda: len(host.received_watch_registers) >= 1, description="ACP reg")
            reg = host.received_watch_registers[0]
            assert reg["target"] == {"mcpServerId": "acp_agent", "protocolTaskId": "acp_task_7"}
            await _push_proto(host, reg["watchId"], job_id, kind="target_lost", elapsed=3)
            failed = await _wait_state(owner, job_id, "failed")
            assert failed["stopReason"] == "target_lost"
            inbox = (await owner.list_wake_inbox()).json()["items"]
            assert next(i for i in inbox if i["jobId"] == job_id)["kind"] == "job_failed"
        finally:
            await host.disconnect()
            await owner.aclose()


async def test_t6_04_negative_fail_closed_and_estop_protocol_async(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
) -> None:
    """Missing `protocolTaskId`, post-wake hallucination (`no_tool_executed`), and E-Stop."""
    dsn = pg_dsn(clean_watch_db)

    def _respond(req: ChatRequest, _idx: int) -> list[Any]:
        joined = "\n".join(str(m.content) for m in req.messages)
        if "MISSING_TASK" in joined:
            return _tc(
                "w_miss",
                "job__wait",
                mode="protocol_async",
                mcpServerId="build_mcp",
                doneCriteria="Missing task id",
            )
        if "HALLUCINATE" in joined:
            if "[Host watch 'settled']" in joined:
                return [TextDelta("Claiming done without mcp.task_result!"), done_event()]
            return _tc(
                "w_hal",
                "job__wait",
                mode="protocol_async",
                mcpServerId="build_mcp",
                protocolTaskId="task_hal_1",
                doneCriteria="Verify via mcp.task_result",
            )
        return _tc(
            "w_estop",
            "job__wait",
            mode="protocol_async",
            mcpServerId="build_mcp",
            protocolTaskId="task_estop_1",
            doneCriteria="Stopped by E-Stop",
        )

    llm = ScriptedTextGenerator(responder=_respond)
    async with watch_server_factory(generator=llm) as srv:
        host = _host()
        owner, pet_id, host_id = await _enroll(srv, host, "NegProtoMac")
        try:
            # 1) Missing protocolTaskId fails closed before watchRegister reaches host
            j_miss = await _job(
                owner,
                pet_id,
                host_id,
                "MISSING_TASK",
                [{"id": "s", "title": "M", "description": "M"}],
            )
            assert (await _wait_state(owner, j_miss, "failed"))["stopReason"] == "watch_unsupported"
            assert host.received_watch_registers == []

            # 2) Post-wake hallucination without mcp__task_result -> no_tool_executed
            j_hal = await _job(
                owner,
                pet_id,
                host_id,
                "HALLUCINATE",
                [{"id": "s", "title": "H", "description": "H"}],
            )
            await wait_until(lambda: len(host.received_watch_registers) >= 1, description="hal reg")
            await _push_proto(host, host.received_watch_registers[-1]["watchId"], j_hal, elapsed=3)
            assert (await _wait_state(owner, j_hal, "failed"))["stopReason"] == "no_tool_executed"
            assert not any(c["name"] == "mcp.task_result" for c in host.received_tool_calls)

            # 3) HTTP POST /v1/actions/stop cancels active watch & ignores late watchEvent
            j_stop = await _job(
                owner,
                pet_id,
                host_id,
                "ESTOP_PROTO",
                [{"id": "s", "title": "S", "description": "S"}],
            )
            await wait_until(
                lambda: len(host.received_watch_registers) >= 2, description="stop reg"
            )
            w_stop = host.received_watch_registers[-1]["watchId"]
            assert w_stop in host.active_watches

            stop_resp = await owner.emergency_stop(reason="owner_app")
            assert stop_resp.status_code == 200, stop_resp.text
            await wait_until(lambda: len(host.received_watch_cancels) >= 1, description="cancel")
            assert host.received_watch_cancels[-1] == {"schemaVersion": 1, "reason": "estop"}
            assert host.active_watches == {}
            assert (await owner.get_job(j_stop)).json()["state"] == "cancelled"
            with psycopg.connect(dsn) as conn:
                w_st = fetch_scalar(
                    conn, "SELECT status FROM job_watches WHERE watch_id = %s", (uuid.UUID(w_stop),)
                )
            assert w_st == "cancelled"

            await _push_proto(host, w_stop, j_stop, elapsed=10)
            await asyncio.sleep(0.25)
            assert (await owner.get_job(j_stop)).json()["state"] == "cancelled"
        finally:
            await host.disconnect()
            await owner.aclose()
