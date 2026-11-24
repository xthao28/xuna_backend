"""Unit tests for AW2-M3 backend integration of `protocol_async` watch mode."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.modules.actions.test_tool_discovery import _descriptor, _grants, _host
from tests.unit.test_async_watch_coordinator_and_runner import _FakeHub

from xuna_backend.modules.actions.application.catalog import (
    _PRIORITY_ORDER,
    CAP_HOST_ACTIONS_V1,
    MAX_CATALOG_TOOLS,
    build_tool_catalog,
)
from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    InMemoryComputerUseSessionStore,
)
from xuna_backend.modules.actions.application.planner_executor import JobPlannerExecutor
from xuna_backend.modules.actions.application.tool_discovery import TOOL_SEARCH_FUNCTION
from xuna_backend.modules.actions.application.watch_coordinator import WatchCoordinator
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    SubGoal,
    SubGoalStatus,
)
from xuna_backend.modules.actions.domain.schema_validator import (
    ArgumentValidationError,
    SchemaValidator,
)
from xuna_backend.modules.actions.domain.watch import (
    JOB_WAIT_SCHEMA,
    WatchCancelReason,
    WatchMode,
    WatchStatus,
)
from xuna_backend.modules.actions.infrastructure.job_store import InMemoryAutonomousJobStore
from xuna_backend.modules.actions.infrastructure.watch_repository import InMemoryJobWatchRepository
from xuna_backend.modules.cognition.domain.types import ToolCall
from xuna_backend.modules.hosts.application.xtp_frames import WatchEventParamsModel
from xuna_backend.modules.hosts.domain.entities import HostId, Manifest
from xuna_backend.modules.hosts.domain.twin import HostPresence

NOW = datetime(2026, 11, 23, 14, 0, 0, tzinfo=UTC)


def _make_job(
    owner_id: uuid.UUID,
    host_id: HostId,
    session_id: str | None = None,
    checkpoint_data: dict[str, Any] | None = None,
) -> AutonomousJob:
    return AutonomousJob(
        job_id=uuid.uuid4(),
        owner_id=owner_id,
        pet_id="pet_w2",
        host_id=host_id,
        session_id=session_id,
        title="Protocol async job",
        goal="Run async MCP task and verify result",
        state=AutonomousJobState.RUNNING,
        sub_goals=(
            SubGoal("sg-1", "Run task", "Run call", SubGoalStatus.IN_PROGRESS, 1, started_at=NOW),
            SubGoal("sg-2", "Verify", "Inspect result", SubGoalStatus.PENDING),
        ),
        active_sub_goal_index=0,
        checkpoint_data=checkpoint_data,
        created_at=NOW,
        updated_at=NOW,
        started_at=NOW,
    )


def _make_cu_state(
    owner_id: uuid.UUID,
    host_id: HostId,
    last_window: dict[str, str],
    allowed_apps: tuple[str, ...] = (),
) -> ComputerUseSessionState:
    return ComputerUseSessionState(
        session_id=f"cu_{uuid.uuid4().hex[:8]}",
        owner_id=owner_id,
        host_id=host_id,
        goal="Async protocol session",
        allowed_apps=allowed_apps,
        allowed_domains=(),
        max_steps=10,
        max_minutes=15,
        started_at_utc=NOW,
        last_heartbeat_at=NOW,
        last_window=last_window,
    )


def test_job_wait_schema_accepts_and_validates_protocol_async_fields() -> None:
    SchemaValidator.validate_schema_definition(JOB_WAIT_SCHEMA)
    clean_with, target = SchemaValidator.validate_arguments(
        JOB_WAIT_SCHEMA,
        {
            "mode": "protocol_async",
            "doneCriteria": "Task reaches completed state",
            "mcpServerId": "build_mcp.v1",
            "protocolTaskId": "task-01_abc",
        },
        allow_target=False,
    )
    assert target is None
    assert clean_with["mcpServerId"] == "build_mcp.v1"
    assert clean_with["protocolTaskId"] == "task-01_abc"

    clean_without, _ = SchemaValidator.validate_arguments(
        JOB_WAIT_SCHEMA,
        {"mode": "protocol_async", "doneCriteria": "Task finishes"},
        allow_target=False,
    )
    assert "mcpServerId" not in clean_without
    assert "protocolTaskId" not in clean_without

    for bad_args in (
        {"mode": "protocol_async", "doneCriteria": "x", "mcpServerId": "INVALID"},
        {"mode": "protocol_async", "doneCriteria": "x", "mcpServerId": "a/b"},
        {"mode": "protocol_async", "doneCriteria": "x", "mcpServerId": ""},
        {"mode": "protocol_async", "doneCriteria": "x", "mcpServerId": "a" * 65},
        {"mode": "protocol_async", "doneCriteria": "x", "protocolTaskId": ""},
        {"mode": "protocol_async", "doneCriteria": "x", "protocolTaskId": "t" * 129},
        {"mode": "protocol_async", "doneCriteria": "x", "extraKey": 1},
    ):
        with pytest.raises(ArgumentValidationError):
            SchemaValidator.validate_arguments(JOB_WAIT_SCHEMA, bad_args, allow_target=False)


def test_bind_target_protocol_async_priority_isolation_and_fail_closed() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        hub = _FakeHub(host_id, owner_id)
        hub.twin = hub.twin.__class__(
            host_id=str(host_id),
            owner_id=owner_id,
            presence=HostPresence(0, False, False, NOW, foreground_app_id="1password"),
        )
        cu_store = InMemoryComputerUseSessionStore()
        cu_state = _make_cu_state(
            owner_id,
            host_id,
            {"appId": "1password", "mcpServerId": "cu_srv", "protocolTaskId": "cu_task"},
            allowed_apps=("1password",),
        )
        await cu_store.start(cu_state, NOW)

        coord = WatchCoordinator(
            watch_repo=InMemoryJobWatchRepository(),
            job_store=InMemoryAutonomousJobStore(),
            hub=hub,  # type: ignore[arg-type]
            cu_session_store=cu_store,
            clock_utc=lambda: NOW,
        )

        sid = cu_state.session_id

        async def _bind(c: WatchCoordinator, **kw: Any):
            return await c.bind_target(
                owner_id=owner_id,
                host_id=host_id,
                mode=WatchMode.PROTOCOL_ASYNC,
                signals=(),
                **kw,
            )

        # UI watch on the same hard-denied foreground app IS blocked, while protocol_async is not
        t_ui, err_ui = await coord.bind_target(
            owner_id=owner_id,
            host_id=host_id,
            mode=WatchMode.UI_QUIESCENCE,
            signals=(),
            session_id=sid,
        )
        assert t_ui is None and err_ui == "policy_blocked"

        # 1. Explicit args override checkpoint and active_cu
        fb_lw = {"mcpServerId": "cp_srv", "protocolTaskId": "cp_task"}
        t1, err1 = await _bind(
            coord,
            session_id=sid,
            mcp_server_id="arg_srv",
            protocol_task_id="arg_task",
            fallback_last_window=fb_lw,
        )
        assert err1 is None and t1 is not None
        assert (t1.mcp_server_id, t1.protocol_task_id) == ("arg_srv", "arg_task")

        # 2. Checkpoint fallback works without CU session
        coord_no_cu = WatchCoordinator(
            watch_repo=InMemoryJobWatchRepository(),
            job_store=InMemoryAutonomousJobStore(),
            hub=hub,  # type: ignore[arg-type]
            clock_utc=lambda: NOW,
        )
        t2, err2 = await _bind(coord_no_cu, fallback_last_window=fb_lw)
        assert err2 is None and t2 is not None
        assert (t2.mcp_server_id, t2.protocol_task_id) == ("cp_srv", "cp_task")

        # 3. Missing either ID (incl. partial tier with lower tiers populated) -> watch_unsupported
        for partial_kw in (
            {"mcp_server_id": "only_srv"},
            {"protocol_task_id": "only_task"},
            {"session_id": sid, "mcp_server_id": "srv_b", "fallback_last_window": fb_lw},
            {"session_id": sid, "protocol_task_id": "task_b", "fallback_last_window": fb_lw},
            {"session_id": sid, "fallback_last_window": {"mcpServerId": "cp_srv"}},
            {"session_id": sid, "fallback_last_window": {"protocolTaskId": "cp_task"}},
        ):
            tm, errm = await _bind(
                coord if "session_id" in partial_kw else coord_no_cu, **partial_kw
            )
            assert tm is None and errm == "watch_unsupported"

        # 4. Invalid formats fail closed with invalid_args (before missing-half check)
        for bad_kw in (
            {"mcp_server_id": "BadCaps", "protocol_task_id": "ok_task"},
            {"mcp_server_id": "srv\n", "protocol_task_id": "ok_task"},
            {"mcp_server_id": "BadCaps"},
            {"mcp_server_id": "ok_srv", "protocol_task_id": ""},
            {"mcp_server_id": "ok_srv", "protocol_task_id": "   "},
            {"mcp_server_id": "ok_srv", "protocol_task_id": "bad\x00task"},
            {"mcp_server_id": "ok_srv", "protocol_task_id": "x" * 129},
            {"protocol_task_id": "bad\x00task", "fallback_last_window": fb_lw},
            {"protocol_task_id": "   "},
            {"protocol_task_id": "x" * 129},
        ):
            tb, errb = await _bind(coord_no_cu, **bad_kw)
            assert tb is None and errb == "invalid_args"

        # 5. infer_watch_for_detach picks protocol_async when active_cu.last_window has both IDs
        d_mode, d_sigs, d_target, d_err = await coord.infer_watch_for_detach(
            owner_id=owner_id,
            host_id=host_id,
            session_id=cu_state.session_id,
        )
        assert d_err is None and d_mode == WatchMode.PROTOCOL_ASYNC and d_sigs == ()
        assert d_target is not None
        assert (d_target.mcp_server_id, d_target.protocol_task_id) == ("cu_srv", "cu_task")

    asyncio.run(_run())


def test_register_wake_prompt_and_cancel_for_protocol_async() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        hub = _FakeHub(host_id, owner_id)
        watch_repo = InMemoryJobWatchRepository()
        job_store = InMemoryAutonomousJobStore()
        coord = WatchCoordinator(
            watch_repo=watch_repo,
            job_store=job_store,
            hub=hub,  # type: ignore[arg-type]
            clock_utc=lambda: NOW,
        )

        async def _reg_wait(j: AutonomousJob, args: dict[str, object], call_id: str):
            return await coord.register_from_job_wait(
                job=j,
                clean_args=args,
                sub_goals=j.sub_goals,
                active_sub_goal_index=0,
                active_turns=[],
                compacted_history=[],
                assistant_call=ToolCall(id=call_id, name="job__wait", arguments=args),
                assistant_text="",
                steering_notes=[],
                now=NOW,
            )

        lw_cp = {"mcpServerId": "build_mcp", "protocolTaskId": "task_cp"}
        job = _make_job(owner_id, host_id, checkpoint_data={"last_window": lw_cp})
        await job_store.save(job)
        for bad_args in (
            {
                "mode": "protocol_async",
                "doneCriteria": "Bad",
                "mcpServerId": 123,
                "protocolTaskId": "t",
            },
            {"mode": "protocol_async", "doneCriteria": "Bad", "protocolTaskId": True},
        ):
            bad_reg = await _reg_wait(job, bad_args, "call_bad")
            assert not bad_reg.accepted and bad_reg.error_code == "invalid_args"

        reg = await _reg_wait(
            job, {"mode": "protocol_async", "doneCriteria": "Verify build outcome"}, "call_w1"
        )
        assert reg.accepted is True and reg.watch is not None
        assert (reg.watch.target.mcp_server_id, reg.watch.target.protocol_task_id) == (
            "build_mcp",
            "task_cp",
        )
        reg_frames = [p for _, m, p in hub.sent if m == "xuna/watchRegister"]
        assert len(reg_frames) == 1 and reg_frames[0]["requireInitialChange"] is False
        assert reg_frames[0]["target"] == lw_cp

        # Wake event produces H1 prompt instructing mcp__task_result with exact prefix
        await coord.on_watch_event(
            owner_id,
            host_id,
            WatchEventParamsModel(
                schemaVersion=1,
                watchId=str(reg.watch.watch_id),
                jobId=str(job.job_id),
                kind="settled",
                observedSignals=["protocol_async"],
                elapsedSeconds=12,
                changeCount=2,
                digestChanged=True,
                occurredAt=NOW.isoformat(),
            ),
        )
        woken = await job_store.get(job.job_id)
        assert woken is not None and woken.checkpoint_data is not None
        wake_msg = str(woken.checkpoint_data["active_turns"][-1]["content"])
        assert wake_msg.startswith("[Host watch 'settled']")
        assert "mcp__task_result" in wake_msg and "build_mcp" in wake_msg and "task_cp" in wake_msg

        # Cancel watch for job + E-Stop cancel_all_for_owner
        running_again = woken.with_state(AutonomousJobState.RUNNING, now=NOW)
        await job_store.save(running_again)
        args_2: dict[str, object] = {
            "mode": "protocol_async",
            "doneCriteria": "Wait second round",
            "mcpServerId": "acp_harness",
            "protocolTaskId": "task_override",
        }
        reg2 = await _reg_wait(running_again, args_2, "call_w2")
        assert reg2.accepted and reg2.watch is not None
        assert (
            await coord.cancel_watches_for_job(job.job_id, reason=WatchCancelReason.OWNER_CANCEL)
            == 1
        )
        assert [p for _, m, p in hub.sent if m == "xuna/watchCancel"][-1]["watchId"] == str(
            reg2.watch.watch_id
        )

        job_estop = _make_job(owner_id, host_id)
        await job_store.save(job_estop)
        args_3: dict[str, object] = {
            "mode": "protocol_async",
            "doneCriteria": "E-Stop check",
            "mcpServerId": "acp_harness",
            "protocolTaskId": "task_estop",
        }
        reg3 = await _reg_wait(job_estop, args_3, "call_w3")
        assert reg3.accepted and reg3.watch is not None
        assert await coord.cancel_all_for_owner(owner_id, reason="estop") >= 1
        w3_after = await watch_repo.get(reg3.watch.watch_id)
        assert w3_after is not None and w3_after.status == WatchStatus.CANCELLED
        assert [p for _, m, p in hub.sent if m == "xuna/watchCancel"][-1]["reason"] == "estop"

    asyncio.run(_run())


def test_planner_executor_persists_and_carries_forward_last_window() -> None:
    planner = JobPlannerExecutor(clock_utc=lambda: NOW)
    owner_id, host_id = uuid.uuid4(), HostId.generate()
    job = _make_job(owner_id=owner_id, host_id=host_id, checkpoint_data=None)

    # Non-running status does NOT fabricate last_window
    out_done = planner.process_tool_result(
        job=job,
        tool_name="mcp.call",
        tool_result={
            "status": "ok",
            "output": {"status": "completed", "mcpServerId": "s1", "protocolTaskId": "t1"},
        },
        now=NOW,
    )
    assert out_done.checkpoint_data is not None and "last_window" not in out_done.checkpoint_data

    # status=="running" persists last_window and preserves across subsequent steps
    out1 = planner.process_tool_result(
        job=job,
        tool_name="mcp.call",
        tool_result={
            "status": "ok",
            "output": {"status": "running", "mcpServerId": "build_mcp", "protocolTaskId": "t99"},
        },
        mark_sub_goal_complete=False,
        now=NOW,
    )
    assert out1.kind == "progress" and out1.checkpoint_data is not None
    expected_lw = {"mcpServerId": "build_mcp", "protocolTaskId": "t99"}
    assert out1.checkpoint_data["last_window"] == expected_lw

    job2 = _make_job(owner_id=owner_id, host_id=host_id, checkpoint_data=out1.checkpoint_data)
    out2 = planner.process_tool_result(
        job=job2,
        tool_name="host.status",
        tool_result={"status": "ok", "output": {"healthy": True}},
        mark_sub_goal_complete=False,
        now=NOW,
    )
    assert out2.kind == "progress" and out2.checkpoint_data is not None
    assert out2.checkpoint_data["last_window"] == expected_lw

    job3 = _make_job(owner_id=owner_id, host_id=host_id, checkpoint_data=out2.checkpoint_data)
    out3 = planner.complete_sub_goal_after_verified_tools(job3, now=NOW)
    assert out3.kind == "progress" and out3.checkpoint_data is not None
    assert out3.checkpoint_data["last_window"] == expected_lw


def test_catalog_exposes_mcp_task_tools_at_slots_30_31_keeping_max_36() -> None:
    assert MAX_CATALOG_TOOLS == 36
    assert _PRIORITY_ORDER[29] == "cu.session_end"
    assert _PRIORITY_ORDER[30:32] == ("mcp.task_result", "mcp.task_cancel")
    assert _PRIORITY_ORDER[32] == "cu.session_extend"

    unique_priority = tuple(dict.fromkeys(_PRIORITY_ORDER))
    host = _host(uuid.uuid4())
    descriptors = tuple(_descriptor(name) for name in unique_priority)
    manifest = Manifest(host_id=host.host_id, tools=descriptors, updated_at=NOW)
    grants = _grants(host.host_id, {d.family for d in descriptors})

    specs_reserved_0 = build_tool_catalog(
        hosts=[host],
        manifests_by_host={host.host_id: manifest},
        grants_by_host={host.host_id: grants},
        online_host_ids={str(host.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1,),
        host_screen_consent=True,
        now=NOW,
    )
    names_0 = [s.name for s in specs_reserved_0]
    assert len(specs_reserved_0) == MAX_CATALOG_TOOLS == 36
    assert TOOL_SEARCH_FUNCTION in names_0
    assert "mcp__task_result" in names_0
    assert "mcp__task_cancel" in names_0
    assert {n.replace(".", "__") for n in _PRIORITY_ORDER[:30]} <= set(names_0)
