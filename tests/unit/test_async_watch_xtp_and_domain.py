"""Unit tests for AW-M2 XTP watch frames (24 valid + 12 invalid parity) and domain models."""

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from xuna_backend.modules.actions.application.catalog import build_tool_catalog
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    JobStepOutcome,
)
from xuna_backend.modules.actions.domain.schema_validator import SchemaValidator
from xuna_backend.modules.actions.domain.watch import (
    JOB_WAIT_FUNCTION_NAME,
    JOB_WAIT_SCHEMA,
    ContinuationMode,
    JobWatch,
    TurnDetachedEvent,
    WakeInboxItem,
    WakeInboxKind,
    WakeSuggestedAction,
    WatchCompositeSignal,
    WatchGestureCue,
    WatchMode,
    WatchStatus,
    WatchTarget,
    compute_backoff_quiescence,
    validate_workspace_alias,
)
from xuna_backend.modules.hosts.application.xtp_frames import (
    ParsedWatchEvent,
    WatchTargetModel,
    build_watch_cancel_params,
    build_watch_register_params,
    parse_agent_frame,
    parse_watch_cancel_params,
    parse_watch_cancel_result,
    parse_watch_register_params,
    parse_watch_register_result,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
)
from xuna_backend.modules.hosts.domain.session import HostSessionRejected

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "xtp" / "watch_frames.json"


def test_watch_frames_golden_fixture_all_24_valid_and_12_invalid_parity() -> None:
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    frames: dict[str, dict[str, object]] = raw["frames"]
    invalid_frames: dict[str, dict[str, object]] = raw["invalidFrames"]

    assert len(frames) == 24
    assert len(invalid_frames) == 12

    for name, frame in frames.items():
        if "method" in frame and "id" in frame:
            method = frame["method"]
            params = frame["params"]
            assert isinstance(params, dict)
            if method == "xuna/watchRegister":
                parsed_reg = parse_watch_register_params(params)
                assert parsed_reg.schemaVersion == 1, name
            elif method == "xuna/watchCancel":
                parsed_cancel = parse_watch_cancel_params(params)
                assert parsed_cancel.schemaVersion == 1, name
            else:
                raise AssertionError(f"Unexpected request method in {name}: {method}")
        elif "method" in frame and "id" not in frame:
            msg = parse_agent_frame(json.dumps(frame))
            assert isinstance(msg, ParsedWatchEvent), name
            assert msg.params.parsed_occurred_at().tzinfo == UTC
        elif "result" in frame:
            result = frame["result"]
            assert isinstance(result, dict)
            if "accepted" in result:
                reg_res = parse_watch_register_result(result)
                assert reg_res.schemaVersion == 1, name
            elif "cancelledCount" in result:
                cancel_res = parse_watch_cancel_result(result)
                assert cancel_res.schemaVersion == 1, name
            else:
                raise AssertionError(f"Unexpected result shape in {name}")
        else:
            raise AssertionError(f"Unknown frame shape: {name}")

    for name, entry in invalid_frames.items():
        frame = entry["frame"]
        assert isinstance(frame, dict), name
        method = frame.get("method")
        if method == "xuna/watchEvent":
            with pytest.raises(HostSessionRejected):
                parse_agent_frame(json.dumps(frame))
        elif method == "xuna/watchRegister":
            params = frame["params"]
            assert isinstance(params, dict)
            with pytest.raises(ValueError):
                parse_watch_register_params(params)
        elif method == "xuna/watchCancel":
            params = frame["params"]
            assert isinstance(params, dict)
            with pytest.raises(ValueError):
                parse_watch_cancel_params(params)
        else:
            raise AssertionError(f"Unexpected invalid frame method in {name}: {method}")


def test_watch_frames_null_vs_omitted_optional_fields_and_control_char_rejection() -> None:
    params_with_nulls: dict[str, object] = {
        "schemaVersion": 1,
        "watchId": "f1111111-1111-4111-8111-111111111111",
        "jobId": "a2222222-2222-4222-8222-222222222222",
        "mode": "ui_quiescence",
        "target": {
            "pid": None,
            "windowId": None,
            "appId": "editor",
            "workspaceAlias": None,
            "tabId": None,
            "mcpServerId": None,
            "protocolTaskId": None,
        },
        "quiescenceSeconds": 8,
        "minWaitSeconds": 0,
        "maxWaitSeconds": 300,
        "requireInitialChange": True,
        "releaseFocus": True,
    }
    parsed = parse_watch_register_params(params_with_nulls)
    assert parsed.target.appId == "editor"
    assert parsed.target.pid is None

    cancel_with_null = parse_watch_cancel_params(
        {
            "schemaVersion": 1,
            "watchId": None,
            "jobId": "a2222222-2222-4222-8222-222222222222",
            "reason": "job_ended",
        }
    )
    assert cancel_with_null.watchId is None

    # Outbound builders omit None fields
    wire_reg = build_watch_register_params(
        watch_id="f1111111-1111-4111-8111-111111111111",
        job_id="a2222222-2222-4222-8222-222222222222",
        mode="ui_quiescence",
        signals=None,
        target=WatchTargetModel(appId="editor"),
        quiescence_seconds=8,
        min_wait_seconds=0,
        max_wait_seconds=300,
        require_initial_change=True,
        release_focus=True,
    )
    assert "signals" not in wire_reg
    assert wire_reg["target"] == {"appId": "editor"}

    wire_cancel = build_watch_cancel_params(reason="estop")
    assert wire_cancel == {"schemaVersion": 1, "reason": "estop"}

    # Trailing newline / control char rejection
    with pytest.raises(ValueError):
        WatchTargetModel(appId="editor\n")
    with pytest.raises(ValueError):
        WatchTargetModel(workspaceAlias="repo\r")
    with pytest.raises(ValueError):
        WatchTargetModel(tabId="tab\x001")
    with pytest.raises(ValueError):
        WatchTargetModel(protocolTaskId="task\x1b1")


def test_job_wait_schema_validates_and_is_excluded_from_interactive_catalog() -> None:
    SchemaValidator.validate_schema_definition(JOB_WAIT_SCHEMA)
    clean_args, target = SchemaValidator.validate_arguments(
        JOB_WAIT_SCHEMA,
        {
            "mode": "composite",
            "signals": ["ui_quiescence", "process_idle"],
            "doneCriteria": "Build output settles and worker process becomes idle",
            "quiescenceSeconds": 8,
            "maxWaitSeconds": 300,
            "continuationMode": "auto_continue",
            "workspaceAlias": "xuna_pet",
        },
        allow_target=False,
    )
    assert target is None
    assert clean_args["mode"] == "composite"

    # Interactive catalog never includes job__wait
    desc = ToolDescriptor(
        name="ui.inspect",
        description="Inspect active window tree.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=ToolFamily.UI,
        risk=RiskTier.R0,
        observes=True,
        reversible=True,
        permission=ToolPermissionState.GRANTED,
    )
    host_id = HostId.generate()
    host_obj = Host(
        host_id=host_id,
        owner_id=uuid.uuid4(),
        alias=HostAlias("desktop"),
        platform=HostPlatform.MACOS,
        hostname="desktop",
        os_version="14.0",
        agent_version="0.2.0",
        public_key=b"\x04" + bytes(range(1, 65)),
        status=HostStatus.ACTIVE,
        created_at=datetime.now(UTC),
    )
    manifest = Manifest(host_id=host_id, tools=(desc,), updated_at=datetime.now(UTC))
    grant = Grant(
        host_id=host_id,
        family=ToolFamily.UI,
        max_risk=RiskTier.R0,
        enabled=True,
        updated_at=datetime.now(UTC),
    )
    catalog = build_tool_catalog(
        hosts=[host_obj],
        manifests_by_host={host_id: manifest},
        grants_by_host={host_id: [grant]},
        online_host_ids=[str(host_id)],
        client_caps=("host_actions_v1",),
        now=datetime.now(UTC),
    )
    names = {spec.name for spec in catalog}
    assert JOB_WAIT_FUNCTION_NAME not in names


def test_domain_watch_entities_job_transitions_and_backoff() -> None:
    assert compute_backoff_quiescence(8) == 12
    assert compute_backoff_quiescence(12) == 18
    assert compute_backoff_quiescence(50) == 60
    assert compute_backoff_quiescence(1) == 2

    with pytest.raises(ValueError):
        validate_workspace_alias("../secret")
    with pytest.raises(ValueError):
        validate_workspace_alias("a/b")

    target = WatchTarget(pid=1234, window_id=0, app_id="editor", workspace_alias="xuna_pet")
    assert target.to_wire_dict() == {
        "pid": 1234,
        "windowId": 0,
        "appId": "editor",
        "workspaceAlias": "xuna_pet",
    }

    now = datetime.now(UTC)
    watch = JobWatch(
        watch_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        host_id=HostId.generate(),
        mode=WatchMode.COMPOSITE,
        signals=(WatchCompositeSignal.UI_QUIESCENCE, WatchCompositeSignal.PROCESS_IDLE),
        target=target,
        done_criteria="Wait until build completes",
        continuation_mode=ContinuationMode.REPORT_AND_ASK,
        expires_at=now + timedelta(seconds=300),
        created_at=now,
        updated_at=now,
    )
    summary = watch.to_summary_dict(now=now + timedelta(seconds=15))
    assert summary["mode"] == "composite"
    assert summary["elapsedSeconds"] == 15
    assert summary["status"] == WatchStatus.ACTIVE.value

    inbox = WakeInboxItem(
        item_id=uuid.uuid4(),
        owner_id=watch.owner_id,
        pet_id="pet_1",
        job_id=watch.job_id,
        host_id=watch.host_id,
        kind=WakeInboxKind.JOB_NEEDS_DECISION,
        title="Ready to continue",
        summary="Build completed",
        suggested_actions=(WakeSuggestedAction.CONTINUE_JOB, WakeSuggestedAction.DISMISS),
        gesture_cue=WatchGestureCue.LOOK_AROUND,
        created_at=now,
    )
    wire_inbox = inbox.to_dict()
    assert wire_inbox["suggestedActions"] == ["continue_job", "dismiss"]
    assert wire_inbox["gestureCue"] == "look_around"

    detached = TurnDetachedEvent(
        job_id=str(watch.job_id),
        watch_id=str(watch.watch_id),
        host_id=str(watch.host_id),
        title="Long build",
        state=AutonomousJobState.WAITING_WATCH.value,
    )
    assert detached.wire_payload()["reason"] == "yield_ms"

    # AutonomousJob WAITING_WATCH transitions
    job = AutonomousJob(
        job_id=watch.job_id,
        owner_id=watch.owner_id,
        pet_id="pet_1",
        host_id=watch.host_id,
        session_id="cu_1",
        title="Long build",
        goal="Run and inspect build",
        state=AutonomousJobState.RUNNING,
    )
    waiting = job.with_state(AutonomousJobState.WAITING_WATCH, now=now)
    assert waiting.is_active is True
    assert waiting.is_terminal is False
    resumed = waiting.with_state(AutonomousJobState.RUNNING, now=now)
    assert resumed.state == AutonomousJobState.RUNNING
    paused = waiting.with_state(AutonomousJobState.PAUSED, now=now)
    rewatched = paused.with_state(AutonomousJobState.WAITING_WATCH, now=now)
    assert rewatched.state == AutonomousJobState.WAITING_WATCH
    assert JobStepOutcome(kind="waiting_watch").kind == "waiting_watch"


@pytest.mark.parametrize("bad_version", [True, False, 1.0, "1", 0, 2, None])
def test_watch_frames_strict_schema_version_rejects_bool_float_and_wrong_int(
    bad_version: object,
) -> None:
    watch_id = "f1111111-1111-4111-8111-111111111111"
    job_id = "a2222222-2222-4222-8222-222222222222"

    with pytest.raises(ValueError):
        parse_watch_register_params(
            {
                "schemaVersion": bad_version,
                "watchId": watch_id,
                "jobId": job_id,
                "mode": "timer",
                "target": {},
                "quiescenceSeconds": 8,
                "minWaitSeconds": 0,
                "maxWaitSeconds": 300,
                "requireInitialChange": False,
                "releaseFocus": True,
            }
        )

    with pytest.raises(ValueError):
        parse_watch_register_result(
            {
                "schemaVersion": bad_version,
                "accepted": True,
                "watchId": watch_id,
                "activeSignals": ["timer"],
                "eventDriven": True,
                "fallbackUsed": False,
            }
        )

    with pytest.raises(ValueError):
        parse_watch_register_result(
            {
                "schemaVersion": bad_version,
                "accepted": False,
                "watchId": watch_id,
                "errorCode": "target_not_found",
            }
        )

    with pytest.raises(ValueError):
        parse_watch_cancel_params(
            {
                "schemaVersion": bad_version,
                "watchId": watch_id,
                "jobId": job_id,
                "reason": "owner_cancel",
            }
        )

    with pytest.raises(ValueError):
        parse_watch_cancel_result(
            {
                "schemaVersion": bad_version,
                "cancelledCount": 1,
            }
        )

    event_frame = {
        "jsonrpc": "2.0",
        "method": "xuna/watchEvent",
        "params": {
            "schemaVersion": bad_version,
            "watchId": watch_id,
            "jobId": job_id,
            "kind": "settled",
            "observedSignals": ["timer"],
            "elapsedSeconds": 5,
            "changeCount": 1,
            "digestChanged": False,
            "occurredAt": datetime.now(UTC).isoformat(),
        },
    }
    with pytest.raises(HostSessionRejected):
        parse_agent_frame(json.dumps(event_frame))
