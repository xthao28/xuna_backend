"""Upgrade plan P1.1 (U3): `exec.run` `expectFiles` on the backend side.

The host-advertised descriptor (golden fixture) must stay in the catalog, which
needs the schema subset to accept objects as array items; the model's
`expectFiles` arguments are validated like any untrusted input; the host's
`expected_output_*` result keeps its `artifacts`; the approval card says how
many output files the host will check.
"""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.test_wave_c_e2e import (
    NOW,
    EventCollector,
    _FakeAuditLog,
    _FakeHostRepository,
    _FakeSessionHub,
    _make_action_context,
    _make_host,
)

from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.domain.exec_fs_summary import exec_fs_pending_text
from xuna_backend.modules.actions.domain.schema_validator import (
    ArgumentValidationError,
    SchemaDefinitionError,
    SchemaValidator,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import ToolCall
from xuna_backend.modules.hosts.application.xtp_frames import parse_preflight_result
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    HostPlatform,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "xtp" / "exec_fs_frames.json"


def _frames() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    frames: dict[str, Any] = data["frames"]
    return frames


def _run_schema() -> dict[str, object]:
    tools = _frames()["toolsListExecFsResponse"]["result"]["tools"]
    schema: dict[str, object] = next(t for t in tools if t["name"] == "exec.run")["inputSchema"]
    return schema


def test_exec_run_descriptor_with_expect_files_stays_a_valid_catalog_schema() -> None:
    SchemaValidator.validate_schema_definition(_run_schema())


def test_fixture_request_arguments_validate() -> None:
    args = _frames()["toolsCallExecRunExpectFilesRequest"]["params"]["arguments"]
    clean, target = SchemaValidator.validate_arguments(_run_schema(), args)
    assert clean == args and target is None


@pytest.mark.parametrize(
    "expect_files",
    [
        [],
        [{}],
        [{"path": ""}],
        [{"path": "/w/a", "extra": 1}],
        [{"path": "/w/a", "minBytes": -1}],
        [{"path": "/w/a", "minBytes": True}],
        [{"path": "/w/a", "magic": "zz"}],
        [{"path": "/w/a", "sha256": "abc"}],
        [{"path": "/w/a"}] * 17,
        ["/w/a"],
        {"path": "/w/a"},
    ],
)
def test_malformed_expect_files_are_rejected(expect_files: object) -> None:
    with pytest.raises(ArgumentValidationError):
        SchemaValidator.validate_arguments(
            _run_schema(), {"command": "npm test", "expectFiles": expect_files}
        )


def _array_of(item: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"xs": {"type": "array", "items": item}},
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    "item",
    [
        # Item objects must forbid extra keys.
        {"type": "object", "properties": {"a": {"type": "string"}}},
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": True,
        },
        # No further nesting.
        {
            "type": "object",
            "properties": {"a": {"type": "object"}},
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"a": {"type": "array", "items": {"type": "string"}}},
            "additionalProperties": False,
        },
        # Required keys must be declared; unknown keywords are refused.
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["b"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
            "patternProperties": {},
        },
        {"type": "object", "properties": {}, "additionalProperties": False},
    ],
)
def test_unsafe_item_object_schemas_are_refused(item: dict[str, object]) -> None:
    with pytest.raises(SchemaDefinitionError):
        SchemaValidator.validate_schema_definition(_array_of(item))


def test_top_level_object_properties_are_still_refused() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "o": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "additionalProperties": False,
    }
    with pytest.raises(SchemaDefinitionError):
        SchemaValidator.validate_schema_definition(schema)


def test_expected_output_failure_keeps_artifacts_and_error_code() -> None:
    result = _frames()["toolsCallExecRunExpectedOutputMissingResponse"]["result"]
    assert result["isError"] is True
    assert result["_meta"]["xuna/status"] == "failed"
    assert result["_meta"]["xuna/errorCode"] == "expected_output_missing"
    sc = result["structuredContent"]
    assert sc["status"] == "failed" and sc["exitCode"] == 0
    assert sc["artifacts"] == [
        {"path": "~/Projects/app/dist/report.pdf", "exists": False, "problem": "missing"}
    ]
    done = _frames()["toolsCallExecRunExpectFilesCompletedResponse"]["result"]
    artifact = done["structuredContent"]["artifacts"][0]
    assert set(artifact) == {"path", "exists", "sizeBytes", "sha256", "magicOk"}
    assert len(artifact["sha256"]) == 64


def test_secret_expected_path_is_a_refused_preflight() -> None:
    refused = parse_preflight_result(
        _frames()["preflightExecRunExpectFilesSecretResponse"]["result"]
    )
    assert refused.status == "denied"
    assert refused.error_code == "fs_secret_file_forbidden"
    assert refused.effective_risk is None


def _orchestrator_with_fixture_descriptor() -> tuple[Orchestrator, _FakeSessionHub, uuid.UUID]:
    owner_id = uuid.uuid4()
    host_repo = _FakeHostRepository()
    hub = _FakeSessionHub()
    host = _make_host(owner_id, platform=HostPlatform.WINDOWS)
    host_repo.hosts[host.host_id] = host
    tools = _frames()["toolsListExecFsResponse"]["result"]["tools"]
    run = ToolDescriptor.from_wire(next(t for t in tools if t["name"] == "exec.run"))
    host_repo.manifests[host.host_id] = Manifest(
        host_id=host.host_id, tools=(run,), updated_at=NOW, schema_version=1
    )
    host_repo.grants[host.host_id] = [
        Grant(
            host_id=host.host_id,
            family=ToolFamily.EXEC,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=NOW,
        )
    ]
    hub.custom_responses["xuna/preflight"] = {
        "preflight": {"status": "ok", "effectiveRisk": "R2", "reasonCode": "local_command"}
    }
    orch = Orchestrator(
        host_repo=host_repo,
        hub=hub,
        approval_repo=InMemoryApprovalRepository(),
        approval_broker=InMemoryApprovalBroker(),
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=_FakeAuditLog(),
        scope_store=InMemoryApprovalScopeStore(),
        clock=lambda: NOW,
    )
    return orch, hub, owner_id


@pytest.mark.anyio
async def test_model_sees_the_failure_code_and_artifacts_of_an_exit_zero_run() -> None:
    orch, hub, owner_id = _orchestrator_with_fixture_descriptor()
    hub.custom_responses["exec.run"] = _frames()["toolsCallExecRunExpectedOutputMissingResponse"][
        "result"
    ]
    collector = EventCollector(orch, owner_id)
    args = _frames()["toolsCallExecRunExpectFilesRequest"]["params"]["arguments"]
    msg = await orch.handle_call(
        ToolCall(id="p11", name="exec__run", arguments=dict(args)),
        _make_action_context(owner_id, collector),
    )
    sent = [c for c in hub.tool_calls if c["tool"] == "exec.run"]
    assert len(sent) == 1 and sent[0]["arguments"]["expectFiles"] == args["expectFiles"]
    body = json.loads(msg.content)
    assert body["status"] == "failed"
    output = body["output"]
    assert output["status"] == "failed" and output["exitCode"] == 0
    assert output["errorCode"] == "expected_output_missing"
    assert output["artifacts"] == [
        {"path": "~/Projects/app/dist/report.pdf", "exists": False, "problem": "missing"}
    ]


def test_approval_card_counts_expected_output_files() -> None:
    args: dict[str, object] = {
        "command": "npm run build",
        "cwd": "~/app",
        "expectFiles": [{"path": "~/app/a"}, {"path": "~/app/b"}],
    }
    vi = exec_fs_pending_text("exec.run", args, "Laptop", english=False)
    en = exec_fs_pending_text("exec.run", args, "Laptop", english=True)
    assert vi is not None and "kiểm tra 2 tệp đầu ra" in vi
    assert en is not None and "check 2 output file(s)" in en
    plain = exec_fs_pending_text("exec.run", {"command": "npm test"}, "Laptop", english=False)
    assert plain == "Chạy lệnh `npm test` trong ~ trên Laptop"
