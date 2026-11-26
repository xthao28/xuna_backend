"""Unit & contract tests for Milestone 3 (P1.2 / U4):
- `xuna/resources.toolchains` strict XTP parsing & RAM-only `HostResources`
- Generic `[HOST_ROSTER]`, `mesh__find_resource(kind="toolchain")`, and `score_resource_match`
- `structuredContent.remediation` fail-closed validation, `ActionResultEvent` forwarding, & hint
- Rule H5 compliance: zero hardcoded toolchain/package-manager names in backend generic layers
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.test_wave_c_e2e import (
    EventCollector,
    _FakeAuditLog,
    _FakeHostRepository,
    _FakeSessionHub,
    _make_action_context,
    _make_host,
)

from xuna_backend.modules.actions.application.mesh_prober import MeshProber, ProbeTarget
from xuna_backend.modules.actions.application.mesh_router import format_host_roster
from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.application.prompts import (
    encode_tool_payload,
    state_driven_hint,
    truncate_tool_value,
)
from xuna_backend.modules.actions.application.tool_discovery import (
    MESH_FIND_RESOURCE_FUNCTION,
    MetaToolArgumentError,
    build_mesh_tool_specs,
    parse_mesh_find_resource_args,
)
from xuna_backend.modules.actions.domain.entities import ActionResultEvent
from xuna_backend.modules.actions.domain.mesh_scoring import score_resource_match
from xuna_backend.modules.actions.domain.remediation import sanitize_remediation
from xuna_backend.modules.actions.domain.schema_validator import SchemaValidator
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import ToolCall
from xuna_backend.modules.hosts.application.xtp_frames import ParsedResources, parse_agent_frame
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
)
from xuna_backend.modules.hosts.domain.session import HostSessionRejected
from xuna_backend.modules.hosts.domain.twin import HostResources, HostToolchain
from xuna_backend.modules.hosts.infrastructure.session_hub import InMemoryHostSessionHub

NOW = datetime(2026, 11, 26, 7, 0, 0, tzinfo=UTC)
PUB = b"\x04" + b"\x44" * 64
FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "xtp"
BACKEND_ROOT = Path(__file__).resolve().parents[3]


def _load_fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))
    return data


def _make_simple_host(owner_id: uuid.UUID, alias: str, platform: HostPlatform) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname=alias.lower(),
        os_version="15.0",
        agent_version="0.3.0",
        public_key=PUB,
        status=HostStatus.ACTIVE,
        created_at=NOW,
    )


def _exec_run_descriptor_from_fixture() -> ToolDescriptor:
    frames = _load_fixture("exec_fs_frames.json")["frames"]
    tools = frames["toolsListExecFsResponse"]["result"]["tools"]
    raw = next(t for t in tools if t["name"] == "exec.run")
    return ToolDescriptor.from_wire(raw)


# -----------------------------------------------------------------------------
# 1. XTP `xuna/resources.toolchains` Parsing & Domain Invariants
# -----------------------------------------------------------------------------


def test_resources_with_toolchains_golden_fixture_parses_and_stores_in_ram() -> None:
    frames = _load_fixture("mesh_frames.json")["frames"]

    parsed_with_tc = parse_agent_frame(json.dumps(frames["resourcesWithToolchainsNotification"]))
    assert isinstance(parsed_with_tc, ParsedResources)
    assert [(tc.id, tc.version) for tc in parsed_with_tc.params.toolchains] == [
        ("git", "2.47.1"),
        ("node", "22.11.0"),
    ]

    # Backward compatibility when toolchains is omitted
    parsed_legacy = parse_agent_frame(json.dumps(frames["resourcesNotification"]))
    assert isinstance(parsed_legacy, ParsedResources)
    assert parsed_legacy.params.toolchains == []

    # Domain mapping
    resources = HostResources(
        installed_apps=tuple(parsed_with_tc.params.installedApps),
        workspaces=tuple(parsed_with_tc.params.workspaces),
        updated_at=NOW,
        capabilities_version=parsed_with_tc.params.capabilitiesVersion,
        toolchains=tuple(
            HostToolchain(id=tc.id, version=tc.version) for tc in parsed_with_tc.params.toolchains
        ),
    )
    assert resources.toolchains == (
        HostToolchain(id="git", version="2.47.1"),
        HostToolchain(id="node", version="22.11.0"),
    )


@pytest.mark.parametrize(
    "toolchains_payload",
    [
        [{"id": "/usr/bin/node", "version": "22.11.0"}],
        [{"id": "C:\\node", "version": "22.11.0"}],
        [{"id": "..", "version": "1.0.0"}],
        [{"id": ".", "version": "1.0.0"}],
        [{"id": "foo..bar", "version": "1.0.0"}],
        [{"id": "git\n", "version": "2.47.1"}],
        [{"id": "git\x00", "version": "2.47.1"}],
        [{"id": "git ", "version": "2.47.1"}],
        [{"id": "git", "version": "/opt/git/2.47"}],
        [{"id": "git", "version": "2.47\\1"}],
        [{"id": "git", "version": "2.47..1"}],
        [{"id": "git", "version": "2.47.1\n"}],
        [{"id": "git", "version": "2.47.1\x7f"}],
        [{"id": "git", "version": "2.47 1"}],
        [{"id": "git", "version": "2.47.1", "path": "/usr/bin/git"}],
        [{"id": "git", "version": "2.47.1"}, {"id": "git", "version": "2.48.0"}],
        [{"id": f"tool_{i}", "version": f"1.0.{i}"} for i in range(65)],
        None,
    ],
)
def test_resources_toolchains_fail_closed_in_xtp_parser(toolchains_payload: object) -> None:
    frame = {
        "jsonrpc": "2.0",
        "method": "xuna/resources",
        "params": {
            "schemaVersion": 1,
            "installedApps": [],
            "workspaces": [],
            "toolchains": toolchains_payload,
        },
    }
    with pytest.raises(HostSessionRejected):
        parse_agent_frame(json.dumps(frame))


def test_host_toolchain_and_resources_domain_reject_invalid_values() -> None:
    for bad_id in ("/bin/sh", "a\\b", ".", "..", "a..b", "git\n", "git ", "UPPER"):
        with pytest.raises(ValueError):
            HostToolchain(id=bad_id, version="1.0.0")

    for bad_ver in ("", "/1.0", "1\\0", "1..0", "1.0\r", "1.0\x01", "1.0 beta", "-1.0"):
        with pytest.raises(ValueError):
            HostToolchain(id="git", version=bad_ver)

    with pytest.raises(ValueError):
        HostResources(
            installed_apps=(),
            workspaces=(),
            updated_at=NOW,
            toolchains=(HostToolchain("git", "2.47.1"), HostToolchain("git", "2.48.0")),
        )

    with pytest.raises(ValueError):
        HostResources(
            installed_apps=(),
            workspaces=(),
            updated_at=NOW,
            toolchains=tuple(HostToolchain(f"t{i}", "1.0") for i in range(65)),
        )


# -----------------------------------------------------------------------------
# 2. `[HOST_ROSTER]`, `mesh__find_resource`, & `score_resource_match`
# -----------------------------------------------------------------------------


def test_host_roster_includes_bounded_toolchains_without_paths() -> None:
    import asyncio

    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        host = _make_simple_host(owner, "DevBox", HostPlatform.WINDOWS)
        await hub.open_session(
            str(host.host_id), object(), lambda _: _noop(), _noop, _noop, owner_id=owner
        )
        many_tc = tuple(HostToolchain(id=f"tool_{i:02d}", version=f"1.{i}") for i in range(30))
        hub.update_resources(
            str(host.host_id),
            HostResources(
                installed_apps=("editor",),
                workspaces=("repo",),
                updated_at=NOW,
                toolchains=many_tc,
            ),
        )
        roster = format_host_roster(hosts=[host], hub=hub, owner_id=owner)
        assert "toolchains=[" in roster
        assert "tool_00@1.0" in roster
        assert "tool_23@1.23" in roster
        assert "tool_24@1.24" not in roster
        assert "+6" in roster

    async def _noop() -> None:
        return None

    asyncio.run(_run())


def test_mesh_find_resource_supports_toolchain_and_passes_schema_validator() -> None:
    import asyncio

    specs = build_mesh_tool_specs()
    find_spec = next(s for s in specs if s.name == MESH_FIND_RESOURCE_FUNCTION)
    SchemaValidator.validate_schema_definition(find_spec.parameters)
    assert "toolchain" in find_spec.parameters["properties"]["kind"]["enum"]

    assert parse_mesh_find_resource_args({"query": "node", "kind": "toolchain"}) == (
        "node",
        "toolchain",
    )
    with pytest.raises(MetaToolArgumentError):
        parse_mesh_find_resource_args({"query": "node", "kind": "unknown_kind"})

    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        prober = MeshProber(hub=hub, clock=lambda: NOW)

        h_with_tc = _make_simple_host(owner, "BuildWin", HostPlatform.WINDOWS)
        h_no_tc = _make_simple_host(owner, "PlainMac", HostPlatform.MACOS)

        async def _noop() -> None:
            return None

        await hub.open_session(
            str(h_with_tc.host_id), object(), lambda _: _noop(), _noop, _noop, owner_id=owner
        )
        await hub.open_session(
            str(h_no_tc.host_id), object(), lambda _: _noop(), _noop, _noop, owner_id=owner
        )
        # Only toolchains populated (apps/workspaces empty) -> has_twin_resources must be True
        hub.update_resources(
            str(h_with_tc.host_id),
            HostResources(
                installed_apps=(),
                workspaces=(),
                updated_at=NOW,
                toolchains=(
                    HostToolchain(id="git", version="2.47.1"),
                    HostToolchain(id="node", version="22.11.0"),
                ),
            ),
        )
        hub.update_resources(
            str(h_no_tc.host_id),
            HostResources(installed_apps=(), workspaces=(), updated_at=NOW, toolchains=()),
        )
        targets = [
            ProbeTarget(host=h_with_tc, manifest=None, grants=(), online=True),
            ProbeTarget(host=h_no_tc, manifest=None, grants=(), online=True),
        ]

        res_exact = await prober.find_resource(
            owner_id=owner, targets=targets, query="node", kind="toolchain"
        )
        assert res_exact["matches"] == [
            {
                "hostId": str(h_with_tc.host_id),
                "alias": "BuildWin",
                "platform": "windows",
                "kind": "toolchain",
                "id": "node",
                "version": "22.11.0",
                "exact": True,
                "source": "twin",
            }
        ]

        res_all = await prober.find_resource(
            owner_id=owner, targets=targets, query="git", kind=None
        )
        assert len(res_all["matches"]) == 1  # type: ignore[arg-type]
        assert res_all["matches"][0]["kind"] == "toolchain"  # type: ignore[index]

        res_miss = await prober.find_resource(
            owner_id=owner, targets=targets, query="nonexistent_tc", kind="toolchain"
        )
        assert res_miss["matches"] == []

    asyncio.run(_run())


def test_score_resource_match_considers_toolchains_generically() -> None:
    desc = ToolDescriptor(
        name="exec.run",
        description="Run command",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=ToolFamily.EXEC,
        risk=RiskTier.R2,
    )
    has_tc = HostResources(
        installed_apps=(),
        workspaces=(),
        updated_at=NOW,
        toolchains=(HostToolchain(id="runtime_alpha", version="2.1.0"),),
    )
    miss_tc = HostResources(
        installed_apps=(),
        workspaces=(),
        updated_at=NOW,
        toolchains=(HostToolchain(id="other_beta", version="1.0.0"),),
    )
    assert score_resource_match(desc, has_tc, None, "runtime_alpha") == 1.00
    assert score_resource_match(desc, has_tc, None, "runtime") == 0.85
    assert score_resource_match(desc, miss_tc, None, "runtime_alpha") == 0.25


# -----------------------------------------------------------------------------
# 3. `remediation` Validation, `ActionResultEvent` Forwarding & Hint
# -----------------------------------------------------------------------------


def test_sanitize_remediation_accepts_valid_and_rejects_malformed_or_leaky() -> None:
    valid = sanitize_remediation(
        {
            "summary": "  Runtime is required to run this command.  ",
            "command": "  winget install OpenJS.NodeJS.LTS  ",
            "kind": "install",
        }
    )
    assert valid == {
        "summary": "Runtime is required to run this command.",
        "command": "winget install OpenJS.NodeJS.LTS",
        "kind": "install",
    }

    # Valid without command + https URL allowed
    valid_url = sanitize_remediation(
        {
            "summary": "Download from https://example.com/setup",
            "kind": "manual",
        }
    )
    assert valid_url == {
        "summary": "Download from https://example.com/setup",
        "kind": "manual",
    }

    # Valid slash-containing package names and https docs URLs allowed
    for cmd_payload in (
        {
            "summary": "Install Dart SDK.",
            "command": "brew install dart-lang/dart/dart",
            "kind": "install",
        },
        {
            "summary": "Install Claude Code CLI.",
            "command": "npm install -g @anthropic-ai/claude-code",
            "kind": "install",
        },
        {
            "summary": "Install Dart SDK from https://dart.dev/get-dart and add to PATH.",
            "kind": "manual",
        },
    ):
        assert sanitize_remediation(cmd_payload) == cmd_payload

    # Fail-closed malformed / path-leaking / secret-leaking payloads
    bad_inputs: list[object] = [
        None,
        "install node",
        {},
        {"summary": "Missing tool", "kind": "install", "extra": "nope"},
        {"summary": "Missing tool", "kind": "auto_run"},
        {"summary": "", "kind": "install"},
        {"summary": "   ", "kind": "config"},
        {"summary": "x" * 241, "kind": "install"},
        {"summary": "Line 1\nLine 2", "kind": "install"},
        {"summary": "Has\x00nul", "kind": "install"},
        {"summary": "Missing tool", "command": "  ", "kind": "install"},
        {"summary": "Missing tool", "command": "cmd\r\necho 1", "kind": "install"},
        {"summary": "Check /Users/alice/bin/tool", "kind": "config"},
        {
            "summary": "Missing tool",
            "command": "/home/bob/.local/bin/install.sh",
            "kind": "install",
        },
        {"summary": "Missing tool", "command": "C:\\Users\\alice\\setup.exe", "kind": "install"},
        {"summary": "Missing tool", "command": "D:/Tools/setup.exe", "kind": "install"},
        {
            "summary": "Missing tool",
            "command": "\\\\fileserver\\share\\setup.exe",
            "kind": "install",
        },
        {"summary": "Missing tool", "command": "file:///etc/passwd", "kind": "manual"},
        {"summary": "Missing tool", "command": "~/bin/install", "kind": "install"},
        {"summary": "Run ~\\secret", "kind": "install"},
        {"summary": "Run .\\local\\bin", "kind": "install"},
        {"summary": "Run ..\\secret", "kind": "install"},
        {"summary": "Run ./local/bin", "kind": "install"},
        {"summary": "Run ../secret", "kind": "install"},
        {"summary": "Check /etc", "kind": "config"},
        {"summary": "Check /etc.", "kind": "config"},
        {"summary": "Check /Users", "kind": "config"},
        {"summary": "Check /Applications", "kind": "config"},
        {"summary": "Missing tool", "command": "echo sk-1234567890abcdef1234", "kind": "install"},
        {
            "summary": "Missing tool",
            "command": "curl -H 'Authorization: Bearer secret_token_12345' https://example.com",
            "kind": "install",
        },
    ]
    for bad in bad_inputs:
        assert sanitize_remediation(bad) is None, f"Expected None for {bad!r}"


def test_action_result_event_wire_payload_remediation() -> None:
    without_rem = ActionResultEvent(
        invocation_id="inv-1",
        target="WorkPC",
        summary="Failed",
        status="failed",
        error_code="toolchain_missing",
    )
    assert "remediation" not in without_rem.wire_payload()

    rem: dict[str, object] = {
        "summary": "Runtime missing",
        "command": "winget install Example.Tool",
        "kind": "install",
    }
    with_rem = ActionResultEvent(
        invocation_id="inv-2",
        target="WorkPC",
        summary="Failed",
        status="failed",
        error_code="toolchain_missing",
        remediation=rem,
    )
    wire = with_rem.wire_payload()
    assert wire["remediation"] == rem


@pytest.mark.anyio
async def test_orchestrator_forwards_golden_remediation_and_drops_malformed_or_ok() -> None:
    frames = _load_fixture("exec_fs_frames.json")["frames"]
    golden_result = frames["toolsCallToolchainMissingRemediationResponse"]["result"]

    def _build_orch() -> tuple[Orchestrator, _FakeSessionHub, uuid.UUID]:
        owner_id = uuid.uuid4()
        host_repo = _FakeHostRepository()
        hub = _FakeSessionHub()
        host = _make_host(owner_id, platform=HostPlatform.WINDOWS)
        host_repo.hosts[host.host_id] = host
        desc = _exec_run_descriptor_from_fixture()
        host_repo.manifests[host.host_id] = Manifest(
            host_id=host.host_id, tools=(desc,), updated_at=NOW, schema_version=1
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

    # 1. Golden fixture -> forwards remediation to ActionResultEvent and emits hint to LLM
    orch1, hub1, owner1 = _build_orch()
    hub1.custom_responses["exec.run"] = golden_result
    collector1 = EventCollector(orch1, owner1)
    msg1 = await orch1.handle_call(
        ToolCall(id="rem-1", name="exec__run", arguments={"command": "node -v"}),
        _make_action_context(owner1, collector1),
    )
    res_ev1 = collector1.result_events[0]
    assert res_ev1.status == "failed"
    assert res_ev1.error_code == "toolchain_missing"
    assert res_ev1.remediation == golden_result["structuredContent"]["remediation"]
    assert (
        res_ev1.wire_payload()["remediation"] == golden_result["structuredContent"]["remediation"]
    )
    body1 = json.loads(msg1.content)
    assert body1["status"] == "failed"
    assert body1["output"]["remediation"] == golden_result["structuredContent"]["remediation"]
    assert "hint" in body1
    assert "not" in body1["hint"].lower() or "never" in body1["hint"].lower()

    # 2. Path-leaking remediation is dropped from ActionResultEvent AND Role.TOOL output
    orch2, hub2, owner2 = _build_orch()
    hub2.custom_responses["exec.run"] = {
        "_meta": {
            "xuna/status": "failed",
            "xuna/errorCode": "toolchain_missing",
            "xuna/durationMs": 4,
        },
        "structuredContent": {
            "status": "failed",
            "errorCode": "toolchain_missing",
            "remediation": {
                "summary": "Install dependency",
                "command": "C:\\Users\\alice\\installer.exe",
                "kind": "install",
            },
        },
        "isError": True,
    }
    collector2 = EventCollector(orch2, owner2)
    msg2 = await orch2.handle_call(
        ToolCall(id="rem-2", name="exec__run", arguments={"command": "node -v"}),
        _make_action_context(owner2, collector2),
    )
    res_ev2 = collector2.result_events[0]
    assert res_ev2.remediation is None
    assert "remediation" not in res_ev2.wire_payload()
    body2 = json.loads(msg2.content)
    assert "remediation" not in body2["output"]

    # 3. structuredContent.status == "failed" with _meta xuna/status == "ok" still propagates
    orch3, hub3, owner3 = _build_orch()
    hub3.custom_responses["exec.run"] = {
        "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
        "structuredContent": {
            "status": "failed",
            "exitCode": 127,
            "output": "not found",
            "truncated": False,
            "durationMs": 5,
            "errorCode": "toolchain_missing",
            "remediation": {
                "summary": "Missing dependency on host.",
                "command": "brew install example-pkg",
                "kind": "install",
            },
        },
        "isError": False,
    }
    collector3 = EventCollector(orch3, owner3)
    await orch3.handle_call(
        ToolCall(id="rem-3", name="exec__run", arguments={"command": "node -v"}),
        _make_action_context(owner3, collector3),
    )
    res_ev3 = collector3.result_events[0]
    assert res_ev3.error_code == "toolchain_missing"
    assert res_ev3.remediation == {
        "summary": "Missing dependency on host.",
        "command": "brew install example-pkg",
        "kind": "install",
    }

    # 4. Remediation on true `ok` result is stripped and never forwarded
    orch4, hub4, owner4 = _build_orch()
    hub4.custom_responses["exec.run"] = {
        "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
        "structuredContent": {
            "status": "completed",
            "exitCode": 0,
            "output": "ok",
            "truncated": False,
            "durationMs": 5,
            "remediation": {
                "summary": "Should not leak on ok",
                "kind": "manual",
            },
        },
        "isError": False,
    }
    collector4 = EventCollector(orch4, owner4)
    msg4 = await orch4.handle_call(
        ToolCall(id="rem-4", name="exec__run", arguments={"command": "echo ok"}),
        _make_action_context(owner4, collector4),
    )
    res_ev4 = collector4.result_events[0]
    assert res_ev4.remediation is None
    assert "remediation" not in json.loads(msg4.content)["output"]


def test_state_driven_hint_for_remediation_is_generic() -> None:
    hint = state_driven_hint(
        {
            "errorCode": "toolchain_missing",
            "remediation": {"summary": "Missing tool", "kind": "install"},
        }
    )
    assert hint is not None
    assert "remediation" in hint.lower()
    assert state_driven_hint({"status": "completed"}) is None
    encoded = json.loads(
        encode_tool_payload(
            "failed",
            {
                "errorCode": "toolchain_missing",
                "remediation": {"summary": "Missing tool", "kind": "install"},
            },
        )
    )
    assert "hint" in encoded


def test_truncate_tool_value_preserves_and_bounds_remediation_beyond_16_keys() -> None:
    payload: dict[str, object] = {f"k{i}": i for i in range(18)}
    payload["remediation"] = {
        "summary": "S" * 420,
        "command": "winget install Example.Tool",
        "kind": "install",
    }
    truncated = truncate_tool_value(payload, max_str=120)
    assert isinstance(truncated, dict)
    assert "remediation" in truncated
    rem = truncated["remediation"]
    assert isinstance(rem, dict)
    assert len(rem["summary"]) == 120
    assert rem["command"] == "winget install Example.Tool"
    assert rem["kind"] == "install"


# -----------------------------------------------------------------------------
# 4. Rule H5: Zero Hardcoded Toolchain Names in Backend Generic Layers
# -----------------------------------------------------------------------------


def test_rule_h5_zero_hardcoded_toolchain_names_in_generic_layers() -> None:
    always_forbidden_toolchains = re.compile(
        r"\b(flutter|dart|cargo|rustc|docker|ffmpeg|soffice|winget|brew|choco|"
        r"npm|pnpm|yarn|pip|pip3|javac|dotnet|python3|nodejs)\b",
        re.IGNORECASE,
    )
    curated_exact_literals = frozenset(
        {
            "git",
            "node",
            "npm",
            "python",
            "python3",
            "dart",
            "flutter",
            "go",
            "cargo",
            "docker",
            "gh",
            "rg",
            "ffmpeg",
            "soffice",
            "winget",
            "brew",
            "choco",
            "uv",
            "pip",
            "rustc",
        }
    )

    generic_files = [
        BACKEND_ROOT / "src/xuna_backend/modules/hosts/application/xtp_frames.py",
        BACKEND_ROOT / "src/xuna_backend/modules/hosts/domain/twin.py",
        BACKEND_ROOT / "src/xuna_backend/modules/hosts/api/host_socket.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/domain/remediation.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/domain/entities.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/domain/mesh_scoring.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/application/mesh_router.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/application/mesh_prober.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/application/tool_discovery.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/application/mesh_call_helper.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/application/prompts.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/application/prompt_dev_modules.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/application/catalog.py",
    ]
    conv_app_dir = BACKEND_ROOT / "src/xuna_backend/modules/conversation/application"
    if conv_app_dir.exists():
        generic_files.extend(sorted(conv_app_dir.rglob("*.py")))

    violations: list[str] = []
    for file_path in generic_files:
        assert file_path.exists(), f"Missing expected generic file: {file_path}"
        source = file_path.read_text(encoding="utf-8")
        for line_num, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            match = always_forbidden_toolchains.search(line)
            if match:
                violations.append(
                    f"{file_path}:{line_num} [toolchain_word]: matched {match.group(0)!r}"
                )

        tree = ast.parse(source, filename=str(file_path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.strip().lower() in curated_exact_literals
            ):
                violations.append(f"{file_path}:{node.lineno} [toolchain_literal]: {node.value!r}")

    assert not violations, "Rule H5 toolchain violations:\n" + "\n".join(violations)
