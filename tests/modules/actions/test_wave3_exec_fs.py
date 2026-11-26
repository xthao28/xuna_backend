"""Wave 3 (ADR-0018) backend tests: exec/fs families, mandatory host preflight,
catalog gating and priority, prompt capability blocks, exec_async watch binding,
approval-card text and the local-only `shellControl` policy boundary."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from pydantic import ValidationError
from tests.e2e.test_wave_c_e2e import (
    NOW,
    EventCollector,
    _FakeAuditLog,
    _FakeHostRepository,
    _FakeSessionHub,
    _make_action_context,
    _make_host,
)

from xuna_backend.modules.actions.application.catalog import (
    _PRIORITY_ORDER,
    MAX_CATALOG_TOOLS,
    is_tool_granted,
)
from xuna_backend.modules.actions.application.exec_fs_support import (
    EXEC_FS_PREFLIGHT_TOOLS,
    bind_exec_async_target,
    exec_async_verify_instruction,
)
from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.application.prompts import tool_system_prompt
from xuna_backend.modules.actions.domain.action_summary import ActionSummary
from xuna_backend.modules.actions.domain.tool_names import from_function_name, to_function_name
from xuna_backend.modules.actions.domain.watch import (
    JOB_WAIT_SCHEMA,
    WatchMode,
    WatchObservedSignal,
    WatchTarget,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.api.schemas import GrantItemInput, SetHostPolicyRequest
from xuna_backend.modules.hosts.application.xtp_frames import (
    WatchEventParamsModel,
    WatchTargetModel,
)
from xuna_backend.modules.hosts.domain.entities import (
    FORBIDDEN_TOOLS,
    Grant,
    HostPlatform,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
    validate_tool_name,
)


def _descriptor(
    name: str,
    family: ToolFamily,
    risk: RiskTier,
    *,
    permission: ToolPermissionState = ToolPermissionState.GRANTED,
    properties: dict[str, object] | None = None,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Tool {name}",
        input_schema={
            "type": "object",
            "properties": properties or {},
            "additionalProperties": False,
        },
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=family,
        risk=risk,
        permission=permission,
        policy_version=1,
    )


_EXEC_RUN_PROPS: dict[str, object] = {
    "command": {"type": "string", "minLength": 1, "maxLength": 4096},
    "cwd": {"type": "string", "minLength": 1, "maxLength": 512},
}


class _PreflightHub(_FakeSessionHub):
    """Fake hub whose `xuna/preflight` answer is scripted per test."""

    def __init__(self, preflight: dict[str, object] | None) -> None:
        super().__init__()
        self.preflight = preflight

    async def send_request(
        self, host_id: str, method: str, params: dict[str, object], timeout_s: float
    ) -> dict[str, object]:
        if method == "xuna/preflight":
            self.sent_requests.append((host_id, method, params, timeout_s))
            if self.preflight is None:
                raise RuntimeError("rpc_error:-32601: Method not found")
            return {"preflight": self.preflight}
        return await super().send_request(host_id, method, params, timeout_s)


def _orchestrator(
    preflight: dict[str, object] | None,
    *,
    exec_grant: RiskTier = RiskTier.R3,
    platform: HostPlatform = HostPlatform.MACOS,
) -> tuple[Orchestrator, _PreflightHub, uuid.UUID]:
    owner_id = uuid.uuid4()
    host_repo = _FakeHostRepository()
    hub = _PreflightHub(preflight)
    host = _make_host(owner_id, platform=platform)
    host_repo.hosts[host.host_id] = host
    host_repo.manifests[host.host_id] = Manifest(
        host_id=host.host_id,
        tools=(_descriptor("exec.run", ToolFamily.EXEC, RiskTier.R2, properties=_EXEC_RUN_PROPS),),
        updated_at=NOW,
        schema_version=1,
    )
    host_repo.grants[host.host_id] = [
        Grant(
            host_id=host.host_id,
            family=ToolFamily.EXEC,
            max_risk=exec_grant,
            enabled=True,
            updated_at=NOW,
        )
    ]
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


def _tool_calls(hub: _PreflightHub) -> list[dict[str, Any]]:
    return [c for c in hub.tool_calls if c["tool"] == "exec.run"]


# --- families, names, forbidden prefixes ------------------------------------------------


def test_exec_and_fs_resolve_to_their_families_and_raw_shell_stays_forbidden() -> None:
    assert validate_tool_name("exec.run") == ToolFamily.EXEC
    assert validate_tool_name("fs.patch_file") == ToolFamily.FILES
    assert to_function_name("exec.run") == "exec__run"
    assert from_function_name("exec__run") == "exec.run"
    assert from_function_name("fs__read_file") == "fs.read_file"
    for raw in ("shell.exec", "bash.run", "powershell.exec", "osascript.run", "cmd.exec"):
        assert raw in FORBIDDEN_TOOLS
        assert from_function_name(raw.replace(".", "__")) is None
        with pytest.raises(ValueError):
            validate_tool_name(raw)
    assert GrantItemInput(family="exec", maxRisk="R3", enabled=True).parsed_family() == (
        ToolFamily.EXEC
    )


# --- local-only shellControl ------------------------------------------------------------


@pytest.mark.parametrize("key", ["shellControl", "shell_control", "shellEnabled"])
def test_remote_policy_write_rejects_shell_control(key: str) -> None:
    with pytest.raises(ValidationError):
        SetHostPolicyRequest.model_validate(
            {"schemaVersion": 1, "trustPreset": "autopilot", key: {"enabled": True}}
        )
    ok = SetHostPolicyRequest.model_validate({"schemaVersion": 1, "trustPreset": "standard"})
    assert "shellControl" not in ok.to_xtp_payload()


# --- mandatory preflight in the orchestrator ------------------------------------------


@pytest.mark.anyio
async def test_exec_run_is_refused_when_the_host_cannot_preflight() -> None:
    orch, hub, owner_id = _orchestrator(preflight=None)
    collector = EventCollector(orch, owner_id)
    msg = await orch.handle_call(
        ToolCall(id="c1", name="exec__run", arguments={"command": "npm test"}),
        _make_action_context(owner_id, collector),
    )
    assert "preflight_required" in msg.content
    assert collector.pending_events == []
    assert _tool_calls(hub) == []


@pytest.mark.anyio
async def test_non_desktop_hosts_are_refused_without_a_backend_estimate() -> None:
    orch, hub, owner_id = _orchestrator(
        preflight={"status": "ok", "effectiveRisk": "R2"}, platform=HostPlatform.LINUX
    )
    collector = EventCollector(orch, owner_id)
    msg = await orch.handle_call(
        ToolCall(id="c2", name="exec__run", arguments={"command": "npm test"}),
        _make_action_context(owner_id, collector),
    )
    assert "preflight_required" in msg.content
    assert _tool_calls(hub) == []


@pytest.mark.anyio
async def test_host_hard_deny_is_returned_to_the_model_without_asking_the_owner() -> None:
    orch, hub, owner_id = _orchestrator(
        preflight={"status": "denied", "errorCode": "exec_hard_denied"}
    )
    collector = EventCollector(orch, owner_id)
    msg = await orch.handle_call(
        ToolCall(id="c3", name="exec__run", arguments={"command": "sudo ls"}),
        _make_action_context(owner_id, collector),
    )
    assert "exec_hard_denied" in msg.content
    assert collector.pending_events == []
    assert _tool_calls(hub) == []


@pytest.mark.anyio
async def test_host_r3_raises_the_card_and_is_never_lowered() -> None:
    orch, hub, owner_id = _orchestrator(
        preflight={"status": "ok", "effectiveRisk": "R3", "reasonCode": "network_command"}
    )
    collector = EventCollector(orch, owner_id, auto_decision=None)
    ctx = _make_action_context(owner_id, collector)
    call = ToolCall(
        id="c4", name="exec__run", arguments={"command": "git push origin dev", "cwd": "~/p"}
    )
    task = asyncio.create_task(orch.handle_call(call, ctx))
    for _ in range(200):
        if collector.pending_events:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    assert len(collector.pending_events) == 1
    pending = collector.pending_events[0]
    assert pending.risk.lower() == "r3"
    assert "git push origin dev" in pending.summary
    assert _tool_calls(hub) == []


@pytest.mark.anyio
async def test_r3_above_the_exec_grant_is_refused() -> None:
    orch, hub, owner_id = _orchestrator(
        preflight={"status": "ok", "effectiveRisk": "R3"}, exec_grant=RiskTier.R2
    )
    collector = EventCollector(orch, owner_id)
    msg = await orch.handle_call(
        ToolCall(id="c5", name="exec__run", arguments={"command": "rm notes.txt"}),
        _make_action_context(owner_id, collector),
    )
    assert collector.pending_events == []
    assert _tool_calls(hub) == []
    assert "denied" in msg.content


def test_mandatory_preflight_set_matches_the_contract() -> None:
    assert (
        frozenset(
            {
                "draft.apply",
                "exec.run",
                "exec.write_stdin",
                "fs.write_file",
                "fs.patch_file",
            }
        )
        == EXEC_FS_PREFLIGHT_TOOLS
    )


# --- catalog ----------------------------------------------------------------------------


def _grant(family: ToolFamily, risk: RiskTier = RiskTier.R3) -> Grant:
    return Grant(
        host_id=_make_host(uuid.uuid4()).host_id,
        family=family,
        max_risk=risk,
        enabled=True,
        updated_at=NOW,
    )


def test_exec_tools_are_eligible_only_when_the_host_declares_shell_enabled() -> None:
    grants = [_grant(ToolFamily.EXEC)]
    disabled = _descriptor(
        "exec.run", ToolFamily.EXEC, RiskTier.R2, permission=ToolPermissionState.DENIED
    )
    enabled = _descriptor("exec.run", ToolFamily.EXEC, RiskTier.R2)
    assert not is_tool_granted(disabled, grants, NOW)
    assert is_tool_granted(enabled, grants, NOW)
    assert not is_tool_granted(enabled, [_grant(ToolFamily.FILES)], NOW)


def test_dev_tools_sit_in_direct_slots_without_moving_cu_invariants() -> None:
    assert MAX_CATALOG_TOOLS == 36
    direct = _PRIORITY_ORDER[:30]
    for name in (
        "fs.read_file",
        "fs.grep",
        "fs.list_files",
        "fs.patch_file",
        "fs.write_file",
        "exec.run",
        "exec.status",
        "exec.cancel",
        "workspace.coding_agent",
    ):
        assert name in direct, name
    assert _PRIORITY_ORDER[20] == "cu.session_start"
    assert _PRIORITY_ORDER[29] == "cu.session_end"
    assert _PRIORITY_ORDER[30:33] == ("mcp.task_result", "mcp.task_cancel", "cu.session_extend")
    assert len(set(_PRIORITY_ORDER)) == len(_PRIORITY_ORDER)


# --- prompts ------------------------------------------------------------------------------


def test_dev_loop_and_job_blocks_appear_only_with_their_tools() -> None:
    base = tool_system_prompt(Locale.EN, {"media__play_pause"})
    assert "exec_async" not in base
    assert "autonomous_job" not in base
    dev = tool_system_prompt(Locale.EN, {"exec__run", "fs__read_file"})
    assert "exec_async" in dev
    assert "fs__patch_file" in dev
    vi = tool_system_prompt(Locale.VI, {"exec__run", "autonomous_job"})
    assert "exec_async" in vi
    assert "autonomous_job" in vi


# --- exec_async watch ---------------------------------------------------------------------


def test_exec_async_binding_requires_an_explicit_task_id() -> None:
    target, err = bind_exec_async_target("exec_7f3a9c")
    assert err is None and target == WatchTarget(exec_task_id="exec_7f3a9c")
    assert target.to_wire_dict() == {"execTaskId": "exec_7f3a9c"}
    assert WatchTarget.from_dict({"execTaskId": "exec_7f3a9c"}) == target
    assert bind_exec_async_target(None) == (None, "watch_unsupported")
    assert bind_exec_async_target("rm -rf /") == (None, "invalid_args")
    with pytest.raises(ValueError):
        WatchTarget(exec_task_id="../etc")
    assert WatchMode("exec_async") is WatchMode.EXEC_ASYNC
    assert WatchObservedSignal("exec_async") is WatchObservedSignal.EXEC_ASYNC
    props = JOB_WAIT_SCHEMA["properties"]
    assert isinstance(props, dict) and "execTaskId" in props
    assert WatchTargetModel(execTaskId="exec_abc").execTaskId == "exec_abc"
    with pytest.raises(ValidationError):
        WatchTargetModel(execTaskId="not a task")
    WatchEventParamsModel.model_validate(
        {
            "schemaVersion": 1,
            "watchId": "f1111111-1111-4111-8111-111111111117",
            "jobId": "a2222222-2222-4222-8222-222222222222",
            "kind": "settled",
            "observedSignals": ["exec_async"],
            "elapsedSeconds": 3,
            "changeCount": 1,
            "digestChanged": True,
            "occurredAt": "2026-11-25T12:00:00Z",
        }
    )
    hint = exec_async_verify_instruction("exec_7f3a9c")
    assert "exec__status" in hint and "exec_7f3a9c" in hint


# --- approval card ------------------------------------------------------------------------


def test_approval_card_shows_the_exact_command_and_cwd() -> None:
    vi = ActionSummary.format_pending(
        "exec.run", {"command": "curl -s https://example.com", "cwd": "~/app"}, "Mac"
    )
    assert "curl -s https://example.com" in vi and "~/app" in vi
    en = ActionSummary.format_pending(
        "fs.write_file", {"path": "~/.zshrc", "content": "x"}, "Mac", Locale.EN
    )
    assert "~/.zshrc" in en
    long_cmd = "echo " + "a" * 400
    assert len(ActionSummary.format_pending("exec.run", {"command": long_cmd}, "Mac")) <= 240
