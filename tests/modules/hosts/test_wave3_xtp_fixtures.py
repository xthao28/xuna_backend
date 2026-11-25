"""Wave 3 (ADR-0018) golden fixtures: byte parity with the desktop copy and parsing
by the backend's own wire models (`exec_fs_frames.json`, `policy_frames.json`, the
`exec_async` frames of `watch_frames.json`)."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from xuna_backend.modules.hosts.api.schemas import SetHostPolicyRequest
from xuna_backend.modules.hosts.application.xtp_frames import (
    WatchEventParamsModel,
    WatchRegisterParamsModel,
    WatchRegisterResultModel,
    parse_preflight_result,
)
from xuna_backend.modules.hosts.domain.entities import RiskTier, ToolDescriptor, ToolFamily

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "xtp"
DESKTOP = (
    Path(__file__).resolve().parents[4]
    / "desktop"
    / "packages"
    / "xuna_xtp"
    / "test"
    / "fixtures"
    / "xtp"
)


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


@pytest.mark.parametrize("name", ["exec_fs_frames.json", "policy_frames.json", "watch_frames.json"])
def test_fixture_is_byte_identical_with_desktop(name: str) -> None:
    desktop = DESKTOP / name
    if not desktop.exists():
        pytest.skip("desktop checkout not present")
    assert (FIXTURES / name).read_bytes() == desktop.read_bytes()


def test_exec_fs_descriptors_parse_with_the_right_families() -> None:
    tools = _load("exec_fs_frames.json")["frames"]["toolsListExecFsResponse"]["result"]["tools"]
    parsed = [ToolDescriptor.from_wire(t) for t in tools]
    assert {d.name for d in parsed} == {
        "fs.list_files",
        "fs.read_file",
        "fs.grep",
        "fs.write_file",
        "fs.patch_file",
        "exec.run",
        "exec.status",
        "exec.write_stdin",
        "exec.cancel",
    }
    for d in parsed:
        expected = ToolFamily.EXEC if d.name.startswith("exec.") else ToolFamily.FILES
        assert d.family == expected
    run = next(d for d in parsed if d.name == "exec.run")
    assert run.risk == RiskTier.R2


def test_preflight_frames_parse_and_keep_host_risk() -> None:
    frames = _load("exec_fs_frames.json")["frames"]
    r2 = parse_preflight_result(frames["preflightExecRunR2Response"]["result"])
    assert r2.status == "ok" and r2.effective_risk == RiskTier.R2
    r3 = parse_preflight_result(frames["preflightExecRunR3Response"]["result"])
    assert r3.effective_risk == RiskTier.R3 and r3.reason_code == "network_command"
    for key, code in (
        ("preflightExecRunHardDeniedResponse", "exec_hard_denied"),
        ("preflightExecRunEnvDumpResponse", "exec_env_dump_forbidden"),
        ("preflightShellNotEnabledResponse", "shell_not_enabled"),
    ):
        refused = parse_preflight_result(frames[key]["result"])
        assert refused.status == "denied" and refused.error_code == code
        assert refused.effective_risk is None


def test_policy_frames_reject_remote_shell_control() -> None:
    fixture = _load("policy_frames.json")
    ok = fixture["frames"]["setPolicyRequest"]["params"]
    assert SetHostPolicyRequest.model_validate(ok).trustPreset == "autopilot"
    for spec in fixture["invalidFrames"].values():
        with pytest.raises(ValidationError):
            SetHostPolicyRequest.model_validate(spec["frame"]["params"])
    disabled = fixture["frames"]["getPolicyShellDisabledResponse"]["result"]
    assert disabled["shellEnabled"] is False
    assert "workspaceRoots" not in json.dumps(fixture["frames"])


def test_exec_async_watch_frames_validate() -> None:
    fixture = _load("watch_frames.json")
    frames = fixture["frames"]
    params = WatchRegisterParamsModel.model_validate(
        frames["watchRegisterExecAsyncRequest"]["params"]
    )
    assert params.mode == "exec_async" and params.requireInitialChange is False
    assert params.target.execTaskId == "exec_7f3a9c"
    WatchRegisterResultModel.model_validate(
        frames["watchRegisterExecAsyncAcceptedResponse"]["result"]
    )
    WatchEventParamsModel.model_validate(frames["watchEventExecAsyncSettledNotification"]["params"])
    bad = fixture["invalidFrames"]["watchRegisterExecAsyncInvalidTaskId"]["frame"]["params"]
    with pytest.raises(ValidationError):
        WatchRegisterParamsModel.model_validate(bad)
