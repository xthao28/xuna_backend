"""Unit tests for `actions.domain.mission_envelope` (UDC-18 + Mesh §5.6.7/§7)."""

from __future__ import annotations

import pytest

from xuna_backend.modules.actions.domain.mission_envelope import (
    MissionEnvelope,
    MissionStep,
    MissionStepStatus,
)
from xuna_backend.modules.hosts.domain.entities import HostPlatform, RiskTier


def test_mission_envelope_preserves_udc18_fields_and_computes_highest_risk() -> None:
    step_r1 = MissionStep(
        host_id="host-1",
        host_alias="MacDesk",
        platform=HostPlatform.MACOS,
        tool="workspace.capture_capsule",
        summary="Capture workspace state",
        risk=RiskTier.R1,
        status=MissionStepStatus.OK,
    )
    step_r3 = MissionStep(
        host_id="host-2",
        host_alias="WinDesk",
        platform=HostPlatform.WINDOWS,
        tool="input.click",
        summary="Submit target action",
        risk=RiskTier.R3,
        status=MissionStepStatus.PENDING,
    )
    env = MissionEnvelope(
        mission_id="mission-123",
        title="Cross-host workflow",
        goal="Transfer work and commit",
        steps=(step_r1, step_r3),
        domains=("example.org",),
    )

    assert env.highest_risk() == RiskTier.R3
    wire = env.to_wire()
    assert wire["missionId"] == "mission-123"
    assert wire["title"] == "Cross-host workflow"
    assert wire["goal"] == "Transfer work and commit"
    assert wire["domains"] == ["example.org"]
    assert wire["apps"] == []
    assert wire["maxSteps"] == 2
    assert wire["maxMinutes"] == 2
    assert wire["highestRisk"] == "r3"
    assert wire["planSteps"] == ["Capture workspace state", "Submit target action"]
    steps_wire = wire["steps"]
    assert isinstance(steps_wire, list)
    assert steps_wire[0]["status"] == "ok"
    assert steps_wire[1]["risk"] == "r3"
    assert steps_wire[1]["status"] == "pending"

    updated = env.with_step_status(1, MissionStepStatus.OK)
    assert updated.to_wire()["steps"][1]["status"] == "ok"  # type: ignore[index]


def test_mission_envelope_rejects_empty_goal_or_title() -> None:
    with pytest.raises(ValueError, match="goal"):
        MissionEnvelope(mission_id="m1", title="Title", goal="   ")
    with pytest.raises(ValueError, match="summary"):
        MissionStep(
            host_id="h1",
            host_alias="Mac",
            platform=HostPlatform.MACOS,
            tool="url.open",
            summary="",
            risk=RiskTier.R2,
        )
