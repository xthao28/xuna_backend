"""Pure-stdlib `MissionEnvelope` builder and step aggregator (Mesh §5.6.7, §7).

Preserves all UDC-18 fields (`goal` non-empty, `apps`, `domains`, `maxSteps`,
`maxMinutes`, `planSteps`) alongside additive Phase 3 spatial fields (`missionId`,
`title`, `steps`, `highestRisk`).
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from xuna_backend.modules.hosts.domain.entities import HostPlatform, RiskTier


class MissionStepStatus(StrEnum):
    """Lifecycle status of an individual step inside `MissionEnvelope.steps`."""

    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"
    DEFERRED_WAKE = "deferred_wake"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class MissionStep:
    """A single host-targeted step grouped under a spatial `MissionEnvelope`."""

    host_id: str
    host_alias: str
    platform: HostPlatform
    tool: str
    summary: str
    risk: RiskTier
    status: MissionStepStatus = MissionStepStatus.PENDING

    def __post_init__(self) -> None:
        if not self.host_id.strip():
            raise ValueError("mission step host_id must be non-empty")
        if not self.host_alias.strip():
            raise ValueError("mission step host_alias must be non-empty")
        if not self.tool.strip():
            raise ValueError("mission step tool must be non-empty")
        if not 1 <= len(self.summary.strip()) <= 240:
            raise ValueError("mission step summary must be 1..240 chars")

    def to_wire(self) -> dict[str, object]:
        return {
            "hostId": self.host_id,
            "hostAlias": self.host_alias,
            "platform": self.platform.value,
            "tool": self.tool,
            "summary": self.summary,
            "risk": self.risk.value.lower(),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class MissionEnvelope:
    """Unified Mission Card metadata grouping cross-host steps (UDC-18 + Mesh §7)."""

    mission_id: str
    title: str
    goal: str
    steps: tuple[MissionStep, ...] = ()
    apps: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    max_steps: int | None = None
    max_minutes: int = 2
    plan_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.mission_id.strip():
            raise ValueError("mission_id must be non-empty")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.goal.strip():
            raise ValueError("goal must be non-empty")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_minutes < 1:
            raise ValueError("max_minutes must be >= 1")

    def highest_risk(self) -> RiskTier:
        """Returns the highest risk tier across all grouped steps (defaults to R1)."""
        if not self.steps:
            return RiskTier.R1
        return max(self.steps, key=lambda step: step.risk.level).risk

    def with_step_status(
        self,
        index: int,
        status: MissionStepStatus,
        *,
        summary: str | None = None,
    ) -> "MissionEnvelope":
        """Returns a new `MissionEnvelope` with step `index` updated."""
        if not 0 <= index < len(self.steps):
            raise IndexError("mission step index out of range")
        mutable = list(self.steps)
        current = mutable[index]
        mutable[index] = replace(
            current,
            status=status,
            summary=summary[:240] if summary is not None else current.summary,
        )
        return replace(self, steps=tuple(mutable))

    def to_wire(self) -> dict[str, object]:
        """Serializes both UDC-18 base keys and Phase 3 spatial step keys."""
        eff_max_steps = self.max_steps if self.max_steps is not None else max(len(self.steps), 1)
        eff_plan = (
            list(self.plan_steps) if self.plan_steps else [step.summary for step in self.steps]
        )
        return {
            "missionId": self.mission_id,
            "title": self.title,
            "goal": self.goal,
            "apps": list(self.apps),
            "domains": list(self.domains),
            "maxSteps": eff_max_steps,
            "maxMinutes": self.max_minutes,
            "planSteps": eff_plan,
            "steps": [step.to_wire() for step in self.steps],
            "highestRisk": self.highest_risk().value.lower(),
        }


def build_cu_session_mission_dict(
    *,
    goal: str,
    apps: Sequence[str] = (),
    domains: Sequence[str] = (),
    max_steps: int = 12,
    max_minutes: int = 5,
    plan_steps: Sequence[str] | None = None,
) -> dict[str, object] | None:
    """Builds a validated CU `MissionEnvelope` wire dict, returning `None` if `goal` is blank."""
    clean_goal = goal.strip()
    if not clean_goal:
        return None
    payload: dict[str, object] = {
        "goal": clean_goal,
        "apps": [a for a in (str(x).strip() for x in apps) if a],
        "domains": [d for d in (str(x).strip() for x in domains) if d],
        "maxSteps": max(max_steps, 1),
        "maxMinutes": max(max_minutes, 1),
    }
    if plan_steps is not None:
        payload["planSteps"] = [s for s in (str(x).strip() for x in plan_steps) if s]
    return payload
