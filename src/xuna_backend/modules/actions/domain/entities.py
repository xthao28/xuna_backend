"""Domain entities, value objects, and stream events for `actions` (UDC-4, UDC-7).

Stdlib only. Summaries and gesture mappings live in `action_summary.py` and are
re-exported here for backwards compatibility.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from xuna_backend.modules.actions.domain.action_summary import (
    _BOARD_KEY_LABELS_EN,
    _BOARD_KEY_LABELS_VI,
    ActionSummary,
    gesture_for_action_status,
)
from xuna_backend.modules.hosts.domain.entities import HostId, RiskTier
from xuna_backend.modules.hosts.domain.twin import HostFocusChangedNotice

__all__ = [
    "APPROVAL_TTL_SECONDS",
    "CONVERSATION_SCOPE_TTL",
    "MAX_CALLS_PER_ROUND",
    "MAX_TOOL_OUTPUT_BYTES",
    "MAX_TOOL_ROUNDS",
    "TURN_BUDGET_SECONDS",
    "_BOARD_KEY_LABELS_EN",
    "_BOARD_KEY_LABELS_VI",
    "ActionPendingEvent",
    "ActionResultEvent",
    "ActionStreamEvent",
    "ActionSummary",
    "ActionTargetRequiredEvent",
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalScope",
    "ApprovalState",
    "BoardHidRequestEvent",
    "EmergencyStopSummary",
    "HostFocusChangedNotice",
    "HostWakeRequestEvent",
    "LocalActionReport",
    "LocalActionRequestEvent",
    "gesture_for_action_status",
]

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

APPROVAL_TTL_SECONDS = 60
CONVERSATION_SCOPE_TTL = timedelta(minutes=15)
# A UI snapshot (compactUiTree, up to 3600 chars) plus status must fit in one tool message.
MAX_TOOL_OUTPUT_BYTES = 6144
# Multi-step host control (open app -> inspect -> act -> verify) needs several rounds;
# TURN_BUDGET_SECONDS remains the hard stop.
MAX_TOOL_ROUNDS = 12
MAX_CALLS_PER_ROUND = 3
TURN_BUDGET_SECONDS = 150.0


class ApprovalState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class ApprovalScope(StrEnum):
    ONCE = "once"
    CONVERSATION_15M = "conversation_15m"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Owner approval request bound to `args_sha256` (no raw arguments stored)."""

    approval_id: uuid.UUID
    owner_id: uuid.UUID
    host_id: HostId | None
    target_alias: str
    tool: str
    args_sha256: str
    risk: RiskTier
    summary: str
    created_at: datetime
    expires_at: datetime
    state: ApprovalState = ApprovalState.PENDING
    decided_at: datetime | None = None
    scope: ApprovalScope = ApprovalScope.ONCE
    signature: bytes | None = None
    signer_public_key: bytes | None = None

    def __post_init__(self) -> None:
        if not _SHA256_HEX_RE.match(self.args_sha256):
            raise ValueError("args_sha256 must be 64 lowercase hex characters")
        if not 1 <= len(self.target_alias) <= 32:
            raise ValueError("target_alias must be 1..32 characters")
        if not 1 <= len(self.summary) <= 240:
            raise ValueError("summary must be 1..240 characters")
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        if self.decided_at is not None and self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")
        if self.signature is not None and len(self.signature) != 64:
            raise ValueError("signature must be 64 bytes")
        if self.signer_public_key is not None and (
            len(self.signer_public_key) != 65 or self.signer_public_key[0] != 0x04
        ):
            raise ValueError("signer_public_key must be 65-byte uncompressed P-256 point")

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


@dataclass(frozen=True, slots=True)
class EmergencyStopSummary:
    stopped: bool
    hosts_notified: int
    approvals_cancelled: int
    grants_suspended: int
    host_id: str | None = None


@dataclass(frozen=True, slots=True)
class ActionPendingEvent:
    """SSE / WebSocket event emitted when an R2+ or board action awaits owner approval."""

    approval_id: str
    target: str
    summary: str
    risk: str
    expires_in: int
    host_id: str | None = None
    args_sha256: str = ""
    expires_at: str = ""
    # Present only when this exact call may be pre-approved (ADR-0014 D1): the tool name and
    # the validated arguments, so the owner's phone can sign a standing approval for them.
    standing_tool: str | None = None
    standing_arguments: dict[str, object] | None = None
    # UDC-17 S17.5: set with the standing offer when this exact call was approved at least
    # `STANDING_SUGGEST_MIN_APPROVALS` times in the window, so the phone can suggest it.
    standing_approved_count: int | None = None
    # Additive metadata for Wave E Mission Card UX (UDC-18). None for standard 1-step approvals.
    mission_envelope: dict[str, object] | None = None
    target_reason: str | None = None
    target_platform: str | None = None
    battery_advisories: tuple[str, ...] = ()

    def standing_payload(self) -> dict[str, object] | None:
        """The `standing` object of `action_pending` on SSE and the voice WebSocket, or
        `None` when this call cannot be pre-approved."""
        if self.standing_tool is None:
            return None
        payload: dict[str, object] = {
            "tool": self.standing_tool,
            "arguments": self.standing_arguments or {},
        }
        if self.standing_approved_count is not None:
            payload["approvedCount"] = self.standing_approved_count
        return payload

    def wire_payload(self) -> dict[str, object]:
        """Wire JSON payload for `action_pending` on SSE and the voice WebSocket.

        Serializes `risk` in lowercase (`r1`..`r3`) to match mobile `ActionEventMapper`.
        """
        payload: dict[str, object] = {
            "approvalId": self.approval_id,
            "target": self.target,
            "summary": self.summary,
            "risk": self.risk.lower(),
            "expiresIn": self.expires_in,
        }
        if self.host_id is not None:
            payload["hostId"] = self.host_id
        if self.args_sha256 is not None:
            payload["argsSha256"] = self.args_sha256
        if self.expires_at is not None:
            payload["expiresAt"] = self.expires_at
        standing = self.standing_payload()
        if standing is not None:
            payload["standing"] = standing
        if self.mission_envelope is not None:
            payload["missionEnvelope"] = self.mission_envelope
        if self.target_reason is not None:
            payload["targetReason"] = self.target_reason
        if self.target_platform is not None:
            payload["targetPlatform"] = self.target_platform
        if self.battery_advisories:
            payload["batteryAdvisories"] = list(self.battery_advisories)
        return payload


@dataclass(frozen=True, slots=True)
class BoardHidRequestEvent:
    """SSE / WebSocket event asking the mobile app to send BLE Opcode 12 (`source = 2`)."""

    invocation_id: str
    approval_id: str
    key: str
    usage_index: int


@dataclass(frozen=True, slots=True)
class LocalActionRequestEvent:
    """SSE / WebSocket event asking the same-device app to execute a host tool (`UDC-5` §5.2)."""

    invocation_id: str
    approval_id: str | None
    target: str
    tool: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class LocalActionReport:
    """Client-reported result for `board_hid_request` or `local_action_request` (`UDC-5` §5.2)."""

    status: str
    output: dict[str, object] | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class HostWakeRequestEvent:
    """SSE / WebSocket event asking the mobile app to send a local-only WoL
    Magic Packet (`UDC-11` S11.1)."""

    invocation_id: str
    host_id: str
    target: str
    mac_address: str | None = None


@dataclass(frozen=True, slots=True)
class ActionTargetRequiredEvent:
    """SSE / WebSocket event emitted when target resolution requires owner choice or repair."""

    invocation_id: str
    tool: str
    reason_code: str
    target_reason: str
    candidates: tuple[dict[str, object], ...] = ()
    error_code: str | None = None

    def wire_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "invocationId": self.invocation_id,
            "tool": self.tool,
            "reasonCode": self.reason_code,
            "targetReason": self.target_reason,
            "candidates": list(self.candidates),
        }
        if self.error_code is not None:
            payload["errorCode"] = self.error_code
        return payload


@dataclass(frozen=True, slots=True)
class ActionResultEvent:
    """SSE / WebSocket event emitted when a tool invocation completes.

    `error_code` and `host_id` are set when the call did not run, so the app can offer the
    owner the place to fix it (UDC-17 S17.3, P7).
    """

    invocation_id: str
    target: str
    summary: str
    status: str
    gesture: str | None = None
    error_code: str | None = None
    host_id: str | None = None
    target_reason: str | None = None
    target_platform: str | None = None
    mission_envelope: dict[str, object] | None = None
    remediation: dict[str, object] | None = None

    def wire_payload(self) -> dict[str, object]:
        """Wire JSON payload for `action_result` on SSE and the voice WebSocket."""
        payload: dict[str, object] = {
            "invocationId": self.invocation_id,
            "target": self.target,
            "summary": self.summary,
            "status": self.status,
        }
        if self.gesture is not None:
            payload["gesture"] = self.gesture
        if self.error_code is not None:
            payload["errorCode"] = self.error_code
        if self.host_id is not None:
            payload["hostId"] = self.host_id
        if self.target_reason is not None:
            payload["targetReason"] = self.target_reason
        if self.target_platform is not None:
            payload["targetPlatform"] = self.target_platform
        if self.mission_envelope is not None:
            payload["missionEnvelope"] = self.mission_envelope
        if self.remediation is not None:
            payload["remediation"] = self.remediation
        return payload


@dataclass(frozen=True, slots=True)
class ComputerUseSessionEvent:
    """SSE / WebSocket `cu_session` event: the live state of a desktop control session for the
    phone's session card (UDC-17 S17.3).

    Carries the owner's own goal and the text typed since the last commit; it is streamed to
    the owner only and never persisted, logged or audited.
    """

    session_id: str
    host_id: str
    target: str
    goal: str
    apps: tuple[str, ...]
    domains: tuple[str, ...]
    step: int
    max_steps: int
    max_minutes: int
    seconds_left: int
    status: str  # "running" | "ended"
    summary: str
    typed: tuple[str, ...] = ()
    end_reason: str | None = None
    # Whether a rerun of this scope would fit its approval card (UDC-17 S17.5 follow-up).
    rerun_fits_card: bool | None = None


ActionStreamEvent = (
    ActionPendingEvent
    | ActionTargetRequiredEvent
    | BoardHidRequestEvent
    | LocalActionRequestEvent
    | HostWakeRequestEvent
    | ActionResultEvent
    | ComputerUseSessionEvent
    | HostFocusChangedNotice
)
