"""Pure domain entities, validation rules, and schemas for Async Watch & Wake.

Stdlib-only, framework-free, and strictly app-agnostic (Rule H5).
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from xuna_backend.modules.hosts.domain.entities import HostId

MAX_WATCH_WAIT_SECONDS = 900
MIN_WATCH_WAIT_SECONDS = 5
JOB_WAIT_MIN_WAIT_SECONDS = 10
DEFAULT_WATCH_WAIT_SECONDS = 300
DEFAULT_QUIESCENCE_SECONDS = 8
MIN_QUIESCENCE_SECONDS = 2
MAX_QUIESCENCE_SECONDS = 120
BACKOFF_QUIESCENCE_CAP_SECONDS = 60
DEFAULT_MAX_WAKE_CHECKS = 8
HARD_MAX_WAKE_CHECKS = 16
MAX_WAKE_INBOX_TITLE_CHARS = 120
MAX_WAKE_INBOX_SUMMARY_CHARS = 2000

_SHORT_CODE_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")
_WORKSPACE_ALIAS_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


class WatchMode(StrEnum):
    """OS-generic watch observation modes (`XUNA_XTP_V1_CONTRACT.md` §4.2.14)."""

    UI_QUIESCENCE = "ui_quiescence"
    UI_CHANGE = "ui_change"
    PROCESS_IDLE = "process_idle"
    FS_QUIESCENCE = "fs_quiescence"
    CDP_IDLE = "cdp_idle"
    COMPOSITE = "composite"
    TIMER = "timer"
    PROTOCOL_ASYNC = "protocol_async"


class WatchCompositeSignal(StrEnum):
    """Primitive signals allowed inside `composite` watch mode."""

    UI_QUIESCENCE = "ui_quiescence"
    PROCESS_IDLE = "process_idle"
    FS_QUIESCENCE = "fs_quiescence"
    CDP_IDLE = "cdp_idle"


class WatchObservedSignal(StrEnum):
    """Signals reported by host watch registration and events."""

    UI_QUIESCENCE = "ui_quiescence"
    UI_CHANGE = "ui_change"
    PROCESS_IDLE = "process_idle"
    FS_QUIESCENCE = "fs_quiescence"
    CDP_IDLE = "cdp_idle"
    TIMER = "timer"
    PROTOCOL_ASYNC = "protocol_async"


class ContinuationMode(StrEnum):
    """Behavior when a watched task wakes and passes verification."""

    REPORT_ONLY = "report_only"
    REPORT_AND_ASK = "report_and_ask"
    AUTO_CONTINUE = "auto_continue"


class WatchStatus(StrEnum):
    """Lifecycle status of a persisted `JobWatch` record."""

    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    FAILED = "failed"


class WatchEventKind(StrEnum):
    """Event kinds emitted by `xuna/watchEvent`."""

    SETTLED = "settled"
    CHANGED = "changed"
    TIMEOUT = "timeout"
    TARGET_LOST = "target_lost"
    ERROR = "error"


class WatchCancelReason(StrEnum):
    """Cancellation reasons for `xuna/watchCancel`."""

    OWNER_CANCEL = "owner_cancel"
    ESTOP = "estop"
    JOB_ENDED = "job_ended"
    REPLACED = "replaced"
    EXPIRED = "expired"


class WakeInboxKind(StrEnum):
    """Notification categories persisted in `wake_inbox_items`."""

    JOB_REPORT = "job_report"
    JOB_NEEDS_DECISION = "job_needs_decision"
    JOB_TIMEOUT = "job_timeout"
    JOB_FAILED = "job_failed"
    WATCH_ALERT = "watch_alert"


class WakeSuggestedAction(StrEnum):
    """Typed action chips displayed in Mobile Wake Inbox cards."""

    EXTEND_WATCH = "extend_watch"
    CONTINUE_JOB = "continue_job"
    OPEN_APPROVAL = "open_approval"
    DISMISS = "dismiss"


class WatchGestureCue(StrEnum):
    """Closed Track A expression cue hints for Wake Inbox items."""

    LOOK_AROUND = "look_around"
    BOUNCE_PLAY = "bounce_play"
    GREET_NUZZLE = "greet_nuzzle"


JOB_WAIT_FUNCTION_NAME = "job__wait"
JOB_WAIT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": [m.value for m in WatchMode],
        },
        "signals": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [s.value for s in WatchCompositeSignal],
            },
            "minItems": 2,
            "maxItems": 3,
        },
        "quiescenceSeconds": {
            "type": "integer",
            "minimum": MIN_QUIESCENCE_SECONDS,
            "maximum": MAX_QUIESCENCE_SECONDS,
            "description": "Stability window in seconds (default 8).",
        },
        "maxWaitSeconds": {
            "type": "integer",
            "minimum": JOB_WAIT_MIN_WAIT_SECONDS,
            "maximum": MAX_WATCH_WAIT_SECONDS,
            "description": "Hard watch timeout in seconds (default 300, max 900).",
        },
        "doneCriteria": {
            "type": "string",
            "minLength": 1,
            "maxLength": 240,
            "description": "Observable completion condition to verify upon wake.",
        },
        "continuationMode": {
            "type": "string",
            "enum": [c.value for c in ContinuationMode],
            "description": "Post-wake behavior (default report_and_ask).",
        },
        "workspaceAlias": {
            "type": "string",
            "minLength": 1,
            "maxLength": 64,
            "description": "Normalized workspace alias for filesystem watch modes.",
        },
    },
    "required": ["mode", "doneCriteria"],
    "additionalProperties": False,
}


def compute_backoff_quiescence(prev_seconds: int) -> int:
    """Computes `NO_ACTION` exponential backoff bounded to `[2, 60]` seconds."""
    scaled = int(prev_seconds * 1.5)
    return min(max(scaled, MIN_QUIESCENCE_SECONDS), BACKOFF_QUIESCENCE_CAP_SECONDS)


def validate_workspace_alias(value: str) -> str:
    """Validates a normalized workspace alias, rejecting paths and traversal tokens."""
    if (
        not value
        or "\n" in value
        or "\r" in value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
        or ".." in value
        or not _WORKSPACE_ALIAS_RE.fullmatch(value)
    ):
        raise ValueError("invalid workspaceAlias")
    return value


def _validate_control_free(value: str, *, max_len: int = 128) -> str:
    if not 1 <= len(value) <= max_len:
        raise ValueError("identifier length out of bounds")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError("control characters are not allowed")
    return value


@dataclass(frozen=True, slots=True)
class WatchTarget:
    """Normalized, content-free target specification for a host watch."""

    pid: int | None = None
    window_id: int | None = None
    app_id: str | None = None
    workspace_alias: str | None = None
    tab_id: str | None = None
    mcp_server_id: str | None = None
    protocol_task_id: str | None = None

    def __post_init__(self) -> None:
        if self.pid is not None and (
            isinstance(self.pid, bool) or not 1 <= self.pid <= 2_147_483_647
        ):
            raise ValueError("pid out of range")
        if self.window_id is not None and (
            isinstance(self.window_id, bool) or not 0 <= self.window_id <= 2_147_483_647
        ):
            raise ValueError("windowId out of range")
        if self.app_id is not None and (
            "\n" in self.app_id or "\r" in self.app_id or not _SHORT_CODE_RE.fullmatch(self.app_id)
        ):
            raise ValueError("invalid appId")
        if self.workspace_alias is not None:
            validate_workspace_alias(self.workspace_alias)
        if self.tab_id is not None:
            _validate_control_free(self.tab_id, max_len=128)
        if self.mcp_server_id is not None and (
            "\n" in self.mcp_server_id
            or "\r" in self.mcp_server_id
            or not _SHORT_CODE_RE.fullmatch(self.mcp_server_id)
        ):
            raise ValueError("invalid mcpServerId")
        if self.protocol_task_id is not None:
            _validate_control_free(self.protocol_task_id, max_len=128)

    def to_wire_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.pid is not None:
            payload["pid"] = self.pid
        if self.window_id is not None:
            payload["windowId"] = self.window_id
        if self.app_id is not None:
            payload["appId"] = self.app_id
        if self.workspace_alias is not None:
            payload["workspaceAlias"] = self.workspace_alias
        if self.tab_id is not None:
            payload["tabId"] = self.tab_id
        if self.mcp_server_id is not None:
            payload["mcpServerId"] = self.mcp_server_id
        if self.protocol_task_id is not None:
            payload["protocolTaskId"] = self.protocol_task_id
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchTarget":
        return cls(
            pid=data.get("pid"),
            window_id=data.get("windowId") if "windowId" in data else data.get("window_id"),
            app_id=data.get("appId") if "appId" in data else data.get("app_id"),
            workspace_alias=(
                data.get("workspaceAlias")
                if "workspaceAlias" in data
                else data.get("workspace_alias")
            ),
            tab_id=data.get("tabId") if "tabId" in data else data.get("tab_id"),
            mcp_server_id=(
                data.get("mcpServerId") if "mcpServerId" in data else data.get("mcp_server_id")
            ),
            protocol_task_id=(
                data.get("protocolTaskId")
                if "protocolTaskId" in data
                else data.get("protocol_task_id")
            ),
        )


@dataclass(frozen=True, slots=True)
class JobWatch:
    """Durable watch registration bound to an `AutonomousJob` in `WAITING_WATCH`."""

    watch_id: uuid.UUID
    job_id: uuid.UUID
    owner_id: uuid.UUID
    host_id: HostId
    mode: WatchMode
    signals: tuple[WatchCompositeSignal, ...]
    target: WatchTarget
    done_criteria: str
    continuation_mode: ContinuationMode = ContinuationMode.REPORT_AND_ASK
    quiescence_seconds: int = DEFAULT_QUIESCENCE_SECONDS
    max_wait_seconds: int = DEFAULT_WATCH_WAIT_SECONDS
    max_wake_checks: int = DEFAULT_MAX_WAKE_CHECKS
    wake_checks_used: int = 0
    status: WatchStatus = WatchStatus.ACTIVE
    last_event_kind: str | None = None
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not 1 <= len(self.done_criteria) <= 240:
            raise ValueError("done_criteria must be 1..240 characters")
        if not MIN_QUIESCENCE_SECONDS <= self.quiescence_seconds <= MAX_QUIESCENCE_SECONDS:
            raise ValueError("quiescence_seconds out of bounds")
        if not MIN_WATCH_WAIT_SECONDS <= self.max_wait_seconds <= MAX_WATCH_WAIT_SECONDS:
            raise ValueError("max_wait_seconds out of bounds")
        if not 1 <= self.max_wake_checks <= HARD_MAX_WAKE_CHECKS:
            raise ValueError("max_wake_checks out of bounds")
        if self.wake_checks_used < 0:
            raise ValueError("wake_checks_used cannot be negative")
        if self.mode == WatchMode.COMPOSITE:
            if not 2 <= len(self.signals) <= 3 or len(set(self.signals)) != len(self.signals):
                raise ValueError("composite mode requires 2..3 distinct signals")
        elif self.signals:
            raise ValueError("signals allowed only when mode is composite")

    @property
    def is_active(self) -> bool:
        return self.status == WatchStatus.ACTIVE

    def to_summary_dict(self, *, now: datetime | None = None) -> dict[str, Any]:
        ref_now = now or datetime.now(UTC)
        elapsed = max(0, int((ref_now - self.created_at).total_seconds()))
        return {
            "watchId": str(self.watch_id),
            "jobId": str(self.job_id),
            "hostId": str(self.host_id),
            "mode": self.mode.value,
            "signals": [s.value for s in self.signals],
            "elapsedSeconds": min(elapsed, self.max_wait_seconds),
            "maxWaitSeconds": self.max_wait_seconds,
            "wakeChecksUsed": self.wake_checks_used,
            "maxWakeChecks": self.max_wake_checks,
            "doneCriteria": self.done_criteria,
            "continuationMode": self.continuation_mode.value,
            "status": self.status.value,
            "lastEventKind": self.last_event_kind,
            "expiresAt": self.expires_at.astimezone(UTC).isoformat(),
            "createdAt": self.created_at.astimezone(UTC).isoformat(),
        }


@dataclass(frozen=True, slots=True)
class WatchEvent:
    """Domain representation of an incoming `xuna/watchEvent` notification."""

    watch_id: uuid.UUID
    job_id: uuid.UUID
    kind: WatchEventKind
    observed_signals: tuple[WatchObservedSignal, ...]
    elapsed_seconds: int
    change_count: int
    digest_changed: bool
    occurred_at: datetime
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class WakeInboxItem:
    """Redacted, content-free Wake Inbox item delivered non-intrusively to the owner."""

    item_id: uuid.UUID
    owner_id: uuid.UUID
    pet_id: str
    job_id: uuid.UUID | None
    host_id: HostId | None
    kind: WakeInboxKind
    title: str
    summary: str
    suggested_actions: tuple[WakeSuggestedAction, ...] = ()
    gesture_cue: WatchGestureCue | None = None
    read_at: datetime | None = None
    dismissed_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not 1 <= len(self.title) <= MAX_WAKE_INBOX_TITLE_CHARS:
            raise ValueError("title must be 1..120 characters")
        if len(self.summary) > MAX_WAKE_INBOX_SUMMARY_CHARS:
            raise ValueError("summary exceeds 2000 characters")
        if len(self.suggested_actions) > 3:
            raise ValueError("suggested_actions may contain at most 3 items")

    def to_dict(self) -> dict[str, Any]:
        return {
            "itemId": str(self.item_id),
            "jobId": str(self.job_id) if self.job_id is not None else None,
            "hostId": str(self.host_id) if self.host_id is not None else None,
            "petId": self.pet_id,
            "kind": self.kind.value,
            "title": self.title,
            "summary": self.summary,
            "suggestedActions": [a.value for a in self.suggested_actions],
            "gestureCue": self.gesture_cue.value if self.gesture_cue is not None else None,
            "readAt": self.read_at.astimezone(UTC).isoformat() if self.read_at else None,
            "dismissedAt": (
                self.dismissed_at.astimezone(UTC).isoformat() if self.dismissed_at else None
            ),
            "createdAt": self.created_at.astimezone(UTC).isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TurnDetachedEvent:
    """Stream event emitted when an interactive turn detaches via `yieldMs`."""

    job_id: str
    watch_id: str | None
    host_id: str | None
    title: str
    state: str
    reason: str = "yield_ms"

    def wire_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "jobId": self.job_id,
            "title": self.title,
            "state": self.state,
            "reason": self.reason,
        }
        if self.watch_id is not None:
            payload["watchId"] = self.watch_id
        if self.host_id is not None:
            payload["hostId"] = self.host_id
        return payload
