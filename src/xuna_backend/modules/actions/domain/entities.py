"""Domain entities, value objects, summaries, and stream events for `actions` (UDC-4, UDC-7).

Stdlib only.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from urllib.parse import urlparse

from xuna_backend.modules.cognition.domain.types import Locale
from xuna_backend.modules.hosts.domain.entities import HostId, RiskTier

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

APPROVAL_TTL_SECONDS = 60
CONVERSATION_SCOPE_TTL = timedelta(minutes=15)
MAX_TOOL_OUTPUT_BYTES = 2048
MAX_TOOL_ROUNDS = 3
MAX_CALLS_PER_ROUND = 3
TURN_BUDGET_SECONDS = 90.0


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
class ActionResultEvent:
    """SSE / WebSocket event emitted when a tool invocation completes."""

    invocation_id: str
    target: str
    summary: str
    status: str
    gesture: str | None = None


ActionStreamEvent = (
    ActionPendingEvent
    | BoardHidRequestEvent
    | LocalActionRequestEvent
    | HostWakeRequestEvent
    | ActionResultEvent
)


_BOARD_KEY_LABELS_VI: dict[str, str] = {
    "play_pause": "Phát/Tạm dừng (▶︎)",
    "next": "Bài tiếp (⏭)",
    "previous": "Bài trước (⏮)",
    "volume_up": "Tăng âm lượng",
    "volume_down": "Giảm âm lượng",
    "mute": "Tắt tiếng",
    "stop": "Dừng phát",
}

_BOARD_KEY_LABELS_EN: dict[str, str] = {
    "play_pause": "Play/Pause (▶︎)",
    "next": "Next track (⏭)",
    "previous": "Previous track (⏮)",
    "volume_up": "Volume Up",
    "volume_down": "Volume Down",
    "mute": "Mute",
    "stop": "Stop",
}


class ActionSummary:
    """Deterministic Vietnamese & English summaries for approval cards and chat chips."""

    @staticmethod
    def format_pending(
        tool: str,
        arguments: dict[str, object],
        target_alias: str,
        locale: Locale = Locale.VI,
    ) -> str:
        if locale == Locale.EN:
            return ActionSummary._format_pending_en(tool, arguments, target_alias)
        return ActionSummary._format_pending_vi(tool, arguments, target_alias)

    @staticmethod
    def _format_pending_vi(
        tool: str,
        arguments: dict[str, object],
        target: str,
    ) -> str:
        match tool:
            case "board.media_key":
                key = str(arguments.get("key", "play_pause"))
                label = _BOARD_KEY_LABELS_VI.get(key, key)
                return f"XUNA sẽ bấm {label} trên {target}"[:240]
            case "media.play_pause":
                return f"Phát/tạm dừng nhạc trên {target}"[:240]
            case "media.next":
                return f"Chuyển bài tiếp theo trên {target}"[:240]
            case "media.previous":
                return f"Quay lại bài trước trên {target}"[:240]
            case "media.volume_set":
                level = arguments.get("level", "")
                return f"Chỉnh âm lượng {level}% trên {target}"[:240]
            case "media.now_playing":
                return f"Xem bài đang phát trên {target}"[:240]
            case "app.launch":
                app_id = str(arguments.get("id", ""))[:40]
                return f"Mở {app_id} trên {target}"[:240]
            case "app.list_allowed":
                return f"Xem danh sách ứng dụng trên {target}"[:240]
            case "url.open":
                raw_url = str(arguments.get("url", ""))
                host = urlparse(raw_url).hostname or raw_url[:40]
                return f"Mở trang {host} trên {target}"[:240]
            case "browser.current_url":
                browser = str(arguments.get("browser", "auto"))[:20]
                return f"Đọc trang đang mở ({browser}) trên {target}"[:240]
            case "workflow.run":
                wf_id = str(arguments.get("id", ""))[:40]
                return f"Chạy quy trình {wf_id} trên {target}"[:240]
            case "workflow.list":
                return f"Xem danh sách quy trình trên {target}"[:240]
            case "host.status":
                return f"Kiểm tra trạng thái {target}"[:240]
            case "host.notify":
                return f"Hiện thông báo trên {target}"[:240]
            case "display.wake":
                return f"Bật sáng màn hình {target}"[:240]
            case "workspace.open":
                proj = str(arguments.get("project", "recent"))[:40]
                ide = str(arguments.get("ide", "auto"))[:24]
                return f"Mở dự án {proj} ({ide}) trên {target}"[:240]
            case "workspace.git_summary":
                proj = str(arguments.get("project", "recent"))[:40]
                return f"Kiểm tra trạng thái Git dự án {proj} trên {target}"[:240]
            case "workspace.coding_agent":
                proj = str(arguments.get("project", "recent"))[:40]
                agent = str(arguments.get("agent", "auto"))[:24]
                return f"Mở {agent} tại dự án {proj} và tiếp tục công việc trên {target}"[:240]
            case _:
                return f"Thực hiện {tool} trên {target}"[:240]

    @staticmethod
    def _format_pending_en(
        tool: str,
        arguments: dict[str, object],
        target: str,
    ) -> str:
        match tool:
            case "board.media_key":
                key = str(arguments.get("key", "play_pause"))
                label = _BOARD_KEY_LABELS_EN.get(key, key)
                return f"XUNA will press {label} on {target}"[:240]
            case "media.play_pause":
                return f"Play/pause media on {target}"[:240]
            case "media.next":
                return f"Skip to next track on {target}"[:240]
            case "media.previous":
                return f"Previous track on {target}"[:240]
            case "media.volume_set":
                level = arguments.get("level", "")
                return f"Set volume to {level}% on {target}"[:240]
            case "media.now_playing":
                return f"Check now playing on {target}"[:240]
            case "app.launch":
                app_id = str(arguments.get("id", ""))[:40]
                return f"Open {app_id} on {target}"[:240]
            case "app.list_allowed":
                return f"List allowed apps on {target}"[:240]
            case "url.open":
                raw_url = str(arguments.get("url", ""))
                host = urlparse(raw_url).hostname or raw_url[:40]
                return f"Open {host} on {target}"[:240]
            case "browser.current_url":
                browser = str(arguments.get("browser", "auto"))[:20]
                return f"Read active browser tab ({browser}) on {target}"[:240]
            case "workflow.run":
                wf_id = str(arguments.get("id", ""))[:40]
                return f"Run workflow {wf_id} on {target}"[:240]
            case "workflow.list":
                return f"List workflows on {target}"[:240]
            case "host.status":
                return f"Check status of {target}"[:240]
            case "host.notify":
                return f"Show notification on {target}"[:240]
            case "display.wake":
                return f"Wake display on {target}"[:240]
            case "workspace.open":
                proj = str(arguments.get("project", "recent"))[:40]
                ide = str(arguments.get("ide", "auto"))[:24]
                return f"Open workspace {proj} ({ide}) on {target}"[:240]
            case "workspace.git_summary":
                proj = str(arguments.get("project", "recent"))[:40]
                return f"Read Git summary for {proj} on {target}"[:240]
            case "workspace.coding_agent":
                proj = str(arguments.get("project", "recent"))[:40]
                agent = str(arguments.get("agent", "auto"))[:24]
                return f"Launch {agent} in {proj} and continue work on {target}"[:240]
            case _:
                return f"Run {tool} on {target}"[:240]

    @staticmethod
    def format_result(
        tool: str,
        arguments: dict[str, object],
        target_alias: str,
        status: str,
        locale: Locale = Locale.VI,
    ) -> str:
        base = ActionSummary.format_pending(tool, arguments, target_alias, locale)
        if locale == Locale.EN:
            if status == "ok":
                return f"✓ {base}"[:240]
            return f"✗ {base} ({status})"[:240]
        if status == "ok":
            return f"✓ Đã thực hiện: {base}"[:240]
        reason_map = {
            "needs_permission": "thiếu quyền hệ thống",
            "denied": "bị từ chối bởi chính sách",
            "rejected": "bạn đã từ chối",
            "expired": "hết thời gian xác nhận",
            "timeout": "quá thời gian phản hồi",
            "unsupported": "không hỗ trợ",
            "failed": "lỗi thực thi",
        }
        reason = reason_map.get(status, status)
        return f"✗ Không thực hiện được: {base} ({reason})"[:240]


def gesture_for_action_status(status: str) -> str | None:
    """Deterministic gesture mapping (UDC-4 §5.4):
    - `ok` -> `bounce_play`
    - `failed`, `timeout`, `needs_permission`, `denied` -> `look_around`
    - `rejected` / `expired` -> None
    """
    if status == "ok":
        return "bounce_play"
    if status in ("failed", "timeout", "needs_permission", "denied"):
        return "look_around"
    return None
