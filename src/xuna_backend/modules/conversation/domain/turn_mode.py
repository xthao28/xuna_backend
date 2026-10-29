"""Turn mode routing rules (KX-1, plan §6.2).

Pure stdlib only: decides whether a turn uses conversational chat mode or
dual-output expert mode (`<say>` + Markdown `detail`).
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class RequestedMode(StrEnum):
    AUTO = "auto"
    CHAT = "chat"
    EXPERT = "expert"


class TurnMode(StrEnum):
    CHAT = "chat"
    EXPERT = "expert"


class Channel(StrEnum):
    TEXT = "text"
    VOICE = "voice"


class ModeReason(StrEnum):
    LEGACY_CLIENT = "legacy_client"
    REQUESTED_CHAT = "requested_chat"
    REQUESTED_EXPERT = "requested_expert"
    CODE_OR_LONG = "code_or_long"
    DOMAIN_TASK = "domain_task"
    CASUAL = "casual"


@dataclass(frozen=True, slots=True)
class ModeDecision:
    mode: TurnMode
    reason: ModeReason


LONG_TEXT_CHARS = 280  # [P]

_TASK_RE = re.compile(
    r"\b("
    r"giải thích|phân tích|so sánh|tại sao|vì sao|làm sao|làm thế nào|"
    r"như thế nào|cách nào|cách để|hướng dẫn|viết|sửa|lỗi|debug|tối ưu|"
    r"thiết kế|review|đánh giá|tư vấn|tính toán|kiểm tra|"
    r"explain|why|how|compare|write|fix|error|bug|optimi[sz]e|design"
    r")\b",
    re.UNICODE,
)


def looks_like_code(text: str) -> bool:
    """True when text contains a fenced code block or at least 3 newlines."""
    return "```" in text or text.count("\n") >= 3


def is_task_request(text: str) -> bool:
    """True when NFC-lowercased text contains `?` or matches a task keyword."""
    norm = unicodedata.normalize("NFC", text).lower()
    if "?" in norm:
        return True
    return _TASK_RE.search(norm) is not None


def decide_mode(
    requested: RequestedMode | None,
    channel: Channel,
    text: str,
    domain_matched: bool,
) -> ModeDecision:
    """Decides the turn mode and reason in strict priority order (§6.2)."""
    if requested is None:
        return ModeDecision(TurnMode.CHAT, ModeReason.LEGACY_CLIENT)
    if requested == RequestedMode.CHAT:
        return ModeDecision(TurnMode.CHAT, ModeReason.REQUESTED_CHAT)
    if requested == RequestedMode.EXPERT:
        return ModeDecision(TurnMode.EXPERT, ModeReason.REQUESTED_EXPERT)

    # AUTO
    if channel == Channel.TEXT and (looks_like_code(text) or len(text.strip()) >= LONG_TEXT_CHARS):
        return ModeDecision(TurnMode.EXPERT, ModeReason.CODE_OR_LONG)
    if domain_matched and is_task_request(text):
        return ModeDecision(TurnMode.EXPERT, ModeReason.DOMAIN_TASK)
    return ModeDecision(TurnMode.CHAT, ModeReason.CASUAL)
