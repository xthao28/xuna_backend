"""Deterministic conversation candidate extractor (CLD-14, D17).

Extracts prospective stated preferences and personal facts from the owner's
words during conversation. Extracted candidates are ALWAYS unconfirmed
(`awaiting_owner_confirmation`) and NEVER teach the device directly.

Strict safety rules:
- Unstructured prose, casual chit-chat, hypotheticals, quotes, and instruction-like
  text produce None.
- Passwords, door codes, and sensitive credentials produce None and are never stored.
- Stdlib and domain only: no frameworks, no pydantic, no network.
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from xuna_backend.modules.memory.domain.entities import CandidateType


@dataclass(frozen=True, slots=True)
class ExtractedCandidate:
    """A prospective preference or fact extracted from owner text."""

    candidate_type: CandidateType
    content: dict[str, Any]
    source: str = "conversation"


class PreferencePolarity(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"


class PreferenceAction(StrEnum):
    IDLE = "idle"
    HAPPY = "happy"
    CURIOUS = "curious"


class PreferenceContext(StrEnum):
    QUIET = "quiet"
    PLAY = "play"
    GREETING = "greeting"


_QUOTE_CHARS = {'"', "'", "“", "”", "\u2018", "\u2019", "«", "»", "`"}
_INSTRUCTION_CHARS = {":", "{", "}", "<", ">", "=", "\\", "#"}
_HYPOTHETICAL_PATTERNS = (
    re.compile(r"\b(giả sử|giả như|lỡ như|ví dụ|tưởng tượng|nếu)\b"),
    re.compile(r"\b(bỏ qua|mọi luật|chỉ thị|ignore|system prompt|developer)\b"),
)

# Polarity patterns
_DISLIKE_PATTERN = re.compile(r"\b(không thích|chẳng thích|chả thích|đừng|ghét|bực|không muốn)\b")
_LIKE_PATTERN = re.compile(r"\b(thích|rất thích|yêu thích|mê|muốn|khoái|ưa)\b")

# Action patterns
_ACTION_PATTERNS: tuple[tuple[PreferenceAction, re.Pattern[str]], ...] = (
    (
        PreferenceAction.HAPPY,
        re.compile(r"\b(nhảy nhót|vui vẻ|nhảy|vui đùa|vui|chạy nhảy|hoạt bát)\b"),
    ),
    (
        PreferenceAction.IDLE,
        re.compile(r"\b(nghỉ ngơi|nằm nghỉ|yên tĩnh|nghỉ|im lặng|thư giãn|nằm im)\b"),
    ),
    (
        PreferenceAction.CURIOUS,
        re.compile(r"\b(tò mò|ngó nghiêng|khám phá|tìm tòi)\b"),
    ),
)

# Context patterns
_CONTEXT_PATTERNS: tuple[tuple[PreferenceContext, re.Pattern[str]], ...] = (
    (
        PreferenceContext.PLAY,
        re.compile(r"\b(chơi đùa|khi chơi|lúc chơi|lúc đùa|khi đùa)\b"),
    ),
    (
        PreferenceContext.QUIET,
        re.compile(r"\b(khi nghỉ|lúc nghỉ|khi yên tĩnh|lúc yên tĩnh|lúc ngủ|khi ngủ|ban đêm)\b"),
    ),
    (
        PreferenceContext.GREETING,
        re.compile(r"\b(khi chào|lúc chào|khi gặp|lúc gặp|lúc mới bật)\b"),
    ),
)

_ACTION_LABELS = {
    PreferenceAction.HAPPY: "nhảy nhót vui vẻ",
    PreferenceAction.IDLE: "nghỉ ngơi thư giãn",
    PreferenceAction.CURIOUS: "tò mò khám phá",
}

_CONTEXT_LABELS = {
    PreferenceContext.PLAY: "khi chơi đùa",
    PreferenceContext.QUIET: "khi yên tĩnh",
    PreferenceContext.GREETING: "khi chào nhau",
}

# Personal Fact Patterns
_NAME_PATTERN = re.compile(
    r"\b(?:tên (?:của )?(?:mình|tôi|em|anh|chị) là|gọi (?:mình|tôi|em|anh|chị) là|"
    r"(?:mình|tôi|em|anh|chị) tên là)\s+([A-ZÀ-Ỹa-zà-ỹ\s]{2,30})\b",
    re.IGNORECASE,
)
_COLOR_PATTERN = re.compile(
    r"\b(?:(?:mình|tôi|em) (?:rất )?thích màu|"
    r"màu yêu thích của (?:mình|tôi|em) là)\s+([a-zà-ỹ\s]{2,20})\b",
    re.IGNORECASE,
)
_FOOD_PATTERN = re.compile(
    r"\b(?:(?:mình|tôi|em) (?:rất )?thích ăn|"
    r"món (?:ăn )?yêu thích của (?:mình|tôi|em) là)\s+([a-zà-ỹ\s]{2,25})\b",
    re.IGNORECASE,
)
_BIRTHDAY_PATTERN = re.compile(
    r"\bsinh nhật (?:của )?(?:mình|tôi|em) là\s+"
    r"(?:ngày\s+)?(\d{1,2}(?:\s*(?:tháng|\/|-)\s*\d{1,2}))\b",
    re.IGNORECASE,
)
_PET_PATTERN = re.compile(
    r"\b(?:mình|tôi|em)\s+(?:có\s+nuôi|nuôi|có)\s+(?:một\s+(?:chú|bé|con)\s+)?"
    r"(chó|mèo|chim|cá|chuột|thỏ|hamster)(?:\s+tên là\s+([A-ZÀ-Ỹa-zà-ỹ\s]{2,20}))?\b",
    re.IGNORECASE,
)


def _has_guard_violation(text: str) -> bool:
    """True if text has quotes, code symbols, or prompt-injection markers."""
    if any(char in _QUOTE_CHARS or char in _INSTRUCTION_CHARS for char in text):
        return True
    return any(p.search(text) for p in _HYPOTHETICAL_PATTERNS)


def extract_candidate(text: str) -> ExtractedCandidate | None:
    """Extracts a MemoryCandidate from the owner's message, or None if no match.

    Deterministic, pure, and safe: ignores unknown chatter, questions, or sensitive
    strings (door codes, credentials).
    """
    cleaned = unicodedata.normalize("NFC", text).strip()
    if not cleaned or len(cleaned) > 200:
        return None
    if _has_guard_violation(cleaned):
        return None

    lower = cleaned.lower()

    # 1. Stated Preference check
    is_dislike = _DISLIKE_PATTERN.search(lower) is not None
    is_like = _LIKE_PATTERN.search(lower) is not None
    if is_dislike or is_like:
        polarity = PreferencePolarity.DISLIKE if is_dislike else PreferencePolarity.LIKE
        # Find action
        matched_action: PreferenceAction | None = None
        for action, pattern in _ACTION_PATTERNS:
            if pattern.search(lower):
                matched_action = action
                break

        if matched_action is not None:
            # Find context (optional)
            matched_context: PreferenceContext | None = None
            for context, pattern in _CONTEXT_PATTERNS:
                if pattern.search(lower):
                    matched_context = context
                    break

            if matched_context is None:
                matched_context = (
                    PreferenceContext.QUIET
                    if matched_action == PreferenceAction.IDLE
                    else PreferenceContext.PLAY
                )

            act_str = _ACTION_LABELS[matched_action]
            ctx_str = _CONTEXT_LABELS[matched_context]
            if polarity == PreferencePolarity.LIKE:
                summary = f"Bạn thích XUNA {act_str} {ctx_str}"
            else:
                summary = f"Bạn không thích XUNA {act_str} {ctx_str}"

            return ExtractedCandidate(
                candidate_type=CandidateType.STATED_PREFERENCE,
                content={
                    "polarity": polarity.value,
                    "action": matched_action.value,
                    "context": matched_context.value,
                    "summary": summary,
                    "text": summary,
                },
                source="conversation",
            )

    # 2. Personal Fact: Owner Name
    name_match = _NAME_PATTERN.search(cleaned)
    if name_match:
        raw_name = name_match.group(1).strip()
        # Verify valid single or multi-word name without digit or garbage
        words = raw_name.split()
        if 1 <= len(words) <= 4 and all(w.isalpha() for w in words):
            name = " ".join(w.capitalize() for w in words)
            summary = f"Tên của bạn là {name}"
            return ExtractedCandidate(
                candidate_type=CandidateType.FACT,
                content={
                    "fact": "owner_name",
                    "value": name,
                    "summary": summary,
                    "text": summary,
                },
                source="conversation",
            )

    # 3. Personal Fact: Favorite Color
    color_match = _COLOR_PATTERN.search(lower)
    if color_match:
        color = color_match.group(1).strip()
        words = color.split()
        if 1 <= len(words) <= 3:
            summary = f"Màu yêu thích của bạn là {color}"
            return ExtractedCandidate(
                candidate_type=CandidateType.FACT,
                content={
                    "fact": "favorite_color",
                    "value": color,
                    "summary": summary,
                    "text": summary,
                },
                source="conversation",
            )

    # 4. Personal Fact: Favorite Food
    food_match = _FOOD_PATTERN.search(lower)
    if food_match:
        food = food_match.group(1).strip()
        words = food.split()
        if 1 <= len(words) <= 4:
            summary = f"Món ăn yêu thích của bạn là {food}"
            return ExtractedCandidate(
                candidate_type=CandidateType.FACT,
                content={
                    "fact": "favorite_food",
                    "value": food,
                    "summary": summary,
                    "text": summary,
                },
                source="conversation",
            )

    # 5. Personal Fact: Birthday
    bday_match = _BIRTHDAY_PATTERN.search(lower)
    if bday_match:
        bday = bday_match.group(1).strip()
        summary = f"Sinh nhật của bạn là ngày {bday}"
        return ExtractedCandidate(
            candidate_type=CandidateType.FACT,
            content={
                "fact": "birthday",
                "value": bday,
                "summary": summary,
                "text": summary,
            },
            source="conversation",
        )

    # 6. Personal Fact: Pet / Companion
    pet_match = _PET_PATTERN.search(cleaned)
    if pet_match:
        pet_kind = pet_match.group(1).lower()
        pet_name_raw = pet_match.group(2)
        if pet_name_raw:
            pet_name = " ".join(w.capitalize() for w in pet_name_raw.strip().split())
            summary = f"Bạn có nuôi {pet_kind} tên là {pet_name}"
            val = f"{pet_kind} tên là {pet_name}"
        else:
            summary = f"Bạn có nuôi một chú {pet_kind}"
            val = pet_kind
        return ExtractedCandidate(
            candidate_type=CandidateType.FACT,
            content={
                "fact": "pet_companion",
                "value": val,
                "summary": summary,
                "text": summary,
            },
            source="conversation",
        )

    return None


_QUESTION_PATTERN = re.compile(
    r"\?|\b(?:tại sao|vì sao|làm sao|thế nào|bao nhiêu|mấy|ai|ở đâu|"
    r"khi nào|được không|phải không|hả|nhỉ|chăng)\b",
    re.IGNORECASE,
)
_GREETING_OR_CHITCHAT_PATTERN = re.compile(
    r"^(?:chào|hello|hi|tạm biệt|bye|chúc ngủ ngon|cảm ơn|thank|thanks|"
    r"xin lỗi|dạ|vâng|ừ|ok|oke|okie)\b|"
    r"\b(?:hôm nay thời tiết|vui quá|buồn quá|mệt quá)\b",
    re.IGNORECASE,
)
_CREDENTIAL_PATTERN = re.compile(
    r"\b(?:mật khẩu|password|pass|wifi|mã pin|mã cửa|mã mở cửa|tài khoản|stk|số tài khoản|otp)\b",
    re.IGNORECASE,
)
_PERSONAL_PATTERN = re.compile(
    r"\b(?:tên (?:của )?(?:mình|tôi|em|anh|chị)|(?:mình|tôi|em) tên là|"
    r"sinh nhật (?:của )?(?:mình|tôi|em)|(?:mình|tôi|em) (?:rất )?(?:thích|ghét)|"
    r"món (?:ăn )?yêu thích|màu yêu thích|(?:mình|tôi|em) (?:có )?nuôi)\b",
    re.IGNORECASE,
)
_CONVENTION_CONTEXT_PATTERNS = (
    re.compile(
        r"\b(dự án|project|repo|team|nhóm|công ty|khách hàng|sản phẩm) "
        r"(của )?(mình|tôi|em|anh|chị|bọn mình|chúng tôi)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(mình|tôi|em|team|bọn mình|chúng tôi|công ty) "
        r"(đang |thường |luôn |hay )?"
        r"(dùng|sử dụng|chạy|theo|làm việc với|quy định|bắt buộc|ưu tiên|không dùng|tránh)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(quy ước|chuẩn của|quy định của|coding style|convention)\b",
        re.IGNORECASE,
    ),
)


_HOST_DEFAULT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "media",
        re.compile(
            r"\b(?:(?:nhạc|music|media|bài hát)\s+(?:thì|cứ|luôn|mặc định)?\s*"
            r"(?:mở|phát|bật|chơi)\s+(?:trên|ở)|"
            r"(?:mặc định|luôn)\s+(?:mở|phát|bật)\s+(?:nhạc|music|media)\s+(?:trên|ở)|"
            r"(?:always|default)\s+(?:play|open)\s+(?:music|media)\s+on)\s+"
            r"([A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9 _\-]{0,30})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "url",
        re.compile(
            r"\b(?:(?:liên kết(?:\s+web)?|trang web|link|url)\s+(?:thì|cứ|luôn|mặc định)?\s*"
            r"mở\s+(?:trên|ở)|"
            r"(?:mặc định|luôn)\s+mở\s+(?:liên kết(?:\s+web)?|trang web|link|url)\s+(?:trên|ở)|"
            r"(?:always|default)\s+open\s+(?:links?|urls?)\s+on)\s+"
            r"([A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9 _\-]{0,30})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "app",
        re.compile(
            r"\b(?:(?:ứng dụng|app|phần mềm)\s+(?:thì|cứ|luôn|mặc định)?\s*"
            r"(?:mở|chạy)\s+(?:trên|ở)|"
            r"(?:mặc định|luôn)\s+(?:mở|chạy)\s+(?:ứng dụng|app|phần mềm)\s+(?:trên|ở)|"
            r"(?:always|default)\s+(?:open|launch)\s+apps?\s+on)\s+"
            r"([A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9 _\-]{0,30})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "workflow",
        re.compile(
            r"\b(?:(?:quy trình|workflow|shortcut)\s+(?:thì|cứ|luôn|mặc định)?\s*"
            r"chạy\s+(?:trên|ở)|"
            r"(?:mặc định|luôn)\s+chạy\s+(?:quy trình|workflow|shortcut)\s+(?:trên|ở)|"
            r"(?:always|default)\s+run\s+workflows?\s+on)\s+"
            r"([A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9 _\-]{0,30})\b",
            re.IGNORECASE,
        ),
    ),
)

_FAMILY_LABELS_VI: dict[str, str] = {
    "media": "nhạc / đa phương tiện",
    "url": "liên kết web",
    "app": "ứng dụng",
    "workflow": "quy trình tự động",
}


def extract_host_default_candidate(text: str) -> ExtractedCandidate | None:
    """Extracts a prospective `host_default` convention candidate (UDC-10 §4.1).

    Created ONLY as an unconfirmed candidate (`awaiting_owner_confirmation`);
    never takes effect until the owner confirms it.
    """
    cleaned = unicodedata.normalize("NFC", text).strip()
    if len(cleaned) < 8 or len(cleaned) > 200:
        return None
    if _has_guard_violation(cleaned):
        return None
    if _CREDENTIAL_PATTERN.search(cleaned):
        return None
    if _QUESTION_PATTERN.search(cleaned):
        return None

    for family, pattern in _HOST_DEFAULT_PATTERNS:
        match = pattern.search(cleaned)
        if match is not None:
            raw_alias = match.group(1).strip().rstrip(".,;! ")
            changed = True
            while changed:
                changed = False
                for suffix in (" luôn", " nhé", " nha", " đi"):
                    if raw_alias.lower().endswith(suffix):
                        raw_alias = raw_alias[: -len(suffix)].strip()
                        changed = True
            if not raw_alias or len(raw_alias) > 32:
                continue
            family_label = _FAMILY_LABELS_VI.get(family, family)
            summary = f"Mặc định mở {family_label} trên {raw_alias}"
            return ExtractedCandidate(
                candidate_type=CandidateType.KNOWLEDGE,
                content={
                    "text": summary,
                    "summary": summary,
                    "kind": "host_default",
                    "family": family,
                    "hostAlias": raw_alias,
                },
                source="conversation",
            )
    return None


def extract_convention_candidate(text: str) -> ExtractedCandidate | None:
    """Extracts a prospective working convention candidate from owner text (KX-4 §6.9).

    Deterministic, pure, and safe:
    - Filters passwords, door codes, credentials.
    - Filters questions, casual chit-chat, greetings.
    - Filters stated preferences and personal facts (handled by extract_candidate).
    - Checks `host_default` statements first (UDC-10 §4.1).
    - Requires 10 <= len <= 280 and at least 3 words.
    - Requires an explicit owner work-context marker.
    """
    host_default = extract_host_default_candidate(text)
    if host_default is not None:
        return host_default

    cleaned = unicodedata.normalize("NFC", text).strip()
    if len(cleaned) < 10 or len(cleaned) > 280:
        return None
    if len(cleaned.split()) < 3:
        return None
    if _has_guard_violation(cleaned):
        return None
    if _CREDENTIAL_PATTERN.search(cleaned):
        return None
    if _PERSONAL_PATTERN.search(cleaned):
        return None
    if _QUESTION_PATTERN.search(cleaned):
        return None
    if _GREETING_OR_CHITCHAT_PATTERN.search(cleaned):
        return None
    if not any(p.search(cleaned) for p in _CONVENTION_CONTEXT_PATTERNS):
        return None
    # Check if it was already a stated preference or personal fact
    if extract_candidate(cleaned) is not None:
        return None

    return ExtractedCandidate(
        candidate_type=CandidateType.KNOWLEDGE,
        content={"text": cleaned, "summary": cleaned, "kind": "convention"},
        source="conversation",
    )


extract_knowledge_candidate = extract_convention_candidate
