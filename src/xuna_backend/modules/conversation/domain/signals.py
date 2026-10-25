"""Closed conversation tags (personality PE-4c, plan §6) from the owner's words.

A cheap deterministic classifier, not an extra LLM call: the same words give
the same tags whatever provider answers, nothing more is sent anywhere, and a
tag never carries text. Tags are weak on purpose (§6.5): the device weighs
each at 0.1, caps them per day and needs ≥ 4 owner sessions to move a trait,
so a missed negation or an odd phrase cannot reshape XUNA on its own.
"""

import re
import unicodedata
from enum import StrEnum


class ConversationSignal(StrEnum):
    """Wire names; the app maps them to BLE opcode 8 tags 1 to 6."""

    PRAISE = "praise"
    SCOLD = "scold"
    AFFECTION = "affection"
    PLAYFUL = "playful"
    CURIOUS_QUESTION = "curious_question"
    LONG_CHAT = "long_chat"


MAX_SIGNALS = 2
# History lines (owner + XUNA) at which a talk counts as long, once.
LONG_CHAT_LINES = 8

# Priority order: the first matches win when more than MAX_SIGNALS apply.
_WORDS: tuple[tuple[ConversationSignal, re.Pattern[str]], ...] = (
    (
        ConversationSignal.SCOLD,
        re.compile(r"\b(hư|ngốc|ngu|lười|bực|mắng|phạt|hư hỏng|tệ quá|dở quá)\b"),
    ),
    (
        ConversationSignal.PRAISE,
        re.compile(
            r"\b(giỏi|ngoan|tuyệt|thông minh|khen|cừ|xuất sắc|hay quá|đáng yêu|dễ thương)\b"
        ),
    ),
    (
        ConversationSignal.AFFECTION,
        re.compile(r"\b(thương|yêu(?! cầu)|cưng|ôm|quý|nhớ (bạn|xuna|em|cậu|mày))\b"),
    ),
    (ConversationSignal.PLAYFUL, re.compile(r"\b(chơi|đùa|trêu|haha|hihi|hehe)\b")),
    (
        ConversationSignal.CURIOUS_QUESTION,
        re.compile(r"\b(tại sao|vì sao|là gì|như thế nào|thế nào|ra sao|có biết)\b"),
    ),
)
_NEGATION = re.compile(r"\b(không|chẳng|chả|đừng)\s+(\w+\s+)?$")


def _said(pattern: re.Pattern[str], text: str) -> bool:
    """True when the words appear without a negation right before them."""
    return any(not _NEGATION.search(text[: m.start()]) for m in pattern.finditer(text))


def conversation_signals(text: str, history_lines: int = 0) -> tuple[ConversationSignal, ...]:
    """At most MAX_SIGNALS tags for one owner line, in priority order.

    `history_lines` is the turn's earlier lines; a long chat is tagged when it
    first reaches LONG_CHAT_LINES (one turn adds one or two lines).
    """
    normalized = unicodedata.normalize("NFC", text).lower()
    found = [signal for signal, words in _WORDS if _said(words, normalized)]
    if LONG_CHAT_LINES <= history_lines < LONG_CHAT_LINES + 2:
        found.append(ConversationSignal.LONG_CHAT)
    return tuple(found[:MAX_SIGNALS])
