"""Dual-output stream splitter for expert mode (`<say>...</say>` + Markdown `detail`, KX-1 §6.3).

Pure stdlib only. Input text has already passed through `GestureTagFilter`.
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum

MAX_SAY_CHARS = 280  # hard cap; prompt asks for <= 200

_OPEN_TAG = "<say>"
_CLOSE_TAG = "</say>"


class Section(StrEnum):
    SAY = "say"
    DETAIL = "detail"


class _State(StrEnum):
    PRE = "pre"
    SAY = "say"
    DETAIL = "detail"
    UNTAGGED = "untagged"


def _close_prefix_len(text: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `</say>`."""
    max_k = min(len(text), len(_CLOSE_TAG) - 1)
    for k in range(max_k, 0, -1):
        if _CLOSE_TAG.startswith(text[-k:]):
            return k
    return 0


def _tag_prefix_len(text: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `<say>` or `</say>`."""
    max_k = min(len(text), len(_CLOSE_TAG) - 1)
    for k in range(max_k, 0, -1):
        suffix = text[-k:]
        if _OPEN_TAG.startswith(suffix) or _CLOSE_TAG.startswith(suffix):
            return k
    return 0


@dataclass
class ReplySectionFilter:
    """Splits streamed expert reply text into `SAY` and `DETAIL` chunks."""

    _state: _State = _State.PRE
    _tagged: bool = False
    _held: str = ""
    _say_chars: int = 0
    _out: list[tuple[Section, str]] = field(default_factory=list)

    @property
    def tagged(self) -> bool:
        """True if a valid `<say>` tag was seen at the start of the reply."""
        return self._tagged

    def feed(self, text: str) -> list[tuple[Section, str]]:
        """Process a chunk of streamed text and return emitted `(Section, str)` pairs."""
        for char in text:
            self._push_char(char)
        self._drain_held_if_safe()
        return self._coalesce_and_clear()

    def flush(self) -> list[tuple[Section, str]]:
        """Flush any held characters at the end of the reply stream."""
        if self._held:
            if self._state == _State.SAY:
                self._emit_say(self._held)
            elif self._state == _State.DETAIL:
                cleaned = self._held.replace(_OPEN_TAG, "").replace(_CLOSE_TAG, "")
                if cleaned:
                    self._out.append((Section.DETAIL, cleaned))
            else:
                # PRE or UNTAGGED -> DETAIL
                self._out.append((Section.DETAIL, self._held))
            self._held = ""
        return self._coalesce_and_clear()

    def _push_char(self, char: str) -> None:
        if self._state == _State.PRE:
            if not self._held and char.isspace():
                return
            self._held += char
            if self._held == _OPEN_TAG:
                self._held = ""
                self._tagged = True
                self._state = _State.SAY
            elif not _OPEN_TAG.startswith(self._held):
                held, self._held = self._held, ""
                self._state = _State.UNTAGGED
                self._out.append((Section.DETAIL, held))
            return

        if self._state == _State.SAY:
            self._held += char
            if self._held.endswith(_CLOSE_TAG):
                before = self._held[: -len(_CLOSE_TAG)]
                self._held = ""
                if before:
                    self._emit_say(before)
                self._state = _State.DETAIL
            return

        if self._state == _State.DETAIL:
            self._held += char
            if self._held.endswith(_OPEN_TAG):
                self._held = self._held[: -len(_OPEN_TAG)]
            elif self._held.endswith(_CLOSE_TAG):
                self._held = self._held[: -len(_CLOSE_TAG)]
            return

        # UNTAGGED
        self._out.append((Section.DETAIL, char))

    def _drain_held_if_safe(self) -> None:
        if self._state == _State.SAY and self._held:
            keep = _close_prefix_len(self._held)
            safe_len = len(self._held) - keep
            if safe_len > 0:
                to_emit = self._held[:safe_len]
                self._held = self._held[safe_len:]
                overflowed = self._emit_say(to_emit)
                if overflowed and self._held:
                    # Transitioned due to MAX_SAY_CHARS overflow; re-evaluate held in DETAIL
                    leftover, self._held = self._held, ""
                    for ch in leftover:
                        self._push_char(ch)
        if self._state == _State.DETAIL and self._held:
            keep = _tag_prefix_len(self._held)
            safe_len = len(self._held) - keep
            if safe_len > 0:
                to_emit = self._held[:safe_len]
                self._held = self._held[safe_len:]
                cleaned = to_emit.replace(_OPEN_TAG, "").replace(_CLOSE_TAG, "")
                if cleaned:
                    self._out.append((Section.DETAIL, cleaned))

    def _emit_say(self, text: str) -> bool:
        if not text:
            return False
        remaining = MAX_SAY_CHARS - self._say_chars
        if len(text) <= remaining:
            self._say_chars += len(text)
            self._out.append((Section.SAY, text))
            return False
        if remaining > 0:
            say_part = text[:remaining]
            self._say_chars += len(say_part)
            self._out.append((Section.SAY, say_part))
        overflow = text[max(0, remaining) :]
        self._state = _State.DETAIL
        cleaned = overflow.replace(_OPEN_TAG, "").replace(_CLOSE_TAG, "")
        if cleaned:
            self._out.append((Section.DETAIL, cleaned))
        return True

    def _coalesce_and_clear(self) -> list[tuple[Section, str]]:
        if not self._out:
            return []
        merged: list[tuple[Section, str]] = []
        for sec, txt in self._out:
            if not txt:
                continue
            if merged and merged[-1][0] == sec:
                merged[-1] = (sec, merged[-1][1] + txt)
            else:
                merged.append((sec, txt))
        self._out.clear()
        return merged


_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_LINE_PREFIX_RE = re.compile(r"(?m)^\s*(?:-\s+|\d+\.\s+)")
_MD_CHARS_TABLE = str.maketrans("", "", "#*_`>|")
_SENTENCE_END_RE = re.compile(r"[.!?]")


def fallback_say(detail: str, locale_vi: bool) -> str:
    """Derives a short spoken summary when the model omitted `<say>` (§6.3)."""
    without_code = _CODE_BLOCK_RE.sub(" ", detail)
    without_prefix = _LINE_PREFIX_RE.sub("", without_code)
    without_md = without_prefix.translate(_MD_CHARS_TABLE)
    collapsed = " ".join(without_md.split()).strip()

    if not collapsed:
        return (
            "Mình để câu trả lời chi tiết trong app nhé."
            if locale_vi
            else "I've put the full answer in the app."
        )

    match = _SENTENCE_END_RE.search(collapsed)
    if match is not None and match.end() <= 200:
        return collapsed[: match.end()].strip()

    if len(collapsed) <= 200:
        return collapsed

    window = collapsed[:200]
    last_space = window.rfind(" ")
    cut = window[:last_space].rstrip() if last_space > 0 else window
    return f"{cut}…"
