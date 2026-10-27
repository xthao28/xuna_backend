"""Reply gestures: an expression cue the model may attach to its reply (owner 2026-10-27).

The closed set is the device's Track A variants, which the app plays through
the "chỉ diễn" preview (opcode 1 args 4-7): expression only, never needs,
drives, personality, learning or the snapshot. The model writes a `<g:name>`
tag; `GestureTagFilter` removes every tag from the streamed text (so it is
neither shown nor spoken) and reports the first valid one. The name is
untrusted model output: anything outside the closed set is dropped, and the
app and the device validate again.
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum


class ReplyGesture(StrEnum):
    """Wire names; the app maps them to Track A variants."""

    LOOK_AROUND = "look_around"
    YAWN_STRETCH = "yawn_stretch"
    BOUNCE_PLAY = "bounce_play"
    GREET_NUZZLE = "greet_nuzzle"


_TAG = re.compile(r"<g:([a-z_]{1,20})>")
# A piece of text that can still grow into a tag: it is held back until it does.
_TAG_PREFIX = re.compile(r"<(g(:[a-z_]{0,20}>?)?)?")


@dataclass
class GestureTagFilter:
    """Streams reply text with `<g:name>` tags removed, wherever they appear.

    Text that might start a tag (from `<`) is held back until it closes or can
    no longer be one, so a tag split across deltas never leaks. Leading
    whitespace left by a tag at the very start is dropped.
    """

    gesture: ReplyGesture | None = None
    _held: str = ""
    _started: bool = False
    _out: list[str] = field(default_factory=list)

    def feed(self, text: str) -> tuple[str, ReplyGesture | None]:
        """Clean text to emit now, and the gesture if this piece chose it."""
        chosen: ReplyGesture | None = None
        for char in text:
            chosen = self._push(char) or chosen
        return self._emit(), chosen

    def flush(self) -> str:
        """End of the reply: an unfinished tag-like piece is plain text."""
        self._out.append(self._held)
        self._held = ""
        return self._emit()

    def _push(self, char: str) -> ReplyGesture | None:
        if not self._held:
            if char == "<":
                self._held = char
            else:
                self._out.append(char)
            return None
        self._held += char
        tag = _TAG.fullmatch(self._held)
        if tag is not None:
            self._held = ""
            return self._choose(tag.group(1))
        if _TAG_PREFIX.fullmatch(self._held) is None:
            # Not a tag after all: release it; its last char may start a new one.
            held, self._held = self._held, ""
            self._out.append(held[:-1])
            return self._push(held[-1])
        return None

    def _choose(self, name: str) -> ReplyGesture | None:
        if self.gesture is not None:
            return None  # one gesture per reply
        try:
            self.gesture = ReplyGesture(name)
        except ValueError:
            return None
        return self.gesture

    def _emit(self) -> str:
        text = "".join(self._out)
        self._out.clear()
        if not self._started:
            text = text.lstrip()
            self._started = bool(text)
        return text
