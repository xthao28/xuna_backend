"""Pure-stdlib secret redaction and payload boundary validation for clipboard transfers.

Matches the desktop security redaction patterns (`[REDACTED]`) and rejects binary,
NUL, control-character, invalid UTF-8, or oversized payloads cleanly with
`ValueError("invalid_args")`.
"""

import re
from dataclasses import dataclass

MAX_SELECTION_CHARS = 1200
MAX_CLIPBOARD_WRITE_CHARS = 4000
REDACTED_MARKER = "[REDACTED]"

_PEM_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
)
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    _PEM_KEY_RE,
    re.compile(r"sk-[a-zA-Z0-9_\-]{12,}"),
    re.compile(r"gh[pousr]_[a-zA-Z0-9_]{12,}"),
    re.compile(r"github_pat_[a-zA-Z0-9_]{12,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])"),
    re.compile(r"xox[baprs]-[a-zA-Z0-9\-]{10,}"),
    re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*", re.IGNORECASE),
    re.compile(
        r"""(?:password|passwd|secret|api_key|apikey|token|access_token)["']?\s*[:=]\s*(?:"""
        r'"(?!\[REDACTED\]")[^"\r\n]{4,}"'
        r"|'(?!\[REDACTED\]')[^'\r\n]{4,}'"
        r"""|(?!\[REDACTED\])[^\s"',;]{4,})""",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Result of running secret-redaction rules over a text payload."""

    text: str
    redacted_count: int


@dataclass(frozen=True, slots=True)
class ClipboardPayloadResult:
    """Validated, redacted, and bounded text ready for cross-host clipboard transfer."""

    text: str
    char_count: int
    redacted_count: int
    truncated: bool


def contains_binary_or_control(text: str) -> bool:
    """Returns `True` if `text` contains NUL/control bytes or invalid UTF-8 surrogates."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True
    for ch in text:
        code = ord(ch)
        if code == 0 or (code < 32 and ch not in ("\n", "\r", "\t")) or code == 127:
            return True
    return False


def redact_sensitive_text(text: str) -> RedactionResult:
    """Replaces known secret token patterns with `[REDACTED]` and counts hits."""
    current = text
    total_hits = 0
    for pattern in _SECRET_PATTERNS:
        current, hits = pattern.subn(REDACTED_MARKER, current)
        total_hits += hits
    return RedactionResult(text=current, redacted_count=total_hits)


def prepare_selection_for_clipboard(
    raw: object,
    *,
    max_chars: int = MAX_SELECTION_CHARS,
) -> ClipboardPayloadResult:
    """Validates a read selection payload, redacts secrets, and bounds length to `max_chars`."""
    if not isinstance(raw, str) or not raw.strip() or contains_binary_or_control(raw):
        raise ValueError("invalid_args")
    redacted = redact_sensitive_text(raw)
    truncated = len(redacted.text) > max_chars
    bounded = redacted.text[:max_chars]
    second_pass = redact_sensitive_text(bounded)
    return ClipboardPayloadResult(
        text=second_pass.text,
        char_count=len(second_pass.text),
        redacted_count=redacted.redacted_count + second_pass.redacted_count,
        truncated=truncated,
    )


def validate_clipboard_payload(
    raw: object,
    *,
    max_chars: int = MAX_CLIPBOARD_WRITE_CHARS,
) -> ClipboardPayloadResult:
    """Strictly validates a clipboard write payload; rejects oversize/binary with `invalid_args`."""
    if (
        not isinstance(raw, str)
        or not raw.strip()
        or len(raw) > max_chars
        or contains_binary_or_control(raw)
    ):
        raise ValueError("invalid_args")
    redacted = redact_sensitive_text(raw)
    if len(redacted.text) > max_chars:
        raise ValueError("invalid_args")
    return ClipboardPayloadResult(
        text=redacted.text,
        char_count=len(redacted.text),
        redacted_count=redacted.redacted_count,
        truncated=False,
    )
