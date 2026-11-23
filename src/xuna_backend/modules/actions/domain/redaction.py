"""Pure-stdlib secret redaction, payload boundary validation, and Consent U5 wake sanitization.

Matches the desktop security redaction patterns (`[REDACTED]`), rejects binary/NUL/control/
invalid-UTF-8/oversized clipboard payloads cleanly with `ValueError("invalid_args")`, and
provides strictly linear-time `O(n)` Consent U5 sanitization (`sanitize_wake_text`) so images,
base64 blobs, raw OS paths, and credentials are never persisted in Wake Inbox items or job
summaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_SELECTION_CHARS = 1200
MAX_CLIPBOARD_WRITE_CHARS = 4000
MAX_WAKE_INPUT_CHARS = 16_384
MAX_WAKE_PREBOUND_CHARS = 65_536

REDACTED_MARKER = "[REDACTED]"
IMAGE_REDACTED_MARKER = "[IMAGE_REDACTED]"
BASE64_REDACTED_MARKER = "[BASE64_REDACTED]"
PATH_REDACTED_MARKER = "[PATH]"

_REDACT_TAGS: tuple[str, ...] = (
    REDACTED_MARKER,
    IMAGE_REDACTED_MARKER,
    BASE64_REDACTED_MARKER,
    PATH_REDACTED_MARKER,
)

_PEM_BEGIN = "-----BEGIN "
_PEM_KEY_SUFFIX = "PRIVATE KEY-----"
_PEM_END = "-----END "

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
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

_UNCLOSED_KV_TAIL_RE = re.compile(
    r"""(?:password|passwd|secret|api_key|apikey|token|access_token)["']?\s*[:=]\s*"""
    r"""(?:["'](?!\[REDACTED\])[^"'\r\n]*$)""",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()]+")
_QUOTED_DOUBLE_FILE_URI_RE = re.compile(r'"file://[^"\r\n]+"', re.IGNORECASE)
_QUOTED_SINGLE_FILE_URI_RE = re.compile(r"'file://[^'\r\n]+'", re.IGNORECASE)
_FILE_URI_RE = re.compile(r"(?<![A-Za-z0-9_])file://[^\s\"'<>()]+", re.IGNORECASE)

_QUOTED_DOUBLE_WIN_UNC_RE = re.compile(r'"\\\\[^"\r\n\\]+\\[^"\r\n]+"')
_QUOTED_SINGLE_WIN_UNC_RE = re.compile(r"'\\\\[^'\r\n\\]+\\[^'\r\n]+'")
_QUOTED_DOUBLE_WIN_DRIVE_RE = re.compile(r'"[A-Za-z]:[\\/][^"\r\n]+"')
_QUOTED_SINGLE_WIN_DRIVE_RE = re.compile(r"'[A-Za-z]:[\\/][^'\r\n]+'")

_QUOTED_DOUBLE_POSIX_PATH_RE = re.compile(r'"(?:~|\.{1,2})?/[^"\r\n/]+/[^"\r\n]+"')
_QUOTED_SINGLE_POSIX_PATH_RE = re.compile(r"'(?:~|\.{1,2})?/[^'\r\n/]+/[^'\r\n]+'")
_POSIX_MULTI_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./])(?:~|\.{1,2})?/(?:[^\s/\"'<>(),;]+/)+[^\s/\"'<>(),;]+"
)

_RAW_BASE64_BLOB_RE = re.compile(
    r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{80,}={0,2}(?![A-Za-z0-9+/_=-])"
)

_WIN_PATH_START_RE = re.compile(r"(?:(?<!\\)\\\\[A-Za-z0-9_.-]+\\|(?<!\w)[A-Za-z]:[\\/])")
_WIN_STOP_CHARS = frozenset("\"'<>()[]{},;\r\n")
_FILE_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")


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


def redact_pem_linear(text: str, *, redact_unclosed: bool = False) -> tuple[str, int]:
    """Replaces PEM private-key blocks with `[REDACTED]` in strictly `O(n)` time."""
    if _PEM_BEGIN not in text:
        return text, 0
    out: list[str] = []
    pos = 0
    hits = 0
    n = len(text)
    while pos < n:
        start = text.find(_PEM_BEGIN, pos)
        if start == -1:
            out.append(text[pos:])
            break
        line_end = text.find("\n", start)
        header_end = n if line_end == -1 else line_end
        header = text[start:header_end]
        key_end_idx = header.find(_PEM_KEY_SUFFIX)
        if key_end_idx == -1:
            out.append(text[pos : start + len(_PEM_BEGIN)])
            pos = start + len(_PEM_BEGIN)
            continue
        label = header[len(_PEM_BEGIN) : key_end_idx + len(_PEM_KEY_SUFFIX)]
        if not all(ch.isupper() or ch == " " or ch == "-" for ch in label):
            out.append(text[pos : start + len(_PEM_BEGIN)])
            pos = start + len(_PEM_BEGIN)
            continue
        search_from = start + key_end_idx + len(_PEM_KEY_SUFFIX)
        end_marker_start = text.find(_PEM_END, search_from)
        matched_end = -1
        while end_marker_start != -1:
            end_line = text.find("\n", end_marker_start)
            end_hdr_limit = n if end_line == -1 else end_line
            end_hdr = text[end_marker_start:end_hdr_limit]
            suffix_pos = end_hdr.find(_PEM_KEY_SUFFIX)
            if suffix_pos != -1:
                matched_end = end_marker_start + suffix_pos + len(_PEM_KEY_SUFFIX)
                break
            end_marker_start = text.find(_PEM_END, end_marker_start + len(_PEM_END))
        if matched_end != -1:
            out.append(text[pos:start])
            out.append(REDACTED_MARKER)
            hits += 1
            pos = matched_end
        elif redact_unclosed:
            out.append(text[pos:start])
            out.append(REDACTED_MARKER)
            hits += 1
            pos = n
        else:
            out.append(text[pos:])
            break
    return "".join(out), hits


def redact_sensitive_text(text: str) -> RedactionResult:
    """Replaces known secret token patterns with `[REDACTED]` and counts hits."""
    current, total_hits = redact_pem_linear(text, redact_unclosed=False)
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


def redact_data_images_linear(text: str) -> str:
    """Replaces `data:image/...` URIs (base64, percent-encoded, or inline SVG) in `O(n)` time."""
    lower = text.lower()
    if "data:image/" not in lower:
        return text
    out: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        idx = lower.find("data:image/", pos)
        if idx == -1:
            out.append(text[pos:])
            break
        out.append(text[pos:idx])
        comma = text.find(",", idx + 11, min(n, idx + 140))
        if comma == -1 or any(ch.isspace() or ch in "\"'" for ch in text[idx:comma]):
            out.append(text[idx : idx + 11])
            pos = idx + 11
            continue
        payload_start = comma + 1
        svg_start = payload_start
        while svg_start < n and text[svg_start] in (" ", "\t"):
            svg_start += 1
        if lower.startswith("<svg", svg_start):
            open_gt = text.find(">", svg_start + 4)
            if open_gt == -1:
                line_end = text.find("\n", svg_start)
                pos = n if line_end == -1 else line_end
            elif text[open_gt - 1] == "/":
                pos = open_gt + 1
            else:
                close_svg = lower.find("</svg>", open_gt + 1)
                if close_svg != -1:
                    pos = close_svg + 6
                else:
                    line_end = text.find("\n", open_gt + 1)
                    eol = n if line_end == -1 else line_end
                    pos = eol if text.find("<", open_gt + 1, eol) != -1 else open_gt + 1
        else:
            end = payload_start
            while end < n and not text[end].isspace() and text[end] not in "\"'<>()":
                end += 1
            pos = end
        out.append(IMAGE_REDACTED_MARKER)
    return "".join(out)


def _trim_sentence_dot(text: str, start: int, end: int) -> int:
    """Preserves a trailing sentence-ending period after a path filename."""
    n = len(text)
    if end > start + 1 and text[end - 1] == "." and (end == n or text[end].isspace()):
        return end - 1
    return end


def _consume_windows_path(text: str, start: int, sep: str) -> int:
    """Advances linearly from a Windows drive/UNC start through path segments (incl. spaces)."""
    n = len(text)
    pos = start
    while pos < n:
        while pos < n and not text[pos].isspace() and text[pos] not in _WIN_STOP_CHARS:
            pos += 1
        trimmed = _trim_sentence_dot(text, start, pos)
        if trimmed != pos:
            pos = trimmed
            break
        if pos >= n or text[pos] != " ":
            break
        # Stop right after a likely filename extension (e.g. `.pdf`, `.txt`, `.xlsx`).
        last_sep = text.rfind(sep, start, pos)
        seg_start = start if last_sep == -1 else last_sep + 1
        if _FILE_EXT_RE.search(text[seg_start:pos]) is not None:
            break
        # Look ahead at most 64 chars (and at most 2 spaces) for a continuation separator.
        look_limit = min(n, pos + 64)
        probe = pos
        spaces_seen = 0
        next_sep = -1
        while probe < look_limit:
            ch = text[probe]
            if ch in _WIN_STOP_CHARS or ch in ("=", ":", "\t"):
                break
            if ch == sep:
                # Reject starting a separate UNC path (`\\host\...`).
                if probe > pos and text[probe - 1] == " " and text[probe : probe + 2] == "\\\\":
                    break
                next_sep = probe
                break
            if ch == " ":
                spaces_seen += 1
                if spaces_seen > 2:
                    break
            probe += 1
        if next_sep != -1:
            pos = next_sep + 1
            continue
        # Handle a single space inside a final filename (e.g. `D:/Tài liệu/bí mật.txt`).
        word_start = pos + 1
        word_end = word_start
        while (
            word_end < look_limit
            and not text[word_end].isspace()
            and text[word_end] not in _WIN_STOP_CHARS
        ):
            word_end += 1
        word_trimmed = _trim_sentence_dot(text, word_start, word_end)
        if word_trimmed > word_start and _FILE_EXT_RE.search(text[word_start:word_trimmed]):
            pos = word_trimmed
        break
    return pos


def redact_unquoted_windows_paths(text: str) -> str:
    """Replaces unquoted Windows drive-letter and UNC paths in `O(n)` time."""
    if "\\" not in text and ":/" not in text:
        return text
    out: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        match = _WIN_PATH_START_RE.search(text, pos)
        if match is None:
            out.append(text[pos:])
            break
        start = match.start()
        matched_prefix = match.group(0)
        sep = "/" if matched_prefix.endswith("/") else "\\"
        end = _consume_windows_path(text, match.end(), sep)
        out.append(text[pos:start])
        out.append(PATH_REDACTED_MARKER)
        pos = end
    return "".join(out)


def _redact_posix_paths_preserving_urls(text: str) -> str:
    """Redacts unquoted multi-segment POSIX paths while leaving `scheme://...` URLs intact."""
    if "/" not in text:
        return text
    url_spans = [(m.start(), m.end()) for m in _URL_RE.finditer(text)]
    if not url_spans:
        return _POSIX_MULTI_PATH_RE.sub(PATH_REDACTED_MARKER, text)
    out: list[str] = []
    pos = 0
    for u_start, u_end in url_spans:
        if u_start > pos:
            out.append(_POSIX_MULTI_PATH_RE.sub(PATH_REDACTED_MARKER, text[pos:u_start]))
        out.append(text[u_start:u_end])
        pos = u_end
    if pos < len(text):
        out.append(_POSIX_MULTI_PATH_RE.sub(PATH_REDACTED_MARKER, text[pos:]))
    return "".join(out)


def _clamp_clean_marker_boundary(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    cut = max_chars
    for tag in _REDACT_TAGS:
        tag_len = len(tag)
        start = max(0, max_chars - tag_len + 1)
        end = min(len(text), max_chars + tag_len - 1)
        idx = text.find(tag, start, end)
        if idx != -1 and idx < cut and idx + tag_len > max_chars:
            cut = idx
    return text[:cut]


def sanitize_wake_text(
    text: str,
    *,
    max_chars: int = 2000,
    max_input_chars: int = MAX_WAKE_INPUT_CHARS,
) -> str:
    """Sanitizes Wake Inbox and job summary text per Consent U5 in strictly `O(n)` time."""
    if not text:
        return ""
    prebound_cap = max(max_input_chars * 4, MAX_WAKE_PREBOUND_CHARS)
    cleaned = text[:prebound_cap]
    cleaned = _CONTROL_CHARS_RE.sub("", cleaned)
    cleaned = redact_data_images_linear(cleaned)
    cleaned, _ = redact_pem_linear(cleaned, redact_unclosed=True)
    cleaned = redact_sensitive_text(cleaned).text
    cleaned = _UNCLOSED_KV_TAIL_RE.sub(REDACTED_MARKER, cleaned)
    cleaned = _QUOTED_DOUBLE_FILE_URI_RE.sub(f'"{PATH_REDACTED_MARKER}"', cleaned)
    cleaned = _QUOTED_SINGLE_FILE_URI_RE.sub(f"'{PATH_REDACTED_MARKER}'", cleaned)
    cleaned = _FILE_URI_RE.sub(PATH_REDACTED_MARKER, cleaned)
    cleaned = _QUOTED_DOUBLE_WIN_UNC_RE.sub(f'"{PATH_REDACTED_MARKER}"', cleaned)
    cleaned = _QUOTED_SINGLE_WIN_UNC_RE.sub(f"'{PATH_REDACTED_MARKER}'", cleaned)
    cleaned = _QUOTED_DOUBLE_WIN_DRIVE_RE.sub(f'"{PATH_REDACTED_MARKER}"', cleaned)
    cleaned = _QUOTED_SINGLE_WIN_DRIVE_RE.sub(f"'{PATH_REDACTED_MARKER}'", cleaned)
    cleaned = redact_unquoted_windows_paths(cleaned)
    cleaned = _RAW_BASE64_BLOB_RE.sub(BASE64_REDACTED_MARKER, cleaned)
    cleaned = _QUOTED_DOUBLE_POSIX_PATH_RE.sub(f'"{PATH_REDACTED_MARKER}"', cleaned)
    cleaned = _QUOTED_SINGLE_POSIX_PATH_RE.sub(f"'{PATH_REDACTED_MARKER}'", cleaned)
    cleaned = _redact_posix_paths_preserving_urls(cleaned)
    bounded = _clamp_clean_marker_boundary(cleaned.strip(), max_chars).strip()
    final = redact_sensitive_text(bounded).text.strip()
    if len(final) > max_chars:
        final = _clamp_clean_marker_boundary(final, max_chars).strip()
    return final
