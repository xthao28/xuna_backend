"""Pure-stdlib validation and sanitization for host-declared `remediation` payloads.

Enforces `XUNA_XTP_V1_CONTRACT.md` §5.1 bounds (`summary`, optional `command`, `kind`),
single-line printable text, and fail-closed rejection of host filesystem paths or secrets.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from xuna_backend.modules.actions.domain.redaction import (
    _FILE_URI_RE,
    _POSIX_MULTI_PATH_RE,
    _QUOTED_DOUBLE_FILE_URI_RE,
    _QUOTED_DOUBLE_POSIX_PATH_RE,
    _QUOTED_DOUBLE_WIN_DRIVE_RE,
    _QUOTED_DOUBLE_WIN_UNC_RE,
    _QUOTED_SINGLE_FILE_URI_RE,
    _QUOTED_SINGLE_POSIX_PATH_RE,
    _QUOTED_SINGLE_WIN_DRIVE_RE,
    _QUOTED_SINGLE_WIN_UNC_RE,
    _UNCLOSED_KV_TAIL_RE,
    _URL_RE,
    redact_data_images_linear,
    redact_pem_linear,
    redact_sensitive_text,
    redact_unquoted_windows_paths,
)

MAX_REMEDIATION_TEXT_CHARS = 240
_ALLOWED_REMEDIATION_KEYS = frozenset({"summary", "command", "kind"})
_ALLOWED_REMEDIATION_KINDS = frozenset({"install", "config", "manual"})
_POSIX_DIR_OR_REL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.:/\\])(?:/[^\s/\"'`<>(),;]+/)"
    r"|(?:^|[\s\"'`(\[{=:,])(?:~|\.{1,2})[/\\][^\s\"'`<>(),;]*"
)
_POSIX_ROOT_DIR_NAMES_LOWER = frozenset(
    {
        "users",
        "home",
        "etc",
        "var",
        "tmp",
        "opt",
        "usr",
        "bin",
        "sbin",
        "root",
        "system",
        "library",
        "applications",
        "volumes",
        "private",
        "dev",
        "proc",
        "sys",
    }
)
_POSIX_SINGLE_ROOT_SEG_RE = re.compile(
    r"(?:^|[\s\"'`(\[{=:,])/(?P<seg>[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)"
    r"(?=$|[\s\"'`)\],;:]|\.(?=$|\s))"
)


def _has_control_or_invalid_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in text)


def _strip_non_file_urls(text: str) -> str:
    if "://" not in text:
        return text
    out: list[str] = []
    pos = 0
    for match in _URL_RE.finditer(text):
        out.append(text[pos : match.start()])
        out.append(" ")
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def _contains_host_path(text: str) -> bool:
    if (
        _QUOTED_DOUBLE_FILE_URI_RE.search(text) is not None
        or _QUOTED_SINGLE_FILE_URI_RE.search(text) is not None
        or _FILE_URI_RE.search(text) is not None
        or _QUOTED_DOUBLE_WIN_UNC_RE.search(text) is not None
        or _QUOTED_SINGLE_WIN_UNC_RE.search(text) is not None
        or _QUOTED_DOUBLE_WIN_DRIVE_RE.search(text) is not None
        or _QUOTED_SINGLE_WIN_DRIVE_RE.search(text) is not None
        or _QUOTED_DOUBLE_POSIX_PATH_RE.search(text) is not None
        or _QUOTED_SINGLE_POSIX_PATH_RE.search(text) is not None
    ):
        return True
    if redact_unquoted_windows_paths(text) != text:
        return True
    without_urls = _strip_non_file_urls(text)
    if _POSIX_MULTI_PATH_RE.search(without_urls) is not None:
        return True
    if _POSIX_DIR_OR_REL_PATH_RE.search(without_urls) is not None:
        return True
    return any(
        m.group("seg").lower() in _POSIX_ROOT_DIR_NAMES_LOWER
        for m in _POSIX_SINGLE_ROOT_SEG_RE.finditer(without_urls)
    )


def _contains_secret(text: str) -> bool:
    if redact_data_images_linear(text) != text:
        return True
    _, pem_hits = redact_pem_linear(text, redact_unclosed=True)
    if pem_hits > 0:
        return True
    if redact_sensitive_text(text).redacted_count > 0:
        return True
    return _UNCLOSED_KV_TAIL_RE.search(text) is not None


def _validate_text_field(raw: object) -> str | None:
    if not isinstance(raw, str) or _has_control_or_invalid_utf8(raw):
        return None
    cleaned = raw.strip()
    if not 1 <= len(cleaned) <= MAX_REMEDIATION_TEXT_CHARS:
        return None
    if _contains_host_path(cleaned) or _contains_secret(cleaned):
        return None
    return cleaned


def sanitize_remediation(raw: object) -> dict[str, object] | None:
    """Validates and normalizes `structuredContent.remediation`; returns `None` fail-closed."""
    if not isinstance(raw, Mapping):
        return None
    if not set(raw.keys()).issubset(_ALLOWED_REMEDIATION_KEYS):
        return None
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in _ALLOWED_REMEDIATION_KINDS:
        return None
    summary = _validate_text_field(raw.get("summary"))
    if summary is None:
        return None
    normalized: dict[str, object] = {"summary": summary, "kind": kind}
    if "command" in raw and raw["command"] is not None:
        command = _validate_text_field(raw["command"])
        if command is None:
            return None
        normalized["command"] = command
    return normalized
