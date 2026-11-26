"""Pure-stdlib validation and sanitization for ephemeral `xuna/preflight` previews.

Enforces `XUNA_XTP_V1_CONTRACT.md` §4.2.8 (`ADR-0019` §2 / `W4-M2`) bounds:
- `kind == "diff"`: `summary` (1..160 single-line chars), `snippet` (1..2048 UTF-8 bytes,
  <= 40 lines, redacted for sensitive tokens, ending with `"…"` when `truncated` is True).
- `kind == "paths"`: `summary`, `paths` (0..20 workspace-relative paths without parent
  traversal, absolute prefixes, or secret/credential path segments), `totalCount >= len(paths)`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from xuna_backend.modules.actions.domain.redaction import (
    redact_data_images_linear,
    redact_pem_linear,
    redact_sensitive_text,
)

MAX_PREVIEW_SUMMARY_CHARS = 160
MAX_PREVIEW_SNIPPET_BYTES = 2048
MAX_PREVIEW_SNIPPET_LINES = 40
MAX_PREVIEW_PATHS = 20
MAX_PREVIEW_PATH_CHARS = 260
PREVIEW_TRUNCATION_MARKER = "\u2026"

_DIFF_KEYS = frozenset({"kind", "summary", "snippet", "truncated"})
_PATHS_KEYS = frozenset({"kind", "summary", "paths", "totalCount", "truncated"})
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_SECRET_DIR_SEGMENTS = frozenset(
    {".ssh", ".gnupg", ".aws", ".kube", ".password-store", "keychains"}
)
_SECRET_EXACT_NAMES = frozenset(
    {".npmrc", ".pypirc", ".pgpass", ".htpasswd", ".netrc", ".git-credentials"}
)
_SECRET_EXTENSIONS = (".pem", ".key", ".p12", ".pfx", ".jks")
_SECRET_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa")


def _has_invalid_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


def _has_single_line_control(text: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in text)


def _has_snippet_control(text: str) -> bool:
    return any((ord(ch) < 32 and ch not in ("\n", "\t")) or ord(ch) == 127 for ch in text)


def _is_forbidden_relative_path(rel_path: str) -> bool:
    if (
        not rel_path
        or len(rel_path) > MAX_PREVIEW_PATH_CHARS
        or rel_path.strip() != rel_path
        or _has_invalid_utf8(rel_path)
        or _has_single_line_control(rel_path)
    ):
        return True
    if rel_path.startswith(("/", "\\", "~")) or _DRIVE_PREFIX_RE.match(rel_path) is not None:
        return True
    normalized = rel_path.replace("\\", "/")
    segments = [seg for seg in normalized.split("/") if seg]
    if not segments or ".." in segments:
        return True
    lower_segs = [seg.lower() for seg in segments]
    for idx, seg in enumerate(lower_segs):
        if seg in _SECRET_DIR_SEGMENTS or seg in _SECRET_EXACT_NAMES:
            return True
        nxt = lower_segs[idx + 1] if idx + 1 < len(lower_segs) else None
        if seg.startswith(".") and nxt in (
            "config",
            "config.json",
            "credentials",
            "credentials.json",
        ):
            return True
    name = lower_segs[-1]
    if name == ".env" or name.startswith(".env."):
        return True
    return name.endswith(_SECRET_EXTENSIONS) or name.startswith(_SECRET_PREFIXES)


def _redact_text(text: str) -> str:
    cleaned = redact_data_images_linear(text)
    cleaned, _ = redact_pem_linear(cleaned, redact_unclosed=True)
    return redact_sensitive_text(cleaned).text


def _truncate_utf8_with_marker(text: str, max_bytes: int) -> str:
    marker_bytes = PREVIEW_TRUNCATION_MARKER.encode("utf-8")
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        if text.endswith(PREVIEW_TRUNCATION_MARKER):
            return text
        if len(raw) + len(marker_bytes) <= max_bytes:
            return f"{text}{PREVIEW_TRUNCATION_MARKER}"
    limit = max(0, max_bytes - len(marker_bytes))
    trimmed = raw[:limit].decode("utf-8", errors="ignore")
    return f"{trimmed}{PREVIEW_TRUNCATION_MARKER}"


def sanitize_preflight_preview(raw: object) -> dict[str, object] | None:
    """Validates and sanitizes an ephemeral preflight preview; returns `None` fail-closed."""
    if not isinstance(raw, Mapping):
        return None
    kind = raw.get("kind")
    truncated = raw.get("truncated")
    summary_raw = raw.get("summary")
    if (
        not isinstance(kind, str)
        or not isinstance(truncated, bool)
        or not isinstance(summary_raw, str)
        or _has_invalid_utf8(summary_raw)
        or _has_single_line_control(summary_raw)
    ):
        return None
    summary = _redact_text(summary_raw).strip()
    if not 1 <= len(summary) <= MAX_PREVIEW_SUMMARY_CHARS or _has_single_line_control(summary):
        return None

    if kind == "diff":
        if set(raw.keys()) != _DIFF_KEYS:
            return None
        snippet_raw = raw.get("snippet")
        if (
            not isinstance(snippet_raw, str)
            or not snippet_raw
            or _has_invalid_utf8(snippet_raw)
            or _has_snippet_control(snippet_raw)
        ):
            return None
        if len(snippet_raw.encode("utf-8")) > MAX_PREVIEW_SNIPPET_BYTES:
            return None
        if len(snippet_raw.split("\n")) > MAX_PREVIEW_SNIPPET_LINES:
            return None
        if truncated and not snippet_raw.endswith(PREVIEW_TRUNCATION_MARKER):
            return None
        redacted_snippet = _redact_text(snippet_raw)
        if _has_snippet_control(redacted_snippet):
            return None
        eff_truncated = truncated
        lines = redacted_snippet.split("\n")
        if len(lines) > MAX_PREVIEW_SNIPPET_LINES:
            lines = [*lines[: MAX_PREVIEW_SNIPPET_LINES - 1], PREVIEW_TRUNCATION_MARKER]
            redacted_snippet = "\n".join(lines)
            eff_truncated = True
        if len(redacted_snippet.encode("utf-8")) > MAX_PREVIEW_SNIPPET_BYTES:
            redacted_snippet = _truncate_utf8_with_marker(
                redacted_snippet, MAX_PREVIEW_SNIPPET_BYTES
            )
            eff_truncated = True
        return {
            "kind": "diff",
            "summary": summary,
            "snippet": redacted_snippet,
            "truncated": eff_truncated,
        }

    if kind == "paths":
        if set(raw.keys()) != _PATHS_KEYS:
            return None
        raw_paths = raw.get("paths")
        total_count = raw.get("totalCount")
        if (
            not isinstance(raw_paths, list)
            or len(raw_paths) > MAX_PREVIEW_PATHS
            or not isinstance(total_count, int)
            or isinstance(total_count, bool)
            or total_count < len(raw_paths)
        ):
            return None
        normalized_paths: list[str] = []
        for entry in raw_paths:
            if not isinstance(entry, str) or _is_forbidden_relative_path(entry):
                return None
            normalized_paths.append(entry.replace("\\", "/"))
        return {
            "kind": "paths",
            "summary": summary,
            "paths": normalized_paths,
            "totalCount": total_count,
            "truncated": truncated,
        }

    return None
