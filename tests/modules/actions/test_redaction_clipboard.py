"""Unit tests for `actions.domain.redaction` (Phase 3 Universal Clipboard guardrails)."""

from __future__ import annotations

import pytest

from xuna_backend.modules.actions.domain.redaction import (
    MAX_CLIPBOARD_WRITE_CHARS,
    MAX_SELECTION_CHARS,
    REDACTED_MARKER,
    contains_binary_or_control,
    prepare_selection_for_clipboard,
    redact_sensitive_text,
    validate_clipboard_payload,
)


@pytest.mark.parametrize(
    "secret",
    [
        "sk-1234567890abcdef",
        "ghp_1234567890abcdef",
        "github_pat_1234567890abcdef",
        "AIzaSyA1234567890abcdefg",
        "xoxb-1234567890-abcdef",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def",
        "password=superSecret99",
        "api_key: 'my_secret_value'",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
    ],
)
def test_redact_sensitive_text_replaces_known_secret_patterns(secret: str) -> None:
    res = redact_sensitive_text(f"prefix {secret} suffix")
    assert secret not in res.text
    assert REDACTED_MARKER in res.text
    assert res.redacted_count >= 1


def test_prepare_selection_for_clipboard_redacts_and_caps_at_1200_chars() -> None:
    token_value = "sk-" + "secretValue123456"
    raw = f"Header token={token_value} " + ("a" * 1500)
    res = prepare_selection_for_clipboard(raw)
    assert token_value not in res.text
    assert REDACTED_MARKER in res.text
    assert len(res.text) <= MAX_SELECTION_CHARS
    assert res.char_count == len(res.text)
    assert res.truncated is True
    assert res.redacted_count >= 1


@pytest.mark.parametrize(
    "bad_input",
    [
        None,
        123,
        "",
        "   \n\t  ",
        "hello\x00world",
        "ctrl\x07bell",
        "del\x7fchar",
        "bad\ud800surrogate",
    ],
)
def test_clipboard_validators_fail_closed_on_binary_empty_or_invalid_utf8(
    bad_input: object,
) -> None:
    if isinstance(bad_input, str) and bad_input.strip():
        assert contains_binary_or_control(bad_input) is True
    with pytest.raises(ValueError, match="invalid_args"):
        prepare_selection_for_clipboard(bad_input)
    with pytest.raises(ValueError, match="invalid_args"):
        validate_clipboard_payload(bad_input)


def test_validate_clipboard_payload_rejects_oversized_and_allows_newlines_tabs() -> None:
    ok = validate_clipboard_payload("line1\n\tline2\r\nline3")
    assert ok.text == "line1\n\tline2\r\nline3"
    assert ok.redacted_count == 0

    with pytest.raises(ValueError, match="invalid_args"):
        validate_clipboard_payload("x" * (MAX_CLIPBOARD_WRITE_CHARS + 1))

    with pytest.raises(ValueError, match="invalid_args"):
        validate_clipboard_payload("x" * (MAX_SELECTION_CHARS + 1), max_chars=MAX_SELECTION_CHARS)


@pytest.mark.parametrize(
    ("secret_line", "leaked_fragment"),
    [
        ('password="my secret pass 1234"', "my secret pass 1234"),
        ("api_key: 'my spaced secret key 42'", "my spaced secret key 42"),
        ('token="sk-1234567890abcdef extra words"', "extra words"),
    ],
)
def test_redact_sensitive_text_handles_quoted_secrets_with_spaces_and_idempotency(
    secret_line: str,
    leaked_fragment: str,
) -> None:
    res = redact_sensitive_text(f"before {secret_line} after")
    assert leaked_fragment not in res.text
    assert REDACTED_MARKER in res.text
    assert res.redacted_count >= 1
    assert res.text == "before [REDACTED] after"

    for already in ('password="[REDACTED]"', "api_key: '[REDACTED]'", "token=[REDACTED]"):
        idem = redact_sensitive_text(already)
        assert idem.redacted_count == 0
        assert idem.text == already


def test_redact_sensitive_text_aws_access_key_parity_and_boundaries() -> None:
    """Tier 5 (Gap 3): AWS AKIA/ASIA access key IDs match xuna_xtp redaction parity."""
    for aws_key in ("AKIAIOSFODNN7EXAMPLE", "ASIA1234567890ABCDEF"):
        res = redact_sensitive_text(f"aws={aws_key} done")
        assert aws_key not in res.text
        assert res.text == f"aws={REDACTED_MARKER} done"
        assert res.redacted_count == 1

        clip = validate_clipboard_payload(aws_key)
        assert clip.text == REDACTED_MARKER
        assert clip.redacted_count == 1

        sel = prepare_selection_for_clipboard(f"key {aws_key}")
        assert aws_key not in sel.text
        assert sel.redacted_count >= 1

    # Boundary: adjacent alphanumeric characters must not trigger false positives
    for non_key in ("XAKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLEZ"):
        res = redact_sensitive_text(non_key)
        assert res.text == non_key
        assert res.redacted_count == 0
