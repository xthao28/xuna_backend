"""Tests for bilingual Vietnamese / English language detector."""

import pytest

from xuna_backend.modules.cognition.domain.language_detector import detect_text_language
from xuna_backend.modules.cognition.domain.types import Locale


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Xin chào bạn, hôm nay thế nào?", Locale.VI),
        ("Mình là XUNA, rất vui được gặp bạn.", Locale.VI),
        ("Nấu ăn món này ngon lắm.", Locale.VI),
        ("Hôm nay trời đẹp thật đấy!", Locale.VI),
        ("Chào nhé", Locale.VI),
        ("ban co khoe khong", Locale.VI),
        ("cam on ban rat nhieu", Locale.VI),
        ("Hello XUNA, how are you today?", Locale.EN),
        ("I am fine, thank you very much!", Locale.EN),
        ("What can you do?", Locale.EN),
        ("That sounds awesome, tell me more about it.", Locale.EN),
        ("Yes, I would love to learn more.", Locale.EN),
        ("Good morning!", Locale.EN),
        ("Can you speak English?", Locale.EN),
        ("", Locale.VI),
        ("   ", Locale.VI),
        ("12345", Locale.VI),
    ],
)
def test_detect_text_language(text: str, expected: Locale) -> None:
    assert detect_text_language(text) == expected
