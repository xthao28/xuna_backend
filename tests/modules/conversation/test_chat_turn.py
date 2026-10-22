"""Persona rules of the text turn (no database)."""

import pytest

from xuna_backend.modules.cognition.domain.types import Locale, Role
from xuna_backend.modules.conversation.application.chat_turn import PERSONA_VI, build_request


def test_persona_hears_the_owner_only_when_they_talk_and_never_sees() -> None:
    # Owner 2026-10-22: speech through the app microphone; no camera, no ambient sound.
    assert "nghe được người dùng qua micro điện thoại" in PERSONA_VI
    assert "không có camera" in PERSONA_VI
    assert "nhìn thấy, nghe thấy tiếng động" in PERSONA_VI
    assert "chỉ đọc được tin nhắn" not in PERSONA_VI


def test_request_puts_the_persona_first_and_only_the_trimmed_text_second() -> None:
    request = build_request("  chào XUNA  ", Locale.VI)
    assert [m.role for m in request.messages] == [Role.SYSTEM, Role.USER]
    assert request.messages[0].content == PERSONA_VI
    assert request.messages[1].content == "chào XUNA"


@pytest.mark.parametrize("text", ["", "   ", "x" * 501])
def test_request_rejects_empty_or_long_text(text: str) -> None:
    with pytest.raises(ValueError):
        build_request(text, Locale.VI)
