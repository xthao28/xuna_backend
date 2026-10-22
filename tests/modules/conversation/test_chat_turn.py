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


def test_context_tells_the_time_day_and_what_the_device_learned() -> None:
    # Owner 2026-10-23: XUNA should know the time and its own name and likes.
    from datetime import datetime, timedelta, timezone

    from xuna_backend.modules.conversation.application.chat_turn import (
        HistoryLine,
        PetAction,
        PetSelf,
        TurnContext,
    )

    context = TurnContext(
        local_time=datetime(2026, 10, 23, 21, 5, tzinfo=timezone(timedelta(hours=7))),
        history=(HistoryLine(True, "mình tên Hiếu"), HistoryLine(False, "Chào Hiếu!")),
        pet=PetSelf(
            active_seconds=3 * 86400,
            learned_events=12,
            current=PetAction.CURIOUS,
            quiet_choice=PetAction.IDLE,
            play_choice=PetAction.HAPPY,
        ),
    )
    request = build_request("mấy giờ rồi?", Locale.VI, context=context)
    system = request.messages[0].content
    assert system.startswith(PERSONA_VI)
    assert "Bây giờ là 21:05, thứ Sáu, ngày 23/10/2026" in system
    assert "khoảng 3 ngày" in system and "12 lần" in system
    assert "khi yên tĩnh bạn thích nằm nghỉ" in system
    assert "khi chơi đùa bạn thích nhảy nhót" in system
    assert [m.role for m in request.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]
    assert request.messages[-1].content == "mấy giờ rồi?"


def test_context_without_time_or_teaching_says_so_honestly() -> None:
    from xuna_backend.modules.conversation.application.chat_turn import PetSelf, TurnContext

    system = (
        build_request("chào", Locale.VI, context=TurnContext(pet=PetSelf(active_seconds=600)))
        .messages[0]
        .content
    )
    assert "không biết bây giờ là mấy giờ" in system
    assert "chưa có sở thích rõ ràng" in system


def test_context_rejects_naive_time_and_long_history() -> None:
    from datetime import datetime

    from xuna_backend.modules.conversation.application.chat_turn import (
        HistoryLine,
        TurnContext,
    )

    with pytest.raises(ValueError):
        TurnContext(local_time=datetime(2026, 10, 23, 21, 5))
    with pytest.raises(ValueError):
        TurnContext(history=tuple(HistoryLine(True, "a") for _ in range(13)))
