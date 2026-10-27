"""Persona rules of the text turn (no database)."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Locale,
    Role,
    TextDelta,
)
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


def test_persona_leaves_the_character_to_the_device() -> None:
    # PE-4b: "tò mò" is no longer fixed; it comes from the personality label.
    assert "tò mò" not in PERSONA_VI


def test_context_voices_the_device_personality_and_mood() -> None:
    from xuna_backend.modules.conversation.application.chat_turn import (
        PetMood,
        PetSelf,
        PetTemperament,
        TurnContext,
    )

    def system(pet: PetSelf) -> str:
        return build_request("chào", Locale.VI, context=TurnContext(pet=pet)).messages[0].content

    quiet = system(PetSelf(temperament=PetTemperament.QUIET, mood=PetMood.LOW))
    assert "Tính cách của bạn" in quiet and "là trầm" in quiet and "ít lời" in quiet
    assert "Tâm trạng của bạn lúc này là hơi buồn" in quiet
    playful = system(PetSelf(temperament=PetTemperament.PLAYFUL, mood=PetMood.JOYFUL))
    assert "là ham chơi" in playful and "rất vui" in playful
    plain = system(PetSelf())
    assert "Tính cách của bạn" not in plain and "Tâm trạng" not in plain
    # Every label and mood has wording, so a new device value cannot slip through silently.
    for temperament in PetTemperament:
        assert "Tính cách của bạn" in system(PetSelf(temperament=temperament))
    for mood in PetMood:
        assert "Tâm trạng của bạn" in system(PetSelf(mood=mood))


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


def test_context_injects_confirmed_facts_and_personal_name() -> None:
    from xuna_backend.modules.conversation.application.chat_turn import TurnContext

    context = TurnContext(
        confirmed_facts=(
            "Tên của bạn ấy là Thảo",
            "Màu yêu thích của bạn ấy là xanh dương",
            "Món ăn yêu thích của bạn ấy là phở bò",
        )
    )
    request = build_request("chào", Locale.VI, context=context)
    system = request.messages[0].content
    assert "Những điều bạn biết chắc chắn về người bạn đang trò chuyện cùng" in system
    assert "Tên của bạn ấy là Thảo" in system
    assert "Màu yêu thích của bạn ấy là xanh dương" in system
    assert "Món ăn yêu thích của bạn ấy là phở bò" in system
    assert "Người bạn đang nói chuyện cùng tên là Thảo" in system
    assert "gọi bạn ấy là Thảo hoặc bạn Thảo" in system


def test_context_without_confirmed_facts_has_no_fact_injection() -> None:
    from xuna_backend.modules.conversation.application.chat_turn import TurnContext

    context = TurnContext(confirmed_facts=())
    request = build_request("chào", Locale.VI, context=context)
    system = request.messages[0].content
    assert "Những điều bạn biết chắc chắn" not in system
    assert "Người bạn đang nói chuyện cùng tên là" not in system


def test_build_request_auto_detects_language_when_locale_omitted() -> None:
    req_vi = build_request("Chào bạn, hôm nay thế nào?")
    assert req_vi.locale == Locale.VI
    assert "Bạn là XUNA" in req_vi.messages[0].content

    req_en = build_request("Hello, can you help me with Python?")
    assert req_en.locale == Locale.EN
    assert "when speaking English" in req_en.messages[0].content


def test_gesture_tags_are_offered_only_when_the_device_can_play_them() -> None:
    from xuna_backend.modules.conversation.application.chat_turn import TurnContext

    offered = build_request("chào", Locale.VI, context=TurnContext(gestures=True))
    assert "<g:bounce_play>" in offered.messages[0].content
    assert "ngay đầu câu trả lời" in offered.messages[0].content
    plain = build_request("chào", Locale.VI, context=TurnContext())
    assert "<g:" not in plain.messages[0].content


class _ScriptedGenerator:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        for piece in self.pieces:
            yield TextDelta(piece)
        yield ChatFinished("fake", "fake-v1", 1, 1, 0)


def test_turn_sends_the_gesture_first_and_never_the_tag_text() -> None:
    from xuna_backend.modules.conversation.application.chat_turn import GestureChosen, chat_turn
    from xuna_backend.modules.conversation.domain.gestures import ReplyGesture

    async def run() -> list[object]:
        generator = _ScriptedGenerator(["<g:bou", "nce_play> Chơi", " thôi! <g:yawn", "_stretch>"])
        return [event async for event in chat_turn(generator, build_request("chơi", Locale.VI))]

    events = asyncio.run(run())
    assert events[0] == GestureChosen(ReplyGesture.BOUNCE_PLAY)
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "Chơi thôi! "
    assert sum(isinstance(e, GestureChosen) for e in events) == 1
    assert isinstance(events[-1], ChatFinished)
