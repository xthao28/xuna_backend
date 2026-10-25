import pytest

from xuna_backend.modules.conversation.domain.signals import (
    LONG_CHAT_LINES,
    MAX_SIGNALS,
    ConversationSignal,
    conversation_signals,
)

S = ConversationSignal


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("XUNA giỏi quá!", (S.PRAISE,)),
        ("đồ ngốc", (S.SCOLD,)),
        ("mình thương bạn lắm", (S.AFFECTION,)),
        ("nhớ bạn ghê", (S.AFFECTION,)),
        ("mình chơi trò đoán số nhé", (S.PLAYFUL,)),
        ("tại sao trời lại mưa?", (S.CURIOUS_QUESTION,)),
        ("Mặt trăng là gì vậy", (S.CURIOUS_QUESTION,)),
        ("mấy giờ rồi?", ()),
        ("chào XUNA", ()),
    ],
)
def test_owner_words_give_closed_tags(text: str, expected: tuple[ConversationSignal, ...]) -> None:
    assert conversation_signals(text) == expected


def test_lookalikes_and_negations_are_not_tags() -> None:
    assert conversation_signals("đi ngủ thôi") == ()  # "ngủ" is not "ngu"
    assert conversation_signals("mình có yêu cầu này") == ()  # "yêu cầu" is a request
    assert conversation_signals("nhớ là tắt đèn nhé") == ()  # "remember", not "miss you"
    assert conversation_signals("bạn không giỏi lắm") == ()
    assert conversation_signals("đừng chơi nữa") == ()


def test_at_most_two_tags_in_priority_order() -> None:
    tags = conversation_signals("ngoan lắm, thương ghê, mình chơi nhé, tại sao vậy?")
    assert len(tags) == MAX_SIGNALS
    assert tags == (S.PRAISE, S.AFFECTION)
    assert conversation_signals("hư quá nhưng giỏi") == (S.SCOLD, S.PRAISE)


def test_long_chat_is_tagged_once_when_the_talk_gets_long() -> None:
    assert conversation_signals("kể tiếp đi", LONG_CHAT_LINES - 1) == ()
    assert conversation_signals("kể tiếp đi", LONG_CHAT_LINES) == (S.LONG_CHAT,)
    assert conversation_signals("kể tiếp đi", LONG_CHAT_LINES + 2) == ()


def test_unicode_forms_and_case_do_not_matter() -> None:
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "GIỎI QUÁ")
    assert conversation_signals(decomposed) == (S.PRAISE,)
