"""Table-driven tests for ReplySectionFilter and fallback_say (KX-1 §6.3, §9)."""

import pytest

from xuna_backend.modules.conversation.domain.reply_sections import (
    MAX_SAY_CHARS,
    ReplySectionFilter,
    Section,
    fallback_say,
)


def _run_filter(pieces: list[str]) -> tuple[str, str, bool]:
    f = ReplySectionFilter()
    say_parts: list[str] = []
    detail_parts: list[str] = []
    for piece in pieces:
        for sec, txt in f.feed(piece):
            if sec == Section.SAY:
                say_parts.append(txt)
            else:
                detail_parts.append(txt)
    for sec, txt in f.flush():
        if sec == Section.SAY:
            say_parts.append(txt)
        else:
            detail_parts.append(txt)
    return "".join(say_parts), "".join(detail_parts), f.tagged


@pytest.mark.parametrize(
    ("pieces", "expected_say", "expected_detail", "expected_tagged"),
    [
        # Normal single-piece dual output
        (
            ["<say>Ý chính ngắn gọn.</say>## Phân tích\n- Bước 1\n- Bước 2"],
            "Ý chính ngắn gọn.",
            "## Phân tích\n- Bước 1\n- Bước 2",
            True,
        ),
        # Split `<say>` and `</say>` across deltas
        (
            ["  <s", "ay>Chào ", "bạn nhé.</s", "ay>Chi tiết ", "ở đây."],
            "Chào bạn nhé.",
            "Chi tiết ở đây.",
            True,
        ),
        # Missing `<say>` tag -> untagged, everything goes to detail
        (
            ["## Nguyên nhân\nMạch bị sụt áp nguồn 3.3V khi bật phát sóng."],
            "",
            "## Nguyên nhân\nMạch bị sụt áp nguồn 3.3V khi bật phát sóng.",
            False,
        ),
        # Partial `<sa` that turns out not to be `<say>`
        (
            ["<sample>Nội dung</sample>"],
            "",
            "<sample>Nội dung</sample>",
            False,
        ),
        # Short untagged text held in PRE until flush
        (
            ["<sa"],
            "",
            "<sa",
            False,
        ),
        # Stray `<say>` or `</say>` inside DETAIL are stripped
        (
            ["<say>Tóm tắt.</say>Phần đầu <say>giữa</say> phần cuối."],
            "Tóm tắt.",
            "Phần đầu giữa phần cuối.",
            True,
        ),
        # Stray tag split across deltas in DETAIL is stripped
        (
            ["<say>Tóm tắt.</say>Chi tiết </s", "ay> tiếp theo <sa", "y> hết."],
            "Tóm tắt.",
            "Chi tiết  tiếp theo  hết.",
            True,
        ),
        # Whitespace only
        (
            ["   \n\t  "],
            "",
            "",
            False,
        ),
    ],
)
def test_reply_section_filter_cases(
    pieces: list[str],
    expected_say: str,
    expected_detail: str,
    expected_tagged: bool,
) -> None:
    say, detail, tagged = _run_filter(pieces)
    assert say == expected_say
    assert detail == expected_detail
    assert tagged is expected_tagged


def test_reply_section_filter_caps_say_at_max_chars_and_spills_to_detail() -> None:
    long_say = "a" * (MAX_SAY_CHARS + 25)
    say, detail, tagged = _run_filter([f"<say>{long_say}</say> và phần sau"])
    assert tagged is True
    assert len(say) == MAX_SAY_CHARS
    assert say == "a" * MAX_SAY_CHARS
    assert detail == ("a" * 25) + " và phần sau"


def test_gesture_tag_before_say_works_through_gesture_filter() -> None:
    from xuna_backend.modules.conversation.domain.gestures import (
        GestureTagFilter,
        ReplyGesture,
    )

    g_filter = GestureTagFilter()
    s_filter = ReplySectionFilter()
    raw_chunks = ["<g:look_", "around> <s", "ay>Để mình xem.</say>## Chi tiết\nNội dung"]

    chosen = None
    say_out = ""
    detail_out = ""
    for chunk in raw_chunks:
        clean, g = g_filter.feed(chunk)
        if g is not None:
            chosen = g
        if clean:
            for sec, txt in s_filter.feed(clean):
                if sec == Section.SAY:
                    say_out += txt
                else:
                    detail_out += txt
    if rest := g_filter.flush():
        for sec, txt in s_filter.feed(rest):
            if sec == Section.SAY:
                say_out += txt
            else:
                detail_out += txt
    for sec, txt in s_filter.flush():
        if sec == Section.SAY:
            say_out += txt
        else:
            detail_out += txt

    assert chosen == ReplyGesture.LOOK_AROUND
    assert s_filter.tagged is True
    assert say_out == "Để mình xem."
    assert detail_out == "## Chi tiết\nNội dung"


def test_fallback_say_strips_markdown_and_extracts_first_sentence() -> None:
    md = (
        "## Nguyên nhân\n"
        "```c\nint x = 0;\n```\n"
        "- **Mạch nguồn** bị sụt áp khi bật *Wi-Fi*! Cần thêm tụ 470uF.\n"
        "1. Kiểm tra lại đường 3.3V."
    )
    res = fallback_say(md, locale_vi=True)
    assert res == "Nguyên nhân Mạch nguồn bị sụt áp khi bật Wi-Fi!"
    assert len(res) <= 200


def test_fallback_say_truncates_long_sentence_without_punctuation() -> None:
    long_text = " ".join(["từ"] * 80)
    res = fallback_say(long_text, locale_vi=True)
    assert res.endswith("…")
    assert len(res) <= 201


def test_fallback_say_empty_detail_returns_locale_default() -> None:
    assert (
        fallback_say("```python\npass\n```\n# * _", locale_vi=True)
        == "Mình để câu trả lời chi tiết trong app nhé."
    )
    assert fallback_say("   ", locale_vi=False) == "I've put the full answer in the app."
