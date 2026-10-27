"""Reply gesture tags (owner 2026-10-27): removed from the text, first valid one chosen."""

import pytest

from xuna_backend.modules.conversation.domain.gestures import GestureTagFilter, ReplyGesture


def _run(pieces: list[str]) -> tuple[str, list[ReplyGesture]]:
    tags = GestureTagFilter()
    text = ""
    chosen: list[ReplyGesture] = []
    for piece in pieces:
        clean, gesture = tags.feed(piece)
        text += clean
        if gesture is not None:
            chosen.append(gesture)
    return text + tags.flush(), chosen


@pytest.mark.parametrize(
    ("pieces", "text", "chosen"),
    [
        (["Chào bạn!"], "Chào bạn!", []),
        (["<g:bounce_play> Chào bạn!"], "Chào bạn!", [ReplyGesture.BOUNCE_PLAY]),
        # Split across deltas: nothing of the tag leaks.
        (["<", "g:greet", "_nuzzle>", " Chào", " bạn"], "Chào bạn", [ReplyGesture.GREET_NUZZLE]),
        # Anywhere in the text, and only the first valid one counts.
        (
            ["Ừ <g:look_around>nhé <g:yawn_stretch>"],
            "Ừ nhé ",
            [ReplyGesture.LOOK_AROUND],
        ),
        # Unknown names are removed but never chosen.
        (["<g:dance>Vui quá"], "Vui quá", []),
        (["<g:dance><g:yawn_stretch>Buồn ngủ"], "Buồn ngủ", [ReplyGesture.YAWN_STRETCH]),
        # A `<` that is not a tag stays text, including one that starts a real tag.
        (["3 < 5 và 2<g", "o"], "3 < 5 và 2<go", []),
        (["a <<g:bounce_play>b"], "a <b", [ReplyGesture.BOUNCE_PLAY]),
        (["<b>đậm</b>"], "<b>đậm</b>", []),
        # An unfinished tag at the end is plain text.
        (["xong <g:bou"], "xong <g:bou", []),
        # Too long to be a tag name.
        (["<g:" + "a" * 21 + ">"], "<g:" + "a" * 21 + ">", []),
    ],
)
def test_filter(pieces: list[str], text: str, chosen: list[ReplyGesture]) -> None:
    assert _run(pieces) == (text, chosen)


def test_leading_whitespace_after_a_tag_is_dropped_once() -> None:
    tags = GestureTagFilter()
    assert tags.feed("<g:look_around>") == ("", ReplyGesture.LOOK_AROUND)
    assert tags.feed("  ") == ("", None)
    assert tags.feed(" Hả?") == ("Hả?", None)
    assert tags.feed(" Gì") == (" Gì", None)
    assert tags.gesture is ReplyGesture.LOOK_AROUND


def test_wire_names_match_the_device_variants() -> None:
    assert [g.value for g in ReplyGesture] == [
        "look_around",
        "yawn_stretch",
        "bounce_play",
        "greet_nuzzle",
    ]
