"""Approval text for the Android `ui.act` tool (R3): it must name what is touched."""

import pytest

from xuna_backend.modules.actions.domain.action_summary import ActionSummary
from xuna_backend.modules.cognition.domain.types import Locale


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        (Locale.VI, "Chạm vào 'Gửi' trên màn hình Pixel"),
        (Locale.EN, "Tap 'Gửi' on Pixel"),
    ],
)
def test_activate_summary_shows_the_element_label_not_the_ref(
    locale: Locale, expected: str
) -> None:
    text = ActionSummary.format_pending(
        "ui.act", {"type": "activate", "ref": "e3", "label": "Gửi"}, "Pixel", locale
    )
    assert text == expected
    assert "e3" not in text


@pytest.mark.parametrize("locale", [Locale.VI, Locale.EN])
def test_set_text_summary_shows_the_exact_text_and_the_field(locale: Locale) -> None:
    text = ActionSummary.format_pending(
        "ui.act",
        {"type": "set_text", "ref": "e2", "label": "Tìm kiếm", "text": "nhạc chill"},
        "Pixel",
        locale,
    )
    assert "nhạc chill" in text
    assert "Tìm kiếm" in text


@pytest.mark.parametrize("locale", [Locale.VI, Locale.EN])
def test_unknown_act_type_gets_a_generic_but_safe_summary(locale: Locale) -> None:
    text = ActionSummary.format_pending("ui.act", {"type": "tap_xy"}, "Pixel", locale)
    assert "Pixel" in text
