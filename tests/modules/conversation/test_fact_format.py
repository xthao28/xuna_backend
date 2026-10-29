"""Unit tests for personal fact formatting (KX-1 §6.7)."""

from xuna_backend.modules.conversation.domain.fact_format import format_fact


def test_format_fact_structured_fields() -> None:
    assert format_fact({"fact": "owner_name", "value": "Thảo"}) == "Tên của bạn ấy là Thảo"
    assert (
        format_fact({"fact": "favorite_color", "value": "xanh dương"})
        == "Màu yêu thích của bạn ấy là xanh dương"
    )
    assert (
        format_fact({"fact": "favorite_food", "value": "phở bò"})
        == "Món ăn yêu thích của bạn ấy là phở bò"
    )
    assert (
        format_fact({"fact": "birthday", "value": "28/10"}) == "Sinh nhật của bạn ấy là ngày 28/10"
    )
    assert (
        format_fact({"fact": "pet_companion", "value": "mèo tên Miu"})
        == "Bạn ấy có nuôi mèo tên Miu"
    )


def test_format_fact_summary_fallbacks() -> None:
    assert format_fact({"summary": "Tên của bạn là Minh"}) == "Tên của bạn ấy là Minh"
    assert (
        format_fact({"summary": "Màu yêu thích của bạn là tím"})
        == "Màu yêu thích của bạn ấy là tím"
    )
    assert format_fact({"text": "Bạn ấy làm kỹ sư nhúng"}) == "Bạn ấy làm kỹ sư nhúng"
    assert format_fact({}) is None
    assert format_fact({"value": "   "}) is None
