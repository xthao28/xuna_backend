"""Format a confirmed personal fact for the conversation context (KX-1, plan §6.7).

Pure stdlib only.
"""

from typing import Any


def format_fact(content: dict[str, Any]) -> str | None:
    """Format a confirmed fact for inclusion in the conversation context."""
    fact_key = content.get("fact")
    value = content.get("value")
    if isinstance(value, str) and value.strip():
        v = value.strip()
        if fact_key == "owner_name":
            return f"Tên của bạn ấy là {v}"
        if fact_key == "favorite_color":
            return f"Màu yêu thích của bạn ấy là {v}"
        if fact_key == "favorite_food":
            return f"Món ăn yêu thích của bạn ấy là {v}"
        if fact_key == "birthday":
            return f"Sinh nhật của bạn ấy là ngày {v}"
        if fact_key == "pet_companion":
            return f"Bạn ấy có nuôi {v}"

    raw_summary = content.get("summary") or content.get("text")
    if raw_summary and isinstance(raw_summary, str) and raw_summary.strip():
        s = str(raw_summary).strip()
        if s.startswith("Tên của bạn là "):
            return f"Tên của bạn ấy là {s[len('Tên của bạn là ') :]}"
        if s.startswith("Màu yêu thích của bạn là "):
            return f"Màu yêu thích của bạn ấy là {s[len('Màu yêu thích của bạn là ') :]}"
        if s.startswith("Món ăn yêu thích của bạn là "):
            return f"Món ăn yêu thích của bạn ấy là {s[len('Món ăn yêu thích của bạn là ') :]}"
        if s.startswith("Sinh nhật của bạn là "):
            return f"Sinh nhật của bạn ấy là {s[len('Sinh nhật của bạn là ') :]}"
        return s
    return None

