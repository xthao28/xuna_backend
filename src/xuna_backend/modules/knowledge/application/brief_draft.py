"""LLM-assisted draft of a domain expert brief (plan §6.5, Appendix B, KX-3)."""

from __future__ import annotations

import re

from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatRequest,
    Locale,
    Message,
    Role,
    TextDelta,
)

BRIEF_DRAFT_SYSTEM_VI = (
    'Bạn viết "chỉ dẫn cách làm việc" cho một trợ lý AI khi hỗ trợ người dùng '
    "trong một lĩnh vực cụ thể.\n\n"
    "Yêu cầu:\n"
    "- Viết bằng tiếng Việt, dạng 4 đến 6 gạch đầu dòng, tổng cộng tối đa 900 ký tự.\n"
    '- Dòng đầu tiên là: "Cách làm việc chuyên lĩnh vực <tên lĩnh vực>:"\n'
    "- Mỗi gạch đầu dòng nêu một nguyên tắc thực hành cụ thể của lĩnh vực đó: cần kiểm tra "
    "giả định nào, trình bày kết quả theo cấu trúc nào (bảng, công thức, từng bước, ví dụ), "
    "lỗi phổ biến cần tránh, ranh giới an toàn nếu lĩnh vực nhạy cảm.\n"
    "- KHÔNG viết lời chào, KHÔNG đóng vai danh hiệu con người (tiến sĩ, luật sư, bác sĩ...), "
    "KHÔNG bịa thông tin cá nhân của người dùng.\n"
    "- Chỉ xuất đúng nội dung chỉ dẫn, không kèm giải thích ngoài lề."
)

_TAG_RE = re.compile(r"</?say>|<g:[^>]*>", re.IGNORECASE)
_SPACE_RUN_RE = re.compile(r"[^\S\n]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MAX_BRIEF_CHARS = 1500


def _sanitize_draft(raw: str) -> str:
    cleaned = _TAG_RE.sub("", raw)
    lines = [_SPACE_RUN_RE.sub(" ", line).strip() for line in cleaned.splitlines()]
    text = "\n".join(lines).strip()
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    if len(text) <= _MAX_BRIEF_CHARS:
        return text
    cut = text[:_MAX_BRIEF_CHARS]
    last_space = cut.rfind(" ")
    if last_space > 0:
        return cut[:last_space].rstrip()
    return cut.rstrip()


class DraftExpertBriefUseCase:
    def __init__(self, generator: TextGenerator) -> None:
        self._generator = generator

    async def execute(self, name: str, description: str) -> str:
        user_content = f"Tên lĩnh vực: {name}\nMô tả và từ khóa của chủ: {description}"
        request = ChatRequest(
            messages=(
                Message(role=Role.SYSTEM, content=BRIEF_DRAFT_SYSTEM_VI),
                Message(role=Role.USER, content=user_content),
            ),
            locale=Locale.VI,
            max_output_tokens=900,
            timeout_s=30.0,
        )
        chunks: list[str] = []
        async for event in self._generator.stream(request):
            if isinstance(event, TextDelta):
                chunks.append(event.text)
        draft = _sanitize_draft("".join(chunks))
        if not draft:
            raise ProviderUnavailableError("Empty brief draft from provider")
        return draft
