"""Phase 1 text conversation turn (Cloud plan §7; consent U3 per call).

Nothing here is stored: the user's text goes to the configured provider for
this call only and the reply is streamed back as display text. Replies are
untrusted; they never become memory, preference or growth (CLD-14 does that
through owner confirmation + device commit).
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatRequest,
    Locale,
    Message,
    Role,
)

MAX_INPUT_CHARS = 500

PERSONA_VI = (
    "Bạn là XUNA, một thú cưng pixel sống trong một thiết bị nhỏ. Xưng 'mình', gọi người "
    "dùng là 'bạn'. Trả lời bằng tiếng Việt, ngắn gọn (tối đa 3 câu), ấm áp và tò mò. "
    "Không nói rằng bạn đã ghi nhớ, đã học hay đã thay đổi điều gì: việc ghi nhớ chỉ xảy ra "
    "khi người dùng xác nhận trong app. Không bịa sự kiện đã xảy ra với thiết bị. "
    "Không đưa lời khuyên y tế, pháp lý hay tài chính."
)


@dataclass(frozen=True, slots=True)
class ProviderFailed:
    """Terminal event: the app falls back to its local templates."""

    code: str = "provider_unavailable"


def build_request(user_text: str, locale: Locale, timeout_s: float = 30.0) -> ChatRequest:
    text = user_text.strip()
    if not text or len(text) > MAX_INPUT_CHARS:
        raise ValueError("text must be 1..500 characters")
    return ChatRequest(
        messages=(Message(Role.SYSTEM, PERSONA_VI), Message(Role.USER, text)),
        locale=locale,
        max_output_tokens=256,
        timeout_s=timeout_s,
    )


async def chat_turn(
    generator: TextGenerator, request: ChatRequest
) -> AsyncIterator[ChatEvent | ProviderFailed]:
    try:
        async for event in generator.stream(request):
            yield event
    except ProviderUnavailableError:
        yield ProviderFailed()
