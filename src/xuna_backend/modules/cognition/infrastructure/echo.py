"""Deterministic offline adapters for tests and local runs. Not an AI."""

from collections.abc import AsyncIterator, Mapping
from typing import Any

from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.types import (
    Capability,
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Role,
    TextDelta,
)


class EchoTextGenerator:
    """Streams the last user message back word by word."""

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        last = next(m.content for m in reversed(request.messages) if m.role is Role.USER)
        words = last.split()
        for index, word in enumerate(words):
            yield TextDelta(word if index == 0 else f" {word}")
        yield ChatFinished("echo", "echo-v1", len(words), len(words), 0)


def register(registry: ProviderRegistry) -> None:
    def text(_: Mapping[str, Any]) -> object:
        return EchoTextGenerator()

    registry.register(Capability.TEXT, "echo", text)
