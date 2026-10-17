import asyncio

import pytest

from xuna_backend.modules.cognition.application.ports import TextGenerator
from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.types import (
    Capability,
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Message,
    Role,
    TextDelta,
)
from xuna_backend.modules.cognition.infrastructure import echo


def registry() -> ProviderRegistry:
    providers = ProviderRegistry()
    echo.register(providers)
    return providers


def test_registry_selects_adapter_by_name_and_streams() -> None:
    generator = registry().create(Capability.TEXT, "echo", {}, TextGenerator)
    request = ChatRequest(messages=(Message(Role.USER, "xin chào XUNA"),))

    async def collect() -> list[ChatEvent]:
        return [event async for event in generator.stream(request)]

    events = asyncio.run(collect())
    deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(d.text for d in deltas) == "xin chào XUNA"
    assert all(d.untrusted for d in deltas)
    assert isinstance(events[-1], ChatFinished)


def test_unknown_or_duplicate_adapter_fails_fast() -> None:
    providers = registry()
    with pytest.raises(LookupError, match="have \\['echo'\\]"):
        providers.create(Capability.TEXT, "missing", {}, TextGenerator)
    with pytest.raises(ValueError):
        echo.register(providers)
    assert providers.names(Capability.SPEECH_RECOGNITION) == []


def test_chat_request_bounds() -> None:
    with pytest.raises(ValueError):
        ChatRequest(messages=())
    with pytest.raises(ValueError):
        ChatRequest(messages=(Message(Role.USER, "a"),), max_output_tokens=0)
    with pytest.raises(ValueError):
        ChatRequest(messages=(Message(Role.USER, "a"),), timeout_s=0)
