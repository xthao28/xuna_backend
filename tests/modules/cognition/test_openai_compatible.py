import asyncio
import json

import httpx
import pytest

from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.types import (
    Capability,
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Message,
    Role,
    TextDelta,
    ToolCallRequested,
    ToolSpec,
)
from xuna_backend.modules.cognition.infrastructure import openai_compatible
from xuna_backend.modules.cognition.infrastructure.openai_compatible import (
    OpenAICompatibleTextGenerator,
)

REQUEST = ChatRequest(
    messages=(Message(Role.SYSTEM, "Bạn là XUNA."), Message(Role.USER, "Chào XUNA")),
    tools=(ToolSpec("get_time", "Giờ hiện tại", {"type": "object", "properties": {}}),),
)


def sse(*chunks: object) -> bytes:
    lines = [f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def run(handler: httpx.MockTransport, request: ChatRequest = REQUEST) -> list[ChatEvent]:
    generator = OpenAICompatibleTextGenerator(
        "http://ai.internal:8000/v1", "vi-chat", api_key="k", transport=handler
    )

    async def collect() -> list[ChatEvent]:
        return [e async for e in generator.stream(request)]

    return asyncio.run(collect())


def test_streams_vietnamese_text_tools_and_usage() -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        body = sse(
            {"choices": [{"delta": {"content": "Chào "}}]},
            {"choices": [{"delta": {"content": "bạn!"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "get_time", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    events = run(httpx.MockTransport(handle))
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Chào bạn!"
    tool = next(e for e in events if isinstance(e, ToolCallRequested))
    assert (tool.call.name, tool.call.arguments) == ("get_time", {})
    finished = events[-1]
    assert isinstance(finished, ChatFinished)
    assert (finished.input_tokens, finished.output_tokens, finished.model) == (12, 3, "vi-chat")
    assert seen["url"] == "http://ai.internal:8000/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["stream"] is True
    assert body["messages"][1] == {"role": "user", "content": "Chào XUNA"}
    assert body["tools"][0]["function"]["name"] == "get_time"


@pytest.mark.parametrize(
    "handler",
    [
        lambda _r: httpx.Response(503),
        lambda _r: httpx.Response(200, content=b"data: {not json\n\n"),
    ],
)
def test_provider_failures_are_unavailable(handler: object) -> None:
    with pytest.raises(ProviderUnavailableError):
        run(httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_vpn_down_is_unavailable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("vpn down", request=request)

    with pytest.raises(ProviderUnavailableError):
        run(httpx.MockTransport(refuse))


def test_registry_and_config_validation() -> None:
    registry = ProviderRegistry()
    openai_compatible.register(registry)
    created = registry.create(
        Capability.TEXT,
        "openai_compatible",
        {"base_url": "https://x/v1", "model": "m"},
    )
    assert isinstance(created, OpenAICompatibleTextGenerator)
    with pytest.raises(ValueError):
        OpenAICompatibleTextGenerator("ftp://x", "m")
    with pytest.raises(ValueError):
        OpenAICompatibleTextGenerator("https://x", "")
