import asyncio
import json
from typing import cast

import httpx
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
    ToolCall,
    ToolCallRequested,
    ToolSpec,
)
from xuna_backend.modules.cognition.infrastructure import echo
from xuna_backend.modules.cognition.infrastructure.openai_compatible import (
    OpenAICompatibleTextGenerator,
)


def registry() -> ProviderRegistry:
    providers = ProviderRegistry()
    echo.register(providers)
    return providers


def test_registry_selects_adapter_by_name_and_streams() -> None:
    generator = cast(TextGenerator, registry().create(Capability.TEXT, "echo", {}))
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
        providers.create(Capability.TEXT, "missing", {})
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


def test_message_tool_calls_only_allowed_for_assistant_role() -> None:
    call = ToolCall(id="call_1", name="media__play_pause", arguments={"target": "Mac"})
    msg = Message(Role.ASSISTANT, "", tool_calls=(call,))
    assert msg.tool_calls == (call,)

    with pytest.raises(ValueError, match="tool_calls are only valid for assistant"):
        Message(Role.USER, "hi", tool_calls=(call,))


def test_openai_compatible_payload_snapshot_for_assistant_tool_calls_and_tool_message() -> None:
    gen = OpenAICompatibleTextGenerator(
        base_url="https://llm.example.com/v1",
        model="gpt-test",
    )
    call = ToolCall(
        id="call_abc123",
        name="app__launch",
        arguments={"id": "vscode", "target": "PC Phòng Làm Việc"},
    )
    spec = ToolSpec(
        name="app__launch",
        description="Launch allowlisted app",
        parameters={"type": "object", "properties": {"id": {"type": "string"}}},
    )
    req = ChatRequest(
        messages=(
            Message(Role.USER, "Mở VS Code trên PC"),
            Message(Role.ASSISTANT, "", tool_calls=(call,)),
            Message(
                Role.TOOL,
                '{"status":"ok","output":{"id":"vscode"}}',
                tool_call_id="call_abc123",
            ),
        ),
        tools=(spec,),
    )
    payload = gen._payload(req)
    assert payload["messages"] == [
        {"role": "user", "content": "Mở VS Code trên PC"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "app__launch",
                        "arguments": '{"id": "vscode", "target": "PC Phòng Làm Việc"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"status":"ok","output":{"id":"vscode"}}',
            "tool_call_id": "call_abc123",
        },
    ]
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "app__launch",
                "description": "Launch allowlisted app",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
            },
        }
    ]


def test_openai_compatible_stream_assembles_multiple_tool_calls() -> None:
    sse_chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "media__play_pause", "arguments": '{"tar'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'get":"Mac"}'}},
                            {
                                "index": 1,
                                "id": "call_2",
                                "function": {
                                    "name": "media__volume_set",
                                    "arguments": '{"level":40,',
                                },
                            },
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 1, "function": {"arguments": '"target":"Mac"}'}}]
                    }
                }
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 14},
        },
    ]
    body_lines = [f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in sse_chunks]
    body_lines.append("data: [DONE]\n\n")
    raw_sse = "".join(body_lines).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw_sse, headers={"content-type": "text/event-stream"})

    gen = OpenAICompatibleTextGenerator(
        base_url="https://llm.example.com/v1",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )
    req = ChatRequest(messages=(Message(Role.USER, "Tạm dừng nhạc và chỉnh âm lượng 40 trên Mac"),))

    async def collect() -> list[ChatEvent]:
        return [ev async for ev in gen.stream(req)]

    events = asyncio.run(collect())
    tool_events = [ev for ev in events if isinstance(ev, ToolCallRequested)]
    assert len(tool_events) == 2
    assert tool_events[0].call == ToolCall(
        id="call_1",
        name="media__play_pause",
        arguments={"target": "Mac"},
    )
    assert tool_events[1].call == ToolCall(
        id="call_2",
        name="media__volume_set",
        arguments={"level": 40, "target": "Mac"},
    )
    finished = events[-1]
    assert isinstance(finished, ChatFinished)
    assert finished.input_tokens == 25
    assert finished.output_tokens == 14
