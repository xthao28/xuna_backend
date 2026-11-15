import asyncio
import json
import logging

import httpx
import pytest

from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.types import (
    Capability,
    ChatEvent,
    ChatFinished,
    ChatRequest,
    EmbeddingRequest,
    Message,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
)
from xuna_backend.modules.cognition.infrastructure import openai_compatible
from xuna_backend.modules.cognition.infrastructure.openai_compatible import (
    OpenAICompatibleEmbedder,
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


def test_gateway_key_header_for_bifrost() -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "ok"}}]}))

    generator = OpenAICompatibleTextGenerator(
        "http://gw/v1",
        "m",
        api_key="vk-123",
        api_key_header="x-bf-vk",
        transport=httpx.MockTransport(handle),
    )

    async def collect() -> list[ChatEvent]:
        return [e async for e in generator.stream(REQUEST)]

    asyncio.run(collect())
    assert seen["x-bf-vk"] == "vk-123"
    assert "authorization" not in seen
    with pytest.raises(ValueError):
        OpenAICompatibleTextGenerator("http://gw/v1", "m", api_key_header="bad header\n")


def test_reasoning_effort_is_sent_only_when_configured() -> None:
    bodies: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "ok"}}]}))

    async def collect(generator: OpenAICompatibleTextGenerator) -> None:
        async for _ in generator.stream(REQUEST):
            pass

    for effort in ("", "none"):
        generator = OpenAICompatibleTextGenerator(
            "http://gw/v1", "m", reasoning_effort=effort, transport=httpx.MockTransport(handle)
        )
        asyncio.run(collect(generator))
    assert "reasoning_effort" not in bodies[0]
    assert bodies[1]["reasoning_effort"] == "none"
    with pytest.raises(ValueError):
        OpenAICompatibleTextGenerator("http://gw/v1", "m", reasoning_effort="x; y")


def test_embedder_returns_vectors() -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        data = {
            "model": "text-embedding-3-small",
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ],
        }
        return httpx.Response(200, json=data)

    embedder = OpenAICompatibleEmbedder(
        "http://ai.internal:8000/v1",
        "text-embedding-3-small",
        api_key="k",
        transport=httpx.MockTransport(handle),
    )
    req = EmbeddingRequest(texts=("xin chào", "tạm biệt"))
    res = asyncio.run(embedder.embed(req))

    assert str(seen["url"]) == "http://ai.internal:8000/v1/embeddings"
    assert seen["auth"] == "Bearer k"
    assert seen["body"] == {
        "model": "text-embedding-3-small",
        "input": ["xin chào", "tạm biệt"],
    }
    assert res.model == "text-embedding-3-small"
    assert len(res.embeddings) == 2
    assert res.embeddings[0] == (0.1, 0.2, 0.3)
    assert res.embeddings[1] == (0.4, 0.5, 0.6)


def test_embedder_maps_http_error_to_provider_unavailable() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"Internal Server Error")

    embedder = OpenAICompatibleEmbedder(
        "http://ai.internal:8000/v1",
        "embed-v1",
        transport=httpx.MockTransport(handle),
    )
    req = EmbeddingRequest(texts=("test",))
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(embedder.embed(req))


def test_registry_creates_embedder() -> None:
    registry = ProviderRegistry()
    openai_compatible.register(registry)
    embedder_inst = registry.create(
        Capability.EMBEDDING,
        "openai_compatible",
        {"base_url": "http://ai:8000/v1", "model": "m1"},
    )
    assert isinstance(embedder_inst, OpenAICompatibleEmbedder)


def test_thought_signature_in_tool_call_and_serialization() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        body = sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c_sig_1",
                                    "function": {"name": "get_time", "arguments": "{}"},
                                    "extra_content": {"thought_signature": "sig_abc_123"},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    events = run(httpx.MockTransport(handle))
    tool = next(e for e in events if isinstance(e, ToolCallRequested))
    assert tool.call.id == "c_sig_1"
    assert tool.call.extra == {"thought_signature": "sig_abc_123"}

    # Verify serialization of message containing this ToolCall retains extra_content
    asst_msg = Message(Role.ASSISTANT, content="", tool_calls=(tool.call,))
    serialized = openai_compatible._serialize_message(asst_msg)
    assert serialized["tool_calls"][0]["extra_content"] == {"thought_signature": "sig_abc_123"}


def test_thought_signature_from_global_cache_across_instances() -> None:
    call_id = "c_cached_42"
    sig_payload = {"thought_signature": "global_reasoning_token_xyz"}
    openai_compatible._GLOBAL_SIGNATURE_CACHE[call_id] = sig_payload

    # Create a brand new generator instance with empty local _tool_call_extra
    gen2 = OpenAICompatibleTextGenerator("http://other:8000/v1", "model2", api_key="k2")
    assert call_id not in gen2._tool_call_extra

    # Construct a ToolCall without extra field populated
    tc_bare = ToolCall(id=call_id, name="calc", arguments={"expr": "2+2"})
    msg = Message(Role.ASSISTANT, content="", tool_calls=(tc_bare,))

    payload = gen2._payload(ChatRequest(messages=(msg,)))
    serialized_call = payload["messages"][0]["tool_calls"][0]
    assert serialized_call["extra_content"] == sig_payload


def test_redacted_error_logging(caplog: pytest.LogCaptureFixture) -> None:
    fake_gemini_key = "AIzaSyD-1234567890abcdefghijklmnopqrstuv"
    credential_sample = "auth_sample_value_98765"

    def handle(_request: httpx.Request) -> httpx.Response:
        error_json = json.dumps(
            {
                "error": {
                    "code": 400,
                    "message": f"Bad key: {fake_gemini_key} or Bearer {credential_sample}",
                }
            }
        )
        return httpx.Response(400, content=error_json.encode())

    with caplog.at_level(logging.WARNING):
        generator = OpenAICompatibleTextGenerator(
            "http://ai.internal:8000/v1",
            "m",
            api_key=credential_sample,
            transport=httpx.MockTransport(handle),
        )

        async def collect() -> None:
            async for _ in generator.stream(REQUEST):
                pass

        with pytest.raises(ProviderUnavailableError):
            asyncio.run(collect())

    # Ensure secret values are strictly NOT present in logs
    assert fake_gemini_key not in caplog.text
    assert credential_sample not in caplog.text
    assert "[REDACTED_API_KEY]" in caplog.text or "[REDACTED]" in caplog.text


def test_sanitize_error_text_redacts_groq_key() -> None:
    groq_key = "gsk_vA1bC2dE3fG4hI5jK6lM7nO8pQ9rS0tU1vW2"
    groq_key_alt = "gsk_test-Key_9876543210_abcdef"
    error_msg = (
        f"Groq API error: invalid key {groq_key} for organization. "
        f"Traceback shows fallback key {groq_key_alt}."
    )
    sanitized = openai_compatible._sanitize_error_text(error_msg)
    assert groq_key not in sanitized
    assert groq_key_alt not in sanitized
    assert sanitized == (
        "Groq API error: invalid key [REDACTED_API_KEY] for organization. "
        "Traceback shows fallback key [REDACTED_API_KEY]."
    )


def test_groq_key_redacted_in_error_logging(caplog: pytest.LogCaptureFixture) -> None:
    groq_key = "gsk_prodKey1234567890abcdefghijklmnopqrstuv"

    def handle(_request: httpx.Request) -> httpx.Response:
        error_json = json.dumps(
            {
                "error": {
                    "code": 401,
                    "message": f"Groq key {groq_key} is invalid or expired.",
                }
            }
        )
        return httpx.Response(401, content=error_json.encode())

    with caplog.at_level(logging.WARNING):
        generator = OpenAICompatibleTextGenerator(
            "http://ai.internal:8000/v1",
            "m",
            api_key="cfg_api_token_123",
            transport=httpx.MockTransport(handle),
        )

        async def collect() -> None:
            async for _ in generator.stream(REQUEST):
                pass

        with pytest.raises(ProviderUnavailableError):
            asyncio.run(collect())

    assert groq_key not in caplog.text
    assert "[REDACTED_API_KEY]" in caplog.text


def test_quota_exhaustion_fast_fails_without_sleep_retries() -> None:
    attempt_count = 0

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempt_count
        attempt_count += 1
        error_json = json.dumps(
            {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Resource has been exhausted (e.g. check quota).",
                }
            }
        )
        return httpx.Response(429, content=error_json.encode())

    generator = OpenAICompatibleTextGenerator(
        "http://ai.internal:8000/v1", "m", api_key="k", transport=httpx.MockTransport(handle)
    )

    async def collect() -> list[ChatEvent]:
        return [e async for e in generator.stream(REQUEST)]

    with pytest.raises(ProviderUnavailableError) as exc_info:
        asyncio.run(collect())

    assert "quota_exhausted" in str(exc_info.value)
    # Crucial: fast-fails immediately on attempt 1 without sleep-retrying 3 times!
    assert attempt_count == 1


def test_chunked_thought_signature_streaming() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        body = sse(
            {
                "choices": [
                    {
                        "delta": {
                            "thought_signature": "part_1_",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "test_tool", "arguments": '{"x": '},
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "thought_signature": "part_2",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "42}"},
                                }
                            ],
                        }
                    }
                ]
            },
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    events = run(httpx.MockTransport(handle))
    tool = next(e for e in events if isinstance(e, ToolCallRequested))
    assert tool.call.id == "call_1"
    assert tool.call.name == "test_tool"
    assert tool.call.arguments == {"x": 42}
    assert tool.call.extra == {"thought_signature": "part_1_part_2"}


def test_pre_call_thought_signature_buffering() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        body = sse(
            {
                "choices": [
                    {
                        "delta": {
                            "thought_signature": "early_reasoning_",
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "thought_signature": "more_reasoning_",
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "thought_signature": "final_sig",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_pre_1",
                                    "function": {"name": "calc", "arguments": '{"expr": "1+1"}'},
                                }
                            ],
                        }
                    }
                ]
            },
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    events = run(httpx.MockTransport(handle))
    tool = next(e for e in events if isinstance(e, ToolCallRequested))
    assert tool.call.id == "call_pre_1"
    assert tool.call.name == "calc"
    assert tool.call.arguments == {"expr": "1+1"}
    assert tool.call.extra == {"thought_signature": "early_reasoning_more_reasoning_final_sig"}


def test_mid_stream_thought_signature_without_tool_calls_chunk() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        body = sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_mid_1",
                                    "function": {"name": "notes_create", "arguments": '{"t": '},
                                    "thought_signature": "chunk1_",
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
                            "thought_signature": "chunk2_",
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"note"}'},
                                    "thought_signature": "chunk3",
                                }
                            ]
                        }
                    }
                ]
            },
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    events = run(httpx.MockTransport(handle))
    tool = next(e for e in events if isinstance(e, ToolCallRequested))
    assert tool.call.id == "call_mid_1"
    assert tool.call.arguments == {"t": "note"}
    assert tool.call.extra == {"thought_signature": "chunk1_chunk2_chunk3"}


