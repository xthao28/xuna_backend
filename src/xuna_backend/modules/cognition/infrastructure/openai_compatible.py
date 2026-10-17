"""Adapter for any OpenAI-style `POST {base}/chat/completions` streaming server.

Covers the owner's internal AI (reached over OpenVPN), vLLM, Ollama `/v1`,
LM Studio, LiteLLM and most hosted vendors, so switching to a rented provider
later is configuration only. Network reachability (VPN) is the host's job;
this adapter only needs a base URL. Prompts and outputs are never logged.
"""

import json
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.types import (
    Capability,
    ChatEvent,
    ChatFinished,
    ChatRequest,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)

NAME = "openai_compatible"


class OpenAICompatibleTextGenerator:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
        provider_label: str = NAME,
    ) -> None:
        if not base_url.startswith(("http://", "https://")) or not model:
            raise ValueError("base_url (http/https) and model are required")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._transport = transport
        self._label = provider_label

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": True,
            "max_tokens": request.max_output_tokens,
            "messages": [
                {"role": m.role.value, "content": m.content}
                | ({"tool_call_id": m.tool_call_id} if m.tool_call_id else {})
                for m in request.messages
            ],
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
        return payload

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        started = time.perf_counter()
        timeout = httpx.Timeout(request.timeout_s, connect=min(5.0, request.timeout_s))
        usage: dict[str, int] = {}
        calls: dict[int, dict[str, str]] = {}
        try:
            async with (
                httpx.AsyncClient(timeout=timeout, transport=self._transport) as client,
                client.stream(
                    "POST", self._url, json=self._payload(request), headers=self._headers
                ) as response,
            ):
                if response.status_code != 200:
                    raise ProviderUnavailableError(f"status {response.status_code}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            yield TextDelta(str(delta["content"]))
                        for call in delta.get("tool_calls") or []:
                            slot = calls.setdefault(
                                int(call.get("index", 0)), {"id": "", "name": "", "args": ""}
                            )
                            function = call.get("function") or {}
                            slot["id"] = call.get("id") or slot["id"]
                            slot["name"] = function.get("name") or slot["name"]
                            slot["args"] += function.get("arguments") or ""
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            raise ProviderUnavailableError(type(error).__name__) from error
        for slot in calls.values():
            try:
                arguments = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if isinstance(arguments, dict) and slot["name"]:
                yield ToolCallRequested(ToolCall(slot["id"], slot["name"], arguments))
        yield ChatFinished(
            provider=self._label,
            model=self._model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def register(registry: ProviderRegistry) -> None:
    def text(options: Mapping[str, Any]) -> object:
        return OpenAICompatibleTextGenerator(
            base_url=str(options.get("base_url", "")),
            model=str(options.get("model", "")),
            api_key=str(options.get("api_key", "")),
        )

    registry.register(Capability.TEXT, NAME, text)
