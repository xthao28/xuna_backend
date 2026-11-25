"""Adapter for any OpenAI-style `POST {base}/chat/completions` streaming server.

Covers the owner's internal AI (reached over OpenVPN), vLLM, Ollama `/v1`,
LM Studio, LiteLLM and most hosted vendors, so switching to a rented provider
later is configuration only. Network reachability (VPN) is the host's job;
this adapter only needs a base URL. Prompts and outputs are never logged.
"""

import asyncio
import json
import logging
import re
import secrets
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any

import httpx

from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.types import (
    Capability,
    ChatEvent,
    ChatFinished,
    ChatRequest,
    EmbeddingRequest,
    EmbeddingResult,
    Message,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)

NAME = "openai_compatible"

logger = logging.getLogger(__name__)

_GLOBAL_SIGNATURE_CACHE: dict[str, dict[str, Any]] = {}
_MAX_GLOBAL_SIGNATURE_CACHE = 512

_GEMINI_KEY_RE = re.compile(r"AIza[0-9A-Za-z\-_]{35}")
_OPENAI_KEY_RE = re.compile(r"sk-(?:proj-)?[A-Za-z0-9\-_]{20,}")
_GROQ_KEY_RE = re.compile(r"gsk_[0-9A-Za-z_\-]{20,}")
_BEARER_TOKEN_RE = re.compile(r"Bearer\s+[A-Za-z0-9\-_.]{8,}", re.IGNORECASE)

_QUOTA_EXHAUSTED_KEYWORDS = (
    "resource_exhausted",
    "daily quota",
    "quota exceeded",
    "ratelimitexceeded",
    "quota_exhausted",
    "exceeded your current quota",
    "insufficient_quota",
)
_RETRY_HINT_RE = re.compile(
    r"retry\s+(?:in|after)\s+(?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)
_DAILY_QUOTA_MARKERS = (
    "per day",
    "daily",
    "requests_per_day",
    "insufficient_quota",
    "billing",
)
_MINUTE_QUOTA_MARKERS = (
    "per minute",
    "requests per minute",
    "tokens per minute",
    "rpm",
    "tpm",
)
_DAILY_COOLDOWN_MIN_S = 600.0
_DAILY_COOLDOWN_MAX_S = 1800.0
_MINUTE_COOLDOWN_MIN_S = 15.0
_MINUTE_COOLDOWN_MAX_S = 65.0
_DEFAULT_QUOTA_COOLDOWN_S = 300.0


def _cache_signature(call_id: str, extra: dict[str, Any]) -> None:
    if len(_GLOBAL_SIGNATURE_CACHE) >= _MAX_GLOBAL_SIGNATURE_CACHE:
        try:
            oldest = next(iter(_GLOBAL_SIGNATURE_CACHE))
            del _GLOBAL_SIGNATURE_CACHE[oldest]
        except StopIteration:
            _GLOBAL_SIGNATURE_CACHE.clear()
    _GLOBAL_SIGNATURE_CACHE[call_id] = extra


def _normalize_thought_signature(val: Any) -> dict[str, Any] | None:
    if isinstance(val, dict):
        return val if val else None
    if isinstance(val, str) and val.strip():
        s = val.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed if parsed else None
            except json.JSONDecodeError:
                pass
        return {"thought_signature": val}
    return None


def _sanitize_error_text(text: str, secrets: Iterable[str] = ()) -> str:
    sanitized = text
    for secret in secrets:
        if secret and len(secret) >= 4:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    sanitized = _GEMINI_KEY_RE.sub("[REDACTED_API_KEY]", sanitized)
    sanitized = _OPENAI_KEY_RE.sub("[REDACTED_API_KEY]", sanitized)
    sanitized = _GROQ_KEY_RE.sub("[REDACTED_API_KEY]", sanitized)
    sanitized = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", sanitized)
    return sanitized


def _is_quota_exhausted(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _QUOTA_EXHAUSTED_KEYWORDS)


def _parse_quota_cooldown_s(text: str, retry_after_header: str = "") -> float:
    parsed_s: float | None = None
    if retry_after_header.strip():
        try:
            header_val = float(retry_after_header.strip())
            if header_val > 0:
                parsed_s = header_val
        except ValueError:
            pass
    if parsed_s is None:
        match = _RETRY_HINT_RE.search(text)
        if match is not None:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = float(match.group(3))
            parsed_s = hours * 3600.0 + minutes * 60.0 + seconds

    lower = text.lower()
    has_explicit_daily = any(
        marker in lower for marker in ("per day", "per_day", "perday", "daily", "requests_per_day")
    )
    has_explicit_minute = any(marker in lower for marker in _MINUTE_QUOTA_MARKERS)
    if has_explicit_daily or (parsed_s is not None and parsed_s >= 120.0):
        base = parsed_s if parsed_s is not None else _DAILY_COOLDOWN_MIN_S
        return min(max(base, _DAILY_COOLDOWN_MIN_S), _DAILY_COOLDOWN_MAX_S)
    if has_explicit_minute:
        base = parsed_s if parsed_s is not None else _MINUTE_COOLDOWN_MIN_S
        return min(max(base, _MINUTE_COOLDOWN_MIN_S), _MINUTE_COOLDOWN_MAX_S)
    if any(marker in lower for marker in _DAILY_QUOTA_MARKERS):
        base = parsed_s if parsed_s is not None else _DAILY_COOLDOWN_MIN_S
        return min(max(base, _DAILY_COOLDOWN_MIN_S), _DAILY_COOLDOWN_MAX_S)
    if parsed_s is not None and parsed_s > 0:
        return min(max(parsed_s, _MINUTE_COOLDOWN_MIN_S), _DAILY_COOLDOWN_MAX_S)
    return _DEFAULT_QUOTA_COOLDOWN_S


def _serialize_message(
    m: Message,
    extra_by_call_id: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if m.role is Role.ASSISTANT and m.tool_calls:
        serialized_calls: list[dict[str, Any]] = []
        for tc in m.tool_calls:
            item: dict[str, Any] = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            extra = (
                tc.extra
                or (extra_by_call_id.get(tc.id) if extra_by_call_id else None)
                or _GLOBAL_SIGNATURE_CACHE.get(tc.id)
            )
            if extra:
                item["extra_content"] = extra
            serialized_calls.append(item)
        return {
            "role": Role.ASSISTANT.value,
            "content": m.content or None,
            "tool_calls": serialized_calls,
        }
    content_val: Any
    if m.image_base64:
        blocks: list[dict[str, Any]] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{m.image_mime_type};base64,{m.image_base64}",
                },
            }
        )
        content_val = blocks
    else:
        content_val = m.content

    msg: dict[str, Any] = {"role": m.role.value, "content": content_val}
    if m.tool_call_id:
        msg["tool_call_id"] = m.tool_call_id
    return msg


class OpenAICompatibleTextGenerator:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
        api_key_header: str = "authorization",
        provider_label: str = NAME,
        reasoning_effort: str = "",
        max_attempts: int = 5,
        retry_base_delay_s: float = 1.0,
        retry_max_delay_s: float = 8.0,
        quota_cooldown_enabled: bool = True,
    ) -> None:
        if not base_url.startswith(("http://", "https://")) or not model:
            raise ValueError("base_url (http/https) and model are required")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        header = api_key_header.strip().lower()
        if not header.replace("-", "").isalnum():
            raise ValueError("invalid api_key_header")
        if not api_key:
            self._headers: dict[str, str] = {}
        elif header == "authorization":
            self._headers = {"Authorization": f"Bearer {api_key}"}
        else:
            self._headers = {header: api_key}  # e.g. Bifrost virtual key x-bf-vk
        if reasoning_effort and not reasoning_effort.isalpha():
            raise ValueError("invalid reasoning_effort")
        self._reasoning_effort = reasoning_effort
        self._transport = transport
        self._label = provider_label
        self._api_key = api_key
        self._max_attempts = max(1, max_attempts)
        self._retry_base_delay_s = max(0.0, retry_base_delay_s)
        self._retry_max_delay_s = max(0.0, retry_max_delay_s)
        self._quota_cooldown_enabled = quota_cooldown_enabled
        self._quota_cooldown_until: float = 0.0
        self._tool_call_extra: dict[str, dict[str, Any]] = {}

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        serialized_messages: list[dict[str, Any]] = []
        pending_tool_images: list[tuple[str, str]] = []

        def _flush_pending_images() -> None:
            if not pending_tool_images:
                return
            blocks: list[dict[str, Any]] = [
                {"type": "text", "text": "Attached screenshot from tool observation."}
            ]
            for mime, b64 in pending_tool_images:
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )
            serialized_messages.append({"role": Role.USER.value, "content": blocks})
            pending_tool_images.clear()

        for m in request.messages:
            if m.role is Role.TOOL and m.image_base64:
                # OpenAI & Gemini OpenAI-compat reject `image_url` parts inside `role="tool"`.
                # Send text-only tool result first, then attach screenshot in a `role="user"` turn.
                msg: dict[str, Any] = {"role": Role.TOOL.value, "content": m.content}
                if m.tool_call_id:
                    msg["tool_call_id"] = m.tool_call_id
                serialized_messages.append(msg)
                pending_tool_images.append((m.image_mime_type, m.image_base64))
                continue
            if m.role is not Role.TOOL:
                _flush_pending_images()
            serialized_messages.append(_serialize_message(m, self._tool_call_extra))
        _flush_pending_images()

        payload: dict[str, Any] = {
            "model": self._model,
            "stream": True,
            "max_tokens": request.max_output_tokens,
            "messages": serialized_messages,
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
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
        if self._quota_cooldown_enabled:
            now_mono = time.monotonic()
            if now_mono < self._quota_cooldown_until:
                remaining_cooldown = max(1, int(self._quota_cooldown_until - now_mono))
                raise ProviderUnavailableError(
                    f"quota_exhausted: cooldown active ({remaining_cooldown}s remaining)"
                )
        started = time.perf_counter()
        deadline = started + request.timeout_s
        timeout = httpx.Timeout(request.timeout_s, connect=min(10.0, request.timeout_s))
        usage: dict[str, int] = {}
        calls: dict[int, dict[str, Any]] = {}
        pre_call_extra: dict[str, Any] = {}
        emitted_any = False
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
                for attempt in range(self._max_attempts):
                    calls.clear()
                    pre_call_extra.clear()
                    remaining = max(deadline - time.perf_counter(), 1.0)
                    if attempt == 0 and self._max_attempts > 1 and remaining >= 45.0:
                        read_timeout = min(remaining * 0.6, 35.0)
                    else:
                        read_timeout = min(request.timeout_s, remaining)
                    attempt_timeout = httpx.Timeout(
                        read_timeout,
                        connect=min(10.0, remaining),
                    )
                    try:
                        async with client.stream(
                            "POST",
                            self._url,
                            json=self._payload(request),
                            headers=self._headers,
                            timeout=attempt_timeout,
                        ) as response:
                            if response.status_code != 200:
                                error_bytes = b""
                                try:
                                    async for chunk in response.aiter_bytes():
                                        error_bytes += chunk
                                        if len(error_bytes) >= 8192:
                                            error_bytes = error_bytes[:8192]
                                            break
                                except httpx.HTTPError:
                                    pass
                                raw_snippet = error_bytes.decode("utf-8", errors="replace")
                                secrets_to_redact = [self._api_key]
                                for val in self._headers.values():
                                    secrets_to_redact.append(val)
                                    if val.lower().startswith("bearer "):
                                        secrets_to_redact.append(val[7:].strip())
                                sanitized_snippet = _sanitize_error_text(
                                    raw_snippet, secrets_to_redact
                                )
                                bounded_snippet = sanitized_snippet[:1024]
                                logger.warning(
                                    "Provider error (status %d): %s",
                                    response.status_code,
                                    bounded_snippet,
                                )
                                if _is_quota_exhausted(sanitized_snippet):
                                    if self._quota_cooldown_enabled:
                                        cooldown_s = _parse_quota_cooldown_s(
                                            sanitized_snippet,
                                            response.headers.get("retry-after", ""),
                                        )
                                        self._quota_cooldown_until = time.monotonic() + cooldown_s
                                        logger.info(
                                            "Quota cooldown armed for %.0fs on model=%s",
                                            cooldown_s,
                                            self._model,
                                        )
                                    raise ProviderUnavailableError(
                                        f"quota_exhausted: {bounded_snippet}"
                                    )
                                if (
                                    response.status_code in (429, 503)
                                    and attempt + 1 < self._max_attempts
                                ):
                                    retry_after_raw = response.headers.get("retry-after", "")
                                    try:
                                        delay_s = min(max(float(retry_after_raw), 0.8), 10.0)
                                    except ValueError:
                                        exp_delay = min(
                                            self._retry_max_delay_s,
                                            self._retry_base_delay_s * (2**attempt),
                                        )
                                        jitter = (
                                            secrets.randbelow(351) / 1000.0
                                            if self._retry_base_delay_s > 0
                                            else 0.0
                                        )
                                        delay_s = exp_delay + jitter
                                    remaining_after = deadline - time.perf_counter()
                                    if remaining_after > 1.5:
                                        sleep_s = min(delay_s, remaining_after - 1.5)
                                        if sleep_s > 0:
                                            await asyncio.sleep(sleep_s)
                                        continue
                                raise ProviderUnavailableError(
                                    f"status {response.status_code}: {bounded_snippet}"
                                    if bounded_snippet
                                    else f"status {response.status_code}"
                                )
                            async for line in response.aiter_lines():
                                if not line.startswith("data:"):
                                    continue
                                data = line[5:].strip()
                                if data == "[DONE]":
                                    break
                                parsed_chunk = json.loads(data)
                                if not isinstance(parsed_chunk, dict):
                                    continue
                                usage = parsed_chunk.get("usage") or usage
                                for choice in parsed_chunk.get("choices") or []:
                                    delta = choice.get("delta") or {}
                                    if delta.get("content"):
                                        emitted_any = True
                                        yield TextDelta(str(delta["content"]))
                                    tool_calls = delta.get("tool_calls") or []
                                    if tool_calls:
                                        emitted_any = True
                                    delta_thought = delta.get("thought_signature") or delta.get(
                                        "extra_content"
                                    )
                                    if delta_thought is not None and not tool_calls:
                                        norm_delta_extra = _normalize_thought_signature(
                                            delta_thought
                                        )
                                        if norm_delta_extra:
                                            if not calls:
                                                for k, v in norm_delta_extra.items():
                                                    if (
                                                        k in pre_call_extra
                                                        and isinstance(pre_call_extra[k], str)
                                                        and isinstance(v, str)
                                                    ):
                                                        pre_call_extra[k] += v
                                                    else:
                                                        pre_call_extra[k] = v
                                            else:
                                                slot = calls[max(calls.keys())]
                                                if slot["extra"] is None:
                                                    slot["extra"] = dict(norm_delta_extra)
                                                elif isinstance(slot["extra"], dict):
                                                    for k, v in norm_delta_extra.items():
                                                        if (
                                                            k in slot["extra"]
                                                            and isinstance(slot["extra"][k], str)
                                                            and isinstance(v, str)
                                                        ):
                                                            slot["extra"][k] += v
                                                        else:
                                                            slot["extra"][k] = v
                                    for call in tool_calls:
                                        slot = calls.setdefault(
                                            int(call.get("index", 0)),
                                            {"id": "", "name": "", "args": "", "extra": None},
                                        )
                                        if pre_call_extra:
                                            if slot["extra"] is None:
                                                slot["extra"] = dict(pre_call_extra)
                                            elif isinstance(slot["extra"], dict):
                                                for k, v in pre_call_extra.items():
                                                    if (
                                                        k in slot["extra"]
                                                        and isinstance(slot["extra"][k], str)
                                                        and isinstance(v, str)
                                                    ):
                                                        slot["extra"][k] = v + slot["extra"][k]
                                                    else:
                                                        slot["extra"].setdefault(k, v)
                                            pre_call_extra.clear()
                                        function = call.get("function") or {}
                                        slot["id"] = call.get("id") or slot["id"]
                                        slot["name"] = function.get("name") or slot["name"]
                                        slot["args"] += function.get("arguments") or ""
                                        raw_extra = (
                                            call.get("extra_content")
                                            or call.get("thought_signature")
                                            or delta.get("thought_signature")
                                        )
                                        if raw_extra is not None:
                                            norm_extra = _normalize_thought_signature(raw_extra)
                                            if norm_extra:
                                                if slot["extra"] is None:
                                                    slot["extra"] = dict(norm_extra)
                                                elif isinstance(slot["extra"], dict):
                                                    for k, v in norm_extra.items():
                                                        if (
                                                            k in slot["extra"]
                                                            and isinstance(slot["extra"][k], str)
                                                            and isinstance(v, str)
                                                        ):
                                                            slot["extra"][k] += v
                                                        else:
                                                            slot["extra"][k] = v
                            break
                    except (
                        httpx.TimeoutException,
                        httpx.ConnectError,
                        httpx.RemoteProtocolError,
                    ) as error:
                        logger.warning("Provider transport error (%s)", type(error).__name__)
                        remaining_after = deadline - time.perf_counter()
                        if (
                            not emitted_any
                            and attempt + 1 < self._max_attempts
                            and remaining_after > 1.5
                        ):
                            exp_delay = min(
                                self._retry_max_delay_s,
                                self._retry_base_delay_s * (2**attempt),
                            )
                            jitter = (
                                secrets.randbelow(351) / 1000.0
                                if self._retry_base_delay_s > 0
                                else 0.0
                            )
                            sleep_s = min(exp_delay + jitter, remaining_after - 1.5)
                            if sleep_s > 0:
                                await asyncio.sleep(sleep_s)
                            continue
                        raise ProviderUnavailableError(type(error).__name__) from error
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            logger.warning("Provider transport error (%s)", type(error).__name__)
            raise ProviderUnavailableError(type(error).__name__) from error
        for slot in calls.values():
            try:
                arguments = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if isinstance(arguments, dict) and slot["name"]:
                call_id = str(slot["id"])
                extra_dict = slot.get("extra") if isinstance(slot.get("extra"), dict) else None
                if extra_dict and call_id:
                    if len(self._tool_call_extra) >= 128:
                        self._tool_call_extra.clear()
                    self._tool_call_extra[call_id] = extra_dict
                    _cache_signature(call_id, extra_dict)
                yield ToolCallRequested(
                    ToolCall(call_id, str(slot["name"]), arguments, extra=extra_dict)
                )
        yield ChatFinished(
            provider=self._label,
            model=self._model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
        api_key_header: str = "authorization",
        provider_label: str = NAME,
        timeout_s: float = 10.0,
    ) -> None:
        if not base_url.startswith(("http://", "https://")) or not model:
            raise ValueError("base_url (http/https) and model are required")
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        header = api_key_header.strip().lower()
        if not header.replace("-", "").isalnum():
            raise ValueError("invalid api_key_header")
        if not api_key:
            self._headers: dict[str, str] = {}
        elif header == "authorization":
            self._headers = {"Authorization": f"Bearer {api_key}"}
        else:
            self._headers = {header: api_key}
        self._transport = transport
        self._label = provider_label
        self._timeout_s = timeout_s

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        payload = {
            "model": self._model,
            "input": list(request.texts),
        }
        timeout = min(request.timeout_s, self._timeout_s)
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=timeout, headers=self._headers
            ) as client:
                response = await client.post(self._url, json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            logger.warning("Provider transport error (%s)", type(error).__name__)
            raise ProviderUnavailableError(type(error).__name__) from error

        items = data.get("data")
        if not isinstance(items, list):
            raise ProviderUnavailableError("invalid embedding response format")

        embeddings: list[tuple[float, ...]] = []
        for item in items:
            vec = item.get("embedding")
            if isinstance(vec, list):
                embeddings.append(tuple(float(x) for x in vec))
            else:
                embeddings.append(())

        return EmbeddingResult(
            embeddings=tuple(embeddings),
            model=str(data.get("model") or self._model),
        )


def register(registry: ProviderRegistry) -> None:
    def text(options: Mapping[str, Any]) -> object:
        return OpenAICompatibleTextGenerator(
            base_url=str(options.get("base_url", "")),
            model=str(options.get("model", "")),
            api_key=str(options.get("api_key", "")),
            api_key_header=str(options.get("api_key_header", "authorization")),
            reasoning_effort=str(options.get("reasoning_effort", "")),
        )

    def embedding(options: Mapping[str, Any]) -> object:
        return OpenAICompatibleEmbedder(
            base_url=str(options.get("base_url", "")),
            model=str(options.get("model", "")),
            api_key=str(options.get("api_key", "")),
            api_key_header=str(options.get("api_key_header", "authorization")),
            timeout_s=float(options.get("timeout_s", 10.0)),
        )

    registry.register(Capability.TEXT, NAME, text)
    registry.register(Capability.EMBEDDING, NAME, embedding)
