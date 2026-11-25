"""Ordered multi-model failover for text generation (W3-M1).

Switches candidate only while nothing has been yielded yet; once a chunk is out,
errors propagate unchanged so output is never duplicated or spliced across
models. Logs carry labels (model names) only, never keys or prompts.
"""

import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence

from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.types import ChatEvent, ChatRequest
from xuna_backend.modules.cognition.infrastructure.openai_compatible import cooldown_hint_s

logger = logging.getLogger(__name__)

ALL_UNAVAILABLE = "all_text_providers_unavailable"
_DEFAULT_COOLDOWN_S = 60.0
_MAX_COOLDOWN_S = 1800.0
_AUTH_MARKERS = (
    "status 401",
    "status 403",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "forbidden",
)


def _is_auth_failure(message: str) -> bool:
    lower = message.lower()
    return any(marker in lower for marker in _AUTH_MARKERS)


class FallbackTextGenerator:
    def __init__(
        self,
        candidates: Sequence[tuple[str, TextGenerator]],
        clock: Callable[[], float] = time.monotonic,
        default_cooldown_s: float = _DEFAULT_COOLDOWN_S,
    ) -> None:
        if not candidates:
            raise ValueError("at least one text candidate is required")
        self._candidates = tuple(candidates)
        self._clock = clock
        self._default_cooldown_s = default_cooldown_s
        self._cooling_until: dict[int, float] = {}

    def _cooldown_for(self, message: str) -> float:
        hint = cooldown_hint_s(message)
        seconds = hint if hint is not None else self._default_cooldown_s
        return min(max(seconds, 1.0), _MAX_COOLDOWN_S)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        attempted = False
        for index, (label, generator) in enumerate(self._candidates):
            if self._clock() < self._cooling_until.get(index, 0.0):
                continue
            attempted = True
            emitted = False
            try:
                async for event in generator.stream(request):
                    emitted = True
                    yield event
                return
            except ProviderUnavailableError as error:
                if emitted:
                    raise
                message = str(error)
                if _is_auth_failure(message):
                    reason = "auth"
                else:
                    reason = "unavailable"
                    self._cooling_until[index] = self._clock() + self._cooldown_for(message)
                logger.warning(
                    "text_provider_failed_over",
                    extra={
                        "event": "primary_failed_over"
                        if index == 0
                        else "text_provider_failed_over",
                        "candidate_index": index,
                        "candidate": label,
                        "reason": reason,
                        "next_candidate": (
                            self._candidates[index + 1][0]
                            if index + 1 < len(self._candidates)
                            else None
                        ),
                    },
                )
        if not attempted:
            logger.warning("text_providers_all_cooling", extra={"event": "all_cooling"})
        raise ProviderUnavailableError(ALL_UNAVAILABLE)
