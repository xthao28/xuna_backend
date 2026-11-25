import asyncio
import logging
from collections.abc import AsyncIterator

import pytest

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import text_generator
from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Message,
    Role,
    TextDelta,
)
from xuna_backend.modules.cognition.infrastructure.fallback_text_generator import (
    ALL_UNAVAILABLE,
    FallbackTextGenerator,
)
from xuna_backend.modules.cognition.infrastructure.openai_compatible import (
    OpenAICompatibleTextGenerator,
)

REQUEST = ChatRequest(messages=(Message(Role.USER, "secret user text"),))
SECRET = "sk-super-secret-value-123456"  # noqa: S105


class Fake:
    def __init__(self, label: str, error: str | None = None, fail_after: int | None = None):
        self.label = label
        self.error = error
        self.fail_after = fail_after
        self.calls = 0

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.calls += 1
        if self.error is not None and self.fail_after is None:
            raise ProviderUnavailableError(self.error)
        yield TextDelta(f"hello from {self.label}")
        if self.error is not None:
            raise ProviderUnavailableError(self.error)
        yield ChatFinished("fake", self.label, 0, 0, 0)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def collect(gen: FallbackTextGenerator) -> list[ChatEvent]:
    async def go() -> list[ChatEvent]:
        return [e async for e in gen.stream(REQUEST)]

    return asyncio.run(go())


def texts(events: list[ChatEvent]) -> list[str]:
    return [e.text for e in events if isinstance(e, TextDelta)]


def test_daily_quota_fails_over_and_yields_backup_once() -> None:
    primary = Fake("a", "quota_exhausted: daily quota exceeded (per day)")
    backup = Fake("b")
    events = collect(FallbackTextGenerator([("a", primary), ("b", backup)], Clock()))
    assert texts(events) == ["hello from b"]
    assert sum(isinstance(e, ChatFinished) for e in events) == 1
    assert primary.calls == 1 and backup.calls == 1


def test_503_fails_over(caplog: pytest.LogCaptureFixture) -> None:
    primary = Fake("a", "status 503: overloaded")
    backup = Fake("b")
    with caplog.at_level(logging.WARNING):
        events = collect(FallbackTextGenerator([("a", primary), ("b", backup)], Clock()))
    assert texts(events) == ["hello from b"]
    record = next(r for r in caplog.records if getattr(r, "event", "") == "primary_failed_over")
    assert record.candidate == "a"  # type: ignore[attr-defined]
    assert "secret user text" not in caplog.text


def test_error_after_first_chunk_propagates_without_backup() -> None:
    primary = Fake("a", "status 503: boom", fail_after=1)
    backup = Fake("b")
    gen = FallbackTextGenerator([("a", primary), ("b", backup)], Clock())
    seen: list[ChatEvent] = []

    async def go() -> None:
        async for event in gen.stream(REQUEST):
            seen.append(event)

    with pytest.raises(ProviderUnavailableError, match="boom"):
        asyncio.run(go())
    assert texts(seen) == ["hello from a"]
    assert backup.calls == 0


def test_all_fail_raises_content_free_error() -> None:
    gen = FallbackTextGenerator(
        [
            ("a", Fake("a", f"status 503: leaked {SECRET}")),
            ("b", Fake("b", "status 401: unauthorized")),
        ],
        Clock(),
    )
    with pytest.raises(ProviderUnavailableError) as info:
        collect(gen)
    assert str(info.value) == ALL_UNAVAILABLE
    assert SECRET not in str(info.value)


def test_auth_failure_tries_next_but_does_not_cool_down() -> None:
    primary = Fake("a", "status 401: invalid api key")
    backup = Fake("b")
    gen = FallbackTextGenerator([("a", primary), ("b", backup)], Clock())
    collect(gen)
    collect(gen)
    assert primary.calls == 2 and backup.calls == 2


def test_cooled_primary_skipped_then_retried_after_expiry() -> None:
    clock = Clock()
    primary = Fake("a", "status 503: down")
    backup = Fake("b")
    gen = FallbackTextGenerator([("a", primary), ("b", backup)], clock, default_cooldown_s=60.0)
    collect(gen)
    assert primary.calls == 1
    collect(gen)
    assert primary.calls == 1 and backup.calls == 2
    clock.now += 61
    collect(gen)
    assert primary.calls == 2 and backup.calls == 3


def test_all_cooling_fails_fast_without_calling() -> None:
    a, b = Fake("a", "status 503: x"), Fake("b", "status 503: y")
    gen = FallbackTextGenerator([("a", a), ("b", b)], Clock())
    with pytest.raises(ProviderUnavailableError):
        collect(gen)
    with pytest.raises(ProviderUnavailableError, match=ALL_UNAVAILABLE):
        collect(gen)
    assert a.calls == 1 and b.calls == 1


def test_quota_cooldown_seconds_taken_from_error_text() -> None:
    clock = Clock()
    primary = Fake("a", "quota_exhausted: cooldown active (900s remaining)")
    backup = Fake("b")
    gen = FallbackTextGenerator([("a", primary), ("b", backup)], clock, default_cooldown_s=60.0)
    collect(gen)
    clock.now += 120
    collect(gen)
    assert primary.calls == 1  # still cooling past the 60s default
    clock.now += 800
    collect(gen)
    assert primary.calls == 2


def test_empty_candidates_rejected() -> None:
    with pytest.raises(ValueError):
        FallbackTextGenerator([])


# --- config parsing -------------------------------------------------------

BASE_ENV = {"XUNA_DATABASE_URL": "sqlite://x"}


def test_fallback_models_parsed_trimmed() -> None:
    s = Settings.from_env({**BASE_ENV, "XUNA_TEXT_FALLBACK_MODELS": " m1 , ,m2,, "})
    assert s.text_fallback_models == ("m1", "m2")
    assert Settings.from_env(BASE_ENV).text_fallback_models == ()


def test_fallback_providers_parsed() -> None:
    raw = (
        '[{"provider":"openai_compatible","base_url":"https://x/v1",'
        f'"api_key":"{SECRET}","model":"m","api_key_header":"x-bf-vk"}}]'
    )
    s = Settings.from_env({**BASE_ENV, "XUNA_TEXT_FALLBACK_PROVIDERS": raw})
    assert s.text_fallback_providers[0]["model"] == "m"
    assert SECRET not in repr(s)


def _entry(**override: object) -> str:
    import json

    base: dict[str, object] = {
        "provider": "p",
        "base_url": "u",
        "api_key": SECRET,
        "model": "m",
    }
    base.update(override)
    base = {k: v for k, v in base.items() if v is not None}
    return json.dumps([base])


@pytest.mark.parametrize(
    "raw,fragment",
    [
        ("{not json", "not valid JSON"),
        ('{"a":1}', "JSON array"),
        (_entry(bogus=1), "bogus"),
        (_entry(model=None), "model is required"),
        (_entry(model=5), "model must be"),
        ('["x"]', "must be an object"),
    ],
)
def test_fallback_providers_rejected_without_leaking_key(raw: str, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment) as info:
        Settings.from_env({**BASE_ENV, "XUNA_TEXT_FALLBACK_PROVIDERS": raw})
    assert SECRET not in str(info.value)


# --- main wiring ----------------------------------------------------------


def _settings(**kw: object) -> Settings:
    return Settings(
        environment=Environment.TEST,
        database_url="sqlite://x",
        text_provider="openai_compatible",
        text_base_url="https://example.test/v1",
        text_model="primary",
        text_api_key="k",
        **kw,  # type: ignore[arg-type]
    )


def test_wiring_returns_bare_primary_when_unconfigured() -> None:
    assert isinstance(text_generator(_settings()), OpenAICompatibleTextGenerator)


def test_wiring_echo_ignores_fallback() -> None:
    s = Settings(
        environment=Environment.TEST,
        database_url="sqlite://x",
        text_fallback_models=("m",),
    )
    assert not isinstance(text_generator(s), FallbackTextGenerator)


def test_wiring_builds_fallback_chain() -> None:
    provider = {
        "provider": "openai_compatible",
        "base_url": "https://other.test/v1",
        "api_key": "k2",
        "model": "other",
    }
    gen = text_generator(
        _settings(text_fallback_models=("backup",), text_fallback_providers=(provider,))
    )
    assert isinstance(gen, FallbackTextGenerator)
    assert [label for label, _ in gen._candidates] == [
        "primary",
        "backup",
        "openai_compatible:other",
    ]
