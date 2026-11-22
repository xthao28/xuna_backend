"""XTP v1 WebSocket session constants, close codes and rate limiter (UDC-3 §5.2)."""

import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

SUBPROTOCOL = "xtp/1"
ENROLLMENT_TTL_S = 300
APPROVAL_TTL_S = 60
MAX_STRUCTURED_CONTENT_BYTES = 16_384
MAX_MEDIA_STRUCTURED_CONTENT_BYTES = 458_752


def has_media_structured_content(sc: Mapping[str, object]) -> bool:
    """Matches `frames.dart:_hasMediaStructuredContent` for 448 KiB media payloads."""
    screenshot = sc.get("screenshot")
    if isinstance(screenshot, Mapping) and isinstance(screenshot.get("dataBase64"), str):
        return True
    if isinstance(sc.get("dataBase64"), str):
        return True
    image = sc.get("image")
    return isinstance(image, Mapping) and isinstance(image.get("dataBase64"), str)


def structured_content_byte_size(sc: Mapping[str, object]) -> int:
    """Compact UTF-8 JSON byte length matching `utf8.encode(jsonEncode(sc)).length`."""
    return len(json.dumps(sc, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def structured_content_exceeds_limit(
    sc: Mapping[str, object],
    *,
    allow_media: bool = False,
    max_ordinary_bytes: int = MAX_STRUCTURED_CONTENT_BYTES,
    max_media_bytes: int = MAX_MEDIA_STRUCTURED_CONTENT_BYTES,
) -> bool:
    """Returns True if `sc` exceeds the 16 KiB ordinary or 448 KiB media ceiling."""
    limit = (
        max_media_bytes if allow_media and has_media_structured_content(sc) else max_ordinary_bytes
    )
    try:
        return structured_content_byte_size(sc) > limit
    except (TypeError, ValueError):
        return True


@dataclass(frozen=True, slots=True)
class HostSessionClose:
    """WebSocket close code and reason string."""

    code: int
    reason: str


UNAUTHORIZED = HostSessionClose(1008, "unauthorized")
RATE_LIMITED = HostSessionClose(1008, "rate_limited")
UNSUPPORTED_PROTOCOL = HostSessionClose(1002, "unsupported_protocol")
FRAME_TOO_LARGE = HostSessionClose(1009, "frame_too_large")
IDLE_TIMEOUT = HostSessionClose(1001, "idle_timeout")
GOING_AWAY = HostSessionClose(1001, "going_away")
DUPLICATE_SESSION = HostSessionClose(4001, "duplicate_session")


class HostSessionRejected(Exception):
    def __init__(self, close: HostSessionClose) -> None:
        super().__init__(close.reason)
        self.close = close


@dataclass(frozen=True, slots=True)
class HostGatewayLimits:
    hello_timeout_s: float = 10.0
    challenge_ttl_s: float = 60.0
    enroll_wait_timeout_s: float = 300.0
    heartbeat_s: int = 30
    default_call_deadline_ms: int = 10_000
    max_frame_bytes: int = 524_288
    max_structured_content_bytes: int = MAX_STRUCTURED_CONTENT_BYTES
    max_media_structured_content_bytes: int = MAX_MEDIA_STRUCTURED_CONTENT_BYTES
    claims_per_owner_per_min: int = 30
    connects_per_peer_per_min: int = 30

    @property
    def idle_timeout_s(self) -> float:
        return 2.0 * self.heartbeat_s


class HostRateLimiter:
    """Sliding-window rate limiter (`limit` events per `window_s` per key)."""

    def __init__(self, limit: int, window_s: float, max_keys: int = 10_000) -> None:
        if limit < 1 or window_s <= 0:
            raise ValueError("limit and window must be positive")
        self._limit = limit
        self._window = window_s
        self._max_keys = max_keys
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str, now: float) -> bool:
        events = self._events.get(key)
        if events is None:
            if len(self._events) >= self._max_keys:
                self._prune(now)
            events = self._events.setdefault(key, deque())
        while events and now - events[0] >= self._window:
            events.popleft()
        if len(events) >= self._limit:
            return False
        events.append(now)
        return True

    def _prune(self, now: float) -> None:
        stale = [k for k, v in self._events.items() if not v or now - v[-1] >= self._window]
        for k in stale:
            del self._events[k]


@dataclass(frozen=True, slots=True)
class SelfEnrollmentChallenge:
    """Short-lived challenge issued for same-device self-enrollment (UDC-5 §5.2)."""

    nonce: bytes
    code: bytes
    expires_at: float


class SelfEnrollmentChallengeStore:
    """In-memory 60 s challenge store for `POST /v1/hosts/self/challenge`."""

    def __init__(self, ttl_s: float = 60.0) -> None:
        self._ttl_s = ttl_s
        self._challenges: dict[str, SelfEnrollmentChallenge] = {}

    def issue(
        self,
        owner_key: str,
        nonce: bytes,
        code: bytes,
        monotonic_now: float,
    ) -> SelfEnrollmentChallenge:
        challenge = SelfEnrollmentChallenge(
            nonce=nonce,
            code=code,
            expires_at=monotonic_now + self._ttl_s,
        )
        self._challenges[owner_key] = challenge
        return challenge

    def consume(self, owner_key: str, monotonic_now: float) -> SelfEnrollmentChallenge | None:
        challenge = self._challenges.pop(owner_key, None)
        if challenge is None or monotonic_now >= challenge.expires_at:
            return None
        return challenge
