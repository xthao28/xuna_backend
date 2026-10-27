"""Device-direct session rules (DC-3; contract draft "Device connection", ADR-0010).

The board opens one WebSocket over TLS, proves itself with its bound device key
(proof purpose 3) and then only sends heartbeats (and, from DC-5, projections).
Nothing in a session changes device state. Stdlib only.
"""

from collections import deque
from dataclasses import dataclass

SUBPROTOCOL = "xuna.device.v1"


@dataclass(frozen=True, slots=True)
class SessionClose:
    """WebSocket close code plus the stable reason code sent as the close reason."""

    code: int
    reason: str


UNAUTHORIZED = SessionClose(1008, "unauthorized")
NOT_BOUND = SessionClose(1008, "not_bound")
CONSENT_REQUIRED = SessionClose(1008, "consent_required")
RATE_LIMITED = SessionClose(1008, "rate_limited")
UNSUPPORTED_PROTOCOL = SessionClose(1002, "unsupported_protocol")
FRAME_TOO_LARGE = SessionClose(1009, "frame_too_large")
IDLE_TIMEOUT = SessionClose(1001, "idle_timeout")
GOING_AWAY = SessionClose(1001, "going_away")
REPLACED = SessionClose(4000, "replaced")


class SessionRejected(Exception):
    def __init__(self, close: SessionClose) -> None:
        super().__init__(close.reason)
        self.close = close


@dataclass(frozen=True, slots=True)
class GatewayLimits:
    hello_timeout_s: float = 10.0
    challenge_ttl_s: float = 60.0
    heartbeat_s: int = 30
    max_frame_bytes: int = 4096
    max_projection_bytes: int = 16384
    challenges_per_pet: int = 10
    challenge_window_s: float = 600.0
    connects_per_peer: int = 30
    connect_window_s: float = 60.0

    def __post_init__(self) -> None:
        if self.heartbeat_s < 10:
            raise ValueError("heartbeat must be at least 10 s (contract)")
        if self.hello_timeout_s <= 0 or self.challenge_ttl_s <= 0:
            raise ValueError("timeouts must be positive")

    @property
    def idle_timeout_s(self) -> float:
        return 3.0 * self.heartbeat_s


class SlidingWindowLimiter:
    """At most `limit` events per `window_s` per key. Process-local (one API
    process today); the caller passes a monotonic `now`."""

    def __init__(self, limit: int, window_s: float, max_keys: int = 10_000) -> None:
        if limit < 1 or window_s <= 0:
            raise ValueError("limit and window must be positive")
        self._limit, self._window, self._max_keys = limit, window_s, max_keys
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
        for key in [k for k, v in self._events.items() if not v or now - v[-1] >= self._window]:
            del self._events[key]
