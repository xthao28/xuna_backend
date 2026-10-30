"""XTP v1 WebSocket session constants, close codes and rate limiter (UDC-3 §5.2)."""

from collections import deque
from dataclasses import dataclass

SUBPROTOCOL = "xtp/1"
ENROLLMENT_TTL_S = 300
APPROVAL_TTL_S = 60


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
    max_frame_bytes: int = 65_536
    max_structured_content_bytes: int = 16_384
    claims_per_owner_per_min: int = 5
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
