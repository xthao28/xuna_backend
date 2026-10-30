"""Process-local session hub for Host Agents over XTP v1 (UDC-3 §5.4).

Manages:
1. Enrollment waiting callbacks (`pending_id -> callback`) so `POST /v1/hosts/{id}/confirm`
   immediately pushes `xuna/enrolled` to the waiting WebSocket.
2. Single live session per `HostId` (ADR-0013): a second concurrent session closes
   BOTH the existing and the new session with `4001 duplicate_session`.
3. JSON-RPC request/response correlation (`tools/list`, `tools/call`, `ping`),
   per-call timeout, and automatic `notifications/cancelled` on timeout.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from xuna_backend.modules.hosts.application.xtp_frames import (
    rpc_notification_frame,
    rpc_request_frame,
)
from xuna_backend.modules.hosts.domain.entities import HostAlias, HostId


@dataclass
class _LiveSession:
    token: object
    send_text: Callable[[str], Awaitable[None]]
    close_duplicate: Callable[[], Awaitable[None]]
    close_revoked: Callable[[], Awaitable[None]]
    next_rpc_id: int = 1
    in_flight: dict[int | str, asyncio.Future[dict[str, object]]] = field(default_factory=dict)


class InMemoryHostSessionHub:
    def __init__(self) -> None:
        self._pending: dict[str, Callable[[HostId, HostAlias], Awaitable[None]]] = {}
        self._live: dict[str, _LiveSession] = {}

    def register_pending(
        self,
        pending_id: str,
        on_enrolled: Callable[[HostId, HostAlias], Awaitable[None]],
    ) -> None:
        self._pending[pending_id] = on_enrolled

    def unregister_pending(self, pending_id: str) -> None:
        self._pending.pop(pending_id, None)

    async def notify_enrolled(self, pending_id: str, host_id: HostId, alias: HostAlias) -> None:
        callback = self._pending.pop(pending_id, None)
        if callback is not None:
            with contextlib.suppress(Exception):
                await callback(host_id, alias)

    async def open_session(
        self,
        host_id: str,
        token: object,
        send_text: Callable[[str], Awaitable[None]],
        close_duplicate: Callable[[], Awaitable[None]],
        close_revoked: Callable[[], Awaitable[None]],
    ) -> bool:
        """Registers a live session. If a session is already active for `host_id`,
        closes the existing session with `duplicate_session` and returns `False`
        so the caller also closes the new session with `duplicate_session`."""
        existing = self._live.pop(host_id, None)
        if existing is not None and existing.token is not token:
            for fut in existing.in_flight.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("duplicate_session"))
            with contextlib.suppress(Exception):
                await existing.close_duplicate()
            return False

        self._live[host_id] = _LiveSession(
            token=token,
            send_text=send_text,
            close_duplicate=close_duplicate,
            close_revoked=close_revoked,
        )
        return True

    def close_session(self, host_id: str, token: object) -> bool:
        current = self._live.get(host_id)
        if current is None or current.token is not token:
            return False
        del self._live[host_id]
        for fut in current.in_flight.values():
            if not fut.done():
                fut.set_exception(RuntimeError("session_closed"))
        return True

    def on_response(self, host_id: str, request_id: int | str, result: dict[str, object]) -> None:
        current = self._live.get(host_id)
        if current is None:
            return
        fut = current.in_flight.pop(request_id, None)
        if fut is not None and not fut.done():
            fut.set_result(result)

    def on_error_response(
        self, host_id: str, request_id: int | str, code: int, message: str
    ) -> None:
        current = self._live.get(host_id)
        if current is None:
            return
        fut = current.in_flight.pop(request_id, None)
        if fut is not None and not fut.done():
            fut.set_exception(RuntimeError(f"rpc_error:{code}:{message}"))

    def is_online(self, host_id: str) -> bool:
        return host_id in self._live

    async def send_request(
        self,
        host_id: str,
        method: str,
        params: dict[str, object],
        timeout_s: float,
    ) -> dict[str, object]:
        current = self._live.get(host_id)
        if current is None:
            raise RuntimeError("host_offline")
        rpc_id = current.next_rpc_id
        current.next_rpc_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, object]] = loop.create_future()
        current.in_flight[rpc_id] = fut
        try:
            await current.send_text(rpc_request_frame(rpc_id, method, params))
            return await asyncio.wait_for(fut, max(timeout_s, 0.05))
        except TimeoutError:
            current.in_flight.pop(rpc_id, None)
            with contextlib.suppress(Exception):
                await current.send_text(
                    rpc_notification_frame(
                        "notifications/cancelled",
                        {"requestId": rpc_id, "reason": "timeout"},
                    )
                )
            raise
        finally:
            current.in_flight.pop(rpc_id, None)

    async def notify(
        self,
        host_id: str,
        method: str,
        params: dict[str, object],
    ) -> None:
        current = self._live.get(host_id)
        if current is None:
            return
        await current.send_text(rpc_notification_frame(method, params))

    async def disconnect_host(self, host_id: str, reason: str = "revoked") -> None:
        current = self._live.pop(host_id, None)
        if current is None:
            return
        for fut in current.in_flight.values():
            if not fut.done():
                fut.set_exception(RuntimeError(reason))
        with contextlib.suppress(Exception):
            await current.send_text(rpc_notification_frame("xuna/stop", {"reason": reason}))
        with contextlib.suppress(Exception):
            await current.close_revoked()
