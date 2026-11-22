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
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from xuna_backend.modules.hosts.application.xtp_frames import (
    rpc_notification_frame,
    rpc_request_frame,
)
from xuna_backend.modules.hosts.domain.entities import HostAlias, HostId
from xuna_backend.modules.hosts.domain.twin import (
    HostFocusChangedNotice,
    HostFocusState,
    HostPresence,
    HostResources,
    HostTwin,
)
from xuna_backend.modules.hosts.infrastructure.host_twin import InMemoryHostTwinStore


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
        self._twin_store = InMemoryHostTwinStore()
        self._focus_listeners: dict[
            uuid.UUID, list[Callable[[HostFocusChangedNotice], Awaitable[None]]]
        ] = {}

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
        owner_id: uuid.UUID | None = None,
    ) -> bool:
        """Registers a live session. If a session is already active for `host_id`,
        closes the existing session with `duplicate_session` and returns `False`
        so the caller also closes the new session with `duplicate_session`."""
        existing = self._live.pop(host_id, None)
        if existing is not None and existing.token is not token:
            self._twin_store.evict(host_id)
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
        self._twin_store.register_online(host_id, owner_id=owner_id)
        return True

    def close_session(self, host_id: str, token: object) -> bool:
        current = self._live.get(host_id)
        if current is None or current.token is not token:
            return False
        del self._live[host_id]
        self._twin_store.evict(host_id)
        for fut in current.in_flight.values():
            if not fut.done():
                fut.set_exception(RuntimeError("session_closed"))
        return True

    def get_twin(self, host_id: str) -> HostTwin | None:
        return self._twin_store.get(host_id)

    def get_twins_for_owner(self, owner_id: uuid.UUID) -> list[HostTwin]:
        return self._twin_store.list_for_owner(owner_id)

    def update_presence(self, host_id: str, presence: HostPresence) -> HostTwin | None:
        return self._twin_store.update_presence(host_id, presence)

    def update_resources(self, host_id: str, resources: HostResources) -> HostTwin | None:
        return self._twin_store.update_resources(host_id, resources)

    def update_trust_preset(self, host_id: str, trust_preset: str) -> HostTwin | None:
        return self._twin_store.update_trust_preset(host_id, trust_preset)

    def get_focus_state(self, owner_id: uuid.UUID) -> HostFocusState | None:
        return self._twin_store.get_focus(owner_id)

    def set_focus_state(
        self,
        owner_id: uuid.UUID,
        *,
        focused_host_id: HostId | None = None,
        keep_focused_host: bool = False,
        follow_me: bool | None = None,
        source: str,
        now: datetime,
    ) -> HostFocusState:
        return self._twin_store.set_focus(
            owner_id,
            focused_host_id=focused_host_id,
            keep_focused_host=keep_focused_host,
            follow_me=follow_me,
            source=source,
            now=now,
        )

    def subscribe_focus(
        self,
        owner_id: uuid.UUID,
        listener: Callable[[HostFocusChangedNotice], Awaitable[None]],
    ) -> Callable[[], None]:
        listeners = self._focus_listeners.setdefault(owner_id, [])
        listeners.append(listener)

        def _unsubscribe() -> None:
            bucket = self._focus_listeners.get(owner_id)
            if bucket is None:
                return
            with contextlib.suppress(ValueError):
                bucket.remove(listener)
            if not bucket:
                self._focus_listeners.pop(owner_id, None)

        return _unsubscribe

    async def publish_focus_changed(self, notice: HostFocusChangedNotice) -> None:
        listeners = list(self._focus_listeners.get(notice.owner_id, ()))
        for listener in listeners:
            with contextlib.suppress(Exception):
                await listener(notice)

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

    async def stop_host(self, host_id: str, reason: str = "owner_app") -> bool:
        """Cancels in-flight calls and sends `xuna/stop` without closing the WebSocket."""
        current = self._live.get(host_id)
        if current is None:
            return False
        for fut in list(current.in_flight.values()):
            if not fut.done():
                fut.set_exception(RuntimeError(f"stopped:{reason}"))
        current.in_flight.clear()
        with contextlib.suppress(Exception):
            await current.send_text(rpc_notification_frame("xuna/stop", {"reason": reason}))
        return True

    async def disconnect_host(self, host_id: str, reason: str = "revoked") -> None:
        self._twin_store.evict(host_id)
        if reason == "revoked":
            cleared = self._twin_store.clear_focus_if_host(
                host_id, now=datetime.now(UTC), source="revoked"
            )
            for state in cleared:
                await self.publish_focus_changed(
                    HostFocusChangedNotice(
                        owner_id=state.owner_id,
                        host_id=None,
                        alias=None,
                        platform=None,
                        follow_me=state.follow_me,
                        source="revoked",
                        updated_at=state.updated_at,
                    )
                )
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
