"""Ports for the Host Registry & XTP v1 module (UDC-3 §5.4)."""

import uuid
from collections.abc import Awaitable, Callable
from typing import Protocol

from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    Manifest,
    PendingEnrollment,
)


class EnrollmentRepository(Protocol):
    async def save_pending(self, enrollment: PendingEnrollment) -> None: ...

    async def get_pending_by_code_hash(self, code_sha256: str) -> PendingEnrollment | None: ...

    async def get_pending_by_id(self, pending_id: HostId) -> PendingEnrollment | None: ...

    async def delete_pending(self, pending_id: HostId) -> None: ...


class HostRepository(Protocol):
    async def get_host(self, host_id: HostId) -> Host | None: ...

    async def save_host(self, host: Host) -> None: ...

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]: ...

    async def get_grants(self, host_id: HostId) -> list[Grant]: ...

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None: ...

    async def get_manifest(self, host_id: HostId) -> Manifest | None: ...

    async def save_manifest(self, manifest: Manifest) -> None: ...


class HostSessionHub(Protocol):
    """Process-local hub managing enrollment waiting sockets and live host sessions."""

    def register_pending(
        self,
        pending_id: str,
        on_enrolled: Callable[[HostId, HostAlias], Awaitable[None]],
    ) -> None: ...

    def unregister_pending(self, pending_id: str) -> None: ...

    async def notify_enrolled(self, pending_id: str, host_id: HostId, alias: HostAlias) -> None: ...

    async def open_session(
        self,
        host_id: str,
        token: object,
        send_text: Callable[[str], Awaitable[None]],
        close_duplicate: Callable[[], Awaitable[None]],
        close_revoked: Callable[[], Awaitable[None]],
    ) -> bool:
        """Registers a live session. Returns `False` if a session was already live,
        after closing the previous session with `duplicate_session`."""
        ...

    def close_session(self, host_id: str, token: object) -> bool: ...

    def on_response(
        self, host_id: str, request_id: int | str, result: dict[str, object]
    ) -> None: ...

    def on_error_response(
        self, host_id: str, request_id: int | str, code: int, message: str
    ) -> None: ...

    def is_online(self, host_id: str) -> bool: ...

    async def send_request(
        self,
        host_id: str,
        method: str,
        params: dict[str, object],
        timeout_s: float,
    ) -> dict[str, object]: ...

    async def notify(
        self,
        host_id: str,
        method: str,
        params: dict[str, object],
    ) -> None: ...

    async def disconnect_host(self, host_id: str, reason: str = "revoked") -> None: ...
