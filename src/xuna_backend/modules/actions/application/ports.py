"""Ports for the `actions` application layer (UDC-4 §5.2)."""

import uuid
from datetime import datetime
from typing import Protocol

from xuna_backend.modules.actions.domain.entities import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalState,
    LocalActionReport,
)


class ApprovalRepository(Protocol):
    """Durable persistence for `ApprovalRequest` records (`action_approvals`)."""

    async def save(self, req: ApprovalRequest) -> None: ...

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None: ...


class ApprovalBroker(Protocol):
    """Process-local coordination for pending owner approvals (60 s TTL)."""

    async def create(self, req: ApprovalRequest) -> None: ...

    async def wait(self, approval_id: uuid.UUID, timeout_s: float) -> ApprovalState: ...

    async def decide(
        self,
        approval_id: uuid.UUID,
        owner_id: uuid.UUID,
        decision: ApprovalDecision,
        now: datetime,
    ) -> ApprovalRequest | None:
        """Applies `decision` if `owner_id` owns the request. Idempotent on repeat calls."""
        ...

    def cancel(self, approval_id: uuid.UUID) -> None: ...


class LocalResultBroker(Protocol):
    """Process-local coordination for client-executed actions."""

    def register(self, invocation_id: str, owner_id: uuid.UUID) -> None: ...

    async def wait(self, invocation_id: str, timeout_s: float) -> str: ...

    async def wait_report(self, invocation_id: str, timeout_s: float) -> LocalActionReport: ...

    def report(
        self,
        invocation_id: str,
        owner_id: uuid.UUID,
        status: str,
        output: dict[str, object] | None = None,
        error_code: str | None = None,
    ) -> bool: ...

    def cancel(self, invocation_id: str) -> None: ...


class LastTargetStore(Protocol):
    """In-memory `(owner_id, pet_id) -> (alias, timestamp)` store (5-minute TTL)."""

    def get_last_target(
        self, owner_id: uuid.UUID, pet_id: str, now: datetime
    ) -> tuple[str, datetime] | None: ...

    def set_last_target(
        self, owner_id: uuid.UUID, pet_id: str, alias: str, now: datetime
    ) -> None: ...
