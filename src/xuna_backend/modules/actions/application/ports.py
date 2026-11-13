"""Ports for the `actions` application layer (UDC-4 §5.2, UDC-7)."""

import uuid
from datetime import datetime
from typing import Protocol

from xuna_backend.modules.actions.application.computer_use import ComputerUseSessionState
from xuna_backend.modules.actions.domain.entities import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScope,
    ApprovalState,
    LocalActionReport,
)
from xuna_backend.modules.actions.domain.standing_approval import StandingApproval
from xuna_backend.modules.hosts.domain.entities import HostId


class ApprovalRepository(Protocol):
    """Durable persistence for `ApprovalRequest` records (`action_approvals`)."""

    async def save(self, req: ApprovalRequest) -> None: ...

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None: ...

    async def reject_pending_for_owner(self, owner_id: uuid.UUID, now: datetime) -> int: ...

    async def count_approved(
        self,
        owner_id: uuid.UUID,
        host_id: HostId,
        tool: str,
        args_sha256: str,
        since: datetime,
    ) -> int:
        """Approved requests for exactly this call (host, tool, `args_sha256`) decided at or
        after `since`, by any route (owner decision, 15-minute scope or standing approval)."""
        ...


class StandingApprovalRepository(Protocol):
    """Durable store for `StandingApproval` records (`host_standing_approvals`, ADR-0014)."""

    async def save(self, approval: StandingApproval) -> None: ...

    async def get(self, owner_id: uuid.UUID, approval_id: uuid.UUID) -> StandingApproval | None: ...

    async def list_active(self, owner_id: uuid.UUID, now: datetime) -> list[StandingApproval]: ...

    async def consume_match(
        self,
        owner_id: uuid.UUID,
        host_id: HostId,
        tool: str,
        args_sha256: str,
        now: datetime,
    ) -> StandingApproval | None:
        """Atomically finds a usable approval for exactly this call and counts one use."""
        ...

    async def revoke(self, owner_id: uuid.UUID, approval_id: uuid.UUID, now: datetime) -> bool:
        """Marks an owner's approval revoked. `False` when it does not exist or was revoked."""
        ...


class ApprovalBroker(Protocol):
    """Process-local coordination for pending owner approvals (60 s TTL)."""

    async def create(self, req: ApprovalRequest) -> None: ...

    async def wait(self, approval_id: uuid.UUID, timeout_s: float) -> ApprovalState: ...

    def get_slot_request(self, approval_id: uuid.UUID) -> ApprovalRequest | None: ...

    async def decide(
        self,
        approval_id: uuid.UUID,
        owner_id: uuid.UUID,
        decision: ApprovalDecision,
        now: datetime,
        *,
        scope: ApprovalScope = ApprovalScope.ONCE,
        signature: bytes | None = None,
        signer_public_key: bytes | None = None,
    ) -> ApprovalRequest | None:
        """Applies `decision` if `owner_id` owns the request. Idempotent on repeat calls."""
        ...

    def reject_all_for_owner(self, owner_id: uuid.UUID, now: datetime) -> list[ApprovalRequest]: ...

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

    def cancel_for_owner(self, owner_id: uuid.UUID) -> int: ...

    def cancel(self, invocation_id: str) -> None: ...


class LastTargetStore(Protocol):
    """In-memory `(owner_id, pet_id) -> (alias, timestamp)` store (5-minute TTL)."""

    def get_last_target(
        self, owner_id: uuid.UUID, pet_id: str, now: datetime
    ) -> tuple[str, datetime] | None: ...

    def set_last_target(
        self, owner_id: uuid.UUID, pet_id: str, alias: str, now: datetime
    ) -> None: ...

    def clear_owner(self, owner_id: uuid.UUID) -> None: ...


class ApprovalScopeStore(Protocol):
    """In-memory 15-minute `conversation_15m` R2 approval scope store (`UDC-7` S7.1)."""

    def has_valid_scope(
        self,
        owner_id: uuid.UUID,
        pet_id: str,
        host_id: HostId,
        tool: str,
        now: datetime,
    ) -> bool: ...

    def grant_scope(
        self,
        owner_id: uuid.UUID,
        pet_id: str,
        host_id: HostId,
        tool: str,
        now: datetime,
    ) -> datetime: ...

    def clear_owner(self, owner_id: uuid.UUID) -> int: ...


class ComputerUseSessionStore(Protocol):
    """Desktop control sessions as the backend enforces them (UDC-8, UDC-17 S17.6).

    One running session per owner. Writes go through these methods so a durable store
    persists them; callers never rely on mutating a returned state. `cached_nodes` and
    `typed_since_commit` stay in process memory in every implementation.
    """

    async def get(self, session_id: str) -> ComputerUseSessionState | None: ...

    async def get_active_for_owner(self, owner_id: uuid.UUID) -> ComputerUseSessionState | None:
        """The owner's running session, before budgets and the heartbeat are applied."""
        ...

    async def start(self, state: ComputerUseSessionState, now: datetime) -> None:
        """Stores a new running session; any other running one of the owner ends as
        `replaced`."""
        ...

    async def mark_ended(
        self, session_id: str, reason: str, now: datetime
    ) -> ComputerUseSessionState | None:
        """Ends the session (no-op if already ended; the first reason is kept) and drops its
        goal, cached UI tree and draft text."""
        ...

    async def stop_all_for_owner(self, owner_id: uuid.UUID, reason: str, now: datetime) -> int: ...

    async def record_step(self, session_id: str) -> int | None:
        """Atomically counts one step of a running session; returns the new count, or `None`
        when the session is not running."""
        ...

    async def set_scope(
        self, session_id: str, apps: tuple[str, ...], domains: tuple[str, ...]
    ) -> None: ...

    async def touch_heartbeat(self, session_id: str, owner_id: uuid.UUID, now: datetime) -> bool:
        """Records a phone heartbeat for the owner's running session; `False` otherwise."""
        ...

    async def set_typed_count(self, session_id: str, count: int) -> None: ...

    async def record_prompt(self, session_id: str) -> int | None:
        """Counts one owner approval asked during a running session; returns the new count,
        or `None` when the session is not running."""
        ...
