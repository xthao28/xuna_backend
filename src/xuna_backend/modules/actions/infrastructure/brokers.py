"""In-memory brokers for approvals, local HID results, and last-target memory (UDC-4 §5.2)."""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from xuna_backend.modules.actions.domain.entities import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalState,
    LocalActionReport,
)
from xuna_backend.modules.actions.domain.target_resolver import LAST_TARGET_TTL


@dataclass
class _ApprovalSlot:
    request: ApprovalRequest
    event: asyncio.Event = field(default_factory=asyncio.Event)


class InMemoryApprovalBroker:
    """Single-process asyncio coordinator for owner approvals (60 s TTL)."""

    def __init__(self) -> None:
        self._slots: dict[uuid.UUID, _ApprovalSlot] = {}

    async def create(self, req: ApprovalRequest) -> None:
        self._slots[req.approval_id] = _ApprovalSlot(request=req)

    async def wait(self, approval_id: uuid.UUID, timeout_s: float) -> ApprovalState:
        slot = self._slots.get(approval_id)
        if slot is None:
            return ApprovalState.EXPIRED
        if slot.request.state != ApprovalState.PENDING:
            return slot.request.state
        try:
            await asyncio.wait_for(slot.event.wait(), timeout=timeout_s)
        except TimeoutError:
            current = self._slots.get(approval_id)
            if current is not None and current.request.state == ApprovalState.PENDING:
                current.request = ApprovalRequest(
                    approval_id=current.request.approval_id,
                    owner_id=current.request.owner_id,
                    host_id=current.request.host_id,
                    target_alias=current.request.target_alias,
                    tool=current.request.tool,
                    args_sha256=current.request.args_sha256,
                    risk=current.request.risk,
                    summary=current.request.summary,
                    created_at=current.request.created_at,
                    expires_at=current.request.expires_at,
                    state=ApprovalState.EXPIRED,
                    decided_at=current.request.expires_at,
                )
            return ApprovalState.EXPIRED
        current = self._slots.get(approval_id)
        if current is None:
            return ApprovalState.EXPIRED
        return current.request.state

    async def decide(
        self,
        approval_id: uuid.UUID,
        owner_id: uuid.UUID,
        decision: ApprovalDecision,
        now: datetime,
    ) -> ApprovalRequest | None:
        slot = self._slots.get(approval_id)
        if slot is None:
            return None
        if slot.request.owner_id != owner_id:
            return None
        # Idempotent if already decided
        if slot.request.state != ApprovalState.PENDING:
            return slot.request
        if slot.request.is_expired(now):
            slot.request = ApprovalRequest(
                approval_id=slot.request.approval_id,
                owner_id=slot.request.owner_id,
                host_id=slot.request.host_id,
                target_alias=slot.request.target_alias,
                tool=slot.request.tool,
                args_sha256=slot.request.args_sha256,
                risk=slot.request.risk,
                summary=slot.request.summary,
                created_at=slot.request.created_at,
                expires_at=slot.request.expires_at,
                state=ApprovalState.EXPIRED,
                decided_at=now,
            )
            slot.event.set()
            return slot.request

        new_state = (
            ApprovalState.APPROVED
            if decision == ApprovalDecision.APPROVE
            else ApprovalState.REJECTED
        )
        slot.request = ApprovalRequest(
            approval_id=slot.request.approval_id,
            owner_id=slot.request.owner_id,
            host_id=slot.request.host_id,
            target_alias=slot.request.target_alias,
            tool=slot.request.tool,
            args_sha256=slot.request.args_sha256,
            risk=slot.request.risk,
            summary=slot.request.summary,
            created_at=slot.request.created_at,
            expires_at=slot.request.expires_at,
            state=new_state,
            decided_at=now,
        )
        slot.event.set()
        return slot.request

    def cancel(self, approval_id: uuid.UUID) -> None:
        slot = self._slots.pop(approval_id, None)
        if slot is not None:
            slot.event.set()


@dataclass
class _LocalResultSlot:
    owner_id: uuid.UUID
    report: LocalActionReport | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)


class InMemoryLocalResultBroker:
    """Single-process coordinator for `board_hid_request` and `local_action_request` results."""

    def __init__(self) -> None:
        self._slots: dict[str, _LocalResultSlot] = {}

    def register(self, invocation_id: str, owner_id: uuid.UUID) -> None:
        self._slots[invocation_id] = _LocalResultSlot(owner_id=owner_id)

    async def wait(self, invocation_id: str, timeout_s: float) -> str:
        res = await self.wait_report(invocation_id, timeout_s)
        return res.status

    async def wait_report(self, invocation_id: str, timeout_s: float) -> LocalActionReport:
        slot = self._slots.get(invocation_id)
        if slot is None:
            return LocalActionReport(status="timeout")
        if slot.report is not None:
            self._slots.pop(invocation_id, None)
            return slot.report
        try:
            await asyncio.wait_for(slot.event.wait(), timeout=timeout_s)
        except TimeoutError:
            self._slots.pop(invocation_id, None)
            return LocalActionReport(status="timeout")
        current = self._slots.pop(invocation_id, None)
        if current is None or current.report is None:
            return LocalActionReport(status="timeout")
        return current.report

    def report(
        self,
        invocation_id: str,
        owner_id: uuid.UUID,
        status: str,
        output: dict[str, object] | None = None,
        error_code: str | None = None,
    ) -> bool:
        slot = self._slots.get(invocation_id)
        if slot is None or slot.owner_id != owner_id:
            return False
        if slot.report is None:
            slot.report = LocalActionReport(
                status=status,
                output=dict(output) if isinstance(output, dict) else None,
                error_code=error_code,
            )
            slot.event.set()
        return True

    def cancel(self, invocation_id: str) -> None:
        slot = self._slots.pop(invocation_id, None)
        if slot is not None:
            slot.event.set()


class InMemoryLastTargetStore:
    """In-memory `(owner_id, pet_id) -> (alias, timestamp)` store with 5-minute TTL."""

    def __init__(self) -> None:
        self._entries: dict[tuple[uuid.UUID, str], tuple[str, datetime]] = {}

    def get_last_target(
        self, owner_id: uuid.UUID, pet_id: str, now: datetime
    ) -> tuple[str, datetime] | None:
        entry = self._entries.get((owner_id, pet_id))
        if entry is None:
            return None
        _, ts = entry
        if now - ts > LAST_TARGET_TTL:
            self._entries.pop((owner_id, pet_id), None)
            return None
        return entry

    def set_last_target(self, owner_id: uuid.UUID, pet_id: str, alias: str, now: datetime) -> None:
        self._entries[(owner_id, pet_id)] = (alias, now)
