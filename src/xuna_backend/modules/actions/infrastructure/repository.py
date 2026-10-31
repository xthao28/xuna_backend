"""SQLAlchemy and in-memory repositories for `ApprovalRequest` (UDC-4 §5.2, UDC-7)."""

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.actions.domain.entities import (
    ApprovalRequest,
    ApprovalScope,
    ApprovalState,
)
from xuna_backend.modules.actions.infrastructure.models import ActionApprovalModel
from xuna_backend.modules.hosts.domain.entities import HostId, RiskTier


def _to_domain(row: ActionApprovalModel) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=row.approval_id,
        owner_id=row.owner_id,
        host_id=HostId(row.host_id) if row.host_id is not None else None,
        target_alias=row.target_alias,
        tool=row.tool,
        args_sha256=row.args_sha256,
        risk=RiskTier(row.risk),
        summary=row.summary,
        created_at=row.created_at,
        expires_at=row.expires_at,
        state=ApprovalState(row.state),
        decided_at=row.decided_at,
        scope=ApprovalScope(row.scope) if row.scope else ApprovalScope.ONCE,
        signature=bytes(row.signature) if row.signature is not None else None,
        signer_public_key=(
            bytes(row.signer_public_key) if row.signer_public_key is not None else None
        ),
    )


class SqlApprovalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, req: ApprovalRequest) -> None:
        row = await self._session.get(ActionApprovalModel, req.approval_id)
        if row is None:
            self._session.add(
                ActionApprovalModel(
                    approval_id=req.approval_id,
                    owner_id=req.owner_id,
                    host_id=req.host_id.value if req.host_id is not None else None,
                    target_alias=req.target_alias,
                    tool=req.tool,
                    args_sha256=req.args_sha256,
                    risk=req.risk.value,
                    summary=req.summary,
                    state=req.state.value,
                    created_at=req.created_at,
                    expires_at=req.expires_at,
                    decided_at=req.decided_at,
                    scope=req.scope.value,
                    signature=req.signature,
                    signer_public_key=req.signer_public_key,
                )
            )
        else:
            row.state = req.state.value
            row.decided_at = req.decided_at
            row.scope = req.scope.value
            row.signature = req.signature
            row.signer_public_key = req.signer_public_key
        await self._session.flush()

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        row = await self._session.get(ActionApprovalModel, approval_id)
        if row is None:
            return None
        return _to_domain(row)

    async def reject_pending_for_owner(
        self, owner_id: uuid.UUID, now: datetime
    ) -> int:
        stmt = select(ActionApprovalModel).where(
            ActionApprovalModel.owner_id == owner_id,
            ActionApprovalModel.state == ApprovalState.PENDING.value,
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        for row in rows:
            row.state = ApprovalState.REJECTED.value
            row.decided_at = now
        if rows:
            await self._session.flush()
        return len(rows)


class InMemoryApprovalRepository:
    def __init__(self) -> None:
        self._items: dict[uuid.UUID, ApprovalRequest] = {}

    async def save(self, req: ApprovalRequest) -> None:
        self._items[req.approval_id] = req

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        return self._items.get(approval_id)

    async def reject_pending_for_owner(
        self, owner_id: uuid.UUID, now: datetime
    ) -> int:
        count = 0
        for aid, req in list(self._items.items()):
            if req.owner_id == owner_id and req.state == ApprovalState.PENDING:
                self._items[aid] = ApprovalRequest(
                    approval_id=req.approval_id,
                    owner_id=req.owner_id,
                    host_id=req.host_id,
                    target_alias=req.target_alias,
                    tool=req.tool,
                    args_sha256=req.args_sha256,
                    risk=req.risk,
                    summary=req.summary,
                    created_at=req.created_at,
                    expires_at=req.expires_at,
                    state=ApprovalState.REJECTED,
                    decided_at=now,
                    scope=req.scope,
                )
                count += 1
        return count
