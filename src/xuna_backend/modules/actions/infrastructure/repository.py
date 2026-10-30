"""SQLAlchemy and in-memory repositories for `ApprovalRequest` (UDC-4 §5.2)."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.actions.domain.entities import (
    ApprovalRequest,
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
                )
            )
        else:
            row.state = req.state.value
            row.decided_at = req.decided_at
        await self._session.flush()

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        row = await self._session.get(ActionApprovalModel, approval_id)
        if row is None:
            return None
        return _to_domain(row)


class InMemoryApprovalRepository:
    def __init__(self) -> None:
        self._items: dict[uuid.UUID, ApprovalRequest] = {}

    async def save(self, req: ApprovalRequest) -> None:
        self._items[req.approval_id] = req

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        return self._items.get(approval_id)
