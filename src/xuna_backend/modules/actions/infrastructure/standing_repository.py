"""SQLAlchemy and in-memory repositories for `StandingApproval` (ADR-0014 D1)."""

import uuid
from dataclasses import replace
from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.actions.domain.standing_approval import StandingApproval
from xuna_backend.modules.actions.infrastructure.models import HostStandingApprovalModel
from xuna_backend.modules.hosts.domain.entities import HostId


def _to_domain(row: HostStandingApprovalModel) -> StandingApproval:
    return StandingApproval(
        approval_id=row.approval_id,
        owner_id=row.owner_id,
        host_id=HostId(row.host_id),
        target_alias=row.target_alias,
        tool=row.tool,
        args_sha256=row.args_sha256,
        summary=row.summary,
        created_at=row.created_at,
        expires_at=row.expires_at,
        max_uses=row.max_uses,
        signature=bytes(row.signature),
        signer_public_key=bytes(row.signer_public_key),
        uses=row.uses,
        revoked_at=row.revoked_at,
    )


class SqlStandingApprovalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, approval: StandingApproval) -> None:
        self._session.add(
            HostStandingApprovalModel(
                approval_id=approval.approval_id,
                owner_id=approval.owner_id,
                host_id=approval.host_id.value,
                target_alias=approval.target_alias,
                tool=approval.tool,
                args_sha256=approval.args_sha256,
                summary=approval.summary,
                created_at=approval.created_at,
                expires_at=approval.expires_at,
                max_uses=approval.max_uses,
                uses=approval.uses,
                revoked_at=approval.revoked_at,
                signature=approval.signature,
                signer_public_key=approval.signer_public_key,
            )
        )
        await self._session.flush()

    async def get(self, owner_id: uuid.UUID, approval_id: uuid.UUID) -> StandingApproval | None:
        row = await self._session.get(HostStandingApprovalModel, approval_id)
        if row is None or row.owner_id != owner_id:
            return None
        return _to_domain(row)

    async def list_active(self, owner_id: uuid.UUID, now: datetime) -> list[StandingApproval]:
        stmt = (
            select(HostStandingApprovalModel)
            .where(
                HostStandingApprovalModel.owner_id == owner_id,
                HostStandingApprovalModel.revoked_at.is_(None),
                HostStandingApprovalModel.expires_at > now,
                or_(
                    HostStandingApprovalModel.max_uses.is_(None),
                    HostStandingApprovalModel.uses < HostStandingApprovalModel.max_uses,
                ),
            )
            .order_by(HostStandingApprovalModel.created_at.desc())
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [_to_domain(row) for row in rows]

    async def consume_match(
        self,
        owner_id: uuid.UUID,
        host_id: HostId,
        tool: str,
        args_sha256: str,
        now: datetime,
    ) -> StandingApproval | None:
        stmt = (
            select(HostStandingApprovalModel)
            .where(
                HostStandingApprovalModel.owner_id == owner_id,
                HostStandingApprovalModel.host_id == host_id.value,
                HostStandingApprovalModel.tool == tool,
                HostStandingApprovalModel.args_sha256 == args_sha256,
                HostStandingApprovalModel.revoked_at.is_(None),
                HostStandingApprovalModel.expires_at > now,
                or_(
                    HostStandingApprovalModel.max_uses.is_(None),
                    HostStandingApprovalModel.uses < HostStandingApprovalModel.max_uses,
                ),
            )
            .order_by(HostStandingApprovalModel.expires_at.asc())
            .limit(1)
            .with_for_update()
        )
        row = (await self._session.execute(stmt)).scalars().first()
        if row is None:
            return None
        row.uses = row.uses + 1
        await self._session.flush()
        return _to_domain(row)

    async def revoke(self, owner_id: uuid.UUID, approval_id: uuid.UUID, now: datetime) -> bool:
        row = await self._session.get(HostStandingApprovalModel, approval_id)
        if row is None or row.owner_id != owner_id or row.revoked_at is not None:
            return False
        row.revoked_at = now
        await self._session.flush()
        return True


class InMemoryStandingApprovalRepository:
    def __init__(self) -> None:
        self._items: dict[uuid.UUID, StandingApproval] = {}

    async def save(self, approval: StandingApproval) -> None:
        self._items[approval.approval_id] = approval

    async def get(self, owner_id: uuid.UUID, approval_id: uuid.UUID) -> StandingApproval | None:
        item = self._items.get(approval_id)
        return item if item is not None and item.owner_id == owner_id else None

    async def list_active(self, owner_id: uuid.UUID, now: datetime) -> list[StandingApproval]:
        items = [a for a in self._items.values() if a.owner_id == owner_id and a.is_usable(now)]
        return sorted(items, key=lambda a: a.created_at, reverse=True)

    async def consume_match(
        self,
        owner_id: uuid.UUID,
        host_id: HostId,
        tool: str,
        args_sha256: str,
        now: datetime,
    ) -> StandingApproval | None:
        candidates = [
            a
            for a in self._items.values()
            if a.owner_id == owner_id
            and a.host_id == host_id
            and a.tool == tool
            and a.args_sha256 == args_sha256
            and a.is_usable(now)
        ]
        if not candidates:
            return None
        chosen = min(candidates, key=lambda a: a.expires_at)
        updated = replace(chosen, uses=chosen.uses + 1)
        self._items[chosen.approval_id] = updated
        return updated

    async def revoke(self, owner_id: uuid.UUID, approval_id: uuid.UUID, now: datetime) -> bool:
        item = self._items.get(approval_id)
        if item is None or item.owner_id != owner_id or item.revoked_at is not None:
            return False
        self._items[approval_id] = replace(item, revoked_at=now)
        return True
