"""SQL implementation of TransportStatusRepository (CLD-40, CLD-41, E14)."""

import uuid
from datetime import UTC

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.transport.domain.entities import (
    DeviceConnectionStatus,
    DeviceStatusRecord,
)
from xuna_backend.modules.transport.infrastructure.models import DeviceTransportStatusRow


class SqlTransportStatusRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_status(self, record: DeviceStatusRecord) -> None:
        stmt = (
            insert(DeviceTransportStatusRow)
            .values(
                device_id=record.device_id,
                pet_id=record.pet_id,
                status=record.status.value,
                last_seen_at=record.last_seen_at,
                ip_address=record.ip_address,
                client_id=record.client_id,
                firmware=record.firmware,
                connected_at=record.connected_at,
            )
            .on_conflict_do_update(
                index_elements=["device_id"],
                set_={
                    "status": record.status.value,
                    "last_seen_at": record.last_seen_at,
                    "ip_address": record.ip_address,
                    "client_id": record.client_id,
                    "firmware": record.firmware,
                    "connected_at": record.connected_at,
                },
            )
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def get_status(self, device_id: uuid.UUID) -> DeviceStatusRecord | None:
        result = await self._session.execute(
            select(DeviceTransportStatusRow).where(DeviceTransportStatusRow.device_id == device_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def list_statuses_for_pet(self, pet_id: str) -> list[DeviceStatusRecord]:
        result = await self._session.execute(
            select(DeviceTransportStatusRow)
            .where(DeviceTransportStatusRow.pet_id == pet_id)
            .order_by(DeviceTransportStatusRow.last_seen_at.desc())
        )
        return [self._to_entity(r) for r in result.scalars().all()]

    def _to_entity(self, row: DeviceTransportStatusRow) -> DeviceStatusRecord:
        last_seen = (
            row.last_seen_at
            if row.last_seen_at.tzinfo is not None
            else row.last_seen_at.replace(tzinfo=UTC)
        )
        return DeviceStatusRecord(
            device_id=row.device_id,
            pet_id=row.pet_id,
            status=DeviceConnectionStatus(row.status),
            last_seen_at=last_seen,
            ip_address=row.ip_address,
            client_id=row.client_id,
            firmware=row.firmware,
            connected_at=row.connected_at,
        )
