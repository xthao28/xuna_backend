"""SQLAlchemy models for Direct Device Transport (CLD-40, CLD-41, E14)."""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from xuna_backend.core.db import Base


class DeviceTransportStatusRow(Base):
    __tablename__ = "device_transport_statuses"
    __table_args__ = (
        Index("ix_device_transport_statuses_pet_id", "pet_id"),
        Index("ix_device_transport_statuses_status", "status"),
    )

    device_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("devices.device_id", ondelete="CASCADE"),
        primary_key=True,
    )
    pet_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("pets.pet_id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    client_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    firmware: Mapped[str | None] = mapped_column(String(32), nullable=True)
    connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
