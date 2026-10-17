import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from xuna_backend.core.db import Base


class AuditRow(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_owner_time", "owner_id", "occurred_at"),)

    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    action: Mapped[str] = mapped_column(String(48))
    subject: Mapped[str] = mapped_column(String(64))
    request_id: Mapped[str] = mapped_column(String(64))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
