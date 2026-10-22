import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from xuna_backend.core.db import Base


class ConsentRow(Base):
    """Append-only history; the highest version is the current consent."""

    __tablename__ = "consent_records"

    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    aggregate_metrics: Mapped[bool] = mapped_column(Boolean)
    experience_events: Mapped[bool] = mapped_column(Boolean)
    conversation_text: Mapped[bool] = mapped_column(Boolean)
    conversation_voice: Mapped[bool] = mapped_column(Boolean, server_default="false")
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    policy: Mapped[str] = mapped_column(String(32))
