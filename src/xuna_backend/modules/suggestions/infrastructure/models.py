"""SQLAlchemy models for cloud suggestions (CLD-30, CLD-31, H1-H8)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from xuna_backend.core.db import Base


class CloudSuggestionRow(Base):
    __tablename__ = "cloud_suggestions"
    __table_args__ = (
        Index("ix_cloud_suggestions_pet_status", "pet_id", "status"),
        Index("ix_cloud_suggestions_pet_id", "pet_id"),
        Index("ix_cloud_suggestions_brain_id", "brain_id"),
        Index("ix_cloud_suggestions_expires_at", "expires_at"),
    )

    suggestion_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    pet_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("pets.pet_id", ondelete="CASCADE"), index=True
    )
    brain_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("brains.brain_id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    target: Mapped[str] = mapped_column(String(64), nullable=False)
    delta: Mapped[float] = mapped_column(Float, nullable=False)
    bounds: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    provenance: Mapped[str] = mapped_column(String(32), nullable=False)
    evidence_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
