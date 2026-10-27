"""SQLAlchemy models for memory projections and candidates (CLD-14)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from xuna_backend.core.db import Base


class ProjectionRow(Base):
    __tablename__ = "memory_projections"
    __table_args__ = (
        UniqueConstraint(
            "pet_id", "snapshot_generation", name="uq_memory_projections_pet_generation"
        ),
        Index("ix_memory_projections_pet_created", "pet_id", "created_at"),
    )

    projection_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    pet_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("pets.pet_id", ondelete="CASCADE"), index=True
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("owners.owner_id", ondelete="CASCADE"), index=True
    )
    snapshot_generation: Mapped[int] = mapped_column(BigInteger, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    device_attested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MemoryCandidateRow(Base):
    __tablename__ = "memory_candidates"
    __table_args__ = (
        Index("ix_memory_candidates_pet_status", "pet_id", "status"),
        Index("ix_memory_candidates_owner_pet", "owner_id", "pet_id"),
    )

    candidate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("owners.owner_id", ondelete="CASCADE"), index=True
    )
    pet_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("pets.pet_id", ondelete="CASCADE"), index=True
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    candidate_type: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    committed_generation: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    domain_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_domains.domain_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    relation: Mapped[str | None] = mapped_column(String(16), nullable=True)
    related_candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memory_candidates.candidate_id", ondelete="SET NULL"),
        nullable=True,
    )
    fact_key: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)
