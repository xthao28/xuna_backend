"""SQLAlchemy models for Knowledge Domains (plan §6.2, §6.3, KD-3)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from xuna_backend.core.db import Base


class KnowledgeDomainRow(Base):
    __tablename__ = "knowledge_domains"
    __table_args__ = (
        Index("ix_knowledge_domains_pet_status", "pet_id", "status"),
        Index("ix_knowledge_domains_pet_slot", "pet_id", "device_slot"),
    )

    domain_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    pet_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("pets.pet_id", ondelete="CASCADE"), index=True
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("owners.owner_id", ondelete="CASCADE"), index=True
    )
    device_slot: Mapped[int | None] = mapped_column(Integer, nullable=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active", index=True)
    origin: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)
    pack_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    expert_brief: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    safety_tier: Mapped[str] = mapped_column(
        String(16), nullable=False, default="general", server_default="general"
    )


class KnowledgeSuggestionRow(Base):
    __tablename__ = "knowledge_suggestions"
    __table_args__ = (Index("ix_knowledge_suggestions_pet_status", "pet_id", "status"),)

    suggestion_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("owners.owner_id", ondelete="CASCADE"), index=True
    )
    pet_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("pets.pet_id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    proposed_name: Mapped[str] = mapped_column(String(64), nullable=False)
    proposed_description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    member_candidate_ids: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    target_domain_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_domains.domain_id", ondelete="SET NULL"),
        nullable=True,
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class KnowledgeSummaryRow(Base):
    __tablename__ = "knowledge_summaries"

    domain_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_domains.domain_id", ondelete="CASCADE"),
        primary_key=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_candidate_ids: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
