"""SQLAlchemy models for backup manifests and deletion tombstones (CLD-20)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
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


class BackupManifestRow(Base):
    __tablename__ = "backup_manifests"
    __table_args__ = (
        Index("ix_backup_manifests_pet_created", "pet_id", "created_at"),
        Index("ix_backup_manifests_owner_id", "owner_id"),
        Index("ix_backup_manifests_brain_id", "brain_id"),
    )

    backup_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    pet_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("pets.pet_id", ondelete="CASCADE"), index=True
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("owners.owner_id", ondelete="CASCADE"), index=True
    )
    brain_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("brains.brain_id", ondelete="CASCADE"), index=True
    )
    snapshot_generation: Mapped[int] = mapped_column(BigInteger, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    encrypted_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DeletionTombstoneRow(Base):
    __tablename__ = "deletion_tombstones"
    __table_args__ = (
        UniqueConstraint("pet_id", "item_type", "item_id", name="uq_deletion_tombstones_item"),
        Index("ix_deletion_tombstones_pet_type", "pet_id", "item_type"),
        Index("ix_deletion_tombstones_owner_pet", "owner_id", "pet_id"),
    )

    tombstone_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    pet_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("pets.pet_id", ondelete="CASCADE"), index=True
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("owners.owner_id", ondelete="CASCADE"), index=True
    )
    item_type: Mapped[str] = mapped_column(String(32), nullable=False)
    item_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
