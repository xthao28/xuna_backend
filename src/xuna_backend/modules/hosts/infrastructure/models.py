"""SQLAlchemy ORM models for the Host Registry & XTP v1 module (UDC-3 §5.4).

Tables: `hosts`, `host_enrollments`, `host_manifests`, `host_grants`.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from xuna_backend.core.db import Base


class HostModel(Base):
    __tablename__ = "hosts"

    host_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("owners.owner_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    alias: Mapped[str] = mapped_column(String(24), nullable=False)
    platform: Mapped[str] = mapped_column(String(16), nullable=False)
    hostname: Mapped[str] = mapped_column(String(32), nullable=False)
    os_version: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_version: Mapped[str] = mapped_column(String(32), nullable=False)
    public_key: Mapped[bytes] = mapped_column(LargeBinary(65), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lan_fingerprint: Mapped[str | None] = mapped_column(String(96), nullable=True)
    lan_port: Mapped[int | None] = mapped_column(Integer(), nullable=True)


class HostEnrollmentModel(Base):
    __tablename__ = "host_enrollments"

    pending_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    code_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    public_key: Mapped[bytes] = mapped_column(LargeBinary(65), nullable=False)
    key_fingerprint: Mapped[str] = mapped_column(String(32), nullable=False)
    sas: Mapped[str] = mapped_column(String(6), nullable=False)
    platform: Mapped[str] = mapped_column(String(16), nullable=False)
    os_version: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_version: Mapped[str] = mapped_column(String(32), nullable=False)
    hostname: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("owners.owner_id", ondelete="CASCADE"),
        nullable=True,
    )
    alias: Mapped[str | None] = mapped_column(String(24), nullable=True)


class HostManifestModel(Base):
    __tablename__ = "host_manifests"

    host_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hosts.host_id", ondelete="CASCADE"),
        primary_key=True,
    )
    schema_version: Mapped[int] = mapped_column(Integer(), nullable=False, default=1)
    tools_json: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class HostGrantModel(Base):
    __tablename__ = "host_grants"
    __table_args__ = (UniqueConstraint("host_id", "family", name="uq_host_grants_host_family"),)

    grant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    host_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hosts.host_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    family: Mapped[str] = mapped_column(String(16), nullable=False)
    max_risk: Mapped[str] = mapped_column(String(4), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
