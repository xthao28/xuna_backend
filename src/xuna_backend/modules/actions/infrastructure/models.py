"""SQLAlchemy ORM models for `action_approvals` (UDC-4 §5.2, UDC-7),
`host_standing_approvals` (ADR-0014 D1) and `cu_sessions` (UDC-17 S17.6).

No raw arguments are stored — only `args_sha256` and the deterministic summary.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from xuna_backend.core.db import Base


class ActionApprovalModel(Base):
    __tablename__ = "action_approvals"

    approval_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("owners.owner_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    host_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hosts.host_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    target_alias: Mapped[str] = mapped_column(String(32), nullable=False)
    tool: Mapped[str] = mapped_column(String(64), nullable=False)
    args_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    risk: Mapped[str] = mapped_column(String(4), nullable=False)
    summary: Mapped[str] = mapped_column(String(240), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scope: Mapped[str] = mapped_column(String(24), nullable=False, default="once")
    signature: Mapped[bytes | None] = mapped_column(LargeBinary(64), nullable=True)
    signer_public_key: Mapped[bytes | None] = mapped_column(LargeBinary(65), nullable=True)


class HostStandingApprovalModel(Base):
    __tablename__ = "host_standing_approvals"

    approval_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("owners.owner_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    host_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hosts.host_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_alias: Mapped[str] = mapped_column(String(32), nullable=False)
    tool: Mapped[str] = mapped_column(String(64), nullable=False)
    args_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(String(240), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    max_uses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    signature: Mapped[bytes] = mapped_column(LargeBinary(64), nullable=False)
    signer_public_key: Mapped[bytes] = mapped_column(LargeBinary(65), nullable=False)


class CuSessionModel(Base):
    """A desktop control session as the backend enforces it (UDC-17 S17.6).

    Content-free apart from `goal` (the owner's stated goal, at most 200 characters), which
    is blanked when the session ends. UI trees and draft text are never stored.
    """

    __tablename__ = "cu_sessions"
    __table_args__ = (
        Index(
            "uq_cu_sessions_owner_active",
            "owner_id",
            unique=True,
            postgresql_where=text("active"),
        ),
    )

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("owners.owner_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    host_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hosts.host_id", ondelete="CASCADE"),
        nullable=False,
    )
    target_alias: Mapped[str] = mapped_column(String(32), nullable=False)
    goal: Mapped[str] = mapped_column(String(200), nullable=False)
    apps: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    domains: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    max_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    max_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    steps_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    typed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    owner_prompts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    stop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
