import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from xuna_backend.core.db import Base


class PetRow(Base):
    __tablename__ = "pets"
    __table_args__ = (CheckConstraint("pet_id ~ '^[0-9a-f]{32}$'", name="pet_id_hex"),)

    pet_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("owners.owner_id"), index=True)
    origin: Mapped[str] = mapped_column(String(32))
    last_generation: Mapped[int] = mapped_column(BigInteger)
    bound_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class BrainRow(Base):
    __tablename__ = "brains"
    __table_args__ = (
        Index(
            "uq_brains_one_active_per_pet",
            "pet_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    brain_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    pet_id: Mapped[str] = mapped_column(ForeignKey("pets.pet_id"))
    status: Mapped[str] = mapped_column(String(16))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DeviceRow(Base):
    __tablename__ = "devices"

    device_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    pet_id: Mapped[str] = mapped_column(ForeignKey("pets.pet_id"), index=True)
    public_key: Mapped[bytes] = mapped_column(LargeBinary(65), unique=True)
    bound_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChallengeRow(Base):
    __tablename__ = "device_challenges"

    challenge_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("owners.owner_id"))
    pet_id: Mapped[str] = mapped_column(String(32))
    nonce: Mapped[bytes] = mapped_column(LargeBinary(32))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    consumed: Mapped[bool] = mapped_column(Boolean)
