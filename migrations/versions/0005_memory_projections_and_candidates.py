"""Memory projections and candidates (CLD-14).

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_projections",
        sa.Column("projection_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "pet_id",
            sa.String(32),
            sa.ForeignKey("pets.pet_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("snapshot_generation", sa.BigInteger(), nullable=False),
        sa.Column("schema_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("payload_sha256", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("device_attested", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "pet_id", "snapshot_generation", name="uq_memory_projections_pet_generation"
        ),
    )
    op.create_index(
        "ix_memory_projections_pet_created",
        "memory_projections",
        ["pet_id", "created_at"],
    )
    op.create_index(
        "ix_memory_projections_owner_id",
        "memory_projections",
        ["owner_id"],
    )
    op.create_index(
        "ix_memory_projections_pet_id",
        "memory_projections",
        ["pet_id"],
    )

    op.create_table(
        "memory_candidates",
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "pet_id",
            sa.String(32),
            sa.ForeignKey("pets.pet_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("candidate_type", sa.String(32), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("committed_generation", sa.BigInteger(), nullable=True),
    )
    op.create_index(
        "ix_memory_candidates_pet_status",
        "memory_candidates",
        ["pet_id", "status"],
    )
    op.create_index(
        "ix_memory_candidates_owner_pet",
        "memory_candidates",
        ["owner_id", "pet_id"],
    )
    op.create_index(
        "ix_memory_candidates_owner_id",
        "memory_candidates",
        ["owner_id"],
    )
    op.create_index(
        "ix_memory_candidates_pet_id",
        "memory_candidates",
        ["pet_id"],
    )
    op.create_index(
        "ix_memory_candidates_status",
        "memory_candidates",
        ["status"],
    )


def downgrade() -> None:
    op.drop_table("memory_candidates")
    op.drop_table("memory_projections")
