"""Cloud suggestions table (CLD-30, CLD-31, H1-H8).

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cloud_suggestions",
        sa.Column("suggestion_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "pet_id",
            sa.String(32),
            sa.ForeignKey("pets.pet_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "brain_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("brains.brain_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("target", sa.String(64), nullable=False),
        sa.Column("delta", sa.Float(), nullable=False),
        sa.Column("bounds", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("provenance", sa.String(32), nullable=False),
        sa.Column("evidence_ref", sa.String(128), nullable=True),
        sa.Column("status", sa.String(32), server_default="pending", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_cloud_suggestions_pet_status",
        "cloud_suggestions",
        ["pet_id", "status"],
    )
    op.create_index(
        "ix_cloud_suggestions_pet_id",
        "cloud_suggestions",
        ["pet_id"],
    )
    op.create_index(
        "ix_cloud_suggestions_brain_id",
        "cloud_suggestions",
        ["brain_id"],
    )
    op.create_index(
        "ix_cloud_suggestions_expires_at",
        "cloud_suggestions",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_cloud_suggestions_expires_at", table_name="cloud_suggestions")
    op.drop_index("ix_cloud_suggestions_brain_id", table_name="cloud_suggestions")
    op.drop_index("ix_cloud_suggestions_pet_id", table_name="cloud_suggestions")
    op.drop_index("ix_cloud_suggestions_pet_status", table_name="cloud_suggestions")
    op.drop_table("cloud_suggestions")
