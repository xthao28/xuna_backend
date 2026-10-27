"""Backup manifests and deletion tombstones (CLD-20).

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "backup_manifests",
        sa.Column("backup_id", postgresql.UUID(as_uuid=True), primary_key=True),
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
        sa.Column(
            "brain_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("brains.brain_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("snapshot_generation", sa.BigInteger(), nullable=False),
        sa.Column("schema_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("payload_sha256", sa.String(64), nullable=False),
        sa.Column("encrypted_data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_backup_manifests_pet_created",
        "backup_manifests",
        ["pet_id", "created_at"],
    )
    op.create_index(
        "ix_backup_manifests_owner_id",
        "backup_manifests",
        ["owner_id"],
    )
    op.create_index(
        "ix_backup_manifests_brain_id",
        "backup_manifests",
        ["brain_id"],
    )
    op.create_index(
        "ix_backup_manifests_pet_id",
        "backup_manifests",
        ["pet_id"],
    )

    op.create_table(
        "deletion_tombstones",
        sa.Column("tombstone_id", postgresql.UUID(as_uuid=True), primary_key=True),
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
        sa.Column("item_type", sa.String(32), nullable=False),
        sa.Column("item_id", sa.String(128), nullable=False),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("pet_id", "item_type", "item_id", name="uq_deletion_tombstones_item"),
    )
    op.create_index(
        "ix_deletion_tombstones_pet_type",
        "deletion_tombstones",
        ["pet_id", "item_type"],
    )
    op.create_index(
        "ix_deletion_tombstones_owner_pet",
        "deletion_tombstones",
        ["owner_id", "pet_id"],
    )
    op.create_index(
        "ix_deletion_tombstones_pet_id",
        "deletion_tombstones",
        ["pet_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_deletion_tombstones_pet_id", table_name="deletion_tombstones")
    op.drop_index("ix_deletion_tombstones_owner_pet", table_name="deletion_tombstones")
    op.drop_index("ix_deletion_tombstones_pet_type", table_name="deletion_tombstones")
    op.drop_table("deletion_tombstones")

    op.drop_index("ix_backup_manifests_pet_id", table_name="backup_manifests")
    op.drop_index("ix_backup_manifests_brain_id", table_name="backup_manifests")
    op.drop_index("ix_backup_manifests_owner_id", table_name="backup_manifests")
    op.drop_index("ix_backup_manifests_pet_created", table_name="backup_manifests")
    op.drop_table("backup_manifests")
