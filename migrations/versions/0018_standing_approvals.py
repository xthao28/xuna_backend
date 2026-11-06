"""ADR-0014 D1: standing (pre-issued) approvals for exactly-constrained R2 host calls.

Revision ID: 0018
Revises: 0017
Create Date: 2026-11-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "host_standing_approvals",
        sa.Column("approval_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "host_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("hosts.host_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("target_alias", sa.String(32), nullable=False),
        sa.Column("tool", sa.String(64), nullable=False),
        sa.Column("args_sha256", sa.String(64), nullable=False),
        sa.Column("summary", sa.String(240), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("max_uses", sa.Integer(), nullable=True),
        sa.Column("uses", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("signature", sa.LargeBinary(64), nullable=False),
        sa.Column("signer_public_key", sa.LargeBinary(65), nullable=False),
    )
    op.create_index("ix_host_standing_approvals_owner_id", "host_standing_approvals", ["owner_id"])
    op.create_index("ix_host_standing_approvals_host_id", "host_standing_approvals", ["host_id"])


def downgrade() -> None:
    op.drop_index("ix_host_standing_approvals_host_id", table_name="host_standing_approvals")
    op.drop_index("ix_host_standing_approvals_owner_id", table_name="host_standing_approvals")
    op.drop_table("host_standing_approvals")
