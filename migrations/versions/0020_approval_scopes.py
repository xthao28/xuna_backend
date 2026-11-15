"""Wave A R2: persistent approval scopes (`approval_scopes`).

Persists 15-minute `conversation_15m` R2 approval scopes in PostgreSQL so that
session grants survive server restarts and work across multiple workers.

Revision ID: 0020
Revises: 0019
Create Date: 2026-11-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "approval_scopes",
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("pet_id", sa.String(64), primary_key=True, nullable=False),
        sa.Column(
            "host_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("hosts.host_id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("tool", sa.String(64), primary_key=True, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_approval_scopes_owner_id", "approval_scopes", ["owner_id"])
    op.create_index("ix_approval_scopes_expires_at", "approval_scopes", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_approval_scopes_expires_at", table_name="approval_scopes")
    op.drop_index("ix_approval_scopes_owner_id", table_name="approval_scopes")
    op.drop_table("approval_scopes")
