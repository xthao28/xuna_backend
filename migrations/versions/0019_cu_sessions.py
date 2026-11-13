"""UDC-17 S17.6: durable desktop control sessions (`cu_sessions`).

Replaces the in-memory session store so a session survives a backend restart. Content-free
apart from the running session's goal; UI trees and draft text stay in process memory.

Revision ID: 0019
Revises: 0018
Create Date: 2026-11-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cu_sessions",
        sa.Column("session_id", sa.String(64), primary_key=True),
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
        sa.Column("goal", sa.String(200), nullable=False),
        sa.Column("apps", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("domains", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("max_steps", sa.Integer(), nullable=False),
        sa.Column("max_minutes", sa.Integer(), nullable=False),
        sa.Column("steps_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("typed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("owner_prompts", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("stop_reason", sa.String(64), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_cu_sessions_owner_id", "cu_sessions", ["owner_id"])
    # At most one running session per owner, also across workers.
    op.create_index(
        "uq_cu_sessions_owner_active",
        "cu_sessions",
        ["owner_id"],
        unique=True,
        postgresql_where=sa.text("active"),
    )


def downgrade() -> None:
    op.drop_index("uq_cu_sessions_owner_active", table_name="cu_sessions")
    op.drop_index("ix_cu_sessions_owner_id", table_name="cu_sessions")
    op.drop_table("cu_sessions")
