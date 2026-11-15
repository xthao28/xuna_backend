"""Wave D M1: durable autonomous jobs queue and persistence (`autonomous_jobs`).

Persists AutonomousJob records in PostgreSQL so that long-running jobs (40-50 steps)
survive detached background execution outside single SSE connections.

Revision ID: 0021
Revises: 0020
Create Date: 2026-11-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "autonomous_jobs",
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pet_id", sa.String(64), nullable=False),
        sa.Column(
            "host_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("hosts.host_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column(
            "sub_goals",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("active_sub_goal_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_steps", sa.Integer(), nullable=False, server_default="50"),
        sa.Column("steps_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pending_approval_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("checkpoint_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("stop_reason", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_autonomous_jobs_owner_id", "autonomous_jobs", ["owner_id"])
    op.create_index("ix_autonomous_jobs_state", "autonomous_jobs", ["state"])
    op.create_index("ix_autonomous_jobs_created_at", "autonomous_jobs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_autonomous_jobs_created_at", table_name="autonomous_jobs")
    op.drop_index("ix_autonomous_jobs_state", table_name="autonomous_jobs")
    op.drop_index("ix_autonomous_jobs_owner_id", table_name="autonomous_jobs")
    op.drop_table("autonomous_jobs")
