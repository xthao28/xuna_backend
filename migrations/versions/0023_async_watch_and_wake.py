"""Async Watch & Wake persistence tables (`job_watches`, `wake_inbox_items`).

Revision ID: 0023
Revises: 0022
Create Date: 2026-11-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    jsonb_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

    op.create_table(
        "job_watches",
        sa.Column("watch_id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("autonomous_jobs.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
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
        sa.Column("mode", sa.String(24), nullable=False),
        sa.Column(
            "signals",
            jsonb_type,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "target_spec",
            jsonb_type,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("done_criteria", sa.String(240), nullable=False),
        sa.Column(
            "continuation_mode",
            sa.String(24),
            nullable=False,
            server_default="report_and_ask",
        ),
        sa.Column("quiescence_seconds", sa.Integer(), nullable=False, server_default="8"),
        sa.Column("max_wait_seconds", sa.Integer(), nullable=False, server_default="300"),
        sa.Column("max_wake_checks", sa.Integer(), nullable=False, server_default="8"),
        sa.Column("wake_checks_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("last_event_kind", sa.String(24), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_job_watches_job_id", "job_watches", ["job_id"])
    op.create_index("ix_job_watches_owner_id", "job_watches", ["owner_id"])
    op.create_index("ix_job_watches_host_id", "job_watches", ["host_id"])
    op.create_index("ix_job_watches_status", "job_watches", ["status"])
    op.create_index("ix_job_watches_expires_at", "job_watches", ["expires_at"])

    op.create_table(
        "wake_inbox_items",
        sa.Column("item_id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pet_id", sa.String(64), nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("autonomous_jobs.job_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "host_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("hosts.host_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("title", sa.String(120), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column(
            "suggested_actions",
            jsonb_type,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("gesture_cue", sa.String(24), nullable=True),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_wake_inbox_items_owner_id", "wake_inbox_items", ["owner_id"])
    op.create_index("ix_wake_inbox_items_read_at", "wake_inbox_items", ["read_at"])
    op.create_index("ix_wake_inbox_items_created_at", "wake_inbox_items", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_wake_inbox_items_created_at", table_name="wake_inbox_items")
    op.drop_index("ix_wake_inbox_items_read_at", table_name="wake_inbox_items")
    op.drop_index("ix_wake_inbox_items_owner_id", table_name="wake_inbox_items")
    op.drop_table("wake_inbox_items")

    op.drop_index("ix_job_watches_expires_at", table_name="job_watches")
    op.drop_index("ix_job_watches_status", table_name="job_watches")
    op.drop_index("ix_job_watches_host_id", table_name="job_watches")
    op.drop_index("ix_job_watches_owner_id", table_name="job_watches")
    op.drop_index("ix_job_watches_job_id", table_name="job_watches")
    op.drop_table("job_watches")
