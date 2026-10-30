"""Host Registry tables: hosts, host_enrollments, host_manifests (UDC-3 S3.2).

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "hosts",
        sa.Column("host_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("alias", sa.String(24), nullable=False),
        sa.Column("platform", sa.String(16), nullable=False),
        sa.Column("hostname", sa.String(32), nullable=False),
        sa.Column("os_version", sa.String(64), nullable=False),
        sa.Column("agent_version", sa.String(32), nullable=False),
        sa.Column("public_key", sa.LargeBinary(65), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "host_enrollments",
        sa.Column("pending_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "code_sha256",
            sa.String(64),
            nullable=False,
            unique=True,
            index=True,
        ),
        sa.Column("public_key", sa.LargeBinary(65), nullable=False),
        sa.Column("key_fingerprint", sa.String(32), nullable=False),
        sa.Column("sas", sa.String(6), nullable=False),
        sa.Column("platform", sa.String(16), nullable=False),
        sa.Column("os_version", sa.String(64), nullable=False),
        sa.Column("agent_version", sa.String(32), nullable=False),
        sa.Column("hostname", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("alias", sa.String(24), nullable=True),
    )

    op.create_table(
        "host_manifests",
        sa.Column(
            "host_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("hosts.host_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "tools_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("host_manifests")
    op.drop_table("host_enrollments")
    op.drop_table("hosts")
