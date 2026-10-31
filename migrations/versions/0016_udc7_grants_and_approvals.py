"""UDC-7 multi-host grants expiry/suspension, LAN peer metadata, and approval scopes/signatures.

Revision ID: 0016
Revises: 0015
Create Date: 2026-10-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "hosts",
        sa.Column("lan_fingerprint", sa.String(96), nullable=True),
    )
    op.add_column(
        "hosts",
        sa.Column("lan_port", sa.Integer(), nullable=True),
    )
    op.add_column(
        "host_grants",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "host_grants",
        sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "action_approvals",
        sa.Column("scope", sa.String(24), nullable=False, server_default="once"),
    )
    op.add_column(
        "action_approvals",
        sa.Column("signature", sa.LargeBinary(64), nullable=True),
    )
    op.add_column(
        "action_approvals",
        sa.Column("signer_public_key", sa.LargeBinary(65), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("action_approvals", "signer_public_key")
    op.drop_column("action_approvals", "signature")
    op.drop_column("action_approvals", "scope")
    op.drop_column("host_grants", "suspended_at")
    op.drop_column("host_grants", "expires_at")
    op.drop_column("hosts", "lan_port")
    op.drop_column("hosts", "lan_fingerprint")

