"""Consent class U3V conversation_voice (owner D16, 2026-10-22).

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-22

Adding a column with a constant default rewrites no rows, so the append-only
trigger on consent_records is not involved; existing history reads as "voice off".
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "consent_records",
        sa.Column("conversation_voice", sa.Boolean(), server_default="false", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("consent_records", "conversation_voice")
