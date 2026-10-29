"""Knowledge expert briefs and curated pack fields (KX-3).

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_domains",
        sa.Column("pack_id", sa.String(32), nullable=True),
    )
    op.add_column(
        "knowledge_domains",
        sa.Column("expert_brief", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "knowledge_domains",
        sa.Column("safety_tier", sa.String(16), nullable=False, server_default="general"),
    )


def downgrade() -> None:
    op.drop_column("knowledge_domains", "safety_tier")
    op.drop_column("knowledge_domains", "expert_brief")
    op.drop_column("knowledge_domains", "pack_id")
