"""Add `approval_version` and `effect_sha256` to `action_approvals` (`ADR-0019` §4 / `W4-M4`).

Revision ID: 0024
Revises: 0023
Create Date: 2026-11-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: str | None = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "action_approvals",
        sa.Column(
            "approval_version",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )
    op.add_column(
        "action_approvals",
        sa.Column("effect_sha256", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("action_approvals", "effect_sha256")
    op.drop_column("action_approvals", "approval_version")
