"""UDC-8 consent classes U5 (host_screen) and U6 (host_notifications).

Revision ID: 0017
Revises: 0016
Create Date: 2026-10-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "consent_records",
        sa.Column(
            "host_screen",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "consent_records",
        sa.Column(
            "host_notifications",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("consent_records", "host_notifications")
    op.drop_column("consent_records", "host_screen")

