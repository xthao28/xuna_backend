"""Device-direct session status: firmware and connected-at (DC-3, ADR-0010).

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("device_transport_statuses", sa.Column("firmware", sa.String(32), nullable=True))
    op.add_column(
        "device_transport_statuses",
        sa.Column("connected_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("device_transport_statuses", "connected_at")
    op.drop_column("device_transport_statuses", "firmware")
