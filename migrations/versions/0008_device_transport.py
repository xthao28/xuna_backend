"""Device transport status table (CLD-40, CLD-41, E14).

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "device_transport_statuses",
        sa.Column(
            "device_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("devices.device_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "pet_id",
            sa.String(32),
            sa.ForeignKey("pets.pet_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ip_address", sa.String(64), nullable=True),
        sa.Column("client_id", sa.String(128), nullable=True),
    )
    op.create_index(
        "ix_device_transport_statuses_pet_id",
        "device_transport_statuses",
        ["pet_id"],
    )
    op.create_index(
        "ix_device_transport_statuses_status",
        "device_transport_statuses",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_device_transport_statuses_status",
        table_name="device_transport_statuses",
    )
    op.drop_index(
        "ix_device_transport_statuses_pet_id",
        table_name="device_transport_statuses",
    )
    op.drop_table("device_transport_statuses")
