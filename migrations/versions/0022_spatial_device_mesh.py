"""Spatial Device Mesh: host spatial fields, per-host CU lanes, habits & defaults.

Revision ID: 0022
Revises: 0021
Create Date: 2026-11-20
"""

import unicodedata
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _fold_text(raw: str) -> str:
    lowered = raw.strip().lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(stripped.split())


def upgrade() -> None:
    # 1. hosts spatial columns + deterministic backfill + partial unique active alias index
    op.add_column(
        "hosts",
        sa.Column("host_kind", sa.String(16), nullable=False, server_default="desktop"),
    )
    op.add_column("hosts", sa.Column("place", sa.String(32), nullable=True))
    op.add_column(
        "hosts",
        sa.Column(
            "nicknames",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "hosts",
        sa.Column("alias_folded", sa.String(32), nullable=False, server_default=""),
    )
    op.add_column("hosts", sa.Column("wol_mac", sa.String(17), nullable=True))

    op.execute(
        "UPDATE hosts "
        "SET host_kind = CASE WHEN platform IN ('android', 'ios') THEN 'mobile' ELSE 'desktop' END"
    )

    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT host_id, owner_id, alias, status "
            "FROM hosts "
            "ORDER BY owner_id ASC, created_at ASC, host_id ASC"
        )
    ).fetchall()
    used_active: dict[object, set[str]] = {}
    for host_id, owner_id, alias, status in rows:
        base = _fold_text(str(alias))[:32] or "host"
        candidate = base
        if status == "active":
            used = used_active.setdefault(owner_id, set())
            n = 2
            while candidate in used:
                suffix = f"_{n}"
                candidate = f"{base[: 32 - len(suffix)]}{suffix}"
                n += 1
            used.add(candidate)
        bind.execute(
            sa.text("UPDATE hosts SET alias_folded = :folded WHERE host_id = :host_id"),
            {"folded": candidate, "host_id": host_id},
        )

    op.create_index(
        "uq_hosts_owner_alias_folded_active",
        "hosts",
        ["owner_id", "alias_folded"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    # 2. cu_sessions per-(owner_id, host_id) execution lanes
    op.drop_index("uq_cu_sessions_owner_active", table_name="cu_sessions")
    op.create_index(
        "uq_cu_sessions_owner_host_active",
        "cu_sessions",
        ["owner_id", "host_id"],
        unique=True,
        postgresql_where=sa.text("active"),
    )
    op.create_index("ix_cu_sessions_owner_host", "cu_sessions", ["owner_id", "host_id"])

    # 3. host_routing_habits
    op.create_table(
        "host_routing_habits",
        sa.Column("habit_id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pet_id", sa.String(64), nullable=False),
        sa.Column("context_key", sa.String(128), nullable=False),
        sa.Column(
            "host_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("hosts.host_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("positive_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("negative_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("decay_gamma", sa.Float(), nullable=False, server_default="0.85"),
        sa.Column("last_reason_code", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "owner_id",
            "pet_id",
            "context_key",
            "host_id",
            name="uq_host_routing_habits_key",
        ),
    )
    op.create_index("ix_host_routing_habits_owner_id", "host_routing_habits", ["owner_id"])

    # 4. host_defaults
    op.create_table(
        "host_defaults",
        sa.Column("default_id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pet_id", sa.String(64), nullable=False),
        sa.Column("scope_key", sa.String(64), nullable=False),
        sa.Column(
            "host_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("hosts.host_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source", sa.String(24), nullable=False, server_default="confirmed"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "owner_id",
            "pet_id",
            "scope_key",
            name="uq_host_defaults_owner_pet_scope",
        ),
    )
    op.create_index("ix_host_defaults_owner_id", "host_defaults", ["owner_id"])


def downgrade() -> None:
    op.drop_index("ix_host_defaults_owner_id", table_name="host_defaults")
    op.drop_table("host_defaults")

    op.drop_index("ix_host_routing_habits_owner_id", table_name="host_routing_habits")
    op.drop_table("host_routing_habits")

    op.drop_index("ix_cu_sessions_owner_host", table_name="cu_sessions")
    op.drop_index("uq_cu_sessions_owner_host_active", table_name="cu_sessions")
    op.create_index(
        "uq_cu_sessions_owner_active",
        "cu_sessions",
        ["owner_id"],
        unique=True,
        postgresql_where=sa.text("active"),
    )

    op.drop_index("uq_hosts_owner_alias_folded_active", table_name="hosts")
    op.drop_column("hosts", "wol_mac")
    op.drop_column("hosts", "alias_folded")
    op.drop_column("hosts", "nicknames")
    op.drop_column("hosts", "place")
    op.drop_column("hosts", "host_kind")
