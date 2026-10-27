"""Knowledge domains, suggestions, summaries, and candidate knowledge fields (KD-3).

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Knowledge domains table
    op.create_table(
        "knowledge_domains",
        sa.Column("domain_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "pet_id",
            sa.String(32),
            sa.ForeignKey("pets.pet_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("device_slot", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("origin", sa.String(16), nullable=False, server_default="user"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("embedding", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        "ix_knowledge_domains_pet_status",
        "knowledge_domains",
        ["pet_id", "status"],
    )
    op.create_index(
        "ix_knowledge_domains_pet_slot",
        "knowledge_domains",
        ["pet_id", "device_slot"],
    )

    # 2. Knowledge suggestions table
    op.create_table(
        "knowledge_suggestions",
        sa.Column("suggestion_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("owners.owner_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "pet_id",
            sa.String(32),
            sa.ForeignKey("pets.pet_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("proposed_name", sa.String(64), nullable=False),
        sa.Column("proposed_description", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "member_candidate_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "target_domain_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("knowledge_domains.domain_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_knowledge_suggestions_pet_status",
        "knowledge_suggestions",
        ["pet_id", "status"],
    )

    # 3. Knowledge summaries table
    op.create_table(
        "knowledge_summaries",
        sa.Column(
            "domain_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("knowledge_domains.domain_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "source_candidate_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stale", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )

    # 4. Add knowledge columns to memory_candidates table
    op.add_column(
        "memory_candidates",
        sa.Column(
            "domain_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("knowledge_domains.domain_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "memory_candidates",
        sa.Column("relation", sa.String(16), nullable=True),
    )
    op.add_column(
        "memory_candidates",
        sa.Column(
            "related_candidate_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("memory_candidates.candidate_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "memory_candidates",
        sa.Column("fact_key", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "memory_candidates",
        sa.Column("embedding", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        "ix_memory_candidates_domain_id",
        "memory_candidates",
        ["domain_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_candidates_domain_id", table_name="memory_candidates")
    op.drop_column("memory_candidates", "embedding")
    op.drop_column("memory_candidates", "fact_key")
    op.drop_column("memory_candidates", "related_candidate_id")
    op.drop_column("memory_candidates", "relation")
    op.drop_column("memory_candidates", "domain_id")
    op.drop_table("knowledge_summaries")
    op.drop_index(
        "ix_knowledge_suggestions_pet_status",
        table_name="knowledge_suggestions",
    )
    op.drop_table("knowledge_suggestions")
    op.drop_index("ix_knowledge_domains_pet_slot", table_name="knowledge_domains")
    op.drop_index("ix_knowledge_domains_pet_status", table_name="knowledge_domains")
    op.drop_table("knowledge_domains")
