"""Alembic environment. URL comes only from XUNA_DATABASE_URL."""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# Import every module's ORM models here so autogenerate sees them.
import xuna_backend.core.idempotency
import xuna_backend.modules.accounts.infrastructure.models
import xuna_backend.modules.audit.infrastructure.models
import xuna_backend.modules.backups.infrastructure.models
import xuna_backend.modules.consent.infrastructure.models
import xuna_backend.modules.identity.infrastructure.models
import xuna_backend.modules.knowledge.infrastructure.models
import xuna_backend.modules.memory.infrastructure.models
import xuna_backend.modules.suggestions.infrastructure.models
import xuna_backend.modules.transport.infrastructure.models  # noqa: F401
import xuna_backend.modules.hosts.infrastructure.models  # noqa: F401
from xuna_backend.core.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def _url() -> str:
    url = os.environ.get("XUNA_DATABASE_URL", "")
    if not url:
        raise RuntimeError("XUNA_DATABASE_URL is required")
    return url


def run_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
