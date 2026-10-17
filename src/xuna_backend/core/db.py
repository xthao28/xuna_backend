"""Async SQLAlchemy 2 engine/session and the declarative base.

All tables use the naming convention below so Alembic autogenerate produces
stable constraint names. Models live in each module's `infrastructure/`.
"""

from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import MetaData, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from xuna_backend.core.errors import AppError

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Database:
    """Owns one engine per process; created in the app lifespan."""

    def __init__(self, url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(
            url,
            pool_pre_ping=True,
            # Every session in UTC: API times are RFC 3339 UTC (contract §1).
            connect_args={"options": "-c timezone=UTC"},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions() as session:
            yield session

    async def ping(self) -> None:
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def dispose(self) -> None:
        await self.engine.dispose()


async def request_session(request: Request) -> AsyncIterator[AsyncSession]:
    """One session per request; routers commit through `commit_or_conflict`."""
    database: Database = request.app.state.database
    async with database.sessions() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(request_session)]


async def commit_or_conflict(session: AsyncSession) -> None:
    """Commit; a unique-constraint race becomes `409 conflict_retry`."""
    try:
        await session.commit()
    except IntegrityError as error:
        await session.rollback()
        raise AppError("conflict_retry", HTTPStatus.CONFLICT) from error
