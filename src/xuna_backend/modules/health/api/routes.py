"""Liveness and readiness probes for the hosting platform."""

from http import HTTPStatus

from fastapi import APIRouter, Request

from xuna_backend.core.db import Database
from xuna_backend.core.errors import AppError

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Process is up. Never touches dependencies."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, str]:
    """Database reachable. 503 `database_unavailable` otherwise."""
    database: Database = request.app.state.database
    try:
        await database.ping()
    except Exception as error:
        raise AppError("database_unavailable", HTTPStatus.SERVICE_UNAVAILABLE) from error
    return {"status": "ready"}
