"""Composition root: builds the FastAPI app. Run: `uvicorn xuna_backend.main:app`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from xuna_backend import __version__
from xuna_backend.core.config import Settings
from xuna_backend.core.db import Database
from xuna_backend.core.errors import install_error_handlers
from xuna_backend.core.observability import configure_logging, install_request_context
from xuna_backend.modules.health.api.routes import router as health_router


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.database = Database(resolved.database_url)
        try:
            yield
        finally:
            await app.state.database.dispose()

    app = FastAPI(title="XUNA Cloud", version=__version__, lifespan=lifespan)
    app.state.settings = resolved
    install_error_handlers(app)
    install_request_context(app)
    app.include_router(health_router)
    return app


def __getattr__(name: str) -> FastAPI:
    # Lazy `app` for uvicorn so importing this module never reads the env.
    if name == "app":
        return create_app()
    raise AttributeError(name)
