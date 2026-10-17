"""Composition root: builds the FastAPI app. Run: `uvicorn xuna_backend.main:app`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from xuna_backend import __version__
from xuna_backend.core.config import Settings
from xuna_backend.core.db import Database
from xuna_backend.core.errors import install_error_handlers
from xuna_backend.core.observability import configure_logging, install_request_context
from xuna_backend.modules.accounts.api.routes import router as accounts_router
from xuna_backend.modules.accounts.application.ports import TokenVerifier
from xuna_backend.modules.accounts.infrastructure.oidc import (
    OidcTokenVerifier,
    UnconfiguredTokenVerifier,
    jwks_resolver,
)
from xuna_backend.modules.accounts.infrastructure.repository import SqlOwnerRepository
from xuna_backend.modules.health.api.routes import router as health_router
from xuna_backend.modules.identity.api.routes import router as identity_router
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier
from xuna_backend.modules.identity.infrastructure.repository import SqlIdentityRepository


def token_verifier(settings: Settings) -> TokenVerifier:
    if not settings.oidc_issuer:
        return UnconfiguredTokenVerifier()
    jwks_url = settings.oidc_jwks_url or (
        settings.oidc_issuer.rstrip("/") + "/.well-known/jwks.json"
    )
    return OidcTokenVerifier(settings.oidc_issuer, settings.oidc_audience, jwks_resolver(jwks_url))


def create_app(
    settings: Settings | None = None, *, verifier: TokenVerifier | None = None
) -> FastAPI:
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
    # Ports for routers (api never imports infrastructure).
    app.state.token_verifier = verifier or token_verifier(resolved)
    app.state.owner_repository = SqlOwnerRepository
    app.state.identity_repository = SqlIdentityRepository
    app.state.signature_verifier = P256SignatureVerifier()
    install_error_handlers(app)
    install_request_context(app)
    app.include_router(health_router)
    app.include_router(accounts_router)
    app.include_router(identity_router)
    return app


def __getattr__(name: str) -> FastAPI:
    # Lazy `app` for uvicorn so importing this module never reads the env.
    if name == "app":
        return create_app()
    raise AttributeError(name)
