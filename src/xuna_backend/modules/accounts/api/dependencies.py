"""FastAPI dependencies shared by routers: DB session and current owner.

Ports are placed on `app.state` by the composition root (`main.py`); this
module never imports infrastructure.
"""

from collections.abc import Callable
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import Depends, Request

from xuna_backend.core.db import DbSession
from xuna_backend.core.errors import AppError
from xuna_backend.modules.accounts.application.authenticate import authenticate
from xuna_backend.modules.accounts.application.ports import (
    AuthUnavailableError,
    InvalidTokenError,
    OwnerRepository,
    TokenVerifier,
)
from xuna_backend.modules.accounts.domain.owner import Owner

Session = DbSession


def _bearer(request: Request) -> str:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AppError("unauthorized", HTTPStatus.UNAUTHORIZED)
    return token


async def current_owner(request: Request, session: Session) -> Owner:
    verifier: TokenVerifier = request.app.state.token_verifier
    owners_for: Callable[[Any], OwnerRepository] = request.app.state.owner_repository
    try:
        owner = await authenticate(_bearer(request), verifier, owners_for(session))
    except InvalidTokenError as error:
        raise AppError("unauthorized", HTTPStatus.UNAUTHORIZED) from error
    except AuthUnavailableError as error:
        raise AppError("auth_unavailable", HTTPStatus.SERVICE_UNAVAILABLE) from error
    await session.commit()
    return owner


CurrentOwner = Annotated[Owner, Depends(current_owner)]


def state(request: Request, name: str) -> Any:
    return getattr(request.app.state, name)
