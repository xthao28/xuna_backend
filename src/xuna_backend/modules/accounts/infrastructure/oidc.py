"""Generic OIDC access-token verifier (RS256/ES256 via JWKS).

Works with any managed provider; only issuer, audience and JWKS URL are
configured. JWKS fetching is blocking in PyJWT, so it runs in a thread with a
timeout; keys are cached by PyJWKClient.
"""

import asyncio
from collections.abc import Callable
from typing import Any

import jwt

from xuna_backend.modules.accounts.application.ports import (
    AuthUnavailableError,
    InvalidTokenError,
)
from xuna_backend.modules.accounts.domain.owner import OwnerClaims

KeyResolver = Callable[[str], Any]
ALGORITHMS = ["RS256", "ES256"]


def jwks_resolver(jwks_url: str, timeout_s: float = 5.0) -> KeyResolver:
    client = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=3600, timeout=int(timeout_s))

    def resolve(token: str) -> Any:
        return client.get_signing_key_from_jwt(token).key

    return resolve


class OidcTokenVerifier:
    def __init__(self, issuer: str, audience: str, resolve_key: KeyResolver) -> None:
        if not issuer or not audience:
            raise ValueError("issuer and audience are required")
        self._issuer = issuer
        self._audience = audience
        self._resolve_key = resolve_key

    async def verify(self, token: str) -> OwnerClaims:
        try:
            key = await asyncio.wait_for(asyncio.to_thread(self._resolve_key, token), timeout=6)
        except (jwt.PyJWKClientConnectionError, TimeoutError) as error:
            raise AuthUnavailableError("jwks unavailable") from error
        except jwt.PyJWTError as error:
            raise InvalidTokenError("unknown signing key") from error
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=ALGORITHMS,
                audience=self._audience,
                issuer=self._issuer,
                leeway=30,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
            return OwnerClaims(issuer=claims["iss"], subject=str(claims["sub"]))
        except (jwt.PyJWTError, ValueError) as error:
            raise InvalidTokenError("invalid token") from error


class UnconfiguredTokenVerifier:
    """Used when XUNA_OIDC_ISSUER is empty: every request gets 503."""

    async def verify(self, token: str) -> OwnerClaims:
        raise AuthUnavailableError("owner auth is not configured")
