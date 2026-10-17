"""Ports for owner authentication. The OIDC vendor is not chosen (ADR-0009)."""

from typing import Protocol

from xuna_backend.modules.accounts.domain.owner import Owner, OwnerClaims


class InvalidTokenError(Exception):
    """Missing, malformed, expired or wrongly issued token."""


class AuthUnavailableError(Exception):
    """Auth is not configured or the key set cannot be fetched."""


class TokenVerifier(Protocol):
    async def verify(self, token: str) -> OwnerClaims: ...


class OwnerRepository(Protocol):
    async def get_or_create(self, claims: OwnerClaims) -> Owner: ...
