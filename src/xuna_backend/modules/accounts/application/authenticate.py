"""Use case: resolve a bearer token to an owner, provisioning on first use."""

from xuna_backend.modules.accounts.application.ports import (
    OwnerRepository,
    OwnerRepositoryWithStatus,
    TokenVerifier,
)
from xuna_backend.modules.accounts.domain.owner import Owner


async def authenticate(token: str, verifier: TokenVerifier, owners: OwnerRepository) -> Owner:
    """Raises InvalidTokenError or AuthUnavailableError."""
    claims = await verifier.verify(token)
    return await owners.get_or_create(claims)


async def authenticate_with_status(
    token: str, verifier: TokenVerifier, owners: OwnerRepository
) -> tuple[Owner, bool]:
    """Resolves bearer token to `(owner, created)` (True only when newly provisioned)."""
    claims = await verifier.verify(token)
    if isinstance(owners, OwnerRepositoryWithStatus):
        return await owners.get_or_create_with_status(claims)
    return (await owners.get_or_create(claims), True)
