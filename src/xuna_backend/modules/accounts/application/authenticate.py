"""Use case: resolve a bearer token to an owner, provisioning on first use."""

from xuna_backend.modules.accounts.application.ports import OwnerRepository, TokenVerifier
from xuna_backend.modules.accounts.domain.owner import Owner


async def authenticate(token: str, verifier: TokenVerifier, owners: OwnerRepository) -> Owner:
    """Raises InvalidTokenError or AuthUnavailableError."""
    claims = await verifier.verify(token)
    return await owners.get_or_create(claims)
