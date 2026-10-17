import asyncio
import time

import pytest

from tests.helpers import ISSUER, LocalIssuer
from xuna_backend.modules.accounts.application.ports import (
    AuthUnavailableError,
    InvalidTokenError,
)
from xuna_backend.modules.accounts.infrastructure.oidc import UnconfiguredTokenVerifier


def test_valid_token_yields_issuer_and_subject() -> None:
    issuer = LocalIssuer()
    claims = asyncio.run(issuer.verifier().verify(issuer.token("abc")))
    assert (claims.issuer, claims.subject) == (ISSUER, "abc")


@pytest.mark.parametrize(
    "overrides",
    [
        {"iss": "https://evil"},
        {"aud": "other"},
        {"exp": int(time.time()) - 3600},
        {"sub": ""},
    ],
)
def test_rejects_wrong_claims(overrides: dict[str, object]) -> None:
    issuer = LocalIssuer()
    with pytest.raises(InvalidTokenError):
        asyncio.run(issuer.verifier().verify(issuer.token(**overrides)))  # type: ignore[arg-type]


def test_rejects_other_signer_and_garbage() -> None:
    issuer = LocalIssuer()
    with pytest.raises(InvalidTokenError):
        asyncio.run(issuer.verifier().verify(LocalIssuer().token()))
    with pytest.raises(InvalidTokenError):
        asyncio.run(issuer.verifier().verify("not-a-jwt"))


def test_unconfigured_auth_is_unavailable() -> None:
    with pytest.raises(AuthUnavailableError):
        asyncio.run(UnconfiguredTokenVerifier().verify("x"))
