import asyncio

import pytest

from tests.helpers import LocalIssuer
from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import token_verifier
from xuna_backend.modules.accounts.application.ports import InvalidTokenError
from xuna_backend.modules.accounts.infrastructure.oidc import (
    FIREBASE_JWKS_URL,
    OidcTokenVerifier,
    UnconfiguredTokenVerifier,
    firebase_settings,
)


def test_firebase_preset_values() -> None:
    issuer, audience, jwks = firebase_settings("xuna-dev")
    assert issuer == "https://securetoken.google.com/xuna-dev"
    assert audience == "xuna-dev"
    assert jwks == FIREBASE_JWKS_URL
    with pytest.raises(ValueError):
        firebase_settings("bad/project")


def test_settings_select_firebase_without_network() -> None:
    base = Settings(environment=Environment.TEST, database_url="x")
    assert isinstance(token_verifier(base), UnconfiguredTokenVerifier)
    firebase = Settings(
        environment=Environment.TEST, database_url="x", firebase_project_id="xuna-dev"
    )
    assert isinstance(token_verifier(firebase), OidcTokenVerifier)


def test_firebase_shaped_token_is_verified() -> None:
    signer = LocalIssuer()
    issuer, audience, _ = firebase_settings("xuna-dev")
    verifier = OidcTokenVerifier(issuer, audience, lambda _t: signer.key.public_key())
    token = signer.token("firebase-uid-123", iss=issuer, aud=audience, auth_time=1)
    claims = asyncio.run(verifier.verify(token))
    assert claims.subject == "firebase-uid-123"
    with pytest.raises(InvalidTokenError):
        asyncio.run(verifier.verify(signer.token("u", iss=issuer, aud="other-project")))


def test_settings_repr_hides_secrets() -> None:
    settings = Settings(
        environment=Environment.TEST,
        database_url="postgresql://u:secret@h/db",
        text_api_key="sk-secret",
    )
    assert "secret" not in repr(settings)
