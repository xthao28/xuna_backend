"""Shared test helpers: local OIDC issuer and a simulated device key."""

import base64
import time
import uuid
from typing import Any

import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from xuna_backend.modules.accounts.infrastructure.oidc import OidcTokenVerifier

ISSUER = "https://auth.test.xuna"
AUDIENCE = "xuna-api"


class LocalIssuer:
    def __init__(self) -> None:
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def verifier(self) -> OidcTokenVerifier:
        public = self.key.public_key()
        return OidcTokenVerifier(ISSUER, AUDIENCE, lambda _token: public)

    def token(self, subject: str = "owner-1", **overrides: Any) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": subject,
            "iat": now,
            "exp": now + 300,
        }
        claims.update(overrides)
        return jwt.encode(claims, self.key, algorithm="RS256")


class SimulatedDevice:
    """Stands in for firmware Identity V2 (ECDSA P-256)."""

    def __init__(self) -> None:
        self.key = ec.generate_private_key(ec.SECP256R1())

    @property
    def public_key(self) -> bytes:
        return self.key.public_key().public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )

    def sign(self, message: bytes) -> bytes:
        r, s = decode_dss_signature(self.key.sign(message, ec.ECDSA(hashes.SHA256())))
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def idem() -> dict[str, str]:
    return {"Idempotency-Key": str(uuid.uuid4())}
