"""Cryptographic proof, SAS, key fingerprint and canonical JSON (ADR-0013, XTP v1 §3).

Stdlib only. Matches `desktop/packages/xuna_xtp` and `tests/fixtures/xtp/golden_vectors.json`
byte-for-byte.
"""

import base64
import hashlib
import json
from enum import IntEnum

from xuna_backend.modules.hosts.domain.entities import HostId

DOMAIN = b"xuna.host.proof.v1\x00"
SAS_DOMAIN = b"xuna.host.sas.v1"
NONCE_SIZE = 32
ENROLLMENT_CODE_SIZE = 16
PUBLIC_KEY_SIZE = 65
SIGNATURE_SIZE = 64
PROOF_MESSAGE_SIZE = 100


class HostProofPurpose(IntEnum):
    ENROLL = 1
    SESSION = 2


def validate_public_key(public_key: bytes) -> None:
    if len(public_key) != PUBLIC_KEY_SIZE or public_key[0] != 0x04:
        raise ValueError("public key must be a 65-byte uncompressed P-256 point")


def host_proof_message(
    nonce: bytes,
    purpose: HostProofPurpose,
    payload: bytes,
    host_id: HostId | None = None,
) -> bytes:
    """Builds the 100-byte host proof message signed with ECDSA P-256 over SHA-256."""
    if len(nonce) != NONCE_SIZE:
        raise ValueError("nonce must be 32 bytes")
    if purpose == HostProofPurpose.ENROLL:
        if len(payload) != ENROLLMENT_CODE_SIZE:
            raise ValueError("enroll proof payload must be 16-byte enrollment code")
        if host_id is not None:
            raise ValueError("enroll proof host_id must be None (16 zero bytes on wire)")
        host_id_bytes = bytes(16)
    elif purpose == HostProofPurpose.SESSION:
        validate_public_key(payload)
        if host_id is None:
            raise ValueError("session proof requires host_id")
        host_id_bytes = host_id.bytes
    else:
        raise ValueError(f"unsupported proof purpose: {purpose!r}")

    msg = DOMAIN + nonce + host_id_bytes + bytes([int(purpose)]) + hashlib.sha256(payload).digest()
    if len(msg) != PROOF_MESSAGE_SIZE:
        raise ValueError("unexpected proof message length")
    return msg


def sas(enrollment_code: bytes, public_key: bytes) -> str:
    """Computes the 6-digit Short Authentication String (SAS) per ADR-0013."""
    if len(enrollment_code) != ENROLLMENT_CODE_SIZE:
        raise ValueError("enrollment_code must be 16 bytes")
    validate_public_key(public_key)
    digest = hashlib.sha256(SAS_DOMAIN + enrollment_code + public_key).digest()
    value = int.from_bytes(digest[:4], "big") % 1_000_000
    return f"{value:06d}"


def public_key_fingerprint_b64url(public_key: bytes) -> str:
    """First 16 bytes of SHA-256(public_key), unpadded base64url (22 chars)."""
    validate_public_key(public_key)
    fp = hashlib.sha256(public_key).digest()[:16]
    return base64.urlsafe_b64encode(fp).rstrip(b"=").decode("ascii")


def decode_b64url_16b(encoded: str) -> bytes:
    """Decodes an unpadded or padded base64url string of exactly 16 bytes."""
    clean = encoded.strip()
    rem = len(clean) % 4
    if rem == 2:
        clean += "=="
    elif rem == 3:
        clean += "="
    elif rem != 0:
        raise ValueError("invalid base64url length")
    try:
        raw = base64.urlsafe_b64decode(clean.encode("ascii"))
    except Exception as error:
        raise ValueError("invalid base64url encoding") from error
    if len(raw) != 16:
        raise ValueError("expected 16 decoded bytes")
    return raw


def canonical_json(value: object) -> str:
    """Encodes `value` into XTP v1 canonical JSON (keys sorted, no whitespace, no floats)."""
    _reject_floats(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def args_sha256(arguments: dict[str, object]) -> str:
    """Computes `hex(sha256(canonical_json(arguments)))`."""
    return hashlib.sha256(canonical_json(arguments).encode("utf-8")).hexdigest()


def _reject_floats(value: object) -> None:
    if isinstance(value, float):
        raise ValueError("floating-point numbers are not allowed in XTP v1 canonical JSON")
    if isinstance(value, list):
        for item in value:
            _reject_floats(item)
    elif isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("canonical JSON object keys must be strings")
            _reject_floats(v)
