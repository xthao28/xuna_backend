"""Device proof message (contract draft §4, threat model §4).

The device signs SHA-256 of these bytes with its ECDSA P-256 key:

    b"xuna.proof.v1" 0x00 | nonce[32] | petId[16 raw] | purpose u8
    | generation u32 LE | payloadSha256[32]

Firmware must build byte-identical messages; keep the golden vector in
`tests/modules/identity/test_device_proof.py` in sync with firmware tests.
"""

import hashlib
import struct
from enum import IntEnum

from xuna_backend.modules.identity.domain.pet_id import PetId

DOMAIN = b"xuna.proof.v1\x00"
NONCE_SIZE = 32
PUBLIC_KEY_SIZE = 65  # uncompressed SEC1 P-256 point
SIGNATURE_SIZE = 64  # r || s


class ProofPurpose(IntEnum):
    BIND = 1
    PROJECTION = 2
    SESSION = 3  # device-direct connection (DC-3); payload = the device's own public key
    APPROVE = 4  # R3 pet co-signing with matching 6-digit code (ADR-0020 F3-M1/M2)
    STOP = 5  # cryptographically signed emergency stop (ADR-0020 F3-M3)


COSIGN_CODE_DOMAIN = b"xuna.cosign.code.v1\x00"
STOP_PROOF_PAYLOAD = b"xuna.estop.v1"


def cosign_short_code(nonce: bytes, target_sha256: bytes) -> str:
    """Deterministic 6-digit matching code shown on both phone and XUNA pet screen."""
    if len(nonce) != NONCE_SIZE:
        raise ValueError("nonce must be 32 bytes")
    if len(target_sha256) != 32:
        raise ValueError("target_sha256 must be 32 bytes")
    digest = hashlib.sha256(COSIGN_CODE_DOMAIN + nonce + target_sha256).digest()
    num = int.from_bytes(digest[:4], "big") % 1_000_000
    return f"{num:06d}"


def approve_proof_payload(target_sha256: bytes, short_code: str) -> bytes:
    """Canonical 38-byte payload signed for `ProofPurpose.APPROVE` (ADR-0020)."""
    if len(target_sha256) != 32:
        raise ValueError("target_sha256 must be 32 bytes")
    if len(short_code) != 6 or not short_code.isascii() or not short_code.isdigit():
        raise ValueError("short_code must be 6 ASCII digits")
    return target_sha256 + short_code.encode("ascii")


def stop_proof_payload() -> bytes:
    """Canonical payload signed for `ProofPurpose.STOP` (ADR-0020 F3-M3)."""
    return STOP_PROOF_PAYLOAD


def proof_message(
    nonce: bytes,
    pet_id: PetId,
    purpose: ProofPurpose,
    generation: int,
    payload: bytes,
) -> bytes:
    if len(nonce) != NONCE_SIZE:
        raise ValueError("nonce must be 32 bytes")
    if not 0 <= generation <= 0xFFFFFFFF:
        raise ValueError("generation must be a u32")
    return (
        DOMAIN
        + nonce
        + bytes.fromhex(pet_id.value)
        + struct.pack("<BI", purpose, generation)
        + hashlib.sha256(payload).digest()
    )


def validate_public_key(public_key: bytes) -> None:
    if len(public_key) != PUBLIC_KEY_SIZE or public_key[0] != 0x04:
        raise ValueError("public key must be an uncompressed P-256 point")
