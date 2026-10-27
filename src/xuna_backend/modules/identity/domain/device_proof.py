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
