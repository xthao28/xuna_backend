"""Golden vector for the device proof message. Firmware Identity V2 checks
the same SHA-256 in `firmware/test/test_device_proof` (golden_message_matches_backend);
change both together."""

import hashlib

import pytest

from tests.helpers import SimulatedDevice
from xuna_backend.modules.identity.domain.device_proof import ProofPurpose, proof_message
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier

PET = PetId("d945c77026fd4f4a8a899bbff4aeed86")
NONCE = bytes(range(32))


def test_golden_message_bytes() -> None:
    message = proof_message(NONCE, PET, ProofPurpose.BIND, 38, b"payload")
    assert message[:14] == b"xuna.proof.v1\x00"
    assert message[14:46] == NONCE
    assert message[46:62].hex() == PET.value
    assert message[62:67] == bytes([1, 38, 0, 0, 0])
    assert message[67:] == hashlib.sha256(b"payload").digest()
    assert len(message) == 99
    assert (
        hashlib.sha256(message).hexdigest()
        == hashlib.sha256(
            b"xuna.proof.v1\x00"
            + NONCE
            + bytes.fromhex(PET.value)
            + b"\x01\x26\x00\x00\x00"
            + hashlib.sha256(b"payload").digest()
        ).hexdigest()
    )


def test_session_purpose_golden() -> None:
    """Device-direct session proof (DC-3/DC-4); firmware test_cloud_session checks the same hash."""
    message = proof_message(NONCE, PET, ProofPurpose.SESSION, 38, b"payload")
    assert message[62] == 3
    assert (
        hashlib.sha256(message).hexdigest()
        == "448f0b9047eebce3a1cffe8f87c9f49f06ac73ff61aedd5855645906cc4b1aa4"
    )


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        proof_message(NONCE[:31], PET, ProofPurpose.BIND, 0, b"")
    with pytest.raises(ValueError):
        proof_message(NONCE, PET, ProofPurpose.BIND, 1 << 32, b"")


def test_p256_verifier_accepts_only_matching_signatures() -> None:
    device = SimulatedDevice()
    message = proof_message(NONCE, PET, ProofPurpose.BIND, 1, device.public_key)
    signature = device.sign(message)
    verifier = P256SignatureVerifier()
    assert verifier.verify(device.public_key, message, signature)
    assert not verifier.verify(device.public_key, message + b"x", signature)
    assert not verifier.verify(SimulatedDevice().public_key, message, signature)
    assert not verifier.verify(b"\x04" + bytes(64), message, signature)
    assert not verifier.verify(device.public_key, message, bytes(64))
