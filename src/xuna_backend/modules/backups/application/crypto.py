"""Cryptographic primitives for backup encryption and decryption (CLD-20).

Uses standard AES-256-GCM authenticated encryption.
"""

import base64
import hashlib
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from xuna_backend.modules.backups.domain.entities import EncryptedData

NONCE_SIZE = 12
TAG_SIZE = 16


def derive_key(secret: str | bytes) -> bytes:
    """Derives a 256-bit AES key from a passphrase or secret."""
    raw = secret.encode("utf-8") if isinstance(secret, str) else secret
    return hashlib.sha256(raw).digest()


def encrypt_payload(data: bytes, key: bytes, key_id: str = "primary") -> EncryptedData:
    """Encrypts bytes using AES-256-GCM into an EncryptedData domain entity."""
    if len(key) != 32:
        raise ValueError("AES-256 requires a 32-byte key")
    nonce = os.urandom(NONCE_SIZE)
    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(nonce, data, None)
    ciphertext = ciphertext_and_tag[:-TAG_SIZE]
    tag = ciphertext_and_tag[-TAG_SIZE:]
    return EncryptedData(
        algorithm="aes-256-gcm",
        key_id=key_id,
        nonce=base64.b64encode(nonce).decode("ascii"),
        ciphertext=base64.b64encode(ciphertext).decode("ascii"),
        tag=base64.b64encode(tag).decode("ascii"),
    )


def decrypt_payload(encrypted: EncryptedData, key: bytes) -> bytes:
    """Decrypts EncryptedData using AES-256-GCM. Raises ValueError on corruption."""
    if len(key) != 32:
        raise ValueError("AES-256 requires a 32-byte key")
    if encrypted.algorithm != "aes-256-gcm":
        raise ValueError(f"unsupported algorithm: {encrypted.algorithm}")
    try:
        nonce = base64.b64decode(encrypted.nonce)
        ciphertext = base64.b64decode(encrypted.ciphertext)
        tag = base64.b64decode(encrypted.tag)
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext + tag, None)
    except Exception as error:
        raise ValueError("decryption failed or invalid tag") from error
