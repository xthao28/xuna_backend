"""Unit tests for backup encryption / decryption (CLD-20)."""

import pytest

from xuna_backend.modules.backups.application.crypto import (
    decrypt_payload,
    derive_key,
    encrypt_payload,
)
from xuna_backend.modules.backups.domain.entities import EncryptedData


def test_derive_key_produces_32_bytes() -> None:
    key = derive_key("my-secret-passphrase")
    assert len(key) == 32
    assert key == derive_key("my-secret-passphrase")


def test_encrypt_decrypt_roundtrip() -> None:
    key = derive_key("test-key")
    data = b"hello xuna life snapshot"
    encrypted = encrypt_payload(data, key, key_id="v1")
    assert isinstance(encrypted, EncryptedData)
    assert encrypted.algorithm == "aes-256-gcm"
    assert encrypted.key_id == "v1"

    decrypted = decrypt_payload(encrypted, key)
    assert decrypted == data


def test_decrypt_with_wrong_key_fails() -> None:
    key1 = derive_key("key-1")
    key2 = derive_key("key-2")
    data = b"sensitive memory data"
    encrypted = encrypt_payload(data, key1)

    with pytest.raises(ValueError, match="decryption failed"):
        decrypt_payload(encrypted, key2)


def test_decrypt_corrupted_ciphertext_fails() -> None:
    key = derive_key("key-1")
    encrypted = encrypt_payload(b"secret", key)
    corrupted = EncryptedData(
        algorithm=encrypted.algorithm,
        key_id=encrypted.key_id,
        nonce=encrypted.nonce,
        ciphertext="AAAA" + encrypted.ciphertext[4:],
        tag=encrypted.tag,
    )
    with pytest.raises(ValueError, match="decryption failed"):
        decrypt_payload(corrupted, key)


def test_decrypt_corrupted_tag_fails() -> None:
    key = derive_key("key-1")
    encrypted = encrypt_payload(b"secret", key)
    corrupted = EncryptedData(
        algorithm=encrypted.algorithm,
        key_id=encrypted.key_id,
        nonce=encrypted.nonce,
        ciphertext=encrypted.ciphertext,
        tag="AAAA" + encrypted.tag[4:],
    )
    with pytest.raises(ValueError, match="decryption failed"):
        decrypt_payload(corrupted, key)
