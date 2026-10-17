from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature


class P256SignatureVerifier:
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        try:
            key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), public_key)
            der = encode_dss_signature(
                int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big")
            )
            key.verify(der, message, ec.ECDSA(hashes.SHA256()))
        except (InvalidSignature, ValueError):
            return False
        return True
