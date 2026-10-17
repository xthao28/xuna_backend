"""`xuna.export.v1` integrity check (gate c format, cloud-gate sprint record).

`sha256` is SHA-256 of the UTF-8 compact JSON of `payload` in stored key
order. This is the server-side mirror of mobile `ExportDocumentCodec`; keep
both in lockstep and update the shared golden fixture together.
"""

import hashlib
import json
from typing import Any

FORMAT = "xuna.export.v1"


class ExportDocumentError(ValueError):
    """The document is not a valid, untampered `xuna.export.v1` envelope."""


def payload_hash(payload: dict[str, Any]) -> str:
    compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def verified_payload(document: str | bytes) -> dict[str, Any]:
    """Returns the payload of an intact envelope; raises ExportDocumentError."""
    try:
        envelope = json.loads(document)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ExportDocumentError("not JSON") from error
    if not isinstance(envelope, dict) or set(envelope) != {"format", "sha256", "payload"}:
        raise ExportDocumentError("unexpected envelope keys")
    if envelope["format"] != FORMAT:
        raise ExportDocumentError("unsupported format")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise ExportDocumentError("payload must be an object")
    if envelope["sha256"] != payload_hash(payload):
        raise ExportDocumentError("hash mismatch")
    return payload
