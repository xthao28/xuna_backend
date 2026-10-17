"""Use case: verify an owner-supplied export file. Pure; stores nothing."""

from pydantic import ValidationError

from xuna_backend.modules.exports.application.schemas import ExportPayloadV1
from xuna_backend.modules.exports.domain.export_document import (
    ExportDocumentError,
    verified_payload,
)


def verify_export(document: str | bytes) -> ExportPayloadV1:
    """Raises ExportDocumentError for a bad envelope, hash or payload shape."""
    payload = verified_payload(document)
    try:
        return ExportPayloadV1.model_validate(payload)
    except ValidationError as error:
        raise ExportDocumentError("payload does not match xuna.export.v1") from error
