"""Golden fixture produced by mobile `DeviceExportRepositoryImpl` (Dart).

Regenerate only together with a mobile export format change; see
`docs/ARCHITECTURE.md` §Contracts. The hash must verify byte-for-byte.
"""

import json
from pathlib import Path

import pytest

from xuna_backend.modules.exports.application.verify_export import verify_export
from xuna_backend.modules.exports.domain.export_document import (
    ExportDocumentError,
    payload_hash,
)

GOLDEN = Path(__file__).resolve().parents[2] / "fixtures" / "xuna_export_v1_golden.json"


def golden() -> str:
    return GOLDEN.read_text(encoding="utf-8")


def test_dart_golden_verifies_and_parses() -> None:
    export = verify_export(golden())
    assert export.identity.petId == "d945c77026fd4f4a8a899bbff4aeed86"
    assert export.identity.origin == "migrated_from_v1"
    assert export.identity.snapshotGeneration == 38
    assert export.life.feedbackCount == 5
    assert export.explanation is not None
    assert export.explanation.probabilityQ8 == 203
    assert [m.source for m in export.memories] == ["button", "text"]


def _rewrite(mutate: object) -> str:
    envelope = json.loads(golden())
    assert callable(mutate)
    mutate(envelope)
    return json.dumps(envelope, ensure_ascii=False)


def _rehash(envelope: dict[str, object]) -> None:
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    envelope["sha256"] = payload_hash(payload)


def test_tampering_breaks_the_hash() -> None:
    def bump(envelope: dict[str, object]) -> None:
        envelope["payload"]["life"]["feedbackCount"] = 6  # type: ignore[index]

    with pytest.raises(ExportDocumentError, match="hash mismatch"):
        verify_export(_rewrite(bump))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e["payload"]["identity"].update(petId="D945C77026FD4F4A8A899BBFF4AEED86"),
        lambda e: e["payload"]["identity"].update(origin="cloud"),
        lambda e: e["payload"]["life"].update(activeSeconds=-1),
        lambda e: e["payload"]["life"].update(extra=1),
        lambda e: e["payload"]["memories"].append({"context": "x"}),
        lambda e: e["payload"]["life"].update(canFeedback="yes"),
    ],
)
def test_rehashed_but_invalid_payload_is_rejected(mutate: object) -> None:
    def apply(envelope: dict[str, object]) -> None:
        assert callable(mutate)
        mutate(envelope)
        _rehash(envelope)

    with pytest.raises(ExportDocumentError, match="does not match"):
        verify_export(_rewrite(apply))


def test_reaction_and_greeting_memories_are_accepted() -> None:
    # Device-button learning (D6 revised 2026-10-19): source "reaction" and
    # context "greeting" appear in mobile exports from then on.
    def add(envelope: dict[str, object]) -> None:
        envelope["payload"]["memories"].append(  # type: ignore[index]
            {"context": "greeting", "action": "curious", "approved": False, "source": "reaction"}
        )
        _rehash(envelope)

    export = verify_export(_rewrite(add))
    assert export.memories[-1].source == "reaction"
    assert export.memories[-1].context == "greeting"


@pytest.mark.parametrize(
    "document",
    ["not json", "[]", '{"format":"xuna.export.v2","sha256":"","payload":{}}', '{"format":1}'],
)
def test_bad_envelopes(document: str) -> None:
    with pytest.raises(ExportDocumentError):
        verify_export(document)
