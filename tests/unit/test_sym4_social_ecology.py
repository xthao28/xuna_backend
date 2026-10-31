"""SYM-4 (`S4.2`) unit tests for XUNA-to-XUNA Social Ecology:

- Tests `SocialEcologyService.record_encounter`:
  - Synthesizes a first-person ('mình - bạn') `SocialEncounterNote` for the Evolution /
    Friendship Journal based on peer temperament (`curiosity`/`playfulness` ->
    `harmonious_spark`, `sensitivity` -> `shy_peek`, `sociability` -> `warm_nuzzle`).
  - Produces a cryptographically signed 64-hex-char SHA-256 `SocialPostcard` sharing
    a U0-safe trick/workflow (`spin_dance`, `curious_peek`, `cozy_nuzzle`, etc.).
  - Rejects self-encounters (`my_short_id == peer_short_id`).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from xuna_backend.modules.symbiotic.application.services import SocialEcologyService

NOW = datetime(2026, 10, 31, 19, 0, 0, tzinfo=UTC)


def test_social_ecology_service_records_encounter_and_signed_postcard() -> None:
    service = SocialEcologyService()
    note, postcard = service.record_encounter(
        my_short_id="A1B2C3D4",
        peer_short_id="11223344",
        peer_trait="playfulness",
        peer_tint="golden_amber",
        shared_trick="spin_dance",
        now=NOW,
    )
    assert note.encounter_id.startswith("enc-")
    assert note.peer_short_id == "11223344"
    assert note.greeting_kind == "harmonious_spark"
    assert "11223344" in note.narrative_vi
    assert note.to_wire()["sharedTrick"] == "spin_dance"

    assert postcard.postcard_id.startswith("psc-")
    assert postcard.from_short_id == "11223344"
    assert postcard.to_short_id == "A1B2C3D4"
    assert postcard.shared_skill_id == "spin_dance"
    assert len(postcard.signature_hex) == 64
    assert postcard.to_wire()["signatureHex"] == postcard.signature_hex


def test_social_ecology_service_temperament_variants_and_self_rejection() -> None:
    service = SocialEcologyService()
    shy_note, _ = service.record_encounter(
        my_short_id="A1B2C3D4",
        peer_short_id="55667788",
        peer_trait="sensitivity",
        peer_tint="calm_violet",
        shared_trick="curious_peek",
        now=NOW,
    )
    assert shy_note.greeting_kind == "shy_peek"

    warm_note, _ = service.record_encounter(
        my_short_id="A1B2C3D4",
        peer_short_id="99AABBCC",
        peer_trait="sociability",
        peer_tint="warm_coral",
        shared_trick="cozy_nuzzle",
        now=NOW,
    )
    assert warm_note.greeting_kind == "warm_nuzzle"

    with pytest.raises(ValueError, match="cannot record social encounter with self"):
        service.record_encounter(
            my_short_id="A1B2C3D4",
            peer_short_id="a1b2c3d4",
            now=NOW,
        )
