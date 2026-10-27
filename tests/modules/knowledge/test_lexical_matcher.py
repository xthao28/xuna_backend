"""Unit tests for Vietnamese lexical matching (KD-3)."""

from xuna_backend.modules.knowledge.domain.lexical_matcher import (
    compute_lexical_similarity,
)
from xuna_backend.modules.knowledge.domain.vietnamese_text import (
    extract_syllables_and_bigrams,
    normalize_vietnamese,
    remove_accents,
)


def test_vietnamese_text_normalization() -> None:
    text = "  Hà  Nội,  Việt  Nam!  "
    assert normalize_vietnamese(text) == "hà nội việt nam"

    assert remove_accents("hà nội") == "ha noi"
    assert remove_accents("đà nẵng") == "da nang"
    assert remove_accents("điện biên") == "dien bien"


def test_extract_syllables_and_bigrams() -> None:
    features = extract_syllables_and_bigrams("thủ đô hà nội")
    assert "thủ" in features
    assert "đô" in features
    assert "thu" in features  # unaccented
    assert "hà nội" in features
    assert "ha noi" in features  # unaccented bigram


def test_compute_lexical_similarity() -> None:
    query = "Hà Nội là thủ đô"
    targets = ["Thủ đô Hà Nội của Việt Nam"]

    score = compute_lexical_similarity(query, targets)
    assert score > 0.5  # high overlap

    # Dissimilar
    unrelated_targets = ["Mèo con thích ăn cá và ngủ cả ngày"]
    unrelated_score = compute_lexical_similarity(query, unrelated_targets)
    assert unrelated_score < 0.2

    # Accent resilience: "ha noi thu do" vs "Thủ đô Hà Nội"
    accentless_score = compute_lexical_similarity("ha noi thu do", targets)
    assert accentless_score > 0.4
