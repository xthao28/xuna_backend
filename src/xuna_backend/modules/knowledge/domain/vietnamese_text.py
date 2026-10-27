"""Pure stdlib Vietnamese text normalization and n-gram extraction (plan §6.3.1).

Domain rules: stdlib only. No external NLP dependencies (e.g. underthesea).
"""

from __future__ import annotations

import re
import unicodedata

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_vietnamese(text: str) -> str:
    """NFC normalization, lowercase, trimmed, collapse internal whitespace."""
    nfc = unicodedata.normalize("NFC", text.strip().lower())
    clean = _PUNCTUATION_RE.sub(" ", nfc)
    return _WHITESPACE_RE.sub(" ", clean).strip()


def remove_accents(text: str) -> str:
    """Removes Vietnamese diacritics using stdlib unicode decomposition.

    Explicitly maps 'đ' -> 'd'.
    """
    nfd = unicodedata.normalize("NFD", text)
    without_marks = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    without_d = without_marks.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", without_d)


def extract_syllables_and_bigrams(text: str) -> set[str]:
    """Extracts accented and unaccented syllable unigrams and bigrams for matching."""
    norm = normalize_vietnamese(text)
    clean_text = _PUNCTUATION_RE.sub(" ", norm)
    syllables = [s for s in _WHITESPACE_RE.split(clean_text) if s]

    features: set[str] = set()
    for s in syllables:
        features.add(s)
        unaccented = remove_accents(s)
        if unaccented != s:
            features.add(unaccented)

    for i in range(len(syllables) - 1):
        bigram = f"{syllables[i]} {syllables[i + 1]}"
        features.add(bigram)
        unaccented_bigram = remove_accents(bigram)
        if unaccented_bigram != bigram:
            features.add(unaccented_bigram)

    return features
