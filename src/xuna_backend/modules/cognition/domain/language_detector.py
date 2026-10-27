"""Pure stdlib language detection for bilingual Vietnamese / English interaction."""

from __future__ import annotations

import re

from xuna_backend.modules.cognition.domain.types import Locale

_VI_DIACRITICS = re.compile(
    r"[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)

# Common English words for language identification
_EN_WORDS = frozenset(
    {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
        "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
        "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "take", "people", "into", "year", "your", "good", "some", "could",
        "them", "see", "other", "than", "then", "now", "look", "only", "come",
        "its", "over", "think", "also", "back", "after", "use", "two", "how",
        "our", "work", "first", "well", "way", "even", "new", "want", "because",
        "any", "these", "give", "day", "most", "us", "am", "is", "are", "was",
        "were", "hello", "hi", "xuna", "pet", "yes", "please", "sorry", "thank",
        "thanks", "today", "very", "here", "where", "why", "fine", "great",
        "sure", "nice", "meet", "much", "learn", "english", "speak", "tell",
    }
)

_VI_UNACCENTED_WORDS = frozenset(
    {
        "ban", "minh", "toi", "chao", "khong", "duoc", "nguoi", "rat", "nay",
        "nhung", "cac", "tieng", "viet", "xin", "cam", "on", "nhe", "a", "da",
    }
)


def detect_text_language(text: str, default: Locale = Locale.VI) -> Locale:
    """Detects whether text is primarily Vietnamese or English.

    Pure stdlib:
    - Any Vietnamese diacritics -> Locale.VI
    - Unaccented words matching Vietnamese markers -> Locale.VI
    - Strong English word presence without Vietnamese diacritics -> Locale.EN
    - Fallback -> default
    """
    clean = text.strip()
    if not clean:
        return default

    if _VI_DIACRITICS.search(clean):
        return Locale.VI

    words = re.findall(r"\b[a-zA-Z]+\b", clean.lower())
    if not words:
        return default

    vi_unaccented_hits = sum(1 for w in words if w in _VI_UNACCENTED_WORDS)
    en_hits = sum(1 for w in words if w in _EN_WORDS)

    if vi_unaccented_hits > en_hits:
        return Locale.VI

    if en_hits >= 1 or len(words) >= 3:
        return Locale.EN

    return default
