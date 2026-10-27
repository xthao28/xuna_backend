"""Knowledge domain package."""

from xuna_backend.modules.knowledge.domain.entities import (
    DomainOrigin,
    DomainStatus,
    KnowledgeDomain,
    KnowledgeSuggestion,
    KnowledgeSummary,
    RelationKind,
    SuggestionKind,
    SuggestionStatus,
    compute_fact_key,
)
from xuna_backend.modules.knowledge.domain.lexical_matcher import (
    compute_lexical_similarity,
)
from xuna_backend.modules.knowledge.domain.vietnamese_text import (
    extract_syllables_and_bigrams,
    normalize_vietnamese,
    remove_accents,
)

__all__ = [
    "DomainOrigin",
    "DomainStatus",
    "KnowledgeDomain",
    "KnowledgeSuggestion",
    "KnowledgeSummary",
    "RelationKind",
    "SuggestionKind",
    "SuggestionStatus",
    "compute_fact_key",
    "compute_lexical_similarity",
    "extract_syllables_and_bigrams",
    "normalize_vietnamese",
    "remove_accents",
]
