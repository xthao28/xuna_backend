"""Pure stdlib lexical similarity matching for Vietnamese knowledge domains (plan §6.3.1, §6.3.2).

Domain rules: stdlib only.
"""

from __future__ import annotations

from xuna_backend.modules.knowledge.domain.vietnamese_text import (
    extract_syllables_and_bigrams,
)


def compute_lexical_similarity(query_text: str, target_texts: list[str]) -> float:
    """Computes a lexical overlap/Jaccard similarity score in [0.0, 1.0].

    Combines containment of query n-grams with overall Jaccard index.
    """
    q_feats = extract_syllables_and_bigrams(query_text)
    if not q_feats:
        return 0.0

    t_feats: set[str] = set()
    for t in target_texts:
        t_feats.update(extract_syllables_and_bigrams(t))

    if not t_feats:
        return 0.0

    intersection = len(q_feats & t_feats)
    if intersection == 0:
        return 0.0

    containment = intersection / len(q_feats)
    jaccard = intersection / len(q_feats | t_feats)
    # Blend: containment weights query relevance, jaccard penalizes massive mismatched domains
    return 0.7 * containment + 0.3 * jaccard
