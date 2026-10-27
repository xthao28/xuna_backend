"""Pure stdlib clustering and duplicate detection for knowledge suggestions (plan §6.3.4).

Domain rules: stdlib only. No frameworks, no I/O, no DB.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

from xuna_backend.modules.knowledge.domain.lexical_matcher import (
    compute_lexical_similarity,
)
from xuna_backend.modules.knowledge.domain.vietnamese_text import (
    extract_syllables_and_bigrams,
    normalize_vietnamese,
)

if TYPE_CHECKING:
    from xuna_backend.modules.knowledge.domain.entities import KnowledgeDomain
    from xuna_backend.modules.memory.domain.entities import MemoryCandidate

_STOPWORDS = frozenset(
    {
        "và", "là", "của", "cho", "với", "trong", "để", "khi", "có", "một",
        "các", "những", "này", "được", "rất", "thì", "làm", "phải", "nhiều",
        "ít", "bằng", "vào", "ra", "lên", "xuống", "sau", "trước", "giúp",
        "bạn", "mình", "người", "gì", "nào", "đó", "sẽ", "đã", "đang",
    }
)


def is_suggestion_expired(
    created_at: datetime, now: datetime, expiration_days: int = 14
) -> bool:
    """True if suggestion was created more than expiration_days ago (plan §6.3.4)."""
    return (now - created_at).total_seconds() >= expiration_days * 86400


def compute_pairwise_clean_similarity(text1: str, text2: str) -> float:
    """Computes similarity between two short candidate texts using informative features."""
    feats1 = {
        x
        for x in extract_syllables_and_bigrams(text1)
        if not any(w in _STOPWORDS for w in x.split()) and len(x) > 1
    }
    feats2 = {
        x
        for x in extract_syllables_and_bigrams(text2)
        if not any(w in _STOPWORDS for w in x.split()) and len(x) > 1
    }
    if not feats1 or not feats2:
        return 0.0
    inter = len(feats1 & feats2)
    if inter == 0:
        return 0.0
    overlap = inter / min(len(feats1), len(feats2))
    jaccard = inter / len(feats1 | feats2)
    return 0.7 * overlap + 0.3 * jaccard


def cluster_candidates_lexical(
    candidates: list[MemoryCandidate],
    min_similarity: float = 0.02,
    min_cluster_size: int = 4,
) -> list[list[MemoryCandidate]]:
    """Clusters unsorted knowledge candidates based on lexical similarity.

    Uses connected components graph clustering on informative features.
    Returns clusters with at least min_cluster_size items, ordered by size descending.
    """
    n = len(candidates)
    if n < min_cluster_size:
        return []

    texts = [c.content.get("text", "") for c in candidates]

    # Build adjacency list
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_pairwise_clean_similarity(texts[i], texts[j])
            if sim >= min_similarity:
                adj[i].add(j)
                adj[j].add(i)

    # Find connected components
    visited: set[int] = set()
    components: list[list[int]] = []
    for i in range(n):
        if i not in visited:
            comp: list[int] = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    # Filter by min_cluster_size and sort descending
    valid_clusters = [
        [candidates[idx] for idx in comp]
        for comp in components
        if len(comp) >= min_cluster_size
    ]
    valid_clusters.sort(key=len, reverse=True)
    return valid_clusters


def check_cluster_duplicate(
    cluster: list[MemoryCandidate],
    domains: list[KnowledgeDomain],
    duplicate_threshold: float = 0.10,
) -> tuple[KnowledgeDomain | None, float]:
    """Checks if cluster is similar enough to an existing domain to be add_to_domain (plan §6.3.4).

    Returns (matching_domain, score) if score >= duplicate_threshold, else (None, score).
    """
    best_domain: KnowledgeDomain | None = None
    best_score = 0.0

    for domain in domains:
        target_texts = [domain.name, domain.description]
        scores: list[float] = []
        for cand in cluster:
            text = cand.content.get("text", "")
            s = compute_lexical_similarity(text, target_texts)
            scores.append(s)

        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            # Weighted combine max and avg
            combined = 0.5 * max_score + 0.5 * avg_score
            if combined > best_score:
                best_score = combined
                best_domain = domain

    if best_score >= duplicate_threshold and best_domain is not None:
        return best_domain, best_score
    return None, best_score


def derive_cluster_name_and_description(
    cluster: list[MemoryCandidate],
) -> tuple[str, str]:
    """Generates a concise Vietnamese proposed name (<= 40 chars) and description (<= 160 chars)."""
    counts: Counter[str] = Counter()
    for cand in cluster:
        text = cand.content.get("text", "")
        feats = extract_syllables_and_bigrams(text)
        for f in feats:
            # Check if feature has words not in stopwords
            words = f.split()
            if any(w in _STOPWORDS for w in words):
                continue
            if all(len(w) > 1 for w in words):
                counts[f] += 1

    common = [word for word, _ in counts.most_common(5)]
    top_bigrams = [f for f in common if " " in f]
    top_single = [f for f in common if " " not in f]

    seen_norm: set[str] = set()
    chosen: list[str] = []
    for f in top_bigrams + top_single:
        norm = normalize_vietnamese(f)
        if norm not in seen_norm:
            seen_norm.add(norm)
            chosen.append(f)
        if len(chosen) >= 2:
            break

    if chosen:
        topic_words = ", ".join(chosen)
        name = f"Chủ đề: {topic_words}"
    else:
        name = "Chủ đề kiến thức mới"

    if len(name) > 40:
        name = name[:40].rstrip()

    details = ", ".join(chosen) if chosen else "nội dung tương đồng"
    desc = f"Gồm các kiến thức liên quan đến {details} do bạn dạy."
    if len(desc) > 160:
        desc = desc[:160].rstrip()

    return name, desc
