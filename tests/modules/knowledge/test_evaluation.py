"""KD-4 Evaluation Benchmark for Vietnamese Knowledge Domains (plan §6.3.1, §6.3.2, §6.3.4, §9).

Evaluates:
1. Lexical classification accuracy across >= 60 Vietnamese knowledge facts in 6 domains.
2. Noise item rejection rate (passwords, questions, chit-chat, personal facts).
3. Clustering and duplicate domain detection.
4. Embedding-based classification comparison.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path

from xuna_backend.modules.conversation.domain.candidate_extractor import (
    extract_knowledge_candidate,
)
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.domain.clustering import (
    check_cluster_duplicate,
    cluster_candidates_lexical,
    derive_cluster_name_and_description,
    is_suggestion_expired,
)
from xuna_backend.modules.knowledge.domain.entities import (
    DomainOrigin,
    DomainStatus,
    KnowledgeDomain,
)
from xuna_backend.modules.knowledge.domain.lexical_matcher import (
    compute_lexical_similarity,
)
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
)

FIXTURES_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "knowledge_vi.json"


def _load_benchmark_data() -> (
    tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]
):
    with FIXTURES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["domains"], data["items"], data["noise_items"]


VALID_PET_ID = PetId("a" * 12 + "4" + "a" * 3 + "8" + "a" * 15)


def test_benchmark_dataset_integrity() -> None:
    domains, items, noise = _load_benchmark_data()
    assert len(domains) == 6
    assert len(items) >= 60
    assert len(noise) >= 10

    # Ensure all domains are represented
    domain_ids = {d["id"] for d in domains}
    item_domain_ids = {item["domain_id"] for item in items}
    assert item_domain_ids == domain_ids


def test_lexical_classification_accuracy() -> None:
    """Evaluates lexical matcher accuracy against the 60 benchmark items."""
    domains_data, items, _ = _load_benchmark_data()

    domains = [
        KnowledgeDomain(
            domain_id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
            pet_id=VALID_PET_ID,
            name=d["name"],
            description=d["description"],
            device_slot=i if i < 5 else None,
            status=DomainStatus.ACTIVE,
            origin=DomainOrigin.OWNER,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        for i, d in enumerate(domains_data)
    ]
    domain_map = {domains_data[i]["id"]: domains[i] for i in range(len(domains))}

    correct_top1 = 0
    tau_threshold = 0.05
    above_tau = 0

    for item in items:
        expected_domain_id = item["domain_id"]
        text = item["text"]

        scores = [
            (
                d,
                compute_lexical_similarity(text, [d.name, d.description]),
            )
            for d in domains
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        best_domain, best_score = scores[0]

        if best_domain == domain_map[expected_domain_id]:
            correct_top1 += 1
        if best_score >= tau_threshold:
            above_tau += 1

    accuracy = correct_top1 / len(items)
    coverage = above_tau / len(items)

    # Benchmark report: top-1 accuracy >= 85%, coverage >= 80% with empirical tau = 0.05
    assert accuracy >= 0.85, f"Lexical top-1 accuracy too low: {accuracy:.2%}"
    assert coverage >= 0.80, f"Coverage above threshold {tau_threshold} too low: {coverage:.2%}"


def test_noise_rejection_rate() -> None:
    """Confirms all noise items (credentials, questions, chit-chat) are rejected."""
    _, _, noise_items = _load_benchmark_data()

    rejected_count = 0
    for noise in noise_items:
        extracted = extract_knowledge_candidate(noise["text"])
        if extracted is None:
            rejected_count += 1

    rejection_rate = rejected_count / len(noise_items)
    # Target: 100% of noise items must be rejected
    assert rejection_rate == 1.0, f"False positives on noise items: {1.0 - rejection_rate:.2%}"


def test_clustering_and_duplicate_detection() -> None:
    """Tests clustering of unsorted candidates and duplicate detection against existing domains."""
    domains_data, items, _ = _load_benchmark_data()

    # Create 8 cooking candidates and 8 programming candidates as unsorted
    owner_id = uuid.uuid4()
    pet_id = VALID_PET_ID
    now = datetime.now(UTC)

    unsorted_cands: list[MemoryCandidate] = []
    cooking_items = [it for it in items if it["domain_id"] == "nau_an"][:8]
    coding_items = [it for it in items if it["domain_id"] == "lap_trinh"][:8]

    for item in cooking_items + coding_items:
        unsorted_cands.append(
            MemoryCandidate(
                candidate_id=uuid.uuid4(),
                owner_id=owner_id,
                pet_id=pet_id,
                source="conversation",
                candidate_type=CandidateType.KNOWLEDGE,
                content={"text": item["text"]},
                status=CandidateStatus.CONFIRMED,
                created_at=now,
                updated_at=now,
            )
        )

    # Cluster with min_cluster_size=4
    clusters = cluster_candidates_lexical(unsorted_cands, min_similarity=0.10, min_cluster_size=4)
    assert len(clusters) >= 2

    # Test title and description derivation
    name, desc = derive_cluster_name_and_description(clusters[0])
    assert len(name) <= 40
    assert len(desc) <= 160
    assert "Chủ đề" in name

    # Test duplicate detection against existing domain
    existing_domain = KnowledgeDomain(
        domain_id=uuid.uuid4(),
        owner_id=owner_id,
        pet_id=pet_id,
        name=domains_data[0]["name"],
        description=domains_data[0]["description"],
        device_slot=0,
        status=DomainStatus.ACTIVE,
        origin=DomainOrigin.OWNER,
        created_at=now,
        updated_at=now,
    )
    dup, score = check_cluster_duplicate(clusters[0], [existing_domain], duplicate_threshold=0.10)
    assert dup is not None
    assert dup.name == domains_data[0]["name"]


def test_suggestion_expiration() -> None:
    now = datetime.now(UTC)
    old = datetime(2026, 1, 1, tzinfo=UTC)
    recent = datetime(2026, 10, 25, tzinfo=UTC)

    # 14 days expiration
    assert is_suggestion_expired(old, now, expiration_days=14) is True
    check_time = datetime(2026, 10, 27, tzinfo=UTC)
    assert is_suggestion_expired(recent, check_time, expiration_days=14) is False


def test_embedding_comparison_simulation() -> None:
    """Simulates embedding cosine similarity classification using sub-word n-gram features."""
    domains_data, items, _ = _load_benchmark_data()
    from xuna_backend.modules.knowledge.domain.vietnamese_text import (
        extract_syllables_and_bigrams,
    )

    # Build simulated vocabulary embeddings (n-gram term frequency vector)
    vocab: dict[str, int] = {}
    for d in domains_data:
        for feat in extract_syllables_and_bigrams(d["name"] + " " + d["description"]):
            if feat not in vocab:
                vocab[feat] = len(vocab)
    for it in items:
        for feat in extract_syllables_and_bigrams(it["text"]):
            if feat not in vocab:
                vocab[feat] = len(vocab)

    def embed(text: str) -> list[float]:
        vec = [0.0] * len(vocab)
        feats = extract_syllables_and_bigrams(text)
        for feat in feats:
            if feat in vocab:
                vec[vocab[feat]] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def cosine(v1: list[float], v2: list[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2, strict=False))

    domain_vecs = {d["id"]: embed(d["name"] + " " + d["description"]) for d in domains_data}

    correct = 0
    for item in items:
        expected = item["domain_id"]
        v = embed(item["text"])
        best_id = max(domain_vecs.keys(), key=lambda did: cosine(v, domain_vecs[did]))
        if best_id == expected:
            correct += 1

    accuracy = correct / len(items)
    assert accuracy >= 0.85, f"Simulated embedding accuracy: {accuracy:.2%}"
