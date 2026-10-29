"""BuildOwnerContextUseCase: retrieves domain brief, conventions and cases for a turn (KX-1 §6.6).

Application rules: stdlib, domain, application only. No LLM or embedder calls in KX-1.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.domain.entities import DomainStatus, KnowledgeDomain
from xuna_backend.modules.knowledge.domain.lexical_matcher import (
    compute_lexical_similarity,
)
from xuna_backend.modules.knowledge.domain.owner_context import (
    DeviceSlotView,
    DomainBrief,
    OwnerContext,
)
from xuna_backend.modules.memory.domain.entities import CandidateType, MemoryCandidate

if TYPE_CHECKING:
    from xuna_backend.modules.knowledge.application.ports import (
        KnowledgeCandidatePort,
        KnowledgeRepository,
    )

TURN_DOMAIN_THRESHOLD = 0.20  # [P]; KX-1 measures it on tests/fixtures/turn_domain_vi.json
MAX_CONVENTIONS = 6
MAX_CASES = 3
MAX_UNSORTED = 2
CONVENTION_CHARS = 1200
CASE_CHARS = 900


def _item_text(candidate: MemoryCandidate) -> str:
    raw = candidate.content.get("text") or candidate.content.get("summary")
    return raw.strip() if isinstance(raw, str) else ""


def _kind_weight(kind: object) -> float:
    if kind == "convention":
        return 1.0
    if kind == "case":
        return 0.6
    return 0.8


def _recency_score(candidate: MemoryCandidate, now: datetime) -> float:
    ref = candidate.confirmed_at or candidate.updated_at
    age_days = max(0.0, (now - ref).total_seconds() / 86400.0)
    return 1.0 / (1.0 + age_days / 30.0)


def _rank_candidate(query: str, candidate: MemoryCandidate, now: datetime) -> float:
    text = _item_text(candidate)
    lex = compute_lexical_similarity(query, [text]) if text else 0.0
    rec = _recency_score(candidate, now)
    kw = _kind_weight(candidate.content.get("kind"))
    return 0.6 * lex + 0.25 * rec + 0.15 * kw


def _cap_items(texts: list[str], max_count: int, max_chars: int) -> list[str]:
    """Keeps whole items up to `max_count` and total length `<= max_chars`."""
    selected: list[str] = []
    used_chars = 0
    for text in texts:
        if len(selected) >= max_count:
            break
        if not text:
            continue
        if used_chars + len(text) > max_chars:
            continue
        selected.append(text)
        used_chars += len(text)
    return selected


class BuildOwnerContextUseCase:
    def __init__(
        self,
        repo: KnowledgeRepository,
        candidates: KnowledgeCandidatePort,
    ) -> None:
        self._repo = repo
        self._candidates = candidates

    async def execute(
        self,
        pet_id: PetId,
        text: str,
        device_slots: tuple[DeviceSlotView, ...] | None,
        explicit_domain_id: uuid.UUID | None,
        now: datetime,
    ) -> OwnerContext:
        domains = await self._repo.list_domains(pet_id, include_deleted=False)
        slot_map: dict[int, DeviceSlotView] = (
            {s.slot: s for s in device_slots} if device_slots is not None else {}
        )

        active_domains: list[KnowledgeDomain] = []
        for dom in domains:
            if dom.status != DomainStatus.ACTIVE:
                continue
            if device_slots is not None and dom.device_slot is not None:
                slot_view = slot_map.get(dom.device_slot)
                if slot_view is None or slot_view.state != "active":
                    continue
            active_domains.append(dom)

        domain_items: dict[uuid.UUID, list[MemoryCandidate]] = {}
        for dom in active_domains:
            raw_items = await self._candidates.list_candidates_by_domain(
                dom.domain_id, confirmed_only=True
            )
            domain_items[dom.domain_id] = [
                c
                for c in raw_items
                if c.candidate_type == CandidateType.KNOWLEDGE
                and c.content.get("kind") != "document"
            ]

        chosen_domain: KnowledgeDomain | None = None
        domain_explicit = False
        best_score = 0.0

        if explicit_domain_id is not None:
            for dom in active_domains:
                if dom.domain_id == explicit_domain_id:
                    chosen_domain = dom
                    domain_explicit = True
                    items_for_dom = domain_items.get(dom.domain_id, [])
                    item_texts = [t for c in items_for_dom if (t := _item_text(c))]
                    best_score = compute_lexical_similarity(
                        text, [dom.name, dom.description, *item_texts]
                    )
                    break

        if chosen_domain is None:
            for dom in active_domains:
                items_for_dom = domain_items.get(dom.domain_id, [])
                item_texts = [t for c in items_for_dom if (t := _item_text(c))]
                score = compute_lexical_similarity(text, [dom.name, dom.description, *item_texts])
                if score > best_score:
                    best_score = score
                    if score >= TURN_DOMAIN_THRESHOLD:
                        chosen_domain = dom

        conventions: list[str] = []
        cases: list[str] = []

        if chosen_domain is not None:
            items = domain_items.get(chosen_domain.domain_id, [])
            ranked = sorted(
                items,
                key=lambda c: _rank_candidate(text, c, now),
                reverse=True,
            )
            conv_texts: list[str] = []
            case_texts: list[str] = []
            for cand in ranked:
                t = _item_text(cand)
                if not t:
                    continue
                if cand.content.get("kind") == "case":
                    case_texts.append(t)
                else:
                    conv_texts.append(t)

            conventions = _cap_items(conv_texts, MAX_CONVENTIONS, CONVENTION_CHARS)
            cases = _cap_items(case_texts, MAX_CASES, CASE_CHARS)

        # Unsorted knowledge conventions with lex >= 0.20
        unsorted_raw = await self._candidates.list_unsorted_knowledge(pet_id, confirmed_only=True)
        unsorted_scored: list[tuple[float, str]] = []
        for cand in unsorted_raw:
            if cand.candidate_type != CandidateType.KNOWLEDGE:
                continue
            if cand.content.get("kind") in ("case", "document"):
                continue
            t = _item_text(cand)
            if not t or t in conventions:
                continue
            lex = compute_lexical_similarity(text, [t])
            if lex >= TURN_DOMAIN_THRESHOLD:
                unsorted_scored.append((lex, t))

        unsorted_scored.sort(key=lambda pair: pair[0], reverse=True)
        used_conv_chars = sum(len(x) for x in conventions)
        added_unsorted = 0
        for _, t in unsorted_scored:
            if added_unsorted >= MAX_UNSORTED or len(conventions) >= MAX_CONVENTIONS:
                break
            if used_conv_chars + len(t) > CONVENTION_CHARS:
                continue
            conventions.append(t)
            used_conv_chars += len(t)
            added_unsorted += 1

        if chosen_domain is None:
            return OwnerContext(
                domain=None,
                conventions=tuple(conventions),
                cases=(),
                snippets=(),
                domain_score=best_score,
                domain_explicit=False,
            )

        level: int | None = None
        if device_slots is not None and chosen_domain.device_slot is not None:
            sv = slot_map.get(chosen_domain.device_slot)
            if sv is not None:
                level = sv.level

        brief_text = getattr(chosen_domain, "expert_brief", "") or chosen_domain.description
        safety_tier = getattr(chosen_domain, "safety_tier", "general") or "general"

        domain_brief = DomainBrief(
            domain_id=chosen_domain.domain_id,
            name=chosen_domain.name,
            brief=brief_text,
            level=level,
            safety_tier=safety_tier,
        )
        return OwnerContext(
            domain=domain_brief,
            conventions=tuple(conventions),
            cases=tuple(cases),
            snippets=(),
            domain_score=best_score,
            domain_explicit=domain_explicit,
        )

