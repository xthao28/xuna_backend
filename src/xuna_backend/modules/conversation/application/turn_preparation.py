"""Turn preparation use case: gathers personal facts, owner context and mode decision (KX-1 §6.7).

Application layer: stdlib, domain, application only.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING

from xuna_backend.modules.conversation.application.chat_turn import TurnContext
from xuna_backend.modules.conversation.domain.fact_format import format_fact
from xuna_backend.modules.conversation.domain.turn_mode import (
    Channel,
    ModeDecision,
    RequestedMode,
    decide_mode,
)
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.owner_context import (
    TURN_DOMAIN_THRESHOLD,
    BuildOwnerContextUseCase,
)
from xuna_backend.modules.knowledge.domain.owner_context import (
    DeviceSlotView,
    OwnerContext,
)
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
)

if TYPE_CHECKING:
    from xuna_backend.modules.memory.application.ports import MemoryRepository

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedTurn:
    context: TurnContext
    decision: ModeDecision
    domain_id: uuid.UUID | None


async def prepare_turn(
    *,
    pet_id: PetId,
    text: str,
    requested: RequestedMode | None,
    channel: Channel,
    base: TurnContext,
    device_slots: tuple[DeviceSlotView, ...] | None,
    explicit_domain_id: uuid.UUID | None,
    memory_repo: MemoryRepository,
    owner_context: BuildOwnerContextUseCase,
    now: datetime,
) -> PreparedTurn:
    confirmed_facts: list[str] = []
    try:
        confirmed_candidates = await memory_repo.list_candidates(
            pet_id, status=CandidateStatus.CONFIRMED
        )
        seen_facts: set[str] = set()
        for cand in confirmed_candidates:
            if cand.candidate_type == CandidateType.FACT:
                raw_fact_key = cand.content.get("fact")
                dedup_key = (
                    str(raw_fact_key) if raw_fact_key else str(cand.content.get("summary", ""))
                )
                if dedup_key in seen_facts:
                    continue
                seen_facts.add(dedup_key)
                formatted = format_fact(cand.content)
                if formatted and len(confirmed_facts) < 5:
                    confirmed_facts.append(formatted)
    except Exception:
        _logger.debug("fact_lookup_failed", exc_info=True)

    owner = OwnerContext()
    try:
        owner = await owner_context.execute(
            pet_id=pet_id,
            text=text,
            device_slots=device_slots,
            explicit_domain_id=explicit_domain_id,
            now=now,
        )
    except Exception:
        _logger.debug("owner_context_failed", exc_info=True)

    domain_matched = owner.domain is not None and (
        owner.domain_explicit or owner.domain_score >= TURN_DOMAIN_THRESHOLD
    )
    decision = decide_mode(
        requested=requested,
        channel=channel,
        text=text,
        domain_matched=domain_matched,
    )
    domain_id = owner.domain.domain_id if owner.domain is not None else None
    updated_context = replace(
        base,
        confirmed_facts=tuple(confirmed_facts),
        owner=owner,
        mode=decision.mode,
    )
    return PreparedTurn(
        context=updated_context,
        decision=decision,
        domain_id=domain_id,
    )

