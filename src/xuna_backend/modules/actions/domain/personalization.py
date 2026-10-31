"""Deterministic personalization & learning rules for UDC-10 (S10.1-S10.3).

Rules (UDC-9-10-workflows-learning.md §4):
- 4.1 Host defaults (`host_default {family, hostAlias}`):
  - Created ONLY after owner confirmation:
    (a) 3-pick rule: after the owner picks the same host 3 times in a row for a
        tool family, offer `"Lần sau mở {family} trên {hostAlias} luôn nhé?"`
        (explicitly labelled as a rule, never called "AI learning").
    (b) Stated in chat (`candidate_type=KNOWLEDGE`, `kind="host_default"`,
        `status=AWAITING_OWNER_CONFIRMATION` -> `CONFIRMED`).
  - Acts ONLY as a tie-breaker in `TargetResolver` when no explicit alias is
    given and several hosts are eligible.
  - NEVER skips an R2+ approval.
  - Deleting a `host_default` convention takes effect immediately.
- 4.2 Routine-based offers:
  - If the owner ran an R1 action in `family` on `host_alias` in the same
    `hour_of_week` (0..167) on >= 3 of the last 4 weeks, and Routine V1 says
    they are usually around, produce an offer card (`"gợi ý theo thói quen"`)
    with a "Why?" explanation showing the rule and the counts.
  - An offer NEVER executes by itself.
  - Ignored or declined twice in a row -> pauses that `(family, host_alias)`
    offer for 2 weeks (14 days).
- 4.3 Bond level affects offer frequency ONLY:
  - Mới quen (0) -> max 0 offers/day
  - Bắt nhịp (1) -> max 1 offer/day
  - Ăn ý (2)     -> max 2 offers/day
  - Đồng đội (3) -> max 3 offers/day
  - Bond level NEVER changes grants, risk tiers, or approvals.
- 4.4 Personality:
  - Changes offer wording only (temperament phrasing table), never cap or execution.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum

from xuna_backend.modules.hosts.domain.entities import Grant, RiskTier, ToolDescriptor, ToolFamily
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
)

HOST_DEFAULT_PICK_THRESHOLD = 3
ROUTINE_MIN_WEEKS_OUT_OF_FOUR = 3
ROUTINE_WINDOW_WEEKS = 4
ROUTINE_DECLINE_PAUSE_THRESHOLD = 2
ROUTINE_PAUSE_DURATION = timedelta(days=14)
OFFER_LABEL_HABIT_VI = "gợi ý theo thói quen"
OFFER_LABEL_RULE_VI = "gợi ý theo quy tắc"


class BondLevel(IntEnum):
    """Bond level from Knowledge Profile (0..3)."""

    MOI_QUEN = 0  # Mới quen
    BAT_NHIP = 1  # Bắt nhịp
    AN_Y = 2  # Ăn ý
    DONG_DOI = 3  # Đồng đội


BOND_LEVEL_LABELS_VI: dict[BondLevel, str] = {
    BondLevel.MOI_QUEN: "Mới quen",
    BondLevel.BAT_NHIP: "Bắt nhịp",
    BondLevel.AN_Y: "Ăn ý",
    BondLevel.DONG_DOI: "Đồng đội",
}

BOND_LEVEL_MAX_OFFERS_PER_DAY: dict[BondLevel, int] = {
    BondLevel.MOI_QUEN: 0,
    BondLevel.BAT_NHIP: 1,
    BondLevel.AN_Y: 2,
    BondLevel.DONG_DOI: 3,
}


def max_offers_per_day(bond_level: BondLevel | int) -> int:
    """Returns the daily routine-offer cap for `bond_level` (UDC-10 §4.3)."""
    try:
        lvl = BondLevel(int(bond_level))
    except ValueError:
        return 0
    return BOND_LEVEL_MAX_OFFERS_PER_DAY[lvl]


_FAMILY_PROMPT_VI: dict[str, str] = {
    "media": "mở nhạc",
    "app": "mở ứng dụng",
    "url": "mở liên kết web",
    "workflow": "chạy quy trình",
}


@dataclass(frozen=True, slots=True)
class HostDefaultConvention:
    """Confirmed `host_default` convention for a `(owner_id, pet_id, family)`."""

    candidate_id: uuid.UUID
    owner_id: uuid.UUID
    pet_id: str
    family: str
    host_alias: str


@dataclass(frozen=True, slots=True)
class HostDefaultRuleOffer:
    """Offer produced after the owner picked the same host 3 times in a row."""

    family: str
    host_alias: str
    consecutive_picks: int
    label: str
    prompt_vi: str
    why_vi: str


class HostPickTracker:
    """Tracks consecutive owner host picks per `(owner_id, pet_id, family)` (UDC-10 §4.1a)."""

    def __init__(self) -> None:
        self._state: dict[tuple[uuid.UUID, str, str], tuple[str, int]] = {}

    def record_pick(
        self,
        *,
        owner_id: uuid.UUID,
        pet_id: str,
        family: str,
        host_alias: str,
        existing_default_alias: str | None = None,
    ) -> HostDefaultRuleOffer | None:
        """Records an explicit host pick and returns a rule offer on the 3rd consecutive pick."""
        norm_family = family.strip().lower()
        clean_alias = host_alias.strip()
        if not norm_family or not clean_alias:
            return None
        if (
            existing_default_alias is not None
            and existing_default_alias.strip().lower() == clean_alias.lower()
        ):
            return None

        key = (owner_id, pet_id, norm_family)
        prev = self._state.get(key)
        count = prev[1] + 1 if prev is not None and prev[0].lower() == clean_alias.lower() else 1
        self._state[key] = (clean_alias, count)

        if count == HOST_DEFAULT_PICK_THRESHOLD:
            verb = _FAMILY_PROMPT_VI.get(norm_family, f"dùng {norm_family}")
            return HostDefaultRuleOffer(
                family=norm_family,
                host_alias=clean_alias,
                consecutive_picks=count,
                label=OFFER_LABEL_RULE_VI,
                prompt_vi=f"Lần sau {verb} trên {clean_alias} luôn nhé?",
                why_vi=(
                    f"Quy tắc: Bạn đã chọn {clean_alias} {count} lần liên tiếp "
                    f"khi {verb}."
                ),
            )
        return None


class InMemoryHostDefaultStore:
    """Store of owner-confirmed `host_default` conventions with immediate deletion effect."""

    def __init__(self) -> None:
        self._items: dict[uuid.UUID, HostDefaultConvention] = {}

    def add_confirmed(
        self,
        *,
        owner_id: uuid.UUID,
        pet_id: str,
        family: str,
        host_alias: str,
        candidate_id: uuid.UUID | None = None,
    ) -> HostDefaultConvention:
        cid = candidate_id or uuid.uuid4()
        norm_family = family.strip().lower()
        # Replace any prior default for the same (owner_id, pet_id, family)
        to_remove = [
            k
            for k, v in self._items.items()
            if v.owner_id == owner_id and v.pet_id == pet_id and v.family == norm_family
        ]
        for k in to_remove:
            self._items.pop(k, None)
        conv = HostDefaultConvention(
            candidate_id=cid,
            owner_id=owner_id,
            pet_id=pet_id,
            family=norm_family,
            host_alias=host_alias.strip(),
        )
        self._items[cid] = conv
        return conv

    def delete(self, candidate_id: uuid.UUID, *, owner_id: uuid.UUID | None = None) -> bool:
        existing = self._items.get(candidate_id)
        if existing is None:
            return False
        if owner_id is not None and existing.owner_id != owner_id:
            return False
        self._items.pop(candidate_id, None)
        return True

    def get_default_alias(
        self,
        *,
        owner_id: uuid.UUID,
        pet_id: str,
        family: ToolFamily | str,
    ) -> str | None:
        norm_family = (
            family.value if isinstance(family, ToolFamily) else str(family).strip().lower()
        )
        for conv in self._items.values():
            if conv.owner_id == owner_id and conv.pet_id == pet_id and conv.family == norm_family:
                return conv.host_alias
        return None

    def list_for_pet(self, *, owner_id: uuid.UUID, pet_id: str) -> list[HostDefaultConvention]:
        return [
            conv
            for conv in self._items.values()
            if conv.owner_id == owner_id and conv.pet_id == pet_id
        ]


def extract_active_host_defaults(
    candidates: Sequence[MemoryCandidate],
) -> dict[str, str]:
    """Extracts `{family: hostAlias}` from owner-confirmed `MemoryCandidate`s only.

    Unconfirmed (`awaiting_owner_confirmation`), rejected, or deleted candidates
    are strictly ignored (UDC-10 §4.1).
    """
    defaults: dict[str, str] = {}
    for cand in candidates:
        if cand.candidate_type != CandidateType.KNOWLEDGE:
            continue
        if cand.status not in (
            CandidateStatus.CONFIRMED,
            CandidateStatus.AWAITING_DEVICE_COMMIT,
        ):
            continue
        if cand.content.get("kind") != "host_default":
            continue
        family = cand.content.get("family")
        host_alias = cand.content.get("hostAlias")
        if isinstance(family, str) and isinstance(host_alias, str):
            norm_family = family.strip().lower()
            clean_alias = host_alias.strip()
            if norm_family and clean_alias:
                defaults[norm_family] = clean_alias
    return defaults


def hour_of_week(dt: datetime) -> int:
    """Returns `0..167` (`weekday * 24 + hour`, Monday = 0 .. Sunday = 6)."""
    return dt.weekday() * 24 + dt.hour


def format_routine_offer_wording(
    *,
    family: str,
    host_alias: str,
    hour: int,
    temperament: str = "balanced",
) -> str:
    """Temperament-based phrasing table for routine offers (UDC-10 §4.4).

    Personality changes ONLY wording, never cap or execution.
    """
    verb = _FAMILY_PROMPT_VI.get(family.strip().lower(), "mở nhạc")
    day_prefix = (
        "Sáng rồi"
        if 5 <= hour < 11
        else ("Trưa rồi" if 11 <= hour < 14 else ("Chiều rồi" if 14 <= hour < 18 else "Tối rồi"))
    )
    temp = temperament.strip().lower()
    if temp in ("playful", "lively"):
        return f"{day_prefix} nè, mình {verb} trên {host_alias} cùng bạn nhé?"
    if temp in ("quiet", "composed", "unhurried"):
        return f"Đã đến giờ quen thuộc, bạn có muốn {verb} trên {host_alias} không?"
    if temp == "affectionate":
        return f"Mình nhớ giờ này bạn hay {verb} trên {host_alias}, mở luôn nhé?"
    return f"{day_prefix}, {verb} trên {host_alias} nhé?"


@dataclass(frozen=True, slots=True)
class RoutineHistoryBucket:
    """Content-free R1 action count for `(family, host_alias, hour_of_week, week_index)`."""

    family: str
    host_alias: str
    hour_of_week: int
    week_index: int  # 0..3 within the 4-week rolling window
    count: int = 1


@dataclass(frozen=True, slots=True)
class RoutineActionOffer:
    """Explainable routine-based action offer (UDC-10 §4.2-§4.4). Never auto-executes."""

    family: str
    host_alias: str
    tool_name: str
    hour_of_week: int
    weeks_matched: int
    total_count: int
    bond_level: BondLevel
    temperament: str
    label: str
    text_vi: str
    why_vi: str
    auto_executed: bool = False


class RoutineOfferEngine:
    """Deterministic routine offer evaluator with 2-week pause & bond-level cap."""

    def __init__(self) -> None:
        self._consecutive_declines: dict[tuple[str, str], int] = {}
        self._paused_until: dict[tuple[str, str], datetime] = {}

    def record_response(
        self,
        *,
        family: str,
        host_alias: str,
        accepted: bool,
        now: datetime,
    ) -> None:
        key = (family.strip().lower(), host_alias.strip())
        if accepted:
            self._consecutive_declines[key] = 0
            self._paused_until.pop(key, None)
            return
        declines = self._consecutive_declines.get(key, 0) + 1
        self._consecutive_declines[key] = declines
        if declines >= ROUTINE_DECLINE_PAUSE_THRESHOLD:
            self._paused_until[key] = now + ROUTINE_PAUSE_DURATION

    def is_paused(self, *, family: str, host_alias: str, now: datetime) -> bool:
        key = (family.strip().lower(), host_alias.strip())
        until = self._paused_until.get(key)
        if until is None:
            return False
        if now >= until:
            self._paused_until.pop(key, None)
            self._consecutive_declines[key] = 0
            return False
        return True

    def evaluate(
        self,
        *,
        buckets: Sequence[RoutineHistoryBucket],
        now: datetime,
        routine_around: bool,
        bond_level: BondLevel | int,
        offers_shown_today: int = 0,
        temperament: str = "balanced",
    ) -> list[RoutineActionOffer]:
        """Evaluates content-free 4-week history for `now` (UDC-10 §4.2-§4.4)."""
        if not routine_around:
            return []
        cap = max_offers_per_day(bond_level)
        remaining_cap = max(0, cap - offers_shown_today)
        if remaining_cap <= 0:
            return []

        target_how = hour_of_week(now)
        grouped_weeks: dict[tuple[str, str], set[int]] = {}
        grouped_counts: dict[tuple[str, str], int] = {}

        for b in buckets:
            if b.hour_of_week != target_how or b.count <= 0:
                continue
            if not 0 <= b.week_index < ROUTINE_WINDOW_WEEKS:
                continue
            key = (b.family.strip().lower(), b.host_alias.strip())
            grouped_weeks.setdefault(key, set()).add(b.week_index)
            grouped_counts[key] = grouped_counts.get(key, 0) + b.count

        candidates: list[tuple[int, int, str, str]] = []
        for (fam, alias), weeks in grouped_weeks.items():
            if len(weeks) < ROUTINE_MIN_WEEKS_OUT_OF_FOUR:
                continue
            if self.is_paused(family=fam, host_alias=alias, now=now):
                continue
            candidates.append((len(weeks), grouped_counts[(fam, alias)], fam, alias))

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        resolved_bond = BondLevel(int(bond_level))
        offers: list[RoutineActionOffer] = []

        for weeks_matched, total_count, fam, alias in candidates[:remaining_cap]:
            verb = _FAMILY_PROMPT_VI.get(fam, fam)
            tool_name = "media.play_pause" if fam == "media" else f"{fam}.open"
            text_vi = format_routine_offer_wording(
                family=fam,
                host_alias=alias,
                hour=now.hour,
                temperament=temperament,
            )
            why_vi = (
                f"Quy tắc ({OFFER_LABEL_HABIT_VI}): Bạn đã {verb} trên {alias} "
                f"vào khung {now.hour:02d}:00 trong {weeks_matched}/4 tuần gần đây "
                f"(tổng {total_count} lần, trùng giờ bạn thường có mặt; "
                f"mức gắn kết: {BOND_LEVEL_LABELS_VI[resolved_bond]})."
            )
            offers.append(
                RoutineActionOffer(
                    family=fam,
                    host_alias=alias,
                    tool_name=tool_name,
                    hour_of_week=target_how,
                    weeks_matched=weeks_matched,
                    total_count=total_count,
                    bond_level=resolved_bond,
                    temperament=temperament,
                    label=OFFER_LABEL_HABIT_VI,
                    text_vi=text_vi,
                    why_vi=why_vi,
                    auto_executed=False,
                )
            )
        return offers


@dataclass(frozen=True, slots=True)
class PolicyDecisionSnapshot:
    """Snapshot of policy evaluation to prove bond-level invariance (UDC-10 §4.3)."""

    granted: bool
    risk: RiskTier
    requires_approval: bool
    requires_phone_signature: bool


def evaluate_policy_at_bond_level(
    *,
    descriptor: ToolDescriptor,
    grants: Sequence[Grant],
    now: datetime,
    bond_level: BondLevel,
) -> PolicyDecisionSnapshot:
    """Evaluates tool grant, risk, and approval requirements at `bond_level`.

    Architecture §8 & UDC-10 §4.3 invariant: `bond_level` is deliberately unused
    in policy evaluation — grants, risk tiers, and approvals are 100% identical
    across all bond levels.
    """
    _ = bond_level  # Bond level never changes grants, risk tiers, or approvals.
    granted = any(
        g.family == descriptor.family
        and g.is_active(now)
        and g.max_risk.level >= descriptor.risk.level
        for g in grants
    )
    requires_approval = descriptor.risk in (RiskTier.R2, RiskTier.R3)
    requires_phone_signature = descriptor.risk == RiskTier.R3
    return PolicyDecisionSnapshot(
        granted=granted,
        risk=descriptor.risk,
        requires_approval=requires_approval,
        requires_phone_signature=requires_phone_signature,
    )
