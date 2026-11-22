"""Pure stdlib Contextual Habit Learner for spatial multi-host routing (Mesh §5.4.1 #3).

Maintains exponential-decay positive and negative preference evidence per
`(owner_id, pet_id, context_key, host_id)` with dual-bucket updates (`specific` +
family-wildcard `-`) and cold-start neutrality (`0.5`). No I/O, clock, or randomness.
"""

import math
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum

from xuna_backend.modules.actions.domain.habits import (
    DEFAULT_HABIT_GAMMA,
    HostRoutingHabit,
)
from xuna_backend.modules.hosts.domain.entities import HostId, ToolFamily, fold_text

HABIT_LAMBDA_NEG = 1.25
HABIT_ALPHA_0 = 2.0
NEUTRAL_HABIT_SCORE = 0.5
MAX_CONTEXT_KEY_LENGTH = 128

_FAMILY_KEY_RE = re.compile(r"^[a-z0-9_.-]+$")
_NON_BUCKET_CHARS_RE = re.compile(r"[^a-z0-9_.-]+")


class DayPart(StrEnum):
    """Approved S10 day parts for contextual habit bucketing."""

    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


class HabitReasonCode(StrEnum):
    """Short content-free codes describing why a habit row was updated."""

    CONFIRMED_PICK = "confirmed_pick"
    REDIRECT_PENALTY = "redirect_penalty"
    EXECUTION_OK = "execution_ok"


@dataclass(frozen=True, slots=True)
class HabitLookupResult:
    """Result of resolving a habit score with specific-to-wildcard fallback."""

    score: float
    habit: HostRoutingHabit | None
    context_key: str
    used_fallback: bool


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def resolve_day_part(dt: datetime) -> DayPart:
    """Maps a timezone-aware datetime to the S10 day-part bucket."""
    if dt.tzinfo is None:
        raise ValueError("dt must be timezone-aware")
    hour = dt.hour
    if 5 <= hour < 11:
        return DayPart.MORNING
    if 11 <= hour < 17:
        return DayPart.AFTERNOON
    if 17 <= hour < 22:
        return DayPart.EVENING
    return DayPart.NIGHT


def day_part_for_dt(now: datetime, tz_offset_minutes: int | None = None) -> DayPart:
    """Converts `now` by optional UTC offset minutes before resolving the S10 day part."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if tz_offset_minutes is None:
        return resolve_day_part(now)
    target_tz = timezone(timedelta(minutes=tz_offset_minutes))
    return resolve_day_part(now.astimezone(target_tz))


def normalize_resource_bucket(resource_id: str | None) -> str:
    """Normalizes an optional resource identifier for `context_key` (uses `-`, never `*`)."""
    if resource_id is None:
        return "-"
    if _has_control_chars(resource_id):
        raise ValueError("resource_id must not contain control characters")
    folded = fold_text(resource_id).replace(" ", "_").replace(":", "_")
    cleaned = _NON_BUCKET_CHARS_RE.sub("_", folded).strip("_")
    if not cleaned or cleaned in ("*", "-"):
        return "-"
    return cleaned


def build_context_key(
    family: ToolFamily | str,
    resource_id: str | None,
    day_part: DayPart | str,
) -> str:
    """Builds `f'{family}:{resource_id_or_dash}:{day_part}'` within `128` chars (never `*`)."""
    raw_fam = family.value if isinstance(family, ToolFamily) else str(family)
    if _has_control_chars(raw_fam):
        raise ValueError("family must not contain control characters")
    fam = raw_fam.strip(" ").lower()
    if not fam or not _FAMILY_KEY_RE.fullmatch(fam):
        raise ValueError("family must be a valid lowercase identifier")
    dp = DayPart(day_part).value
    res = normalize_resource_bucket(resource_id)
    max_res_len = MAX_CONTEXT_KEY_LENGTH - len(fam) - len(dp) - 2
    if max_res_len < 1:
        raise ValueError("family and day_part exceed context_key budget")
    if len(res) > max_res_len:
        res = res[:max_res_len].rstrip("_") or "-"
    return f"{fam}:{res}:{dp}"


def wildcard_context_key(family: ToolFamily | str, day_part: DayPart | str) -> str:
    """Builds the family-level fallback bucket `f'{family}:-:{day_part}'`."""
    return build_context_key(family, None, day_part)


def context_keys_for_update(
    family: ToolFamily | str,
    resource_id: str | None,
    day_part: DayPart | str,
) -> tuple[str, ...]:
    """Returns `(specific_key, wildcard_key)` when `resource_id` is concrete, else `(wildcard,)`."""
    specific = build_context_key(family, resource_id, day_part)
    wildcard = wildcard_context_key(family, day_part)
    if specific == wildcard:
        return (wildcard,)
    return (specific, wildcard)


def compute_habit_score(
    positive_score: float,
    negative_score: float,
    *,
    sample_count: int | None = None,
    lambda_neg: float = HABIT_LAMBDA_NEG,
    alpha_0: float = HABIT_ALPHA_0,
) -> float:
    """Computes normalized habit score `M_hab` in `[0.0, 1.0]` (`0.5` at cold start)."""
    if (
        not math.isfinite(positive_score)
        or not math.isfinite(negative_score)
        or positive_score < 0.0
        or negative_score < 0.0
    ):
        raise ValueError("positive_score and negative_score must be finite floats >= 0.0")
    if not math.isfinite(lambda_neg) or lambda_neg <= 0.0:
        raise ValueError("lambda_neg must be > 0.0")
    if not math.isfinite(alpha_0) or alpha_0 <= 0.0:
        raise ValueError("alpha_0 must be > 0.0")
    if sample_count == 0 or (positive_score == 0.0 and negative_score == 0.0):
        return NEUTRAL_HABIT_SCORE
    weighted_neg = lambda_neg * negative_score
    denom = 2.0 * (positive_score + weighted_neg + alpha_0)
    raw = 0.5 + (positive_score - weighted_neg) / denom
    return max(0.0, min(1.0, raw))


def habit_score(habit: HostRoutingHabit | None) -> float:
    """Convenience scorer for a `HostRoutingHabit` row."""
    if habit is None or habit.sample_count == 0:
        return NEUTRAL_HABIT_SCORE
    return compute_habit_score(
        habit.positive_score,
        habit.negative_score,
        sample_count=habit.sample_count,
    )


def lookup_habit_score(
    habits_by_key_host: Mapping[tuple[str, HostId], HostRoutingHabit],
    *,
    family: ToolFamily | str,
    resource_id: str | None,
    day_part: DayPart | str,
    host_id: HostId,
) -> HabitLookupResult:
    """Looks up specific `family:q:day_part` first; falls back to `family:-:day_part`."""
    specific_key = build_context_key(family, resource_id, day_part)
    wildcard_key = wildcard_context_key(family, day_part)
    if specific_key != wildcard_key:
        spec_habit = habits_by_key_host.get((specific_key, host_id))
        if spec_habit is not None and spec_habit.sample_count > 0:
            return HabitLookupResult(
                score=habit_score(spec_habit),
                habit=spec_habit,
                context_key=specific_key,
                used_fallback=False,
            )
    wild_habit = habits_by_key_host.get((wildcard_key, host_id))
    if wild_habit is not None and wild_habit.sample_count > 0:
        return HabitLookupResult(
            score=habit_score(wild_habit),
            habit=wild_habit,
            context_key=wildcard_key,
            used_fallback=(specific_key != wildcard_key),
        )
    return HabitLookupResult(
        score=NEUTRAL_HABIT_SCORE,
        habit=None,
        context_key=specific_key,
        used_fallback=False,
    )


def apply_habit_observation(
    existing: HostRoutingHabit | None,
    *,
    habit_id: uuid.UUID,
    owner_id: uuid.UUID,
    pet_id: str,
    context_key: str,
    host_id: HostId,
    positive: bool,
    now: datetime,
    reason_code: HabitReasonCode | str,
    gamma: float | None = None,
) -> HostRoutingHabit:
    """Applies one decayed positive confirmation or negative correction to a habit row."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    resolved_gamma = (
        existing.decay_gamma
        if (existing is not None and gamma is None)
        else (DEFAULT_HABIT_GAMMA if gamma is None else gamma)
    )
    if not math.isfinite(resolved_gamma) or not (0.0 < resolved_gamma < 1.0):
        raise ValueError("gamma must be finite in (0.0, 1.0)")
    old_pos = existing.positive_score if existing is not None else 0.0
    old_neg = existing.negative_score if existing is not None else 0.0
    old_count = existing.sample_count if existing is not None else 0
    if positive:
        new_pos = resolved_gamma * old_pos + 1.0
        new_neg = resolved_gamma * old_neg
    else:
        new_pos = resolved_gamma * old_pos
        new_neg = resolved_gamma * old_neg + 1.0
    reason_str = reason_code.value if isinstance(reason_code, HabitReasonCode) else str(reason_code)
    return HostRoutingHabit(
        habit_id=existing.habit_id if existing is not None else habit_id,
        owner_id=owner_id,
        pet_id=pet_id,
        context_key=context_key,
        host_id=host_id,
        positive_score=new_pos,
        negative_score=new_neg,
        sample_count=old_count + 1,
        decay_gamma=resolved_gamma,
        last_reason_code=reason_str,
        created_at=existing.created_at if existing is not None else now,
        updated_at=now,
    )


def apply_dual_bucket_observation(
    existing_by_key_host: Mapping[tuple[str, HostId], HostRoutingHabit],
    *,
    owner_id: uuid.UUID,
    pet_id: str,
    family: ToolFamily | str,
    resource_id: str | None,
    day_part: DayPart | str,
    host_id: HostId,
    positive: bool,
    now: datetime,
    reason_code: HabitReasonCode | str,
    habit_id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
    habit_ids: Mapping[str, uuid.UUID] | None = None,
    gamma: float | None = None,
) -> tuple[HostRoutingHabit, ...]:
    """Updates both the specific resource bucket and the `family:-:day_part` wildcard bucket."""
    keys = context_keys_for_update(family, resource_id, day_part)
    updated: list[HostRoutingHabit] = []
    for ctx_key in keys:
        existing = existing_by_key_host.get((ctx_key, host_id))
        hid = (
            existing.habit_id
            if existing is not None
            else (
                habit_ids[ctx_key]
                if habit_ids is not None and ctx_key in habit_ids
                else habit_id_factory()
            )
        )
        updated.append(
            apply_habit_observation(
                existing,
                habit_id=hid,
                owner_id=owner_id,
                pet_id=pet_id,
                context_key=ctx_key,
                host_id=host_id,
                positive=positive,
                now=now,
                reason_code=reason_code,
                gamma=gamma,
            )
        )
    return tuple(updated)


def explain_habit(habit: HostRoutingHabit) -> str:
    """Formats a concise content-free explanation of a learned habit row (`why`)."""
    parts = habit.context_key.split(":", 2)
    family = parts[0] if parts else "action"
    day_part = parts[2] if len(parts) == 3 else "day"
    score = habit_score(habit)
    if habit.negative_score > habit.positive_score and habit.last_reason_code == "redirect_penalty":
        return (
            f"Redirected away in {habit.sample_count} samples for {family} in {day_part}; "
            f"active confidence {score:.2f}"
        )
    return (
        f"Picked {habit.sample_count} times for {family} in {day_part}; "
        f"active confidence {score:.2f}"
    )
