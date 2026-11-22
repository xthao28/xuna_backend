"""Unit tests for pure Contextual Habit Learner (`actions/domain/habit_learner.py`)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from xuna_backend.modules.actions.domain.habit_learner import (
    DEFAULT_HABIT_GAMMA,
    MAX_CONTEXT_KEY_LENGTH,
    DayPart,
    HabitReasonCode,
    apply_dual_bucket_observation,
    apply_habit_observation,
    build_context_key,
    compute_habit_score,
    context_keys_for_update,
    day_part_for_dt,
    explain_habit,
    habit_score,
    lookup_habit_score,
    resolve_day_part,
)
from xuna_backend.modules.actions.domain.habits import HostRoutingHabit
from xuna_backend.modules.hosts.domain.entities import HostId, ToolFamily

NOW = datetime(2026, 11, 20, 19, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (4, 59, DayPart.NIGHT),
        (5, 0, DayPart.MORNING),
        (10, 59, DayPart.MORNING),
        (11, 0, DayPart.AFTERNOON),
        (16, 59, DayPart.AFTERNOON),
        (17, 0, DayPart.EVENING),
        (21, 59, DayPart.EVENING),
        (22, 0, DayPart.NIGHT),
        (0, 0, DayPart.NIGHT),
    ],
)
def test_resolve_day_part_s10_boundaries(hour: int, minute: int, expected: DayPart) -> None:
    dt = datetime(2026, 11, 20, hour, minute, 0, tzinfo=UTC)
    assert resolve_day_part(dt) == expected


def test_resolve_day_part_rejects_naive_and_applies_offset() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_day_part(datetime(2026, 11, 20, 10, 0, 0))
    # 12:00 UTC + 420 min (UTC+7) = 19:00 -> evening
    assert day_part_for_dt(datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC), 420) == DayPart.EVENING


def test_build_context_key_bounds_and_never_wildcard_star() -> None:
    assert build_context_key(ToolFamily.MEDIA, None, DayPart.EVENING) == "media:-:evening"
    assert build_context_key(ToolFamily.MEDIA, "*", DayPart.EVENING) == "media:-:evening"
    assert build_context_key(ToolFamily.APP, "Audio Player X", DayPart.MORNING) == (
        "app:audio_player_x:morning"
    )
    long_key = build_context_key(ToolFamily.FILES, "proj_" * 60, DayPart.AFTERNOON)
    assert len(long_key) <= MAX_CONTEXT_KEY_LENGTH
    assert "*" not in long_key

    with pytest.raises(ValueError):
        build_context_key(ToolFamily.APP, "bad\napp", DayPart.MORNING)


def test_habit_learner_gamma_085_positive_and_negative_checkpoints() -> None:
    assert DEFAULT_HABIT_GAMMA == 0.85
    assert compute_habit_score(0.0, 0.0, sample_count=0) == 0.5

    owner_id = uuid.uuid4()
    host_id = HostId.generate()

    # 3 consecutive positive picks: P = 1 + 0.85 + 0.85^2 = 2.5725
    row: HostRoutingHabit | None = None
    for i in range(3):
        row = apply_habit_observation(
            row,
            habit_id=uuid.uuid4(),
            owner_id=owner_id,
            pet_id="pet-1",
            context_key="app:audio_player_x:evening",
            host_id=host_id,
            positive=True,
            now=NOW + timedelta(minutes=i),
            reason_code=HabitReasonCode.EXECUTION_OK,
        )
    assert row is not None
    assert row.positive_score == pytest.approx(2.5725)
    assert row.negative_score == 0.0
    assert row.sample_count == 3

    # Steady-state positive asymptote -> ~0.885
    steady = compute_habit_score(1.0 / (1.0 - 0.85), 0.0, sample_count=100)
    assert steady == pytest.approx(0.8846, abs=0.002)

    # From steady-state positive, 4 negative corrections drop score < 0.50 (~0.473)
    steady_row = HostRoutingHabit(
        habit_id=uuid.uuid4(),
        owner_id=owner_id,
        pet_id="pet-1",
        context_key="media:-:evening",
        host_id=host_id,
        positive_score=1.0 / (1.0 - 0.85),
        negative_score=0.0,
        sample_count=50,
        created_at=NOW,
        updated_at=NOW,
    )
    for step in range(1, 5):
        steady_row = apply_habit_observation(
            steady_row,
            habit_id=steady_row.habit_id,
            owner_id=owner_id,
            pet_id="pet-1",
            context_key="media:-:evening",
            host_id=host_id,
            positive=False,
            now=NOW + timedelta(minutes=step),
            reason_code=HabitReasonCode.REDIRECT_PENALTY,
        )
    assert habit_score(steady_row) < 0.50
    assert habit_score(steady_row) == pytest.approx(0.473, abs=0.01)


def test_dual_bucket_update_and_fallback_lookup() -> None:
    owner_id = uuid.uuid4()
    host_id = HostId.generate()
    keys = context_keys_for_update(ToolFamily.APP, "tool_alpha", DayPart.MORNING)
    assert keys == ("app:tool_alpha:morning", "app:-:morning")

    updated = apply_dual_bucket_observation(
        {},
        owner_id=owner_id,
        pet_id="pet-1",
        family=ToolFamily.APP,
        resource_id="tool_alpha",
        day_part=DayPart.MORNING,
        host_id=host_id,
        positive=True,
        now=NOW,
        reason_code=HabitReasonCode.CONFIRMED_PICK,
    )
    assert len(updated) == 2
    by_key_host = {(r.context_key, r.host_id): r for r in updated}

    # Specific lookup succeeds directly
    res_alpha = lookup_habit_score(
        by_key_host,
        family=ToolFamily.APP,
        resource_id="tool_alpha",
        day_part=DayPart.MORNING,
        host_id=host_id,
    )
    assert res_alpha.used_fallback is False
    assert res_alpha.score > 0.5

    # Unseen resource falls back to app:-:morning
    res_beta = lookup_habit_score(
        by_key_host,
        family=ToolFamily.APP,
        resource_id="tool_beta",
        day_part=DayPart.MORNING,
        host_id=host_id,
    )
    assert res_beta.used_fallback is True
    assert res_beta.context_key == "app:-:morning"
    assert res_beta.score == pytest.approx(res_alpha.score)

    # sample_count == 0 on specific row still falls back to wildcard
    zero_spec = HostRoutingHabit(
        habit_id=uuid.uuid4(),
        owner_id=owner_id,
        pet_id="pet-1",
        context_key="app:tool_beta:morning",
        host_id=host_id,
        positive_score=0.0,
        negative_score=0.0,
        sample_count=0,
        created_at=NOW,
        updated_at=NOW,
    )
    by_key_host[("app:tool_beta:morning", host_id)] = zero_spec
    res_zero = lookup_habit_score(
        by_key_host,
        family=ToolFamily.APP,
        resource_id="tool_beta",
        day_part=DayPart.MORNING,
        host_id=host_id,
    )
    assert res_zero.used_fallback is True
    assert res_zero.score > 0.5
    assert "Picked 1 times" in explain_habit(updated[0])
