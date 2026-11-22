import uuid
from datetime import UTC, datetime

import pytest

from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent

NOW = datetime(2026, 10, 17, tzinfo=UTC)


def event(**kw: object) -> AuditEvent:
    return AuditEvent(owner_id=uuid.uuid4(), action=AuditAction.PET_BOUND, occurred_at=NOW, **kw)  # type: ignore[arg-type]


def test_accepts_codes_numbers_and_flags() -> None:
    e = event(
        subject="d945c77026fd4f4a8a899bbff4aeed86", metadata={"code": "ok_1", "n": 3, "on": True}
    )
    assert e.metadata["n"] == 3


@pytest.mark.parametrize(
    "kw",
    [
        {"metadata": {"text": "mình thích chơi buổi tối"}},  # free text is rejected
        {"metadata": {"Bad": 1}},
        {"metadata": {f"k{i}": i for i in range(9)}},
        {"subject": "a b"},
        {"request_id": "x" * 65},
    ],
)
def test_rejects_anything_that_could_carry_content(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        event(**kw)


def test_requires_aware_time() -> None:
    with pytest.raises(ValueError):
        AuditEvent(
            owner_id=uuid.uuid4(), action=AuditAction.PET_BOUND, occurred_at=datetime(2026, 1, 1)
        )


def test_mesh_audit_actions_and_content_free_metadata() -> None:
    assert AuditAction.HOST_UPDATED.value == "host.updated"
    assert AuditAction.HOST_HABIT_DELETED.value == "host.habit_deleted"

    host_id = str(uuid.uuid4())
    habit_id = str(uuid.uuid4())

    updated_evt = AuditEvent(
        owner_id=uuid.uuid4(),
        action=AuditAction.HOST_UPDATED,
        occurred_at=NOW,
        subject=host_id,
        metadata={"hostId": host_id, "fieldsCount": 3},
    )
    assert updated_evt.metadata["fieldsCount"] == 3

    deleted_single = AuditEvent(
        owner_id=uuid.uuid4(),
        action=AuditAction.HOST_HABIT_DELETED,
        occurred_at=NOW,
        subject=habit_id,
        metadata={"habitId": habit_id},
    )
    assert deleted_single.metadata["habitId"] == habit_id

    deleted_bulk = AuditEvent(
        owner_id=uuid.uuid4(),
        action=AuditAction.HOST_HABIT_DELETED,
        occurred_at=NOW,
        subject=str(uuid.uuid4()),
        metadata={"deletedCount": 4},
    )
    assert deleted_bulk.metadata["deletedCount"] == 4

    # Raw contextKey (>64 chars or free text) must be rejected by AuditEvent
    with pytest.raises(ValueError):
        AuditEvent(
            owner_id=uuid.uuid4(),
            action=AuditAction.HOST_HABIT_DELETED,
            occurred_at=NOW,
            metadata={"contextKey": "media:" + "x" * 70},
        )
