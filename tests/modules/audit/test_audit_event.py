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
