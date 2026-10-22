import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from tests.helpers import MemoryAuditLog
from xuna_backend.modules.consent.application.use_cases import (
    ConsentError,
    UpdateConsent,
    require_consent,
    update_consent,
)
from xuna_backend.modules.consent.domain.consent import U0, ConsentClass, ConsentState

OWNER = uuid.uuid4()
NOW = datetime(2026, 10, 17, tzinfo=UTC)


class MemoryConsents:
    def __init__(self) -> None:
        self.history: list[ConsentState] = []

    async def current(self, owner_id: uuid.UUID) -> ConsentState:
        return self.history[-1] if self.history else U0

    async def append(self, owner_id: uuid.UUID, state: ConsentState) -> bool:
        if any(s.version == state.version for s in self.history):
            return False
        self.history.append(state)
        return True


def update(
    repo: MemoryConsents, audit: MemoryAuditLog, version: int, **flags: bool | None
) -> ConsentState:
    command = UpdateConsent(
        aggregate_metrics=bool(flags.get("u1", False)),
        experience_events=bool(flags.get("u2", False)),
        conversation_text=bool(flags.get("u3", False)),
        expected_version=version,
        conversation_voice=flags.get("voice"),
    )
    return asyncio.run(update_consent(repo, audit, OWNER, command, NOW, "req12345"))


@pytest.mark.parametrize("consent", list(ConsentClass))
def test_u0_default_denies_every_class(consent: ConsentClass) -> None:
    with pytest.raises(ConsentError, match="consent_required"):
        asyncio.run(require_consent(MemoryConsents(), OWNER, consent))


def test_each_class_is_a_separate_opt_in() -> None:
    repo, audit = MemoryConsents(), MemoryAuditLog()
    state = update(repo, audit, 0, u2=True)
    assert state.version == 1
    asyncio.run(require_consent(repo, OWNER, ConsentClass.EXPERIENCE_EVENTS))
    for other in (ConsentClass.AGGREGATE_METRICS, ConsentClass.CONVERSATION_TEXT):
        with pytest.raises(ConsentError):
            asyncio.run(require_consent(repo, OWNER, other))
    assert audit.events[0].metadata == {"version": 1, "experienceevents": True}


def test_history_is_appended_and_withdrawal_takes_effect() -> None:
    repo, audit = MemoryConsents(), MemoryAuditLog()
    update(repo, audit, 0, u3=True)
    update(repo, audit, 1)
    assert [s.version for s in repo.history] == [1, 2]
    with pytest.raises(ConsentError):
        asyncio.run(require_consent(repo, OWNER, ConsentClass.CONVERSATION_TEXT))
    assert len(audit.events) == 2


def test_voice_is_its_own_opt_in_and_omitting_it_keeps_it() -> None:
    repo, audit = MemoryConsents(), MemoryAuditLog()
    update(repo, audit, 0, u3=True)
    with pytest.raises(ConsentError):
        asyncio.run(require_consent(repo, OWNER, ConsentClass.CONVERSATION_VOICE))
    update(repo, audit, 1, u3=True, voice=True)
    asyncio.run(require_consent(repo, OWNER, ConsentClass.CONVERSATION_VOICE))
    assert audit.events[-1].metadata == {"version": 2, "conversationvoice": True}
    # A client that predates voice (field omitted) changes text consent only.
    kept = update(repo, audit, 2, u3=False)
    assert kept.conversation_voice is True
    assert kept.conversation_text is False
    withdrawn = update(repo, audit, 3, voice=False)
    assert withdrawn.conversation_voice is False


def test_stale_version_conflicts_and_no_change_records_nothing() -> None:
    repo, audit = MemoryConsents(), MemoryAuditLog()
    update(repo, audit, 0, u1=True)
    with pytest.raises(ConsentError, match="consent_version_conflict"):
        update(repo, audit, 0, u1=False)
    same = update(repo, audit, 1, u1=True)
    assert same.version == 1
    assert len(repo.history) == 1
    assert len(audit.events) == 1
