"""Owner consent (D23). U0 — nothing leaves the phone — is the default and
the state of every owner who never chose otherwise. Each class is a separate
opt-in; none implies another."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class ConsentClass(StrEnum):
    AGGREGATE_METRICS = "aggregate_metrics"  # U1
    EXPERIENCE_EVENTS = "experience_events"  # U2
    CONVERSATION_TEXT = "conversation_text"  # U3 (per provider call, not retained)


@dataclass(frozen=True, slots=True)
class ConsentState:
    aggregate_metrics: bool = False
    experience_events: bool = False
    conversation_text: bool = False
    version: int = 0  # 0 = never recorded (U0 default)
    updated_at: datetime | None = None

    def allows(self, consent: ConsentClass) -> bool:
        match consent:
            case ConsentClass.AGGREGATE_METRICS:
                return self.aggregate_metrics
            case ConsentClass.EXPERIENCE_EVENTS:
                return self.experience_events
            case ConsentClass.CONVERSATION_TEXT:
                return self.conversation_text

    def changed(self, other: "ConsentState") -> list[ConsentClass]:
        return [c for c in ConsentClass if self.allows(c) != other.allows(c)]


U0 = ConsentState()
