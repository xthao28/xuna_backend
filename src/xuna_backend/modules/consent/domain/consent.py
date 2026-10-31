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
    # U3V (owner D16, 2026-10-22): speech sent to the ASR provider per call; raw
    # audio and transcripts are never stored. Separate from U3: hearing is not reading.
    CONVERSATION_VOICE = "conversation_voice"
    OWNER_DOCUMENTS = "owner_documents"  # U4
    # U5 (UDC-8): desktop screenshots and accessibility trees sent per computer-use
    # step to the AI provider; never retained in DB or logs.
    HOST_SCREEN = "host_screen"
    # U6 (UDC-8 / UDC-10): host notification titles/bodies sent per call; never retained.
    HOST_NOTIFICATIONS = "host_notifications"


@dataclass(frozen=True, slots=True)
class ConsentState:
    aggregate_metrics: bool = False
    experience_events: bool = False
    conversation_text: bool = False
    conversation_voice: bool = False
    owner_documents: bool = False
    host_screen: bool = False
    host_notifications: bool = False
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
            case ConsentClass.CONVERSATION_VOICE:
                return self.conversation_voice
            case ConsentClass.OWNER_DOCUMENTS:
                return self.owner_documents
            case ConsentClass.HOST_SCREEN:
                return self.host_screen
            case ConsentClass.HOST_NOTIFICATIONS:
                return self.host_notifications

    def changed(self, other: "ConsentState") -> list[ConsentClass]:
        return [c for c in ConsentClass if self.allows(c) != other.allows(c)]


U0 = ConsentState()
