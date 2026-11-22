"""RAM-only ephemeral HostTwin store for online hosts (Spatial Mesh §5.2.1).

Never persisted to PostgreSQL and never logged. Offline hosts are immediately evicted
and have no `HostTwin` (presence=None, resources=None) so stale data is never served.
"""

import uuid
from dataclasses import replace
from datetime import datetime

from xuna_backend.modules.hosts.domain.entities import HostId
from xuna_backend.modules.hosts.domain.twin import (
    HostFocusChangedNotice as HostFocusChangedNotice,
)
from xuna_backend.modules.hosts.domain.twin import (
    HostFocusState as HostFocusState,
)
from xuna_backend.modules.hosts.domain.twin import (
    HostPresence as HostPresence,
)
from xuna_backend.modules.hosts.domain.twin import (
    HostResources as HostResources,
)
from xuna_backend.modules.hosts.domain.twin import (
    HostTwin as HostTwin,
)


class InMemoryHostTwinStore:
    """Process-local RAM store of online `HostTwin` projections and `owner_id` focus state."""

    def __init__(self) -> None:
        self._twins: dict[str, HostTwin] = {}
        self._focus_states: dict[uuid.UUID, HostFocusState] = {}

    @property
    def twins(self) -> dict[str, HostTwin]:
        return dict(self._twins)

    @property
    def focus_states(self) -> dict[uuid.UUID, HostFocusState]:
        return dict(self._focus_states)

    def register_online(self, host_id: str, owner_id: uuid.UUID | None = None) -> HostTwin:
        twin = HostTwin(host_id=HostId.from_str(host_id), owner_id=owner_id)
        self._twins[host_id] = twin
        return twin

    def evict(self, host_id: str) -> HostTwin | None:
        return self._twins.pop(host_id, None)

    def get(self, host_id: str) -> HostTwin | None:
        return self._twins.get(host_id)

    def list_for_owner(self, owner_id: uuid.UUID) -> list[HostTwin]:
        return [twin for twin in self._twins.values() if twin.owner_id == owner_id]

    def update_presence(self, host_id: str, presence: HostPresence) -> HostTwin | None:
        current = self._twins.get(host_id)
        if current is None:
            return None
        updated = replace(current, presence=presence)
        self._twins[host_id] = updated
        return updated

    def update_resources(self, host_id: str, resources: HostResources) -> HostTwin | None:
        current = self._twins.get(host_id)
        if current is None:
            return None
        updated = replace(current, resources=resources)
        self._twins[host_id] = updated
        return updated

    def update_trust_preset(self, host_id: str, trust_preset: str) -> HostTwin | None:
        current = self._twins.get(host_id)
        if current is None:
            return None
        updated = replace(current, trust_preset=trust_preset)
        self._twins[host_id] = updated
        return updated

    def get_focus(self, owner_id: uuid.UUID) -> HostFocusState | None:
        return self._focus_states.get(owner_id)

    def set_focus(
        self,
        owner_id: uuid.UUID,
        *,
        focused_host_id: HostId | None = None,
        keep_focused_host: bool = False,
        follow_me: bool | None = None,
        source: str,
        now: datetime,
    ) -> HostFocusState:
        prev = self._focus_states.get(owner_id)
        resolved_host = (
            prev.focused_host_id if keep_focused_host and prev is not None else focused_host_id
        )
        resolved_follow_me = (
            follow_me if follow_me is not None else (prev.follow_me if prev is not None else False)
        )
        state = HostFocusState(
            owner_id=owner_id,
            focused_host_id=resolved_host,
            follow_me=resolved_follow_me,
            updated_at=now,
            source=source,
        )
        self._focus_states[owner_id] = state
        return state

    def clear_focus_if_host(
        self,
        host_id: str,
        *,
        now: datetime,
        source: str = "revoked",
    ) -> list[HostFocusState]:
        parsed = HostId.from_str(host_id)
        cleared: list[HostFocusState] = []
        for owner_id, state in list(self._focus_states.items()):
            if state.focused_host_id == parsed:
                updated = HostFocusState(
                    owner_id=owner_id,
                    focused_host_id=None,
                    follow_me=state.follow_me,
                    updated_at=now,
                    source=source,
                )
                self._focus_states[owner_id] = updated
                cleared.append(updated)
        return cleared
