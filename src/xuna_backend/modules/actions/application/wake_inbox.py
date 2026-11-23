"""Wake Inbox service with Consent U5 redaction and non-intrusive owner pub/sub (AW-M2)."""

import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from xuna_backend.modules.actions.domain.ports import WakeInboxRepository
from xuna_backend.modules.actions.domain.redaction import sanitize_wake_text
from xuna_backend.modules.actions.domain.watch import (
    MAX_WAKE_INBOX_SUMMARY_CHARS,
    MAX_WAKE_INBOX_TITLE_CHARS,
    WakeInboxItem,
    WakeInboxKind,
    WakeSuggestedAction,
    WatchGestureCue,
)
from xuna_backend.modules.hosts.domain.entities import HostId

_logger = logging.getLogger(__name__)


class WakeInboxService:
    """Persists U5-redacted wake notifications and pushes to subscribed owner sockets."""

    def __init__(
        self,
        repo: WakeInboxRepository,
        *,
        clock_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repo = repo
        self._clock = clock_utc
        self._listeners: dict[uuid.UUID, list[Callable[[dict[str, Any]], Awaitable[None]]]] = {}
        self._silent_no_action_count = 0

    @property
    def silent_no_action_count(self) -> int:
        return self._silent_no_action_count

    def subscribe(
        self,
        owner_id: uuid.UUID,
        listener: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> Callable[[], None]:
        """Subscribes an owner transport (e.g. voice WS) to `wake_inbox_item` pushes."""
        bucket = self._listeners.setdefault(owner_id, [])
        bucket.append(listener)

        def _unsubscribe() -> None:
            current = self._listeners.get(owner_id)
            if current is None:
                return
            with contextlib.suppress(ValueError):
                current.remove(listener)
            if not current:
                self._listeners.pop(owner_id, None)

        return _unsubscribe

    async def emit_item(
        self,
        *,
        owner_id: uuid.UUID,
        pet_id: str,
        kind: WakeInboxKind,
        title: str,
        summary: str,
        job_id: uuid.UUID | None = None,
        host_id: HostId | None = None,
        suggested_actions: tuple[WakeSuggestedAction, ...] = (),
        gesture_cue: WatchGestureCue | None = None,
    ) -> WakeInboxItem:
        """Sanitizes, persists, and pushes a `WakeInboxItem` to active owner subscribers."""
        clean_title = (
            sanitize_wake_text(title, max_chars=MAX_WAKE_INBOX_TITLE_CHARS) or "Task update"
        )
        clean_summary = sanitize_wake_text(summary, max_chars=MAX_WAKE_INBOX_SUMMARY_CHARS)
        now = self._clock()
        item = WakeInboxItem(
            item_id=uuid.uuid4(),
            owner_id=owner_id,
            pet_id=pet_id[:64] or "xuna_pet",
            job_id=job_id,
            host_id=host_id,
            kind=kind,
            title=clean_title,
            summary=clean_summary,
            suggested_actions=suggested_actions[:3],
            gesture_cue=gesture_cue,
            created_at=now,
        )
        await self._repo.save(item)
        payload = item.to_dict()
        listeners = list(self._listeners.get(owner_id, ()))
        for listener in listeners:
            with contextlib.suppress(Exception):
                await listener(payload)
        return item

    def record_no_action_silent(
        self,
        *,
        job_id: uuid.UUID,
        watch_id: uuid.UUID,
        wake_checks_used: int,
    ) -> None:
        """Records a silent `NO_ACTION` re-watch cycle without emitting a Wake Inbox item."""
        self._silent_no_action_count += 1
        _logger.debug(
            "watch_no_action_silent job=%s watch=%s checks=%d",
            job_id,
            watch_id,
            wake_checks_used,
        )

    async def list_items(
        self,
        owner_id: uuid.UUID,
        *,
        unread_only: bool = False,
        include_dismissed: bool = False,
        limit: int = 50,
    ) -> list[WakeInboxItem]:
        return await self._repo.list_for_owner(
            owner_id,
            unread_only=unread_only,
            include_dismissed=include_dismissed,
            limit=limit,
        )

    async def ack_item(
        self,
        owner_id: uuid.UUID,
        item_id: uuid.UUID,
        *,
        dismiss: bool = False,
    ) -> WakeInboxItem | None:
        return await self._repo.mark_read_or_dismiss(
            item_id,
            owner_id,
            now=self._clock(),
            dismiss=dismiss,
        )
