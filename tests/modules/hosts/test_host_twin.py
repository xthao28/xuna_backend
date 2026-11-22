"""Unit tests for RAM-only `HostTwin` store and `InMemoryHostSessionHub` lifecycle."""

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from xuna_backend.modules.hosts.domain.entities import HostId
from xuna_backend.modules.hosts.infrastructure.host_twin import (
    HostFocusChangedNotice,
    HostPresence,
    HostResources,
    InMemoryHostTwinStore,
)
from xuna_backend.modules.hosts.infrastructure.session_hub import InMemoryHostSessionHub

NOW = datetime(2026, 11, 20, 16, 0, tzinfo=UTC)


def test_host_twin_domain_validation_negative_cases() -> None:
    with pytest.raises(ValueError):
        HostPresence(
            idle_seconds=-1,
            screen_locked=False,
            media_playing=False,
            updated_at=NOW,
        )
    with pytest.raises(ValueError):
        HostPresence(
            idle_seconds=0,
            screen_locked=False,
            media_playing=False,
            updated_at=datetime(2026, 11, 20, 16, 0),
        )
    with pytest.raises(ValueError):
        HostPresence(
            idle_seconds=0,
            screen_locked=False,
            media_playing=False,
            updated_at=NOW,
            battery_percent=101,
        )
    with pytest.raises(ValueError):
        HostPresence(
            idle_seconds=0,
            screen_locked=False,
            media_playing=False,
            updated_at=NOW,
            foreground_app_id="vscode\n",
        )
    with pytest.raises(ValueError):
        HostResources(
            installed_apps=(),
            workspaces=("/Users/owner/project",),
            updated_at=NOW,
        )
    with pytest.raises(ValueError):
        HostResources(
            installed_apps=(),
            workspaces=("C:\\work\\proj",),
            updated_at=NOW,
        )
    for bad_ws in (".", "..", "foo..bar"):
        with pytest.raises(ValueError):
            HostResources(
                installed_apps=(),
                workspaces=(bad_ws,),
                updated_at=NOW,
            )


def test_host_twin_store_offline_fail_closed_and_owner_focus_isolation() -> None:
    store = InMemoryHostTwinStore()
    owner_a = uuid.uuid4()
    owner_b = uuid.uuid4()
    host_a = HostId.generate()
    host_b = HostId.generate()

    presence = HostPresence(
        idle_seconds=5,
        screen_locked=False,
        media_playing=True,
        updated_at=NOW,
        battery_percent=80,
        is_charging=True,
        subnet_hash="a1b2c3d4e5f60718293a4b5c6d7e8f90",
        foreground_app_id="vscode",
    )
    resources = HostResources(
        installed_apps=("vscode", "obsidian"),
        workspaces=("xuna_pet",),
        updated_at=NOW,
        capabilities_version="r1_0001",
    )

    # Offline host rejects presence/resources updates (H1/H2: no stale/fabricated twin)
    assert store.get(str(host_a)) is None
    assert store.update_presence(str(host_a), presence) is None
    assert store.update_resources(str(host_a), resources) is None

    twin = store.register_online(str(host_a), owner_id=owner_a)
    assert twin.presence is None
    assert twin.resources is None
    assert store.update_presence(str(host_a), presence) is not None
    updated = store.update_resources(str(host_a), resources)
    assert updated is not None
    assert updated.presence == presence
    assert updated.resources == resources
    assert len(store.list_for_owner(owner_a)) == 1
    assert store.list_for_owner(owner_b) == []

    # Focus state keyed strictly by owner_id
    s1 = store.set_focus(owner_a, focused_host_id=host_a, follow_me=True, source="hotkey", now=NOW)
    assert s1.focused_host_id == host_a
    assert s1.follow_me is True
    assert store.get_focus(owner_b) is None

    # Preserves follow_me when omitted, keeps focused_host_id when requested
    s2 = store.set_focus(owner_a, focused_host_id=host_b, follow_me=None, source="tray", now=NOW)
    assert s2.focused_host_id == host_b
    assert s2.follow_me is True

    s3 = store.set_focus(owner_a, keep_focused_host=True, follow_me=False, source="app", now=NOW)
    assert s3.focused_host_id == host_b
    assert s3.follow_me is False

    # Revoking host_b clears only focus pointing to host_b
    store.set_focus(owner_b, focused_host_id=host_a, follow_me=True, source="app", now=NOW)
    cleared = store.clear_focus_if_host(str(host_b), now=NOW, source="revoked")
    assert len(cleared) == 1
    assert store.get_focus(owner_a).focused_host_id is None  # type: ignore[union-attr]
    assert store.get_focus(owner_b).focused_host_id == host_a  # type: ignore[union-attr]


def test_session_hub_evicts_twin_on_close_duplicate_and_revoke() -> None:
    async def _run() -> None:
        hub = InMemoryHostSessionHub()
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        tok1 = object()
        tok2 = object()
        closed_dup = False
        closed_rev = False

        async def _send(_txt: str) -> None:
            pass

        async def _dup() -> None:
            nonlocal closed_dup
            closed_dup = True

        async def _rev() -> None:
            nonlocal closed_rev
            closed_rev = True

        assert await hub.open_session(str(host_id), tok1, _send, _dup, _rev, owner_id=owner_id)
        twin = hub.get_twin(str(host_id))
        assert twin is not None
        assert twin.presence is None and twin.resources is None

        hub.update_presence(
            str(host_id),
            HostPresence(idle_seconds=0, screen_locked=False, media_playing=False, updated_at=NOW),
        )
        assert hub.get_twin(str(host_id)) is not None

        # Wrong token close does NOT evict
        assert not hub.close_session(str(host_id), object())
        assert hub.get_twin(str(host_id)) is not None

        # Duplicate session closes old session and immediately evicts HostTwin
        assert not await hub.open_session(str(host_id), tok2, _send, _dup, _rev, owner_id=owner_id)
        assert closed_dup
        assert hub.get_twin(str(host_id)) is None

        # Re-open, pin focus, subscribe, publish, then revoke -> evicts twin and clears focus
        assert await hub.open_session(str(host_id), tok2, _send, _dup, _rev, owner_id=owner_id)
        hub.set_focus_state(
            owner_id, focused_host_id=host_id, follow_me=True, source="hotkey", now=NOW
        )
        received: list[HostFocusChangedNotice] = []

        async def _good_listener(notice: HostFocusChangedNotice) -> None:
            received.append(notice)

        async def _bad_listener(_notice: HostFocusChangedNotice) -> None:
            raise RuntimeError("boom")

        unsub_good = hub.subscribe_focus(owner_id, _good_listener)
        hub.subscribe_focus(owner_id, _bad_listener)

        notice = HostFocusChangedNotice(
            owner_id=owner_id,
            host_id=host_id,
            alias="LinuxBox",
            platform="linux",
            follow_me=True,
            source="hotkey",
            updated_at=NOW,
        )
        await hub.publish_focus_changed(notice)
        assert len(received) == 1
        assert notice.wire_payload()["hostId"] == str(host_id)

        unsub_good()
        await hub.publish_focus_changed(notice)
        assert len(received) == 1

        await hub.disconnect_host(str(host_id), reason="revoked")
        assert closed_rev
        assert hub.get_twin(str(host_id)) is None
        assert hub.get_focus_state(owner_id).focused_host_id is None  # type: ignore[union-attr]

    asyncio.run(_run())
