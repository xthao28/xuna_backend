"""Ephemeral, content-free HostTwin domain value objects (XUNA Spatial Device Mesh §5.2.1).

Stdlib only. Holds live host presence, normalized resource digests, and owner-scoped
focus state. These objects live strictly in RAM inside `InMemoryHostSessionHub` and
are never persisted to PostgreSQL or logged.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime

from xuna_backend.modules.hosts.domain.entities import HostId

_SUBNET_HASH_RE = re.compile(r"[0-9a-f]{16,64}")
_APP_ID_RE = re.compile(r"[a-z0-9_.-]{1,64}")
_WORKSPACE_ID_RE = re.compile(r"[a-zA-Z0-9_.-]{1,64}")
_CAP_VERSION_RE = re.compile(r"[a-zA-Z0-9_.-]{1,32}")
_FOCUS_SOURCES = frozenset({"app", "slash_cmd", "hotkey", "tray", "follow_me", "system", "revoked"})
_MAX_INSTALLED_APPS = 256
_MAX_WORKSPACES = 128


@dataclass(frozen=True, slots=True)
class HostPresence:
    """Live presence reported by an online Host Agent via `xuna/presence`."""

    idle_seconds: int
    screen_locked: bool
    media_playing: bool
    updated_at: datetime
    battery_percent: int | None = None
    is_charging: bool | None = None
    subnet_hash: str | None = None
    foreground_app_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.idle_seconds) is not int or self.idle_seconds < 0:
            raise ValueError("idle_seconds must be a non-negative integer")
        if not isinstance(self.screen_locked, bool):
            raise ValueError("screen_locked must be a boolean")
        if not isinstance(self.media_playing, bool):
            raise ValueError("media_playing must be a boolean")
        if self.updated_at.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware")
        if self.battery_percent is not None and (
            type(self.battery_percent) is not int or not 0 <= self.battery_percent <= 100
        ):
            raise ValueError("battery_percent must be an integer in 0..100")
        if self.is_charging is not None and not isinstance(self.is_charging, bool):
            raise ValueError("is_charging must be a boolean or None")
        if self.subnet_hash is not None and not _SUBNET_HASH_RE.fullmatch(self.subnet_hash):
            raise ValueError("subnet_hash must be 16..64 lowercase hex characters")
        if self.foreground_app_id is not None and not _APP_ID_RE.fullmatch(self.foreground_app_id):
            raise ValueError("foreground_app_id must match ^[a-z0-9_.-]{1,64}$")


@dataclass(frozen=True, slots=True)
class HostResources:
    """Normalized app and workspace aliases reported via `xuna/resources` (zero paths)."""

    installed_apps: tuple[str, ...]
    workspaces: tuple[str, ...]
    updated_at: datetime
    capabilities_version: str | None = None

    def __post_init__(self) -> None:
        if self.updated_at.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware")
        if len(self.installed_apps) > _MAX_INSTALLED_APPS:
            raise ValueError(f"installed_apps exceeds {_MAX_INSTALLED_APPS} items")
        for app_id in self.installed_apps:
            if not _APP_ID_RE.fullmatch(app_id):
                raise ValueError(f"invalid installed app id: {app_id!r}")
        if len(self.workspaces) > _MAX_WORKSPACES:
            raise ValueError(f"workspaces exceeds {_MAX_WORKSPACES} items")
        for ws in self.workspaces:
            if (
                "/" in ws
                or "\\" in ws
                or ws in {".", ".."}
                or ".." in ws
                or not _WORKSPACE_ID_RE.fullmatch(ws)
            ):
                raise ValueError(f"invalid sanitized workspace alias: {ws!r}")
        if self.capabilities_version is not None and not _CAP_VERSION_RE.fullmatch(
            self.capabilities_version
        ):
            raise ValueError("capabilities_version must match ^[a-zA-Z0-9_.-]{1,32}$")


@dataclass(frozen=True, slots=True)
class HostFocusState:
    """Owner-scoped live focus pin and Follow Me toggle (`owner_id` key)."""

    owner_id: uuid.UUID
    focused_host_id: HostId | None
    follow_me: bool
    updated_at: datetime
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.follow_me, bool):
            raise ValueError("follow_me must be a boolean")
        if self.updated_at.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware")
        if self.source not in _FOCUS_SOURCES:
            raise ValueError(f"unsupported focus source: {self.source!r}")


@dataclass(frozen=True, slots=True)
class HostTwin:
    """RAM-only live projection of a currently connected host."""

    host_id: HostId
    owner_id: uuid.UUID | None = None
    presence: HostPresence | None = None
    resources: HostResources | None = None


@dataclass(frozen=True, slots=True)
class HostFocusChangedNotice:
    """Domain notice published when an owner's active host focus changes."""

    owner_id: uuid.UUID
    host_id: HostId | None
    alias: str | None
    platform: str | None
    follow_me: bool
    source: str
    updated_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.follow_me, bool):
            raise ValueError("follow_me must be a boolean")
        if self.updated_at.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware")
        if self.source not in _FOCUS_SOURCES:
            raise ValueError(f"unsupported focus source: {self.source!r}")

    def wire_payload(self) -> dict[str, object]:
        return {
            "hostId": str(self.host_id) if self.host_id is not None else None,
            "alias": self.alias,
            "platform": self.platform,
            "followMe": self.follow_me,
            "source": self.source,
        }
