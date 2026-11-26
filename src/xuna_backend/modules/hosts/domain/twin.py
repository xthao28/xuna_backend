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
_TOOLCHAIN_ID_RE = re.compile(r"[a-z0-9_.-]{1,32}")
_TOOLCHAIN_VERSION_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._+-]{0,31}")
_FOCUS_SOURCES = frozenset({"app", "slash_cmd", "hotkey", "tray", "follow_me", "system", "revoked"})
_TRUST_PRESETS = frozenset({"strict", "standard", "autopilot"})
_MAX_INSTALLED_APPS = 256
_MAX_WORKSPACES = 128
_MAX_TOOLCHAINS = 64


def _has_disallowed_resource_chars(value: str) -> bool:
    return (
        "/" in value
        or "\\" in value
        or ".." in value
        or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value)
    )


@dataclass(frozen=True, slots=True)
class HostToolchain:
    """Normalized host-probed toolchain declaration (`id` and `version`, zero paths)."""

    id: str
    version: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or self.id in {".", ".."}
            or _has_disallowed_resource_chars(self.id)
            or not _TOOLCHAIN_ID_RE.fullmatch(self.id)
        ):
            raise ValueError(f"invalid toolchain id: {self.id!r}")
        if (
            not isinstance(self.version, str)
            or _has_disallowed_resource_chars(self.version)
            or not _TOOLCHAIN_VERSION_RE.fullmatch(self.version)
        ):
            raise ValueError(f"invalid toolchain version: {self.version!r}")


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
    """Normalized app, workspace, and toolchain digests reported via `xuna/resources`."""

    installed_apps: tuple[str, ...]
    workspaces: tuple[str, ...]
    updated_at: datetime
    capabilities_version: str | None = None
    toolchains: tuple[HostToolchain, ...] = ()

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
        if len(self.toolchains) > _MAX_TOOLCHAINS:
            raise ValueError(f"toolchains exceeds {_MAX_TOOLCHAINS} items")
        seen_toolchains: set[str] = set()
        for tc in self.toolchains:
            if not isinstance(tc, HostToolchain):
                raise ValueError("toolchains entries must be HostToolchain instances")
            if tc.id in seen_toolchains:
                raise ValueError(f"duplicate toolchain id: {tc.id!r}")
            seen_toolchains.add(tc.id)


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
    trust_preset: str | None = None

    def __post_init__(self) -> None:
        if self.trust_preset is not None and self.trust_preset not in _TRUST_PRESETS:
            raise ValueError(f"unsupported trust_preset: {self.trust_preset!r}")


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
