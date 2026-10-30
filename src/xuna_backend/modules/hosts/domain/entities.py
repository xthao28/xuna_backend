"""Host Registry domain value objects and entities (UDC-3 §5.4).

Stdlib only. Validates aliases, tool descriptors, forbidden tool names, risk
tiers, manifests, grants, pending enrollments and active/revoked hosts.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

_TOOL_NAME_RE = re.compile(r"^[a-z]+\.[a-z_]+$")
_SAS_RE = re.compile(r"^[0-9]{6}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

FORBIDDEN_TOOLS: frozenset[str] = frozenset(
    {
        "shell.exec",
        "shell.run",
        "powershell.run",
        "powershell.exec",
        "osascript.run",
        "osascript.exec",
        "script.run",
        "script.exec",
        "eval.run",
        "eval.exec",
        "cmd.run",
        "cmd.exec",
        "bash.run",
        "bash.exec",
    }
)

FORBIDDEN_TOOL_PREFIXES: tuple[str, ...] = (
    "shell.",
    "powershell.",
    "osascript.",
    "eval.",
    "script.",
    "keyboard.",
    "pointer.",
    "cmd.",
    "bash.",
)


@dataclass(frozen=True, slots=True)
class HostId:
    """UUID wrapper identifying a registered host or pending enrollment."""

    value: uuid.UUID

    @classmethod
    def from_str(cls, raw: str) -> "HostId":
        try:
            parsed = uuid.UUID(raw)
        except ValueError as error:
            raise ValueError(f"invalid HostId UUID: {raw!r}") from error
        if parsed.int == 0:
            raise ValueError("HostId cannot be the zero UUID")
        return cls(parsed)

    @classmethod
    def generate(cls) -> "HostId":
        return cls(uuid.uuid4())

    def __post_init__(self) -> None:
        if self.value.int == 0:
            raise ValueError("HostId cannot be the zero UUID")

    @property
    def bytes(self) -> bytes:
        return self.value.bytes

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class HostAlias:
    """Owner-assigned display name (1..24 chars; letters, digits, space, `-`, `_`)."""

    value: str

    def __post_init__(self) -> None:
        trimmed = self.value.strip()
        if not 1 <= len(trimmed) <= 24:
            raise ValueError("host alias must be 1..24 characters")
        for ch in trimmed:
            if not (ch.isalnum() or ch in (" ", "-", "_")):
                raise ValueError(f"unsupported character in host alias: {ch!r}")
        object.__setattr__(self, "value", trimmed)

    def __str__(self) -> str:
        return self.value


class HostPlatform(StrEnum):
    WINDOWS = "windows"
    MACOS = "macos"
    ANDROID = "android"
    IOS = "ios"
    LINUX = "linux"


class HostStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"


class RiskTier(StrEnum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"

    @property
    def level(self) -> int:
        return {
            RiskTier.R0: 0,
            RiskTier.R1: 1,
            RiskTier.R2: 2,
            RiskTier.R3: 3,
            RiskTier.R4: 4,
        }[self]

    def allows(self, required: "RiskTier") -> bool:
        return self.level >= required.level


class ToolFamily(StrEnum):
    MEDIA = "media"
    APP = "app"
    URL = "url"
    WORKFLOW = "workflow"
    HOST = "host"
    SCREEN = "screen"
    UI = "ui"
    INPUT = "input"
    FILES = "files"


class ToolPermissionState(StrEnum):
    GRANTED = "granted"
    DENIED = "denied"
    NOT_DETERMINED = "not_determined"


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Validated MCP Tool descriptor with `_meta['xuna/policy']`."""

    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    family: ToolFamily
    risk: RiskTier
    observes: bool = False
    reversible: bool = True
    permission: ToolPermissionState = ToolPermissionState.GRANTED
    policy_version: int = 1

    def __post_init__(self) -> None:
        validate_tool_name(self.name, expected_family=self.family)
        if not 1 <= len(self.description) <= 240:
            raise ValueError("tool description must be 1..240 characters")
        if self.policy_version != 1:
            raise ValueError("tool policy version must be 1")
        if (
            self.input_schema.get("type") != "object"
            or self.input_schema.get("additionalProperties") is not False
        ):
            raise ValueError(f"tool {self.name} inputSchema must forbid additionalProperties")
        if (
            self.output_schema.get("type") != "object"
            or self.output_schema.get("additionalProperties") is not False
        ):
            raise ValueError(f"tool {self.name} outputSchema must forbid additionalProperties")

    @classmethod
    def from_wire(cls, payload: dict[str, object]) -> "ToolDescriptor":
        required_keys = {"name", "description", "inputSchema", "outputSchema", "_meta"}
        if set(payload.keys()) != required_keys:
            raise ValueError("ToolDescriptor must have exact keys")
        name = payload["name"]
        description = payload["description"]
        input_schema = payload["inputSchema"]
        output_schema = payload["outputSchema"]
        meta = payload["_meta"]
        if (
            not isinstance(name, str)
            or not isinstance(description, str)
            or not isinstance(input_schema, dict)
            or not isinstance(output_schema, dict)
            or not isinstance(meta, dict)
        ):
            raise ValueError("invalid ToolDescriptor field types")
        policy = meta.get("xuna/policy")
        if not isinstance(policy, dict) or set(meta.keys()) != {"xuna/policy"}:
            raise ValueError("ToolDescriptor._meta must contain only xuna/policy")
        policy_keys = {
            "version",
            "family",
            "risk",
            "observes",
            "reversible",
            "permission",
        }
        if set(policy.keys()) != policy_keys:
            raise ValueError("xuna/policy must have exact keys")
        version = policy["version"]
        observes = policy["observes"]
        reversible = policy["reversible"]
        if (
            type(version) is not int
            or not isinstance(observes, bool)
            or not isinstance(reversible, bool)
        ):
            raise ValueError("invalid xuna/policy scalar types")
        return cls(
            name=name,
            description=description,
            input_schema=dict(input_schema),
            output_schema=dict(output_schema),
            family=ToolFamily(str(policy["family"])),
            risk=RiskTier(str(policy["risk"])),
            observes=observes,
            reversible=reversible,
            permission=ToolPermissionState(str(policy["permission"])),
            policy_version=version,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "_meta": {
                "xuna/policy": {
                    "version": self.policy_version,
                    "family": self.family.value,
                    "risk": self.risk.value,
                    "observes": self.observes,
                    "reversible": self.reversible,
                    "permission": self.permission.value,
                }
            },
        }


def validate_tool_name(name: str, *, expected_family: ToolFamily | None = None) -> ToolFamily:
    """Validates `^[a-z]+\\.[a-z_]+$`, rejects `FORBIDDEN_TOOLS` and prefixes,
    and ensures the prefix is a valid `ToolFamily`."""
    if name in FORBIDDEN_TOOLS:
        raise ValueError(f"forbidden tool name: {name}")
    for prefix in FORBIDDEN_TOOL_PREFIXES:
        if name.startswith(prefix):
            raise ValueError(f"forbidden tool prefix: {name}")
    if not _TOOL_NAME_RE.match(name):
        raise ValueError(f"invalid tool name format: {name}")
    family_str = name.split(".", 1)[0]
    try:
        family = ToolFamily(family_str)
    except ValueError as error:
        raise ValueError(f"unknown tool family in {name}") from error
    if expected_family is not None and family != expected_family:
        raise ValueError(
            f"tool {name} family prefix {family.value} does not match {expected_family.value}"
        )
    return family


@dataclass(frozen=True, slots=True)
class Manifest:
    """Tool manifest advertised by a Host Agent in `tools/list`."""

    host_id: HostId
    tools: tuple[ToolDescriptor, ...]
    updated_at: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.updated_at.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        seen: set[str] = set()
        for tool in self.tools:
            if tool.name in seen:
                raise ValueError(f"duplicate tool in manifest: {tool.name}")
            seen.add(tool.name)

    def find_tool(self, name: str) -> ToolDescriptor | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


@dataclass(frozen=True, slots=True)
class Grant:
    """Owner-configured capability grant for a tool family on a specific host."""

    host_id: HostId
    family: ToolFamily
    max_risk: RiskTier
    enabled: bool
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.updated_at.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware")


def default_initial_grants(host_id: HostId, now: datetime) -> tuple[Grant, ...]:
    """Initial grants created on enrollment confirmation (`UDC-3` §5.1):
    R0 for all MVP families, plus R1 enabled for `media`."""
    return (
        Grant(
            host_id=host_id,
            family=ToolFamily.MEDIA,
            max_risk=RiskTier.R1,
            enabled=True,
            updated_at=now,
        ),
        Grant(
            host_id=host_id,
            family=ToolFamily.APP,
            max_risk=RiskTier.R0,
            enabled=False,
            updated_at=now,
        ),
        Grant(
            host_id=host_id,
            family=ToolFamily.URL,
            max_risk=RiskTier.R0,
            enabled=False,
            updated_at=now,
        ),
        Grant(
            host_id=host_id,
            family=ToolFamily.WORKFLOW,
            max_risk=RiskTier.R0,
            enabled=False,
            updated_at=now,
        ),
        Grant(
            host_id=host_id,
            family=ToolFamily.HOST,
            max_risk=RiskTier.R1,
            enabled=True,
            updated_at=now,
        ),
    )


@dataclass(frozen=True, slots=True)
class PendingEnrollment:
    """Short-lived (5 min) enrollment record awaiting owner claim and SAS confirmation."""

    pending_id: HostId
    code_sha256: str
    public_key: bytes
    key_fingerprint: str
    sas: str
    platform: HostPlatform
    os_version: str
    agent_version: str
    hostname: str
    created_at: datetime
    expires_at: datetime
    owner_id: uuid.UUID | None = None
    alias: HostAlias | None = None

    def __post_init__(self) -> None:
        if not _SHA256_HEX_RE.match(self.code_sha256):
            raise ValueError("code_sha256 must be 64 hex chars")
        if len(self.public_key) != 65 or self.public_key[0] != 0x04:
            raise ValueError("public_key must be a 65-byte uncompressed P-256 point")
        if not _SAS_RE.match(self.sas):
            raise ValueError("sas must be 6 digits")
        if not 1 <= len(self.hostname) <= 32:
            raise ValueError("hostname must be 1..32 characters")
        if not 1 <= len(self.os_version) <= 64:
            raise ValueError("os_version must be 1..64 characters")
        if not 1 <= len(self.agent_version) <= 32:
            raise ValueError("agent_version must be 1..32 characters")
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


@dataclass(frozen=True, slots=True)
class Host:
    """Enrolled host entity (`hosts` table)."""

    host_id: HostId
    owner_id: uuid.UUID
    alias: HostAlias
    platform: HostPlatform
    hostname: str
    os_version: str
    agent_version: str
    public_key: bytes
    status: HostStatus
    created_at: datetime
    last_seen_at: datetime | None = field(default=None)

    def __post_init__(self) -> None:
        if len(self.public_key) != 65 or self.public_key[0] != 0x04:
            raise ValueError("public_key must be a 65-byte uncompressed P-256 point")
        if not 1 <= len(self.hostname) <= 32:
            raise ValueError("hostname must be 1..32 characters")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.last_seen_at is not None and self.last_seen_at.tzinfo is None:
            raise ValueError("last_seen_at must be timezone-aware")
