"""Strict JSON-RPC 2.0 frame parsing and builders for XTP v1 (UDC-3 §5.2)."""

import base64
import binascii
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from xuna_backend.modules.hosts.domain.crypto import (
    NONCE_SIZE,
    PUBLIC_KEY_SIZE,
    SIGNATURE_SIZE,
    HostProofPurpose,
    decode_b64url_16b,
    validate_public_key,
)
from xuna_backend.modules.hosts.domain.entities import (
    HostId,
    HostPlatform,
    RiskTier,
    ToolDescriptor,
)
from xuna_backend.modules.hosts.domain.session import (
    FRAME_TOO_LARGE,
    MAX_MEDIA_STRUCTURED_CONTENT_BYTES,
    MAX_STRUCTURED_CONTENT_BYTES,
    UNAUTHORIZED,
    UNSUPPORTED_PROTOCOL,
    HostSessionRejected,
    has_media_structured_content,
    structured_content_byte_size,
    structured_content_exceeds_limit,
)

MAX_FRAME_BYTES_DEFAULT = 524_288


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EnrollHelloPayload(_StrictModel):
    code: str = Field(min_length=20, max_length=28)
    publicKey: str = Field(min_length=80, max_length=96)
    platform: Literal["windows", "macos", "linux"]
    osVersion: str = Field(min_length=1, max_length=64)
    agentVersion: str = Field(min_length=1, max_length=32)
    hostname: str = Field(min_length=1, max_length=32)

    def code_bytes(self) -> bytes:
        return decode_b64url_16b(self.code)

    def public_key_bytes(self) -> bytes:
        raw = decode_b64_exact(self.publicKey, PUBLIC_KEY_SIZE)
        validate_public_key(raw)
        return raw

    def parsed_platform(self) -> HostPlatform:
        return HostPlatform(self.platform)


class HelloEnrollParamsModel(_StrictModel):
    protocols: list[str] = Field(min_length=1, max_length=8)
    enroll: EnrollHelloPayload


class HelloReconnectParamsModel(_StrictModel):
    protocols: list[str] = Field(min_length=1, max_length=8)
    hostId: str = Field(min_length=36, max_length=36)
    platform: Literal["windows", "macos", "android", "ios", "linux"]
    osVersion: str = Field(min_length=1, max_length=64)
    agentVersion: str = Field(min_length=1, max_length=32)

    def parsed_host_id(self) -> HostId:
        return HostId.from_str(self.hostId)

    def parsed_platform(self) -> HostPlatform:
        return HostPlatform(self.platform)


class ProofParamsModel(_StrictModel):
    purpose: Literal[1, 2]
    signature: str = Field(min_length=80, max_length=96)

    def parsed_purpose(self) -> HostProofPurpose:
        return HostProofPurpose(self.purpose)

    def signature_bytes(self) -> bytes:
        return decode_b64_exact(self.signature, SIGNATURE_SIZE)


class ProgressParamsModel(_StrictModel):
    progressToken: str = Field(min_length=1, max_length=64)
    progress: int = Field(ge=0)
    total: int | None = Field(default=None, ge=0)
    message: str | None = Field(default=None, pattern=r"^[a-z0-9_.-]{1,64}$")


class StopParamsModel(_StrictModel):
    reason: Literal["owner_app", "owner_host", "owner_board", "revoked"]


class LanPeerParamsModel(_StrictModel):
    enabled: bool
    port: int | None = Field(default=47821, ge=1024, le=65535)
    certFingerprint: str | None = Field(default=None, min_length=16, max_length=96)


class PresenceParamsModel(_StrictModel):
    schemaVersion: Literal[1]
    idleSeconds: int = Field(ge=0)
    screenLocked: bool
    mediaPlaying: bool
    batteryPercent: int | None = Field(default=None, ge=0, le=100)
    isCharging: bool | None = None
    subnetHash: str | None = Field(default=None, pattern=r"^[0-9a-f]{16,64}$")
    foregroundAppId: str | None = Field(default=None, pattern=r"^[a-z0-9_.-]{1,64}$")
    updatedAt: str = Field(min_length=10, max_length=64)

    @field_validator("subnetHash", "foregroundAppId")
    @classmethod
    def _reject_newlines(cls, value: str | None) -> str | None:
        if value is not None and ("\n" in value or "\r" in value):
            raise ValueError("newline characters are not allowed")
        return value

    @field_validator("updatedAt")
    @classmethod
    def _validate_updated_at(cls, value: str) -> str:
        if "T" not in value and "t" not in value:
            raise ValueError("updatedAt must be an ISO-8601 datetime string")
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("updatedAt must be timezone-aware")
        return value

    def parsed_updated_at(self) -> datetime:
        return datetime.fromisoformat(self.updatedAt).astimezone(UTC)


_TOOLCHAIN_ID_RE = re.compile(r"^[a-z0-9_.-]{1,32}$")
_TOOLCHAIN_VERSION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._+-]{0,31}$")


class ToolchainModel(_StrictModel):
    """Strict host-declared toolchain entry (`XUNA_XTP_V1_CONTRACT.md` §4.2.11)."""

    id: str = Field(min_length=1, max_length=32)
    version: str = Field(min_length=1, max_length=32)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if (
            "/" in value
            or "\\" in value
            or ".." in value
            or value in {".", ".."}
            or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value)
            or not _TOOLCHAIN_ID_RE.fullmatch(value)
        ):
            raise ValueError("invalid toolchain id")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if (
            "/" in value
            or "\\" in value
            or ".." in value
            or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value)
            or not _TOOLCHAIN_VERSION_RE.fullmatch(value)
        ):
            raise ValueError("invalid toolchain version")
        return value


class ResourcesParamsModel(_StrictModel):
    schemaVersion: Literal[1]
    installedApps: list[Annotated[str, Field(pattern=r"^[a-z0-9_.-]{1,64}$")]] = Field(
        max_length=256
    )
    workspaces: list[Annotated[str, Field(pattern=r"^[a-zA-Z0-9_.-]{1,64}$")]] = Field(
        max_length=128
    )
    capabilitiesVersion: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]{1,32}$")
    toolchains: list[ToolchainModel] = Field(default_factory=list, max_length=64)

    @field_validator("installedApps", "workspaces")
    @classmethod
    def _reject_list_newlines(cls, items: list[str]) -> list[str]:
        for item in items:
            if "\n" in item or "\r" in item or "/" in item or "\\" in item:
                raise ValueError("invalid character in resource identifier")
        return items

    @field_validator("workspaces")
    @classmethod
    def _reject_workspace_traversal(cls, items: list[str]) -> list[str]:
        for item in items:
            if item in {".", ".."} or ".." in item:
                raise ValueError("path traversal tokens are not allowed in workspaces")
        return items

    @field_validator("capabilitiesVersion")
    @classmethod
    def _reject_cap_newlines(cls, value: str | None) -> str | None:
        if value is not None and ("\n" in value or "\r" in value):
            raise ValueError("newline characters are not allowed")
        return value

    @field_validator("toolchains")
    @classmethod
    def _reject_duplicate_toolchains(cls, items: list[ToolchainModel]) -> list[ToolchainModel]:
        seen: set[str] = set()
        for item in items:
            if item.id in seen:
                raise ValueError("duplicate toolchain id")
            seen.add(item.id)
        return items


class FocusRequestParamsModel(_StrictModel):
    schemaVersion: Literal[1]
    source: Literal["hotkey", "tray"]
    followMe: bool | None = None


_UUID_CANONICAL_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_WATCH_SHORT_CODE_RE = r"^[a-z0-9_.-]{1,64}$"
_WATCH_WORKSPACE_ALIAS_RE = r"^[a-zA-Z0-9_.-]{1,64}$"

WatchModeWire = Literal[
    "ui_quiescence",
    "ui_change",
    "process_idle",
    "fs_quiescence",
    "cdp_idle",
    "composite",
    "timer",
    "protocol_async",
    "exec_async",
]
WatchCompositeSignalWire = Literal[
    "ui_quiescence",
    "process_idle",
    "fs_quiescence",
    "cdp_idle",
]
WatchObservedSignalWire = Literal[
    "ui_quiescence",
    "ui_change",
    "process_idle",
    "fs_quiescence",
    "cdp_idle",
    "timer",
    "protocol_async",
    "exec_async",
]
WatchRegisterErrorWire = Literal[
    "watch_unsupported",
    "target_not_found",
    "workspace_not_found",
    "watch_limit_reached",
    "needs_permission",
    "policy_blocked",
    "stopped",
    "invalid_args",
]
WatchCancelReasonWire = Literal[
    "owner_cancel",
    "estop",
    "job_ended",
    "replaced",
    "expired",
]
WatchEventKindWire = Literal["settled", "changed", "timeout", "target_lost", "error"]


def _validate_non_zero_uuid_str(value: str) -> str:
    if "\n" in value or "\r" in value or not _UUID_CANONICAL_RE.fullmatch(value):
        raise ValueError("must be a canonical UUID")
    parsed = uuid.UUID(value)
    if parsed.int == 0:
        raise ValueError("UUID cannot be zero")
    return value


def _reject_control_chars(value: str | None, field_name: str) -> str | None:
    if value is not None and any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"control characters are not allowed in {field_name}")
    return value


def _require_strict_schema_version_1(value: object) -> int:
    if type(value) is not int or value != 1:
        raise ValueError("schemaVersion must be strict integer 1")
    return value


StrictSchemaVersion1 = Annotated[Literal[1], BeforeValidator(_require_strict_schema_version_1)]


class WatchTargetModel(_StrictModel):
    """Normalized, content-free target specification (`XUNA_XTP_V1_CONTRACT.md` §4.2.14)."""

    pid: int | None = Field(default=None, ge=1, le=2_147_483_647)
    windowId: int | None = Field(default=None, ge=0, le=2_147_483_647)
    appId: str | None = Field(default=None, pattern=_WATCH_SHORT_CODE_RE)
    workspaceAlias: str | None = Field(default=None, pattern=_WATCH_WORKSPACE_ALIAS_RE)
    tabId: str | None = Field(default=None, min_length=1, max_length=128)
    mcpServerId: str | None = Field(default=None, pattern=_WATCH_SHORT_CODE_RE)
    protocolTaskId: str | None = Field(default=None, min_length=1, max_length=128)
    execTaskId: str | None = Field(default=None, pattern=r"^exec_[a-zA-Z0-9_-]{1,64}$")

    @field_validator("appId", "mcpServerId")
    @classmethod
    def _reject_code_newlines(cls, value: str | None) -> str | None:
        if value is not None and ("\n" in value or "\r" in value):
            raise ValueError("newline characters are not allowed")
        return value

    @field_validator("workspaceAlias")
    @classmethod
    def _validate_workspace_alias(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if "\n" in value or "\r" in value or "/" in value or "\\" in value:
            raise ValueError("invalid character in workspaceAlias")
        if value in {".", ".."} or ".." in value:
            raise ValueError("path traversal tokens are not allowed in workspaceAlias")
        return value

    @field_validator("tabId", "protocolTaskId")
    @classmethod
    def _validate_opaque_ids(cls, value: str | None) -> str | None:
        return _reject_control_chars(value, "target identifier")


class WatchRegisterParamsModel(_StrictModel):
    """`xuna/watchRegister` request params (`XUNA_XTP_V1_CONTRACT.md` §4.2.14)."""

    schemaVersion: StrictSchemaVersion1
    watchId: str
    jobId: str
    mode: WatchModeWire
    signals: list[WatchCompositeSignalWire] | None = None
    target: WatchTargetModel
    quiescenceSeconds: int = Field(ge=2, le=120)
    minWaitSeconds: int = Field(ge=0, le=60)
    maxWaitSeconds: int = Field(ge=5, le=900)
    requireInitialChange: bool
    releaseFocus: bool

    @field_validator("watchId", "jobId")
    @classmethod
    def _validate_uuids(cls, value: str) -> str:
        return _validate_non_zero_uuid_str(value)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "WatchRegisterParamsModel":
        if self.minWaitSeconds > self.maxWaitSeconds:
            raise ValueError("minWaitSeconds must not exceed maxWaitSeconds")
        if self.mode == "composite":
            if (
                self.signals is None
                or not 2 <= len(self.signals) <= 3
                or len(set(self.signals)) != len(self.signals)
            ):
                raise ValueError("composite mode requires 2..3 distinct signals")
        elif "signals" in self.model_fields_set or self.signals is not None:
            raise ValueError("signals is forbidden when mode is not composite")
        return self


class WatchRegisterResultModel(_StrictModel):
    """`xuna/watchRegister` response body (`XUNA_XTP_V1_CONTRACT.md` §4.2.14)."""

    schemaVersion: StrictSchemaVersion1
    accepted: bool
    watchId: str
    activeSignals: list[WatchObservedSignalWire] | None = None
    eventDriven: bool | None = None
    fallbackUsed: bool | None = None
    errorCode: WatchRegisterErrorWire | None = None

    @field_validator("watchId")
    @classmethod
    def _validate_watch_id(cls, value: str) -> str:
        return _validate_non_zero_uuid_str(value)

    @model_validator(mode="after")
    def _validate_outcome(self) -> "WatchRegisterResultModel":
        if self.accepted:
            if (
                self.activeSignals is None
                or not 1 <= len(self.activeSignals) <= 3
                or len(set(self.activeSignals)) != len(self.activeSignals)
            ):
                raise ValueError(
                    "accepted watchRegister result requires 1..3 distinct activeSignals"
                )
            if self.eventDriven is None or self.fallbackUsed is None:
                raise ValueError(
                    "accepted watchRegister result requires eventDriven and fallbackUsed"
                )
            if "errorCode" in self.model_fields_set or self.errorCode is not None:
                raise ValueError("accepted watchRegister result must omit errorCode")
        else:
            if self.errorCode is None:
                raise ValueError("rejected watchRegister result requires errorCode")
            if any(
                k in self.model_fields_set for k in ("activeSignals", "eventDriven", "fallbackUsed")
            ):
                raise ValueError("rejected watchRegister result must omit acceptance fields")
        return self


class WatchCancelParamsModel(_StrictModel):
    """`xuna/watchCancel` request params (`XUNA_XTP_V1_CONTRACT.md` §4.2.15)."""

    schemaVersion: StrictSchemaVersion1
    watchId: str | None = None
    jobId: str | None = None
    reason: WatchCancelReasonWire

    @field_validator("watchId", "jobId")
    @classmethod
    def _validate_optional_uuids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_zero_uuid_str(value)

    @model_validator(mode="after")
    def _validate_target_ids(self) -> "WatchCancelParamsModel":
        if self.reason != "estop" and self.watchId is None and self.jobId is None:
            raise ValueError("watchId or jobId is required unless reason is estop")
        return self


class WatchCancelResultModel(_StrictModel):
    """`xuna/watchCancel` response body (`XUNA_XTP_V1_CONTRACT.md` §4.2.15)."""

    schemaVersion: StrictSchemaVersion1
    cancelledCount: int = Field(ge=0)


class WatchEventParamsModel(_StrictModel):
    """`xuna/watchEvent` notification params (`XUNA_XTP_V1_CONTRACT.md` §4.2.16)."""

    schemaVersion: StrictSchemaVersion1
    watchId: str
    jobId: str
    kind: WatchEventKindWire
    observedSignals: list[WatchObservedSignalWire] = Field(max_length=3)
    elapsedSeconds: int = Field(ge=0)
    changeCount: int = Field(ge=0)
    digestChanged: bool
    occurredAt: str = Field(min_length=10, max_length=64)
    errorCode: str | None = Field(default=None, pattern=_WATCH_SHORT_CODE_RE)

    @field_validator("watchId", "jobId")
    @classmethod
    def _validate_uuids(cls, value: str) -> str:
        return _validate_non_zero_uuid_str(value)

    @field_validator("errorCode")
    @classmethod
    def _reject_error_code_newlines(cls, value: str | None) -> str | None:
        if value is not None and ("\n" in value or "\r" in value):
            raise ValueError("newline characters are not allowed")
        return value

    @field_validator("occurredAt")
    @classmethod
    def _validate_occurred_at(cls, value: str) -> str:
        if "\n" in value or "\r" in value or ("T" not in value and "t" not in value):
            raise ValueError("occurredAt must be an ISO-8601 datetime string")
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("occurredAt must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_event_fields(self) -> "WatchEventParamsModel":
        if len(set(self.observedSignals)) != len(self.observedSignals):
            raise ValueError("observedSignals must be distinct")
        if self.kind in ("settled", "changed") and not self.observedSignals:
            raise ValueError("observedSignals must be non-empty for settled/changed")
        if self.kind == "error":
            if self.errorCode is None:
                raise ValueError("errorCode is required when kind is error")
        elif "errorCode" in self.model_fields_set or self.errorCode is not None:
            raise ValueError("errorCode is forbidden when kind is not error")
        return self

    def parsed_occurred_at(self) -> datetime:
        return datetime.fromisoformat(self.occurredAt).astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ParsedHelloEnroll:
    request_id: int | str
    params: HelloEnrollParamsModel


@dataclass(frozen=True, slots=True)
class ParsedHelloReconnect:
    request_id: int | str
    params: HelloReconnectParamsModel


@dataclass(frozen=True, slots=True)
class ParsedProof:
    request_id: int | str
    params: ProofParamsModel


@dataclass(frozen=True, slots=True)
class ParsedPing:
    request_id: int | str


@dataclass(frozen=True, slots=True)
class ParsedToolsListChanged:
    pass


@dataclass(frozen=True, slots=True)
class ParsedProgress:
    params: ProgressParamsModel


@dataclass(frozen=True, slots=True)
class ParsedStop:
    params: StopParamsModel


@dataclass(frozen=True, slots=True)
class ParsedLanPeer:
    params: LanPeerParamsModel


@dataclass(frozen=True, slots=True)
class ParsedPresence:
    params: PresenceParamsModel


@dataclass(frozen=True, slots=True)
class ParsedResources:
    params: ResourcesParamsModel


@dataclass(frozen=True, slots=True)
class ParsedFocusRequest:
    params: FocusRequestParamsModel


@dataclass(frozen=True, slots=True)
class ParsedWatchEvent:
    params: WatchEventParamsModel


@dataclass(frozen=True, slots=True)
class ParsedRpcResponse:
    request_id: int | str
    result: dict[str, object]


@dataclass(frozen=True, slots=True)
class ParsedRpcError:
    request_id: int | str
    code: int
    message: str


InboundAgentMessage = (
    ParsedHelloEnroll
    | ParsedHelloReconnect
    | ParsedProof
    | ParsedPing
    | ParsedToolsListChanged
    | ParsedProgress
    | ParsedStop
    | ParsedLanPeer
    | ParsedPresence
    | ParsedResources
    | ParsedFocusRequest
    | ParsedWatchEvent
    | ParsedRpcResponse
    | ParsedRpcError
)


def decode_b64_exact(value: str, expected_len: int) -> bytes:
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HostSessionRejected(UNSUPPORTED_PROTOCOL) from error
    if len(raw) != expected_len:
        raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
    return raw


def parse_agent_frame(
    text: str, *, max_bytes: int = MAX_FRAME_BYTES_DEFAULT
) -> InboundAgentMessage:
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        raise HostSessionRejected(FRAME_TOO_LARGE)
    try:
        raw = json.loads(text)
    except ValueError as error:
        raise HostSessionRejected(UNSUPPORTED_PROTOCOL) from error
    if not isinstance(raw, dict) or raw.get("jsonrpc") != "2.0":
        raise HostSessionRejected(UNSUPPORTED_PROTOCOL)

    if "method" in raw:
        method = raw.get("method")
        params = raw.get("params")
        if not isinstance(method, str) or not isinstance(params, dict):
            raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
        has_id = "id" in raw
        expected_keys = (
            {"jsonrpc", "id", "method", "params"} if has_id else {"jsonrpc", "method", "params"}
        )
        if set(raw.keys()) != expected_keys:
            raise HostSessionRejected(UNSUPPORTED_PROTOCOL)

        if has_id:
            req_id = raw["id"]
            if type(req_id) is not int and not isinstance(req_id, str):
                raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
            return _parse_agent_request(req_id, method, params)
        return _parse_agent_notification(method, params)

    if "result" in raw:
        if set(raw.keys()) != {"jsonrpc", "id", "result"}:
            raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
        req_id = raw["id"]
        result = raw["result"]
        if (type(req_id) is not int and not isinstance(req_id, str)) or not isinstance(
            result, dict
        ):
            raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
        return ParsedRpcResponse(request_id=req_id, result=result)

    if "error" in raw:
        if set(raw.keys()) != {"jsonrpc", "id", "error"}:
            raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
        req_id = raw["id"]
        err = raw["error"]
        if (
            (type(req_id) is not int and not isinstance(req_id, str))
            or not isinstance(err, dict)
            or set(err.keys()) != {"code", "message"}
            or type(err.get("code")) is not int
            or not isinstance(err.get("message"), str)
        ):
            raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
        return ParsedRpcError(
            request_id=req_id,
            code=int(err["code"]),
            message=str(err["message"]),
        )

    raise HostSessionRejected(UNSUPPORTED_PROTOCOL)


def _parse_agent_request(
    req_id: int | str, method: str, params: dict[str, object]
) -> InboundAgentMessage:
    try:
        if method == "xuna/hello":
            protocols = params.get("protocols")
            if not isinstance(protocols, list) or "xtp/1" not in protocols:
                raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
            if "enroll" in params and "hostId" not in params:
                parsed_enroll = HelloEnrollParamsModel.model_validate(params)
                # Eagerly validate base64 and public key point
                parsed_enroll.enroll.code_bytes()
                parsed_enroll.enroll.public_key_bytes()
                return ParsedHelloEnroll(request_id=req_id, params=parsed_enroll)
            if "hostId" in params and "enroll" not in params:
                parsed_reconnect = HelloReconnectParamsModel.model_validate(params)
                parsed_reconnect.parsed_host_id()
                return ParsedHelloReconnect(request_id=req_id, params=parsed_reconnect)
            raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
        if method == "xuna/proof":
            parsed_proof = ProofParamsModel.model_validate(params)
            parsed_proof.signature_bytes()
            return ParsedProof(request_id=req_id, params=parsed_proof)
        if method == "ping":
            if params:
                raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
            return ParsedPing(request_id=req_id)
    except (ValidationError, ValueError) as error:
        if method == "xuna/hello" and "hostId" in params:
            raise HostSessionRejected(UNAUTHORIZED) from error
        raise HostSessionRejected(UNSUPPORTED_PROTOCOL) from error
    raise HostSessionRejected(UNSUPPORTED_PROTOCOL)


def _parse_agent_notification(method: str, params: dict[str, object]) -> InboundAgentMessage:
    try:
        if method == "notifications/tools/list_changed":
            if params:
                raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
            return ParsedToolsListChanged()
        if method == "notifications/progress":
            return ParsedProgress(params=ProgressParamsModel.model_validate(params))
        if method == "xuna/stop":
            return ParsedStop(params=StopParamsModel.model_validate(params))
        if method == "xuna/lanPeer":
            return ParsedLanPeer(params=LanPeerParamsModel.model_validate(params))
        if method == "xuna/presence":
            return ParsedPresence(params=PresenceParamsModel.model_validate(params))
        if method == "xuna/resources":
            return ParsedResources(params=ResourcesParamsModel.model_validate(params))
        if method == "xuna/focusRequest":
            return ParsedFocusRequest(params=FocusRequestParamsModel.model_validate(params))
        if method == "xuna/watchEvent":
            return ParsedWatchEvent(params=WatchEventParamsModel.model_validate(params))
    except (ValidationError, ValueError) as error:
        raise HostSessionRejected(UNSUPPORTED_PROTOCOL) from error
    raise HostSessionRejected(UNSUPPORTED_PROTOCOL)


_SHORT_CODE = r"^[a-z0-9_.-]{1,64}$"
_PREFLIGHT_STATUSES = Literal[
    "ok",
    "denied",
    "needs_permission",
    "invalid_args",
    "unsupported",
    "failed",
    "timeout",
    "cancelled",
]


class PreflightElementModel(_StrictModel):
    role: str = Field(pattern=_SHORT_CODE)
    name: str = Field(min_length=1, max_length=80, pattern=r"^[^\x00-\x1f\x7f]*$")


_PREVIEW_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_PREVIEW_SECRET_DIRS = frozenset(
    {".ssh", ".gnupg", ".aws", ".kube", ".password-store", "keychains"}
)
_PREVIEW_SECRET_EXACT = frozenset(
    {".npmrc", ".pypirc", ".pgpass", ".htpasswd", ".netrc", ".git-credentials"}
)
_PREVIEW_SECRET_EXTS = (".pem", ".key", ".p12", ".pfx", ".jks")
_PREVIEW_SECRET_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa")


class PreflightPreviewModel(_StrictModel):
    """Optional bounded read-only preview on `ok` `xuna/preflight` (§4.2.8, ADR-0019 §2)."""

    kind: Literal["diff", "paths"]
    summary: str = Field(min_length=1, max_length=160, pattern=r"^[^\x00-\x1f\x7f]*$")
    truncated: bool
    snippet: str | None = None
    paths: list[str] | None = None
    totalCount: int | None = None

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> "PreflightPreviewModel":
        if not self.summary.strip():
            raise ValueError("preview summary must not be blank")
        if self.kind == "diff":
            if self.snippet is None or self.paths is not None or self.totalCount is not None:
                raise ValueError("diff preview requires snippet and forbids paths/totalCount")
            if not self.snippet or any(
                (ord(ch) < 32 and ch not in ("\n", "\t")) or ord(ch) == 127 for ch in self.snippet
            ):
                raise ValueError("invalid diff preview snippet characters")
            if len(self.snippet.encode("utf-8")) > 2048:
                raise ValueError("diff preview snippet exceeds 2048 UTF-8 bytes")
            if len(self.snippet.split("\n")) > 40:
                raise ValueError("diff preview snippet exceeds 40 lines")
            if self.truncated and not self.snippet.endswith("\u2026"):
                raise ValueError("truncated diff preview snippet must end with ellipsis")
            return self
        if self.paths is None or self.totalCount is None or self.snippet is not None:
            raise ValueError("paths preview requires paths/totalCount and forbids snippet")
        if len(self.paths) > 20:
            raise ValueError("paths preview exceeds 20 entries")
        if self.totalCount < len(self.paths):
            raise ValueError("paths preview totalCount must be >= len(paths)")
        for raw_path in self.paths:
            if (
                not raw_path
                or len(raw_path) > 260
                or raw_path.strip() != raw_path
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in raw_path)
            ):
                raise ValueError("invalid paths preview entry")
            if raw_path.startswith(("/", "\\", "~")) or _PREVIEW_DRIVE_RE.match(raw_path):
                raise ValueError("paths preview entry must be workspace-relative")
            segs = [s for s in raw_path.replace("\\", "/").split("/") if s]
            if not segs or ".." in segs:
                raise ValueError("paths preview entry must not contain parent traversal")
            lower_segs = [s.lower() for s in segs]
            for idx, seg in enumerate(lower_segs):
                if seg in _PREVIEW_SECRET_DIRS or seg in _PREVIEW_SECRET_EXACT:
                    raise ValueError("paths preview entry must not contain secret path")
                nxt = lower_segs[idx + 1] if idx + 1 < len(lower_segs) else None
                if seg.startswith(".") and nxt in (
                    "config",
                    "config.json",
                    "credentials",
                    "credentials.json",
                ):
                    raise ValueError("paths preview entry must not contain credential path")
            leaf = lower_segs[-1]
            if (
                leaf == ".env"
                or leaf.startswith(".env.")
                or leaf.endswith(_PREVIEW_SECRET_EXTS)
                or leaf.startswith(_PREVIEW_SECRET_PREFIXES)
            ):
                raise ValueError("paths preview entry must not contain secret file")
        return self

    def to_wire_dict(self) -> dict[str, object]:
        if self.kind == "diff":
            return {
                "kind": "diff",
                "summary": self.summary.strip(),
                "snippet": self.snippet or "",
                "truncated": self.truncated,
            }
        return {
            "kind": "paths",
            "summary": self.summary.strip(),
            "paths": [p.replace("\\", "/") for p in (self.paths or [])],
            "totalCount": self.totalCount if self.totalCount is not None else 0,
            "truncated": self.truncated,
        }


class PreflightResultModel(_StrictModel):
    """`xuna/preflight` result body (`XUNA_XTP_V1_CONTRACT.md` §4.2.8)."""

    status: _PREFLIGHT_STATUSES
    effectiveRisk: Literal["R0", "R1", "R2", "R3"] | None = None
    reasonCode: str | None = Field(default=None, pattern=_SHORT_CODE)
    errorCode: str | None = Field(default=None, pattern=_SHORT_CODE)
    element: PreflightElementModel | None = None
    preview: PreflightPreviewModel | None = None


@dataclass(frozen=True, slots=True)
class HostPreflight:
    """What the host would enforce for a call (UDC-17 S17.1, ADR-0019 §2).

    `status == "ok"` carries `effective_risk`; any other status is a refusal with
    `error_code`. `element_name` and `preview` are ephemeral host content, so they
    may be shown to the owner on the approval card but never logged, audited, or persisted.
    """

    status: str
    effective_risk: RiskTier | None
    reason_code: str | None
    error_code: str | None
    element_role: str | None
    element_name: str | None
    preview: dict[str, object] | None = None


def parse_preflight_result(result: dict[str, object]) -> HostPreflight:
    """Strictly parses an `xuna/preflight` result; raises `ValueError` on any deviation."""
    if set(result.keys()) != {"preflight"} or not isinstance(result["preflight"], dict):
        raise ValueError("xuna/preflight result must contain only a 'preflight' object")
    try:
        body = PreflightResultModel.model_validate(result["preflight"])
    except ValidationError as error:
        raise ValueError("invalid xuna/preflight result") from error
    if body.status == "ok":
        if body.effectiveRisk is None or body.errorCode is not None:
            raise ValueError("an ok preflight carries a risk and no error code")
    elif (
        body.errorCode is None
        or body.effectiveRisk is not None
        or body.element is not None
        or body.preview is not None
    ):
        raise ValueError("a refused preflight carries an error code and nothing else")
    return HostPreflight(
        status=body.status,
        effective_risk=RiskTier(body.effectiveRisk) if body.effectiveRisk else None,
        reason_code=body.reasonCode,
        error_code=body.errorCode,
        element_role=body.element.role if body.element else None,
        element_name=body.element.name if body.element else None,
        preview=body.preview.to_wire_dict() if body.preview is not None else None,
    )


def parse_tools_list_result(result: dict[str, object]) -> list[ToolDescriptor]:
    raw_tools = result.get("tools")
    if set(result.keys()) != {"tools"} or not isinstance(raw_tools, list):
        raise ValueError("tools/list result must contain a 'tools' list")
    descriptors: list[ToolDescriptor] = []
    for item in raw_tools:
        if not isinstance(item, dict):
            raise ValueError("each tool in tools/list must be an object")
        descriptors.append(ToolDescriptor.from_wire(item))
    return descriptors


def parse_watch_register_params(params: dict[str, object]) -> WatchRegisterParamsModel:
    """Strictly validates `xuna/watchRegister` params; raises `ValueError` on deviation."""
    try:
        return WatchRegisterParamsModel.model_validate(params)
    except ValidationError as error:
        raise ValueError("invalid xuna/watchRegister params") from error


def parse_watch_register_result(result: dict[str, object]) -> WatchRegisterResultModel:
    """Strictly validates `xuna/watchRegister` result; raises `ValueError` on deviation."""
    try:
        return WatchRegisterResultModel.model_validate(result)
    except ValidationError as error:
        raise ValueError("invalid xuna/watchRegister result") from error


def parse_watch_cancel_params(params: dict[str, object]) -> WatchCancelParamsModel:
    """Strictly validates `xuna/watchCancel` params; raises `ValueError` on deviation."""
    try:
        return WatchCancelParamsModel.model_validate(params)
    except ValidationError as error:
        raise ValueError("invalid xuna/watchCancel params") from error


def parse_watch_cancel_result(result: dict[str, object]) -> WatchCancelResultModel:
    """Strictly validates `xuna/watchCancel` result; raises `ValueError` on deviation."""
    try:
        return WatchCancelResultModel.model_validate(result)
    except ValidationError as error:
        raise ValueError("invalid xuna/watchCancel result") from error


def build_watch_register_params(
    *,
    watch_id: str,
    job_id: str,
    mode: WatchModeWire,
    signals: list[WatchCompositeSignalWire] | None,
    target: WatchTargetModel,
    quiescence_seconds: int,
    min_wait_seconds: int,
    max_wait_seconds: int,
    require_initial_change: bool,
    release_focus: bool,
) -> dict[str, object]:
    """Builds and validates `xuna/watchRegister` JSON-RPC params, omitting `None` fields."""
    kwargs: dict[str, object] = {
        "schemaVersion": 1,
        "watchId": watch_id,
        "jobId": job_id,
        "mode": mode,
        "target": target,
        "quiescenceSeconds": quiescence_seconds,
        "minWaitSeconds": min_wait_seconds,
        "maxWaitSeconds": max_wait_seconds,
        "requireInitialChange": require_initial_change,
        "releaseFocus": release_focus,
    }
    if signals is not None:
        kwargs["signals"] = signals
    model = WatchRegisterParamsModel.model_validate(kwargs)
    return model.model_dump(mode="json", exclude_none=True)


def build_watch_cancel_params(
    *,
    reason: WatchCancelReasonWire,
    watch_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, object]:
    """Builds and validates `xuna/watchCancel` JSON-RPC params, omitting `None` fields."""
    kwargs: dict[str, object] = {"schemaVersion": 1, "reason": reason}
    if watch_id is not None:
        kwargs["watchId"] = watch_id
    if job_id is not None:
        kwargs["jobId"] = job_id
    model = WatchCancelParamsModel.model_validate(kwargs)
    return model.model_dump(mode="json", exclude_none=True)


def hello_response_frame(request_id: int | str) -> str:
    return _dump({"jsonrpc": "2.0", "id": request_id, "result": {"protocol": "xtp/1"}})


def proof_response_frame(request_id: int | str) -> str:
    return _dump({"jsonrpc": "2.0", "id": request_id, "result": {"verified": True}})


def ping_response_frame(request_id: int | str) -> str:
    return _dump({"jsonrpc": "2.0", "id": request_id, "result": {}})


def challenge_notification_frame(purpose: HostProofPurpose, nonce: bytes, expires_in_s: int) -> str:
    if len(nonce) != NONCE_SIZE:
        raise ValueError("nonce must be 32 bytes")
    return _dump(
        {
            "jsonrpc": "2.0",
            "method": "xuna/challenge",
            "params": {
                "purpose": int(purpose),
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "expiresIn": expires_in_s,
            },
        }
    )


def enroll_pending_notification_frame(expires_in_s: int) -> str:
    return _dump(
        {
            "jsonrpc": "2.0",
            "method": "xuna/enrollPending",
            "params": {"expiresIn": expires_in_s},
        }
    )


def enrolled_notification_frame(host_id: HostId, alias: str) -> str:
    return _dump(
        {
            "jsonrpc": "2.0",
            "method": "xuna/enrolled",
            "params": {"hostId": str(host_id), "alias": alias},
        }
    )


def ready_notification_frame(server_time_iso: str, heartbeat_s: int) -> str:
    return _dump(
        {
            "jsonrpc": "2.0",
            "method": "xuna/ready",
            "params": {"serverTime": server_time_iso, "heartbeatS": heartbeat_s},
        }
    )


def lan_peer_notification_frame(phone_public_keys_b64: list[str]) -> str:
    return _dump(
        {
            "jsonrpc": "2.0",
            "method": "xuna/lanPeer",
            "params": {"phonePublicKeys": phone_public_keys_b64},
        }
    )


def rpc_request_frame(request_id: int | str, method: str, params: dict[str, object]) -> str:
    return _dump({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})


def rpc_notification_frame(method: str, params: dict[str, object]) -> str:
    return _dump({"jsonrpc": "2.0", "method": method, "params": params})


def _dump(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def validate_structured_content_size(
    structured_content: Mapping[str, object],
    *,
    allow_media: bool = False,
) -> tuple[bool, int, int]:
    """Returns `(within_limit, byte_size, limit_bytes)` for `tools/call` `structuredContent`."""
    is_media = allow_media and has_media_structured_content(structured_content)
    limit = MAX_MEDIA_STRUCTURED_CONTENT_BYTES if is_media else MAX_STRUCTURED_CONTENT_BYTES
    if structured_content_exceeds_limit(structured_content, allow_media=allow_media):
        try:
            byte_size = structured_content_byte_size(structured_content)
        except (TypeError, ValueError):
            byte_size = limit + 1
        return False, byte_size, limit
    return True, structured_content_byte_size(structured_content), limit
