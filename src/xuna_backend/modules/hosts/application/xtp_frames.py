"""Strict JSON-RPC 2.0 frame parsing and builders for XTP v1 (UDC-3 §5.2)."""

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    ToolDescriptor,
)
from xuna_backend.modules.hosts.domain.session import (
    FRAME_TOO_LARGE,
    UNAUTHORIZED,
    UNSUPPORTED_PROTOCOL,
    HostSessionRejected,
)

MAX_FRAME_BYTES_DEFAULT = 65_536


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EnrollHelloPayload(_StrictModel):
    code: str = Field(min_length=20, max_length=28)
    publicKey: str = Field(min_length=80, max_length=96)
    platform: Literal["windows", "macos"]
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
    platform: Literal["windows", "macos", "android", "ios"]
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
    except (ValidationError, ValueError) as error:
        raise HostSessionRejected(UNSUPPORTED_PROTOCOL) from error
    raise HostSessionRejected(UNSUPPORTED_PROTOCOL)


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


def rpc_request_frame(request_id: int | str, method: str, params: dict[str, object]) -> str:
    return _dump({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})


def rpc_notification_frame(method: str, params: dict[str, object]) -> str:
    return _dump({"jsonrpc": "2.0", "method": method, "params": params})


def _dump(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
