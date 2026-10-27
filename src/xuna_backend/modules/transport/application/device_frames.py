"""Wire frames of the device-direct session (contract draft "Device connection").

Device frames are strict: an unknown field, type or schema version, malformed
JSON or a wrong value closes the session with `unsupported_protocol`.
"""

import base64
import binascii
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from xuna_backend.modules.transport.domain.device_session import (
    UNSUPPORTED_PROTOCOL,
    SessionRejected,
)

U32 = Annotated[int, Field(ge=0, le=0xFFFFFFFF)]
B64 = Annotated[str, Field(min_length=4, max_length=128, pattern=r"^[A-Za-z0-9+/]+={0,2}$")]


class _Frame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schemaVersion: Literal[1]


class HelloFrame(_Frame):
    type: Literal["hello"]
    petId: str = Field(pattern=r"^[0-9a-f]{32}$")
    # Same charset as audit metadata values, so it can be recorded as a code.
    firmware: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_.:-]+$")
    capabilities: list[Annotated[str, Field(pattern=r"^[a-z0-9_.]{1,32}$")]] = Field(max_length=8)


class ProofFrame(_Frame):
    type: Literal["proof"]
    generation: U32
    publicKey: B64 | None = None
    signature: B64


class PingFrame(_Frame):
    type: Literal["ping"]
    seq: U32


class ProjectionFrame(_Frame):
    """Accepted from DC-5; until then a projection closes the session."""

    type: Literal["projection"]
    generation: U32
    export: dict[str, object]


DeviceFrame = Annotated[
    HelloFrame | ProofFrame | PingFrame | ProjectionFrame, Field(discriminator="type")
]
_DEVICE_FRAME: TypeAdapter[HelloFrame | ProofFrame | PingFrame | ProjectionFrame] = TypeAdapter(
    DeviceFrame
)


def parse_device_frame(text: str) -> HelloFrame | ProofFrame | PingFrame | ProjectionFrame:
    try:
        return _DEVICE_FRAME.validate_json(text)
    except (ValidationError, ValueError) as error:
        raise SessionRejected(UNSUPPORTED_PROTOCOL) from error


def decode_b64(value: str, size: int) -> bytes:
    """Standard base64 of exactly `size` bytes, else `unsupported_protocol`."""
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise SessionRejected(UNSUPPORTED_PROTOCOL) from error
    if len(raw) != size:
        raise SessionRejected(UNSUPPORTED_PROTOCOL)
    return raw


def challenge_frame(nonce: bytes, expires_in_s: int) -> str:
    return _dump(
        {
            "type": "challenge",
            "schemaVersion": 1,
            "nonce": base64.b64encode(nonce).decode(),
            "expiresIn": expires_in_s,
        }
    )


def ready_frame(server_time: str, heartbeat_s: int) -> str:
    return _dump(
        {"type": "ready", "schemaVersion": 1, "serverTime": server_time, "heartbeatS": heartbeat_s}
    )


def pong_frame(seq: int) -> str:
    return _dump({"type": "pong", "schemaVersion": 1, "seq": seq})


def _dump(frame: dict[str, object]) -> str:
    return json.dumps(frame, separators=(",", ":"))
