"""Domain and golden-fixture tests for the `hosts` module (UDC-3 S3.2)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from xuna_backend.modules.hosts.application.xtp_frames import (
    parse_agent_frame,
    parse_tools_list_result,
)
from xuna_backend.modules.hosts.domain.crypto import (
    HostProofPurpose,
    args_sha256,
    canonical_json,
    decode_b64url_16b,
    host_proof_message,
    public_key_fingerprint_b64url,
    sas,
)
from xuna_backend.modules.hosts.domain.entities import (
    FORBIDDEN_TOOLS,
    HostAlias,
    HostId,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    validate_tool_name,
)
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "xtp"


def _load_golden() -> dict[str, object]:
    raw: dict[str, object] = json.loads(
        (FIXTURES / "golden_vectors.json").read_text(encoding="utf-8")
    )
    return raw


def _load_sample_frames() -> dict[str, object]:
    raw: dict[str, object] = json.loads(
        (FIXTURES / "sample_frames.json").read_text(encoding="utf-8")
    )
    return raw


@pytest.mark.parametrize(
    "raw_alias,expected",
    [
        ("PC phòng làm việc", "PC phòng làm việc"),
        ("  Mac của Thảo  ", "Mac của Thảo"),
        ("Work-PC_01", "Work-PC_01"),
        ("A" * 24, "A" * 24),
    ],
)
def test_valid_host_aliases(raw_alias: str, expected: str) -> None:
    alias = HostAlias(raw_alias)
    assert alias.value == expected


@pytest.mark.parametrize(
    "bad_alias",
    [
        "",
        "   ",
        "A" * 25,
        "PC/bad",
        "PC;drop",
        "PC\nnewline",
    ],
)
def test_invalid_host_aliases_rejected(bad_alias: str) -> None:
    with pytest.raises(ValueError):
        HostAlias(bad_alias)


@pytest.mark.parametrize(
    "forbidden",
    [
        *sorted(FORBIDDEN_TOOLS),
        "shell.custom",
        "powershell.invoke",
        "osascript.apple",
        "script.python",
        "eval.js",
        "keyboard.type",
        "pointer.click",
        "unknown.tool",
        "INVALID.TOOL",
        "media",
    ],
)
def test_forbidden_and_malformed_tool_names_rejected(forbidden: str) -> None:
    with pytest.raises(ValueError):
        validate_tool_name(forbidden)


def test_golden_sas_and_fingerprint_vectors() -> None:
    golden = _load_golden()
    sas_vec = golden["sas"]
    assert isinstance(sas_vec, dict)

    code_bytes = bytes.fromhex(str(sas_vec["codeHex"]))
    pub_bytes = bytes.fromhex(str(sas_vec["publicKeyHex"]))

    assert decode_b64url_16b(str(sas_vec["codeB64Url"])) == code_bytes
    assert public_key_fingerprint_b64url(pub_bytes) == str(sas_vec["keyFingerprintB64Url"])
    assert sas(code_bytes, pub_bytes) == str(sas_vec["sas"])


def test_golden_host_proof_vectors_and_p256_verification() -> None:
    golden = _load_golden()
    hp = golden["hostProof"]
    assert isinstance(hp, dict)

    pub_bytes = bytes.fromhex(str(hp["publicKeyHex"]))
    nonce = bytes.fromhex(str(hp["nonceHex"]))
    verifier = P256SignatureVerifier()

    enroll_vec = hp["enrollProof"]
    assert isinstance(enroll_vec, dict)
    enroll_payload = bytes.fromhex(str(enroll_vec["payloadHex"]))
    enroll_msg = host_proof_message(
        nonce=nonce,
        purpose=HostProofPurpose.ENROLL,
        payload=enroll_payload,
    )
    assert enroll_msg.hex() == str(enroll_vec["messageHex"])
    enroll_sig = bytes.fromhex(str(enroll_vec["signatureHex"]))
    assert verifier.verify(pub_bytes, enroll_msg, enroll_sig)

    session_vec = hp["sessionProof"]
    assert isinstance(session_vec, dict)
    host_id = HostId.from_str(str(session_vec["hostIdUuid"]))
    session_msg = host_proof_message(
        nonce=nonce,
        purpose=HostProofPurpose.SESSION,
        payload=pub_bytes,
        host_id=host_id,
    )
    assert session_msg.hex() == str(session_vec["messageHex"])
    session_sig = bytes.fromhex(str(session_vec["signatureHex"]))
    assert verifier.verify(pub_bytes, session_msg, session_sig)
    assert not verifier.verify(pub_bytes, session_msg + b"x", session_sig)


def test_golden_canonical_json_and_args_sha256() -> None:
    golden = _load_golden()
    vectors = golden["canonicalJson"]
    assert isinstance(vectors, list)
    for item in vectors:
        assert isinstance(item, dict)
        inp = item["input"]
        assert isinstance(inp, dict)
        assert canonical_json(inp) == item["canonical"]
        assert args_sha256(inp) == item["argsSha256"]

    with pytest.raises(ValueError):
        canonical_json({"bad_float": 1.25})


def test_sample_frames_parse_cleanly() -> None:
    sample = _load_sample_frames()
    frames = sample["frames"]
    assert isinstance(frames, dict)

    for key in (
        "helloEnrollRequest",
        "helloReconnectRequest",
        "proofEnrollRequest",
        "proofSessionRequest",
        "toolsListResponse",
        "toolsCallOkResponse",
        "toolsCallErrorResponse",
        "progressNotification",
        "toolsListChangedNotification",
        "stopNotification",
        "pingRequest",
        "pingResponse",
        "jsonRpcMethodNotFoundError",
    ):
        text = json.dumps(frames[key])
        parsed = parse_agent_frame(text)
        assert parsed is not None

    tools_list_resp = frames["toolsListResponse"]
    assert isinstance(tools_list_resp, dict)
    result = tools_list_resp["result"]
    assert isinstance(result, dict)
    descriptors = parse_tools_list_result(result)
    assert len(descriptors) == 3
    manifest = Manifest(
        host_id=HostId.generate(),
        tools=tuple(descriptors),
        updated_at=datetime.now(UTC),
    )
    assert manifest.find_tool("media.play_pause") is not None
    assert manifest.find_tool("app.launch") is not None
    assert manifest.find_tool("app.launch").risk == RiskTier.R2  # type: ignore[union-attr]
    assert manifest.find_tool("host.status").family == ToolFamily.HOST  # type: ignore[union-attr]
    for desc in descriptors:
        assert ToolDescriptor.from_wire(desc.to_wire()) == desc
