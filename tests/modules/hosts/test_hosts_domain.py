"""Domain and golden-fixture tests for the `hosts` module (UDC-3 S3.2)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from xuna_backend.modules.hosts.application.xtp_frames import (
    ParsedFocusRequest,
    ParsedHelloEnroll,
    ParsedHelloReconnect,
    ParsedPresence,
    ParsedResources,
    ParsedRpcResponse,
    parse_agent_frame,
    parse_preflight_result,
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
    Host,
    HostAlias,
    HostId,
    HostKind,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    default_host_kind_for_platform,
    fold_text,
    normalize_nicknames,
    normalize_place,
    normalize_wol_mac,
    validate_tool_name,
)
from xuna_backend.modules.hosts.domain.session import HostSessionRejected
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
        "Mac\n",
        "\nMac",
        "\tMac",
        "Mac\r",
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

    assert canonical_json({"zero": -0.0, "one": 1.0}) == '{"one":1,"zero":0}'
    assert canonical_json({"zero": 0, "one": 1}) == '{"one":1,"zero":0}'

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_json({"bad_float": bad})


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

    # UDC-17 S17.1: the preflight frames the Dart builder must produce parse on this side.
    ok_frame = frames["preflightOkResponse"]
    denied_frame = frames["preflightDeniedResponse"]
    assert isinstance(ok_frame, dict) and isinstance(denied_frame, dict)
    for frame in (ok_frame, denied_frame):
        parsed_frame = parse_agent_frame(json.dumps(frame))
        assert isinstance(parsed_frame, ParsedRpcResponse)
    ok = parse_preflight_result(ok_frame["result"])
    assert ok.effective_risk == RiskTier.R3
    assert ok.element_name == "Gửi báo cáo"
    denied = parse_preflight_result(denied_frame["result"])
    assert denied.status == "denied"
    assert denied.error_code == "password_field_forbidden"
    assert denied.effective_risk is None


def test_host_kind_and_fold_text_and_spatial_validation() -> None:
    assert HostKind.DESKTOP == "desktop"
    assert HostKind.MOBILE == "mobile"
    assert HostKind.SERVER == "server"
    assert HostKind.BRIDGE == "bridge"

    assert default_host_kind_for_platform(HostPlatform.WINDOWS) == HostKind.DESKTOP
    assert default_host_kind_for_platform(HostPlatform.MACOS) == HostKind.DESKTOP
    assert default_host_kind_for_platform(HostPlatform.LINUX) == HostKind.DESKTOP
    assert default_host_kind_for_platform(HostPlatform.ANDROID) == HostKind.MOBILE
    assert default_host_kind_for_platform(HostPlatform.IOS) == HostKind.MOBILE

    assert fold_text("  Máy   của   Thảo  ") == "may cua thao"
    assert fold_text("ĐÈN PHÒNG NGỦ") == "den phong ngu"
    assert fold_text("điện thoại") == "dien thoai"

    now = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
    pub = b"\x04" + bytes(range(1, 65))
    host = Host(
        host_id=HostId.generate(),
        owner_id=HostId.generate().value,
        alias=HostAlias("Máy của Thảo"),
        platform=HostPlatform.MACOS,
        hostname="macbook",
        os_version="15.1",
        agent_version="0.1.0",
        public_key=pub,
        status=HostStatus.ACTIVE,
        created_at=now,
        wol_mac="aa:bb:cc:dd:ee:ff",
        host_kind=HostKind.DESKTOP,
        place="  office  ",
        nicknames=("  mac nha  ", "studio"),
    )
    assert host.wol_mac == "AA:BB:CC:DD:EE:FF"
    assert host.place == "office"
    assert host.nicknames == ("mac nha", "studio")
    assert host.alias_folded == "may cua thao"

    base_kwargs: dict[str, object] = {
        "host_id": HostId.generate(),
        "owner_id": HostId.generate().value,
        "alias": HostAlias("Mac"),
        "platform": HostPlatform.MACOS,
        "hostname": "macbook",
        "os_version": "15.1",
        "agent_version": "0.1.0",
        "public_key": pub,
        "status": HostStatus.ACTIVE,
        "created_at": now,
    }
    for bad_kw in (
        {"place": ""},
        {"place": "   "},
        {"place": "x" * 33},
        {"place": "room\n1"},
        {"place": "office\n"},
        {"place": "\toffice"},
        {"place": "office\r"},
        {"nicknames": tuple(f"n{i}" for i in range(11))},
        {"nicknames": ("   ",)},
        {"nicknames": ("n" * 25,)},
        {"nicknames": ("bad\x00nick",)},
        {"nicknames": ("nick\n",)},
        {"wol_mac": "not-a-mac"},
        {"wol_mac": "AA-BB-CC-DD-EE-FF"},
        {"wol_mac": "AA:BB:CC:DD:EE:GG"},
        {"wol_mac": "AA:BB:CC:DD:EE:FF\n"},
        {"alias_folded": "x" * 33},
        {"alias_folded": "mac\n"},
    ):
        with pytest.raises(ValueError):
            Host(**{**base_kwargs, **bad_kw})  # type: ignore[arg-type]

    for bad_place in ("office\n", "\toffice", "office\r"):
        with pytest.raises(ValueError):
            normalize_place(bad_place)
    for bad_nicks in (("nick\n",), ("\tnick",), ("nick\r",)):
        with pytest.raises(ValueError):
            normalize_nicknames(bad_nicks)
    for bad_mac in ("AA:BB:CC:DD:EE:FF\n", "\tAA:BB:CC:DD:EE:FF", "AA:BB:CC:DD:EE:FF\r"):
        with pytest.raises(ValueError):
            normalize_wol_mac(bad_mac)


def test_mesh_frames_golden_fixture_and_negative_validation() -> None:
    raw_mesh = json.loads((FIXTURES / "mesh_frames.json").read_text(encoding="utf-8"))
    frames = raw_mesh["frames"]
    assert isinstance(frames, dict)

    enroll_linux = parse_agent_frame(json.dumps(frames["helloEnrollLinuxRequest"]))
    assert isinstance(enroll_linux, ParsedHelloEnroll)
    assert enroll_linux.params.enroll.parsed_platform() == HostPlatform.LINUX

    reconnect_linux = parse_agent_frame(json.dumps(frames["helloReconnectLinuxRequest"]))
    assert isinstance(reconnect_linux, ParsedHelloReconnect)
    assert reconnect_linux.params.parsed_platform() == HostPlatform.LINUX

    presence = parse_agent_frame(json.dumps(frames["presenceNotification"]))
    assert isinstance(presence, ParsedPresence)
    assert presence.params.idleSeconds == 12
    assert presence.params.screenLocked is False
    assert presence.params.mediaPlaying is True
    assert presence.params.batteryPercent == 78
    assert presence.params.isCharging is True
    assert presence.params.foregroundAppId == "vscode"
    assert presence.params.parsed_updated_at() == datetime(2026, 11, 20, 16, 0, 0, tzinfo=UTC)

    no_batt = parse_agent_frame(json.dumps(frames["presenceDesktopNoBatteryNotification"]))
    assert isinstance(no_batt, ParsedPresence)
    assert no_batt.params.batteryPercent is None
    assert no_batt.params.isCharging is None
    assert no_batt.params.foregroundAppId is None

    resources = parse_agent_frame(json.dumps(frames["resourcesNotification"]))
    assert isinstance(resources, ParsedResources)
    assert resources.params.installedApps == ["vscode", "obsidian"]
    assert resources.params.workspaces == ["xuna_pet", "Notes_Vault"]
    assert resources.params.capabilitiesVersion == "r1_0001"

    focus_hotkey = parse_agent_frame(json.dumps(frames["focusRequestHotkeyNotification"]))
    assert isinstance(focus_hotkey, ParsedFocusRequest)
    assert focus_hotkey.params.source == "hotkey"
    assert focus_hotkey.params.followMe is True

    focus_tray = parse_agent_frame(json.dumps(frames["focusRequestTrayNotification"]))
    assert isinstance(focus_tray, ParsedFocusRequest)
    assert focus_tray.params.source == "tray"
    assert focus_tray.params.followMe is None

    # Negative matrix: every malformed / unstrict mesh frame must fail closed
    base_presence = {
        "schemaVersion": 1,
        "idleSeconds": 0,
        "screenLocked": False,
        "mediaPlaying": False,
        "updatedAt": "2026-11-20T16:00:00Z",
    }
    bad_frames: list[dict[str, object]] = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "xuna/presence",
            "params": base_presence,
        },  # notification with id
        {
            "jsonrpc": "2.0",
            "method": "xuna/presence",
            "params": {**base_presence, "extraField": 1},
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/presence",
            "params": {**base_presence, "schemaVersion": 2},
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/presence",
            "params": {**base_presence, "idleSeconds": -1},
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/presence",
            "params": {**base_presence, "batteryPercent": 101},
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/presence",
            "params": {**base_presence, "batteryPercent": True},
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/presence",
            "params": {**base_presence, "subnetHash": "NOT_HEX"},
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/presence",
            "params": {**base_presence, "foregroundAppId": "Bad App"},
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/presence",
            "params": {**base_presence, "updatedAt": "2026-11-20T16:00:00"},
        },  # naive
        {
            "jsonrpc": "2.0",
            "method": "xuna/resources",
            "params": {
                "schemaVersion": 1,
                "installedApps": ["App Upper"],
                "workspaces": [],
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/resources",
            "params": {
                "schemaVersion": 1,
                "installedApps": [],
                "workspaces": ["/Users/alice/secret"],
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/resources",
            "params": {
                "schemaVersion": 1,
                "installedApps": [],
                "workspaces": ["repo\n"],
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/resources",
            "params": {
                "schemaVersion": 1,
                "installedApps": [],
                "workspaces": ["."],
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/resources",
            "params": {
                "schemaVersion": 1,
                "installedApps": [],
                "workspaces": [".."],
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/resources",
            "params": {
                "schemaVersion": 1,
                "installedApps": [],
                "workspaces": ["foo..bar"],
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/resources",
            "params": {
                "schemaVersion": 1,
                "installedApps": [f"a{i}" for i in range(257)],
                "workspaces": [],
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "xuna/focusRequest",
            "params": {"schemaVersion": 1, "source": "cli"},
        },
    ]
    for bad in bad_frames:
        with pytest.raises(HostSessionRejected):
            parse_agent_frame(json.dumps(bad))
