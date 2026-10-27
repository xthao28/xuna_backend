"""E14 dev tooling: pinned dev CA and the WSS echo server (DC-1, ADR-0010)."""

import asyncio
import datetime
import ipaddress
import sys
from pathlib import Path

from cryptography import x509

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import e14_server
import make_dev_ca

NOW = datetime.datetime(2026, 10, 27, tzinfo=datetime.UTC)


def test_server_certificate_names_ip_for_python_and_mbedtls() -> None:
    ca = make_dev_ca.issue_ca("test CA", NOW)
    server = make_dev_ca.issue_server(ca, ["192.168.1.20"], ["xuna-dev.local"], NOW)
    san = server.cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.IPAddress) == [ipaddress.ip_address("192.168.1.20")]
    # mbedTLS 2.28 matches dNSName only, so the dotted IP is also a dNSName.
    assert san.get_values_for_type(x509.DNSName) == ["192.168.1.20", "xuna-dev.local"]
    assert server.cert.issuer == ca.cert.subject
    constraints = ca.cert.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert constraints.ca and constraints.path_length == 0


def test_server_certificate_needs_a_name() -> None:
    ca = make_dev_ca.issue_ca("test CA", NOW)
    try:
        make_dev_ca.issue_server(ca, [], [], NOW)
    except ValueError:
        return
    raise AssertionError("issued a certificate without a name")


def test_ca_is_reused_and_header_pins_it(tmp_path: Path) -> None:
    ca = make_dev_ca.issue_ca("test CA", NOW)
    make_dev_ca.write_bundle(
        tmp_path, ca, make_dev_ca.issue_server(ca, ["10.0.0.2"], [], NOW), write_ca=True
    )
    loaded = make_dev_ca.load_ca(tmp_path)
    assert loaded is not None and loaded.cert == ca.cert
    header = make_dev_ca.firmware_header(ca.cert, NOW)
    assert make_dev_ca.fingerprint(ca.cert)[:16] in header
    assert make_dev_ca.cert_pem(ca.cert).decode() in header
    assert "PRIVATE KEY" not in header


def test_frame_round_trip_masked_and_extended_length() -> None:
    async def run() -> None:
        for size in (0, 125, 126, 1024, 4096):
            payload = bytes(range(256)) * (size // 256) + bytes(size % 256)
            reader = asyncio.StreamReader()
            reader.feed_data(e14_server.encode_frame(e14_server.OP_TEXT, payload, mask=True))
            assert await e14_server.read_frame(reader, expect_mask=True, wait_s=1) == (
                e14_server.OP_TEXT,
                payload,
            )

    asyncio.run(run())


def test_accept_key_rfc6455_example() -> None:
    assert e14_server.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_self_test_passes_over_real_tls(tmp_path: Path) -> None:
    assert asyncio.run(e14_server.self_test(tmp_path)) == []
