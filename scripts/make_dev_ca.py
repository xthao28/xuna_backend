"""Make the LAN dev CA and a server certificate for device-direct tests (E14, DC-D8).

The board pins this CA (ADR-0010). The CA is reused when it already exists, so
re-issuing the server certificate for a new LAN address needs no reflash; pass
`--new-ca` to start over (then reflash the firmware with the new header).

Private keys are written to `--out` (default `~/.xuna/dev-ca`, outside every
repository) and must never be committed. The firmware header holds only the
public CA certificate, but it is machine-specific, so it is git-ignored too.

    .venv/Scripts/python.exe scripts/make_dev_ca.py --ip 192.168.1.20 \
        --firmware-header ../firmware/include/E14DevCa.h

mbedTLS 2.28 (ESP-IDF 4.4) matches the server name only against dNSName
entries, so every `--ip` is written both as an iPAddress SAN (standard, used by
Python and phones) and as a dNSName holding the same dotted string (used by the
board).
"""

import argparse
import datetime
import hashlib
import ipaddress
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

DEFAULT_OUT = Path.home() / ".xuna" / "dev-ca"
CA_DAYS = 3650
SERVER_DAYS = 825


@dataclass(frozen=True)
class Issued:
    cert: x509.Certificate
    key: ec.EllipticCurvePrivateKey


def _name(common_name: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "XUNA dev"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )


def issue_ca(common_name: str, now: datetime.datetime) -> Issued:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = _name(common_name)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=CA_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    return Issued(cert, key)


def issue_server(
    ca: Issued, ips: list[str], dns_names: list[str], now: datetime.datetime
) -> Issued:
    if not ips and not dns_names:
        raise ValueError("at least one --ip or --dns is required")
    names: list[x509.GeneralName] = []
    for ip in ips:
        names.append(x509.IPAddress(ipaddress.ip_address(ip)))
        names.append(x509.DNSName(str(ipaddress.ip_address(ip))))
    names.extend(x509.DNSName(name) for name in dns_names)
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name((ips or dns_names)[0]))
        .issuer_name(ca.cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=SERVER_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.key.public_key()),
            critical=False,
        )
        .sign(ca.key, hashes.SHA256())
    )
    return Issued(cert, key)


def fingerprint(cert: x509.Certificate) -> str:
    return hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


def cert_pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def _write_private(path: Path, key: ec.EllipticCurvePrivateKey) -> None:
    data = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    path.write_bytes(data)
    path.chmod(0o600)  # best effort; Windows keeps its own ACLs


def write_bundle(out: Path, ca: Issued, server: Issued, *, write_ca: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if write_ca:
        _write_private(out / "ca.key", ca.key)
        (out / "ca.pem").write_bytes(cert_pem(ca.cert))
    _write_private(out / "server.key", server.key)
    (out / "server.pem").write_bytes(cert_pem(server.cert))


def load_ca(out: Path) -> Issued | None:
    cert_path, key_path = out / "ca.pem", out / "ca.key"
    if not cert_path.exists() or not key_path.exists():
        return None
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"{key_path} is not an EC private key")
    return Issued(cert, key)


def firmware_header(ca: x509.Certificate, now: datetime.datetime) -> str:
    pem = cert_pem(ca).decode("ascii")
    return (
        f"// Generated by backend/scripts/make_dev_ca.py on {now.date().isoformat()}.\n"
        "// Pinned LAN dev CA for E14 / device-direct tests (ADR-0010, DC-D8).\n"
        "// Public certificate only, but machine-specific: git-ignored, never commit.\n"
        "#pragma once\n"
        f'#define XUNA_E14_CA_FINGERPRINT "{fingerprint(ca)[:16]}"\n'
        f'static const char kE14CaPem[] = R"PEM({pem})PEM";\n'
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ip", action="append", default=[], help="LAN IP of the dev machine")
    parser.add_argument("--dns", action="append", default=[], help="host name of the dev machine")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"default {DEFAULT_OUT}")
    parser.add_argument("--new-ca", action="store_true", help="replace an existing CA")
    parser.add_argument("--firmware-header", type=Path, help="write the pinned-CA C++ header here")
    args = parser.parse_args()

    now = datetime.datetime.now(datetime.UTC)
    ca = None if args.new_ca else load_ca(args.out)
    created_ca = ca is None
    if ca is None:
        ca = issue_ca(f"XUNA dev CA ({socket.gethostname()})", now)
    try:
        server = issue_server(ca, args.ip, args.dns, now)
    except ValueError as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    write_bundle(args.out, ca, server, write_ca=created_ca)
    if args.firmware_header:
        args.firmware_header.write_text(firmware_header(ca.cert, now), encoding="ascii")

    sans = ", ".join([*args.ip, *args.dns])
    sys.stdout.write(
        f"CA {'created' if created_ca else 'reused'}: {args.out / 'ca.pem'}\n"
        f"  SHA-256 {fingerprint(ca.cert)}\n"
        f"server certificate for {sans}: {args.out / 'server.pem'}\n"
        f"private keys: {args.out / 'ca.key'}, {args.out / 'server.key'} (keep out of git)\n"
    )
    if args.firmware_header:
        sys.stdout.write(f"firmware header: {args.firmware_header}\n")
    if created_ca and not args.firmware_header:
        sys.stdout.write("new CA: rerun with --firmware-header to update the board's pin\n")
    if os.name == "nt":
        sys.stdout.write("Windows: private key files are not ACL-restricted; keep --out private\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
