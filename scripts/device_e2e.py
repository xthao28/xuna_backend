"""Phone-free end-to-end check: simulated board + app-shaped requests → real server.

Starts uvicorn as a subprocess against `--database-url`, with a throwaway local
OIDC issuer (JWKS served from this script) in place of Firebase, then runs the
bind → consent → conversation flow over HTTP exactly as the mobile app does:

- the device side speaks the firmware Identity V2 frames (40-byte request,
  141-byte READ value, `firmware/lib/xuna_core/include/xuna/identity/DeviceProof.h`)
  and signs with a software P-256 key;
- request bodies mirror `mobile/lib/features/account/data/repository/
  cloud_repository_impl.dart` (same fields, standard base64, fresh
  Idempotency-Key per mutation, `schemaVersion: 1`).

Every run uses a new random PetId and owner subjects, so it never collides with
real bindings; it does add owner/pet/audit rows to that database (use a dev or
test database, never production). The text provider comes from the usual
`XUNA_TEXT_*` variables (`echo` when unset). Prompts are synthetic.

    .venv/bin/python scripts/device_e2e.py \
        --database-url postgresql+psycopg://$(whoami)@localhost/xuna_local

Exit code 0 only when every step passes. This does not exercise BLE, the
Flutter UI or Firebase; those still need the phone and the board.
"""

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

ROOT = Path(__file__).resolve().parents[1]
AUDIENCE = "xuna-device-e2e"
KID = "device-e2e"
PROMPT = "Kể cho mình nghe một câu chuyện ngắn về con mèo."

# Firmware DeviceProof.h constants.
REQUEST_SIZE = 40
RESPONSE_SIZE = 141
PURPOSE_BIND = 1
STATUS_OK = 2
STATUS_INVALID_REQUEST = 3
STATUS_UNSUPPORTED_PURPOSE = 4


class SimulatedBoard:
    """Software stand-in for firmware `ProofService` + mbedTLS signer."""

    def __init__(self, pet_id: str, generation: int) -> None:
        self.pet_id = pet_id
        self.generation = generation
        self._key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self._key.public_key().public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )

    def answer(self, request: bytes) -> bytes:
        """BLE WRITE of a 40-byte request, then the 141-byte READ value."""
        status, signature = STATUS_INVALID_REQUEST, bytes(64)
        purpose = request[2] if len(request) > 2 else 0
        request_id = struct.unpack_from("<I", request, 4)[0] if len(request) >= 8 else 0
        well_formed = (
            len(request) == REQUEST_SIZE
            and request[0] == 1
            and request[1] == REQUEST_SIZE
            and request[3] == 0
            and request_id != 0
        )
        if well_formed and purpose != PURPOSE_BIND:
            status = STATUS_UNSUPPORTED_PURPOSE
        elif well_formed:
            message = (
                b"xuna.proof.v1\x00"
                + request[8:40]
                + bytes.fromhex(self.pet_id)
                + struct.pack("<BI", PURPOSE_BIND, self.generation)
                + hashlib.sha256(self.public_key).digest()
            )
            r, s = decode_dss_signature(self._key.sign(message, ec.ECDSA(hashes.SHA256())))
            status, signature = STATUS_OK, r.to_bytes(32, "big") + s.to_bytes(32, "big")
        signed = status == STATUS_OK
        return (
            bytes([1, RESPONSE_SIZE, purpose, status])
            + struct.pack("<II", request_id, self.generation if signed else 0)
            + (self.public_key if signed else bytes(65))
            + signature
        )


def encode_request(request_id: int, nonce: bytes) -> bytes:
    """Mobile `DeviceProofCodec.encodeBindRequest`."""
    if len(nonce) != 32:
        raise ValueError("nonce must be 32 bytes")
    return bytes([1, REQUEST_SIZE, PURPOSE_BIND, 0]) + struct.pack("<I", request_id) + nonce


def decode_response(frame: bytes, request_id: int) -> tuple[int, bytes, bytes]:
    """Mobile `DeviceProofCodec.decodeResponse` + repository status check."""
    if len(frame) != RESPONSE_SIZE or frame[0] != 1 or frame[1] != RESPONSE_SIZE:
        raise ValueError("malformed proof frame")
    status = frame[3]
    answered, generation = struct.unpack_from("<II", frame, 4)
    if answered != request_id or status != STATUS_OK or frame[2] != PURPOSE_BIND:
        raise ValueError(f"device proof status {status}")
    public_key, signature = frame[12:77], frame[77:141]
    if public_key[0] != 0x04 or not any(signature):
        raise ValueError("incomplete device proof")
    return generation, public_key, signature


class LocalIssuer:
    """Throwaway RS256 issuer whose JWKS is served on localhost."""

    def __init__(self) -> None:
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self._key.public_key()))
        body = json.dumps({"keys": [{**jwk, "kid": KID, "use": "sig", "alg": "RS256"}]})
        payload = body.encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def token(self, subject: str) -> str:
        now = int(time.time())
        claims = {"iss": self.url, "aud": AUDIENCE, "sub": subject, "iat": now, "exp": now + 600}
        return jwt.encode(claims, self._key, algorithm="RS256", headers={"kid": KID})

    def close(self) -> None:
        self._server.shutdown()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def sse_events(response: httpx.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    event, data = "message", []
    for line in response.iter_lines():
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))


class Run:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.failures += 0 if ok else 1
        sys.stdout.write(f"{'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}\n")
        return ok


def flow(api: httpx.Client, issuer: LocalIssuer, run: Run) -> None:
    owner = {"Authorization": f"Bearer {issuer.token(f'e2e-owner-{uuid.uuid4()}')}"}
    intruder = {"Authorization": f"Bearer {issuer.token(f'e2e-intruder-{uuid.uuid4()}')}"}
    pet_id = uuid.uuid4().hex  # device-issued UUIDv4 in the firmware format
    board = SimulatedBoard(pet_id, generation=39)

    def idem() -> dict[str, str]:
        return {"Idempotency-Key": str(uuid.uuid4())}

    me = api.get("/v1/me", headers=owner)
    run.check("GET /v1/me provisions the owner", me.status_code == 200, f"{me.status_code}")

    consents = api.get("/v1/consents", headers=owner)
    body = consents.json()
    run.check(
        "consents default to U0 (all off)",
        consents.status_code == 200
        and not (body["aggregateMetrics"] or body["experienceEvents"] or body["conversationText"]),
    )

    def bind(headers: dict[str, str], request_id: int) -> httpx.Response:
        challenge = api.post(
            "/v1/devices/challenges",
            json={"schemaVersion": 1, "petId": pet_id},
            headers={**headers, **idem()},
        )
        challenge.raise_for_status()
        data = challenge.json()
        frame = board.answer(encode_request(request_id, base64.b64decode(data["nonce"])))
        generation, public_key, signature = decode_response(frame, request_id)
        return api.post(
            "/v1/pets",
            json={
                "schemaVersion": 1,
                "petId": pet_id,
                "origin": "migrated_from_v1",
                "snapshotGeneration": generation,
                "deviceProof": {
                    "challengeId": data["challengeId"],
                    "publicKey": base64.b64encode(public_key).decode(),
                    "signature": base64.b64encode(signature).decode(),
                },
            },
            headers={**headers, **idem()},
        )

    bad = board.answer(bytes([1, REQUEST_SIZE, 2, 0]) + struct.pack("<I", 7) + bytes(32))
    run.check("board refuses a projection proof", bad[3] == STATUS_UNSUPPORTED_PURPOSE)

    bound = bind(owner, request_id=1)
    run.check("bind with board proof → 201", bound.status_code == 201, bound.text[:120])
    again = bind(owner, request_id=2)
    run.check("re-bind same owner + key → 200", again.status_code == 200, f"{again.status_code}")
    taken = bind(intruder, request_id=3)
    run.check(
        "other owner → 409 pet_already_bound",
        taken.status_code == 409 and taken.json().get("code") == "pet_already_bound",
    )
    pets = api.get("/v1/pets", headers=owner).json()["items"]
    run.check(
        "GET /v1/pets lists the pet with the signed generation",
        any(p["petId"] == pet_id and p["lastGeneration"] == 39 for p in pets),
    )
    hidden = api.get(f"/v1/pets/{pet_id}", headers=intruder)
    run.check("other owner cannot see the pet (404)", hidden.status_code == 404)

    turn_body = {"schemaVersion": 1, "text": PROMPT, "locale": "vi"}
    blocked = api.post(f"/v1/conversations/{pet_id}/turns", json=turn_body, headers=owner)
    run.check(
        "turn without U3 → 403 consent_required",
        blocked.status_code == 403 and blocked.json().get("code") == "consent_required",
    )

    updated = api.put(
        "/v1/consents",
        json={
            "schemaVersion": 1,
            "aggregateMetrics": False,
            "experienceEvents": False,
            "conversationText": True,
            "expectedVersion": body["version"],
        },
        headers={**owner, **idem()},
    )
    run.check(
        "U3 on (expectedVersion)",
        updated.status_code == 200 and updated.json()["conversationText"] is True,
        updated.text[:120],
    )

    reply, finished, error = [], None, None
    with api.stream(
        "POST",
        f"/v1/conversations/{pet_id}/turns",
        json=turn_body,
        headers={**owner, "Accept": "text/event-stream"},
        timeout=40,
    ) as stream:
        if run.check(
            "turn opens an SSE stream", stream.status_code == 200, f"{stream.status_code}"
        ):
            for event, data in sse_events(stream):
                if event == "delta":
                    reply.append(data["text"])
                elif event == "finished":
                    finished = data
                elif event == "error":
                    error = data
    text = "".join(reply)
    run.check(
        "reply streams deltas then finished",
        finished is not None and bool(text),
        f"error={error['code']}" if error else f"{len(text)} chars, {finished}",
    )
    if text:
        sys.stdout.write(f"      reply: {text}\n")

    foreign = api.post(f"/v1/conversations/{uuid.uuid4().hex}/turns", json=turn_body, headers=owner)
    run.check("turn for an unbound pet → 404", foreign.status_code == 404)

    trail = api.get("/v1/audit-events", headers=owner).json()["items"]
    actions = [event["action"] for event in trail]
    run.check(
        "audit trail has pet.bound and consent.changed",
        "pet.bound" in actions and "consent.changed" in actions,
        ", ".join(actions),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--database-url", required=True, help="dev/test PostgreSQL (rows are added)"
    )
    args = parser.parse_args()

    issuer = LocalIssuer()
    port = free_port()
    env = {
        **os.environ,
        "XUNA_ENV": "local",
        "XUNA_DATABASE_URL": args.database_url,
        "XUNA_OIDC_ISSUER": issuer.url,
        "XUNA_OIDC_AUDIENCE": AUDIENCE,
        "XUNA_OIDC_JWKS_URL": f"{issuer.url}/jwks.json",
        "XUNA_FIREBASE_PROJECT_ID": "",
    }
    bin_dir = Path(sys.executable).parent
    subprocess.run(  # noqa: S603 - fixed argv, dev-only script
        [str(bin_dir / "alembic"), "upgrade", "head"], cwd=ROOT, env=env, check=True
    )
    server = subprocess.Popen(  # noqa: S603 - fixed argv, dev-only script
        [
            str(bin_dir / "uvicorn"),
            "xuna_backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
    )
    run = Run()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15) as api:
            deadline = time.monotonic() + 20
            while True:
                try:
                    if api.get("/readyz").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline or server.poll() is not None:
                    run.check("server ready", False, "no /readyz 200 within 20 s")
                    return 1
                time.sleep(0.2)
            run.check("server ready (/readyz)", True)
            flow(api, issuer, run)
    finally:
        server.terminate()
        server.wait(timeout=10)
        issuer.close()
    sys.stdout.write(f"\n{'ALL PASS' if run.failures == 0 else f'{run.failures} FAILED'}\n")
    return 0 if run.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
