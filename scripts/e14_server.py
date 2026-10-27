"""E14 WSS echo server: the board's TLS + WebSocket test peer (DC-1, ADR-0010).

Standard library only (asyncio, ssl, RFC 6455 framing), so the experiment adds
no dependency. Every text or binary frame the board sends is echoed back
unchanged. Logs are content-free: peer address, frame counts and sizes, never
payloads.

    .venv/Scripts/python.exe scripts/e14_server.py --port 8765
    .venv/Scripts/python.exe scripts/e14_server.py --port 8765 --drop-every 120 --down-for 10
    .venv/Scripts/python.exe scripts/e14_server.py --self-test

`--drop-every N` simulates a server kill every N seconds: it aborts every
connection, stops listening for `--down-for` seconds, then listens again. The
board must reconnect by itself within 2 minutes after the server returns.

Certificates come from `make_dev_ca.py` (default `~/.xuna/dev-ca`). The server
listens on all interfaces by default so the board can reach it over the LAN;
Windows needs an inbound firewall rule for the port (see the E14 runbook).
"""

import argparse
import asyncio
import base64
import datetime
import hashlib
import os
import ssl
import struct
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import make_dev_ca

GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_HEAD = 4096
MAX_PAYLOAD = 4096
HEAD_TIMEOUT_S = 10.0
IDLE_TIMEOUT_S = 90.0
OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA
CLOSE_NORMAL, CLOSE_GOING_AWAY, CLOSE_PROTOCOL, CLOSE_UNSUPPORTED, CLOSE_TOO_BIG = (
    1000,
    1001,
    1002,
    1003,
    1009,
)


CLOSE_ABNORMAL = 1006  # logged only: the peer vanished without a close frame


class ProtocolError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def expect(condition: bool) -> None:
    if not condition:
        raise AssertionError("unexpected result")


def log(line: str) -> None:
    stamp = datetime.datetime.now().astimezone().strftime("%H:%M:%S")
    sys.stdout.write(f"{stamp} {line}\n")
    sys.stdout.flush()


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1(key.encode("ascii") + GUID).digest()).decode("ascii")  # noqa: S324 (RFC 6455)


def encode_frame(opcode: int, payload: bytes, *, mask: bool) -> bytes:
    head = bytearray([0x80 | opcode])
    size = len(payload)
    flag = 0x80 if mask else 0
    if size < 126:
        head.append(flag | size)
    elif size < 1 << 16:
        head.append(flag | 126)
        head += struct.pack("!H", size)
    else:
        head.append(flag | 127)
        head += struct.pack("!Q", size)
    if not mask:
        return bytes(head) + payload
    key = os.urandom(4)
    return bytes(head) + key + bytes(b ^ key[i % 4] for i, b in enumerate(payload))


async def read_frame(
    reader: asyncio.StreamReader, *, expect_mask: bool, wait_s: float
) -> tuple[int, bytes]:
    """One complete frame; fragmented messages are not used by the board."""
    first, second = await asyncio.wait_for(reader.readexactly(2), wait_s)
    if first & 0x70:
        raise ProtocolError(CLOSE_PROTOCOL)
    opcode, masked, size = first & 0x0F, bool(second & 0x80), second & 0x7F
    if not first & 0x80 or opcode == OP_CONT:
        raise ProtocolError(CLOSE_UNSUPPORTED)
    if masked != expect_mask:
        raise ProtocolError(CLOSE_PROTOCOL)
    if size == 126:
        (size,) = struct.unpack("!H", await reader.readexactly(2))
    elif size == 127:
        (size,) = struct.unpack("!Q", await reader.readexactly(8))
    if size > MAX_PAYLOAD:
        raise ProtocolError(CLOSE_TOO_BIG)
    key = await reader.readexactly(4) if masked else b""
    payload = await reader.readexactly(size)
    if masked:
        payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def close_payload(code: int) -> bytes:
    return struct.pack("!H", code)


@dataclass
class Stats:
    connections: int = 0
    active: int = 0
    frames: int = 0
    bytes: int = 0
    rejected: int = 0


@dataclass
class EchoServer:
    path: str
    context: ssl.SSLContext
    stats: Stats = field(default_factory=Stats)
    writers: set[asyncio.StreamWriter] = field(default_factory=set)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        host = peer[0] if peer else "?"
        started, frames, size, code = time.monotonic(), 0, 0, CLOSE_NORMAL
        self.writers.add(writer)
        try:
            if not await self._upgrade(reader, writer):
                self.stats.rejected += 1
                return
            self.stats.connections += 1
            self.stats.active += 1
            log(f"open peer={host} active={self.stats.active}")
            try:
                while True:
                    opcode, payload = await read_frame(
                        reader, expect_mask=True, wait_s=IDLE_TIMEOUT_S
                    )
                    if opcode == OP_CLOSE:
                        writer.write(encode_frame(OP_CLOSE, payload[:2], mask=False))
                        await writer.drain()
                        break
                    if opcode == OP_PING:
                        writer.write(encode_frame(OP_PONG, payload, mask=False))
                    elif opcode in (OP_TEXT, OP_BINARY):
                        writer.write(encode_frame(opcode, payload, mask=False))
                        frames += 1
                        size += len(payload)
                    await writer.drain()
            except ProtocolError as error:
                code = error.code
                writer.write(encode_frame(OP_CLOSE, close_payload(code), mask=False))
                await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionError):
                code = CLOSE_ABNORMAL
            except TimeoutError:
                code = CLOSE_GOING_AWAY
                writer.write(encode_frame(OP_CLOSE, close_payload(code), mask=False))
                await writer.drain()
            finally:
                self.stats.active -= 1
                self.stats.frames += frames
                self.stats.bytes += size
                log(
                    f"close peer={host} code={code} frames={frames} bytes={size} "
                    f"seconds={time.monotonic() - started:.0f}"
                )
        except (ConnectionError, asyncio.IncompleteReadError, ssl.SSLError, TimeoutError):
            pass
        finally:
            self.writers.discard(writer)
            writer.close()

    async def _upgrade(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT_S)
        except (asyncio.LimitOverrunError, asyncio.IncompleteReadError, TimeoutError):
            return False
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        headers = {
            name.strip().lower(): value.strip()
            for name, _, value in (line.partition(":") for line in lines[1:] if line)
        }
        if len(parts) != 3 or parts[0] != "GET" or parts[1] != self.path:
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            return False
        key = headers.get("sec-websocket-key", "")
        if (
            headers.get("upgrade", "").lower() != "websocket"
            or "upgrade" not in headers.get("connection", "").lower()
            or headers.get("sec-websocket-version") != "13"
            or len(base64.b64decode(key, validate=False) if key else b"") != 16
        ):
            writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            return False
        writer.write(
            b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            + f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n".encode("ascii")
        )
        await writer.drain()
        return True

    def abort_all(self) -> None:
        for writer in list(self.writers):
            writer.transport.abort()


def server_context(certs: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certs / "server.pem", certs / "server.key")
    return context


async def serve(args: argparse.Namespace) -> None:
    echo = EchoServer(args.path, server_context(args.certs))
    drops = 0

    async def listen() -> asyncio.Server:
        return await asyncio.start_server(
            echo.handle, args.host, args.port, ssl=echo.context, limit=MAX_HEAD
        )

    server = await listen()
    log(f"listening wss://{args.host}:{args.port}{args.path} certs={args.certs}")
    last_summary = time.monotonic()
    next_drop = time.monotonic() + args.drop_every if args.drop_every else None
    while True:
        await asyncio.sleep(1)
        now = time.monotonic()
        if next_drop is not None and now >= next_drop:
            drops += 1
            server.close()
            echo.abort_all()
            await server.wait_closed()
            log(f"drop n={drops} down_s={args.down_for}")
            await asyncio.sleep(args.down_for)
            server = await listen()
            log(f"back n={drops}")
            next_drop = time.monotonic() + args.drop_every
        if now - last_summary >= 60:
            s = echo.stats
            log(
                f"summary active={s.active} connections={s.connections} frames={s.frames} "
                f"bytes={s.bytes} rejected={s.rejected} drops={drops}"
            )
            last_summary = now


# --- self-test: real TLS on 127.0.0.1 with a throwaway CA -------------------------------


async def _client(
    port: int, path: str, context: ssl.SSLContext
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", port, ssl=context, server_hostname="127.0.0.1"
    )
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    writer.write(
        f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
        f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n".encode("ascii")
    )
    await writer.drain()
    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
    if not head.startswith(b"HTTP/1.1 101") or accept_key(key).encode() not in head:
        raise AssertionError("upgrade refused")
    return reader, writer


async def self_test(work: Path) -> list[str]:
    """Return the names of failed checks (empty = all passed)."""
    now = datetime.datetime.now(datetime.UTC)
    ca = make_dev_ca.issue_ca("XUNA E14 self-test CA", now)
    server_cert = make_dev_ca.issue_server(ca, ["127.0.0.1"], [], now)
    make_dev_ca.write_bundle(work, ca, server_cert, write_ca=True)
    other = make_dev_ca.issue_ca("Unrelated CA", now)
    (work / "other.pem").write_bytes(make_dev_ca.cert_pem(other.cert))

    echo = EchoServer("/e14", server_context(work))
    server = await asyncio.start_server(echo.handle, "127.0.0.1", 0, ssl=echo.context)
    port = server.sockets[0].getsockname()[1]
    pinned = ssl.create_default_context(cafile=str(work / "ca.pem"))
    failures: list[str] = []

    async def check(name: str, body: Callable[[], Awaitable[None]]) -> None:
        try:
            await asyncio.wait_for(body(), 10)
            sys.stdout.write(f"PASS  {name}\n")
        except Exception as error:  # every failure is reported, none hidden
            failures.append(name)
            sys.stdout.write(f"FAIL  {name} ({type(error).__name__})\n")

    async def echo_round_trip() -> None:
        reader, writer = await _client(port, "/e14", pinned)
        for seq in range(3):
            payload = f"e14 seq={seq} ".encode().ljust(1024, b".")
            writer.write(encode_frame(OP_TEXT, payload, mask=True))
            await writer.drain()
            opcode, got = await read_frame(reader, expect_mask=False, wait_s=5)
            expect(opcode == OP_TEXT and got == payload)
        writer.write(encode_frame(OP_PING, b"hb", mask=True))
        await writer.drain()
        expect(await read_frame(reader, expect_mask=False, wait_s=5) == (OP_PONG, b"hb"))
        writer.write(encode_frame(OP_CLOSE, close_payload(CLOSE_NORMAL), mask=True))
        await writer.drain()
        opcode, _ = await read_frame(reader, expect_mask=False, wait_s=5)
        expect(opcode == OP_CLOSE)
        writer.close()

    async def unknown_ca_refused() -> None:
        context = ssl.create_default_context(cafile=str(work / "other.pem"))
        try:
            await _client(port, "/e14", context)
        except ssl.SSLCertVerificationError:
            return
        raise AssertionError("connected with an unrelated CA")

    async def wrong_name_refused() -> None:
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", port, ssl=pinned, server_hostname="not-this-host.lan"
            )
        except ssl.SSLCertVerificationError:
            return
        writer.close()
        raise AssertionError("accepted a certificate for another name")

    async def expect_close(frame: bytes, code: int) -> None:
        reader, writer = await _client(port, "/e14", pinned)
        writer.write(frame)
        await writer.drain()
        opcode, payload = await read_frame(reader, expect_mask=False, wait_s=5)
        expect(opcode == OP_CLOSE and payload == close_payload(code))
        writer.close()

    async def wrong_path_refused() -> None:
        try:
            await _client(port, "/other", pinned)
        except AssertionError:
            return
        raise AssertionError("upgraded on a wrong path")

    await check("echo 3 x 1 KB text frames, ping/pong, close", echo_round_trip)
    await check("unrelated CA refused (pinning)", unknown_ca_refused)
    await check("certificate name checked", wrong_name_refused)
    await check(
        "unmasked client frame closed 1002",
        lambda: expect_close(encode_frame(OP_TEXT, b"x", mask=False), CLOSE_PROTOCOL),
    )
    await check(
        "oversize frame closed 1009",
        lambda: expect_close(
            encode_frame(OP_TEXT, b"x" * (MAX_PAYLOAD + 1), mask=True), CLOSE_TOO_BIG
        ),
    )
    await check("wrong path refused", wrong_path_refused)
    server.close()
    echo.abort_all()
    await server.wait_closed()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0", help="listen address (default all)")  # noqa: S104 - the board connects over the LAN
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--path", default="/e14")
    parser.add_argument("--certs", type=Path, default=make_dev_ca.DEFAULT_OUT)
    parser.add_argument("--drop-every", type=int, default=0, help="seconds; 0 = never")
    parser.add_argument("--down-for", type=int, default=10, help="seconds offline per drop")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        with tempfile.TemporaryDirectory() as work:
            failures = asyncio.run(self_test(Path(work)))
        sys.stdout.write("ALL PASS\n" if not failures else f"{len(failures)} FAILED\n")
        return 1 if failures else 0
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
