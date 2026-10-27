"""Run the API with uvicorn on any machine, including Windows (DC-D8 dev server).

psycopg's async mode cannot use Windows' default ProactorEventLoop, which
plain `uvicorn` picks there, so this sets the Selector loop first. TLS is for
the device-direct LAN test: pass the certificate from `make_dev_ca.py`.

    .venv/Scripts/python.exe scripts/serve.py --host 0.0.0.0 --port 8443 \
        --tls ~/.xuna/dev-ca

Configuration still comes from `XUNA_*` environment variables. The access log
stays off (the app writes its own content-free one).
"""

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tls", type=Path, help="folder with server.pem and server.key")
    args = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    tls: dict[str, str] = {}
    if args.tls:
        tls = {
            "ssl_certfile": str(args.tls.expanduser() / "server.pem"),
            "ssl_keyfile": str(args.tls.expanduser() / "server.key"),
        }
    uvicorn.run(
        "xuna_backend.main:app",
        host=args.host,
        port=args.port,
        loop="asyncio",
        ws="websockets",
        access_log=False,
        **tls,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
