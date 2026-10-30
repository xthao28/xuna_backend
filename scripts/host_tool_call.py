"""Dev CLI to invoke a tool on an enrolled Host Agent (UDC-3 §5.4).

Requires the backend to run with `XUNA_DEV_HOST_TOOLS=1` (never enabled in
production).

Usage:
    python scripts/host_tool_call.py --host "PC phòng làm việc" --tool media.play_pause --args "{}"
"""

import argparse
import json
import os
import sys
import uuid
from typing import Any

import httpx


def resolve_host_id(
    client: httpx.Client,
    base_url: str,
    token: str,
    host_alias_or_id: str,
) -> str:
    resp = client.get(
        f"{base_url.rstrip('/')}/v1/hosts",
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    items: list[dict[str, Any]] = resp.json().get("items", [])
    for item in items:
        if item.get("hostId") == host_alias_or_id or item.get("alias") == host_alias_or_id:
            return str(item["hostId"])
    raise LookupError(f"Host not found by alias or id: {host_alias_or_id!r}")


def invoke_host_tool(
    base_url: str,
    token: str,
    host_alias_or_id: str,
    tool_name: str,
    arguments: dict[str, object],
    *,
    deadline_ms: int = 10_000,
    approve: bool = True,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    owns_client = client is None
    http = client or httpx.Client(timeout=max(deadline_ms / 1000.0 + 5.0, 15.0))
    try:
        host_id = resolve_host_id(http, base_url, token, host_alias_or_id)
        resp = http.post(
            f"{base_url.rstrip('/')}/v1/hosts/{host_id}/tools:invoke",
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": str(uuid.uuid4()),
            },
            json={
                "schemaVersion": 1,
                "tool": tool_name,
                "arguments": arguments,
                "deadlineMs": deadline_ms,
                "approve": approve,
            },
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result
    finally:
        if owns_client:
            http.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Invoke an XTP v1 tool on an enrolled host.")
    parser.add_argument(
        "--base-url", default=os.environ.get("XUNA_API_BASE_URL", "http://localhost:8000")
    )
    parser.add_argument("--token", default=os.environ.get("XUNA_OWNER_TOKEN", "dev-token"))
    parser.add_argument("--host", required=True, help="Host alias or UUID")
    parser.add_argument("--tool", required=True, help="Tool name (e.g. media.play_pause)")
    parser.add_argument("--args", default="{}", help="JSON object string of arguments")
    parser.add_argument("--deadline-ms", type=int, default=10_000)
    parser.add_argument("--no-approve", action="store_true", help="Omit dev approval token")
    parsed = parser.parse_args(argv)

    raw_args = json.loads(parsed.args)
    if not isinstance(raw_args, dict):
        raise SystemExit("--args must decode to a JSON object")

    output = invoke_host_tool(
        base_url=parsed.base_url,
        token=parsed.token,
        host_alias_or_id=parsed.host,
        tool_name=parsed.tool,
        arguments=raw_args,
        deadline_ms=parsed.deadline_ms,
        approve=not parsed.no_approve,
    )
    sys.stdout.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
