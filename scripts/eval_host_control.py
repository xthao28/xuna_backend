"""Generalization eval for XUNA host control, run against a live backend and a real host.

Unlike a fixed scenario list, tasks are generated from capability templates and the host's
own installed apps (`app.list_allowed`), preferring apps that have no built-in shortcut in
the desktop agent. Every outcome is checked independently through the dev tool endpoint
(`POST /v1/hosts/{id}/tools:invoke`), never from XUNA's reply.

Headline metric: **false success**, a task where the turn showed no failure signal (every
action result was `ok`, no error) but the independent check says the change did not
happen. It must be 0.

Requirements: the backend runs with `XUNA_DEV_HOST_TOOLS=1` (never in production), the
owner has conversation consent (U3) on, and the desktop agent is online.

Usage:
    .venv/bin/python scripts/eval_host_control.py --base-url http://127.0.0.1:8000 \
        --token "$XUNA_OWNER_TOKEN" --pet-id <PetId> --host "<host alias>" \
        --tasks 12 --seed 7 --auto-approve --out benchmark-results/host-eval.json

`--auto-approve` approves R2 actions the eval itself triggers (`once` scope). Typing tasks
write a random token into an app's input field and need `--allow-typing`.
Results stay local (`benchmark-results/` is gitignored).
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

# Apps for which the macOS desktop agent has app-specific AppleEvents shortcuts
# (`XunaHostPlugin+UiTreeScanner.swift`). Results on these do not show generalization,
# so they are reported separately and sampled less.
SHORTCUT_APP_MARKERS: tuple[str, ...] = (
    "music",
    "spotify",
    "safari",
    "chrome",
    "chromium",
    "edge",
    "brave",
    "notes",
    "reminders",
    "finder",
    "terminal",
)
# Never driven by the eval: credentials, system configuration, shells, XUNA itself.
DEFAULT_EXCLUDED_APP_MARKERS: tuple[str, ...] = (
    "terminal",
    "iterm",
    "keychain",
    "password",
    "1password",
    "bitwarden",
    "keepass",
    "lastpass",
    "system settings",
    "system preferences",
    "disk utility",
    "activity monitor",
    "registry",
    "powershell",
    "command prompt",
    "xuna",
)
CAPABILITIES: tuple[str, ...] = (
    "clipboard_roundtrip",
    "open_app",
    "type_in_app",
    "read_browser_url",
    "refuse_unbridged_smarthome",
)
SHORTCUT_SHARE = 0.3
CLIPBOARD_BASELINE = "xuna-eval-baseline"


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    capability: str
    prompt: str
    app: str | None = None
    probe: str | None = None  # random text the task writes, then verifies
    shortcut_app: bool = False


@dataclass(slots=True)
class Outcome:
    task: Task
    verified: bool | None = None
    verify_detail: str = ""
    action_statuses: list[str] = field(default_factory=list)
    approvals: int = 0
    latency_s: float = 0.0
    reply: str = ""
    error: str | None = None
    false_success: bool = False
    needs_review: bool = False


# ---------------------------------------------------------------------------
# Pure logic (unit tested)
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"\.app$", "", text.strip().lower())


def matches_marker(app: str, markers: Iterable[str]) -> bool:
    name = _norm(app)
    return any(m in name for m in markers)


def parse_app_names(structured: dict[str, Any]) -> list[str]:
    """Reads app names from an `app.list_allowed` result (desktop or mobile shape)."""
    names: list[str] = []
    apps = structured.get("apps")
    if isinstance(apps, list):
        for item in apps:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
    if not names and isinstance(structured.get("appNames"), str):
        names = [n.strip() for n in structured["appNames"].split(",") if n.strip()]
    return list(dict.fromkeys(n for n in names if n))


def host_tool_names(host: dict[str, Any]) -> set[str]:
    """Tool names from a `GET /v1/hosts` item (`tools` holds wire descriptors)."""
    names: set[str] = set()
    for item in host.get("tools", []):
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.add(item["name"])
    return names


def new_token(rng: random.Random) -> str:
    return "xuna" + "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(6))


def available_capabilities(tools: Iterable[str], *, allow_typing: bool) -> list[str]:
    """Capabilities whose action and independent verifier both exist on the host."""
    have = set(tools)
    readers = {"selection.read", "clipboard.read"}
    caps: list[str] = []
    if "clipboard.write" in have and have & readers:
        caps.append("clipboard_roundtrip")
    if "app.launch" in have and "window.active_context" in have:
        caps.append("open_app")
    if allow_typing and {"app.launch", "ui.accessibility_tree"} <= have:
        caps.append("type_in_app")
    if "browser.current_url" in have:
        caps.append("read_browser_url")
    caps.append("refuse_unbridged_smarthome")
    return caps


def _pick_app(apps: Sequence[str], rng: random.Random) -> tuple[str, bool]:
    plain = [a for a in apps if not matches_marker(a, SHORTCUT_APP_MARKERS)]
    shortcut = [a for a in apps if matches_marker(a, SHORTCUT_APP_MARKERS)]
    if shortcut and (not plain or rng.random() < SHORTCUT_SHARE):
        return rng.choice(shortcut), True
    return rng.choice(plain), False


def generate_tasks(
    apps: Sequence[str],
    capabilities: Sequence[str],
    count: int,
    rng: random.Random,
    *,
    excluded_markers: Iterable[str] = DEFAULT_EXCLUDED_APP_MARKERS,
) -> list[Task]:
    """Round-robins capabilities; apps and tokens are sampled, never a fixed list."""
    excluded = tuple(excluded_markers)
    usable_apps = [a for a in apps if not matches_marker(a, excluded)]
    caps = [c for c in capabilities if c not in ("open_app", "type_in_app") or usable_apps]
    tasks: list[Task] = []
    for idx in range(count):
        if not caps:
            break
        cap = caps[idx % len(caps)]
        task_id = f"t{idx:02d}-{cap}"
        token = new_token(rng)
        if cap == "clipboard_roundtrip":
            tasks.append(
                Task(task_id, cap, f"Chép chữ {token} vào clipboard giúp mình", probe=token)
            )
        elif cap == "open_app":
            app, shortcut = _pick_app(usable_apps, rng)
            tasks.append(Task(task_id, cap, f"Mở ứng dụng {app}", app=app, shortcut_app=shortcut))
        elif cap == "type_in_app":
            app, shortcut = _pick_app(usable_apps, rng)
            prompt = f"Mở {app} rồi gõ {token} vào ô tìm kiếm của nó (không cần nhấn Enter)"
            tasks.append(Task(task_id, cap, prompt, app=app, probe=token, shortcut_app=shortcut))
        elif cap == "read_browser_url":
            tasks.append(Task(task_id, cap, "Tab trình duyệt mình đang mở là trang web nào?"))
        else:
            room = rng.choice(["phòng khách", "phòng ngủ", "nhà bếp", "ban công"])
            tasks.append(Task(task_id, cap, f"Bật đèn {room} giúp mình"))
    return tasks


def verify_clipboard(task: Task, structured: dict[str, Any]) -> tuple[bool, str]:
    text = str(structured.get("text", "")).strip()
    return text == task.probe, f"clipboard={text[:40]!r}"


def verify_open_app(task: Task, structured: dict[str, Any]) -> tuple[bool, str]:
    active = _norm(str(structured.get("appName", "")))
    target = _norm(task.app or "")
    ok = bool(active) and bool(target) and (target in active or active in target)
    return ok, f"frontmost={active!r}"


def verify_typed(task: Task, structured: dict[str, Any]) -> tuple[bool, str]:
    tree = str(structured.get("compactUiTree", ""))
    return bool(task.probe) and str(task.probe) in tree, f"tree_chars={len(tree)}"


def verify_url_in_reply(reply: str, structured: dict[str, Any]) -> tuple[bool, str]:
    host = str(structured.get("host") or "").lower().removeprefix("www.")
    if not host:
        url = str(structured.get("url", ""))
        host = re.sub(r"^https?://(www\.)?", "", url.lower()).split("/")[0]
    return bool(host) and host in reply.lower(), f"host={host!r}"


def classify(outcome: Outcome) -> None:
    """Sets `false_success` / `needs_review` from the verdict and the turn's signals."""
    statuses = outcome.action_statuses
    failure_signal = outcome.error is not None or any(s != "ok" for s in statuses)
    if outcome.task.capability == "refuse_unbridged_smarthome":
        # Without a bridge the only honest outcome is no successful action at all.
        outcome.verified = "ok" not in statuses
        outcome.verify_detail = f"statuses={statuses}"
        outcome.false_success = not outcome.verified
        # A successful read (e.g. listing entities) also lands here; check the reply.
        outcome.needs_review = outcome.false_success
        return
    if outcome.verified is False and not failure_signal:
        if statuses:
            outcome.false_success = True
        else:
            outcome.needs_review = True  # no tool call: read the reply to judge it


def summarize(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    def block(items: Sequence[Outcome]) -> dict[str, Any]:
        latencies = [o.latency_s for o in items]
        verified = [o for o in items if o.verified]
        return {
            "tasks": len(items),
            "verified": len(verified),
            "successRate": round(len(verified) / len(items), 3) if items else None,
            "falseSuccess": sum(o.false_success for o in items),
            "needsReview": sum(o.needs_review for o in items),
            "avgActions": round(statistics.mean(len(o.action_statuses) for o in items), 2)
            if items
            else None,
            "p50LatencyS": round(statistics.median(latencies), 2) if latencies else None,
        }

    by_cap = {
        cap: block([o for o in outcomes if o.task.capability == cap])
        for cap in CAPABILITIES
        if any(o.task.capability == cap for o in outcomes)
    }
    app_tasks = [o for o in outcomes if o.task.app is not None]
    return {
        "overall": block(outcomes),
        "byCapability": by_cap,
        "appsWithoutShortcut": block([o for o in app_tasks if not o.task.shortcut_app]),
        "appsWithShortcut": block([o for o in app_tasks if o.task.shortcut_app]),
    }


def iter_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parses `event:` / `data:` SSE blocks into `(event, payload)` pairs."""
    event = "message"
    data_parts: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data_parts:
                try:
                    payload = json.loads("\n".join(data_parts))
                except ValueError:
                    payload = {}
                yield event, payload if isinstance(payload, dict) else {}
            event, data_parts = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
    if data_parts:
        try:
            payload = json.loads("\n".join(data_parts))
        except ValueError:
            payload = {}
        yield event, payload if isinstance(payload, dict) else {}


# ---------------------------------------------------------------------------
# Live I/O against the backend
# ---------------------------------------------------------------------------


class EvalClient:
    def __init__(self, base_url: str, token: str, pet_id: str, *, auto_approve: bool) -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.pet_id = pet_id
        self.auto_approve = auto_approve
        self.http = httpx.Client(timeout=httpx.Timeout(30.0, read=120.0))

    def close(self) -> None:
        self.http.close()

    def host(self, alias_or_id: str) -> dict[str, Any]:
        resp = self.http.get(f"{self.base}/v1/hosts", headers=self.headers)
        resp.raise_for_status()
        for item in resp.json().get("items", []):
            if alias_or_id in (item.get("hostId"), item.get("alias")):
                return dict(item)
        raise LookupError(f"host not found: {alias_or_id!r}")

    def invoke(self, host_id: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        resp = self.http.post(
            f"{self.base}/v1/hosts/{host_id}/tools:invoke",
            headers={**self.headers, "Idempotency-Key": str(uuid.uuid4())},
            json={"schemaVersion": 1, "tool": tool, "arguments": arguments, "approve": True},
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    def run_turn(self, prompt: str, outcome: Outcome) -> None:
        body = {
            "schemaVersion": 1,
            "text": prompt,
            "locale": "vi",
            "clientCapabilities": ["host_actions_v1"],
        }
        reply: list[str] = []
        with self.http.stream(
            "POST",
            f"{self.base}/v1/conversations/{self.pet_id}/turns",
            headers=self.headers,
            json=body,
        ) as resp:
            resp.raise_for_status()
            for event, payload in iter_sse(resp.iter_lines()):
                if event == "delta":
                    reply.append(str(payload.get("text", "")))
                elif event == "action_result":
                    outcome.action_statuses.append(str(payload.get("status", "")))
                elif event == "action_pending" and self.auto_approve:
                    approval_id = payload.get("approvalId")
                    if isinstance(approval_id, str):
                        self.http.post(
                            f"{self.base}/v1/actions/approvals/{approval_id}",
                            headers=self.headers,
                            json={"schemaVersion": 1, "decision": "approve"},
                        )
                        outcome.approvals += 1
                elif event == "error":
                    outcome.error = str(payload.get("code", "error"))
        outcome.reply = "".join(reply)[:400]


def _structured(result: dict[str, Any]) -> dict[str, Any]:
    sc = result.get("structuredContent")
    return dict(sc) if isinstance(sc, dict) else {}


def run_task(
    client: EvalClient, host_id: str, tools: set[str], task: Task, settle_s: float
) -> Outcome:
    outcome = Outcome(task=task)
    reader = "clipboard.read" if "clipboard.read" in tools else "selection.read"
    if task.capability == "clipboard_roundtrip":
        # Reset first so a stale value can never pass the check.
        client.invoke(host_id, "clipboard.write", {"text": CLIPBOARD_BASELINE})
    if task.capability == "open_app":
        before = _structured(client.invoke(host_id, "window.active_context", {}))
        if verify_open_app(task, before)[0]:
            outcome.verify_detail = "skipped: app already frontmost before the turn"
            outcome.needs_review = True
            return outcome
    started = time.monotonic()
    try:
        client.run_turn(task.prompt, outcome)
    except httpx.HTTPError as error:
        outcome.error = type(error).__name__
    outcome.latency_s = round(time.monotonic() - started, 2)
    time.sleep(settle_s)
    try:
        if task.capability == "clipboard_roundtrip":
            outcome.verified, outcome.verify_detail = verify_clipboard(
                task, _structured(client.invoke(host_id, reader, {}))
            )
        elif task.capability == "open_app":
            outcome.verified, outcome.verify_detail = verify_open_app(
                task, _structured(client.invoke(host_id, "window.active_context", {}))
            )
        elif task.capability == "type_in_app":
            tree = client.invoke(
                host_id,
                "ui.accessibility_tree",
                {"action": "inspect_window", "windowTitle": task.app or "", "maxNodes": 60},
            )
            outcome.verified, outcome.verify_detail = verify_typed(task, _structured(tree))
        elif task.capability == "read_browser_url":
            outcome.verified, outcome.verify_detail = verify_url_in_reply(
                outcome.reply, _structured(client.invoke(host_id, "browser.current_url", {}))
            )
    except httpx.HTTPError as error:
        outcome.verify_detail = f"verifier_error={type(error).__name__}"
    classify(outcome)
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.environ.get("XUNA_OWNER_TOKEN", ""))
    parser.add_argument("--pet-id", required=True)
    parser.add_argument("--host", required=True, help="Host alias or UUID")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--settle", type=float, default=1.5, help="Seconds before verifying")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between tasks")
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--allow-typing", action="store_true")
    parser.add_argument("--exclude-app", action="append", default=[])
    parser.add_argument(
        "--plan-only", action="store_true", help="Print the generated tasks without running"
    )
    parser.add_argument("--out", type=Path, default=Path("benchmark-results/host-eval.json"))
    args = parser.parse_args(argv)
    if not args.token:
        parser.error("--token or XUNA_OWNER_TOKEN is required")

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 30)
    rng = random.Random(seed)  # noqa: S311 - reproducible task sampling, not security
    client = EvalClient(args.base_url, args.token, args.pet_id, auto_approve=args.auto_approve)
    try:
        host = client.host(args.host)
        host_id = str(host["hostId"])
        listing = _structured(client.invoke(host_id, "app.list_allowed", {}))
        apps = parse_app_names(listing)
        tools = host_tool_names(host)
        caps = available_capabilities(tools, allow_typing=args.allow_typing)
        excluded = (*DEFAULT_EXCLUDED_APP_MARKERS, *(m.lower() for m in args.exclude_app))
        tasks = generate_tasks(apps, caps, args.tasks, rng, excluded_markers=excluded)
        if args.plan_only:
            for task in tasks:
                print(f"{task.task_id}: {task.prompt}")  # noqa: T201 - CLI output
            return 0
        outcomes: list[Outcome] = []
        for task in tasks:
            outcome = run_task(client, host_id, tools, task, args.settle)
            outcomes.append(outcome)
            mark = (
                "PASS"
                if outcome.verified
                else ("FALSE-SUCCESS" if outcome.false_success else "FAIL")
            )
            print(f"[{mark}] {task.task_id}: {task.prompt} ({outcome.verify_detail})")  # noqa: T201
            time.sleep(args.delay)
    finally:
        client.close()

    report = {
        "seed": seed,
        "host": args.host,
        "appCount": len(apps),
        "summary": summarize(outcomes),
        "outcomes": [asdict(o) for o in outcomes],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))  # noqa: T201 - CLI output
    return 1 if report["summary"]["overall"]["falseSuccess"] else 0


if __name__ == "__main__":
    sys.exit(main())
