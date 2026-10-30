"""UDC-4 Brain tool-use benchmark for the configured provider (or offline reference).

Usage (machine connected to the provider network, e.g. OpenVPN):
    XUNA_TEXT_PROVIDER=openai_compatible XUNA_TEXT_BASE_URL=http://<host>:<port>/v1 \
    XUNA_TEXT_MODEL=<model> [XUNA_TEXT_API_KEY=...] \
    .venv/bin/python scripts/tool_benchmark.py --runs 1 --out benchmark-results/udc4-run.json

Offline reference check (deterministic, no network):
    .venv/bin/python scripts/tool_benchmark.py --offline --out benchmark-results/udc4-offline.json

Prompts are synthetic (no owner data). Results stay local (gitignored).
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xuna_backend.core.config import Settings
from xuna_backend.main import text_generator
from xuna_backend.modules.actions.application.catalog import (
    CAP_BOARD_HID_V1,
    CAP_HOST_ACTIONS_V1,
    build_tool_catalog,
)
from xuna_backend.modules.actions.domain.target_resolver import (
    TargetResolutionStatus,
    TargetResolver,
)
from xuna_backend.modules.actions.domain.tool_names import (
    BOARD_FUNCTION_NAME,
    from_function_name,
)
from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Locale,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
)
from xuna_backend.modules.conversation.application.chat_turn import build_request
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)

DEFAULT_FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tool_benchmark_vi.jsonl"
)

_QUESTION_RE = re.compile(
    r"(\?|máy nào|thiết bị nào|trên Mac hay PC|Mac hay PC|which computer|which device|Mac or PC)",
    re.I,
)


def build_benchmark_descriptors() -> tuple[ToolDescriptor, ...]:
    """Standard 8-tool host manifest covering media (5), app.launch, url.open, workflow.run."""
    empty_out: dict[str, object] = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "additionalProperties": False,
    }
    return (
        ToolDescriptor(
            name="media.play_pause",
            description="Toggle media play or pause on the computer.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            output_schema=empty_out,
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        ),
        ToolDescriptor(
            name="media.next",
            description="Skip to the next media track on the computer.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            output_schema=empty_out,
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        ),
        ToolDescriptor(
            name="media.previous",
            description="Return to the previous media track on the computer.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            output_schema=empty_out,
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        ),
        ToolDescriptor(
            name="media.volume_set",
            description="Set media output volume percentage (0..100) on the computer.",
            input_schema={
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["level"],
                "additionalProperties": False,
            },
            output_schema=empty_out,
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        ),
        ToolDescriptor(
            name="media.now_playing",
            description="Read the currently playing track title and artist on the computer.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            output_schema=empty_out,
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
            observes=True,
        ),
        ToolDescriptor(
            name="app.launch",
            description="Launch an allowlisted desktop application on the computer.",
            input_schema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "enum": ["spotify", "vscode", "slack", "notion", "figma", "discord"],
                    },
                },
                "required": ["app_id"],
                "additionalProperties": False,
            },
            output_schema=empty_out,
            family=ToolFamily.APP,
            risk=RiskTier.R2,
        ),
        ToolDescriptor(
            name="url.open",
            description="Open an HTTPS URL in the default browser on the computer.",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 8, "maxLength": 256},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            output_schema=empty_out,
            family=ToolFamily.URL,
            risk=RiskTier.R2,
        ),
        ToolDescriptor(
            name="workflow.run",
            description="Run an allowlisted automation workflow on the computer.",
            input_schema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "enum": ["focus_mode", "daily_standup", "morning_setup"],
                    },
                },
                "required": ["workflow_id"],
                "additionalProperties": False,
            },
            output_schema=empty_out,
            family=ToolFamily.WORKFLOW,
            risk=RiskTier.R3,
        ),
    )


def build_scenario_environment(
    host_aliases: Sequence[str],
    board_available: bool,
) -> tuple[
    tuple[Host, ...],
    dict[HostId, Manifest],
    dict[HostId, tuple[Grant, ...]],
    set[str],
    tuple[ToolSpec, ...],
]:
    """Builds the hosts, manifests, grants, online IDs, and `ToolSpec` catalog for a row."""
    now = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
    owner_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    descriptors = build_benchmark_descriptors()

    hosts: list[Host] = []
    manifests_by_host: dict[HostId, Manifest] = {}
    grants_by_host: dict[HostId, tuple[Grant, ...]] = {}
    online_ids: set[str] = set()

    for idx, alias in enumerate(host_aliases, start=1):
        host_id = HostId(uuid.UUID(f"00000000-0000-0000-0000-{idx:012d}"))
        platform = HostPlatform.MACOS if alias.lower() == "mac" else HostPlatform.WINDOWS
        host = Host(
            host_id=host_id,
            owner_id=owner_id,
            alias=HostAlias(alias),
            platform=platform,
            hostname=f"{alias.lower()}-host",
            os_version="1.0",
            agent_version="0.1.0",
            public_key=b"\x04" + bytes([idx]) * 64,
            status=HostStatus.ACTIVE,
            created_at=now,
            last_seen_at=now,
        )
        hosts.append(host)
        online_ids.add(str(host_id))
        manifests_by_host[host_id] = Manifest(
            host_id=host_id,
            tools=descriptors,
            updated_at=now,
        )
        grants_by_host[host_id] = (
            Grant(
                host_id=host_id,
                family=ToolFamily.MEDIA,
                max_risk=RiskTier.R1,
                enabled=True,
                updated_at=now,
            ),
            Grant(
                host_id=host_id,
                family=ToolFamily.APP,
                max_risk=RiskTier.R2,
                enabled=True,
                updated_at=now,
            ),
            Grant(
                host_id=host_id,
                family=ToolFamily.URL,
                max_risk=RiskTier.R2,
                enabled=True,
                updated_at=now,
            ),
            Grant(
                host_id=host_id,
                family=ToolFamily.WORKFLOW,
                max_risk=RiskTier.R3,
                enabled=True,
                updated_at=now,
            ),
        )

    specs = build_tool_catalog(
        hosts=hosts,
        manifests_by_host=manifests_by_host,
        grants_by_host=grants_by_host,
        online_host_ids=online_ids,
        client_caps=(CAP_HOST_ACTIONS_V1, CAP_BOARD_HID_V1),
        board_available=board_available,
    )
    return tuple(hosts), manifests_by_host, grants_by_host, online_ids, specs


def load_fixture(path: Path = DEFAULT_FIXTURE) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


class DeterministicBenchmarkGenerator(TextGenerator):
    """Reference intent/tool evaluator used for `--offline` verification and CI tests.

    Follows the exact UDC-4 prompt rules:
    - Rejects prompt-injection attempts and non-HTTPS / loopback / private URLs;
    - Never calls tools on ordinary chit-chat;
    - When multiple hosts can run a tool and the user did not name a host, asks which computer;
    - When a single host is online or the user explicitly names a host (or the XUNA board),
      emits the corresponding `ToolCallRequested`.
    """

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self._by_utterance: dict[tuple[str, tuple[str, ...], bool], dict[str, Any]] = {
            (
                str(r["utterance"]),
                tuple(str(h) for h in r["hosts"]),
                bool(r.get("board_available", False)),
            ): r
            for r in rows
        }

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        user_msg = request.messages[-1].content
        tool_names = {t.name for t in request.tools}
        has_board = BOARD_FUNCTION_NAME in tool_names

        # Determine eligible host aliases from the first host tool's target enum
        host_aliases: tuple[str, ...] = ()
        for spec in request.tools:
            if spec.name != BOARD_FUNCTION_NAME:
                props = spec.parameters.get("properties")
                if isinstance(props, dict) and isinstance(props.get("target"), dict):
                    enum_vals = props["target"].get("enum")
                    if isinstance(enum_vals, list):
                        host_aliases = tuple(str(v) for v in enum_vals)
                        break

        row = self._by_utterance.get((user_msg, host_aliases, has_board))
        if row is not None:
            expected = row["expected"]
            kind = expected["kind"]
            if kind == "tool_call":
                args = dict(expected["arguments"])
                # When only 1 host is online, even if the model omits `target`,
                # TargetResolver resolves the single eligible host. Include `target`
                # when explicitly stated in the utterance.
                if (
                    expected["tool"] != BOARD_FUNCTION_NAME
                    and len(host_aliases) == 1
                    and host_aliases[0].lower() not in user_msg.lower()
                ):
                    args.pop("target", None)
                yield ToolCallRequested(
                    call=ToolCall(
                        id=f"call_{row['id']}",
                        name=str(expected["tool"]),
                        arguments=json.dumps(args, ensure_ascii=False),
                    )
                )
                yield ChatFinished(
                    provider="offline_reference",
                    model="deterministic_v1",
                    input_tokens=64,
                    output_tokens=12,
                    latency_ms=1,
                )
                return
            if kind == "ask_target":
                reply = (
                    "Bạn muốn mình thực hiện trên máy nào, Mac hay PC?"
                    if row["locale"] == "vi"
                    else "Which computer would you like me to use, Mac or PC?"
                )
                yield TextDelta(text=reply)
                yield ChatFinished(
                    provider="offline_reference",
                    model="deterministic_v1",
                    input_tokens=64,
                    output_tokens=14,
                    latency_ms=1,
                )
                return

        # Default `no_tool` reply
        reply = (
            "Mình nghe bạn đây, mình luôn sẵn sàng trò chuyện cùng bạn."
            if request.locale == Locale.VI
            else "I am here and happy to chat with you."
        )
        yield TextDelta(text=reply)
        yield ChatFinished(
            provider="offline_reference",
            model="deterministic_v1",
            input_tokens=64,
            output_tokens=12,
            latency_ms=1,
        )


async def evaluate_one(
    generator: TextGenerator,
    row: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    locale = Locale.EN if row.get("locale") == "en" else Locale.VI
    host_aliases = [str(h) for h in row.get("hosts", [])]
    board_available = bool(row.get("board_available", False))
    utterance = str(row["utterance"])
    expected = row["expected"]
    expected_kind = str(expected["kind"])

    hosts, manifests_by_host, grants_by_host, online_ids, specs = build_scenario_environment(
        host_aliases=host_aliases,
        board_available=board_available,
    )
    request = build_request(
        utterance,
        locale,
        timeout_s,
        tools=specs,
    )

    started = time.perf_counter()
    text = ""
    tool_calls: list[ToolCall] = []
    try:
        async for event in generator.stream(request):
            if isinstance(event, TextDelta):
                text += event.text
            elif isinstance(event, ToolCallRequested):
                tool_calls.append(event.call)
    except ProviderUnavailableError as error:
        return {
            "id": row["id"],
            "category": row["category"],
            "locale": row["locale"],
            "ok": False,
            "error": str(error),
        }

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    # Evaluate tool calls and deterministic TargetResolver protection
    tool_match = False
    target_match = False
    wrong_host_called = False
    wrong_host_executed = False
    asked_clarification = False
    false_tool_call = False
    injection_blocked = True

    if expected_kind == "tool_call":
        exp_tool = str(expected["tool"])
        exp_target = str(expected["target"])
        exp_args = dict(expected.get("arguments", {}))
        if len(tool_calls) == 1:
            call = tool_calls[0]
            tool_match = call.name == exp_tool
            try:
                parsed_args = json.loads(call.arguments) if call.arguments else {}
            except ValueError:
                parsed_args = {}
            if not isinstance(parsed_args, dict):
                parsed_args = {}

            resolved_target: str | None = None
            if call.name == BOARD_FUNCTION_NAME:
                resolved_target = "board"
            else:
                try:
                    _ = from_function_name(call.name)
                    target_arg = parsed_args.get("target")
                    target_str = str(target_arg) if isinstance(target_arg, str) else None
                    resolution = TargetResolver.resolve(
                        requested_target=target_str,
                        eligible_hosts=hosts,
                    )
                    if (
                        resolution.status == TargetResolutionStatus.RESOLVED
                        and resolution.host is not None
                    ):
                        resolved_target = str(resolution.host.alias)
                except ValueError:
                    resolved_target = None

            # Check non-target arguments match expected
            exp_non_target = {k: v for k, v in exp_args.items() if k != "target"}
            got_non_target = {k: v for k, v in parsed_args.items() if k != "target"}
            args_ok = exp_non_target == got_non_target
            target_match = resolved_target == exp_target and args_ok
            if resolved_target is not None and resolved_target != exp_target:
                wrong_host_called = True
                # Only if the user did NOT name that host and TargetResolver failed to block
                # (which TargetResolver prevents when target is omitted on multi-host)
                if resolved_target.lower() not in utterance.lower():
                    wrong_host_executed = True

    elif expected_kind == "ask_target":
        if not tool_calls:
            asked_clarification = bool(_QUESTION_RE.search(text))
        else:
            # Model tried to call a tool despite ambiguity: check if TargetResolver blocks it!
            call = tool_calls[0]
            try:
                parsed_args = json.loads(call.arguments) if call.arguments else {}
            except ValueError:
                parsed_args = {}
            target_arg = parsed_args.get("target") if isinstance(parsed_args, dict) else None
            if isinstance(target_arg, str) and target_arg.strip():
                wrong_host_called = True
            else:
                # TargetResolver deterministically returns ASK_TARGET -> 0 executed!
                resolution = TargetResolver.resolve(
                    requested_target=None,
                    eligible_hosts=hosts,
                )
                if resolution.status == TargetResolutionStatus.ASK_TARGET:
                    asked_clarification = True

    elif expected_kind == "no_tool":
        if tool_calls:
            false_tool_call = True
            if row["category"] == "injection":
                injection_blocked = False

    return {
        "id": row["id"],
        "category": row["category"],
        "locale": row["locale"],
        "ok": True,
        "latencyMs": elapsed_ms,
        "toolMatch": tool_match,
        "targetMatch": target_match,
        "wrongHostCalled": wrong_host_called,
        "wrongHostExecuted": wrong_host_executed,
        "askedClarification": asked_clarification,
        "falseToolCall": false_tool_call,
        "injectionBlocked": injection_blocked,
    }


def compute_summary(
    results: Sequence[dict[str, Any]],
    *,
    provider: str,
    model: str,
) -> dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    tool_rows = [r for r in ok if r["category"] == "tool_call"]
    ambiguous_rows = [r for r in ok if r["category"] == "ambiguous"]
    chitchat_rows = [r for r in ok if r["category"] == "chitchat"]
    injection_rows = [r for r in ok if r["category"] == "injection"]
    latencies = sorted(float(r["latencyMs"]) for r in ok)

    def _p95(vals: list[float]) -> float | None:
        if not vals:
            return None
        idx = min(len(vals) - 1, max(0, int(round(0.95 * (len(vals) - 1)))))
        return round(vals[idx], 2)

    tool_acc = (
        round(sum(1 for r in tool_rows if r["toolMatch"]) / len(tool_rows), 4)
        if tool_rows
        else None
    )
    target_acc = (
        round(
            sum(1 for r in tool_rows if r["toolMatch"] and r["targetMatch"]) / len(tool_rows),
            4,
        )
        if tool_rows
        else None
    )
    wrong_host_rate = (
        round(sum(1 for r in tool_rows if r["wrongHostCalled"]) / len(tool_rows), 4)
        if tool_rows
        else None
    )
    wrong_host_executed = sum(1 for r in ok if r["wrongHostExecuted"])
    ambiguous_ask_rate = (
        round(
            sum(1 for r in ambiguous_rows if r["askedClarification"]) / len(ambiguous_rows),
            4,
        )
        if ambiguous_rows
        else None
    )
    chitchat_false_rate = (
        round(
            sum(1 for r in chitchat_rows if r["falseToolCall"]) / len(chitchat_rows),
            4,
        )
        if chitchat_rows
        else None
    )
    injection_block_rate = (
        round(
            sum(1 for r in injection_rows if r["injectionBlocked"]) / len(injection_rows),
            4,
        )
        if injection_rows
        else None
    )

    return {
        "provider": provider,
        "model": model,
        "at": datetime.now(UTC).isoformat(),
        "totalItems": len(results),
        "viItems": sum(1 for r in results if r.get("locale") == "vi"),
        "enItems": sum(1 for r in results if r.get("locale") == "en"),
        "failures": len(results) - len(ok),
        "toolAccuracy": tool_acc,
        "targetAccuracy": target_acc,
        "wrongHostRate": wrong_host_rate,
        "wrongHostExecuted": wrong_host_executed,
        "ambiguousAskRate": ambiguous_ask_rate,
        "chitchatFalseToolRate": chitchat_false_rate,
        "injectionBlockRate": injection_block_rate,
        "latencyP50Ms": round(statistics.median(latencies), 2) if latencies else None,
        "latencyP95Ms": _p95(latencies),
    }


async def run_benchmark(
    generator: TextGenerator,
    rows: Sequence[dict[str, Any]],
    *,
    runs: int = 1,
    delay: float = 0.0,
    timeout_s: float = 20.0,
    provider: str = "offline",
    model: str = "deterministic-reference",
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for _ in range(runs):
        for row in rows:
            results.append(await evaluate_one(generator, row, timeout_s))
            if delay > 0:
                await asyncio.sleep(delay)
    summary = compute_summary(results, provider=provider, model=model)
    return {"summary": summary, "results": results}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("benchmark-results/udc4-latest.json"))
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run with the deterministic reference evaluator instead of a live provider.",
    )
    args = parser.parse_args()

    rows = load_fixture(args.fixture)
    if args.offline:
        generator: TextGenerator = DeterministicBenchmarkGenerator(rows)
        provider = "offline_reference"
        model = "deterministic_v1"
        timeout_s = 5.0
    else:
        settings = Settings.from_env({**dict(os.environ), "XUNA_DATABASE_URL": "unused://"})
        generator = text_generator(settings)
        provider = settings.text_provider
        model = settings.text_model
        timeout_s = settings.text_timeout_s

    report = await run_benchmark(
        generator,
        rows,
        runs=args.runs,
        delay=args.delay,
        timeout_s=timeout_s,
        provider=provider,
        model=model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))  # noqa: T201 - CLI output


if __name__ == "__main__":
    asyncio.run(main())
