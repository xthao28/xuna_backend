"""Deferred host-tool discovery: `tool_search`, `tool_describe`, `tool_invoke`.

A host can advertise 70+ XTP tools, too many to send as function schemas on every turn.
Following the OpenClaw Tool Search pattern, the catalog sends a fixed set of core host
tools directly and makes every other eligible host tool reachable through three
backend-owned meta tools:

1. `tool_search(query)`: compact matches (name, short description, hosts).
2. `tool_describe(name)`: the exact argument signature of one tool.
3. `tool_invoke(name, argumentsJson, target?)`: calls it.

`tool_invoke` re-enters the normal orchestrator pipeline (schema validation, target
resolution, grants, owner approval, audit), so deferral never bypasses a gate. Only tools
in the same eligible set the catalog uses (granted, consented, schema-valid) are found or
invoked.
"""

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from xuna_backend.modules.actions.domain.tool_names import from_function_name, to_function_name
from xuna_backend.modules.cognition.domain.types import ToolSpec
from xuna_backend.modules.hosts.domain.entities import ToolDescriptor

TOOL_SEARCH_FUNCTION = "tool_search"
TOOL_DESCRIBE_FUNCTION = "tool_describe"
TOOL_INVOKE_FUNCTION = "tool_invoke"
META_FUNCTION_NAMES: frozenset[str] = frozenset(
    {TOOL_SEARCH_FUNCTION, TOOL_DESCRIBE_FUNCTION, TOOL_INVOKE_FUNCTION}
)

META_TOOL_COUNT = 3
DEFAULT_SEARCH_LIMIT = 8
MAX_SEARCH_LIMIT = 15
MAX_DIRECTORY_CHARS = 1500
MAX_DESCRIBE_BYTES = 4096
MAX_INVOKE_ARGUMENTS_CHARS = 4000

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class EligibleHostTool:
    """A host tool the owner may use this turn, with the online host aliases offering it."""

    descriptor: ToolDescriptor
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InvokeRequest:
    """A validated `tool_invoke` request, ready for the orchestrator pipeline."""

    tool_name: str
    function_name: str
    arguments: dict[str, object]


class MetaToolArgumentError(ValueError):
    """Raised when a meta-tool call has malformed arguments; the code is returned to the model."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _tokens(text: str) -> set[str]:
    norm = unicodedata.normalize("NFC", text).lower()
    return {t for t in _TOKEN_RE.findall(norm) if len(t) >= 2}


def _name_tokens(tool_name: str) -> set[str]:
    return _tokens(tool_name.replace(".", " ").replace("_", " "))


def _normalize_tool_name(raw: object) -> str | None:
    """Accepts `media.next` or `media__next`; returns the dotted name or `None`."""
    if not isinstance(raw, str):
        return None
    clean = raw.strip()
    if not clean or len(clean) > 64:
        return None
    return from_function_name(clean)


def build_meta_tool_specs(
    eligible: Mapping[str, EligibleHostTool],
    direct_names: frozenset[str],
) -> tuple[ToolSpec, ...]:
    """Builds the three meta-tool specs; `tool_search` carries a bounded directory of the
    deferred (not directly listed) tool names so the model knows what exists."""
    deferred = [name for name in eligible if name not in direct_names]
    directory_parts: list[str] = []
    used = 0
    for idx, name in enumerate(deferred):
        cost = len(name) + 2
        if used + cost > MAX_DIRECTORY_CHARS:
            directory_parts.append(f"(+{len(deferred) - idx} more, find them by search)")
            break
        directory_parts.append(name)
        used += cost
    directory = ", ".join(directory_parts) if directory_parts else "none"
    search_description = (
        "Find host tools that are not in your direct tool list. Query with short English "
        "keywords (e.g. 'tv volume', 'smart home light', 'clipboard'). Then call "
        f"tool_describe for the exact arguments and tool_invoke to run it. Deferred tools: "
        f"{directory}"
    )
    return (
        ToolSpec(
            name=TOOL_SEARCH_FUNCTION,
            description=search_description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 120},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_LIMIT},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name=TOOL_DESCRIBE_FUNCTION,
            description=(
                "Return the exact argument signature, risk tier and available hosts of one "
                "host tool found by tool_search. Call it before tool_invoke."
            ),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "maxLength": 64}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name=TOOL_INVOKE_FUNCTION,
            description=(
                "Run one host tool found by tool_search. `argumentsJson` is a JSON object "
                "string matching tool_describe's signature; `target` is a host alias when "
                "several hosts offer the tool. Normal approvals and permissions still apply."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 64},
                    "argumentsJson": {
                        "type": "string",
                        "maxLength": MAX_INVOKE_ARGUMENTS_CHARS,
                    },
                    "target": {"type": "string", "maxLength": 64},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
    )


def encode_meta_payload(status: str, output: dict[str, object]) -> str:
    """Encodes a meta-tool result as compact JSON within `MAX_DESCRIBE_BYTES`, dropping
    trailing list entries (matches / names) rather than truncating to an unusable blob."""
    body: dict[str, object] = {"status": status, "output": output}
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    for key in ("allTools", "matches"):
        items = output.get(key)
        while isinstance(items, list) and items:
            if len(encoded.encode("utf-8")) <= MAX_DESCRIBE_BYTES:
                return encoded
            items = items[:-1]
            output[key] = items
            encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return encoded


def search_tools(
    eligible: Mapping[str, EligibleHostTool],
    query: object,
    limit: object = None,
) -> dict[str, object]:
    """Ranks eligible tools against `query` by name and description token overlap.

    Returns the whole (bounded) name list when nothing matches, so the model can still
    pick a tool instead of guessing.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > 120:
        raise MetaToolArgumentError("invalid_args")
    max_results = DEFAULT_SEARCH_LIMIT
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise MetaToolArgumentError("invalid_args")
        max_results = max(1, min(limit, MAX_SEARCH_LIMIT))

    query_norm = unicodedata.normalize("NFC", query).lower().strip()
    query_tokens = _tokens(query_norm)
    scored: list[tuple[int, int, str]] = []
    for order, (name, item) in enumerate(eligible.items()):
        score = 0
        if name in query_norm or to_function_name(name) in query_norm:
            score += 100
        name_toks = _name_tokens(name)
        score += 10 * len(query_tokens & name_toks)
        for tok in query_tokens:
            if len(tok) >= 3 and tok not in name_toks and tok in name:
                score += 4
        score += 2 * len(query_tokens & _tokens(item.descriptor.description))
        if score > 0:
            scored.append((-score, order, name))
    scored.sort()

    def _entry(name: str) -> dict[str, object]:
        item = eligible[name]
        return {
            "name": name,
            "description": item.descriptor.description[:160],
            "hosts": list(item.aliases),
        }

    if scored:
        return {
            "matches": [_entry(name) for _, _, name in scored[:max_results]],
            "next": "Call tool_describe(name) for the exact arguments, then tool_invoke.",
        }
    return {
        "matches": [],
        "allTools": list(eligible.keys())[:60],
        "next": "No keyword match; pick a name from allTools and call tool_describe.",
    }


def _render_property(prop: object, required: bool) -> str:
    if not isinstance(prop, Mapping):
        return "any"
    parts: list[str] = [str(prop.get("type", "any"))]
    enum_vals = prop.get("enum")
    if isinstance(enum_vals, list) and enum_vals:
        shown = "|".join(str(v) for v in enum_vals[:16])
        more = f"|…(+{len(enum_vals) - 16})" if len(enum_vals) > 16 else ""
        parts.append(f"one of {shown}{more}")
    minimum = prop.get("minimum")
    maximum = prop.get("maximum")
    if isinstance(minimum, int | float) or isinstance(maximum, int | float):
        low = minimum if minimum is not None else ""
        high = maximum if maximum is not None else ""
        parts.append(f"range {low}..{high}")
    max_len = prop.get("maxLength")
    if isinstance(max_len, int):
        parts.append(f"max {max_len} chars")
    desc = prop.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append(desc.strip()[:100])
    if required:
        parts.append("REQUIRED")
    return "; ".join(parts)


def describe_tool(eligible: Mapping[str, EligibleHostTool], name: object) -> str:
    """Returns an encoded `Role.TOOL` payload with a compact, bounded argument signature."""
    tool_name = _normalize_tool_name(name)
    if tool_name is None or tool_name not in eligible:
        raise MetaToolArgumentError("unknown_tool")
    item = eligible[tool_name]
    schema = item.descriptor.input_schema
    props_raw = schema.get("properties")
    props = props_raw if isinstance(props_raw, Mapping) else {}
    required_raw = schema.get("required")
    required = set(required_raw) if isinstance(required_raw, list) else set()
    args = {str(k): _render_property(v, k in required) for k, v in props.items()}
    output: dict[str, object] = {
        "name": tool_name,
        "description": item.descriptor.description,
        "risk": item.descriptor.risk.value,
        "hosts": list(item.aliases),
        "args": args,
        "next": "Call tool_invoke with name and argumentsJson (a JSON object string).",
    }
    encoded = json.dumps(
        {"status": "ok", "output": output}, ensure_ascii=False, separators=(",", ":")
    )
    while len(encoded.encode("utf-8")) > MAX_DESCRIBE_BYTES and len(args) > 1:
        args = {k: v[:60] for k, v in list(args.items())[: max(len(args) - 4, 1)]}
        output["args"] = args
        output["argsTruncated"] = True
        encoded = json.dumps(
            {"status": "ok", "output": output}, ensure_ascii=False, separators=(",", ":")
        )
    return encoded


def parse_invoke(
    eligible: Mapping[str, EligibleHostTool],
    arguments: Mapping[str, object],
) -> InvokeRequest:
    """Validates a `tool_invoke` call; the inner arguments are validated again against the
    host's schema by the orchestrator."""
    tool_name = _normalize_tool_name(arguments.get("name"))
    if tool_name is None or tool_name not in eligible:
        raise MetaToolArgumentError("unknown_tool")
    raw_json = arguments.get("argumentsJson")
    inner: dict[str, object] = {}
    if raw_json is not None:
        if not isinstance(raw_json, str) or len(raw_json) > MAX_INVOKE_ARGUMENTS_CHARS:
            raise MetaToolArgumentError("invalid_args")
        if raw_json.strip():
            try:
                parsed = json.loads(raw_json)
            except ValueError as error:
                raise MetaToolArgumentError("invalid_args") from error
            if not isinstance(parsed, dict):
                raise MetaToolArgumentError("invalid_args")
            inner = {str(k): v for k, v in parsed.items()}
    target = arguments.get("target")
    if target is not None:
        if not isinstance(target, str) or len(target) > 64:
            raise MetaToolArgumentError("invalid_args")
        if target.strip():
            inner["target"] = target.strip()
    return InvokeRequest(
        tool_name=tool_name,
        function_name=to_function_name(tool_name),
        arguments=inner,
    )
