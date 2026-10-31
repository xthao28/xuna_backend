"""Mapping between XTP dotted tool names and OpenAI-safe function names (UDC-4 §5.1-§5.2).

OpenAI-compatible chat completion APIs require function names matching
`^[a-zA-Z0-9_-]{1,64}$` and reject dots. We map `family.action` <-> `family__action`
and enforce the forbidden-tool list from ADR-0012 / UDC-3.
"""

import re

from xuna_backend.modules.hosts.domain.entities import (
    FORBIDDEN_TOOL_PREFIXES,
    FORBIDDEN_TOOLS,
    validate_tool_name,
)

FUNCTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

BOARD_TOOL = "board.media_key"
BOARD_FUNCTION_NAME = "board__media_key"
BOARD_MEDIA_KEY_DESCRIPTION = (
    "Send a single Consumer Control media key press via XUNA's Bluetooth HID "
    "to the currently paired host (requires owner confirmation)."
)

WORKFLOW_TEMPLATE_TOOL = "workflow.template"
WORKFLOW_TEMPLATE_FUNCTION_NAME = "workflow_template"
WORKFLOW_TEMPLATE_DESCRIPTION = (
    "Run a deterministic cross-device workflow template (`link_to_phone`, "
    "`continue_media`, `pause_everywhere`, `resume_work`, or `wrap_up_work`) "
    "with a single owner approval."
)
WORKFLOW_TEMPLATES: tuple[str, ...] = (
    "link_to_phone",
    "continue_media",
    "pause_everywhere",
    "resume_work",
    "wrap_up_work",
)
WORKFLOW_TEMPLATE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "template": {
            "type": "string",
            "enum": list(WORKFLOW_TEMPLATES),
        },
        "source_target": {
            "type": "string",
            "maxLength": 24,
        },
        "destination_target": {
            "type": "string",
            "maxLength": 24,
        },
        "browser": {
            "type": "string",
            "enum": ["auto", "edge", "chrome", "safari"],
        },
        "player_id": {
            "type": "string",
            "maxLength": 64,
        },
        "project": {
            "type": "string",
            "maxLength": 64,
        },
        "agent": {
            "type": "string",
            "enum": ["auto", "claude_code", "antigravity", "cursor", "vscode", "codex"],
        },
        "prompt": {
            "type": "string",
            "maxLength": 500,
        },
    },
    "required": ["template"],
    "additionalProperties": False,
}

# Normative Consumer Control usage index mapping (BLE Opcode 12 arg 1..7, UDC-2 §5.1).
BOARD_MEDIA_KEY_TO_USAGE_INDEX: dict[str, int] = {
    "play_pause": 1,
    "next": 2,
    "previous": 3,
    "volume_up": 4,
    "volume_down": 5,
    "mute": 6,
    "stop": 7,
}

BOARD_MEDIA_KEYS: tuple[str, ...] = tuple(BOARD_MEDIA_KEY_TO_USAGE_INDEX.keys())

BOARD_MEDIA_KEY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "key": {
            "type": "string",
            "enum": list(BOARD_MEDIA_KEYS),
        }
    },
    "required": ["key"],
    "additionalProperties": False,
}


def is_forbidden_tool_name(dotted_name: str) -> bool:
    if dotted_name in FORBIDDEN_TOOLS:
        return True
    return any(dotted_name.startswith(prefix) for prefix in FORBIDDEN_TOOL_PREFIXES)


def to_function_name(tool_name: str) -> str:
    """Converts a validated dotted tool name (`media.play_pause` or `board.media_key`)
    into an OpenAI-compatible function name (`media__play_pause`)."""
    if tool_name == WORKFLOW_TEMPLATE_TOOL:
        return WORKFLOW_TEMPLATE_FUNCTION_NAME
    if tool_name != BOARD_TOOL:
        validate_tool_name(tool_name)
    func_name = tool_name.replace(".", "__")
    if not FUNCTION_NAME_RE.match(func_name):
        raise ValueError(f"function name {func_name!r} violates ^[a-zA-Z0-9_-]{{1,64}}$")
    return func_name


def from_function_name(func_name: str) -> str | None:
    """Maps an LLM function name (`media__play_pause` or `media.play_pause`) back to
    its canonical dotted tool name. Returns `None` if unknown, malformed, or forbidden."""
    if not func_name or len(func_name) > 64:
        return None
    if func_name in (
        WORKFLOW_TEMPLATE_FUNCTION_NAME,
        "workflow__template",
        WORKFLOW_TEMPLATE_TOOL,
    ):
        return WORKFLOW_TEMPLATE_TOOL
    if "__" in func_name:
        parts = func_name.split("__")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        dotted = f"{parts[0]}.{parts[1]}"
    elif "." in func_name:
        dotted = func_name
    else:
        return None

    if dotted == BOARD_TOOL:
        return BOARD_TOOL
    if is_forbidden_tool_name(dotted):
        return None
    try:
        validate_tool_name(dotted)
    except ValueError:
        return None
    return dotted
