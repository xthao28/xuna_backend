"""Delegated Code Specialist sub-agent for LATM tool synthesis & self-debugging (`FORGE-LATM`).

Allows XUNA to separate conversational turns (`text_generator`, Model 1) from
executable script synthesis and `stderrPreview` self-debugging (`code_text_generator`, Model 2).
"""

import json
import re
from dataclasses import dataclass

from xuna_backend.modules.cognition.application.ports import TextGenerator
from xuna_backend.modules.cognition.domain.types import (
    ChatRequest,
    Locale,
    Message,
    Role,
    TextDelta,
)

_ALLOWED_RUNTIMES = frozenset({"python", "node", "powershell", "bash"})
_FENCED_BLOCK_RE = re.compile(
    r"```(?:(python|py|javascript|js|node|powershell|ps1|bash|sh|json))?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9_]+")

CODE_SPECIALIST_SYSTEM_PROMPT = (
    "You are XUNA's dedicated Code Specialist Sub-Agent (LATM Toolsmith). "
    "Write or repair a safe, self-contained executable script for the owner's tool. "
    "Rules:\n"
    "1. Return ONLY a JSON object with keys: "
    '{"runtime": "python"|"node"|"powershell"|"bash", '
    '"toolName": "<snake_case_slug>", '
    '"scriptCode": "<executable code>", '
    '"testArgsJson": "<valid JSON object string>"}.\n'
    "2. The script MUST read input arguments from the environment variable "
    "`XUNA_TOOL_ARGS_JSON` (default `{}`) and print a JSON object to stdout "
    '(including `"processedCount"` when applicable).\n'
    "3. NEVER use destructive filesystem commands (`rm -rf`, `shutil.rmtree`, "
    "`Format-Volume`), NEVER read `.ssh`/`.aws`/`.env` credentials, NEVER use "
    "`eval()`/`exec()`/`Invoke-Expression`, and NEVER spawn `subprocess` with `shell=True`."
)


@dataclass(frozen=True, slots=True)
class SynthesizedCodeDraft:
    runtime: str
    tool_name: str
    script_code: str
    test_args_json: str


def _normalize_slug(raw: str, fallback: str = "custom_tool") -> str:
    cleaned = _SLUG_CLEAN_RE.sub("_", raw.strip().lower()).strip("_")
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned[:40]


def _normalize_runtime(raw: str | None, fallback: str = "python") -> str:
    if not raw:
        return fallback
    low = raw.strip().lower()
    mapping = {
        "py": "python",
        "python3": "python",
        "js": "node",
        "javascript": "node",
        "ps1": "powershell",
        "pwsh": "powershell",
        "sh": "bash",
    }
    resolved = mapping.get(low, low)
    return resolved if resolved in _ALLOWED_RUNTIMES else fallback


def parse_code_specialist_reply(
    raw_text: str,
    *,
    default_runtime: str = "python",
    default_tool_name: str = "custom_tool",
    default_test_args_json: str = "{}",
) -> SynthesizedCodeDraft | None:
    """Extracts a `SynthesizedCodeDraft` from JSON or fenced code returned by the Code Model."""
    text = raw_text.strip()
    if not text:
        return None

    candidates = [text]
    for match in _FENCED_BLOCK_RE.finditer(text):
        lang = (match.group(1) or "").strip().lower()
        body = match.group(2).strip()
        if lang in ("", "json") and body.startswith("{"):
            candidates.insert(0, body)

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if 0 <= brace_start < brace_end:
        candidates.append(text[brace_start : brace_end + 1])

    for cand in candidates:
        if not cand.startswith("{"):
            continue
        try:
            parsed = json.loads(cand)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("scriptCode"), str):
            code = parsed["scriptCode"].strip()
            if not code:
                continue
            rt = _normalize_runtime(
                str(parsed.get("runtime", default_runtime)), default_runtime
            )
            slug = _normalize_slug(
                str(parsed.get("toolName", default_tool_name)), default_tool_name
            )
            raw_test_args = parsed.get("testArgsJson", default_test_args_json)
            if isinstance(raw_test_args, dict):
                test_args = json.dumps(raw_test_args, ensure_ascii=False)
            elif isinstance(raw_test_args, str) and raw_test_args.strip():
                test_args = raw_test_args.strip()
            else:
                test_args = default_test_args_json
            return SynthesizedCodeDraft(
                runtime=rt,
                tool_name=slug,
                script_code=code[:8000],
                test_args_json=test_args[:2000],
            )

    for match in _FENCED_BLOCK_RE.finditer(text):
        lang = (match.group(1) or "").strip().lower()
        body = match.group(2).strip()
        if lang != "json" and body:
            rt = _normalize_runtime(lang or default_runtime, default_runtime)
            return SynthesizedCodeDraft(
                runtime=rt,
                tool_name=_normalize_slug(default_tool_name),
                script_code=body[:8000],
                test_args_json=default_test_args_json,
            )

    return None


class CodeSpecialistSynthesizer:
    """Invokes the dedicated Code Specialist `TextGenerator` to write or repair tool scripts."""

    def __init__(
        self,
        code_generator: TextGenerator,
        *,
        max_output_tokens: int = 3000,
        timeout_s: float = 45.0,
    ) -> None:
        self._generator = code_generator
        self._max_tokens = max_output_tokens
        self._timeout_s = timeout_s

    async def synthesize_or_refine(
        self,
        *,
        goal: str,
        runtime: str = "python",
        tool_name: str = "custom_tool",
        test_args_json: str = "{}",
        broken_script: str | None = None,
        stderr_preview: str | None = None,
        locale: Locale = Locale.VI,
    ) -> SynthesizedCodeDraft | None:
        eff_runtime = _normalize_runtime(runtime, "python")
        eff_slug = _normalize_slug(tool_name, "custom_tool")
        eff_test_args = test_args_json.strip() or "{}"

        if broken_script and stderr_preview:
            user_prompt = (
                f"Repair this broken `{eff_runtime}` tool script.\n"
                f"Goal: {goal}\n"
                f"Tool name: {eff_slug}\n"
                f"Test args JSON: {eff_test_args}\n"
                f"Broken script:\n```{eff_runtime}\n{broken_script}\n```\n"
                f"Sandbox dry-run error (stderr):\n{stderr_preview}\n"
                "Return the corrected JSON object with runtime, toolName, scriptCode, testArgsJson."
            )
        else:
            user_prompt = (
                f"Synthesize an executable `{eff_runtime}` tool script.\n"
                f"Goal: {goal}\n"
                f"Preferred toolName: {eff_slug}\n"
                f"Sample testArgsJson: {eff_test_args}\n"
                "Return the JSON object with runtime, toolName, scriptCode, testArgsJson."
            )

        req = ChatRequest(
            messages=(
                Message(Role.SYSTEM, CODE_SPECIALIST_SYSTEM_PROMPT),
                Message(Role.USER, user_prompt),
            ),
            locale=locale,
            max_output_tokens=self._max_tokens,
            timeout_s=self._timeout_s,
            tools=(),
        )

        chunks: list[str] = []
        try:
            async for event in self._generator.stream(req):
                if isinstance(event, TextDelta):
                    chunks.append(event.text)
        except Exception:
            return None

        return parse_code_specialist_reply(
            "".join(chunks),
            default_runtime=eff_runtime,
            default_tool_name=eff_slug,
            default_test_args_json=eff_test_args,
        )
