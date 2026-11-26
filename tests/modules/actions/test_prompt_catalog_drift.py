"""Upgrade plan P1.3 (U5): Anti-drift tests between LLM prompts and tool catalog/descriptors.

Guarantees that:
1. Every tool function name (`family__action` or backend built-in/meta tool) and every
   wildcard tool prefix (`family__*`) mentioned in backend prompt modules (`prompts.py`,
   `prompt_dev_modules.py`, `exec_fs_support.py`, `watch_coordinator.py`, `job_runner.py`,
   `cu_template.py`, `tool_discovery.py`) actually exists in the catalog/descriptors and is
   never a forbidden tool name.
2. Vietnamese and English prompt modules reference identical sets of tool names and prefixes.
3. Parameter names and literal argument examples embedded in prompts validate cleanly through
   `SchemaValidator.validate_arguments` against the real tool `inputSchema`s.
4. Backend `_PRIORITY_ORDER` and built-in/meta schemas stay valid and in sync with the real
   desktop (`xuna_xtp`) and mobile host descriptors when sibling repositories are present,
   with zero new descriptors silently dropped by `SchemaValidator`.
"""

import ast
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from xuna_backend.modules.actions.application import (
    catalog,
    cu_template,
    exec_fs_support,
    job_runner,
    prompt_dev_modules,
    prompts,
    tool_discovery,
    watch_coordinator,
)
from xuna_backend.modules.actions.domain import tool_names, watch
from xuna_backend.modules.actions.domain.schema_validator import (
    ArgumentValidationError,
    SchemaDefinitionError,
    SchemaValidator,
)
from xuna_backend.modules.cognition.domain.types import Locale
from xuna_backend.modules.hosts.domain.entities import validate_tool_name

BACKEND_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = BACKEND_ROOT.parent
FIXTURES_DIR = BACKEND_ROOT / "tests" / "fixtures" / "xtp"
XTP_SCHEMAS_DIR = REPO_ROOT / "desktop" / "packages" / "xuna_xtp" / "lib" / "src" / "schemas"
MOBILE_HOSTS_DIR = REPO_ROOT / "mobile" / "lib" / "features" / "hosts"

# Known pre-existing desktop descriptors rejected by backend SchemaValidator (documented in
# PROCESS.md 2026-11-26). Frozen here so no newly added or modified descriptor can ever be
# silently dropped from the catalog by `catalog.collect_eligible_host_tools`.
_PREEXISTING_SCHEMA_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "calendar.list_events",
        "notes.search",
        "shortcuts.run",
        "skill.chain_and_test",
        "web.navigate",
        "web.snapshot",
        "web.wait_for",
        "workflow.batch_steps",
    }
)

_BARE_BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        tool_discovery.TOOL_SEARCH_FUNCTION,
        tool_discovery.TOOL_DESCRIBE_FUNCTION,
        tool_discovery.TOOL_INVOKE_FUNCTION,
        tool_names.AUTONOMOUS_JOB_FUNCTION_NAME,
        tool_names.WORKFLOW_TEMPLATE_FUNCTION_NAME,
    }
)

_FN_TOOL_RE = re.compile(
    r"\b([a-z][a-z0-9_]*__[a-z][a-z0-9_]*|"
    r"tool_search|tool_describe|tool_invoke|autonomous_job|workflow_template)\b"
)
_WILD_PREFIX_RE = re.compile(r"\b([a-z][a-z0-9_]*)__\*")
_CALL_SIG_RE = re.compile(
    r"\b([a-z][a-z0-9_]*__[a-z][a-z0-9_]*|tool_search|tool_describe|tool_invoke|"
    r"autonomous_job|workflow_template)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\)"
)
_DYNAMIC_CUSTOM_PREFIX = "forge__custom_"


# ---------------------------------------------------------------------------
# Minimal stdlib Dart schema literal reader (for cross-repo parity in monorepo)
# ---------------------------------------------------------------------------


def _strip_dart_comments(src: str) -> str:
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch in ("'", '"'):
            raw = i > 0 and src[i - 1] == "r" and (i < 2 or not src[i - 2].isalnum())
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                c = src[i]
                out.append(c)
                if not raw and c == "\\" and i + 1 < n:
                    out.append(src[i + 1])
                    i += 2
                    continue
                i += 1
                if c == quote:
                    break
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_balanced(src: str, open_idx: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    depth = 0
    i = open_idx
    n = len(src)
    in_str: str | None = None
    is_raw = False
    while i < n:
        ch = src[i]
        if in_str is not None:
            if not is_raw and ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_str:
                in_str = None
                is_raw = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_str = ch
            is_raw = i > 0 and src[i - 1] == "r" and (i < 2 or not src[i - 2].isalnum())
            i += 1
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return src[open_idx : i + 1], i + 1
        i += 1
    raise ValueError(f"Unbalanced {open_ch!r} at offset {open_idx}")


class _DartSchemaReader:
    def __init__(self, constants: dict[str, Any]) -> None:
        self.constants: dict[str, Any] = dict(constants)

    def eval_expr(self, expr: str) -> Any:
        s = expr.strip()
        if s.startswith("_obj("):
            inner, _ = _extract_balanced(s, 4, "(", ")")
            return self._eval_obj(inner[1:-1])
        if s.startswith("_str("):
            m = re.fullmatch(r"_str\(\s*(\d+)\s*,\s*(\d+)\s*\)", s)
            if m is None:
                raise ValueError(f"Bad _str call: {s!r}")
            return {"type": "string", "minLength": int(m.group(1)), "maxLength": int(m.group(2))}
        if s.startswith("_int("):
            m = re.fullmatch(r"_int\(\s*([^,]+),\s*([^)]+)\)", s)
            if m is None:
                raise ValueError(f"Bad _int call: {s!r}")
            return {
                "type": "integer",
                "minimum": self._eval_int_expr(m.group(1)),
                "maximum": self._eval_int_expr(m.group(2)),
            }
        s = re.sub(r"\bconst\s+", "", s).strip()
        s = re.sub(r"^<[^>]+>\s*", "", s).strip()
        if s in self.constants:
            return self.constants[s]
        if s == "true":
            return True
        if s == "false":
            return False
        if s == "null":
            return None
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d+\.\d+", s):
            return float(s)
        if s.startswith(("r'", 'r"', "'", '"')):
            return self._eval_dart_string(s)
        if s.startswith("{"):
            return self._eval_map(s)
        if s.startswith("["):
            return self._eval_list(s)
        raise ValueError(f"Unsupported Dart schema expression: {s[:80]!r}")

    @staticmethod
    def _eval_int_expr(raw: str) -> int:
        s = raw.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        shift = re.fullmatch(r"(\d+)\s*<<\s*(\d+)", s)
        if shift is not None:
            return int(shift.group(1)) << int(shift.group(2))
        raise ValueError(f"Unsupported integer expression: {raw!r}")

    @staticmethod
    def _eval_dart_string(s: str) -> str:
        parts: list[str] = []
        i = 0
        n = len(s)
        while i < n:
            while i < n and s[i].isspace():
                i += 1
            if i >= n:
                break
            raw = False
            if s[i] == "r" and i + 1 < n and s[i + 1] in ("'", '"'):
                raw = True
                i += 1
            quote = s[i]
            start = i
            i += 1
            while i < n:
                if not raw and s[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if s[i] == quote:
                    i += 1
                    break
                i += 1
            lit = ("r" if raw else "") + s[start:i]
            val = ast.literal_eval(lit)
            if not isinstance(val, str):
                raise ValueError(f"Expected str literal, got {type(val).__name__}")
            parts.append(val)
        return "".join(parts)

    @staticmethod
    def _split_top_level(inner: str, sep: str = ",") -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        in_str: str | None = None
        is_raw = False
        i = 0
        n = len(inner)
        while i < n:
            ch = inner[i]
            if in_str is not None:
                buf.append(ch)
                if not is_raw and ch == "\\" and i + 1 < n:
                    buf.append(inner[i + 1])
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                    is_raw = False
                i += 1
                continue
            if ch in ("'", '"'):
                in_str = ch
                is_raw = i > 0 and inner[i - 1] == "r" and (i < 2 or not inner[i - 2].isalnum())
                buf.append(ch)
                i += 1
                continue
            if ch == "<" and i + 1 < n and inner[i + 1] == "<":
                buf.append("<<")
                i += 2
                continue
            if ch in "{[(<":
                depth += 1
                buf.append(ch)
            elif ch in "}])>":
                depth -= 1
                buf.append(ch)
            elif ch == sep and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    def _eval_map(self, s: str) -> dict[str, Any]:
        inner = s[1:-1].strip()
        if not inner:
            return {}
        res: dict[str, Any] = {}
        for item in self._split_top_level(inner, ","):
            if item.startswith("if ("):
                continue
            colon_parts = self._split_top_level(item, ":")
            key = self.eval_expr(colon_parts[0])
            val = self.eval_expr(":".join(colon_parts[1:]))
            res[str(key)] = val
        return res

    def _eval_list(self, s: str) -> list[Any]:
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [self.eval_expr(item) for item in self._split_top_level(inner, ",")]

    def _eval_obj(self, inner: str) -> dict[str, Any]:
        args = self._split_top_level(inner, ",")
        props = self.eval_expr(args[0])
        out: dict[str, Any] = {
            "type": "object",
            "properties": props,
            "additionalProperties": False,
        }
        for arg in args[1:]:
            if arg.startswith("required:"):
                req = self.eval_expr(arg.split(":", 1)[1])
                if req:
                    out["required"] = req
        return out


def _load_desktop_dart_descriptors() -> dict[str, dict[str, object]]:
    core_src = _strip_dart_comments((XTP_SCHEMAS_DIR / "core_types.dart").read_text("utf-8"))
    constants: dict[str, Any] = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"const\s+int\s+(\w+)\s*=\s*(\d+)\s*;", core_src)
    }
    reader = _DartSchemaReader(constants)
    reader.constants["emptyObjectSchema"] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    reader.constants["idPropertySchema"] = {
        "type": "object",
        "properties": {"id": {"type": "string", "pattern": r"^[a-z0-9_.-]{1,64}$"}},
        "required": ["id"],
        "additionalProperties": False,
    }
    reader.constants["path"] = {"type": "string", "minLength": 1, "maxLength": 512}

    exec_src = _strip_dart_comments((XTP_SCHEMAS_DIR / "exec_fs_catalog.dart").read_text("utf-8"))
    for m in re.finditer(r"(?:const|final)\s+Map<String,\s*Object\?>\s+(\w+)\s*=\s*", exec_src):
        idx = m.end()
        end_idx = idx
        depth = 0
        in_str: str | None = None
        while end_idx < len(exec_src):
            c = exec_src[end_idx]
            if in_str:
                if c == "\\" and end_idx + 1 < len(exec_src):
                    end_idx += 2
                    continue
                if c == in_str:
                    in_str = None
            else:
                if c in ("'", '"'):
                    in_str = c
                elif c in "{[(":
                    depth += 1
                elif c in "}])":
                    depth -= 1
                elif c == ";" and depth == 0:
                    break
            end_idx += 1
        reader.constants[m.group(1)] = reader.eval_expr(exec_src[idx:end_idx])

    descriptors: dict[str, dict[str, object]] = {}
    for dart_file in sorted(XTP_SCHEMAS_DIR.glob("*.dart")):
        if dart_file.name == "core_types.dart":
            continue
        src = _strip_dart_comments(dart_file.read_text("utf-8"))
        for m in re.finditer(r"ToolDescriptor\s*\(", src):
            block, _ = _extract_balanced(src, m.end() - 1, "(", ")")
            entry: dict[str, Any] = {}
            for field in reader._split_top_level(block[1:-1], ","):
                if ":" not in field:
                    continue
                k, v = field.split(":", 1)
                key = k.strip()
                if key in ("name", "inputSchema"):
                    entry[key] = reader.eval_expr(v)
            descriptors[str(entry["name"])] = entry["inputSchema"]
    return descriptors


def _load_mobile_dart_descriptors() -> dict[str, dict[str, object]]:
    reader = _DartSchemaReader({})
    descriptors: dict[str, dict[str, object]] = {}
    for dart_file in sorted(MOBILE_HOSTS_DIR.rglob("*.dart")):
        src = _strip_dart_comments(dart_file.read_text("utf-8"))
        for m in re.finditer(r"'name'\s*:\s*'([a-z]+\.[a-z0-9_]+)'", src):
            tool_name = m.group(1)
            in_m = re.search(r"'inputSchema'\s*:\s*", src[m.end() : m.end() + 2000])
            if in_m is None:
                continue
            brace = src.find("{", m.end() + in_m.end())
            if brace < 0:
                continue
            block, _ = _extract_balanced(src, brace, "{", "}")
            descriptors[tool_name] = reader.eval_expr(block)
    return descriptors


def _load_fixture_descriptors() -> dict[str, dict[str, object]]:
    descriptors: dict[str, dict[str, object]] = {}
    for fixture_name in ("sample_frames.json", "exec_fs_frames.json"):
        data = json.loads((FIXTURES_DIR / fixture_name).read_text("utf-8"))
        frames = data.get("frames", {})
        if not isinstance(frames, dict):
            continue
        for frame in frames.values():
            if not isinstance(frame, dict):
                continue
            result = frame.get("result")
            if not isinstance(result, dict):
                continue
            tools = result.get("tools")
            if not isinstance(tools, list):
                continue
            for item in tools:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    descriptors[item["name"]] = item["inputSchema"]
    return descriptors


def _backend_builtin_schemas() -> dict[str, dict[str, object]]:
    schemas: dict[str, dict[str, object]] = {
        tool_names.BOARD_FUNCTION_NAME: tool_names.BOARD_MEDIA_KEY_SCHEMA,
        tool_names.WORKFLOW_TEMPLATE_FUNCTION_NAME: tool_names.WORKFLOW_TEMPLATE_SCHEMA,
        tool_names.AUTONOMOUS_JOB_FUNCTION_NAME: tool_names.AUTONOMOUS_JOB_SCHEMA,
        watch.JOB_WAIT_FUNCTION_NAME: watch.JOB_WAIT_SCHEMA,
    }
    for spec in tool_discovery.build_meta_tool_specs({}, frozenset()):
        schemas[spec.name] = spec.parameters
    for spec in tool_discovery.build_mesh_tool_specs():
        schemas[spec.name] = spec.parameters
    for spec in tool_discovery.build_mesh_pipeline_tool_specs():
        schemas[spec.name] = spec.parameters
    return schemas


def _known_backend_function_names_and_prefixes() -> tuple[set[str], set[str]]:
    host_dotted = set(catalog._PRIORITY_ORDER) | set(_load_fixture_descriptors())
    fn_names = {tool_names.to_function_name(d) for d in host_dotted}
    fn_names |= set(_backend_builtin_schemas())
    prefixes = {d.split(".", 1)[0] for d in host_dotted} | {
        fn.split("__", 1)[0] for fn in fn_names if "__" in fn
    }
    return fn_names, prefixes


def _module_path(mod: ModuleType) -> Path:
    file_path = mod.__file__
    assert file_path is not None, f"Module {mod.__name__} has no __file__"
    return Path(file_path)


def _extract_prompt_strings_from_module(path: Path) -> list[tuple[str, str]]:
    """Extracts string constants from module-level prompt globals and hint/prompt functions."""
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    out.append((f"{path.name}:{target.id}", node.value.value))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    out.append((f"{path.name}:{node.name}:L{sub.lineno}", sub.value))
    return out


def _find_unknown_tool_references(
    texts: list[tuple[str, str]],
    known_fn_names: set[str],
    known_prefixes: set[str],
) -> list[str]:
    errors: list[str] = []
    for source, text in texts:
        for fn in sorted(set(_FN_TOOL_RE.findall(text))):
            if fn.startswith(_DYNAMIC_CUSTOM_PREFIX):
                if "forge" not in known_prefixes or "forge__run_skill" not in known_fn_names:
                    errors.append(f"{source}: dynamic prefix {fn!r} has no forge runtime")
                continue
            if fn not in known_fn_names:
                errors.append(f"{source}: unknown tool reference {fn!r}")
            dotted = tool_names.from_function_name(fn)
            if dotted is not None and tool_names.is_forbidden_tool_name(dotted):
                errors.append(f"{source}: forbidden tool referenced {fn!r}")
        for prefix in sorted(set(_WILD_PREFIX_RE.findall(text))):
            if prefix not in known_prefixes:
                errors.append(f"{source}: unknown tool wildcard prefix '{prefix}__*'")
    return errors


def _validate_call_signatures_in_text(
    source: str,
    text: str,
    schemas_by_fn: dict[str, dict[str, object]],
) -> list[str]:
    """Checks every `tool__name(arg1, arg2)` signature written in a prompt against real schemas."""
    errors: list[str] = []
    for match in _CALL_SIG_RE.finditer(text):
        fn_name = match.group(1)
        arg_names = [a.strip() for a in match.group(2).split(",") if a.strip()]
        schema = schemas_by_fn.get(fn_name)
        if schema is None:
            errors.append(f"{source}: call signature {match.group(0)!r} has no schema")
            continue
        props_raw = schema.get("properties")
        props = props_raw if isinstance(props_raw, dict) else {}
        for arg_name in arg_names:
            if arg_name not in props and arg_name != "target":
                errors.append(
                    f"{source}: call signature {match.group(0)!r} mentions unknown arg "
                    f"{arg_name!r} (allowed: {sorted(props)})"
                )
    return errors


def _extract_kwarg_literals(text: str, key: str) -> list[str]:
    """Extracts string literals written as `key="val"` or `key='val'` in prompt text."""
    pat = re.compile(rf"\b{re.escape(key)}\s*=\s*['\"]([a-zA-Z0-9_.-]+)['\"]")
    return pat.findall(text)


# ---------------------------------------------------------------------------
# 1. Backend-only invariants (always run, even when backend repo is standalone)
# ---------------------------------------------------------------------------


def test_catalog_priority_order_and_builtin_schemas_are_valid() -> None:
    priority = catalog._PRIORITY_ORDER
    assert len(priority) == len(set(priority)), "Duplicate entries in catalog._PRIORITY_ORDER"
    for dotted in priority:
        validate_tool_name(dotted)
        assert not tool_names.is_forbidden_tool_name(dotted), f"Forbidden tool in catalog: {dotted}"
        tool_names.to_function_name(dotted)

    for fn_name, schema in _backend_builtin_schemas().items():
        SchemaValidator.validate_schema_definition(schema)
        assert fn_name == fn_name.strip() and fn_name


def test_prompts_and_dev_modules_reference_only_existing_catalog_tools_and_prefixes() -> None:
    known_fn_names, known_prefixes = _known_backend_function_names_and_prefixes()

    # Every gating set in prompts.py / prompt_dev_modules.py / exec_fs_support.py must exist.
    for fn in prompts._UI_AUTOMATION_TOOLS | prompts._FORGE_SYNTHESIS_TOOLS:
        assert fn in known_fn_names, f"Prompt gating set references unknown tool: {fn}"
    for fn in prompt_dev_modules.DIRECT_DEV_TOOLS:
        assert fn in known_fn_names, f"DIRECT_DEV_TOOLS references unknown tool: {fn}"
    assert prompt_dev_modules.AUTONOMOUS_JOB_FUNCTION in known_fn_names
    for dotted in exec_fs_support.EXEC_FS_PREFLIGHT_TOOLS:
        assert tool_names.to_function_name(dotted) in known_fn_names

    prompt_modules = (
        _module_path(prompts),
        _module_path(prompt_dev_modules),
        _module_path(exec_fs_support),
        _module_path(watch_coordinator),
        _module_path(job_runner),
        _module_path(cu_template),
    )
    texts: list[tuple[str, str]] = []
    for mod_path in prompt_modules:
        texts.extend(_extract_prompt_strings_from_module(mod_path))
    # Also exercise rendered runtime outputs (tool_system_prompt, prompt_note, verify instructions).
    texts.extend(
        [
            ("SYSTEM_PROMPT_VI", prompts.SYSTEM_PROMPT_VI),
            ("SYSTEM_PROMPT_EN", prompts.SYSTEM_PROMPT_EN),
            ("exec_async_verify", exec_fs_support.exec_async_verify_instruction("exec_123")),
            (
                "cu_template_vi",
                cu_template.CuTemplatePin.create(
                    name="Demo",
                    host_id="11111111-1111-4111-8111-111111111111",
                    apps=["editor"],
                    domains=["example.com"],
                    max_steps=10,
                    max_minutes=5,
                ).prompt_note(Locale.VI, "HostA"),
            ),
            (
                "cu_template_en",
                cu_template.CuTemplatePin.create(
                    name="Demo",
                    host_id="11111111-1111-4111-8111-111111111111",
                    apps=["editor"],
                    domains=["example.com"],
                    max_steps=10,
                    max_minutes=5,
                ).prompt_note(Locale.EN, "HostA"),
            ),
        ]
    )

    errors = _find_unknown_tool_references(texts, known_fn_names, known_prefixes)
    assert not errors, "Prompt <-> catalog drift detected:\n" + "\n".join(errors)


def test_prompts_vi_and_en_modules_keep_identical_tool_and_prefix_parity() -> None:
    pairs = (
        ("CORE_TOOL_GUARDRAILS", prompts.CORE_TOOL_GUARDRAILS_VI, prompts.CORE_TOOL_GUARDRAILS_EN),
        (
            "_TOOL_DISCOVERY_MODULE",
            prompts._TOOL_DISCOVERY_MODULE_VI,
            prompts._TOOL_DISCOVERY_MODULE_EN,
        ),
        (
            "_EXECUTION_LADDER_MODULE",
            prompts._EXECUTION_LADDER_MODULE_VI,
            prompts._EXECUTION_LADDER_MODULE_EN,
        ),
        ("_UI_SKILL_MODULE", prompts._UI_SKILL_MODULE_VI, prompts._UI_SKILL_MODULE_EN),
        ("_FORGE_SKILL_MODULE", prompts._FORGE_SKILL_MODULE_VI, prompts._FORGE_SKILL_MODULE_EN),
        (
            "_DIRECT_DEV_MODULE",
            prompt_dev_modules._DIRECT_DEV_MODULE_VI,
            prompt_dev_modules._DIRECT_DEV_MODULE_EN,
        ),
        (
            "_JOB_ESCALATION_MODULE",
            prompt_dev_modules._JOB_ESCALATION_MODULE_VI,
            prompt_dev_modules._JOB_ESCALATION_MODULE_EN,
        ),
    )
    for label, vi, en in pairs:
        assert set(_FN_TOOL_RE.findall(vi)) == set(
            _FN_TOOL_RE.findall(en)
        ), f"VI/EN tool-name drift in {label}"
        assert set(_WILD_PREFIX_RE.findall(vi)) == set(
            _WILD_PREFIX_RE.findall(en)
        ), f"VI/EN wildcard-prefix drift in {label}"


def test_backend_owned_and_fixture_prompt_argument_examples_pass_schema_validator() -> None:
    builtin_schemas = _backend_builtin_schemas()
    fixture_schemas = {
        tool_names.to_function_name(k): v for k, v in _load_fixture_descriptors().items()
    }
    combined = {**fixture_schemas, **builtin_schemas}

    # 1. Dynamic check of every `mode='...'` and `continuationMode='...'` in prompts & job_runner
    job_runner_src = Path(job_runner.__file__).read_text("utf-8")
    wait_modes_from_prompts = _extract_kwarg_literals(
        prompt_dev_modules._DIRECT_DEV_MODULE_VI
        + "\n"
        + prompt_dev_modules._DIRECT_DEV_MODULE_EN
        + "\n"
        + job_runner_src,
        "mode",
    )
    assert wait_modes_from_prompts, "Expected at least one job__wait mode example in prompts"
    for mode_val in sorted(set(wait_modes_from_prompts)):
        args: dict[str, object] = {
            "mode": mode_val,
            "doneCriteria": "Build finishes cleanly",
        }
        if mode_val == "exec_async":
            args["execTaskId"] = "exec_abc123"
        elif mode_val == "fs_quiescence":
            args["workspaceAlias"] = "my_project"
        elif mode_val == "protocol_async":
            args["mcpServerId"] = "srv_1"
            args["protocolTaskId"] = "task_1"
        SchemaValidator.validate_arguments(watch.JOB_WAIT_SCHEMA, args)

    cont_modes = _extract_kwarg_literals(job_runner_src, "continuationMode")
    assert cont_modes, "Expected continuationMode example in job_runner"
    for cont_val in sorted(set(cont_modes)):
        SchemaValidator.validate_arguments(
            watch.JOB_WAIT_SCHEMA,
            {
                "mode": "ui_quiescence",
                "doneCriteria": "Task completes",
                "continuationMode": cont_val,
            },
        )

    # 2. Dynamic check of `autonomous_job` action examples in prompt_dev_modules
    job_actions = _extract_kwarg_literals(
        prompt_dev_modules._JOB_ESCALATION_MODULE_VI
        + "\n"
        + prompt_dev_modules._JOB_ESCALATION_MODULE_EN,
        "action",
    )
    assert job_actions, "Expected autonomous_job action example in prompt_dev_modules"
    for action_val in sorted(set(job_actions)):
        SchemaValidator.validate_arguments(
            tool_names.AUTONOMOUS_JOB_SCHEMA,
            {"action": action_val, "goal": "Run build and verify tests"},
        )

    # 3. Call signatures in tool_discovery.py and prompts.py against available schemas
    td_texts = _extract_prompt_strings_from_module(Path(tool_discovery.__file__))
    sig_errors: list[str] = []
    for source, text in td_texts:
        sig_errors.extend(_validate_call_signatures_in_text(source, text, combined))
    assert not sig_errors, "\n".join(sig_errors)

    # 4. Concrete fixture-backed argument examples from prompts & dev modules
    SchemaValidator.validate_arguments(combined["app__launch"], {"id": "demo_app"})
    SchemaValidator.validate_arguments(
        combined["tool_invoke"],
        {"name": "media.next", "argumentsJson": "{}"},
    )
    SchemaValidator.validate_arguments(
        combined["fs__patch_file"],
        {"path": "/workspace/a.txt", "target": "old", "replacement": "new"},
    )
    SchemaValidator.validate_arguments(
        combined["exec__run"],
        {
            "command": "echo ok",
            "cwd": "/workspace",
            "expectFiles": [
                {
                    "path": "/workspace/out.bin",
                    "minBytes": 1,
                    "magic": "25504446",
                    "sha256": "a" * 64,
                }
            ],
        },
    )
    SchemaValidator.validate_arguments(combined["exec__status"], {"taskId": "exec_123"})


def test_drift_detector_fails_closed_on_mutated_tool_prefix_or_invalid_args() -> None:
    known_fn_names, known_prefixes = _known_backend_function_names_and_prefixes()
    bad_texts = [
        ("mutated_tool", "Call `media__play_v2` to resume playback."),
        ("mutated_prefix", "Use `telepathy__*` tools directly."),
        ("forbidden_tool", "Fallback to `shell__exec` when stuck."),
    ]
    errors = _find_unknown_tool_references(bad_texts, known_fn_names, known_prefixes)
    assert any("media__play_v2" in e for e in errors)
    assert any("telepathy__*" in e for e in errors)
    assert any("shell__exec" in e for e in errors)

    with pytest.raises(ArgumentValidationError):
        SchemaValidator.validate_arguments(
            watch.JOB_WAIT_SCHEMA,
            {"mode": "window_idle", "doneCriteria": "Should fail closed"},
        )

    bad_sig_errors = _validate_call_signatures_in_text(
        "mutated_sig",
        "Call `tool_invoke(name, bogusArg)` now.",
        _backend_builtin_schemas(),
    )
    assert any("bogusArg" in e for e in bad_sig_errors)


# ---------------------------------------------------------------------------
# 2. Cross-repo descriptor parity (monorepo / ecosystem-ci)
# ---------------------------------------------------------------------------


def _require_monorepo_descriptors() -> (
    tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]
):
    if not XTP_SCHEMAS_DIR.exists() or not MOBILE_HOSTS_DIR.exists():
        pytest.skip("Sibling desktop/ or mobile/ repository not checked out in standalone CI")
    desktop = _load_desktop_dart_descriptors()
    mobile = _load_mobile_dart_descriptors()
    assert len(desktop) >= 100, f"Expected >= 100 desktop descriptors, parsed {len(desktop)}"
    assert len(mobile) >= 27, f"Expected >= 27 mobile descriptors, parsed {len(mobile)}"
    return desktop, mobile


def test_all_priority_order_and_prompt_tools_exist_in_real_host_descriptors() -> None:
    desktop, mobile = _require_monorepo_descriptors()
    all_host_dotted = set(desktop) | set(mobile)

    # Every entry in catalog._PRIORITY_ORDER must map to a real desktop or mobile descriptor.
    missing_from_hosts = sorted(set(catalog._PRIORITY_ORDER) - all_host_dotted)
    assert not missing_from_hosts, (
        "catalog._PRIORITY_ORDER lists tools absent from desktop/mobile descriptors: "
        f"{missing_from_hosts}"
    )

    # Every tool/prefix referenced in prompts must exist in real descriptors or builtins.
    real_fn_names = {tool_names.to_function_name(d) for d in all_host_dotted} | set(
        _backend_builtin_schemas()
    )
    real_prefixes = {d.split(".", 1)[0] for d in all_host_dotted} | {
        fn.split("__", 1)[0] for fn in real_fn_names if "__" in fn
    }
    texts: list[tuple[str, str]] = []
    for mod in (
        prompts,
        prompt_dev_modules,
        exec_fs_support,
        watch_coordinator,
        job_runner,
        cu_template,
    ):
        texts.extend(_extract_prompt_strings_from_module(_module_path(mod)))
    errors = _find_unknown_tool_references(texts, real_fn_names, real_prefixes)
    assert not errors, "Prompt <-> real descriptor drift detected:\n" + "\n".join(errors)


def test_no_new_host_descriptor_is_silently_dropped_by_schema_validator() -> None:
    desktop, mobile = _require_monorepo_descriptors()

    desktop_rejected: dict[str, str] = {}
    for name, schema in sorted(desktop.items()):
        try:
            SchemaValidator.validate_schema_definition(schema)
        except SchemaDefinitionError as exc:
            desktop_rejected[name] = str(exc)

    unexpected_new = sorted(set(desktop_rejected) - _PREEXISTING_SCHEMA_EXCLUSIONS)
    resolved_old = sorted(_PREEXISTING_SCHEMA_EXCLUSIONS - set(desktop_rejected))
    assert set(desktop_rejected) == _PREEXISTING_SCHEMA_EXCLUSIONS, (
        "Desktop descriptor SchemaValidator rejection set changed! "
        f"Unexpected new rejections: {unexpected_new}; "
        f"Resolved exclusions (update baseline): {resolved_old}; "
        f"Details: {desktop_rejected}"
    )

    mobile_rejected: dict[str, str] = {}
    for name, schema in sorted(mobile.items()):
        try:
            SchemaValidator.validate_schema_definition(schema)
        except SchemaDefinitionError as exc:
            mobile_rejected[name] = str(exc)
    assert not mobile_rejected, f"Mobile descriptors rejected by SchemaValidator: {mobile_rejected}"

    # Fixture descriptors must pass SchemaValidator and stay compatible with Dart builders.
    # `exec_fs_frames.json` is generated directly from `exec_fs_catalog.dart` (exact parity);
    # `sample_frames.json` is a minimal MVP wire sample (every property must still exist in Dart).
    exec_fs_data = json.loads((FIXTURES_DIR / "exec_fs_frames.json").read_text("utf-8"))
    exec_fs_tools = exec_fs_data["frames"]["toolsListExecFsResponse"]["result"]["tools"]
    for item in exec_fs_tools:
        name = item["name"]
        SchemaValidator.validate_schema_definition(item["inputSchema"])
        assert name in desktop, f"exec_fs fixture descriptor {name!r} missing from Dart"
        assert (
            desktop[name] == item["inputSchema"]
        ), f"exec_fs fixture inputSchema drifted from Dart for {name!r}"

    sample_data = json.loads((FIXTURES_DIR / "sample_frames.json").read_text("utf-8"))
    sample_tools = sample_data["frames"]["toolsListResponse"]["result"]["tools"]
    for item in sample_tools:
        name = item["name"]
        SchemaValidator.validate_schema_definition(item["inputSchema"])
        assert name in desktop, f"sample fixture descriptor {name!r} missing from Dart"
        sample_props = item["inputSchema"].get("properties", {})
        dart_props = desktop[name].get("properties", {})
        assert isinstance(sample_props, dict) and isinstance(dart_props, dict)
        for prop_name, prop_schema in sample_props.items():
            assert (
                dart_props.get(prop_name) == prop_schema
            ), f"sample fixture property {name}.{prop_name} drifted from Dart"


def test_all_prompt_call_signatures_and_argument_examples_validate_against_real_descriptors() -> (
    None
):
    desktop, mobile = _require_monorepo_descriptors()
    # Desktop schemas take precedence when a tool exists on both desktop and mobile
    # (e.g. app.launch); mobile-only tools (ui.observe, ui.act, etc.) come from mobile.
    schemas_by_fn: dict[str, dict[str, object]] = {
        **{tool_names.to_function_name(k): v for k, v in mobile.items()},
        **{tool_names.to_function_name(k): v for k, v in desktop.items()},
        **_backend_builtin_schemas(),
    }

    # 1. Every `tool__name(arg1, arg2)` call signature in prompts.py & prompt_dev_modules.py
    sig_errors: list[str] = []
    for mod in (prompts, prompt_dev_modules, tool_discovery):
        for source, text in _extract_prompt_strings_from_module(_module_path(mod)):
            sig_errors.extend(_validate_call_signatures_in_text(source, text, schemas_by_fn))
    assert not sig_errors, "Prompt call signature drift:\n" + "\n".join(sig_errors)

    # 2. Extract and validate every concrete parameter example from prompts.py & companion prompts
    # `app__launch(id, openTarget)` and `app__launch(id)`
    SchemaValidator.validate_arguments(schemas_by_fn["app__launch"], {"id": "demo_app"})
    SchemaValidator.validate_arguments(
        schemas_by_fn["app__launch"],
        {"id": "demo_app", "openTarget": "my_project"},
    )
    # `ui__inspect` (`interactiveOnly: true`)
    assert "interactiveOnly: true" in prompts.SYSTEM_PROMPT_VI
    assert "interactiveOnly: true" in prompts.SYSTEM_PROMPT_EN
    SchemaValidator.validate_arguments(
        schemas_by_fn["ui__inspect"],
        {"scope": "active_window", "maxNodes": 50, "interactiveOnly": True},
    )

    # `cu__session_start` (`goal`, `apps`, `domains`, `maxSteps`, `maxMinutes`)
    for req_key in ("goal", "apps", "domains", "maxSteps", "maxMinutes"):
        assert req_key in prompts.SYSTEM_PROMPT_VI
        assert req_key in prompts.SYSTEM_PROMPT_EN
    SchemaValidator.validate_arguments(
        schemas_by_fn["cu__session_start"],
        {
            "goal": "Inspect active document",
            "apps": ["demo_app"],
            "domains": ["example.com"],
            "maxSteps": 10,
            "maxMinutes": 5,
        },
    )

    # `input__click`: elementId, elementName, xNorm/yNorm, clickCount, hover, menuPath
    click_counts = [
        int(m.group(1))
        for m in re.finditer(
            r"clickCount\s*[:=]\s*(\d+)",
            prompts.SYSTEM_PROMPT_VI
            + "\n"
            + prompts.SYSTEM_PROMPT_EN
            + "\n"
            + (prompts.state_driven_hint({"effect": "no_change"}) or ""),
        )
    ]
    assert click_counts, "Expected clickCount example in prompts.py"
    for count in set(click_counts):
        SchemaValidator.validate_arguments(
            schemas_by_fn["input__click"],
            {"elementId": "@s1:e1", "elementName": "Item", "clickCount": count},
        )

    hover_vals = re.findall(
        r"hover\s*[:=]\s*(true|false)",
        prompts.SYSTEM_PROMPT_VI
        + "\n"
        + prompts.SYSTEM_PROMPT_EN
        + "\n"
        + (prompts.state_driven_hint({"effect": "no_change"}) or ""),
    )
    assert hover_vals, "Expected hover example in prompts.py"
    for raw_bool in set(hover_vals):
        SchemaValidator.validate_arguments(
            schemas_by_fn["input__click"],
            {"xNorm": 0.2, "yNorm": 0.4, "hover": raw_bool == "true"},
        )

    SchemaValidator.validate_arguments(
        schemas_by_fn["input__click"],
        {"menuPath": "File > Open..."},
    )

    # `forge__synthesize_skill` (`scriptCode`, `testArgsJson`, `skillIdToUpdate`)
    dry_run_hint = prompts.state_driven_hint(
        {"dryRunOk": False, "stderrPreview": "SyntaxError", "skillId": "sk_1"}
    )
    assert dry_run_hint is not None and "skillIdToUpdate='sk_1'" in dry_run_hint
    SchemaValidator.validate_arguments(
        schemas_by_fn["forge__synthesize_skill"],
        {
            "goal": "Parse table",
            "scriptCode": "print('{}')",
            "testArgsJson": "{}",
            "skillIdToUpdate": "sk_1",
        },
    )

    # Android `ui__act` (`ref`)
    SchemaValidator.validate_arguments(
        schemas_by_fn["ui__act"],
        {"snapshot_id": "snap_1", "type": "activate", "ref": "e1", "label": "OK"},
    )

    # `mcp__task_result` (`server`, `protocolTaskId` from watch_coordinator wake prompt)
    wc_src = Path(watch_coordinator.__file__).read_text("utf-8")
    assert "server={srv}, protocolTaskId={tid}" in wc_src
    SchemaValidator.validate_arguments(
        schemas_by_fn["mcp__task_result"],
        {"server": "srv_1", "protocolTaskId": "task_1"},
    )
