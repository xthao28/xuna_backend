"""Enforces the layer rules in AGENTS.md §2 on every import in src/."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "xuna_backend"
PKG = "xuna_backend"
FRAMEWORKS = ("fastapi", "starlette", "sqlalchemy", "alembic", "psycopg", "httpx", "uvicorn")
LAYERS = ("domain", "application", "infrastructure", "api")
# layer -> layers of any module it may import
ALLOWED = {
    "domain": {"domain"},
    "application": {"domain", "application"},
    "infrastructure": {"domain", "application", "infrastructure"},
    "api": {"domain", "application", "api"},
}
COMPOSITION_ROOTS = {"main.py", "wiring.py"}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def _layer(parts: tuple[str, ...]) -> tuple[str, str] | None:
    # parts relative to SRC: ("modules", <module>, <layer>, ...)
    if len(parts) >= 3 and parts[0] == "modules" and parts[2] in LAYERS:
        return parts[1], parts[2]
    return None


def test_layers_and_purity() -> None:
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        own = _layer(rel.parts)
        for name in _imports(path):
            top = name.split(".")[0]
            where = f"{rel}: {name}"
            if rel.parts[0] == "core" and name.startswith(f"{PKG}.modules"):
                violations.append(f"{where} (core must not import modules)")
            if own is None or path.name in COMPOSITION_ROOTS:
                continue
            _, layer = own
            if layer in ("domain", "application") and top in FRAMEWORKS:
                violations.append(f"{where} ({layer} must stay framework-free)")
            if layer == "domain" and top == "pydantic":
                violations.append(f"{where} (domain is stdlib-only)")
            if layer in ("domain", "application") and name.startswith(f"{PKG}.core"):
                violations.append(f"{where} ({layer} must not import core)")
            target = _layer(tuple(name.split(".")[1:])) if name.startswith(PKG) else None
            if target is not None and target[1] not in ALLOWED[layer]:
                violations.append(f"{where} ({layer} may not import {target[1]})")
    assert not violations, "\n".join(violations)


def test_no_print_or_vendor_sdk_outside_adapters() -> None:
    vendors = ("openai", "anthropic", "google", "dashscope", "boto3")
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        own = _layer(rel.parts)
        for name in _imports(path):
            if name.split(".")[0] in vendors:
                assert own is not None and own[1] == "infrastructure", f"{rel}: {name}"
