#!/usr/bin/env bash
# One command every agent runs before reporting (AGENTS.md §9).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin}
"$PY/ruff" format --check src tests migrations
"$PY/ruff" check src tests migrations
"$PY/mypy"
if [[ -n "${XUNA_TEST_DATABASE_URL:-}" ]]; then
  XUNA_DATABASE_URL="$XUNA_TEST_DATABASE_URL" "$PY/alembic" upgrade head
fi
"$PY/pytest" -q
