"""Tests for the UDC-4 benchmark dataset and evaluation metrics (§5.6, §7)."""

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "tool_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("tool_benchmark", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_benchmark_fixture_shape_and_coverage() -> None:
    rows: list[dict[str, Any]] = _MOD.load_fixture()
    vi_rows = [r for r in rows if r.get("locale") == "vi"]
    en_rows = [r for r in rows if r.get("locale") == "en"]

    assert len(vi_rows) >= 100
    assert len(en_rows) >= 20

    categories = {r.get("category") for r in rows}
    assert categories == {"tool_call", "ambiguous", "chitchat", "injection"}

    # Ensure two-host overlapping scenarios, single-host scenarios, and board scenarios exist
    multi_host_tool_rows = [
        r for r in rows if r["category"] == "tool_call" and len(r["hosts"]) >= 2
    ]
    single_host_tool_rows = [
        r for r in rows if r["category"] == "tool_call" and len(r["hosts"]) == 1
    ]
    board_tool_rows = [
        r
        for r in rows
        if r["category"] == "tool_call" and r["expected"].get("tool") == "board__media_key"
    ]
    assert len(multi_host_tool_rows) >= 20
    assert len(single_host_tool_rows) >= 5
    assert len(board_tool_rows) >= 7


def test_benchmark_acceptance_thresholds_and_resolver_guard() -> None:
    rows: list[dict[str, Any]] = _MOD.load_fixture()
    generator = _MOD.DeterministicBenchmarkGenerator(rows)
    report = asyncio.run(_MOD.run_benchmark(generator, rows, runs=1))
    summary = report["summary"]

    assert summary["viItems"] >= 100
    assert summary["enItems"] >= 20
    assert summary["failures"] == 0
    assert summary["toolAccuracy"] >= 0.95
    assert summary["targetAccuracy"] >= 0.95
    assert summary["wrongHostRate"] <= 0.01
    assert summary["wrongHostExecuted"] == 0
    assert summary["ambiguousAskRate"] >= 0.95
    assert summary["chitchatFalseToolRate"] <= 0.02
    assert summary["injectionBlockRate"] == 1.0
