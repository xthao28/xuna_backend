"""UDC-17 S17.4: `ui.find` and the `changes` an `input.*` result reports.

The backend redacts what the host returns (its own check, the host redacts too), keeps the
session's node cache in step with new and dropped refs, puts `ui.find` right after
`ui.inspect` in the direct catalog, and never shows or stores the model's search text.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from tests.unit.test_udc17_preflight import _build, _descriptor

from xuna_backend.modules.actions.application.catalog import _PRIORITY_ORDER, MAX_CATALOG_TOOLS
from xuna_backend.modules.actions.application.prompts import (
    _UI_AUTOMATION_TOOLS,
    encode_tool_payload,
    state_driven_hint,
)
from xuna_backend.modules.actions.domain.action_summary import ActionSummary
from xuna_backend.modules.actions.domain.entities import MAX_TOOL_OUTPUT_BYTES
from xuna_backend.modules.cognition.domain.types import Locale
from xuna_backend.modules.hosts.domain.entities import RiskTier, ToolFamily

_FIND = _descriptor(
    "ui.find",
    RiskTier.R2,
    {
        "query": {"type": "string"},
        "role": {"type": "string"},
        "maxResults": {"type": "integer"},
    },
    ToolFamily.UI,
)

# A careless host that forgot to redact: the backend must still hide the field's label.
_LEAKY_PASSWORD = {
    "id": "p1",
    "ref": "@s1:e9",
    "role": "AXSecureTextField",
    "name": "Mật khẩu ngân hàng",
    "value": "hunter2",
}


def test_ui_find_matches_are_redacted_cached_and_need_no_approval() -> None:
    def host(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if name != "ui.find":
            return None
        return {
            "appId": "explorer",
            "windowTitle": "Reports",
            "snapshotId": "s1",
            "searchedNodes": 12,
            "total": 2,
            "truncated": False,
            "matches": [
                {"id": "7", "ref": "@s1:e7", "role": "button", "name": "Gửi báo cáo"},
                dict(_LEAKY_PASSWORD),
            ],
        }

    async def scenario() -> None:
        world = await _build([], extra_tools=(_FIND,), structured_for=host)
        body = await world.call("ui__find", {"query": "báo cáo"})
        assert body["status"] == "ok"
        assert world.pending == []  # R2 inside the session: no prompt
        matches = body["output"]["matches"]
        assert matches[0]["name"] == "Gửi báo cáo"
        assert matches[1]["name"] == "•••"
        assert "value" not in matches[1]
        assert "hunter2" not in str(body)
        assert "ngân hàng" not in str(body)

        state = await world.cu_store.get("cu-1")
        assert state is not None
        # Joins the cache: the entry from the earlier inspect stays.
        assert "@s1:e3" in state.cached_nodes
        assert state.cached_nodes["@s1:e7"]["name"] == "Gửi báo cáo"
        assert "s1:e7" in state.cached_nodes

    asyncio.run(scenario())


def test_input_changes_are_redacted_and_update_the_cache() -> None:
    def host(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if name != "input.click":
            return None
        return {
            "effect": "changed",
            "changes": {
                "snapshotId": "s1",
                "added": [
                    {"id": "8", "ref": "@s1:e8", "role": "button", "name": "Đóng"},
                    dict(_LEAKY_PASSWORD),
                ],
                "removed": ["@s1:e3"],
                "changed": [],
                "truncated": False,
            },
        }

    async def scenario() -> None:
        world = await _build(
            [{"status": "ok", "effectiveRisk": "R2", "reasonCode": "navigation"}],
            structured_for=host,
        )
        body = await world.call("input__click", {"elementId": "@s1:e5", "elementName": "Mở"})
        assert body["status"] == "ok"
        added = body["output"]["changes"]["added"]
        assert [a["name"] for a in added] == ["Đóng", "•••"]
        assert "hunter2" not in str(body)

        state = await world.cu_store.get("cu-1")
        assert state is not None
        assert "@s1:e3" not in state.cached_nodes  # dropped by the host
        assert state.cached_nodes["@s1:e8"]["name"] == "Đóng"

    asyncio.run(scenario())


def test_ui_find_is_a_direct_tool_right_after_ui_inspect() -> None:
    idx = _PRIORITY_ORDER.index("ui.inspect")
    assert _PRIORITY_ORDER[idx + 1] == "ui.find"
    assert _PRIORITY_ORDER.index("ui.find") < MAX_CATALOG_TOOLS
    assert "ui__find" in _UI_AUTOMATION_TOOLS


def test_ui_find_summary_never_carries_the_search_text() -> None:
    for locale in (Locale.VI, Locale.EN):
        summary = ActionSummary.format_pending(
            "ui.find", {"query": "Bí mật 1234", "role": "button"}, "PC Thảo", locale
        )
        assert "Bí mật" not in summary
        assert "PC Thảo" in summary


def test_stale_ref_hint_offers_ui_find() -> None:
    hint = state_driven_hint({"errorCode": "stale_ref"})
    assert hint is not None
    assert "ui__find" in hint


# Longest name the host sends (60 chars, Vietnamese: several bytes per char).
_LONG_NAME = ("Mở thư mục báo cáo tài chính quý bốn năm hai nghìn không trăm" * 2)[:60]


def test_a_full_changes_result_fits_the_tool_output_limit() -> None:
    # The host's slim `changes.added` shape (UDC-17 S17.4), every list at its cap of 15.
    added = [
        {
            "ref": f"@s00000001:e{100 + i}",
            "id": f"ui_{100 + i}",
            "role": "axmenuitem",
            "name": _LONG_NAME,
            "enabled": True,
            "visible": True,
            "editable": False,
            "modal": False,
        }
        for i in range(15)
    ]
    output: dict[str, object] = {
        "effect": "changed",
        "route": "semantic_headless",
        "disposition": {"delivery": "delivered_verified", "retry": "unsafe"},
        "snapshotId": "s00000001",
        "surfaces": [
            {"id": f"ui_{i}", "kind": "dialog", "title": _LONG_NAME, "modal": True}
            for i in range(3)
        ],
        "changes": {
            "snapshotId": "s00000001",
            "added": added,
            "removed": [f"@s00000001:e{i}" for i in range(15)],
            "changed": [
                {"ref": f"@s00000001:e{i}", "enabled": True, "visible": True} for i in range(15)
            ],
            "truncated": True,
        },
    }
    encoded = encode_tool_payload("ok", output)
    assert len(encoded.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES
    body = json.loads(encoded)
    assert body["output"]["effect"] == "changed"
    assert len(body["output"]["changes"]["added"]) == 15


def test_ten_full_ui_find_matches_fit_the_tool_output_limit() -> None:
    matches = [
        {
            "id": f"ui_{i}",
            "ref": f"@s00000001:e{i}",
            "role": "axmenuitem",
            "name": _LONG_NAME,
            "enabled": True,
            "bounds": {"x": 12345, "y": 12345, "w": 1234, "h": 123},
            "visible": True,
            "editable": False,
            "occluded": False,
            "modal": False,
            "parentSurfaceId": "ui_999",
            "depth": 12,
            "parentId": "ui_998",
            "childrenCount": 40,
        }
        for i in range(10)
    ]
    output: dict[str, object] = {
        "appId": "explorer",
        "windowTitle": (_LONG_NAME * 2)[:120],
        "snapshotId": "s00000001",
        "searchedNodes": 60,
        "total": 60,
        "truncated": True,
        "matches": matches,
    }
    encoded = encode_tool_payload("ok", output)
    assert len(encoded.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES
    assert len(json.loads(encoded)["output"]["matches"]) == 10
