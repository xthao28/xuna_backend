"""Pure logic of `scripts/eval_host_control.py`: task generation, verification, verdicts."""
# ruff: noqa: S311 - seeded RNGs make task sampling reproducible; not used for security.

import importlib.util
import random
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval_host_control.py"
_SPEC = importlib.util.spec_from_file_location("eval_host_control", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)

_APPS = ["Spotify", "Zalo", "Microsoft Excel", "Terminal", "1Password 8", "Figma", "Notes"]
_ALL_TOOLS = {
    "clipboard.write",
    "selection.read",
    "app.launch",
    "window.active_context",
    "ui.accessibility_tree",
    "browser.current_url",
}


def test_parse_app_names_handles_desktop_and_string_shapes() -> None:
    desktop = {"apps": [{"id": "a", "name": "Zalo"}, {"id": "b", "name": "Figma"}]}
    assert ev.parse_app_names(desktop) == ["Zalo", "Figma"]
    assert ev.parse_app_names({"appNames": "Zalo, Figma, Zalo"}) == ["Zalo", "Figma"]


def test_host_tool_names_reads_wire_descriptors() -> None:
    host = {"tools": [{"name": "app.launch"}, {"name": "clipboard.write"}, "bad"]}
    assert ev.host_tool_names(host) == {"app.launch", "clipboard.write"}


def test_capabilities_need_both_action_and_independent_verifier() -> None:
    assert "clipboard_roundtrip" not in ev.available_capabilities(
        {"clipboard.write"}, allow_typing=True
    )
    caps = ev.available_capabilities(_ALL_TOOLS, allow_typing=False)
    assert "type_in_app" not in caps  # typing into apps is opt-in
    assert "type_in_app" in ev.available_capabilities(_ALL_TOOLS, allow_typing=True)
    assert "refuse_unbridged_smarthome" in caps


def test_generated_tasks_use_host_apps_and_never_excluded_ones() -> None:
    caps = ev.available_capabilities(_ALL_TOOLS, allow_typing=True)
    tasks = ev.generate_tasks(_APPS, caps, 40, random.Random(3))
    apps = {t.app for t in tasks if t.app}
    assert apps <= set(_APPS)
    assert not apps & {"Terminal", "1Password 8"}  # credentials and shells are excluded
    assert any(not t.shortcut_app for t in tasks if t.app)  # mostly generalization apps
    assert {t.capability for t in tasks} == set(caps)
    tokens = [t.probe for t in tasks if t.probe]
    assert len(set(tokens)) == len(tokens)  # fresh token per task


def test_generation_is_reproducible_with_a_seed() -> None:
    caps = ev.available_capabilities(_ALL_TOOLS, allow_typing=True)
    first = ev.generate_tasks(_APPS, caps, 10, random.Random(7))
    second = ev.generate_tasks(_APPS, caps, 10, random.Random(7))
    assert first == second


def test_verifiers() -> None:
    task = ev.Task("t", "clipboard_roundtrip", "p", probe="xunaabc123")
    assert ev.verify_clipboard(task, {"text": "xunaabc123"})[0]
    assert not ev.verify_clipboard(task, {"text": ev.CLIPBOARD_BASELINE})[0]
    app_task = ev.Task("t", "open_app", "p", app="Microsoft Excel.app")
    assert ev.verify_open_app(app_task, {"appName": "Microsoft Excel"})[0]
    assert not ev.verify_open_app(app_task, {"appName": "Finder"})[0]
    typed = ev.Task("t", "type_in_app", "p", app="Zalo", probe="xunaqq")
    assert ev.verify_typed(typed, {"compactUiTree": '[ref=ui_2] <AXTextField> "Tìm" = xunaqq'})[0]
    assert ev.verify_url_in_reply("Bạn đang xem vnexpress.net nè", {"host": "www.vnexpress.net"})[0]
    assert not ev.verify_url_in_reply("Mình không rõ", {"url": "https://vnexpress.net/a"})[0]


@pytest.mark.parametrize(
    ("capability", "verified", "statuses", "error", "false_success", "review"),
    [
        # All results ok but nothing happened: XUNA effectively claimed success.
        ("open_app", False, ["ok"], None, True, False),
        # A failure was reported honestly: a fail, but not a false success.
        ("open_app", False, ["unsupported"], None, False, False),
        ("open_app", False, ["ok"], "provider_unavailable", False, False),
        # No tool call at all: the reply must be read to judge it.
        ("open_app", False, [], None, False, True),
        ("open_app", True, ["ok"], None, False, False),
        # Without a bridge any successful action is a false success.
        ("refuse_unbridged_smarthome", None, ["unsupported"], None, False, False),
        ("refuse_unbridged_smarthome", None, ["ok"], None, True, True),
    ],
)
def test_classify(
    capability: str,
    verified: bool | None,
    statuses: list[str],
    error: str | None,
    false_success: bool,
    review: bool,
) -> None:
    outcome = ev.Outcome(task=ev.Task("t", capability, "p"), verified=verified)
    outcome.action_statuses = statuses
    outcome.error = error
    ev.classify(outcome)
    assert outcome.false_success is false_success
    assert outcome.needs_review is review


def test_summary_splits_shortcut_and_plain_apps() -> None:
    plain = ev.Outcome(task=ev.Task("a", "open_app", "p", app="Zalo"), verified=True)
    short = ev.Outcome(
        task=ev.Task("b", "open_app", "p", app="Spotify", shortcut_app=True), verified=False
    )
    short.false_success = True
    summary = ev.summarize([plain, short])
    assert summary["appsWithoutShortcut"]["successRate"] == 1.0
    assert summary["appsWithShortcut"]["falseSuccess"] == 1
    assert summary["overall"]["tasks"] == 2


def test_iter_sse_parses_events() -> None:
    lines = [
        "event: delta",
        'data: {"text": "Mở"}',
        "",
        "event: action_result",
        'data: {"status": "ok"}',
        "",
        "event: finished",
        "data: {}",
    ]
    assert list(ev.iter_sse(lines)) == [
        ("delta", {"text": "Mở"}),
        ("action_result", {"status": "ok"}),
        ("finished", {}),
    ]


def test_run_task_streams_turn_auto_approves_and_verifies_independently() -> None:
    import httpx

    calls: list[tuple[str, str]] = []
    clipboard = {"text": ""}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path.endswith("/turns"):
            sse = (
                'event: action_pending\ndata: {"approvalId": "ap-1"}\n\n'
                'event: action_result\ndata: {"status": "ok"}\n\n'
                'event: delta\ndata: {"text": "Xong rồi nè"}\n\n'
            )
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        if path == "/v1/actions/approvals/ap-1":
            return httpx.Response(200, json={})
        if path.endswith("/tools:invoke"):
            body = request.read().decode()
            if '"clipboard.write"' in body:
                clipboard["text"] = "xuna-eval-baseline"
            # The turn "succeeded" but the host clipboard still holds the baseline.
            return httpx.Response(200, json={"structuredContent": dict(clipboard)})
        return httpx.Response(404)

    client = ev.EvalClient("http://x", "tok", "pet", auto_approve=True)
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    task = ev.Task("t0", "clipboard_roundtrip", "Chép chữ xunaabc vào clipboard", probe="xunaabc")
    outcome = ev.run_task(client, "h1", {"clipboard.write", "selection.read"}, task, 0.0)

    assert ("POST", "/v1/actions/approvals/ap-1") in calls
    assert outcome.approvals == 1
    assert outcome.action_statuses == ["ok"]
    assert outcome.reply == "Xong rồi nè"
    assert outcome.verified is False
    assert outcome.false_success is True  # said ok, independent check disagrees
