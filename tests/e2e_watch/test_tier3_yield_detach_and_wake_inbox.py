"""Tier 3 Watch & Wake E2E: `yieldMs` Auto-Detach, Non-Intrusive Wake Inbox, H1 & Schema Parity."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any

import psycopg
import pytest

from tests.e2e_watch.conftest import fetch_scalar, pg_dsn
from tests.e2e_watch.xtp_sim import (
    SimulatedWatchOwnerClient,
    SimulatedWatchXtpHost,
    done_event,
    wait_until,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatRequest,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)
from xuna_backend.modules.conversation.api import routes as conv_routes
from xuna_backend.modules.conversation.api import websocket as conv_ws

pytestmark = [pytest.mark.postgres, pytest.mark.anyio]

_VALID_WAKE_KINDS = frozenset(
    {"job_report", "job_needs_decision", "job_timeout", "job_failed", "watch_alert"}
)
_VALID_SUGGESTED_ACTIONS = frozenset({"extend_watch", "continue_job", "open_approval", "dismiss"})
_VALID_GESTURE_CUES = frozenset({"look_around", "bounce_play", "greet_nuzzle"})
_VALID_WATCH_STATUSES = frozenset({"active", "triggered", "expired", "cancelled", "failed"})


def _assert_tz_iso8601(raw: Any) -> None:
    assert isinstance(raw, str) and ("T" in raw or "t" in raw)
    dt = datetime.fromisoformat(raw)
    assert dt.tzinfo is not None


def _assert_wake_inbox_item_mobile_schema(item: dict[str, Any]) -> None:
    required = {
        "itemId",
        "jobId",
        "hostId",
        "petId",
        "kind",
        "title",
        "summary",
        "suggestedActions",
        "gestureCue",
        "readAt",
        "dismissedAt",
        "createdAt",
    }
    assert required <= set(item.keys())
    assert item["kind"] in _VALID_WAKE_KINDS
    assert 1 <= len(item["title"]) <= 120
    assert len(item["summary"]) <= 2000
    assert isinstance(item["suggestedActions"], list)
    assert set(item["suggestedActions"]) <= _VALID_SUGGESTED_ACTIONS
    if item["gestureCue"] is not None:
        assert item["gestureCue"] in _VALID_GESTURE_CUES
    _assert_tz_iso8601(item["createdAt"])
    if item["readAt"] is not None:
        _assert_tz_iso8601(item["readAt"])
    if item["dismissedAt"] is not None:
        _assert_tz_iso8601(item["dismissedAt"])


def _assert_job_watch_summary_mobile_schema(summary: dict[str, Any]) -> None:
    required = {
        "watchId",
        "jobId",
        "hostId",
        "mode",
        "signals",
        "elapsedSeconds",
        "maxWaitSeconds",
        "wakeChecksUsed",
        "maxWakeChecks",
        "doneCriteria",
        "continuationMode",
        "status",
        "lastEventKind",
        "expiresAt",
        "createdAt",
    }
    assert required <= set(summary.keys())
    assert summary["status"] in _VALID_WATCH_STATUSES
    assert isinstance(summary["signals"], list)
    assert isinstance(summary["elapsedSeconds"], int)
    assert isinstance(summary["maxWaitSeconds"], int)
    assert isinstance(summary["wakeChecksUsed"], int)
    assert isinstance(summary["maxWakeChecks"], int)
    _assert_tz_iso8601(summary["expiresAt"])
    _assert_tz_iso8601(summary["createdAt"])


class _StreamingAwareGenerator:
    """Async generator supporting slow streaming turns + post-wake detached observation."""

    def __init__(self, *, hallucinate_post_wake: bool = False) -> None:
        self.hallucinate_post_wake = hallucinate_post_wake
        self.requests: list[ChatRequest] = []

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        texts = [str(m.content) for m in request.messages]
        last_user = next(
            (str(m.content) for m in reversed(request.messages) if m.role == Role.USER), ""
        )

        # Unrelated quick chat on same Voice WS
        if "2+2" in last_user:
            yield TextDelta("2+2 is 4.")
            yield done_event()
            return

        # Slow streaming turn while background watch settles
        if "Tell me a story" in last_user:
            yield TextDelta("Part 1 ")
            await asyncio.sleep(0.35)
            yield TextDelta("Part 2.")
            yield done_event()
            return

        # Post-wake detached job execution
        if any("[Host watch" in t for t in texts):
            if self.hallucinate_post_wake:
                yield TextDelta("Claiming done without running any tool!")
                yield done_event()
                return
            if not any('"snapshotId"' in t for t in texts):
                yield ToolCallRequested(ToolCall(id="post_obs", name="ui__inspect", arguments={}))
                yield done_event()
                return
            yield TextDelta("Verified post-wake via ui.inspect.")
            yield done_event()
            return

        # Interactive slow tool turn that triggers yieldMs auto-detach
        yield ToolCallRequested(ToolCall(id="slow_media", name="media__play_pause", arguments={}))
        yield done_event()


async def test_t3_01_yield_ms_detach_sse_and_voice_ws_non_intrusive_delivery(
    watch_server_factory: Callable[..., Any],
    clean_watch_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`yieldMs` detaches over SSE & Voice WS, unblocks chat lane, queues WakeInbox mid-turn."""
    monkeypatch.setattr(conv_routes, "DEFAULT_YIELD_MS", 80)
    monkeypatch.setattr(conv_ws, "DEFAULT_YIELD_MS", 80)
    dsn = pg_dsn(clean_watch_db)

    async def _slow_tool_handler(
        tool_name: str, _args: dict[str, Any], _meta: dict[str, Any]
    ) -> dict[str, Any]:
        if tool_name == "media.play_pause":
            await asyncio.sleep(0.45)
            return {"status": "ok", "structuredContent": {"state": "paused"}}
        return {
            "status": "ok",
            "structuredContent": {
                "appId": "editor",
                "pid": 48210,
                "windowId": 1042,
                "snapshotId": "snap_1",
                "nodes": [],
                "status": "settled",
            },
        }

    llm = _StreamingAwareGenerator(hallucinate_post_wake=False)
    async with watch_server_factory(generator=llm) as srv:
        owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer
        )
        host = SimulatedWatchXtpHost(tool_handler=_slow_tool_handler)
        voice_ws = None
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            host_id = await host.connect_enroll(srv.ws_url, owner, "YieldMac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")
            await host.send_presence(foreground_app_id="editor")

            # 1) SSE stream_turn auto-detach
            sse_outcome = await owner.stream_turn(
                pet_id,
                "Run slow task on @YieldMac",
                target_host_id=str(host_id),
            )
            sse_detached = sse_outcome.require("autonomous_job")
            assert sse_detached["state"] == "waiting_watch"
            assert sse_detached["reason"] == "yield_ms"
            assert sse_detached["hostId"] == str(host_id)
            assert sse_detached["watchId"]
            sse_finished = sse_outcome.require("finished")
            assert sse_finished["provider"] == "yield_detach"

            sse_job_id = sse_detached["jobId"]
            sse_job = (await owner.get_job(sse_job_id)).json()
            assert sse_job["state"] == "waiting_watch"
            assert sse_job["subGoals"][0]["toolCallCount"] == 0

            watch_summary = (await owner.get_job_watch(sse_job_id)).json()
            _assert_job_watch_summary_mobile_schema(watch_summary)

            # Cancel SSE detached job so host stays clean for Voice WS test
            await owner.cancel_job(sse_job_id, reason="sse_part_done")

            # 2) Voice WS turn.commit auto-detach
            voice_ws = await owner.connect_voice_ws(pet_id, target_host_id=str(host_id))
            await voice_ws.send_turn_commit(
                "Run slow task on @YieldMac", target_host_id=str(host_id)
            )
            detach_events = await voice_ws.receive_until_turn_done(wait_s=5.0)
            ws_detached = next(
                (ev for ev in detach_events if ev.get("type") == "autonomous_job"), None
            )
            assert ws_detached is not None, f"Expected autonomous_job event, got {detach_events}"
            assert ws_detached["state"] == "waiting_watch"
            assert ws_detached["reason"] == "yield_ms"
            ws_job_id = ws_detached["jobId"]
            ws_watch_id = ws_detached["watchId"]

            # Verify DB sub_goals[0]["toolCallCount"] == 0 pre-wake
            with psycopg.connect(dsn) as conn:
                sg_json = fetch_scalar(
                    conn,
                    "SELECT sub_goals FROM autonomous_jobs WHERE job_id = %s",
                    (uuid.UUID(ws_job_id),),
                )
            assert isinstance(sg_json, list) and sg_json[0]["toolCallCount"] == 0

            # Immediately verify chat lane is unblocked on the SAME Voice WS
            await voice_ws.send_turn_commit("What is 2+2?")
            quick_events = await voice_ws.receive_until_turn_done(wait_s=5.0)
            quick_text = "".join(
                ev.get("text", "") for ev in quick_events if ev.get("type") == "reply.delta"
            )
            assert "4" in quick_text

            # 3) Non-intrusive delivery: start a slow streaming turn, settle watch MID-turn,
            #    and verify `wake_inbox_item` arrives strictly AFTER `reply.finished`!
            await voice_ws.send_turn_commit("Tell me a story")
            first_delta = await voice_ws.recv_event(wait_s=5.0)
            assert first_delta["type"] == "reply.delta"
            assert "Part 1" in first_delta["text"]

            # Trigger watchEvent while active turn is mid-stream
            await host.send_watch_event(
                watch_id=ws_watch_id,
                job_id=ws_job_id,
                kind="settled",
                observed_signals=["ui_quiescence", "process_idle"],
                elapsed_seconds=6,
            )

            collected: list[dict[str, Any]] = [first_delta]
            tail = await voice_ws.recv_until_wake_inbox(wait_s=6.0)
            collected.extend(tail[1])

            types = [ev.get("type") for ev in collected]
            assert "reply.finished" in types
            assert "wake_inbox_item" in types
            finished_idx = types.index("reply.finished")
            wake_idx = types.index("wake_inbox_item")
            assert wake_idx > finished_idx, f"Wake Inbox interrupted mid-turn: {types}"

            wake_item = collected[wake_idx]
            assert wake_item["jobId"] == ws_job_id
            assert wake_item["kind"] == "job_report"
            _assert_wake_inbox_item_mobile_schema(wake_item)

            # Verify completed job + REST wake-inbox Mobile schema parity
            completed_job = (await owner.get_job(ws_job_id)).json()
            assert completed_job["state"] == "completed"
            assert completed_job["subGoals"][0]["toolCallCount"] >= 1

            inbox_list = (await owner.list_wake_inbox()).json()["items"]
            matching = [i for i in inbox_list if i["jobId"] == ws_job_id]
            assert len(matching) == 1
            _assert_wake_inbox_item_mobile_schema(matching[0])
        finally:
            if voice_ws is not None:
                await voice_ws.close()
            await host.disconnect()
            await owner.aclose()


async def test_t3_02_rule_h1_post_wake_hallucination_rejected_on_detached_job(
    watch_server_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule H1: claiming done with `toolCallCount == 0` post-wake fails (`no_tool_executed`)."""
    monkeypatch.setattr(conv_routes, "DEFAULT_YIELD_MS", 80)
    monkeypatch.setattr(conv_ws, "DEFAULT_YIELD_MS", 80)

    async def _slow_tool_handler(
        tool_name: str, _args: dict[str, Any], _meta: dict[str, Any]
    ) -> dict[str, Any]:
        if tool_name == "media.play_pause":
            await asyncio.sleep(0.35)
            return {"status": "ok", "structuredContent": {"state": "paused"}}
        return {"status": "ok", "structuredContent": {"ok": True}}

    llm = _StreamingAwareGenerator(hallucinate_post_wake=True)
    async with watch_server_factory(generator=llm) as srv:
        owner = SimulatedWatchOwnerClient(
            base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer
        )
        host = SimulatedWatchXtpHost(tool_handler=_slow_tool_handler)
        try:
            pet_id = await owner.bind_pet()
            await owner.enable_consents()
            host_id = await host.connect_enroll(srv.ws_url, owner, "H1Mac")
            await owner.grant_all_standard(str(host_id), max_risk="R3")
            await host.send_presence(foreground_app_id="editor")

            sse = await owner.stream_turn(
                pet_id, "Run slow task on @H1Mac", target_host_id=str(host_id)
            )
            detached = sse.require("autonomous_job")
            job_id = detached["jobId"]
            watch_id = detached["watchId"]

            await host.send_watch_event(
                watch_id=watch_id,
                job_id=job_id,
                kind="settled",
                observed_signals=["ui_quiescence"],
                elapsed_seconds=5,
            )

            async def _is_failed() -> bool:
                return (await owner.get_job(job_id)).json()["state"] == "failed"

            await wait_until(_is_failed, description="Rule H1 post-wake hallucination FAILED")
            job_data = (await owner.get_job(job_id)).json()
            assert job_data["state"] == "failed"
            assert job_data["stopReason"] == "no_tool_executed"
            assert job_data["subGoals"][0]["toolCallCount"] == 0

            inbox = (await owner.list_wake_inbox()).json()["items"]
            failed_item = next(i for i in inbox if i["jobId"] == job_id)
            assert failed_item["kind"] == "job_failed"
            _assert_wake_inbox_item_mobile_schema(failed_item)
        finally:
            await host.disconnect()
            await owner.aclose()
