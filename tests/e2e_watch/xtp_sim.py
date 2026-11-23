"""Watch-capable XTP v1 Host Agent simulator & Owner client for E2E Tiers 1-5."""

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import websockets

from tests.e2e_mesh.xtp_sim import (
    ScriptedTextGenerator,
    SimulatedOwnerClient,
    SimulatedXtpHost,
    VoiceWsSession,
    args_sha256,
    b64,
    done_event,
    host_proof_message,
    make_standard_mesh_tools,
    make_tool_descriptor,
)
from tests.helpers import idem
from xuna_backend.modules.hosts.application.xtp_frames import (
    WatchCancelParamsModel,
    WatchEventParamsModel,
    WatchRegisterParamsModel,
    WatchRegisterResultModel,
)


async def wait_until(
    predicate: Callable[[], Awaitable[Any] | Any],
    *,
    timeout_s: float = 5.0,
    interval_s: float = 0.02,
    description: str = "condition",
) -> Any:
    """Polls `predicate` until truthy or raises AssertionError on timeout."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_val: Any = None
    while True:
        res = predicate()
        last_val = await res if asyncio.iscoroutine(res) else res
        if last_val:
            return last_val
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(
                f"Timed out after {timeout_s:.2f}s waiting for {description} (last={last_val!r})"
            )
        await asyncio.sleep(interval_s)


def make_watch_tools(
    *,
    platform: str | None = None,
    include_cu: bool = True,
    include_mail: bool = True,
    include_web: bool = True,
    extra_tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Builds a standard neutral tool manifest covering UI, App, CU, Web, and Mail."""
    tools = make_standard_mesh_tools(platform=platform, include_cu=include_cu)
    tools.extend(
        [
            make_tool_descriptor(
                "ui.inspect",
                family="ui",
                risk="R0",
                observes=True,
                reversible=True,
                platform=platform,
                output_schema={
                    "type": "object",
                    "properties": {
                        "appId": {"type": "string"},
                        "pid": {"type": "integer"},
                        "windowId": {"type": "integer"},
                        "snapshotId": {"type": "string"},
                        "nodes": {"type": "array"},
                        "status": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
            make_tool_descriptor(
                "ui.observe_hybrid",
                family="ui",
                risk="R1",
                observes=True,
                reversible=True,
                platform=platform,
                output_schema={
                    "type": "object",
                    "properties": {
                        "appId": {"type": "string"},
                        "pid": {"type": "integer"},
                        "windowId": {"type": "integer"},
                        "snapshotId": {"type": "string"},
                        "nodes": {"type": "array"},
                        "screenshot": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
            ),
            make_tool_descriptor(
                "ui.click",
                family="ui",
                risk="R2",
                observes=False,
                reversible=False,
                platform=platform,
                input_schema={
                    "type": "object",
                    "properties": {"refId": {"type": "string"}},
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"clicked": {"type": "boolean"}},
                    "additionalProperties": False,
                },
            ),
        ]
    )
    if include_cu:
        # Give `input.click` base risk="R2" (matching ADR-0015 / test_autopilot_permissions.py)
        # and support `menuPath` for high-impact ADR-0015 tests.
        for td in tools:
            if td["name"] == "input.click":
                td["_meta"]["xuna/policy"]["risk"] = "R2"
                td["inputSchema"]["properties"]["menuPath"] = {"type": "string"}
    if include_mail:
        tools.append(
            make_tool_descriptor(
                "mail.send",
                family="app",
                risk="R3",
                observes=False,
                reversible=False,
                platform=platform,
                input_schema={
                    "type": "object",
                    "properties": {
                        "to": {"type": "array", "items": {"type": "string"}},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"sent": {"type": "boolean"}},
                    "additionalProperties": False,
                },
            )
        )
    if include_web:
        tools.append(
            make_tool_descriptor(
                "web.click",
                family="url",
                risk="R2",
                observes=False,
                reversible=False,
                platform=platform,
                input_schema={
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string"},
                        "selector": {"type": "string"},
                        "targetText": {"type": "string"},
                        "elementName": {"type": "string"},
                    },
                    "required": ["ref"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"clicked": {"type": "boolean"}},
                    "additionalProperties": False,
                },
            )
        )
    if extra_tools:
        tools.extend(extra_tools)
    return tools


class SimulatedWatchXtpHost(SimulatedXtpHost):
    """XTP v1 host simulator supporting `xuna/watchRegister`, `xuna/watchCancel`, `watchEvent`."""

    def __init__(
        self,
        *,
        platform: str = "macos",
        os_version: str = "15.0",
        agent_version: str = "0.2.0",
        hostname: str = "watch-host",
        tools: list[dict[str, Any]] | None = None,
        tool_handler: Callable[[str, dict[str, Any], dict[str, Any]], Any] | None = None,
        preflight_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        watch_register_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        max_concurrent_watches: int = 4,
        reject_modes: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            platform=platform,
            os_version=os_version,
            agent_version=agent_version,
            hostname=hostname,
            tools=(tools if tools is not None else make_watch_tools(platform=platform)),
            tool_handler=tool_handler,
            preflight_handler=preflight_handler,
        )
        self.watch_register_handler = watch_register_handler
        self.max_concurrent_watches = max_concurrent_watches
        self.reject_modes: dict[str, str] = (
            dict(reject_modes)
            if reject_modes is not None
            else {"protocol_async": "watch_unsupported"}
        )
        self.received_watch_registers: list[dict[str, Any]] = []
        self.received_watch_cancels: list[dict[str, Any]] = []
        self.active_watches: dict[str, dict[str, Any]] = {}

    async def _complete_session_handshake(self, ws: Any, host_id: uuid.UUID) -> None:
        import base64

        challenge_2 = json.loads(await ws.recv())
        assert challenge_2["method"] == "xuna/challenge"
        assert challenge_2["params"]["purpose"] == 2
        nonce_2 = base64.b64decode(challenge_2["params"]["nonce"])
        sig_2 = self.device.sign(host_proof_message(nonce_2, 2, self.device.public_key, host_id))
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "xuna/proof",
                    "params": {"purpose": 2, "signature": b64(sig_2)},
                }
            )
        )
        proof_resp_2 = json.loads(await ws.recv())
        assert proof_resp_2 == {"jsonrpc": "2.0", "id": 3, "result": {"verified": True}}

        ready_msg = json.loads(await ws.recv())
        assert ready_msg["method"] == "xuna/ready"
        self.received_frames.append(ready_msg)

        # Bounded handshake loop (up to 10 frames) handling interleaved `xuna/lanPeer` AND
        # rehydrated `xuna/watchRegister` / `xuna/watchCancel` before `tools/list` arrives.
        tools_answered = False
        for _ in range(10):
            frame = json.loads(await ws.recv())
            self.received_frames.append(frame)
            if frame.get("method") == "tools/list":
                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": frame["id"],
                            "result": {"tools": self.tools},
                        }
                    )
                )
                tools_answered = True
                break
            await self._dispatch_server_frame(frame)
        assert tools_answered, "Timed out waiting for tools/list during XTP handshake"

        self._reader_task = asyncio.create_task(self._reader_loop())
        await self.ping()

    async def _dispatch_server_frame(self, frame: dict[str, Any]) -> None:
        method = frame.get("method")
        req_id = frame.get("id")
        params = frame.get("params") or {}

        if method == "xuna/watchRegister":
            assert self._ws is not None
            WatchRegisterParamsModel.model_validate(params)
            self.received_watch_registers.append(dict(params))
            result = self._handle_watch_register(params)
            WatchRegisterResultModel.model_validate(result)
            await self._ws.send(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}))
            return

        if method == "xuna/watchCancel":
            assert self._ws is not None
            WatchCancelParamsModel.model_validate(params)
            self.received_watch_cancels.append(dict(params))
            cancelled = self._handle_watch_cancel(params)
            await self._ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"schemaVersion": 1, "cancelledCount": cancelled},
                    }
                )
            )
            return

        if method == "xuna/preflight":
            assert self._ws is not None
            tool = str(params.get("name") or params.get("tool") or "")
            args = dict(params.get("arguments") or {})
            self.received_preflights.append({"tool": tool, "arguments": args})
            if self._preflight_handler is not None:
                preflight = self._preflight_handler(tool, args)
            else:
                risk = "R3"
                for td in self.tools:
                    if td["name"] == tool:
                        risk = td["_meta"]["xuna/policy"]["risk"]
                        break
                preflight = {
                    "status": "ok",
                    "effectiveRisk": risk,
                    "reasonCode": "ok",
                }
            await self._ws.send(
                json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"preflight": preflight}})
            )
            return

        await super()._dispatch_server_frame(frame)

    def _handle_watch_register(self, params: dict[str, Any]) -> dict[str, Any]:
        watch_id = str(params["watchId"])
        mode = str(params["mode"])
        if self.watch_register_handler is not None:
            res = dict(self.watch_register_handler(params))
            if res.get("accepted"):
                self.active_watches[watch_id] = dict(params)
            return res

        if mode in self.reject_modes:
            return {
                "schemaVersion": 1,
                "accepted": False,
                "watchId": watch_id,
                "errorCode": self.reject_modes[mode],
            }

        if len(self.active_watches) >= self.max_concurrent_watches:
            return {
                "schemaVersion": 1,
                "accepted": False,
                "watchId": watch_id,
                "errorCode": "watch_limit_reached",
            }

        signals = params.get("signals")
        active_signals = list(signals) if isinstance(signals, list) and signals else [mode]
        self.active_watches[watch_id] = dict(params)
        return {
            "schemaVersion": 1,
            "accepted": True,
            "watchId": watch_id,
            "activeSignals": active_signals,
            "eventDriven": mode != "timer",
            "fallbackUsed": False,
        }

    def _handle_watch_cancel(self, params: dict[str, Any]) -> int:
        reason = params.get("reason")
        watch_id = params.get("watchId")
        job_id = params.get("jobId")
        if reason == "estop" and watch_id is None and job_id is None:
            count = len(self.active_watches)
            self.active_watches.clear()
            return count
        to_remove = [
            wid
            for wid, w in self.active_watches.items()
            if (watch_id is not None and wid == watch_id)
            or (job_id is not None and w.get("jobId") == job_id)
        ]
        for wid in to_remove:
            self.active_watches.pop(wid, None)
        return len(to_remove)

    @staticmethod
    def _default_tool_result(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "ui.inspect":
            structured: dict[str, Any] = {
                "appId": "editor",
                "pid": 48210,
                "windowId": 1042,
                "snapshotId": "snap_1",
                "nodes": [],
                "status": "settled",
            }
            return {
                "content": [{"type": "text", "text": json.dumps(structured)}],
                "structuredContent": structured,
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 8},
            }
        if tool_name == "cu.session_start":
            structured = {
                "sessionId": f"cu-{uuid.uuid4().hex[:8]}",
                "foregroundApp": "editor",
                "pid": 48210,
                "windowId": 1042,
            }
            return {
                "content": [{"type": "text", "text": json.dumps(structured)}],
                "structuredContent": structured,
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 10},
            }
        if tool_name == "mail.send":
            structured = {"sent": True}
            return {
                "content": [{"type": "text", "text": json.dumps(structured)}],
                "structuredContent": structured,
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 12},
            }
        if tool_name in ("web.click", "input.click", "ui.click"):
            structured = {"clicked": True}
            return {
                "content": [{"type": "text", "text": json.dumps(structured)}],
                "structuredContent": structured,
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 10},
            }
        return SimulatedXtpHost._default_tool_result(tool_name, args)

    async def send_watch_event(
        self,
        *,
        watch_id: str,
        job_id: str,
        kind: str = "settled",
        observed_signals: list[str] | None = None,
        elapsed_seconds: int = 10,
        change_count: int = 1,
        digest_changed: bool = True,
        occurred_at: str | None = None,
        error_code: str | None = None,
        validate: bool = True,
        sync: bool = True,
    ) -> dict[str, Any]:
        assert self._ws is not None
        signals = (
            list(observed_signals)
            if observed_signals is not None
            else (["ui_quiescence"] if kind in ("settled", "changed") else [])
        )
        params: dict[str, Any] = {
            "schemaVersion": 1,
            "watchId": watch_id,
            "jobId": job_id,
            "kind": kind,
            "observedSignals": signals,
            "elapsedSeconds": elapsed_seconds,
            "changeCount": change_count,
            "digestChanged": digest_changed,
            "occurredAt": occurred_at or datetime.now(UTC).isoformat(),
        }
        if error_code is not None:
            params["errorCode"] = error_code
        if validate:
            WatchEventParamsModel.model_validate(params)
        if kind in ("settled", "timeout", "target_lost", "error"):
            self.active_watches.pop(watch_id, None)
        await self._ws.send(
            json.dumps({"jsonrpc": "2.0", "method": "xuna/watchEvent", "params": params})
        )
        if sync:
            await self.ping()
        return params

    async def disconnect(self) -> None:
        for task in list(self._bg_tasks):
            if not task.done():
                task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)
            self._bg_tasks.clear()
        await super().disconnect()


class WatchVoiceWsSession(VoiceWsSession):
    """Voice WS helper with non-intrusive Wake Inbox event collection."""

    async def recv_until_event(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        wait_s: float = 10.0,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        deadline = asyncio.get_running_loop().time() + wait_s
        while True:
            remaining = max(0.05, deadline - asyncio.get_running_loop().time())
            raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            if isinstance(raw, bytes):
                continue
            ev = json.loads(raw)
            events.append(ev)
            if predicate(ev):
                return events

    async def recv_until_wake_inbox(
        self, *, wait_s: float = 10.0
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        events = await self.recv_until_event(
            lambda ev: ev.get("type") == "wake_inbox_item",
            wait_s=wait_s,
        )
        return events[-1], events


class SimulatedWatchOwnerClient(SimulatedOwnerClient):
    """HTTP/SSE/Voice-WS owner client extended with Watch & Wake Inbox endpoints."""

    async def grant_all_standard(self, host_id: str, *, max_risk: str = "R3") -> dict[str, Any]:
        return await self.set_grants(
            host_id,
            [
                {"family": fam, "maxRisk": max_risk, "enabled": True}
                for fam in (
                    "host",
                    "media",
                    "app",
                    "url",
                    "workflow",
                    "cu",
                    "screen",
                    "ui",
                    "input",
                    "files",
                )
            ],
        )

    async def list_wake_inbox(
        self,
        *,
        unread_only: bool = False,
        limit: int = 50,
    ) -> httpx.Response:
        return await self.http.get(
            "/v1/actions/wake-inbox",
            headers=self.headers,
            params={"unreadOnly": str(unread_only).lower(), "limit": limit},
        )

    async def ack_wake_inbox(
        self,
        item_id: str,
        *,
        dismiss: bool = False,
        idempotency_key: str | None = None,
        include_idem: bool = True,
    ) -> httpx.Response:
        headers = dict(self.headers)
        if include_idem:
            headers["Idempotency-Key"] = idempotency_key or str(uuid.uuid4())
        return await self.http.post(
            f"/v1/actions/wake-inbox/{item_id}/ack",
            headers=headers,
            json={"schemaVersion": 1, "dismiss": dismiss},
        )

    async def get_job_watch(self, job_id: str) -> httpx.Response:
        return await self.http.get(
            f"/v1/actions/jobs/{job_id}/watch",
            headers=self.headers,
        )

    async def extend_watch(
        self,
        job_id: str,
        *,
        additional_seconds: int = 300,
        idempotency_key: str | None = None,
    ) -> httpx.Response:
        headers = {**self.headers, "Idempotency-Key": idempotency_key or str(uuid.uuid4())}
        return await self.http.post(
            f"/v1/actions/jobs/{job_id}/extend-watch",
            headers=headers,
            json={"schemaVersion": 1, "additionalSeconds": additional_seconds},
        )

    async def wake_now(
        self,
        job_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> httpx.Response:
        headers = {**self.headers, "Idempotency-Key": idempotency_key or str(uuid.uuid4())}
        return await self.http.post(
            f"/v1/actions/jobs/{job_id}/wake-now",
            headers=headers,
            json={"schemaVersion": 1},
        )

    async def pause_job(self, job_id: str) -> httpx.Response:
        return await self.http.post(
            f"/v1/actions/jobs/{job_id}/pause",
            headers=self.headers,
        )

    async def cancel_job(self, job_id: str, *, reason: str = "user_cancelled") -> httpx.Response:
        return await self.http.post(
            f"/v1/actions/jobs/{job_id}/cancel",
            headers=self.headers,
            json={"schemaVersion": 1, "reason": reason},
        )

    async def resume_job(self, job_id: str, *, approval_id: str | None = None) -> httpx.Response:
        body: dict[str, Any] = {"schemaVersion": 1}
        if approval_id is not None:
            body["approvalId"] = approval_id
        return await self.http.post(
            f"/v1/actions/jobs/{job_id}/resume",
            headers=self.headers,
            json=body,
        )

    async def connect_voice_ws(
        self,
        pet_id: str,
        *,
        target_host_id: str | None = None,
        follow_me: bool | None = None,
        client_subnet_hash: str | None = None,
        client_capabilities: tuple[str, ...] = ("host_actions_v1",),
        speak: bool = False,
    ) -> WatchVoiceWsSession:
        ws = await websockets.connect(f"{self.ws_url}/v1/voice")
        hello: dict[str, Any] = {
            "type": "hello",
            "schemaVersion": 1,
            "auth": {"token": self.jwt},
            "petId": pet_id,
            "speak": speak,
            "clientCapabilities": list(client_capabilities),
        }
        if target_host_id is not None:
            hello["targetHostId"] = target_host_id
        if follow_me is not None:
            hello["followMe"] = follow_me
        if client_subnet_hash is not None:
            hello["clientSubnetHash"] = client_subnet_hash
        await ws.send(json.dumps(hello))
        hello_ok = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
        assert hello_ok["type"] == "hello.ok", hello_ok
        return WatchVoiceWsSession(ws=ws)


def load_watch_frames_fixture() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "xtp" / "watch_frames.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "ScriptedTextGenerator",
    "SimulatedWatchOwnerClient",
    "SimulatedWatchXtpHost",
    "WatchVoiceWsSession",
    "args_sha256",
    "b64",
    "done_event",
    "host_proof_message",
    "idem",
    "load_watch_frames_fixture",
    "make_tool_descriptor",
    "make_watch_tools",
    "wait_until",
]
