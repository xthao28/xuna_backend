"""WebSocket voice session route (VOICE-WS, 2026-10-23; KX-1 2026-10-29).

Protocol:
- Path: `GET /v1/voice` (WebSocket)
- Handshake & auth: Client sends hello with Firebase ID token or device proof within 5s.
- Real-time full-duplex audio & control stream.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.status import (
    WS_1008_POLICY_VIOLATION,
)

from xuna_backend.modules.accounts.application.authenticate import authenticate
from xuna_backend.modules.accounts.application.ports import (
    AuthUnavailableError,
    InvalidTokenError,
    TokenVerifier,
)
from xuna_backend.modules.actions.api.routes import build_stream_orchestrator
from xuna_backend.modules.actions.application.catalog import load_tool_catalog_for_owner
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    EventSink,
    Orchestrator,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionResultEvent,
    ApprovalDecision,
    BoardHidRequestEvent,
    LocalActionRequestEvent,
)
from xuna_backend.modules.cognition.application.ports import TextGenerator
from xuna_backend.modules.cognition.domain.types import (
    ChatFinished,
    Locale,
    Message,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from xuna_backend.modules.consent.application.ports import ConsentRepository
from xuna_backend.modules.consent.domain.consent import ConsentClass
from xuna_backend.modules.conversation.application.chat_turn import (
    MAX_HISTORY_LINE_CHARS,
    MAX_HISTORY_LINES,
    DetailDelta,
    GestureChosen,
    HistoryLine,
    PetAction,
    PetMood,
    PetSelf,
    PetTemperament,
    ReplyBudget,
    ToolCallRunner,
    TurnContext,
)
from xuna_backend.modules.conversation.application.spoken_turn import (
    SpokenAudio,
    SpokenFinished,
)
from xuna_backend.modules.conversation.application.turn_preparation import (
    PreparedTurn,
    prepare_turn,
)
from xuna_backend.modules.conversation.application.voice_session import (
    VoiceSession,
    VoiceSessionError,
)
from xuna_backend.modules.conversation.domain.signals import conversation_signals
from xuna_backend.modules.conversation.domain.turn_mode import (
    Channel,
    RequestedMode,
    TurnMode,
    decide_mode,
)
from xuna_backend.modules.identity.application.ports import IdentityRepository
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.owner_context import BuildOwnerContextUseCase
from xuna_backend.modules.knowledge.domain.owner_context import DeviceSlotView

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])


@dataclass(frozen=True, slots=True)
class _ParsedCommitContext:
    base: TurnContext
    device_slots: tuple[DeviceSlotView, ...] | None
    domain_id: uuid.UUID | None


def _parse_client_time(raw_val: object) -> datetime | None:
    if not isinstance(raw_val, str) or not raw_val.strip():
        return None
    try:
        dt = datetime.fromisoformat(raw_val.strip())
    except ValueError:
        return None
    if dt.utcoffset() is None:
        return None
    return dt


def _parse_device_slots(raw_val: object) -> tuple[DeviceSlotView, ...] | None:
    if raw_val is None or not isinstance(raw_val, list):
        return None
    slots: list[DeviceSlotView] = []
    for item in raw_val:
        if len(slots) >= 5:
            break
        if not isinstance(item, dict):
            continue
        slot_val = item.get("slot")
        level_val = item.get("level")
        state_val = item.get("state")
        if (
            isinstance(slot_val, int)
            and not isinstance(slot_val, bool)
            and isinstance(level_val, int)
            and not isinstance(level_val, bool)
            and isinstance(state_val, str)
        ):
            try:
                slots.append(DeviceSlotView(slot=slot_val, level=level_val, state=state_val))
            except ValueError:
                continue
    return tuple(slots)


def _parse_domain_id(raw_val: object) -> uuid.UUID | None:
    if not isinstance(raw_val, str) or not raw_val.strip():
        return None
    try:
        return uuid.UUID(raw_val.strip())
    except ValueError:
        return None


def _parse_requested_mode(raw_val: object) -> RequestedMode | None:
    if not isinstance(raw_val, str):
        return None
    try:
        return RequestedMode(raw_val)
    except ValueError:
        return None


def _parse_client_caps(raw_val: object) -> tuple[str, ...]:
    if not isinstance(raw_val, list):
        return ()
    return tuple(str(item) for item in raw_val[:16] if isinstance(item, str) and item)


def _parse_board_available(raw_val: object) -> bool | None:
    if isinstance(raw_val, dict) and isinstance(raw_val.get("available"), bool):
        return bool(raw_val["available"])
    return None


def _parse_commit_context(raw: dict[str, Any]) -> _ParsedCommitContext:
    history_items: list[HistoryLine] = []
    raw_history = raw.get("history", [])
    if isinstance(raw_history, list):
        for item in raw_history[:MAX_HISTORY_LINES]:
            if isinstance(item, dict) and "text" in item:
                txt = str(item["text"]).strip()
                if not txt:
                    continue
                if len(txt) > MAX_HISTORY_LINE_CHARS:
                    txt = txt[:MAX_HISTORY_LINE_CHARS]
                from_owner = item.get("role") == "owner" or item.get("from_owner", False)
                history_items.append(HistoryLine(from_owner=from_owner, text=txt))

    pet_self: PetSelf | None = None
    pet_data = raw.get("pet")
    if isinstance(pet_data, dict):

        def _safe_action(key: str) -> PetAction | None:
            if key in pet_data:
                try:
                    return PetAction(pet_data[key])
                except (ValueError, KeyError):
                    return None
            return None

        def _safe_temperament() -> PetTemperament | None:
            if "personality" in pet_data:
                try:
                    return PetTemperament(pet_data["personality"])
                except (ValueError, KeyError):
                    return None
            return None

        def _safe_mood() -> PetMood | None:
            if "mood" in pet_data:
                try:
                    return PetMood(pet_data["mood"])
                except (ValueError, KeyError):
                    return None
            return None

        pet_self = PetSelf(
            active_seconds=int(pet_data.get("activeSeconds", 0)),
            learned_events=int(pet_data.get("learnedEvents", 0)),
            current=_safe_action("current"),
            quiet_choice=_safe_action("quietChoice"),
            play_choice=_safe_action("playChoice"),
            temperament=_safe_temperament(),
            mood=_safe_mood(),
        )

    client_time = _parse_client_time(raw.get("clientTime"))
    raw_facts = raw.get("confirmedFacts", ())
    facts_tuple = (
        tuple(str(f) for f in raw_facts if isinstance(f, str))
        if isinstance(raw_facts, list | tuple)
        else ()
    )

    base = TurnContext(
        local_time=client_time,
        history=tuple(history_items),
        pet=pet_self,
        confirmed_facts=facts_tuple,
        gestures=raw.get("gestures") is True,
    )
    return _ParsedCommitContext(
        base=base,
        device_slots=_parse_device_slots(raw.get("knowledge")),
        domain_id=_parse_domain_id(raw.get("domainId")),
    )


def _parse_turn_context(raw: dict[str, Any]) -> TurnContext:
    return _parse_commit_context(raw).base


@router.websocket("/voice")
async def voice_websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    app = websocket.app

    # 1. Handshake & Auth timeout (5s)
    try:
        first_text = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        hello = json.loads(first_text)
    except TimeoutError:
        await websocket.send_json({"type": "error", "code": "auth_timeout"})
        await websocket.close(code=WS_1008_POLICY_VIOLATION)
        return
    except Exception:
        await websocket.send_json({"type": "error", "code": "invalid_hello"})
        await websocket.close(code=WS_1008_POLICY_VIOLATION)
        return

    if hello.get("type") != "hello" or hello.get("schemaVersion") != 1:
        await websocket.send_json({"type": "error", "code": "unsupported_protocol"})
        await websocket.close(code=WS_1008_POLICY_VIOLATION)
        return

    auth_payload = hello.get("auth", {})
    token = auth_payload.get("token")
    if not token:
        await websocket.send_json({"type": "error", "code": "missing_token"})
        await websocket.close(code=WS_1008_POLICY_VIOLATION)
        return

    verifier: TokenVerifier = app.state.token_verifier
    owner_repo_factory = app.state.owner_repository

    async with app.state.database.sessions() as db_session:
        try:
            owner = await authenticate(token, verifier, owner_repo_factory(db_session))
            await db_session.commit()
        except (InvalidTokenError, AuthUnavailableError):
            await websocket.send_json({"type": "error", "code": "unauthorized"})
            await websocket.close(code=WS_1008_POLICY_VIOLATION)
            return

        pet_id_raw = hello.get("petId")
        try:
            pet_id = PetId(pet_id_raw)
        except Exception:
            await websocket.send_json({"type": "error", "code": "invalid_pet_id"})
            await websocket.close(code=WS_1008_POLICY_VIOLATION)
            return

        id_repo: IdentityRepository = app.state.identity_repository(db_session)
        pet = await id_repo.get_pet(pet_id)
        if pet is None or pet.owner_id != owner.owner_id:
            await websocket.send_json({"type": "error", "code": "pet_not_found"})
            await websocket.close(code=WS_1008_POLICY_VIOLATION)
            return

        consent_repo: ConsentRepository = app.state.consent_repository(db_session)
        consent = await consent_repo.current(owner.owner_id)
        if not consent.allows(ConsentClass.CONVERSATION_VOICE):
            await websocket.send_json({"type": "error", "code": "consent_required"})
            await websocket.close(code=WS_1008_POLICY_VIOLATION)
            return

    session_id = str(uuid.uuid4())
    await websocket.send_json(
        {
            "type": "hello.ok",
            "sessionId": session_id,
            "downlink": {"format": "mp3", "sampleRate": 24000},
            "limits": {"maxUtteranceSeconds": 30},
        }
    )

    audio_params = hello.get("audio", {})
    sample_rate = int(audio_params.get("sampleRate", 16000))

    asr_auto_detect = getattr(getattr(app.state, "settings", None), "asr_auto_detect", True)
    locale_raw = hello.get("locale")
    target_locale: Locale | None = None
    if locale_raw:
        try:
            target_locale = Locale(locale_raw)
        except ValueError:
            target_locale = None
    if target_locale is None and not asr_auto_detect:
        target_locale = Locale.VI

    chosen_voice = hello.get("voice")
    default_client_caps = _parse_client_caps(hello.get("clientCapabilities"))
    default_board_available = bool(_parse_board_available(hello.get("boardHid")))
    raw_hello_self = hello.get("selfHostId")
    default_self_host_id = (
        raw_hello_self.strip()[:64]
        if isinstance(raw_hello_self, str) and raw_hello_self.strip()
        else None
    )

    session = VoiceSession(
        session_id=session_id,
        pet_id=pet_id,
        owner_id=owner.owner_id,
        recognizer=getattr(app.state, "speech_recognizer", None),
        generator=app.state.text_generator,
        synthesizer=getattr(app.state, "speech_synthesizer", None),
        locale=target_locale,
        voice=chosen_voice,
        speak=bool(hello.get("speak", True)),
        sample_rate=sample_rate,
    )

    active_turn_task: asyncio.Task[None] | None = None

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes"):
                try:
                    session.append_audio(message["bytes"])
                except VoiceSessionError as err:
                    await websocket.send_json({"type": "error", "code": err.code})
                continue

            if message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except Exception as err:
                    logger.debug(f"Malformed websocket json text: {err}")
                    continue

                msg_type = payload.get("type")
                if msg_type == "listen.start":
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()
                    session.start_listening()

                elif msg_type == "listen.stop":
                    try:
                        transcript, asr_ms = await session.stop_listening()
                        await websocket.send_json(
                            {"type": "stt.final", "text": transcript, "asrMs": asr_ms}
                        )
                    except VoiceSessionError as err:
                        await websocket.send_json({"type": "error", "code": err.code})

                elif msg_type == "turn.commit":
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()

                    turn_text = str(payload.get("text", "")).strip()
                    raw_ctx = payload.get("context", {})
                    ctx_dict = raw_ctx if isinstance(raw_ctx, dict) else {}
                    parsed_ctx = _parse_commit_context(ctx_dict)
                    requested_mode = _parse_requested_mode(payload.get("mode"))
                    turn_voice = payload.get("voice")

                    turn_caps = (
                        _parse_client_caps(payload.get("clientCapabilities"))
                        or _parse_client_caps(ctx_dict.get("clientCapabilities"))
                        or default_client_caps
                    )
                    override_board = _parse_board_available(payload.get("boardHid"))
                    if override_board is None:
                        override_board = _parse_board_available(ctx_dict.get("boardHid"))
                    turn_board_available = (
                        override_board if override_board is not None else default_board_available
                    )
                    raw_self_host = payload.get("selfHostId") or ctx_dict.get("selfHostId")
                    turn_self_host_id = (
                        raw_self_host.strip()[:64]
                        if isinstance(raw_self_host, str) and raw_self_host.strip()
                        else default_self_host_id
                    )

                    now = datetime.now(UTC)
                    turn_tools: tuple[ToolSpec, ...] = ()
                    try:
                        async with app.state.database.sessions() as db:
                            if turn_caps:
                                try:
                                    turn_tools = await load_tool_catalog_for_owner(
                                        owner_id=owner.owner_id,
                                        client_caps=turn_caps,
                                        board_available=turn_board_available,
                                        host_repo=app.state.host_repository(db),
                                        hub=app.state.host_session_hub,
                                        self_host_id=turn_self_host_id,
                                    )
                                except Exception:
                                    logger.debug(
                                        "Failed to load tool catalog for voice turn",
                                        exc_info=True,
                                    )
                                    turn_tools = ()

                            mem_repo = app.state.memory_repository(db)
                            k_repo = app.state.knowledge_repository(db)
                            c_port = app.state.knowledge_candidate_adapter(db)
                            prepared = await prepare_turn(
                                pet_id=pet_id,
                                text=turn_text,
                                requested=requested_mode,
                                channel=Channel.VOICE,
                                base=parsed_ctx.base,
                                device_slots=parsed_ctx.device_slots,
                                explicit_domain_id=parsed_ctx.domain_id,
                                memory_repo=mem_repo,
                                owner_context=BuildOwnerContextUseCase(
                                    k_repo, c_port, getattr(app.state, "embedder", None)
                                ),
                                now=now,
                            )
                            if (
                                not prepared.context.confirmed_facts
                                and parsed_ctx.base.confirmed_facts
                            ):
                                prepared = replace(
                                    prepared,
                                    context=replace(
                                        prepared.context,
                                        confirmed_facts=parsed_ctx.base.confirmed_facts,
                                    ),
                                )
                    except Exception:
                        fallback_dec = decide_mode(
                            requested_mode, Channel.VOICE, turn_text, domain_matched=False
                        )
                        prepared = PreparedTurn(
                            context=replace(parsed_ctx.base, mode=fallback_dec.mode),
                            decision=fallback_dec,
                            domain_id=None,
                        )

                    settings = getattr(app.state, "settings", None)
                    chosen_mode = prepared.decision.mode
                    if chosen_mode == TurnMode.EXPERT:
                        max_tok = settings.text_expert_max_tokens if settings is not None else 3000
                        timeout_s = settings.text_expert_timeout_s if settings is not None else 60.0
                        budget = ReplyBudget(max_output_tokens=max_tok, timeout_s=timeout_s)
                        expert_gen: TextGenerator | None = getattr(
                            app.state, "expert_text_generator", None
                        )
                        turn_generator: TextGenerator = expert_gen or app.state.text_generator
                    else:
                        budget = ReplyBudget(max_output_tokens=400, timeout_s=30.0)
                        turn_generator = app.state.text_generator

                    turn_tool_runner: ToolCallRunner | None = None
                    if turn_caps:
                        stream_orch = build_stream_orchestrator(app.state)
                        eff_locale = target_locale or Locale.VI

                        async def _run_voice_tool(
                            call: ToolCall,
                            emit: EventSink,
                            *,
                            _orch: Orchestrator = stream_orch,
                            _locale: Locale = eff_locale,
                            _caps: frozenset[str] = frozenset(turn_caps),
                            _board: bool = turn_board_available,
                            _self_host: str | None = turn_self_host_id,
                        ) -> Message:
                            return await _orch.handle_call(
                                call,
                                ActionContext(
                                    owner_id=owner.owner_id,
                                    pet_id=str(pet_id),
                                    locale=_locale,
                                    client_caps=_caps,
                                    board_available=_board,
                                    emit_event=emit,
                                    request_id=session_id,
                                    self_host_id=_self_host,
                                ),
                            )

                        turn_tool_runner = _run_voice_tool

                    async def run_reply(
                        target_text: str,
                        prep: PreparedTurn,
                        req_mode: RequestedMode | None,
                        bud: ReplyBudget,
                        gen: TextGenerator,
                        tools_arg: tuple[ToolSpec, ...],
                        runner_arg: ToolCallRunner | None,
                        v: str | None = None,
                    ) -> None:
                        signals = {
                            "signals": list(
                                conversation_signals(target_text, len(prep.context.history))
                            )
                        }
                        domain_id_str = str(prep.domain_id) if prep.domain_id is not None else None
                        mode_val = prep.decision.mode.value
                        has_detail = False
                        try:
                            if req_mode is not None:
                                await websocket.send_json(
                                    {
                                        "type": "turn.mode",
                                        "mode": mode_val,
                                        "domainId": domain_id_str,
                                        "reason": prep.decision.reason.value,
                                    }
                                )
                            async for event in session.stream_turn_reply(
                                target_text,
                                prep.context,
                                voice=v,
                                mode=prep.decision.mode,
                                budget=bud,
                                generator=gen,
                                tools=tools_arg,
                                tool_runner=runner_arg,
                            ):
                                if isinstance(event, GestureChosen):
                                    await websocket.send_json(
                                        {"type": "gesture", "name": event.gesture.value}
                                    )
                                elif isinstance(event, ActionPendingEvent):
                                    await websocket.send_json(
                                        {
                                            "type": "action_pending",
                                            "approvalId": event.approval_id,
                                            "target": event.target,
                                            "summary": event.summary,
                                            "risk": event.risk,
                                            "expiresIn": event.expires_in,
                                        }
                                    )
                                elif isinstance(event, BoardHidRequestEvent):
                                    await websocket.send_json(
                                        {
                                            "type": "board_hid_request",
                                            "invocationId": event.invocation_id,
                                            "approvalId": event.approval_id,
                                            "key": event.key,
                                            "usageIndex": event.usage_index,
                                        }
                                    )
                                elif isinstance(event, LocalActionRequestEvent):
                                    local_msg: dict[str, Any] = {
                                        "type": "local_action_request",
                                        "invocationId": event.invocation_id,
                                        "target": event.target,
                                        "tool": event.tool,
                                        "arguments": event.arguments,
                                    }
                                    if event.approval_id is not None:
                                        local_msg["approvalId"] = event.approval_id
                                    await websocket.send_json(local_msg)
                                elif isinstance(event, ActionResultEvent):
                                    act_msg: dict[str, Any] = {
                                        "type": "action_result",
                                        "invocationId": event.invocation_id,
                                        "target": event.target,
                                        "summary": event.summary,
                                        "status": event.status,
                                    }
                                    if event.gesture is not None:
                                        act_msg["gesture"] = event.gesture
                                    await websocket.send_json(act_msg)
                                elif isinstance(event, TextDelta):
                                    await websocket.send_json(
                                        {"type": "reply.delta", "text": event.text}
                                    )
                                elif isinstance(event, DetailDelta):
                                    has_detail = True
                                    await websocket.send_json(
                                        {"type": "detail.delta", "text": event.text}
                                    )
                                elif isinstance(event, SpokenAudio):
                                    await websocket.send_json(
                                        {"type": "tts.start", "seq": event.seq, "text": event.text}
                                    )
                                    await websocket.send_bytes(event.data)
                                    await websocket.send_json({"type": "tts.end", "seq": event.seq})
                                elif isinstance(event, SpokenFinished):
                                    await websocket.send_json(
                                        {
                                            "type": "reply.finished",
                                            "latencyMs": event.finished.latency_ms,
                                            "firstAudioMs": event.first_audio_ms,
                                            "spoken": event.spoken,
                                            "mode": mode_val,
                                            "domainId": domain_id_str,
                                            "hasDetail": has_detail,
                                        }
                                        | signals
                                    )
                                elif isinstance(event, ChatFinished):
                                    await websocket.send_json(
                                        {
                                            "type": "reply.finished",
                                            "latencyMs": event.latency_ms,
                                            "firstAudioMs": None,
                                            "spoken": False,
                                            "mode": mode_val,
                                            "domainId": domain_id_str,
                                            "hasDetail": has_detail,
                                        }
                                        | signals
                                    )
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            logger.warning("voice_ws_reply_error")
                            await websocket.send_json({"type": "error", "code": "reply_failed"})

                    active_turn_task = asyncio.create_task(
                        run_reply(
                            turn_text,
                            prepared,
                            requested_mode,
                            budget,
                            turn_generator,
                            turn_tools,
                            turn_tool_runner,
                            v=turn_voice,
                        )
                    )

                elif msg_type == "action.decide":
                    raw_aid = payload.get("approvalId")
                    raw_dec = payload.get("decision")
                    if isinstance(raw_aid, str) and isinstance(raw_dec, str):
                        try:
                            aid = uuid.UUID(raw_aid)
                            dec = ApprovalDecision(raw_dec)
                            ws_orch = build_stream_orchestrator(app.state)
                            await ws_orch.decide_approval(
                                approval_id=aid,
                                owner_id=owner.owner_id,
                                decision=dec,
                                request_id=session_id,
                            )
                        except ValueError:
                            pass

                elif msg_type == "action.result":
                    raw_inv = payload.get("invocationId")
                    raw_st = payload.get("status")
                    raw_out = payload.get("output")
                    raw_err = payload.get("errorCode")
                    if isinstance(raw_inv, str) and isinstance(raw_st, str):
                        app.state.local_result_broker.report(
                            invocation_id=raw_inv,
                            owner_id=owner.owner_id,
                            status=raw_st,
                            output=dict(raw_out) if isinstance(raw_out, dict) else None,
                            error_code=str(raw_err)[:64] if isinstance(raw_err, str) else None,
                        )

                elif msg_type == "abort":
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()
                    session.abort(str(payload.get("reason", "cancel")))

                elif msg_type == "bye":
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        pass
    finally:
        if active_turn_task and not active_turn_task.done():
            active_turn_task.cancel()
