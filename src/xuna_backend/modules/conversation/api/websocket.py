"""WebSocket voice session route (VOICE-WS, 2026-10-23; KX-1 2026-10-29).

Protocol:
- Path: `GET /v1/voice` (WebSocket)
- Handshake & auth: Client sends hello with Firebase ID token or device proof within 5s.
- Real-time full-duplex audio & control stream.
"""

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.status import (
    WS_1008_POLICY_VIOLATION,
)

from xuna_backend.modules.accounts.application.authenticate import authenticate_with_status
from xuna_backend.modules.accounts.application.ports import (
    AuthUnavailableError,
    InvalidTokenError,
    TokenVerifier,
)
from xuna_backend.modules.actions.api.routes import (
    _decode_b64_flex,
    build_mesh_pre_router,
    build_stream_orchestrator,
)
from xuna_backend.modules.actions.application.computer_use import session_event_payload
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    EventSink,
    Orchestrator,
)
from xuna_backend.modules.actions.application.yield_detach import (
    DEFAULT_YIELD_MS,
    YieldDetachRequest,
    YieldDetachResult,
    YieldDetachService,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionResultEvent,
    ActionTargetRequiredEvent,
    ApprovalDecision,
    ApprovalScope,
    BoardHidRequestEvent,
    ComputerUseSessionEvent,
    HostWakeRequestEvent,
    LocalActionRequestEvent,
)
from xuna_backend.modules.actions.domain.watch import TurnDetachedEvent
from xuna_backend.modules.cognition.application.ports import TextGenerator
from xuna_backend.modules.cognition.domain.language_detector import detect_text_language
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
    CuCleanupCallback,
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
    TurnEvent,
    build_request,
    chat_turn,
)
from xuna_backend.modules.conversation.application.mesh_turn_helper import (
    execute_repair_dispatch,
    parse_ws_hello,
    parse_ws_optional_host_id,
    parse_ws_turn_commit,
    prepare_mesh_turn,
    stream_short_circuit_turn,
    stream_with_target_fallback,
    tz_offset_minutes_from_dt,
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
    SessionState,
    VoiceSession,
    VoiceSessionError,
)
from xuna_backend.modules.conversation.domain.signals import conversation_signals
from xuna_backend.modules.conversation.domain.turn_mode import (
    Channel,
    RequestedMode,
    TurnMode,
    decide_mode,
    is_code_or_tool_synthesis_request,
)
from xuna_backend.modules.hosts.domain.entities import HostId
from xuna_backend.modules.hosts.domain.twin import HostFocusChangedNotice
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

    if (
        not isinstance(hello, dict)
        or hello.get("type") != "hello"
        or hello.get("schemaVersion") != 1
    ):
        await websocket.send_json({"type": "error", "code": "unsupported_protocol"})
        await websocket.close(code=WS_1008_POLICY_VIOLATION)
        return

    try:
        parsed_hello = parse_ws_hello(hello)
    except ValidationError:
        await websocket.send_json({"type": "error", "code": "validation_failed"})
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
            owner, created = await authenticate_with_status(
                token, verifier, owner_repo_factory(db_session)
            )
            if created:
                await db_session.commit()
        except (InvalidTokenError, AuthUnavailableError):
            await websocket.send_json({"type": "error", "code": "unauthorized"})
            await websocket.close(code=WS_1008_POLICY_VIOLATION)
            return

        pet_id_raw = hello.get("petId")
        try:
            if not isinstance(pet_id_raw, str):
                raise ValueError("petId must be a string")
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
    try:
        default_target_host_id = (
            HostId(parsed_hello.targetHostId) if parsed_hello.targetHostId is not None else None
        )
    except ValueError:
        await websocket.send_json({"type": "error", "code": "validation_failed"})
        await websocket.close(code=WS_1008_POLICY_VIOLATION)
        return
    default_follow_me = parsed_hello.followMe
    default_client_subnet_hash = parsed_hello.clientSubnetHash

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
        vad_enabled=(hello.get("mode") == "vad"),
    )

    active_turn_task: asyncio.Task[None] | None = None
    active_turn_orch: Orchestrator | None = None
    active_turn_pinned_host: HostId | None = None
    ws_closing = False

    ws_hub = getattr(app.state, "host_session_hub", None)
    ws_focus_unsub: Callable[[], None] | None = None
    ws_wake_unsub: Callable[[], None] | None = None
    pending_wake_items: list[dict[str, Any]] = []
    wake_flush_tasks: set[asyncio.Task[None]] = set()
    wake_send_lock = asyncio.Lock()

    def _turn_is_busy() -> bool:
        return session.state != SessionState.IDLE or (
            active_turn_task is not None and not active_turn_task.done()
        )

    async def _flush_pending_wake_items() -> None:
        async with wake_send_lock:
            while not ws_closing and not _turn_is_busy() and pending_wake_items:
                next_item = pending_wake_items.pop(0)
                with contextlib.suppress(Exception):
                    await websocket.send_json({"type": "wake_inbox_item", **next_item})

    async def _on_ws_wake_inbox(payload: dict[str, Any]) -> None:
        if ws_closing:
            return
        pending_wake_items.append(payload)
        if not _turn_is_busy():
            await _flush_pending_wake_items()

    wake_svc = getattr(app.state, "wake_inbox_service", None)
    wake_sub_fn = getattr(wake_svc, "subscribe", None) if wake_svc is not None else None
    if callable(wake_sub_fn):
        ws_wake_unsub = wake_sub_fn(owner.owner_id, _on_ws_wake_inbox)

    async def _on_ws_focus(notice: HostFocusChangedNotice) -> None:
        if not ws_closing:
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "host_focus_changed", **notice.wire_payload()})

    sub_fn = getattr(ws_hub, "subscribe_focus", None) if ws_hub is not None else None
    if callable(sub_fn):
        ws_focus_unsub = sub_fn(owner.owner_id, _on_ws_focus)

    def _is_closing() -> bool:
        return ws_closing

    async def _cancel_active_turn() -> None:
        nonlocal active_turn_task
        if active_turn_task is not None and not active_turn_task.done():
            active_turn_task.cancel()
            await asyncio.gather(active_turn_task, return_exceptions=True)
        active_turn_task = None

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes"):
                try:
                    vad_event = session.append_audio(message["bytes"])
                    if vad_event == "vad.start":
                        await websocket.send_json({"type": "vad.start"})
                    elif vad_event == "vad.end":
                        await websocket.send_json({"type": "vad.end"})
                        transcript, asr_ms = await session.stop_listening()
                        await websocket.send_json(
                            {"type": "stt.final", "text": transcript, "asrMs": asr_ms}
                        )
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
                    await _cancel_active_turn()
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
                    await _cancel_active_turn()
                    try:
                        parsed_commit, ctx_mesh = parse_ws_turn_commit(payload)
                    except ValidationError:
                        await websocket.send_json({"type": "error", "code": "validation_failed"})
                        continue

                    turn_text = parsed_commit.text.strip()
                    ctx_dict = parsed_commit.context
                    parsed_ctx = _parse_commit_context(ctx_dict)
                    requested_mode = _parse_requested_mode(parsed_commit.mode)
                    turn_voice = parsed_commit.voice

                    turn_caps = (
                        _parse_client_caps(parsed_commit.clientCapabilities)
                        or _parse_client_caps(ctx_dict.get("clientCapabilities"))
                        or default_client_caps
                    )
                    override_board = _parse_board_available(parsed_commit.boardHid)
                    if override_board is None:
                        override_board = _parse_board_available(ctx_dict.get("boardHid"))
                    turn_board_available = (
                        override_board if override_board is not None else default_board_available
                    )
                    raw_self_host = parsed_commit.selfHostId or ctx_dict.get("selfHostId")
                    turn_self_host_id = (
                        raw_self_host.strip()[:64]
                        if isinstance(raw_self_host, str) and raw_self_host.strip()
                        else default_self_host_id
                    )
                    try:
                        turn_target_host_id = (
                            HostId(parsed_commit.targetHostId)
                            if parsed_commit.targetHostId is not None
                            else (
                                HostId(ctx_mesh.targetHostId)
                                if ctx_mesh.targetHostId is not None
                                else default_target_host_id
                            )
                        )
                    except ValueError:
                        await websocket.send_json({"type": "error", "code": "validation_failed"})
                        continue
                    turn_follow_me = (
                        parsed_commit.followMe
                        if parsed_commit.followMe is not None
                        else (
                            ctx_mesh.followMe
                            if ctx_mesh.followMe is not None
                            else default_follow_me
                        )
                    )
                    turn_subnet_hash = (
                        parsed_commit.clientSubnetHash
                        or ctx_mesh.clientSubnetHash
                        or default_client_subnet_hash
                    )
                    turn_tz_offset = tz_offset_minutes_from_dt(parsed_ctx.base.local_time)

                    now = datetime.now(UTC)
                    turn_tools: tuple[ToolSpec, ...] = ()
                    turn_host_screen_consent = False
                    turn_pinned_host_id: HostId | None = None
                    turn_host_roster: str | None = None
                    async with app.state.database.sessions() as db:
                        consent_factory = getattr(app.state, "consent_repository", None)
                        host_factory = getattr(app.state, "host_repository", None)
                        audit_factory = getattr(app.state, "audit_log", None)
                        mesh_prep = await prepare_mesh_turn(
                            owner_id=owner.owner_id,
                            pet_id=str(pet_id),
                            text=turn_text,
                            client_caps=turn_caps,
                            board_available=turn_board_available,
                            self_host_id=turn_self_host_id,
                            target_host_id=turn_target_host_id,
                            follow_me=turn_follow_me,
                            tz_offset_minutes=turn_tz_offset,
                            include_autonomous_job=(
                                getattr(app.state, "job_runner", None) is not None
                            ),
                            consent_repo=(
                                consent_factory(db) if consent_factory is not None else None
                            ),
                            host_repo=host_factory(db) if host_factory is not None else None,
                            hub=ws_hub,
                            audit=audit_factory(db) if audit_factory is not None else None,
                            pre_router=build_mesh_pre_router(app.state),
                        )
                        if mesh_prep.outcome.kind == "not_found":
                            await websocket.send_json({"type": "error", "code": "not_found"})
                            continue
                        turn_tools = mesh_prep.tools
                        turn_host_screen_consent = mesh_prep.host_screen_consent
                        turn_pinned_host_id = mesh_prep.pinned_host_id
                        turn_host_roster = mesh_prep.host_roster
                        active_turn_pinned_host = turn_pinned_host_id
                        routing_text = (
                            mesh_prep.effective_text
                            if mesh_prep.outcome.kind == "repair_dispatch"
                            else turn_text
                        )

                        try:
                            mem_repo = app.state.memory_repository(db)
                            k_repo = app.state.knowledge_repository(db)
                            c_port = app.state.knowledge_candidate_adapter(db)
                            prepared = await prepare_turn(
                                pet_id=pet_id,
                                text=routing_text,
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
                                requested_mode, Channel.VOICE, routing_text, domain_matched=False
                            )
                            prepared = PreparedTurn(
                                context=replace(parsed_ctx.base, mode=fallback_dec.mode),
                                decision=fallback_dec,
                                domain_id=None,
                            )

                    settings = getattr(app.state, "settings", None)
                    chosen_mode = prepared.decision.mode
                    code_gen: TextGenerator | None = getattr(
                        app.state, "code_text_generator", None
                    ) or getattr(app.state, "expert_text_generator", None)
                    expert_gen: TextGenerator | None = (
                        getattr(app.state, "expert_text_generator", None) or code_gen
                    )
                    is_code_turn = is_code_or_tool_synthesis_request(routing_text)
                    if chosen_mode == TurnMode.EXPERT:
                        if is_code_turn and code_gen is not None:
                            max_tok = (
                                getattr(settings, "text_code_max_tokens", 4096)
                                if settings is not None
                                else 4096
                            )
                            timeout_s = (
                                getattr(settings, "text_code_timeout_s", 60.0)
                                if settings is not None
                                else 60.0
                            )
                            budget = ReplyBudget(max_output_tokens=max_tok, timeout_s=timeout_s)
                            turn_generator: TextGenerator = code_gen
                        else:
                            max_tok = (
                                settings.text_expert_max_tokens if settings is not None else 3000
                            )
                            timeout_s = (
                                settings.text_expert_timeout_s if settings is not None else 60.0
                            )
                            budget = ReplyBudget(max_output_tokens=max_tok, timeout_s=timeout_s)
                            turn_generator = expert_gen or app.state.text_generator
                    else:
                        default_timeout = (
                            float(settings.text_timeout_s)
                            if settings is not None and hasattr(settings, "text_timeout_s")
                            else 30.0
                        )
                        budget = ReplyBudget(
                            max_output_tokens=400, timeout_s=max(30.0, default_timeout)
                        )
                        turn_generator = app.state.text_generator

                    turn_tool_runner: ToolCallRunner | None = None
                    turn_cu_note: str | None = None
                    turn_cu_cleanup: CuCleanupCallback | None = None
                    ended_cu_events: list[ComputerUseSessionEvent] = []
                    turn_orch: Orchestrator | None = None
                    eff_locale = target_locale or Locale.VI
                    ws_ctx_factory: Callable[[EventSink], ActionContext] | None = None
                    if turn_caps or mesh_prep.outcome.kind == "repair_dispatch":
                        stream_orch = build_stream_orchestrator(app.state)
                        turn_orch = stream_orch
                        active_turn_orch = stream_orch
                        detected_lang = detect_text_language(routing_text)
                        prompt_locale = (
                            Locale.EN
                            if detected_lang == Locale.EN
                            else (target_locale or Locale.VI)
                        )
                        if turn_tools:
                            try:
                                turn_cu_note = await stream_orch.active_cu_session_prompt_note(
                                    owner.owner_id,
                                    host_id=turn_pinned_host_id,
                                    locale=prompt_locale,
                                    request_id=session_id,
                                )
                            except Exception:
                                logger.debug(
                                    "Failed to read active CU session prompt note",
                                    exc_info=True,
                                )
                                turn_cu_note = None

                        async def _cleanup_ws_cu(
                            reason: str,
                            *,
                            _orch: Orchestrator = stream_orch,
                            _ended: list[ComputerUseSessionEvent] = ended_cu_events,
                            _pin: HostId | None = turn_pinned_host_id,
                        ) -> ComputerUseSessionEvent | None:
                            ev = await _orch.cleanup_dangling_cu_session(
                                owner_id=owner.owner_id,
                                host_id=_pin,
                                reason=reason,
                                request_id=session_id,
                            )
                            if ev is not None:
                                _ended.append(ev)
                            return ev

                        turn_cu_cleanup = _cleanup_ws_cu

                        def _make_ws_ctx(
                            emit: EventSink,
                            *,
                            _locale: Locale = eff_locale,
                            _caps: frozenset[str] = frozenset(mesh_prep.effective_client_caps),
                            _board: bool = turn_board_available,
                            _self_host: str | None = turn_self_host_id,
                            _u5_consent: bool = turn_host_screen_consent,
                            _pinned_host: HostId | None = turn_pinned_host_id,
                            _follow: bool = bool(turn_follow_me),
                            _subnet: str | None = turn_subnet_hash,
                            _tz: int | None = turn_tz_offset,
                            _hab: bool = mesh_prep.habit_recorded_in_preroute,
                            _pre_tr: Any = mesh_prep.pre_route_target_required,
                            _pin_rsn: str | None = mesh_prep.pinned_target_reason,
                        ) -> ActionContext:
                            return ActionContext(
                                owner_id=owner.owner_id,
                                pet_id=str(pet_id),
                                locale=_locale,
                                client_caps=_caps,
                                board_available=_board,
                                emit_event=emit,
                                request_id=session_id,
                                self_host_id=_self_host,
                                host_screen_consent=_u5_consent,
                                pinned_host_id=_pinned_host,
                                follow_me=_follow,
                                client_subnet_hash=_subnet,
                                client_tz_offset_minutes=_tz,
                                habit_recorded_in_preroute=_hab,
                                pre_route_target_required=_pre_tr,
                                pinned_target_reason=_pin_rsn,
                            )

                        ws_ctx_factory = _make_ws_ctx

                        async def _run_voice_tool(
                            call: ToolCall,
                            emit: EventSink,
                            *,
                            _orch: Orchestrator = stream_orch,
                            _ctx_fn: Callable[[EventSink], ActionContext] = _make_ws_ctx,
                        ) -> Message:
                            return await _orch.handle_call(call, _ctx_fn(emit))

                        turn_tool_runner = _run_voice_tool
                    else:
                        active_turn_orch = None

                    async def run_reply(
                        target_text: str,
                        prep: PreparedTurn,
                        req_mode: RequestedMode | None,
                        bud: ReplyBudget,
                        gen: TextGenerator,
                        tools_arg: tuple[ToolSpec, ...],
                        runner_arg: ToolCallRunner | None,
                        v: str | None = None,
                        *,
                        _code_gen: TextGenerator | None = code_gen,
                        _code_timeout_s: float | None = (
                            getattr(settings, "text_code_timeout_s", None)
                            if settings is not None and code_gen is not None
                            else None
                        ),
                        _turn_note: str | None = turn_cu_note,
                        _cu_cleanup: CuCleanupCallback | None = turn_cu_cleanup,
                        _orch: Orchestrator | None = turn_orch,
                        _ended: list[ComputerUseSessionEvent] = ended_cu_events,
                        _roster: str | None = turn_host_roster,
                        _mesh: Any = mesh_prep,
                        _ctx_fn: Callable[[EventSink], ActionContext] | None = ws_ctx_factory,
                        _pin: HostId | None = turn_pinned_host_id,
                        _eff_loc: Locale = eff_locale,
                    ) -> None:
                        yield_svc: YieldDetachService | None = getattr(
                            app.state, "yield_detach_service", None
                        )
                        ws_yield_cb: (
                            Callable[
                                [tuple[ToolCall, ...], int, str | None],
                                Awaitable[YieldDetachResult | None],
                            ]
                            | None
                        ) = None
                        if yield_svc is not None and _orch is not None:
                            _y_svc = yield_svc
                            _y_goal = target_text
                            _y_pin = _pin
                            _y_loc = _eff_loc

                            async def _ws_yield_detach(
                                calls: tuple[ToolCall, ...],
                                elapsed_ms: int,
                                last_summary: str | None,
                            ) -> YieldDetachResult | None:
                                return await _y_svc.try_detach(
                                    YieldDetachRequest(
                                        owner_id=owner.owner_id,
                                        pet_id=pet_id.value,
                                        host_id=_y_pin,
                                        goal=_y_goal,
                                        tool_calls=calls,
                                        elapsed_ms=elapsed_ms,
                                        locale=_y_loc,
                                        last_action_summary=last_summary,
                                        request_id=session_id,
                                    )
                                )

                            ws_yield_cb = _ws_yield_detach
                        signals = {
                            "signals": list(
                                conversation_signals(target_text, len(prep.context.history))
                            )
                        }
                        domain_id_str = str(prep.domain_id) if prep.domain_id is not None else None
                        mode_val = prep.decision.mode.value
                        has_detail = False
                        emitted_cu_ended = False

                        async def _safe_cleanup(reason: str) -> None:
                            if _cu_cleanup is not None:
                                with contextlib.suppress(Exception):
                                    await _cu_cleanup(reason)
                            elif _orch is not None:
                                with contextlib.suppress(Exception):
                                    ev = await _orch.cleanup_dangling_cu_session(
                                        owner_id=owner.owner_id,
                                        host_id=_pin,
                                        reason=reason,
                                        request_id=session_id,
                                    )
                                    if ev is not None:
                                        _ended.append(ev)

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
                            if _mesh.outcome.focus_notice is not None and ws_focus_unsub is None:
                                await websocket.send_json(
                                    {
                                        "type": "host_focus_changed",
                                        **_mesh.outcome.focus_notice.wire_payload(),
                                    }
                                )
                            src_events: AsyncIterator[TurnEvent] | None = None
                            ws_turn_note_fn: Callable[[Locale], str] | None = None
                            if _turn_note:
                                _note_str = _turn_note

                                def _ws_note(_loc: Locale, _n: str = _note_str) -> str:
                                    return _n

                                ws_turn_note_fn = _ws_note
                            if _mesh.short_circuit:
                                src_events = stream_short_circuit_turn(_mesh, _eff_loc)
                            elif (
                                _mesh.outcome.kind == "repair_dispatch"
                                and _orch is not None
                                and _ctx_fn is not None
                            ):
                                base_req = build_request(
                                    _mesh.effective_text,
                                    context=prep.context,
                                    locale=session.locale,
                                    mode=prep.decision.mode,
                                    budget=bud,
                                    tools=tools_arg,
                                    turn_note=ws_turn_note_fn,
                                    host_roster=_roster if tools_arg else None,
                                )
                                src_events = execute_repair_dispatch(
                                    outcome=_mesh.outcome,
                                    generator=gen,
                                    base_request=base_req,
                                    orchestrator=_orch,
                                    action_ctx_factory=_ctx_fn,
                                    sections=(prep.decision.mode == TurnMode.EXPERT),
                                    locale_vi=(base_req.locale == Locale.VI),
                                    code_generator=_code_gen,
                                    code_timeout_s=_code_timeout_s,
                                    cu_cleanup=_cu_cleanup,
                                    yield_detach=ws_yield_cb,
                                    yield_ms=DEFAULT_YIELD_MS,
                                )
                            elif _mesh.pre_route_target_required is not None:
                                ws_req = build_request(
                                    target_text,
                                    context=prep.context,
                                    locale=session.locale,
                                    mode=prep.decision.mode,
                                    budget=bud,
                                    tools=tools_arg,
                                    turn_note=ws_turn_note_fn,
                                    host_roster=_roster if tools_arg else None,
                                )
                                raw_ev = chat_turn(
                                    gen,
                                    ws_req,
                                    sections=(prep.decision.mode == TurnMode.EXPERT),
                                    locale_vi=(ws_req.locale == Locale.VI),
                                    tool_runner=runner_arg,
                                    code_generator=_code_gen,
                                    code_timeout_s=_code_timeout_s,
                                    cu_cleanup=_cu_cleanup,
                                    yield_detach=ws_yield_cb,
                                    yield_ms=DEFAULT_YIELD_MS,
                                )
                                src_events = stream_with_target_fallback(
                                    raw_ev, _mesh.target_required_event
                                )

                            async for event in session.stream_turn_reply(
                                target_text,
                                prep.context,
                                voice=v,
                                mode=prep.decision.mode,
                                budget=bud,
                                generator=gen,
                                code_generator=_code_gen,
                                code_timeout_s=_code_timeout_s,
                                tools=tools_arg,
                                tool_runner=runner_arg,
                                turn_note=_turn_note,
                                cu_cleanup=_cu_cleanup,
                                host_roster=_roster if tools_arg else None,
                                source_events=src_events,
                                yield_detach=ws_yield_cb,
                                yield_ms=DEFAULT_YIELD_MS,
                            ):
                                if isinstance(event, GestureChosen):
                                    await websocket.send_json(
                                        {"type": "gesture", "name": event.gesture.value}
                                    )
                                elif isinstance(event, TurnDetachedEvent):
                                    await websocket.send_json(
                                        {"type": "autonomous_job", **event.wire_payload()}
                                    )
                                elif isinstance(event, ActionPendingEvent):
                                    await websocket.send_json(
                                        {"type": "action_pending", **event.wire_payload()}
                                    )
                                elif isinstance(event, ActionTargetRequiredEvent):
                                    await websocket.send_json(
                                        {"type": "action_target_required", **event.wire_payload()}
                                    )
                                elif isinstance(event, HostFocusChangedNotice):
                                    await websocket.send_json(
                                        {"type": "host_focus_changed", **event.wire_payload()}
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
                                elif isinstance(event, HostWakeRequestEvent):
                                    wake_msg: dict[str, Any] = {
                                        "type": "host_wake_request",
                                        "invocationId": event.invocation_id,
                                        "hostId": event.host_id,
                                        "target": event.target,
                                    }
                                    if event.mac_address is not None:
                                        wake_msg["macAddress"] = event.mac_address
                                    await websocket.send_json(wake_msg)
                                elif isinstance(event, ActionResultEvent):
                                    await websocket.send_json(
                                        {"type": "action_result", **event.wire_payload()}
                                    )
                                elif isinstance(event, ComputerUseSessionEvent):
                                    if event.status == "ended":
                                        emitted_cu_ended = True
                                    await websocket.send_json(
                                        {"type": "cu_session"} | session_event_payload(event)
                                    )
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
                            if _cu_cleanup is not None or _orch is not None:
                                with contextlib.suppress(BaseException):
                                    await asyncio.shield(_safe_cleanup("turn_cancelled"))
                            if not _is_closing() and _ended and not emitted_cu_ended:
                                with contextlib.suppress(Exception):
                                    await websocket.send_json(
                                        {"type": "cu_session"} | session_event_payload(_ended[-1])
                                    )
                        except Exception:
                            logger.warning("voice_ws_reply_error")
                            if _cu_cleanup is not None or _orch is not None:
                                with contextlib.suppress(BaseException):
                                    await asyncio.shield(_safe_cleanup("turn_failed"))
                            if not _is_closing():
                                if _ended and not emitted_cu_ended:
                                    with contextlib.suppress(Exception):
                                        await websocket.send_json(
                                            {"type": "cu_session"}
                                            | session_event_payload(_ended[-1])
                                        )
                                with contextlib.suppress(Exception):
                                    await websocket.send_json(
                                        {"type": "error", "code": "reply_failed"}
                                    )

                    active_turn_task = asyncio.create_task(
                        run_reply(
                            mesh_prep.effective_text,
                            prepared,
                            requested_mode,
                            budget,
                            turn_generator,
                            turn_tools,
                            turn_tool_runner,
                            v=turn_voice,
                        )
                    )

                    def _on_turn_done(_t: asyncio.Task[None]) -> None:
                        if not _is_closing():
                            ft = asyncio.create_task(_flush_pending_wake_items())
                            wake_flush_tasks.add(ft)
                            ft.add_done_callback(wake_flush_tasks.discard)

                    active_turn_task.add_done_callback(_on_turn_done)

                elif msg_type == "action.decide":
                    raw_aid = payload.get("approvalId")
                    raw_dec = payload.get("decision")
                    raw_scope = payload.get("scope")
                    raw_sig = payload.get("signature")
                    raw_pub = payload.get("publicKey")
                    if isinstance(raw_aid, str) and isinstance(raw_dec, str):
                        try:
                            aid = uuid.UUID(raw_aid)
                            dec = ApprovalDecision(raw_dec)
                            scope = (
                                ApprovalScope(raw_scope)
                                if isinstance(raw_scope, str)
                                else ApprovalScope.ONCE
                            )
                            sig_bytes = (
                                _decode_b64_flex(raw_sig, 64) if isinstance(raw_sig, str) else None
                            )
                            pub_bytes = (
                                _decode_b64_flex(raw_pub, 65) if isinstance(raw_pub, str) else None
                            )
                            ws_orch = build_stream_orchestrator(app.state)
                            await ws_orch.decide_approval(
                                approval_id=aid,
                                owner_id=owner.owner_id,
                                decision=dec,
                                request_id=session_id,
                                scope=scope,
                                signature=sig_bytes,
                                signer_public_key=pub_bytes,
                            )
                        except ValueError:
                            pass

                elif msg_type == "action.stop":
                    raw_reason = payload.get("reason")
                    try:
                        stop_host_id = parse_ws_optional_host_id(payload.get("hostId"))
                    except ValueError:
                        await websocket.send_json({"type": "error", "code": "validation_failed"})
                        continue
                    stop_reason = (
                        str(raw_reason)[:64]
                        if isinstance(raw_reason, str) and raw_reason.strip()
                        else "owner_ws"
                    )
                    ws_orch = build_stream_orchestrator(app.state)
                    try:
                        stop_summary = await ws_orch.emergency_stop(
                            owner_id=owner.owner_id,
                            reason=stop_reason,
                            request_id=session_id,
                            host_id=stop_host_id,
                        )
                    except ValueError:
                        await websocket.send_json({"type": "error", "code": "not_found"})
                        continue
                    await websocket.send_json(
                        {
                            "type": "action_stopped",
                            "stopped": stop_summary.stopped,
                            "hostsNotified": stop_summary.hosts_notified,
                            "approvalsCancelled": stop_summary.approvals_cancelled,
                            "grantsSuspended": stop_summary.grants_suspended,
                            "hostId": stop_summary.host_id,
                        }
                    )

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
                    await _cancel_active_turn()
                    session.abort(str(payload.get("reason", "cancel")))
                    await _flush_pending_wake_items()

                elif msg_type == "bye":
                    ws_closing = True
                    was_turn_running = active_turn_task is not None and not active_turn_task.done()
                    await _cancel_active_turn()
                    if was_turn_running and active_turn_orch is not None:
                        with contextlib.suppress(BaseException):
                            await asyncio.shield(
                                active_turn_orch.cleanup_dangling_cu_session(
                                    owner_id=owner.owner_id,
                                    host_id=active_turn_pinned_host,
                                    reason="turn_disconnected",
                                    request_id=session_id,
                                )
                            )
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        ws_closing = True
    finally:
        ws_closing = True
        if callable(ws_focus_unsub):
            with contextlib.suppress(BaseException):
                ws_focus_unsub()
        if callable(ws_wake_unsub):
            with contextlib.suppress(BaseException):
                ws_wake_unsub()
        for ft in list(wake_flush_tasks):
            if not ft.done():
                ft.cancel()
        was_turn_running = active_turn_task is not None and not active_turn_task.done()
        await _cancel_active_turn()
        if was_turn_running and active_turn_orch is not None:
            with contextlib.suppress(BaseException):
                await asyncio.shield(
                    active_turn_orch.cleanup_dangling_cu_session(
                        owner_id=owner.owner_id,
                        host_id=active_turn_pinned_host,
                        reason="turn_disconnected",
                        request_id=session_id,
                    )
                )
