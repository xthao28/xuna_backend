"""WebSocket voice session route (VOICE-WS, 2026-10-23).

Protocol:
- Path: `GET /v1/voice` (WebSocket)
- Handshake & auth: Client sends hello with Firebase ID token or device proof within 5s.
- Real-time full-duplex audio & control stream.
"""

import asyncio
import json
import logging
import uuid
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
from xuna_backend.modules.cognition.domain.types import (
    ChatFinished,
    Locale,
    TextDelta,
)
from xuna_backend.modules.consent.application.ports import ConsentRepository
from xuna_backend.modules.consent.domain.consent import ConsentClass
from xuna_backend.modules.conversation.application.chat_turn import (
    HistoryLine,
    PetAction,
    PetMood,
    PetSelf,
    PetTemperament,
    TurnContext,
)
from xuna_backend.modules.conversation.application.spoken_turn import (
    SpokenAudio,
    SpokenFinished,
)
from xuna_backend.modules.conversation.application.voice_session import (
    VoiceSession,
    VoiceSessionError,
)
from xuna_backend.modules.identity.application.ports import IdentityRepository
from xuna_backend.modules.identity.domain.pet_id import PetId

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])


def _parse_turn_context(raw: dict[str, Any]) -> TurnContext:
    history_items: list[HistoryLine] = []
    for item in raw.get("history", []):
        if isinstance(item, dict) and "text" in item:
            from_owner = item.get("role") == "owner" or item.get("from_owner", False)
            history_items.append(HistoryLine(from_owner=from_owner, text=str(item["text"])))

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

    return TurnContext(
        history=tuple(history_items),
        pet=pet_self,
        confirmed_facts=tuple(raw.get("confirmedFacts", ())),
    )


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
                    context = _parse_turn_context(payload.get("context", {}))
                    turn_voice = payload.get("voice")

                    async def run_reply(
                        target_text: str, ctx: TurnContext, v: str | None = None
                    ) -> None:
                        try:
                            async for event in session.stream_turn_reply(
                                target_text, ctx, voice=v
                            ):
                                if isinstance(event, TextDelta):
                                    await websocket.send_json(
                                        {"type": "reply.delta", "text": event.text}
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
                                        }
                                    )
                                elif isinstance(event, ChatFinished):
                                    await websocket.send_json(
                                        {
                                            "type": "reply.finished",
                                            "latencyMs": event.latency_ms,
                                            "firstAudioMs": None,
                                            "spoken": False,
                                        }
                                    )
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            logger.warning(f"Voice websocket reply error: {e}")
                            await websocket.send_json({"type": "error", "code": "reply_failed"})

                    active_turn_task = asyncio.create_task(
                        run_reply(turn_text, context, v=turn_voice)
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
