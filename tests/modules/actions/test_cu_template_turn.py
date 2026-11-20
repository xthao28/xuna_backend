"""UDC-17 S17.5: a rerun turn tells the model the template's scope up front.

The line is added to the system prompt for that turn only, names the computer by the alias the
backend reads for this owner (never another owner's host), and is absent from ordinary turns.
"""

import asyncio
import uuid

from tests.modules.actions.test_actions_api_and_turn import (
    FAKE_PUBKEY,
    NOW,
    PET_HEX,
    _build_test_harness,
    _done,
    _ScriptedLLM,
)

from xuna_backend.modules.actions.application.cu_template import CuTemplatePin
from xuna_backend.modules.cognition.domain.types import Locale, Role, TextDelta, ToolSpec
from xuna_backend.modules.conversation.application.chat_turn import build_request
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)

_EMPTY: dict[str, object] = {"type": "object", "properties": {}, "additionalProperties": False}


def _pin(host_id: str) -> CuTemplatePin:
    return CuTemplatePin.create(
        name="Báo cáo tuần",
        host_id=host_id,
        apps=["notes", "mail"],
        domains=["docs.example.com"],
        max_steps=20,
        max_minutes=10,
    )


def test_the_note_states_the_exact_scope_in_the_turn_language() -> None:
    pin = _pin(str(uuid.uuid4()))
    vi = pin.prompt_note(Locale.VI, "Mac")
    assert "chạy lại mẫu đã lưu 'Báo cáo tuần'" in vi
    assert "trên máy 'Mac'" in vi
    assert "apps=[mail, notes]" in vi
    assert "domains=[docs.example.com]" in vi
    assert "maxSteps tối đa 20" in vi and "maxMinutes tối đa 10" in vi
    en = pin.prompt_note(Locale.EN, None)
    assert "no longer enrolled" in en
    assert "apps=[mail, notes]" in en


def test_build_request_adds_the_note_only_with_tools() -> None:
    tool = ToolSpec(name="cu__session_start", description="d", parameters=_EMPTY)
    with_tools = build_request(
        "Gửi báo cáo tuần", Locale.VI, tools=(tool,), turn_note=lambda _: "\n- NOTE"
    )
    assert with_tools.messages[0].content.endswith("\n- NOTE")
    without_tools = build_request("Gửi báo cáo tuần", Locale.VI, turn_note=lambda _: "\n- NOTE")
    assert "NOTE" not in without_tools.messages[0].content


def _online_mac(client: object, host_repo: object, owner_id: uuid.UUID) -> Host:
    host = Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias("Mac"),
        platform=HostPlatform.MACOS,
        hostname="mac-host",
        os_version="15.0",
        agent_version="0.1.0",
        public_key=FAKE_PUBKEY,
        status=HostStatus.ACTIVE,
        created_at=NOW,
    )
    asyncio.run(host_repo.save_host(host))  # type: ignore[attr-defined]
    asyncio.run(
        host_repo.save_manifest(  # type: ignore[attr-defined]
            Manifest(
                host_id=host.host_id,
                tools=(
                    ToolDescriptor(
                        name="media.play_pause",
                        description="Toggle media playback.",
                        input_schema=_EMPTY,
                        output_schema=_EMPTY,
                        family=ToolFamily.MEDIA,
                        risk=RiskTier.R1,
                    ),
                ),
                updated_at=NOW,
            )
        )
    )
    asyncio.run(
        host_repo.set_grants(  # type: ignore[attr-defined]
            host.host_id,
            [Grant(host.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW)],
        )
    )
    hub = client.app.state.host_session_hub  # type: ignore[attr-defined]

    async def noop(*_: object) -> None:
        return None

    asyncio.run(hub.open_session(str(host.host_id), object(), noop, noop, noop))
    return host


def _turn(client: object, token: str, cu_template: dict[str, object] | None) -> int:
    body: dict[str, object] = {
        "schemaVersion": 1,
        "text": "Gửi báo cáo tuần cho Lan",
        "clientCapabilities": ["host_actions_v1"],
    }
    if cu_template is not None:
        body["cuTemplate"] = cu_template
    resp = client.post(  # type: ignore[attr-defined]
        f"/v1/conversations/{PET_HEX}/turns",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    )
    return int(resp.status_code)


def _template_body(host_id: str) -> dict[str, object]:
    return {
        "name": "Báo cáo tuần",
        "hostId": host_id,
        "apps": ["notes", "mail"],
        "domains": ["docs.example.com"],
        "maxSteps": 20,
        "maxMinutes": 10,
    }


def test_a_rerun_turn_carries_the_note_with_this_owners_computer() -> None:
    llm = _ScriptedLLM([[TextDelta("Được"), _done(5)]] * 4)
    client, issuer, owner, host_repo, _, _ = _build_test_harness(llm)
    mac = _online_mac(client, host_repo, owner.owner_id)
    token = issuer.token("owner-1")

    assert _turn(client, token, _template_body(str(mac.host_id))) == 200
    system = llm.requests[-1].messages[0]
    assert system.role == Role.SYSTEM
    assert "chạy lại mẫu đã lưu 'Báo cáo tuần'" in system.content
    assert "trên máy 'Mac'" in system.content

    # A host id that is not this owner's computer is never named.
    assert _turn(client, token, _template_body(str(uuid.uuid4()))) == 200
    system = llm.requests[-1].messages[0].content
    assert "không còn được ghép" in system
    assert "'Mac'" not in system

    # An ordinary turn carries no note.
    assert _turn(client, token, None) == 200
    assert "chạy lại mẫu" not in llm.requests[-1].messages[0].content

    # A malformed template is refused before any provider call.
    calls = len(llm.requests)
    bad = {**_template_body(str(mac.host_id)), "apps": ["Not An Id"]}
    assert _turn(client, token, bad) == 422
    assert len(llm.requests) == calls


def test_followup_sse_turn_injects_active_cu_session_note_and_combines_with_template() -> None:
    from datetime import UTC, datetime

    from xuna_backend.modules.actions.application.computer_use import ComputerUseSessionState

    llm = _ScriptedLLM([[TextDelta("Được"), _done(5)]] * 3)
    client, issuer, owner, host_repo, _, _ = _build_test_harness(llm)
    mac = _online_mac(client, host_repo, owner.owner_id)
    token = issuer.token("owner-1")
    cu_store = client.app.state.cu_session_store  # type: ignore[union-attr]
    now_utc = datetime.now(UTC)

    asyncio.run(
        cu_store.start(
            ComputerUseSessionState(
                session_id="cu-sse-followup",
                owner_id=owner.owner_id,
                host_id=mac.host_id,
                goal="Draft note",
                allowed_apps=("notes",),
                allowed_domains=("docs.example.com",),
                max_steps=20,
                max_minutes=10,
                started_at_utc=now_utc,
                last_heartbeat_at=now_utc,
                target_alias="Mac",
                last_window={"appId": "notes", "windowTitle": "Weekly Plan"},
            ),
            now_utc,
        )
    )

    assert _turn(client, token, None) == 200
    sys_prompt = llm.requests[-1].messages[0].content
    assert "cu-sse-followup" in sys_prompt
    assert "Weekly Plan" in sys_prompt

    # Combines cleanly with cuTemplate when both are present
    assert _turn(client, token, _template_body(str(mac.host_id))) == 200
    combined = llm.requests[-1].messages[0].content
    assert "chạy lại mẫu đã lưu 'Báo cáo tuần'" in combined
    assert "cu-sse-followup" in combined

    # Absent once the CU session ends
    asyncio.run(cu_store.mark_ended("cu-sse-followup", "completed", now_utc))
    assert _turn(client, token, None) == 200
    assert "cu-sse-followup" not in llm.requests[-1].messages[0].content
