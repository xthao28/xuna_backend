"""Tests for ActionPendingEvent missionEnvelope wire serialization and propagation (Wave E M1)."""

import uuid

from xuna_backend.modules.actions.domain.entities import ActionPendingEvent


def test_action_pending_wire_payload_without_mission_envelope() -> None:
    """Standard 1-step action pending event omits missionEnvelope on the wire."""
    approval_id = str(uuid.uuid4())
    event = ActionPendingEvent(
        approval_id=approval_id,
        target="MacBook Pro",
        summary="Phát nhạc trên Spotify",
        risk="R2",
        expires_in=120,
        host_id="host-123",
        args_sha256="abc123sha",
        expires_at="2026-11-15T12:00:00Z",
        standing_tool="media.play",
        standing_arguments={"query": "lofi"},
        standing_approved_count=3,
        mission_envelope=None,
    )

    payload = event.wire_payload()

    assert payload["approvalId"] == approval_id
    assert payload["target"] == "MacBook Pro"
    assert payload["summary"] == "Phát nhạc trên Spotify"
    assert payload["risk"] == "r2"  # strictly lowercase for mobile ActionEventMapper
    assert payload["expiresIn"] == 120
    assert payload["hostId"] == "host-123"
    assert payload["argsSha256"] == "abc123sha"
    assert payload["expiresAt"] == "2026-11-15T12:00:00Z"
    assert payload["standing"] == {
        "tool": "media.play",
        "arguments": {"query": "lofi"},
        "approvedCount": 3,
    }
    # Backward compatibility: missionEnvelope key MUST NOT exist when None
    assert "missionEnvelope" not in payload


def test_action_pending_wire_payload_with_mission_envelope() -> None:
    """Multi-step mission / AutonomousJob event includes missionEnvelope on the wire."""
    approval_id = str(uuid.uuid4())
    envelope = {
        "goal": "Viết email báo cáo quý 4 và gửi cho sếp",
        "apps": ["mail", "calc"],
        "domains": ["internal.company.com"],
        "maxSteps": 20,
        "maxMinutes": 10,
        "planSteps": [
            "Mở ứng dụng tính toán để kiểm tra số liệu",
            "Mở Mail và soạn nội dung bản thảo",
            "Gửi email báo cáo",
        ],
    }
    event = ActionPendingEvent(
        approval_id=approval_id,
        target="Office PC",
        summary="Bắt đầu phiên điều khiển: Viết email báo cáo",
        risk="R3",
        expires_in=180,
        host_id="host-456",
        args_sha256="xyz789sha",
        expires_at="2026-11-15T12:03:00Z",
        mission_envelope=envelope,
    )

    payload = event.wire_payload()

    assert payload["approvalId"] == approval_id
    assert payload["target"] == "Office PC"
    assert payload["summary"] == "Bắt đầu phiên điều khiển: Viết email báo cáo"
    assert payload["risk"] == "r3"
    assert payload["expiresIn"] == 180
    assert payload["hostId"] == "host-456"
    assert payload["argsSha256"] == "xyz789sha"
    assert payload["expiresAt"] == "2026-11-15T12:03:00Z"
    assert "standing" not in payload
    # Wire compatibility: missionEnvelope must be present with identical dict
    assert "missionEnvelope" in payload
    assert payload["missionEnvelope"] == envelope
