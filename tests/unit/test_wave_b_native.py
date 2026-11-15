"""Unit and integration tests for Wave B Native OS Bridge (Milestone M1)."""

import pytest

from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.domain.action_summary import ActionSummary
from xuna_backend.modules.cognition.domain.types import Locale
from xuna_backend.modules.hosts.domain.entities import (
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
    validate_tool_name,
)


def _make_descriptor(
    name: str,
    family: ToolFamily,
    risk: RiskTier,
    *,
    input_schema: dict[str, object] | None = None,
    output_schema: dict[str, object] | None = None,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Test descriptor for {name}",
        input_schema=input_schema or {"type": "object", "additionalProperties": False},
        output_schema=output_schema or {"type": "object", "additionalProperties": False},
        family=family,
        risk=risk,
        permission=ToolPermissionState.GRANTED,
        policy_version=1,
    )


class TestWaveBNativeToolNames:
    def test_validate_tool_name_maps_native_and_web_families(self) -> None:
        assert validate_tool_name("notes.create") == ToolFamily.APP
        assert validate_tool_name("notes.append") == ToolFamily.APP
        assert validate_tool_name("notes.search") == ToolFamily.APP
        assert validate_tool_name("reminders.create") == ToolFamily.APP
        assert validate_tool_name("calendar.list_events") == ToolFamily.APP
        assert validate_tool_name("calendar.create_event") == ToolFamily.APP
        assert validate_tool_name("mail.create_draft") == ToolFamily.APP
        assert validate_tool_name("mail.send") == ToolFamily.APP
        assert validate_tool_name("finder.reveal") == ToolFamily.FILES
        assert validate_tool_name("web.navigate") == ToolFamily.URL
        assert validate_tool_name("web.snapshot") == ToolFamily.URL

    def test_validate_tool_name_shortcuts_supports_workflow_and_app(self) -> None:
        assert (
            validate_tool_name("shortcuts.run", expected_family=ToolFamily.WORKFLOW)
            == ToolFamily.WORKFLOW
        )
        assert (
            validate_tool_name("shortcuts.run", expected_family=ToolFamily.APP)
            == ToolFamily.APP
        )

    def test_validate_tool_name_expected_family_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            validate_tool_name("finder.reveal", expected_family=ToolFamily.APP)
        with pytest.raises(ValueError, match="does not match"):
            validate_tool_name("web.navigate", expected_family=ToolFamily.APP)

    def test_validate_tool_name_blocks_forbidden(self) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            validate_tool_name("shell.exec")
        with pytest.raises(ValueError, match="forbidden"):
            validate_tool_name("osascript.run")
        with pytest.raises(ValueError, match="forbidden"):
            validate_tool_name("powershell.exec")


class TestWaveBActionSummary:
    def test_format_pending_bilingual_native_tools(self) -> None:
        target = "Mac Studio"

        # notes.create
        summary_vi = ActionSummary.format_pending(
            "notes.create", {"title": "Meeting Notes"}, target, Locale.VI
        )
        assert "Tạo ghi chú 'Meeting Notes'" in summary_vi
        assert target in summary_vi

        summary_en = ActionSummary.format_pending(
            "notes.create", {"title": "Meeting Notes"}, target, Locale.EN
        )
        assert "Create note 'Meeting Notes'" in summary_en

        # notes.append
        append_vi = ActionSummary.format_pending(
            "notes.append", {"title": "Roadmap", "text": "Next steps"}, target, Locale.VI
        )
        assert "Thêm nội dung vào ghi chú 'Roadmap'" in append_vi

        # calendar.create_event
        cal_vi = ActionSummary.format_pending(
            "calendar.create_event", {"title": "Sprint Sync"}, target, Locale.VI
        )
        assert "Tạo sự kiện lịch 'Sprint Sync'" in cal_vi

        # finder.reveal
        finder_vi = ActionSummary.format_pending(
            "finder.reveal", {"path": "/Users/xuna/data.csv"}, target, Locale.VI
        )
        assert "Hiển thị tệp '/Users/xuna/data.csv'" in finder_vi

        # mail.create_draft (draft only)
        mail_vi = ActionSummary.format_pending(
            "mail.create_draft", {"subject": "Weekly Update"}, target, Locale.VI
        )
        assert "Tạo thư nháp 'Weekly Update'" in mail_vi

        mail_en = ActionSummary.format_pending(
            "mail.create_draft", {"subject": "Weekly Update"}, target, Locale.EN
        )
        assert "Create draft email 'Weekly Update'" in mail_en

    def test_format_result_bilingual(self) -> None:
        target = "Mac Studio"
        ok_res = ActionSummary.format_result(
            "notes.create", {"title": "Task"}, target, "ok", Locale.VI
        )
        assert "✓ Đã thực hiện" in ok_res
        assert "Tạo ghi chú 'Task'" in ok_res

        fail_res = ActionSummary.format_result(
            "notes.append", {"title": "Task"}, target, "failed", Locale.VI
        )
        assert "✗ Không thực hiện được" in fail_res
        assert "lỗi thực thi" in fail_res


class TestWaveBRiskFloors:
    def test_mail_send_is_strictly_r3(self) -> None:
        desc_send = _make_descriptor("mail.send", ToolFamily.APP, RiskTier.R2)
        # Even if descriptor was R2, mail.send MUST be hard-gated at R3
        risk = Orchestrator._effective_tool_risk(desc_send, {}, None)
        assert risk == RiskTier.R3

        floor = Orchestrator._declared_risk_floor(desc_send, {})
        assert floor == RiskTier.R3

    def test_mail_create_draft_is_r2(self) -> None:
        desc_draft = _make_descriptor("mail.create_draft", ToolFamily.APP, RiskTier.R2)
        risk = Orchestrator._effective_tool_risk(desc_draft, {}, None)
        assert risk == RiskTier.R2

        floor = Orchestrator._declared_risk_floor(desc_draft, {})
        assert floor == RiskTier.R2

    def test_notes_and_calendar_risks(self) -> None:
        desc_create = _make_descriptor("notes.create", ToolFamily.APP, RiskTier.R2)
        assert Orchestrator._effective_tool_risk(desc_create, {}, None) == RiskTier.R2

        desc_search = _make_descriptor("notes.search", ToolFamily.APP, RiskTier.R1)
        assert Orchestrator._effective_tool_risk(desc_search, {}, None) == RiskTier.R1

        desc_cal = _make_descriptor("calendar.list_events", ToolFamily.APP, RiskTier.R1)
        assert Orchestrator._effective_tool_risk(desc_cal, {}, None) == RiskTier.R1
