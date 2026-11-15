"""Unit and integration tests for Wave B Browser Engine Integration (Milestone M2)."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.application.prompts import (
    MAX_TOOL_OUTPUT_BYTES,
    encode_tool_payload,
)
from xuna_backend.modules.actions.application.workflows import validate_https_public_url
from xuna_backend.modules.actions.domain.action_summary import ActionSummary
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.domain.entities import (
    HostStatus,
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


class TestWaveBBrowserToolNames:
    def test_validate_tool_name_maps_all_7_web_tools_to_url(self) -> None:
        tools = [
            "web.navigate",
            "web.snapshot",
            "web.click",
            "web.fill",
            "web.select",
            "web.extract",
            "web.wait_for",
        ]
        for t in tools:
            assert validate_tool_name(t) == ToolFamily.URL
            assert validate_tool_name(t, expected_family=ToolFamily.URL) == ToolFamily.URL

    def test_validate_tool_name_expected_family_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            validate_tool_name("web.navigate", expected_family=ToolFamily.APP)
        with pytest.raises(ValueError, match="does not match"):
            validate_tool_name("web.snapshot", expected_family=ToolFamily.HOST)

    def test_browser_tool_descriptors_enforce_strict_schemas(self) -> None:
        for name, risk in [
            ("web.navigate", RiskTier.R2),
            ("web.snapshot", RiskTier.R1),
            ("web.click", RiskTier.R2),
            ("web.fill", RiskTier.R2),
            ("web.select", RiskTier.R2),
            ("web.extract", RiskTier.R1),
            ("web.wait_for", RiskTier.R1),
        ]:
            desc = _make_descriptor(name, ToolFamily.URL, risk)
            assert desc.family == ToolFamily.URL
            assert desc.risk == risk
            assert desc.input_schema["additionalProperties"] is False
            assert desc.output_schema["additionalProperties"] is False


class TestWaveBAntiSSRF:
    def test_validate_https_public_url_allows_valid_domains(self) -> None:
        assert validate_https_public_url("https://example.com/path?q=1") == "example.com"
        assert validate_https_public_url("https://shop.vietnam.vn/cart") == "shop.vietnam.vn"
        assert validate_https_public_url("https://github.com/xthao28/xuna_pet") == "github.com"
        assert validate_https_public_url("https://example.com:8443/api") == "example.com"

    def test_validate_https_public_url_rejects_ssrf_and_private_addresses(self) -> None:
        bad_urls = [
            "http://example.com",
            "ftp://example.com",
            "file:///etc/passwd",
            "https://127.0.0.1/admin",
            "https://10.0.0.1/internal",
            "https://192.168.1.1/router",
            "https://169.254.169.254/latest/meta-data/",
            "https://[::1]/api",
            "https://localhost:8080",
            "https://app.localhost",
            "https://service.local",
            "https://admin:pass@example.com",
            "https://internal",
            "https://localhost.localdomain",
            "https://sub.localhost.localdomain",
            "https://metadata.google.internal",
            "https://instance-data.ec2.internal",
            "https://router.lan",
            "https://home.arpa",
            "https://device.localdomain",
            "https://127.0.0.0x1",
            "https://127.0.0.0x01",
            "https://0x7f.0.0.0x1",
            "https://169.254.169.0xfe",
            "https://10.0.0.0x1",
            "https://192.168.1.0x1",
            "https://127.0x1",
            "https://example.com:0",
            "https://example.com:65536",
            "https://127.0.0.1%00/",
            "https://127.0.0.1%2e/",
            "https://127.0.0.1%23/",
            "https://127.0.0.1%2f/",
            "https://169.254.169.254\x00/",
            "https://10.0.0.1\x00/",
        ]
        for url in bad_urls:
            with pytest.raises(ValueError, match="url_invalid"):
                validate_https_public_url(url)

    def test_validate_https_public_url_rejects_adversarial_evasion_and_null_bytes(self) -> None:
        evasion_vectors = [
            "https://127.0.0.1%00/",
            "https://127.0.0.1%2e/",
            "https://127.0.0.1%23/",
            "https://127.0.0.1%2f/",
            "https://127.0.0.1%3f/",
            "https://127.0.0.1%20/",
            "https://127.0.0.1\x00/",
            "https://169.254.169.254\x00/",
            "https://10.0.0.1\x00/",
            "https://-bad.com/",
            "https://bad-.com/",
            "https://a..b.com/",
        ]
        for url in evasion_vectors:
            with pytest.raises(ValueError, match="url_invalid"):
                validate_https_public_url(url)

    @pytest.mark.anyio
    async def test_orchestrator_handle_host_call_rejects_invalid_url_for_web_navigate(
        self,
    ) -> None:
        orch = object.__new__(Orchestrator)
        orch._clock = lambda: datetime.now(UTC)
        ref_desc = _make_descriptor(
            "web.navigate",
            ToolFamily.URL,
            RiskTier.R2,
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        )
        mock_host = MagicMock()
        mock_host.status = HostStatus.ACTIVE
        mock_host.host_id = "host_1"

        mock_manifest = MagicMock()
        mock_manifest.find_tool = MagicMock(return_value=ref_desc)

        orch._host_repo = MagicMock()
        orch._host_repo.list_hosts = AsyncMock(return_value=[mock_host])
        orch._host_repo.get_manifest = AsyncMock(return_value=mock_manifest)
        orch._host_repo.get_grants = AsyncMock(return_value=[])
        orch._hub = MagicMock()
        orch._hub.is_online = MagicMock(return_value=True)

        ctx = MagicMock()
        ctx.self_host_id = None
        ctx.host_screen_consent = True

        # Call with SSRF private IP
        call = ToolCall(
            id="call_1",
            name="web__navigate",
            arguments={"url": "https://127.0.0.1/admin"},
        )
        msg = await orch._handle_host_call("web.navigate", call, ctx)

        assert msg.role.value == "tool"
        assert msg.tool_call_id == "call_1"
        data = json.loads(msg.content)
        assert data["status"] == "invalid_args"
        assert data["output"]["errorCode"] == "url_invalid"


class TestWaveBBrowserRiskTiers:
    def test_browser_tools_default_risk_tiers(self) -> None:
        nav = _make_descriptor("web.navigate", ToolFamily.URL, RiskTier.R2)
        assert (
            Orchestrator._effective_tool_risk(nav, {"url": "https://example.com"}, None)
            == RiskTier.R2
        )
        assert (
            Orchestrator._declared_risk_floor(nav, {"url": "https://example.com"})
            == RiskTier.R2
        )

        snap = _make_descriptor("web.snapshot", ToolFamily.URL, RiskTier.R1)
        assert Orchestrator._effective_tool_risk(snap, {}, None) == RiskTier.R1
        assert Orchestrator._declared_risk_floor(snap, {}) == RiskTier.R1

        sel = _make_descriptor("web.select", ToolFamily.URL, RiskTier.R2)
        assert (
            Orchestrator._effective_tool_risk(sel, {"ref": "e1", "value": "opt"}, None)
            == RiskTier.R2
        )

        ext = _make_descriptor("web.extract", ToolFamily.URL, RiskTier.R1)
        assert Orchestrator._effective_tool_risk(ext, {"ref": "e1"}, None) == RiskTier.R1

        wait = _make_descriptor("web.wait_for", ToolFamily.URL, RiskTier.R1)
        assert Orchestrator._effective_tool_risk(wait, {"selector": "#test"}, None) == RiskTier.R1

    def test_web_click_elevates_to_r3_for_consequential_targets(self) -> None:
        click_desc = _make_descriptor("web.click", ToolFamily.URL, RiskTier.R2)

        # Benign click remains R2 (including false-positive prevention cases)
        for benign_arg in [
            {"ref": "link_home", "targetText": "Trang chủ"},
            {"ref": "theme_button", "targetText": "Dark Theme"},
            {"ref": "guide_link", "targetText": "view the guide"},
            {"ref": "weather_widget", "targetText": "Today weather"},
            {"ref": "other_options", "targetText": "other options"},
            {"ref": "spin", "targetText": "spin the wheel"},
            {"ref": "opinion", "targetText": "share your opinion"},
            {"ref": "cardigan", "targetText": "wool cardigan"},
            {"ref": "wildcard", "targetText": "wildcard search"},
        ]:
            assert (
                Orchestrator._effective_tool_risk(click_desc, benign_arg, None)
                == RiskTier.R2
            )
            assert (
                Orchestrator._declared_risk_floor(click_desc, benign_arg)
                == RiskTier.R2
            )

        # Consequential English keywords -> R3
        for consequential_arg in [
            {"ref": "btn-checkout"},
            {"ref": "e1", "selector": "#btn-pay-now"},
            {"ref": "e2", "targetText": "Delete Account"},
            {"ref": "e3", "elementName": "Submit Order"},
            {"ref": "payNow"},
            {"ref": "buyNow"},
            {"ref": "submitOrder"},
            {"ref": "deleteItem"},
            {"ref": "wire_transfer", "targetText": "wire money"},
            # Explicit Challenger 2 cases: checkOut, checkout, check_out, DELETE_ITEM, placeOrder
            {"ref": "checkOut"},
            {"ref": "checkout"},
            {"ref": "check_out"},
            {"ref": "DELETE_ITEM"},
            {"ref": "placeOrder"},
        ]:
            assert (
                Orchestrator._effective_tool_risk(click_desc, consequential_arg, None)
                == RiskTier.R3
            )
            assert (
                Orchestrator._declared_risk_floor(click_desc, consequential_arg)
                == RiskTier.R3
            )

        # Consequential Vietnamese keywords -> R3
        for consequential_arg in [
            {"ref": "e4", "targetText": "Xác nhận thanh toán"},
            {"ref": "e5", "targetText": "Xóa tài khoản"},
            {"ref": "btn_chuyen_khoan"},
            {"ref": "e6", "text": "Đặt hàng ngay"},
            {"ref": "btn_huy", "targetText": "hủy"},
            {"ref": "btn_huy_bo", "targetText": "hủy bỏ"},
            {"ref": "btn_huy_tk", "targetText": "hủy tài khoản"},
            {"ref": "btn_nop", "targetText": "nộp bài"},
            {"ref": "btn_go_bo", "targetText": "gỡ bỏ tiện ích"},
        ]:
            assert (
                Orchestrator._effective_tool_risk(click_desc, consequential_arg, None)
                == RiskTier.R3
            )
            assert (
                Orchestrator._declared_risk_floor(click_desc, consequential_arg)
                == RiskTier.R3
            )

    def test_web_fill_elevates_to_r3_for_sensitive_fields(self) -> None:
        fill_desc = _make_descriptor("web.fill", ToolFamily.URL, RiskTier.R2)

        # Benign fill remains R2
        benign_args = {"ref": "search_box", "value": "laptop", "targetText": "Tìm kiếm"}
        assert Orchestrator._effective_tool_risk(fill_desc, benign_args, None) == RiskTier.R2
        assert Orchestrator._declared_risk_floor(fill_desc, benign_args) == RiskTier.R2

        # Sensitive password / card / pin fields -> R3
        for sensitive_arg in [
            {"ref": "input_password", "value": "secret"},
            {"ref": "card_number", "value": "4111222233334444"},
            {"ref": "cvv_box", "value": "123", "selector": "input#cvv"},
            {"ref": "e1", "value": "pass", "targetText": "Mật khẩu đăng nhập"},
            # Sensitive values in value elevate to R3
            {"ref": "input_box", "value": "payNow"},
            {"ref": "input_box", "value": "secret password"},
            {"ref": "input_box", "value": "DELETE_ITEM"},
            {"ref": "input_box", "value": "checkOut"},
            {"ref": "input_box", "value": "placeOrder"},
        ]:
            assert (
                Orchestrator._effective_tool_risk(fill_desc, sensitive_arg, None)
                == RiskTier.R3
            )
            assert (
                Orchestrator._declared_risk_floor(fill_desc, sensitive_arg)
                == RiskTier.R3
            )


class TestWaveBCompactAria:
    def test_encode_tool_payload_bounds_compact_aria_tree_to_6144_bytes(self) -> None:
        # Generate large ARIA tree
        large_aria = "# ARIA Tree — Large Page\n" + (
            "[e{i}] button 'Action {i}'\n" * 400
        ).format(i=1)
        raw_output = {
            "snapshotId": "snap-1",
            "url": "https://example.com",
            "pageTitle": "Large Page",
            "compactAriaTree": large_aria,
            "nodes": [{"id": i} for i in range(100)],
        }
        encoded = encode_tool_payload("ok", raw_output)
        encoded_bytes = encoded.encode("utf-8")
        assert len(encoded_bytes) <= MAX_TOOL_OUTPUT_BYTES
        assert len(encoded_bytes) <= 6144

        decoded = json.loads(encoded)
        assert decoded["status"] == "ok"
        assert "compactAriaTree" in decoded["output"]


class TestWaveBActionSummary:
    def test_format_pending_bilingual_web_tools(self) -> None:
        target = "PC Studio"

        # web.navigate
        nav_vi = ActionSummary.format_pending(
            "web.navigate", {"url": "https://example.com"}, target, Locale.VI
        )
        assert "Điều hướng trình duyệt tới 'https://example.com'" in nav_vi
        nav_en = ActionSummary.format_pending(
            "web.navigate", {"url": "https://example.com"}, target, Locale.EN
        )
        assert "Navigate browser to 'https://example.com'" in nav_en

        # web.snapshot
        snap_vi = ActionSummary.format_pending("web.snapshot", {}, target, Locale.VI)
        assert "Chụp cấu trúc trang web" in snap_vi
        snap_en = ActionSummary.format_pending("web.snapshot", {}, target, Locale.EN)
        assert "Capture web page snapshot" in snap_en

        # web.click
        click_vi = ActionSummary.format_pending(
            "web.click", {"ref": "btn-pay"}, target, Locale.VI
        )
        assert "Bấm phần tử web 'btn-pay'" in click_vi
        click_en = ActionSummary.format_pending(
            "web.click", {"ref": "btn-pay"}, target, Locale.EN
        )
        assert "Click web element 'btn-pay'" in click_en

        # web.fill
        fill_vi = ActionSummary.format_pending(
            "web.fill", {"ref": "txt-name", "value": "Thao"}, target, Locale.VI
        )
        assert "Nhập dữ liệu vào phần tử web 'txt-name'" in fill_vi
        fill_en = ActionSummary.format_pending(
            "web.fill", {"ref": "txt-name", "value": "Thao"}, target, Locale.EN
        )
        assert "Fill web element 'txt-name'" in fill_en

        # web.select
        sel_vi = ActionSummary.format_pending(
            "web.select", {"ref": "sel-opt", "value": "1"}, target, Locale.VI
        )
        assert "Chọn mục trong menu web 'sel-opt'" in sel_vi
        sel_en = ActionSummary.format_pending(
            "web.select", {"ref": "sel-opt", "value": "1"}, target, Locale.EN
        )
        assert "Select option in web element 'sel-opt'" in sel_en

        # web.extract
        ext_vi = ActionSummary.format_pending(
            "web.extract", {"ref": "heading"}, target, Locale.VI
        )
        assert "Trích xuất dữ liệu từ trang web" in ext_vi
        ext_en = ActionSummary.format_pending(
            "web.extract", {"ref": "heading"}, target, Locale.EN
        )
        assert "Extract web data" in ext_en

        # web.wait_for
        wait_vi = ActionSummary.format_pending(
            "web.wait_for", {"selector": "#banner"}, target, Locale.VI
        )
        assert "Chờ phần tử web '#banner'" in wait_vi
        wait_en = ActionSummary.format_pending(
            "web.wait_for", {"selector": "#banner"}, target, Locale.EN
        )
        assert "Wait for web element '#banner'" in wait_en

    def test_format_result_bilingual(self) -> None:
        target = "PC Studio"
        ok_res = ActionSummary.format_result("web.click", {"ref": "e1"}, target, "ok", Locale.VI)
        assert "✓ Đã thực hiện" in ok_res
        assert "Bấm phần tử web 'e1'" in ok_res

        fail_res = ActionSummary.format_result(
            "web.navigate", {"url": "https://example.com"}, target, "failed", Locale.EN
        )
        assert "✗ " in fail_res
        assert "Navigate browser to 'https://example.com'" in fail_res


class TestWaveBH5Compliance:
    def test_no_hardcoded_browser_names_in_generic_layer(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "src" / "xuna_backend" / "modules"
        # In generic prompt & catalog layers, no hardcoded browser names (chrome, safari) or zalo
        for filename in ("prompts.py", "catalog.py"):
            target_path = root / "actions" / "application" / filename
            text = target_path.read_text(encoding="utf-8")
            lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
            code = "\n".join(lines).lower()
            for forbidden in ("chrome", "safari", "zalo", "spotify"):
                assert forbidden not in code, (
                    f"Forbidden hardcoded app {forbidden} found in {filename}"
                )
