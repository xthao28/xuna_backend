"""Comprehensive Unit & Integration Test Suite for Wave C Backend Capabilities:

1. Catalog & Prompts Exposure (`ui.observe_hybrid`, `input.drag`, `input.click`
   xNorm/yNorm, UI Settle telemetry).
2. Ephemeral Multimodal Vision Pipeline & Payload Decoupling (<= 6144 bytes JSON,
   ephemeral Message.image_base64, vision content block serialization,
   thought_signature preservation, Zero-Retention).
3. R3 Safety Gate & Coordinate Anti-Circumvention (bounds check, reverse hit-testing,
   password denial, consequential button R3, uncommitted draft text protection,
   input.drag preflight).
4. Rule H5 App-Agnostic Compliance.
"""

import json
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from xuna_backend.modules.actions.application.catalog import (
    _PRIORITY_ORDER,
    EligibleHostTool,
    _host_tool_spec,
)
from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    hit_test_node_at_coordinates,
)
from xuna_backend.modules.actions.application.orchestrator import (
    _COMMIT_INPUT_TOOLS,
    _PREFLIGHT_TOOLS,
    Orchestrator,
)
from xuna_backend.modules.actions.application.prompts import (
    _UI_AUTOMATION_TOOLS,
    encode_tool_payload,
    tool_system_prompt,
    truncate_tool_value,
)
from xuna_backend.modules.actions.domain.entities import MAX_TOOL_OUTPUT_BYTES
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Locale,
    Message,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
)
from xuna_backend.modules.cognition.infrastructure.openai_compatible import (
    _GLOBAL_SIGNATURE_CACHE,
    _serialize_message,
)
from xuna_backend.modules.conversation.application.chat_turn import chat_turn
from xuna_backend.modules.hosts.domain.entities import (
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)

# -----------------------------------------------------------------------------
# 1. Catalog & Prompts Exposure Tests
# -----------------------------------------------------------------------------


class TestCatalogAndPromptsExposure:
    def test_priority_order_includes_wave_c_tools(self) -> None:
        assert "ui.observe_hybrid" in _PRIORITY_ORDER
        assert "input.drag" in _PRIORITY_ORDER

        # Check ordering invariants
        inspect_idx = _PRIORITY_ORDER.index("ui.inspect")
        find_idx = _PRIORITY_ORDER.index("ui.find")
        observe_hybrid_idx = _PRIORITY_ORDER.index("ui.observe_hybrid")
        click_idx = _PRIORITY_ORDER.index("input.click")
        drag_idx = _PRIORITY_ORDER.index("input.drag")

        # ui.find must stay directly after ui.inspect (test_udc17_ui_find contract)
        assert find_idx == inspect_idx + 1
        # ui.observe_hybrid is directly in the primary inspection group
        assert observe_hybrid_idx == find_idx + 1
        # input.drag is immediately grouped with input.click
        assert drag_idx == click_idx + 1

    def test_input_click_schema_has_normalized_coordinates(self) -> None:
        descriptor = ToolDescriptor(
            name="input.click",
            family=ToolFamily.INPUT,
            risk=RiskTier.R2,
            description="Click UI element or normalized coordinate fallback.",
            input_schema={
                "type": "object",
                "properties": {
                    "elementId": {"type": "string"},
                    "elementName": {"type": "string"},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        )
        item = EligibleHostTool(descriptor=descriptor, aliases=("desktop-test",))
        spec = _host_tool_spec("input.click", item)

        assert spec.name == "input__click"
        props = spec.parameters.get("properties", {})
        assert "xNorm" in props
        assert props["xNorm"]["type"] == "number"
        assert props["xNorm"]["minimum"] == 0.0
        assert props["xNorm"]["maximum"] == 1.0

        assert "yNorm" in props
        assert props["yNorm"]["type"] == "number"
        assert props["yNorm"]["minimum"] == 0.0
        assert props["yNorm"]["maximum"] == 1.0

    def test_ui_automation_tools_includes_wave_c_tools(self) -> None:
        assert "ui__observe_hybrid" in _UI_AUTOMATION_TOOLS
        assert "input__drag" in _UI_AUTOMATION_TOOLS

    def test_tool_system_prompt_triggers_on_wave_c_tools(self) -> None:
        # Vietnamese prompt
        prompt_vi = tool_system_prompt(Locale.VI, ["ui__observe_hybrid"])
        assert "ui__observe_hybrid" in prompt_vi
        assert "xNorm, yNorm" in prompt_vi

        # English prompt
        prompt_en = tool_system_prompt(Locale.EN, ["input__drag"])
        assert "input__drag" in prompt_en
        assert "xNorm, yNorm" in prompt_en

    def test_truncate_tool_value_preserves_settled_and_settlems(self) -> None:
        # Create a dictionary with more than 16 keys to trigger truncation
        data: dict[str, object] = {f"k_{i}": f"val_{i}" for i in range(25)}
        data["settled"] = True
        data["settleMs"] = 120

        truncated = truncate_tool_value(data)
        assert isinstance(truncated, dict)
        assert truncated.get("settled") is True
        assert truncated.get("settleMs") == 120

    def test_encode_tool_payload_preserves_settled_and_settlems_on_slim_and_fallback(self) -> None:
        # Slim out branch (large tree output)
        large_tree_output: dict[str, object] = {
            "compactUiTree": "X" * 5000,
            "settled": True,
            "settleMs": 350,
        }
        encoded_slim = encode_tool_payload("ok", large_tree_output)
        assert len(encoded_slim.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES
        parsed_slim = json.loads(encoded_slim)
        assert parsed_slim["output"]["settled"] is True
        assert parsed_slim["output"]["settleMs"] == 350

        # Minimal fallback branch (unshrinkable huge content)
        huge_output: dict[str, object] = {
            "settled": False,
            "settleMs": 1500,
        }
        extra_bloat: dict[str, object] = {f"pad_{i}": "W" * 300 for i in range(25)}
        encoded_minimal = encode_tool_payload("ok", huge_output, **extra_bloat)
        assert len(encoded_minimal.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES
        parsed_minimal = json.loads(encoded_minimal)
        assert parsed_minimal["output"]["truncated"] is True
        assert parsed_minimal["output"]["settled"] is False
        assert parsed_minimal["output"]["settleMs"] == 1500


# -----------------------------------------------------------------------------
# 2. Ephemeral Multimodal Vision Pipeline Tests
# -----------------------------------------------------------------------------


class TestEphemeralMultimodalVisionPipeline:
    def test_message_ephemeral_image_fields(self) -> None:
        msg = Message(
            role=Role.TOOL,
            content='{"status":"ok"}',
            tool_call_id="call-1",
            image_base64="aW1hZ2VkYXRh",
            image_mime_type="image/jpeg",
        )
        assert msg.image_base64 == "aW1hZ2VkYXRh"
        assert msg.image_mime_type == "image/jpeg"

    def test_serialize_message_with_multimodal_content_blocks(self) -> None:
        msg = Message(
            role=Role.TOOL,
            content='{"status":"ok","windowTitle":"Canvas"}',
            tool_call_id="call-123",
            image_base64="aW1hZ2VkYXRh",
            image_mime_type="image/jpeg",
        )
        serialized = _serialize_message(msg)
        assert serialized["role"] == "tool"
        assert serialized["tool_call_id"] == "call-123"
        content = serialized["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": '{"status":"ok","windowTitle":"Canvas"}'}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,aW1hZ2VkYXRh"},
        }

    def test_payload_decoupling_enforces_max_tool_output_bytes(self) -> None:
        # Simulate 120 KB base64 screenshot data returned by host
        fake_b64 = "A" * 120_000
        structured_content: dict[str, object] = {
            "appId": "com.custom.app",
            "windowTitle": "Visual Editor",
            "snapshotId": "snap-99",
            "nodes": [
                {
                    "id": "e1",
                    "role": "button",
                    "name": "Apply",
                    "bounds": [0.1, 0.2, 0.3, 0.4],
                }
            ],
            "screenshot": {
                "mimeType": "image/jpeg",
                "width": 1280,
                "height": 720,
                "dataBase64": fake_b64,
            },
        }

        # Emulate orchestrator decoupling logic
        sc_obj = structured_content.get("screenshot")
        detached_image_b64: str | None = None
        detached_image_mime = "image/jpeg"
        if isinstance(sc_obj, dict):
            detached_image_b64 = sc_obj.pop("dataBase64", None)  # type: ignore[assignment]
            if isinstance(sc_obj.get("mimeType"), str):
                detached_image_mime = str(sc_obj["mimeType"])

        # Check payload size without the image
        encoded_json = encode_tool_payload("ok", structured_content)
        assert len(encoded_json.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES
        assert "Apply" in encoded_json
        assert fake_b64 not in encoded_json

        # Attached ephemerally
        msg = Message(
            role=Role.TOOL,
            content=encoded_json,
            tool_call_id="call-v1",
            image_base64=detached_image_b64,
            image_mime_type=detached_image_mime,
        )
        assert msg.image_base64 == fake_b64
        assert msg.image_mime_type == "image/jpeg"

    @pytest.mark.anyio
    async def test_chat_turn_delivers_ephemeral_image_to_generator_in_next_round(self) -> None:
        fake_jpeg_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )

        class MockVisionGenerator:
            def __init__(self) -> None:
                self.calls: list[ChatRequest] = []

            async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
                self.calls.append(request)
                if len(self.calls) == 1:
                    # Round 1: requests tool ui__observe_hybrid
                    yield ToolCallRequested(
                        call=ToolCall(
                            id="call-observe-1",
                            name="ui__observe_hybrid",
                            arguments={"scope": "active_window"},
                            extra={"thought_signature": "sig-vision-001"},
                        )
                    )
                else:
                    # Round 2: receives image in messages and answers
                    yield TextDelta("Observed window visually.")
                    yield ChatFinished(
                        provider="mock",
                        model="vision",
                        input_tokens=10,
                        output_tokens=10,
                        latency_ms=50,
                    )

        generator = MockVisionGenerator()

        async def mock_tool_runner(call: ToolCall, emit: object) -> Message:
            assert call.name == "ui__observe_hybrid"
            # Return Message with detached image_base64
            return Message(
                role=Role.TOOL,
                content='{"status":"ok","windowTitle":"Custom Canvas"}',
                tool_call_id=call.id,
                image_base64=fake_jpeg_b64,
                image_mime_type="image/jpeg",
            )

        initial_request = ChatRequest(
            messages=(Message(role=Role.USER, content="Look at the screen"),),
            tools=(
                ToolSpec(
                    name="ui__observe_hybrid",
                    description="observe",
                    parameters={"type": "object"},
                ),
            ),
        )

        events: list[ChatEvent] = []
        async for ev in chat_turn(generator, initial_request, tool_runner=mock_tool_runner):
            events.append(ev)

        assert any(isinstance(e, TextDelta) and "Observed" in e.text for e in events)
        assert len(generator.calls) == 2

        # Verify Round 2 request received the in-memory image
        round2_request = generator.calls[1]
        tool_msg = next((m for m in round2_request.messages if m.role == Role.TOOL), None)
        assert tool_msg is not None
        assert tool_msg.image_base64 == fake_jpeg_b64
        assert tool_msg.image_mime_type == "image/jpeg"

    def test_thought_signature_preservation(self) -> None:
        call_id = "call-signature-wave-c"
        signature_data = {"thought_signature": "wave-c-thought-sig-xyz"}
        _GLOBAL_SIGNATURE_CACHE[call_id] = signature_data

        tc = ToolCall(
            id=call_id,
            name="ui__observe_hybrid",
            arguments={"scope": "active_window"},
            extra=signature_data,
        )
        msg = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=(tc,),
        )
        serialized = _serialize_message(msg)
        assert "tool_calls" in serialized
        assert serialized["tool_calls"][0]["extra_content"] == signature_data


# -----------------------------------------------------------------------------
# 3. R3 Safety Gate & Coordinate Anti-Circumvention Tests
# -----------------------------------------------------------------------------


def _make_descriptor(
    name: str,
    risk: RiskTier = RiskTier.R2,
    family: ToolFamily = ToolFamily.INPUT,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        family=family,
        risk=risk,
        description=name,
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )


class TestR3SafetyGateAndCoordinateAntiCircumvention:
    @pytest.fixture
    def active_cu_state(self) -> ComputerUseSessionState:
        now = datetime.now(UTC)
        return ComputerUseSessionState(
            session_id="cu-test-session-1",
            owner_id=uuid.uuid4(),
            host_id=uuid.uuid4(),
            goal="test coordinate anti-circumvention",
            allowed_apps=("com.test.app",),
            allowed_domains=(),
            max_steps=20,
            max_minutes=10,
            started_at_utc=now,
            last_heartbeat_at=now,
        )

    def test_reverse_hit_test_helper_finds_deepest_leaf_node(self) -> None:
        parent_container = {
            "id": "container-1",
            "role": "group",
            "name": "Form Container",
            "bounds": [0.1, 0.1, 0.9, 0.9],
        }
        child_button = {
            "id": "btn-submit",
            "role": "button",
            "name": "Submit Form",
            "bounds": [0.7, 0.8, 0.85, 0.88],
        }
        other_node = {
            "id": "txt-input",
            "role": "textbox",
            "name": "User Note",
            "bounds": [0.2, 0.2, 0.5, 0.3],
        }
        nodes = [parent_container, child_button, other_node]

        # Coordinate lands inside child_button
        hit = hit_test_node_at_coordinates(nodes, 0.75, 0.84)
        assert hit is not None
        assert hit["id"] == "btn-submit"

        # Coordinate lands inside parent_container but outside children
        hit_parent = hit_test_node_at_coordinates(nodes, 0.15, 0.5)
        assert hit_parent is not None
        assert hit_parent["id"] == "container-1"

        # Coordinate outside any node
        hit_none = hit_test_node_at_coordinates(nodes, 0.05, 0.05)
        assert hit_none is None

    def test_coordinate_click_bounds_check_fails_closed(self) -> None:
        descriptor = _make_descriptor("input.click")

        # Coordinate > 1.0 escalates to R3 in risk evaluator
        risk_high = Orchestrator._effective_tool_risk(
            descriptor,
            {"xNorm": 1.05, "yNorm": 0.5},
            active_cu=None,
        )
        assert risk_high == RiskTier.R3

        # Coordinate < 0.0 escalates to R3 in risk evaluator
        risk_low = Orchestrator._effective_tool_risk(
            descriptor,
            {"xNorm": 0.5, "yNorm": -0.01},
            active_cu=None,
        )
        assert risk_low == RiskTier.R3

    def test_coordinate_drag_bounds_check_fails_closed(self) -> None:
        descriptor = _make_descriptor("input.drag")

        risk_out = Orchestrator._effective_tool_risk(
            descriptor,
            {"fromXNorm": 0.1, "fromYNorm": 0.2, "toXNorm": 1.2, "toYNorm": 0.5},
            active_cu=None,
        )
        assert risk_out == RiskTier.R3

    def test_coordinate_click_reverse_hit_test_password_node_escalates_to_r3(
        self, active_cu_state: ComputerUseSessionState
    ) -> None:
        password_node = {
            "id": "pwd-field",
            "role": "passwordfield",
            "name": "Password Input",
            "isPassword": True,
            "bounds": [0.3, 0.4, 0.6, 0.5],
        }
        active_cu_state.cached_nodes["pwd-field"] = password_node

        descriptor = _make_descriptor("input.click")
        # Click lands right on password field
        risk = Orchestrator._effective_tool_risk(
            descriptor,
            {"xNorm": 0.45, "yNorm": 0.45},
            active_cu=active_cu_state,
        )
        assert risk == RiskTier.R3

    def test_coordinate_click_reverse_hit_test_consequential_escalates_to_r3(
        self, active_cu_state: ComputerUseSessionState
    ) -> None:
        pay_node = {
            "id": "btn-pay",
            "role": "button",
            "name": "Thanh toán",
            "bounds": [0.6, 0.7, 0.85, 0.8],
        }
        active_cu_state.cached_nodes["btn-pay"] = pay_node

        descriptor = _make_descriptor("input.click")
        risk = Orchestrator._effective_tool_risk(
            descriptor,
            {"xNorm": 0.7, "yNorm": 0.75},
            active_cu=active_cu_state,
        )
        assert risk == RiskTier.R3

    def test_coordinate_click_uncommitted_draft_protection_escalates_to_r3(
        self, active_cu_state: ComputerUseSessionState
    ) -> None:
        # Simulate active session with uncommitted draft text
        active_cu_state.typed_count = 2
        active_cu_state.typed_since_commit.extend(["Hello", "World"])

        # Target node is a generic button (not editable text)
        button_node = {
            "id": "btn-custom-canvas",
            "role": "button",
            "name": "Action Option",
            "bounds": [0.5, 0.5, 0.7, 0.6],
        }
        active_cu_state.cached_nodes["btn-custom-canvas"] = button_node

        descriptor = _make_descriptor("input.click")
        # Coordinate click on non-editable field while draft is pending MUST escalate to R3!
        risk = Orchestrator._effective_tool_risk(
            descriptor,
            {"xNorm": 0.6, "yNorm": 0.55},
            active_cu=active_cu_state,
        )
        assert risk == RiskTier.R3

    def test_coordinate_click_uncommitted_draft_allows_r2_on_verified_editable_text_field(
        self, active_cu_state: ComputerUseSessionState
    ) -> None:
        # Active session with uncommitted draft text
        active_cu_state.typed_count = 1
        active_cu_state.typed_since_commit.append("Draft paragraph")

        # Target node is a verified editable text input
        text_field = {
            "id": "txt-edit",
            "role": "textbox",
            "name": "Content Editor",
            "editable": True,
            "readOnly": False,
            "bounds": [0.1, 0.1, 0.5, 0.3],
        }
        active_cu_state.cached_nodes["txt-edit"] = text_field

        descriptor = _make_descriptor("input.click")
        # Coordinate click on safe editable text area remains R2
        risk = Orchestrator._effective_tool_risk(
            descriptor,
            {"xNorm": 0.25, "yNorm": 0.2},
            active_cu=active_cu_state,
        )
        assert risk == RiskTier.R2

    def test_coordinate_click_unclassified_target_escalates_to_r3(
        self, active_cu_state: ComputerUseSessionState
    ) -> None:
        # No cached nodes match coordinates
        active_cu_state.cached_nodes.clear()

        descriptor = _make_descriptor("input.click")
        risk = Orchestrator._effective_tool_risk(
            descriptor,
            {"xNorm": 0.3, "yNorm": 0.4},
            active_cu=active_cu_state,
        )
        assert risk == RiskTier.R3

    def test_drag_anti_circumvention_escalates_to_r3(
        self, active_cu_state: ComputerUseSessionState
    ) -> None:
        descriptor = _make_descriptor("input.drag")

        # 1. Dragging with uncommitted draft text escalates to R3
        active_cu_state.typed_count = 1
        risk_draft = Orchestrator._effective_tool_risk(
            descriptor,
            {"fromXNorm": 0.1, "fromYNorm": 0.1, "toXNorm": 0.5, "toYNorm": 0.5},
            active_cu=active_cu_state,
        )
        assert risk_draft == RiskTier.R3
        active_cu_state.typed_count = 0

        # 2. Dragging to consequential target escalates to R3
        pay_btn = {
            "id": "btn-pay-drop",
            "role": "button",
            "name": "Confirm Payment",
            "bounds": [0.6, 0.6, 0.8, 0.8],
        }
        safe_item = {
            "id": "item-card",
            "role": "group",
            "name": "Item Card",
            "bounds": [0.1, 0.1, 0.3, 0.3],
        }
        active_cu_state.cached_nodes["btn-pay-drop"] = pay_btn
        active_cu_state.cached_nodes["item-card"] = safe_item

        risk_consequential = Orchestrator._effective_tool_risk(
            descriptor,
            {"fromXNorm": 0.2, "fromYNorm": 0.2, "toXNorm": 0.7, "toYNorm": 0.7},
            active_cu=active_cu_state,
        )
        assert risk_consequential == RiskTier.R3

        # 3. Dragging with unclassified destination escalates to R3
        risk_unclassified = Orchestrator._effective_tool_risk(
            descriptor,
            {"fromXNorm": 0.2, "fromYNorm": 0.2, "toXNorm": 0.95, "toYNorm": 0.95},
            active_cu=active_cu_state,
        )
        assert risk_unclassified == RiskTier.R3

    def test_input_drag_registered_in_preflight_and_commit_tools(self) -> None:
        assert "input.drag" in _PREFLIGHT_TOOLS
        assert "input.drag" in _COMMIT_INPUT_TOOLS


# -----------------------------------------------------------------------------
# 4. Rule H5 App-Agnostic Generic Layer Scanner Test
# -----------------------------------------------------------------------------


class TestRuleH5GenericLayerCompliance:
    def test_rule_h5_zero_hardcoded_app_names_or_paths_in_generic_layers(self) -> None:
        app_pattern = re.compile(
            r"\b(spotify|chrome|safari|zalo|figma|slack|notion|telegram|vscode|obsidian)\b",
            re.IGNORECASE,
        )
        host_pattern = re.compile(
            r"\b(macbook|mac\s*studio|pc\s*nha|living_room)\b",
            re.IGNORECASE,
        )
        os_path_pattern = re.compile(r"(/Users/|/home/|C:\\\\|/Applications/|/Program Files)")
        backend_root = Path(__file__).resolve().parents[2]
        generic_dirs: list[tuple[Path, frozenset[str]]] = [
            (
                backend_root / "src/xuna_backend/modules/actions/application",
                frozenset({"workflows.py"}),
            ),
            (
                backend_root / "src/xuna_backend/modules/actions/domain",
                frozenset({"tool_names.py"}),
            ),
            (backend_root / "src/xuna_backend/modules/hosts/application", frozenset()),
            (backend_root / "src/xuna_backend/modules/conversation/application", frozenset()),
        ]
        files_to_scan: list[Path] = []
        for directory, exempt in generic_dirs:
            if directory.exists():
                for p in sorted(directory.rglob("*.py")):
                    if p.name not in exempt:
                        files_to_scan.append(p)

        scanned_names = {p.name for p in files_to_scan}
        for required in (
            "mesh_scoring.py",
            "habit_learner.py",
            "mesh_router.py",
            "mesh_prober.py",
            "mesh_call_helper.py",
            "mesh_turn_helper.py",
            "habits.py",
            "tool_discovery.py",
            "catalog.py",
            "prompts.py",
            "orchestrator.py",
            "cross_host_pipelines.py",
            "redaction.py",
            "battery_advisor.py",
            "mission_envelope.py",
            "wake_queue.py",
            "watch.py",
            "watch_coordinator.py",
            "wake_inbox.py",
            "yield_detach.py",
            "planner_executor.py",
        ):
            assert required in scanned_names, f"Expected {required} in H5 generic scan"

        violations: list[str] = []
        for file_path in files_to_scan:
            with file_path.open(encoding="utf-8") as fp:
                for line_num, line in enumerate(fp, 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    for label, pat in (
                        ("app_name", app_pattern),
                        ("host_literal", host_pattern),
                        ("os_path", os_path_pattern),
                    ):
                        match = pat.search(line)
                        if match:
                            violations.append(
                                f"{file_path}:{line_num} [{label}]: matched '{match.group(0)}' "
                                f"in '{stripped}'"
                            )

        assert not violations, "Rule H5 violations found:\n" + "\n".join(violations)

    def test_rule_h5_watch_routes_zero_hardcoded_apps_or_paths(self) -> None:
        app_pattern = re.compile(
            r"\b(spotify|chrome|safari|zalo|figma|slack|notion|telegram|vscode|obsidian)\b",
            re.IGNORECASE,
        )
        host_pattern = re.compile(
            r"\b(macbook|mac\s*studio|pc\s*nha|living_room)\b",
            re.IGNORECASE,
        )
        os_path_pattern = re.compile(r"(/Users/|/home/|C:\\\\|/Applications/|/Program Files)")
        backend_root = Path(__file__).resolve().parents[2]
        watch_routes = backend_root / "src/xuna_backend/modules/actions/api/watch_routes.py"
        assert watch_routes.exists()
        for line_num, line in enumerate(watch_routes.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            for pat in (app_pattern, host_pattern, os_path_pattern):
                assert not pat.search(line), f"{watch_routes}:{line_num}: H5 violation in {line!r}"

    def test_rule_h5_zero_hardcoded_app_names_in_generic_layers(self) -> None:
        self.test_rule_h5_zero_hardcoded_app_names_or_paths_in_generic_layers()

    def test_rule_h5_no_keyword_intent_tables_in_mesh_modules(self) -> None:
        import ast

        keyword_pat = re.compile(
            r"\b(music|nhac|browser|trinh duyet|editor|terminal|spotify|chrome|macbook)\b",
            re.IGNORECASE,
        )
        dotted_tool_pat = re.compile(r"^[a-z]+\.[a-z_]+$")
        backend_root = Path(__file__).resolve().parents[2]
        mesh_files = [
            backend_root / "src/xuna_backend/modules/actions/domain/mesh_scoring.py",
            backend_root / "src/xuna_backend/modules/actions/domain/habit_learner.py",
            backend_root / "src/xuna_backend/modules/actions/domain/redaction.py",
            backend_root / "src/xuna_backend/modules/actions/domain/battery_advisor.py",
            backend_root / "src/xuna_backend/modules/actions/domain/mission_envelope.py",
            backend_root / "src/xuna_backend/modules/actions/domain/wake_queue.py",
            backend_root / "src/xuna_backend/modules/actions/application/mesh_router.py",
            backend_root / "src/xuna_backend/modules/actions/application/mesh_prober.py",
            backend_root / "src/xuna_backend/modules/actions/application/cross_host_pipelines.py",
        ]
        violations: list[str] = []
        for file_path in mesh_files:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    for k, v in zip(node.keys, node.values, strict=False):
                        if (
                            isinstance(k, ast.Constant)
                            and isinstance(k.value, str)
                            and k.value != "tool_name"
                            and isinstance(v, ast.Constant)
                            and isinstance(v.value, str)
                            and dotted_tool_pat.fullmatch(v.value)
                        ):
                            violations.append(
                                f"{file_path}:{node.lineno}: "
                                f"intent dict key={k.value!r} -> {v.value!r}"
                            )
                elif (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and not dotted_tool_pat.fullmatch(node.value)
                    and not node.value.startswith("Job is terminal")
                    and keyword_pat.search(node.value)
                ):
                    if len(node.value) > 80 and "\n" in node.value:
                        continue
                    violations.append(f"{file_path}:{node.lineno}: keyword match {node.value!r}")
        assert not violations, "Rule H5 AST intent-table violations:\n" + "\n".join(violations)

    def test_rule_h5_watch_cross_repo_and_repository_compliance(self) -> None:
        import ast

        app_pattern = re.compile(
            r"\b(spotify|chrome|safari|zalo|figma|slack|notion|telegram|vscode|obsidian)\b",
            re.IGNORECASE,
        )
        agent_pattern = re.compile(
            r"\b(claude|codex|cursor|antigravity|copilot|windsurf|gemini)\b",
            re.IGNORECASE,
        )
        host_pattern = re.compile(
            r"\b(macbook|mac\s*studio|pc\s*nha|living_room)\b",
            re.IGNORECASE,
        )
        os_path_pattern = re.compile(
            r"(/Users/|/home/|(?<![A-Za-z])[A-Za-z]:(?:\\{1,2}|/(?!/))"
            r"|\\\\[A-Za-z0-9_.-]+\\|/Applications/|/Program Files)"
        )
        keyword_pat = re.compile(
            r"\b(music|nhac|nhạc|browser|trinh duyet|trình duyệt|editor|terminal|spotify"
            r"|chrome|macbook|xoa file|xóa file|xoa tep|xóa tệp|gui tin nhan|gửi tin nhắn"
            r"|gui email|gửi email|mo app|mở app|mo nhac|mở nhạc|mo trinh duyet|mở trình duyệt"
            r"|bat nhac|bật nhạc|tat may|tắt máy|chup man hinh|chụp màn hình)\b",
            re.IGNORECASE,
        )
        tool_value_pat = re.compile(r"^[a-z][a-zA-Z0-9_]*(?:\.|__)[a-zA-Z_][a-zA-Z0-9_]*$")
        dart_string_pat = re.compile(r"""(['"])((?:\\.|(?!\1).)*)\1""")
        dart_kv_tool_pat = re.compile(
            r"""['"][^'"]+['"]\s*:\s*['"]([a-z][a-zA-Z0-9_]*(?:\.|__)[a-zA-Z_][a-zA-Z0-9_]*)['"]"""
        )

        backend_root = Path(__file__).resolve().parents[2]
        repo_root = backend_root.parent
        watch_py_files = [
            backend_root / "src/xuna_backend/modules/actions/domain/watch.py",
            backend_root / "src/xuna_backend/modules/actions/application/watch_coordinator.py",
            backend_root / "src/xuna_backend/modules/actions/application/wake_inbox.py",
            backend_root / "src/xuna_backend/modules/actions/application/yield_detach.py",
            backend_root / "src/xuna_backend/modules/actions/application/planner_executor.py",
            backend_root / "src/xuna_backend/modules/actions/api/watch_routes.py",
            backend_root / "src/xuna_backend/modules/actions/infrastructure/watch_repository.py",
        ]
        watch_dart_files = [
            repo_root / "desktop/lib/features/watch/local_watch_sentinel.dart",
            repo_root / "desktop/lib/features/watch/watch_convergence.dart",
            repo_root / "desktop/lib/features/watch/native_watch_signal_driver.dart",
            repo_root / "desktop/lib/features/watch/cdp_watch_signal_driver.dart",
            repo_root / "desktop/lib/features/watch/protocol_task_registry.dart",
            repo_root / "desktop/lib/features/watch/protocol_async_watch_driver.dart",
            repo_root / "desktop/lib/features/watch/acp_harness_adapter.dart",
            repo_root
            / "mobile/lib/features/home/presentation/service/home_wake_inbox_service.dart",
        ]

        violations: list[str] = []
        for file_path in (*watch_py_files, *watch_dart_files):
            assert file_path.exists(), f"Expected Watch H5 target file missing: {file_path}"
            is_dart = file_path.suffix == ".dart"
            for line_num, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if (is_dart and stripped.startswith("//")) or (
                    not is_dart and stripped.startswith("#")
                ):
                    continue
                for label, pat in (
                    ("app_name", app_pattern),
                    ("agent_name", agent_pattern),
                    ("host_literal", host_pattern),
                    ("os_path", os_path_pattern),
                ):
                    if (
                        label == "agent_name"
                        and file_path.name == "native_watch_signal_driver.dart"
                        and stripped in {"'.claude',", "'.gemini',"}
                    ):
                        continue
                    match = pat.search(line)
                    if match:
                        violations.append(
                            f"{file_path}:{line_num} [{label}]: matched '{match.group(0)}' "
                            f"in '{stripped}'"
                        )

        for file_path in watch_py_files:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    for k, v in zip(node.keys, node.values, strict=False):
                        if (
                            isinstance(k, ast.Constant)
                            and isinstance(k.value, str)
                            and k.value != "tool_name"
                            and isinstance(v, ast.Constant)
                            and isinstance(v.value, str)
                            and tool_value_pat.fullmatch(v.value)
                        ):
                            violations.append(
                                f"{file_path}:{node.lineno}: "
                                f"intent dict key={k.value!r} -> {v.value!r}"
                            )
                elif (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and not tool_value_pat.fullmatch(node.value)
                    and not node.value.startswith("Job is terminal")
                    and keyword_pat.search(node.value)
                ):
                    if len(node.value) > 80 and "\n" in node.value:
                        continue
                    violations.append(f"{file_path}:{node.lineno}: keyword match {node.value!r}")

        for file_path in watch_dart_files:
            for line_num, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("import "):
                    continue
                if dart_kv_tool_pat.search(line):
                    violations.append(
                        f"{file_path}:{line_num} [dart_intent_map]: matched tool mapping in "
                        f"'{stripped}'"
                    )
                for _quote, literal in dart_string_pat.findall(line):
                    if keyword_pat.search(literal):
                        violations.append(
                            f"{file_path}:{line_num} [dart_keyword]: matched {literal!r}"
                        )

        mobile_service = watch_dart_files[-1]
        mobile_text = mobile_service.read_text(encoding="utf-8")
        assert not re.search(
            r"import\s+['\"](?:package:xuna_mobile/features/[^'\"]+/data/|(?:\.\./)+[^'\"]*/data/|(?:\.\./)+data/)",
            mobile_text,
        ), f"Clean Architecture violation: {mobile_service} imports data layer directly"

        assert not violations, "Rule H5 Watch cross-repo violations:\n" + "\n".join(violations)

    def test_udc19_universal_hints_menu_risk_and_summaries(self) -> None:
        from xuna_backend.modules.actions.application.prompts import state_driven_hint
        from xuna_backend.modules.actions.domain.action_summary import ActionSummary

        # 1. Changed action with disposition.retry="unsafe" must NOT emit stale_ref hint
        ok_changed = state_driven_hint(
            {"effect": "changed", "uiChanged": True, "disposition": {"retry": "unsafe"}}
        )
        assert ok_changed is None

        # 2. Action with effect="no_change" guides visual/gesture verification
        no_change_hint = state_driven_hint(
            {"effect": "no_change", "uiChanged": False, "disposition": {"retry": "unsafe"}}
        )
        assert no_change_hint is not None
        assert "ui__observe_hybrid" in no_change_hint
        assert "clickCount=2" in no_change_hint

        # 3. Missing or ambiguous home target produces clear hints
        not_found_hint = state_driven_hint({"errorCode": "target_not_found"})
        assert not_found_hint is not None
        assert "openTarget" in not_found_hint

        # 4. Orchestrator._declared_risk_floor elevates destructive menuPath to R3
        click_desc = ToolDescriptor(
            name="input.click",
            family=ToolFamily.INPUT,
            risk=RiskTier.R2,
            description="click",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        )
        assert (
            Orchestrator._declared_risk_floor(click_desc, {"menuPath": "File > Open Folder..."})
            == RiskTier.R2
        )
        assert (
            Orchestrator._declared_risk_floor(click_desc, {"menuPath": "File > Quit"})
            == RiskTier.R3
        )
        assert (
            Orchestrator._declared_risk_floor(click_desc, {"menuPath": "Message > Send"})
            == RiskTier.R3
        )

        # 5. ActionSummary formats openTarget, menuPath, double-click, and hover accurately
        assert "xuna_pet" in ActionSummary.format_pending(
            "app.launch", {"id": "editor", "openTarget": "xuna_pet"}, "Mac", Locale.VI
        )
        assert "File > Open" in ActionSummary.format_pending(
            "input.click", {"menuPath": "File > Open"}, "Mac", Locale.EN
        )
        assert "Nhấp đúp" in ActionSummary.format_pending(
            "input.click", {"elementName": "Row 1", "clickCount": 2}, "Mac", Locale.VI
        )
        assert "Hover" in ActionSummary.format_pending(
            "input.click", {"elementName": "Row 1", "hover": True}, "Mac", Locale.EN
        )

        # 6. Verbose desktop ui.inspect nodes are compacted so >= 75 controls fit in 6144 B
        verbose_nodes = [
            {
                "id": f"ui_{i}",
                "ref": f"@s00000001:e{i}",
                "role": "row" if i > 20 else "button",
                "name": f"Item {i} · Artist · 04:12",
                "enabled": True,
                "visible": True,
                "editable": False,
                "isPassword": False,
                "bounds": {"x": 240, "y": 40 + i * 24, "width": 600, "height": 24},
            }
            for i in range(1, 90)
        ]
        packed = json.loads(
            encode_tool_payload(
                "ok",
                {"appId": "player", "windowTitle": "Library", "nodes": verbose_nodes},
            )
        )
        kept = packed["output"]["nodes"]
        assert len(kept) >= 72
        assert kept[25]["ref"] == "@s00000001:e26"
        assert "bounds" not in kept[25]
        assert "enabled" not in kept[25]
