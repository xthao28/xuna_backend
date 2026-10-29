"""Unit and evaluation tests for turn mode routing and domain matching (KX-1 §6.2, §9)."""

import json
import unicodedata
from pathlib import Path

import pytest

from xuna_backend.modules.conversation.domain.turn_mode import (
    LONG_TEXT_CHARS,
    Channel,
    ModeDecision,
    ModeReason,
    RequestedMode,
    TurnMode,
    decide_mode,
    is_task_request,
    looks_like_code,
)
from xuna_backend.modules.knowledge.application.owner_context import TURN_DOMAIN_THRESHOLD
from xuna_backend.modules.knowledge.domain.lexical_matcher import compute_lexical_similarity

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "turn_domain_vi.json"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("chào bạn", False),
        ("dòng 1\ndòng 2\ndòng 3", False),
        ("dòng 1\ndòng 2\ndòng 3\ndòng 4", True),
        ("Xem code:\n```python\nprint(1)\n```", True),
    ],
)
def test_looks_like_code(text: str, expected: bool) -> None:
    assert looks_like_code(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hôm nay trời đẹp quá", False),
        ("Mình thích uống trà sữa", False),
        ("Mạch bị treo rồi?", True),
        ("Giải thích nguyên lý hoạt động của I2C", True),
        ("Làm sao để cấu hình timer?", True),
        ("Can you explain this bug", True),
        ("How to optimize SQL query", True),
        (unicodedata.normalize("NFD", "Phân tích báo cáo tài chính"), True),
    ],
)
def test_is_task_request(text: str, expected: bool) -> None:
    assert is_task_request(text) is expected


@pytest.mark.parametrize(
    ("requested", "channel", "text", "domain_matched", "expected"),
    [
        # 1. Legacy client (requested is None) -> always CHAT, LEGACY_CLIENT
        (
            None,
            Channel.TEXT,
            "```c\nint main() {}\n```",
            True,
            ModeDecision(TurnMode.CHAT, ModeReason.LEGACY_CLIENT),
        ),
        (
            None,
            Channel.VOICE,
            "Giải thích cho mình nguyên lý hoạt động?",
            True,
            ModeDecision(TurnMode.CHAT, ModeReason.LEGACY_CLIENT),
        ),
        # 2. Explicit CHAT -> CHAT, REQUESTED_CHAT
        (
            RequestedMode.CHAT,
            Channel.TEXT,
            "```c\nint main() {}\n```",
            True,
            ModeDecision(TurnMode.CHAT, ModeReason.REQUESTED_CHAT),
        ),
        # 3. Explicit EXPERT -> EXPERT, REQUESTED_EXPERT
        (
            RequestedMode.EXPERT,
            Channel.VOICE,
            "chào bạn",
            False,
            ModeDecision(TurnMode.EXPERT, ModeReason.REQUESTED_EXPERT),
        ),
        # 4. AUTO on TEXT with code or >= LONG_TEXT_CHARS -> EXPERT, CODE_OR_LONG
        (
            RequestedMode.AUTO,
            Channel.TEXT,
            "```python\nx = 1\n```",
            False,
            ModeDecision(TurnMode.EXPERT, ModeReason.CODE_OR_LONG),
        ),
        (
            RequestedMode.AUTO,
            Channel.TEXT,
            "a" * LONG_TEXT_CHARS,
            False,
            ModeDecision(TurnMode.EXPERT, ModeReason.CODE_OR_LONG),
        ),
        # Voice does NOT trigger CODE_OR_LONG on long text without domain match
        (
            RequestedMode.AUTO,
            Channel.VOICE,
            "a " * 150,
            False,
            ModeDecision(TurnMode.CHAT, ModeReason.CASUAL),
        ),
        # 5. AUTO with domain_matched and task request -> EXPERT, DOMAIN_TASK
        (
            RequestedMode.AUTO,
            Channel.TEXT,
            "Tại sao ESP32 bị reset khi bật BLE?",
            True,
            ModeDecision(TurnMode.EXPERT, ModeReason.DOMAIN_TASK),
        ),
        (
            RequestedMode.AUTO,
            Channel.VOICE,
            "Giải thích cơ chế ngắt trên STM32",
            True,
            ModeDecision(TurnMode.EXPERT, ModeReason.DOMAIN_TASK),
        ),
        # 6. AUTO fallback -> CHAT, CASUAL
        (
            RequestedMode.AUTO,
            Channel.TEXT,
            "Mình vừa đọc xong tài liệu ESP32",
            True,
            ModeDecision(TurnMode.CHAT, ModeReason.CASUAL),
        ),
        (
            RequestedMode.AUTO,
            Channel.TEXT,
            "Tại sao hôm nay mình thấy hơi buồn ngủ?",
            False,
            ModeDecision(TurnMode.CHAT, ModeReason.CASUAL),
        ),
    ],
)
def test_decide_mode_priority_and_reasons(
    requested: RequestedMode | None,
    channel: Channel,
    text: str,
    domain_matched: bool,
    expected: ModeDecision,
) -> None:
    assert decide_mode(requested, channel, text, domain_matched) == expected


def test_turn_domain_vi_benchmark_meets_thresholds() -> None:
    """Evaluates mode routing (>= 90%) and domain selection (>= 80%) on 40 Vietnamese turns (§9)."""
    raw_items = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert len(raw_items) == 40

    mode_correct = 0
    domain_correct = 0

    for item in raw_items:
        text: str = item["text"]
        channel = Channel(item["channel"])
        req_raw = item["requested"]
        requested = RequestedMode(req_raw) if req_raw is not None else None
        domains = item["domains"]

        best_idx: int | None = None
        best_score = 0.0
        for idx, dom in enumerate(domains):
            targets = [dom["name"], dom["description"], *dom.get("items", [])]
            score = compute_lexical_similarity(text, targets)
            if score > best_score:
                best_score = score
                if score >= TURN_DOMAIN_THRESHOLD:
                    best_idx = idx

        domain_matched = best_idx is not None
        decision = decide_mode(requested, channel, text, domain_matched=domain_matched)

        if decision.mode.value == item["expect_mode"]:
            mode_correct += 1
        if best_idx == item["expect_domain_index"]:
            domain_correct += 1

    mode_acc = mode_correct / len(raw_items)
    domain_acc = domain_correct / len(raw_items)

    assert mode_acc >= 0.90, f"Mode accuracy {mode_acc:.2%} < 90%"
    assert domain_acc >= 0.80, f"Domain accuracy {domain_acc:.2%} < 80%"

