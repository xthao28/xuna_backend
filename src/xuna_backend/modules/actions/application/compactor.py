"""Sub-goal Context Compactor for Autonomous Tasks (Wave D Feature 5).

Compresses tool-call history of completed sub-goals into structured summaries,
enabling 20-50 steps execution without context window or quota overflow.

Enforces two strict architectural invariants:
1. Wave A Invariant: Strictly preserves `thought_signature` on active, uncompacted turns
   (`extra={"thought_signature": ...}` or `call.extra`).
2. Wave C Consent U5 Zero-Retention: Ephemeral base64 vision screenshots
   (`Message.image_base64`, `screenshot_base64`, data URLs, or raw image payloads)
   must NEVER be saved to DB/checkpoint data or stored to disk, and must be completely
   pruned/sanitized from compacted historical turns.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from xuna_backend.modules.actions.domain.job import SubGoal, SubGoalStatus
from xuna_backend.modules.cognition.domain.types import Message, Role, ToolCall

# Threshold for detecting raw base64 data payloads in text strings
_BASE64_LENGTH_THRESHOLD = 512
_BASE64_IMAGE_KEYS = frozenset(
    {
        "image_base64",
        "screenshot_base64",
        "image_bytes",
        "screenshot",
        "image",
    }
)


def _is_base64_vision_payload(key: str, value: Any) -> bool:
    """Detects whether a key-value pair contains base64 image data."""
    key_lower = key.lower()
    if key_lower in _BASE64_IMAGE_KEYS:
        return True
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return True
        if "base64" in key_lower and len(value) > _BASE64_LENGTH_THRESHOLD:
            return True
    return False


def sanitize_vision_data(data: Any) -> Any:
    """Recursively removes base64 image payloads to enforce Consent U5 Zero-Retention."""
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for k, v in data.items():
            if _is_base64_vision_payload(str(k), v):
                continue
            cleaned[k] = sanitize_vision_data(v)
        return cleaned
    if isinstance(data, list):
        return [sanitize_vision_data(item) for item in data]
    if isinstance(data, tuple):
        return tuple(sanitize_vision_data(item) for item in data)
    if isinstance(data, str) and data.startswith("data:image/"):
        return "[IMAGE_DATA_REDACTED_U5]"
    return data


def sanitize_message_vision(msg: Message) -> Message:
    """Prunes vision screenshots from a Message while preserving thought_signature."""
    cleaned_tool_calls: list[ToolCall] = []
    for tc in msg.tool_calls:
        # Sanitize arguments, strictly preserve tc.extra (which holds thought_signature)
        cleaned_args = sanitize_vision_data(tc.arguments)
        cleaned_tool_calls.append(
            ToolCall(
                id=tc.id,
                name=tc.name,
                arguments=cleaned_args if isinstance(cleaned_args, dict) else {},
                extra=tc.extra,
            )
        )

    return Message(
        role=msg.role,
        content=msg.content,
        tool_call_id=msg.tool_call_id,
        tool_calls=tuple(cleaned_tool_calls),
        image_base64=None,  # Zero-Retention: strip vision base64
        image_mime_type=msg.image_mime_type,
    )


@dataclass(frozen=True, slots=True)
class SubGoalSummary:
    """Structured summary of a completed sub-goal for compacted context history."""

    sub_goal_id: str
    sub_goal_index: int
    title: str
    summary_text: str
    tools_used: tuple[str, ...]
    tool_call_count: int
    status: str
    outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes to camelCase dictionary for persistence and checkpoints."""
        return {
            "subGoalId": self.sub_goal_id,
            "subGoalIndex": self.sub_goal_index,
            "title": self.title,
            "summaryText": self.summary_text,
            "toolsUsed": list(self.tools_used),
            "toolCallCount": self.tool_call_count,
            "status": self.status,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubGoalSummary":
        """Deserializes from camelCase or snake_case dictionary."""
        return cls(
            sub_goal_id=str(data.get("subGoalId") or data.get("sub_goal_id", "")),
            sub_goal_index=int(data.get("subGoalIndex", data.get("sub_goal_index", 0))),
            title=str(data.get("title", "")),
            summary_text=str(data.get("summaryText") or data.get("summary_text", "")),
            tools_used=tuple(data.get("toolsUsed") or data.get("tools_used") or ()),
            tool_call_count=int(data.get("toolCallCount", data.get("tool_call_count", 0))),
            status=str(data.get("status", "")),
            outcome=data.get("outcome"),
        )


class SubGoalCompactor:
    """Compresses completed sub-goal history into structured summaries.

    Enforces Wave A thought_signature preservation and Consent U5 vision zero-retention.
    """

    def compact_sub_goal(
        self,
        sub_goal: SubGoal,
        sub_goal_index: int,
        turns: Sequence[dict[str, Any] | Message] = (),
        outcome: str | None = None,
    ) -> SubGoalSummary:
        """Compresses all tool-call turns of a completed sub-goal into a structured summary."""
        tool_counts: dict[str, int] = {}
        total_calls = 0

        for turn in turns:
            if isinstance(turn, Message):
                for tc in turn.tool_calls:
                    total_calls += 1
                    tool_counts[tc.name] = tool_counts.get(tc.name, 0) + 1
            elif isinstance(turn, dict):
                # Turn dict may contain 'tool_name', 'tool_calls', or 'name'
                if "tool_name" in turn:
                    t_name = str(turn["tool_name"])
                    total_calls += 1
                    tool_counts[t_name] = tool_counts.get(t_name, 0) + 1
                elif "name" in turn and turn.get("type") == "tool_call":
                    t_name = str(turn["name"])
                    total_calls += 1
                    tool_counts[t_name] = tool_counts.get(t_name, 0) + 1
                elif "tool_calls" in turn and isinstance(turn["tool_calls"], list):
                    for tc_item in turn["tool_calls"]:
                        if isinstance(tc_item, dict):
                            t_name = str(
                                tc_item.get("name")
                                or tc_item.get("function", {}).get("name")
                                or "tool"
                            )
                            total_calls += 1
                            tool_counts[t_name] = tool_counts.get(t_name, 0) + 1

        effective_count = max(total_calls, sub_goal.tool_call_count)
        if tool_counts:
            tools_parts = [
                f"{name} ({count} call{'s' if count > 1 else ''})" if count > 1 else name
                for name, count in sorted(tool_counts.items())
            ]
            tools_str = ", ".join(tools_parts)
        elif effective_count > 0:
            tools_str = f"{effective_count} calls"
        else:
            tools_str = "none"

        effective_outcome = (
            outcome
            or sub_goal.result_summary
            or (
                "Completed successfully" if sub_goal.status == SubGoalStatus.COMPLETED else "Failed"
            )
        )
        # Ensure outcome text never retains base64
        effective_outcome = sanitize_vision_data(effective_outcome)

        summary_text = (
            f'[SubGoal {sub_goal_index + 1} Completed: "{sub_goal.title}" | '
            f"Tools: {tools_str} | Outcome: {effective_outcome}]"
        )

        return SubGoalSummary(
            sub_goal_id=sub_goal.id,
            sub_goal_index=sub_goal_index,
            title=sub_goal.title,
            summary_text=summary_text,
            tools_used=tuple(sorted(tool_counts.keys())),
            tool_call_count=effective_count,
            status=sub_goal.status.value,
            outcome=effective_outcome,
        )

    def compact_history(
        self,
        sub_goals: Sequence[SubGoal],
        active_sub_goal_index: int,
        active_turns: Sequence[dict[str, Any]],
        existing_compacted_history: Sequence[dict[str, Any] | SubGoalSummary] = (),
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Compacts completed sub-goals and returns updated (compacted_history, active_turns).

        - Completed sub-goals preceding active_sub_goal_index are represented
          as summaries in compacted_history.
        - Active turns for the current active sub-goal are strictly preserved
          with thought_signature intact.
        - Ephemeral vision base64 screenshots are wiped from all history.
        """
        compacted: list[dict[str, Any]] = []
        already_compacted_indices: set[int] = set()

        for item in existing_compacted_history:
            if isinstance(item, SubGoalSummary):
                compacted.append(item.to_dict())
                already_compacted_indices.add(item.sub_goal_index)
            elif isinstance(item, dict):
                compacted.append(sanitize_vision_data(item))
                idx = item.get("subGoalIndex", item.get("sub_goal_index"))
                if idx is not None:
                    already_compacted_indices.add(int(idx))

        # Check completed sub-goals that have not yet been compacted
        for idx, sg in enumerate(sub_goals):
            is_completed = (
                idx < active_sub_goal_index
                or sg.status in {SubGoalStatus.COMPLETED, SubGoalStatus.SKIPPED}
            )
            if is_completed and idx not in already_compacted_indices:
                # Filter active turns belonging to this sub-goal if indexed
                sg_turns = [
                    t
                    for t in active_turns
                    if t.get("sub_goal_index") == idx or t.get("sub_goal_id") == sg.id
                ]
                summary = self.compact_sub_goal(
                    sub_goal=sg,
                    sub_goal_index=idx,
                    turns=sg_turns,
                )
                compacted.append(summary.to_dict())
                already_compacted_indices.add(idx)

        # Retain only turns for the currently active sub-goal
        remaining_active: list[dict[str, Any]] = []
        for t in active_turns:
            t_idx = t.get("sub_goal_index")
            t_id = t.get("sub_goal_id")
            # If untagged or tagged with current active sub-goal, keep in active_turns
            if (
                t_idx is None
                or t_idx == active_sub_goal_index
                or (
                    t_id
                    and active_sub_goal_index < len(sub_goals)
                    and t_id == sub_goals[active_sub_goal_index].id
                )
            ):
                # Sanitize vision data while preserving thought_signature
                cleaned = self.sanitize_active_turn(t)
                remaining_active.append(cleaned)

        return compacted, remaining_active

    def sanitize_active_turn(self, turn: dict[str, Any]) -> dict[str, Any]:
        """Sanitizes an active turn dict.

        Strictly preserves thought_signature while removing vision base64.
        """
        cleaned: dict[str, Any] = {}
        for k, v in turn.items():
            if _is_base64_vision_payload(k, v):
                continue
            if k == "extra" and isinstance(v, dict):
                # Strictly preserve thought_signature in extra
                clean_extra = dict(v)
                for ek, ev in v.items():
                    if _is_base64_vision_payload(ek, ev):
                        clean_extra.pop(ek, None)
                cleaned[k] = clean_extra
            elif k == "tool_calls" and isinstance(v, list):
                clean_calls: list[dict[str, Any]] = []
                for tc in v:
                    if isinstance(tc, dict):
                        tc_clean = dict(tc)
                        if "arguments" in tc_clean:
                            tc_clean["arguments"] = sanitize_vision_data(tc_clean["arguments"])
                        # Preserve extra / thought_signature
                        if "extra" in tc_clean and isinstance(tc_clean["extra"], dict):
                            clean_tc_extra = dict(tc_clean["extra"])
                            for ek, ev in tc_clean["extra"].items():
                                if _is_base64_vision_payload(ek, ev):
                                    clean_tc_extra.pop(ek, None)
                            tc_clean["extra"] = clean_tc_extra
                        clean_calls.append(tc_clean)
                    else:
                        clean_calls.append(tc)
                cleaned[k] = clean_calls
            elif isinstance(v, dict):
                cleaned[k] = sanitize_vision_data(v)
            elif isinstance(v, list):
                cleaned[k] = [sanitize_vision_data(item) for item in v]
            else:
                cleaned[k] = v
        return cleaned

    def sanitize_checkpoint(self, checkpoint_data: dict[str, Any]) -> dict[str, Any]:
        """Sanitizes entire checkpoint data.

        Wipes vision screenshots while preserving thought_signatures.
        """
        cleaned = dict(checkpoint_data)
        if "active_turns" in cleaned and isinstance(cleaned["active_turns"], list):
            cleaned["active_turns"] = [
                self.sanitize_active_turn(t) if isinstance(t, dict) else t
                for t in cleaned["active_turns"]
            ]
        if "compacted_history" in cleaned and isinstance(cleaned["compacted_history"], list):
            cleaned["compacted_history"] = [
                sanitize_vision_data(item) for item in cleaned["compacted_history"]
            ]
        # General sanitization for top-level keys
        res = sanitize_vision_data(cleaned)
        return res if isinstance(res, dict) else cleaned

    def build_context_messages(
        self,
        compacted_history: Sequence[dict[str, Any] | SubGoalSummary],
        active_turns: Sequence[dict[str, Any] | Message],
        system_instruction: str | None = None,
    ) -> tuple[Message, ...]:
        """Constructs provider-agnostic Message sequence for LLM context.

        Embeds compacted sub-goal summaries in a system prompt and appends
        uncompacted active turns with preserved thought_signature.
        """
        messages: list[Message] = []

        history_lines: list[str] = []
        for item in compacted_history:
            if isinstance(item, SubGoalSummary):
                history_lines.append(f"- {item.summary_text}")
            elif isinstance(item, dict):
                summary_text = item.get("summaryText") or item.get("summary_text")
                if summary_text:
                    history_lines.append(f"- {summary_text}")

        system_parts: list[str] = []
        if system_instruction:
            system_parts.append(system_instruction.strip())
        if history_lines:
            system_parts.append("[Completed Sub-Goal History]:\n" + "\n".join(history_lines))

        if system_parts:
            messages.append(
                Message(
                    role=Role.SYSTEM,
                    content="\n\n".join(system_parts),
                )
            )

        for turn in active_turns:
            if isinstance(turn, Message):
                messages.append(sanitize_message_vision(turn))
            elif isinstance(turn, dict):
                if "role" in turn:
                    role_val = turn["role"]
                    role = Role(role_val) if isinstance(role_val, str) else role_val
                    content = str(turn.get("content", ""))
                    msg_extra = turn.get("extra")
                    fallback_extra = msg_extra if isinstance(msg_extra, dict) else None
                    tool_calls: list[ToolCall] = []
                    for tc in turn.get("tool_calls", []):
                        if isinstance(tc, dict):
                            tc_id = str(tc.get("id", ""))
                            tc_name = str(tc.get("name", ""))
                            tc_args = tc.get("arguments", {})
                            tc_extra = tc.get("extra")
                            resolved_extra = (
                                tc_extra if isinstance(tc_extra, dict) else fallback_extra
                            )
                            tool_calls.append(
                                ToolCall(
                                    id=tc_id,
                                    name=tc_name,
                                    arguments=tc_args if isinstance(tc_args, dict) else {},
                                    extra=resolved_extra,
                                )
                            )
                    messages.append(
                        Message(
                            role=role,
                            content=content,
                            tool_call_id=turn.get("tool_call_id"),
                            tool_calls=tuple(tool_calls),
                            image_base64=None,
                        )
                    )
                elif "tool_name" in turn:
                    t_name = str(turn.get("tool_name", "tool"))
                    t_status = str(turn.get("status", "ok"))
                    messages.append(
                        Message(
                            role=Role.USER,
                            content=f"[Observed tool '{t_name}' -> status={t_status}]",
                        )
                    )

        return tuple(messages)

    def calculate_compression_stats(
        self,
        raw_turns_chars: int,
        compacted_history: Sequence[dict[str, Any] | SubGoalSummary],
        active_turns_chars: int,
    ) -> dict[str, Any]:
        """Calculates size reduction and compression metrics for observability."""
        compacted_chars = sum(
            len(
                item.summary_text
                if isinstance(item, SubGoalSummary)
                else (item.get("summaryText") or item.get("summary_text", ""))
            )
            for item in compacted_history
        )
        total_post_chars = compacted_chars + active_turns_chars
        saved = max(0, raw_turns_chars - total_post_chars)
        ratio = (saved / raw_turns_chars) if raw_turns_chars > 0 else 0.0

        return {
            "originalChars": raw_turns_chars,
            "compactedChars": compacted_chars,
            "activeChars": active_turns_chars,
            "totalPostChars": total_post_chars,
            "savedChars": saved,
            "compressionRatio": round(ratio, 4),
        }
