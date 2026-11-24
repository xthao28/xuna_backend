"""Planner-Executor for Multi-Step Autonomous Tasks (Wave D Feature 4).

Generates structured execution plans (3-7 sub-goals) for complex autonomous jobs,
tracks dynamic step progress per sub-goal, and enforces Rule H1 Honest Validation:
- Strictly require real tool `status == "ok"` before marking a sub-goal completed.
- Never mark complete without verified real tool success.
- If a tool fails or throws an exception, the sub-goal MUST be marked `failed`
  (or handled honestly without faking success).
- App-agnostic architecture complying with Rule H5 (zero hardcoded app names).
"""

import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from xuna_backend.modules.actions.application.compactor import SubGoalCompactor
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    JobStepOutcome,
    SubGoal,
    SubGoalStatus,
)
from xuna_backend.modules.actions.domain.redaction import sanitize_wake_text

_logger = logging.getLogger(__name__)
_SHORT_CODE_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")

# Execution plan sub-goal bounds (3-7 sub-goals)
MIN_SUB_GOALS = 3
MAX_SUB_GOALS = 7


class RuleH1ValidationError(RuntimeError):
    """Raised when marking a sub-goal completed without verified real tool success."""


class JobPlannerExecutor:
    """Generates structured execution plans (3-7 sub-goals) and drives honest execution.

    Rule H1 Honest Validation: strictly require real tool status == 'ok' before
    marking a sub-goal completed. Never mark complete without verified real tool success.
    If a tool fails or throws an exception, the sub-goal MUST be marked failed.
    """

    def __init__(
        self,
        compactor: SubGoalCompactor | None = None,
        clock_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._compactor = compactor or SubGoalCompactor()
        self._clock = clock_utc

    def generate_plan(
        self,
        goal: str,
        prompt: str | None = None,
        title: str | None = None,
        *,
        max_steps: int | None = None,
    ) -> tuple[SubGoal, ...]:
        """Generates a structured execution plan containing between 3 and 7 sub-goals.

        Parses explicit steps if present, or decomposes natural-language goals into
        generic, app-agnostic operational phases (Rule H5). When ``max_steps`` is
        tight, caps natural decomposition so early discovery phases do not exhaust
        the step budget before primary execution or ``job__wait`` runs.
        """
        now = self._clock()
        raw_text = (goal or prompt or "").strip()
        parsed_steps = self._extract_explicit_steps(raw_text)

        if not parsed_steps:
            parsed_steps = self._decompose_natural_goal(raw_text, max_steps=max_steps)

        # Enforce boundary: 3 <= len(sub_goals) <= 7
        bounded_steps = self._bound_steps(parsed_steps, raw_text)

        sub_goals: list[SubGoal] = []
        for idx, (step_title, step_desc) in enumerate(bounded_steps):
            is_first = idx == 0
            sub_goals.append(
                SubGoal(
                    id=f"sg-{idx + 1}",
                    title=step_title,
                    description=step_desc,
                    status=SubGoalStatus.IN_PROGRESS if is_first else SubGoalStatus.PENDING,
                    tool_call_count=0,
                    result_summary=None,
                    started_at=now if is_first else None,
                    completed_at=None,
                )
            )

        return tuple(sub_goals)

    def _extract_explicit_steps(self, text: str) -> list[tuple[str, str]]:
        """Extracts numbered or bulleted steps if user/caller specified an explicit plan."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        steps: list[tuple[str, str]] = []
        step_pattern = re.compile(
            r"^(?:(?:step|phase|stage|bước|chặng)\s*\d+[:.-]?|\d+[.)])\s*(.*)$",
            re.IGNORECASE,
        )

        for line in lines:
            match = step_pattern.match(line)
            if match:
                content = match.group(1).strip()
                if content:
                    parts = content.split(":", 1)
                    title = parts[0].strip()[:60]
                    desc = parts[1].strip() if len(parts) > 1 else content
                    steps.append((title or "Execute phase", desc))

        return steps

    def _decompose_natural_goal(
        self,
        goal: str,
        *,
        max_steps: int | None = None,
    ) -> list[tuple[str, str]]:
        """Decomposes a goal into standard app-agnostic phases (Rule H5)."""
        goal_summary = goal[:80] if goal else "autonomous task"
        words_count = len(goal.split())
        is_watch_goal = "job__wait" in goal.lower()

        # Tight step budgets (<= 15) or explicit watch/wait goals use 3 compact
        # phases so primary action / job__wait runs before the budget drains.
        if (max_steps is not None and max_steps <= 15) or is_watch_goal:
            return [
                (
                    "Bind target context and start operation",
                    f"Inspect active context and execute or register watch for: {goal_summary}",
                ),
                (
                    "Verify concrete outcome",
                    "Observe real host deliverables and confirm done criteria are genuinely met",
                ),
                (
                    "Conclude and summarize job",
                    "Record honest verified summary and finalize autonomous job",
                ),
            ]

        if words_count > 25 and (max_steps is None or max_steps >= 25):
            # Complex multi-step task -> 5 phases
            return [
                (
                    "Discover environment and target elements",
                    f"Inspect window tree, UI hierarchy, and target controls for: {goal_summary}",
                ),
                (
                    "Navigate and focus target context",
                    "Establish active window focus and navigate to required view or section",
                ),
                (
                    "Execute primary operational sequence",
                    f"Execute core inputs, actions, or commands required for: {goal_summary}",
                ),
                (
                    "Verify execution results and observe UI changes",
                    "Inspect state changes and verify that actions succeeded as intended",
                ),
                (
                    "Consolidate outcome and finalize task",
                    "Synthesize results, record execution summary, and finalize job",
                ),
            ]
        elif words_count > 10:
            # Medium task -> 4 phases
            return [
                (
                    "Inspect environment and target controls",
                    f"Discover controls and prepare context for: {goal_summary}",
                ),
                (
                    "Execute requested interaction",
                    f"Execute inputs and operations required for: {goal_summary}",
                ),
                (
                    "Verify outcome and validate state",
                    "Inspect UI state to confirm successful completion",
                ),
                (
                    "Conclude and summarize job",
                    "Record completion summary and finalize autonomous job",
                ),
            ]
        else:
            # Concise task -> 3 phases
            return [
                (
                    "Inspect target context",
                    f"Discover controls and verify context for: {goal_summary}",
                ),
                (
                    "Execute target operation",
                    f"Perform required actions for: {goal_summary}",
                ),
                (
                    "Verify and finalize outcome",
                    "Validate execution success and conclude job",
                ),
            ]

    def _bound_steps(
        self,
        steps: list[tuple[str, str]],
        fallback_text: str,
    ) -> list[tuple[str, str]]:
        """Ensures that the number of sub-goals is strictly between 3 and 7."""
        if len(steps) < MIN_SUB_GOALS:
            result = list(steps)
            if not result:
                result.append(
                    ("Inspect context", f"Inspect target context for: {fallback_text[:60]}")
                )
            if len(result) == 1:
                result.append(
                    ("Execute operation", f"Perform core operations for: {fallback_text[:60]}")
                )
            if len(result) == 2:
                result.append(("Verify and finalize", "Validate operational success and conclude"))
            return result

        if len(steps) > MAX_SUB_GOALS:
            # Group excess steps into the penultimate step
            head = steps[: MAX_SUB_GOALS - 2]
            middle_desc = " | ".join(desc for _, desc in steps[MAX_SUB_GOALS - 2 : -1])
            middle = ("Execute extended operations", middle_desc)
            tail = steps[-1]
            return [*head, middle, tail]

        return steps

    async def execute_tool_with_honest_validation(
        self,
        job: AutonomousJob,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_invoker: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
        *,
        mark_sub_goal_complete_on_success: bool = False,
        outcome_summary: str | None = None,
    ) -> tuple[dict[str, Any], JobStepOutcome]:
        """Invokes a tool and validates its result strictly under Rule H1.

        - If tool throws an exception: marks sub-goal FAILED, returns failure outcome.
        - If tool returns status != 'ok': marks sub-goal FAILED, returns failure outcome.
        - Only if tool returns status == 'ok' is the step marked progress/completed.
        """
        now = self._clock()
        try:
            tool_result = await tool_invoker(tool_name, tool_args)
        except Exception as exc:
            _logger.exception(
                "Tool '%s' raised exception during sub-goal execution: %s", tool_name, exc
            )
            tool_result = {
                "status": "failed",
                "errorCode": "tool_exception",
                "error": str(exc),
            }

        outcome = self.process_tool_result(
            job=job,
            tool_name=tool_name,
            tool_result=tool_result,
            mark_sub_goal_complete=mark_sub_goal_complete_on_success,
            outcome_summary=outcome_summary,
            now=now,
        )
        return tool_result, outcome

    def process_tool_result(
        self,
        job: AutonomousJob,
        tool_name: str,
        tool_result: dict[str, Any],
        mark_sub_goal_complete: bool = False,
        outcome_summary: str | None = None,
        now: datetime | None = None,
    ) -> JobStepOutcome:
        """Evaluates tool result against Rule H1 Honest Validation.

        Rule H1: Strictly require real tool `status == "ok"` before marking a sub-goal completed.
        Never mark complete without verified real tool success. If a tool fails, mark failed.
        """
        now_dt = now or self._clock()
        sub_goals = list(job.sub_goals)
        idx = job.active_sub_goal_index

        if not sub_goals or idx >= len(sub_goals):
            return JobStepOutcome(
                kind="failed",
                error_message="No active sub-goal to execute",
                stop_reason="invalid_sub_goal_index",
            )

        current_sg = sub_goals[idx]
        tool_status = str(tool_result.get("status", "")).lower()

        # Rule H1 Honest Validation Check
        is_tool_ok = tool_status == "ok"
        resolved_lw = self._resolve_last_window(job, tool_result, is_tool_ok=is_tool_ok)

        if not is_tool_ok:
            # Rule H1 Violation Prevention: Tool failed!
            raw_error_detail = (
                tool_result.get("error")
                or tool_result.get("errorMessage")
                or tool_result.get("errorCode")
                or f"tool returned status '{tool_status or 'unknown'}'"
            )
            error_detail = sanitize_wake_text(str(raw_error_detail)) or "tool execution failed"
            failed_sg = SubGoal(
                id=current_sg.id,
                title=current_sg.title,
                description=current_sg.description,
                status=SubGoalStatus.FAILED,
                tool_call_count=current_sg.tool_call_count + 1,
                result_summary=(
                    f"Rule H1 check failed: tool '{tool_name}' failed with {error_detail}"
                ),
                started_at=current_sg.started_at or now_dt,
                completed_at=now_dt,
            )
            sub_goals[idx] = failed_sg

            # Compact the failed sub-goal for auditability
            compacted_history = list(
                job.checkpoint_data.get("compacted_history", []) if job.checkpoint_data else []
            )
            failed_summary = self._compactor.compact_sub_goal(
                sub_goal=failed_sg,
                sub_goal_index=idx,
                turns=job.checkpoint_data.get("active_turns", []) if job.checkpoint_data else [],
                outcome=f"Failed: {error_detail}",
            )
            compacted_history.append(failed_summary.to_dict())

            checkpoint: dict[str, Any] = {
                "current_sub_goal_idx": idx,
                "step_count": job.steps_used + 1,
                "sub_goals": [sg.to_dict() for sg in sub_goals],
                "compacted_history": compacted_history,
                "active_turns": [],
                "pending_approval_id": None,
            }
            if resolved_lw is not None:
                checkpoint["last_window"] = resolved_lw

            return JobStepOutcome(
                kind="failed",
                sub_goals=tuple(sub_goals),
                active_sub_goal_index=idx,
                checkpoint_data=self._compactor.sanitize_checkpoint(checkpoint),
                error_message=f"Sub-goal '{current_sg.title}' failed: {error_detail}",
                stop_reason="tool_failure",
            )

        # Tool genuinely succeeded (status == 'ok')
        new_count = current_sg.tool_call_count + 1

        if mark_sub_goal_complete:
            # Honestly complete this sub-goal
            raw_summary = outcome_summary or f"Tool '{tool_name}' executed successfully"
            effective_summary = (
                sanitize_wake_text(raw_summary) or f"Tool '{tool_name}' executed successfully"
            )
            completed_sg = SubGoal(
                id=current_sg.id,
                title=current_sg.title,
                description=current_sg.description,
                status=SubGoalStatus.COMPLETED,
                tool_call_count=new_count,
                result_summary=effective_summary,
                started_at=current_sg.started_at or now_dt,
                completed_at=now_dt,
            )
            sub_goals[idx] = completed_sg

            # Sub-goal Context Compaction across boundary
            existing_compacted = list(
                job.checkpoint_data.get("compacted_history", []) if job.checkpoint_data else []
            )
            active_turns = list(
                job.checkpoint_data.get("active_turns", []) if job.checkpoint_data else []
            )
            active_turns.append(
                {
                    "sub_goal_index": idx,
                    "tool_name": tool_name,
                    "status": "ok",
                    "result": tool_result,
                }
            )

            compacted_history, remaining_active = self._compactor.compact_history(
                sub_goals=sub_goals,
                active_sub_goal_index=idx + 1,
                active_turns=active_turns,
                existing_compacted_history=existing_compacted,
            )

            next_idx = idx + 1
            if next_idx < len(sub_goals):
                next_sg = sub_goals[next_idx]
                sub_goals[next_idx] = SubGoal(
                    id=next_sg.id,
                    title=next_sg.title,
                    description=next_sg.description,
                    status=SubGoalStatus.IN_PROGRESS,
                    tool_call_count=0,
                    started_at=now_dt,
                )

            all_completed = all(sg.status == SubGoalStatus.COMPLETED for sg in sub_goals)
            checkpoint = {
                "current_sub_goal_idx": min(next_idx, len(sub_goals) - 1),
                "step_count": job.steps_used + 1,
                "sub_goals": [sg.to_dict() for sg in sub_goals],
                "compacted_history": compacted_history,
                "active_turns": remaining_active,
                "pending_approval_id": None,
            }
            if resolved_lw is not None:
                checkpoint["last_window"] = resolved_lw

            if all_completed:
                return JobStepOutcome(
                    kind="completed",
                    sub_goals=tuple(sub_goals),
                    active_sub_goal_index=len(sub_goals) - 1,
                    checkpoint_data=self._compactor.sanitize_checkpoint(checkpoint),
                    stop_reason="all_sub_goals_completed",
                )

            return JobStepOutcome(
                kind="progress",
                sub_goals=tuple(sub_goals),
                active_sub_goal_index=next_idx,
                checkpoint_data=self._compactor.sanitize_checkpoint(checkpoint),
            )

        # Still in progress within current sub-goal
        sub_goals[idx] = SubGoal(
            id=current_sg.id,
            title=current_sg.title,
            description=current_sg.description,
            status=SubGoalStatus.IN_PROGRESS,
            tool_call_count=new_count,
            result_summary=current_sg.result_summary,
            started_at=current_sg.started_at or now_dt,
            completed_at=None,
        )

        existing_compacted = list(
            job.checkpoint_data.get("compacted_history", []) if job.checkpoint_data else []
        )
        active_turns = list(
            job.checkpoint_data.get("active_turns", []) if job.checkpoint_data else []
        )
        active_turns.append(
            {
                "sub_goal_index": idx,
                "tool_name": tool_name,
                "status": "ok",
                "result": tool_result,
            }
        )

        checkpoint = {
            "current_sub_goal_idx": idx,
            "step_count": job.steps_used + 1,
            "sub_goals": [sg.to_dict() for sg in sub_goals],
            "compacted_history": existing_compacted,
            "active_turns": active_turns,
            "pending_approval_id": None,
        }
        if resolved_lw is not None:
            checkpoint["last_window"] = resolved_lw

        return JobStepOutcome(
            kind="progress",
            sub_goals=tuple(sub_goals),
            active_sub_goal_index=idx,
            checkpoint_data=self._compactor.sanitize_checkpoint(checkpoint),
        )

    @staticmethod
    def _resolve_last_window(
        job: AutonomousJob,
        tool_result: dict[str, Any] | None = None,
        *,
        is_tool_ok: bool = False,
    ) -> dict[str, Any] | None:
        prev_raw = job.checkpoint_data.get("last_window") if job.checkpoint_data else None
        prev_lw = dict(prev_raw) if isinstance(prev_raw, dict) else None
        if not is_tool_ok or tool_result is None:
            return prev_lw

        raw_output = tool_result.get("output")
        payload: dict[str, Any] = raw_output if isinstance(raw_output, dict) else tool_result
        if str(payload.get("status", "")).lower() != "running":
            return prev_lw

        raw_mcp = payload.get("mcpServerId")
        raw_task = payload.get("protocolTaskId")
        if not isinstance(raw_mcp, str) or not isinstance(raw_task, str):
            return prev_lw

        clean_mcp = raw_mcp.strip()
        clean_task = raw_task.strip()
        if (
            not clean_mcp
            or not clean_task
            or len(clean_mcp) > 64
            or len(clean_task) > 128
            or "\n" in clean_mcp
            or "\r" in clean_mcp
            or not _SHORT_CODE_RE.fullmatch(clean_mcp)
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in clean_task)
        ):
            return prev_lw

        next_lw = dict(prev_lw) if prev_lw is not None else {}
        next_lw["mcpServerId"] = clean_mcp
        next_lw["protocolTaskId"] = clean_task
        return next_lw

    def complete_sub_goal_after_verified_tools(
        self,
        job: AutonomousJob,
        *,
        outcome_summary: str | None = None,
        now: datetime | None = None,
    ) -> JobStepOutcome:
        """Marks the active sub-goal completed ONLY if >= 1 real tool succeeded in it (Rule H1).

        If zero real tools were executed in this sub-goal (`tool_call_count < 1`), raises no fake
        completion and fails closed with `no_tool_executed`.
        """
        now_dt = now or self._clock()
        sub_goals = list(job.sub_goals)
        idx = job.active_sub_goal_index

        if not sub_goals or idx >= len(sub_goals):
            return JobStepOutcome(
                kind="failed",
                error_message="No active sub-goal to complete",
                stop_reason="invalid_sub_goal_index",
            )

        prev_lw = self._resolve_last_window(job)
        current_sg = sub_goals[idx]
        if current_sg.tool_call_count < 1:
            failed_sg = SubGoal(
                id=current_sg.id,
                title=current_sg.title,
                description=current_sg.description,
                status=SubGoalStatus.FAILED,
                tool_call_count=0,
                result_summary="Rule H1 check failed: no real tool executed in sub-goal",
                started_at=current_sg.started_at or now_dt,
                completed_at=now_dt,
            )
            sub_goals[idx] = failed_sg
            checkpoint: dict[str, Any] = {
                "current_sub_goal_idx": idx,
                "step_count": job.steps_used + 1,
                "sub_goals": [sg.to_dict() for sg in sub_goals],
                "compacted_history": list(
                    job.checkpoint_data.get("compacted_history", []) if job.checkpoint_data else []
                ),
                "active_turns": [],
                "pending_approval_id": None,
            }
            if prev_lw is not None:
                checkpoint["last_window"] = prev_lw
            return JobStepOutcome(
                kind="failed",
                sub_goals=tuple(sub_goals),
                active_sub_goal_index=idx,
                checkpoint_data=self._compactor.sanitize_checkpoint(checkpoint),
                error_message=f"Sub-goal '{current_sg.title}' executed no real tools",
                stop_reason="no_tool_executed",
            )

        raw_summary = (
            outcome_summary
            or current_sg.result_summary
            or f"Completed '{current_sg.title}' ({current_sg.tool_call_count} verified tool calls)"
        )
        effective_summary = (
            sanitize_wake_text(raw_summary)
            or f"Completed '{current_sg.title}' ({current_sg.tool_call_count} verified tool calls)"
        )
        completed_sg = SubGoal(
            id=current_sg.id,
            title=current_sg.title,
            description=current_sg.description,
            status=SubGoalStatus.COMPLETED,
            tool_call_count=current_sg.tool_call_count,
            result_summary=effective_summary,
            started_at=current_sg.started_at or now_dt,
            completed_at=now_dt,
        )
        sub_goals[idx] = completed_sg

        existing_compacted = list(
            job.checkpoint_data.get("compacted_history", []) if job.checkpoint_data else []
        )
        active_turns = list(
            job.checkpoint_data.get("active_turns", []) if job.checkpoint_data else []
        )
        compacted_history, remaining_active = self._compactor.compact_history(
            sub_goals=sub_goals,
            active_sub_goal_index=idx + 1,
            active_turns=active_turns,
            existing_compacted_history=existing_compacted,
        )

        next_idx = idx + 1
        if next_idx < len(sub_goals):
            next_sg = sub_goals[next_idx]
            sub_goals[next_idx] = SubGoal(
                id=next_sg.id,
                title=next_sg.title,
                description=next_sg.description,
                status=SubGoalStatus.IN_PROGRESS,
                tool_call_count=0,
                started_at=now_dt,
            )

        all_completed = all(sg.status == SubGoalStatus.COMPLETED for sg in sub_goals)
        checkpoint = {
            "current_sub_goal_idx": min(next_idx, len(sub_goals) - 1),
            "step_count": job.steps_used + 1,
            "sub_goals": [sg.to_dict() for sg in sub_goals],
            "compacted_history": compacted_history,
            "active_turns": remaining_active,
            "pending_approval_id": None,
        }
        if prev_lw is not None:
            checkpoint["last_window"] = prev_lw

        if all_completed:
            return JobStepOutcome(
                kind="completed",
                sub_goals=tuple(sub_goals),
                active_sub_goal_index=len(sub_goals) - 1,
                checkpoint_data=self._compactor.sanitize_checkpoint(checkpoint),
                stop_reason="all_sub_goals_completed",
            )

        return JobStepOutcome(
            kind="progress",
            sub_goals=tuple(sub_goals),
            active_sub_goal_index=next_idx,
            checkpoint_data=self._compactor.sanitize_checkpoint(checkpoint),
        )

    def step(self, job: AutonomousJob) -> JobStepOutcome:
        """Fail-closed fallback when no real `step_executor` is configured (Rule H1 & H2).

        Never fabricates synthetic `status == 'ok'` tool outputs or fake sub-goal completions.
        """
        del job
        return JobStepOutcome(
            kind="failed",
            error_message="Autonomous job step executor is not configured",
            stop_reason="executor_not_configured",
        )
