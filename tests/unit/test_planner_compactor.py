"""Comprehensive Unit & Integration Tests for Planner-Executor + Sub-goal Compaction.

Wave D Milestone M2:
1. Plan Generation: 3-7 sub-goals, explicit vs natural language, Rule H5 app-agnostic.
2. Sub-goal State Transitions: PENDING -> IN_PROGRESS -> COMPLETED/FAILED/SKIPPED.
3. Rule H1 Honest Validation: strictly require real tool status == 'ok' before completing.
   Adversarial check: failed tools or exceptions MUST mark sub-goal FAILED.
4. Sub-goal Compaction: structured summary format, tools aggregation, outcome recording.
5. Context Size Reduction: > 80% reduction across 20-30 steps to prevent overflow.
6. Wave A Invariant: strictly preserve thought_signature on active, uncompacted turns.
7. Consent U5 Zero-Retention: complete eradication of base64 screenshots from history.
8. AutonomousJobRunner Integration: running multi-subgoal task with compaction over 20+ steps.
9. AutonomousJobRunner Integration: honest failure propagation when a tool fails mid-task.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from xuna_backend.modules.actions.application.compactor import (
    SubGoalCompactor,
    sanitize_message_vision,
)
from xuna_backend.modules.actions.application.job_runner import (
    AutonomousJobRunner,
    JobStepOutcome,
)
from xuna_backend.modules.actions.application.planner_executor import (
    MAX_SUB_GOALS,
    MIN_SUB_GOALS,
    JobPlannerExecutor,
)
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    SubGoal,
    SubGoalStatus,
)
from xuna_backend.modules.actions.infrastructure.job_store import InMemoryAutonomousJobStore
from xuna_backend.modules.cognition.domain.types import Message, Role, ToolCall

# ---------------------------------------------------------------------------
# 1. Plan Generation Tests (3-7 Sub-goals & Rule H5 App-Agnostic)
# ---------------------------------------------------------------------------


class TestPlanGeneration:
    def test_concise_goal_generates_minimum_3_sub_goals(self) -> None:
        planner = JobPlannerExecutor()
        plan = planner.generate_plan("Check notification status")
        assert len(plan) >= MIN_SUB_GOALS
        assert len(plan) <= MAX_SUB_GOALS
        assert len(plan) == 3

        # First sub-goal is in progress, others are pending
        assert plan[0].status == SubGoalStatus.IN_PROGRESS
        assert plan[0].started_at is not None
        assert plan[1].status == SubGoalStatus.PENDING
        assert plan[2].status == SubGoalStatus.PENDING

    def test_complex_goal_generates_up_to_5_or_7_sub_goals(self) -> None:
        planner = JobPlannerExecutor()
        complex_goal = (
            "Inspect the current window tree, locate the data export form, fill out the user "
            "filters, submit the request, wait for processing to finish, verify downloaded file, "
            "and record execution metrics for the owner."
        )
        plan = planner.generate_plan(complex_goal)
        assert MIN_SUB_GOALS <= len(plan) <= MAX_SUB_GOALS
        assert len(plan) == 5

    def test_explicit_numbered_steps_are_parsed(self) -> None:
        planner = JobPlannerExecutor()
        explicit_goal = """
        1. Discover UI: Inspect the desktop workspace and locate target control
        2. Focus Window: Bring target window to foreground
        3. Click Button: Trigger execution action
        4. Validate Result: Confirm dialog shows success message
        """
        plan = planner.generate_plan(explicit_goal)
        assert len(plan) == 4
        assert "Discover UI" in plan[0].title
        assert "Focus Window" in plan[1].title
        assert "Click Button" in plan[2].title
        assert "Validate Result" in plan[3].title

    def test_excessive_steps_are_bounded_to_max_7(self) -> None:
        planner = JobPlannerExecutor()
        many_steps = "\n".join(f"Step {i}: Action description for phase {i}" for i in range(1, 12))
        plan = planner.generate_plan(many_steps)
        assert len(plan) <= MAX_SUB_GOALS
        assert len(plan) == 7

    def test_rule_h5_app_agnostic_clean(self) -> None:
        planner = JobPlannerExecutor()
        plan = planner.generate_plan("Open spreadsheet, summarize column C, and save report")
        for sg in plan:
            # Rule H5: generic operational phrases, zero hardcoded vendor apps
            title_lower = sg.title.lower()
            desc_lower = sg.description.lower()
            for forbidden in ("excel", "word", "photoshop", "slack", "chrome"):
                assert forbidden not in title_lower
                assert forbidden not in desc_lower

    def test_tight_step_budget_and_watch_goals_use_compact_3_phase_plan(self) -> None:
        planner = JobPlannerExecutor()
        long_watch_goal = (
            "Bind the currently active workspace context on the host, immediately invoke "
            "job__wait with a 15 second quiescence timeout so zero tokens are spent while "
            "waiting, and after waking inspect changedFiles and report the honest outcome "
            "to the owner."
        )
        # Tight max_steps caps decomposition to 3 compact phases
        compact_by_budget = planner.generate_plan(long_watch_goal, max_steps=10)
        assert len(compact_by_budget) == MIN_SUB_GOALS
        assert "watch" in compact_by_budget[0].description.lower()

        # Explicit job__wait goal also uses 3 compact phases even without tight max_steps
        compact_by_watch = planner.generate_plan(long_watch_goal)
        assert len(compact_by_watch) == MIN_SUB_GOALS
        assert "changedfiles" in compact_by_watch[1].description.lower() or (
            "deliverables" in compact_by_watch[1].description.lower()
        )


# ---------------------------------------------------------------------------
# 2. Sub-goal State Machine & Progress Tracking
# ---------------------------------------------------------------------------


class TestSubGoalStateTransitions:
    def test_sub_goal_lifecycle_transitions(self) -> None:
        now = datetime(2026, 11, 15, 14, 0, 0, tzinfo=UTC)
        sg = SubGoal(
            id="sg-1",
            title="Inspect Context",
            description="Discover controls",
            status=SubGoalStatus.PENDING,
        )
        assert sg.status == SubGoalStatus.PENDING
        assert sg.started_at is None

        # Transition to IN_PROGRESS
        in_prog = SubGoal(
            id=sg.id,
            title=sg.title,
            description=sg.description,
            status=SubGoalStatus.IN_PROGRESS,
            started_at=now,
        )
        assert in_prog.status == SubGoalStatus.IN_PROGRESS
        assert in_prog.started_at == now

        # Transition to COMPLETED
        comp = SubGoal(
            id=sg.id,
            title=sg.title,
            description=sg.description,
            status=SubGoalStatus.COMPLETED,
            tool_call_count=3,
            result_summary="Controls discovered successfully",
            started_at=now,
            completed_at=now + timedelta(seconds=10),
        )
        assert comp.status == SubGoalStatus.COMPLETED
        assert comp.tool_call_count == 3
        assert comp.completed_at == now + timedelta(seconds=10)

    def test_sub_goal_skip_transition(self) -> None:
        now = datetime(2026, 11, 15, 14, 0, 0, tzinfo=UTC)
        sg = SubGoal(
            id="sg-2",
            title="Optional Step",
            description="May be skipped if already satisfied",
            status=SubGoalStatus.PENDING,
        )
        skipped = SubGoal(
            id=sg.id,
            title=sg.title,
            description=sg.description,
            status=SubGoalStatus.SKIPPED,
            result_summary="Condition pre-satisfied, skipped",
            completed_at=now,
        )
        assert skipped.status == SubGoalStatus.SKIPPED
        assert skipped.completed_at == now


# ---------------------------------------------------------------------------
# 3. Rule H1 Honest Validation Tests
# ---------------------------------------------------------------------------


class TestRuleH1HonestValidation:
    def test_honest_completion_requires_tool_status_ok(self) -> None:
        planner = JobPlannerExecutor()
        job = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
            pet_id="xuna_pet",
            host_id=None,
            session_id=None,
            title="Honest Job",
            goal="Execute single verified step",
            state=AutonomousJobState.RUNNING,
            sub_goals=(
                SubGoal(
                    id="sg-1",
                    title="Run Tool",
                    description="Perform real action",
                    status=SubGoalStatus.IN_PROGRESS,
                ),
            ),
            active_sub_goal_index=0,
        )

        outcome = planner.process_tool_result(
            job=job,
            tool_name="ui.inspect",
            tool_result={"status": "ok", "nodes": [{"role": "button", "name": "Submit"}]},
            mark_sub_goal_complete=True,
            outcome_summary="Node verified ok",
        )

        assert outcome.kind == "completed"
        assert outcome.sub_goals is not None
        assert outcome.sub_goals[0].status == SubGoalStatus.COMPLETED
        assert outcome.sub_goals[0].tool_call_count == 1
        assert "Node verified ok" in str(outcome.sub_goals[0].result_summary)

    def test_adversarial_failed_tool_marks_sub_goal_failed(self) -> None:
        """Adversarial check: Even if caller sets mark_sub_goal_complete=True, a failed tool

        MUST mark the sub-goal FAILED and MUST NEVER mark it COMPLETED (Rule H1).
        """
        planner = JobPlannerExecutor()
        job = AutonomousJob(
            job_id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
            pet_id="xuna_pet",
            host_id=None,
            session_id=None,
            title="Adversarial Test",
            goal="Attempt fake completion",
            state=AutonomousJobState.RUNNING,
            sub_goals=(
                SubGoal(
                    id="sg-1",
                    title="Critical Action",
                    description="Must succeed",
                    status=SubGoalStatus.IN_PROGRESS,
                ),
            ),
            active_sub_goal_index=0,
        )

        # Tool returns failed
        outcome = planner.process_tool_result(
            job=job,
            tool_name="input.click",
            tool_result={"status": "failed", "errorCode": "element_not_found"},
            mark_sub_goal_complete=True,  # Caller maliciously attempts to mark complete!
            outcome_summary="Pretending it completed anyway",
        )

        assert outcome.kind == "failed"
        assert outcome.sub_goals is not None
        # Rule H1 check: sub-goal MUST NOT be completed!
        assert outcome.sub_goals[0].status == SubGoalStatus.FAILED
        assert "Rule H1 check failed" in str(outcome.sub_goals[0].result_summary)
        assert outcome.sub_goals[0].completed_at is not None

    def test_tool_exception_marks_sub_goal_failed(self) -> None:
        async def _test() -> None:
            planner = JobPlannerExecutor()
            job = AutonomousJob(
                job_id=uuid.uuid4(),
                owner_id=uuid.uuid4(),
                pet_id="xuna_pet",
                host_id=None,
                session_id=None,
                title="Exception Job",
                goal="Action that raises",
                state=AutonomousJobState.RUNNING,
                sub_goals=(
                    SubGoal(
                        id="sg-1",
                        title="Explosive Tool",
                        description="Throws error",
                        status=SubGoalStatus.IN_PROGRESS,
                    ),
                ),
                active_sub_goal_index=0,
            )

            async def _failing_invoker(name: str, args: dict[str, Any]) -> dict[str, Any]:
                raise ConnectionResetError("Host channel dropped mid-call")

            tool_result, outcome = await planner.execute_tool_with_honest_validation(
                job=job,
                tool_name="network.fetch",
                tool_args={"url": "http://target"},
                tool_invoker=_failing_invoker,
                mark_sub_goal_complete_on_success=True,
            )

            assert tool_result["status"] == "failed"
            assert outcome.kind == "failed"
            assert outcome.sub_goals is not None
            assert outcome.sub_goals[0].status == SubGoalStatus.FAILED
            assert "Host channel dropped mid-call" in str(outcome.sub_goals[0].result_summary)

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# 4. Sub-goal Context Compactor Tests
# ---------------------------------------------------------------------------


class TestSubGoalCompactor:
    def test_structured_summary_format(self) -> None:
        compactor = SubGoalCompactor()
        sg = SubGoal(
            id="sg-1",
            title="Inspect Environment",
            description="Discover windows",
            status=SubGoalStatus.COMPLETED,
            tool_call_count=2,
            result_summary="Target window found",
        )
        turns = [
            {"tool_name": "ui.inspect", "status": "ok"},
            {"tool_name": "ui.inspect", "status": "ok"},
        ]

        summary = compactor.compact_sub_goal(sg, sub_goal_index=0, turns=turns)
        assert summary.sub_goal_id == "sg-1"
        assert summary.sub_goal_index == 0
        assert summary.tool_call_count == 2
        assert "ui.inspect" in summary.tools_used
        assert '[SubGoal 1 Completed: "Inspect Environment"' in summary.summary_text
        assert "Tools: ui.inspect (2 calls)" in summary.summary_text
        assert "Outcome: Target window found" in summary.summary_text

    def test_context_size_reduction_across_completed_sub_goals(self) -> None:
        """Verifies that 20-30 tool calls with large payloads are compacted by > 80%."""
        compactor = SubGoalCompactor()

        # Simulate heavy tool turns across 3 completed sub-goals (10 tool calls each)
        raw_turns: list[dict[str, Any]] = []
        sub_goals: list[SubGoal] = []
        for sg_idx in range(3):
            sg = SubGoal(
                id=f"sg-{sg_idx + 1}",
                title=f"Complex Phase {sg_idx + 1}",
                description="Long description",
                status=SubGoalStatus.COMPLETED,
                tool_call_count=10,
                result_summary="Phase completed successfully",
            )
            sub_goals.append(sg)
            for _call_idx in range(10):
                raw_turns.append(
                    {
                        "sub_goal_index": sg_idx,
                        "tool_name": "ui.inspect_tree",
                        "status": "ok",
                        "arguments": {
                            "depth": 5,
                            "tree_payload": "x" * 2000,  # ~2KB per call
                        },
                        "result": {
                            "nodes": [{"id": f"node_{i}", "data": "y" * 100} for i in range(20)]
                        },
                    }
                )

        raw_size = sum(len(str(t)) for t in raw_turns)
        assert raw_size > 60000  # Over 60 KB of raw context!

        compacted_history, remaining_active = compactor.compact_history(
            sub_goals=sub_goals,
            active_sub_goal_index=3,
            active_turns=raw_turns,
        )

        assert len(compacted_history) == 3
        assert len(remaining_active) == 0

        compacted_size = sum(len(item["summaryText"]) for item in compacted_history)
        stats = compactor.calculate_compression_stats(
            raw_turns_chars=raw_size,
            compacted_history=compacted_history,
            active_turns_chars=0,
        )

        assert stats["compressionRatio"] >= 0.85
        assert compacted_size < 1000  # Less than 1 KB total! Enables running 40-50 steps cleanly!

    def test_wave_a_thought_signature_preservation(self) -> None:
        """Wave A Invariant: thought_signature MUST be preserved on active turns."""
        compactor = SubGoalCompactor()
        sig_data = {"thought_signature": "sig_wave_a_clawbot_autonomous_987"}

        active_turns = [
            {
                "sub_goal_index": 1,
                "role": "assistant",
                "content": "",
                "extra": sig_data,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "ui.inspect",
                        "arguments": {"scope": "active"},
                        "extra": sig_data,
                    }
                ],
            }
        ]

        # Sanitize active turn
        sanitized = compactor.sanitize_active_turn(active_turns[0])
        assert sanitized["extra"] == sig_data
        assert sanitized["tool_calls"][0]["extra"] == sig_data

        # Build context messages and verify thought_signature survives
        messages = compactor.build_context_messages(
            compacted_history=[],
            active_turns=active_turns,
        )
        assert len(messages) == 1
        assert messages[0].tool_calls[0].extra == sig_data

    def test_consent_u5_zero_retention_sanitization(self) -> None:
        """Consent U5 Zero-Retention: base64 screenshots must NEVER persist or leak into history."""
        compactor = SubGoalCompactor()
        fake_b64 = "data:image/jpeg;base64," + ("A" * 1024)

        dirty_turn = {
            "sub_goal_index": 0,
            "image_base64": fake_b64,
            "screenshot_base64": "A" * 2048,
            "tool_calls": [
                {
                    "id": "call-vis-1",
                    "name": "ui.observe_hybrid",
                    "arguments": {"screenshot": fake_b64, "meta": "keep_this"},
                    "extra": {"thought_signature": "sig_vision_turn"},
                }
            ],
        }

        sanitized = compactor.sanitize_active_turn(dirty_turn)
        assert "image_base64" not in sanitized
        assert "screenshot_base64" not in sanitized
        assert "screenshot" not in sanitized["tool_calls"][0]["arguments"]
        assert sanitized["tool_calls"][0]["arguments"]["meta"] == "keep_this"
        # Wave A thought_signature is strictly preserved
        assert sanitized["tool_calls"][0]["extra"] == {"thought_signature": "sig_vision_turn"}

        # Test Message sanitization
        dirty_msg = Message(
            role=Role.ASSISTANT,
            content="Screenshot observed",
            image_base64=fake_b64,
            tool_calls=(
                ToolCall(
                    id="call-2",
                    name="ui.observe_hybrid",
                    arguments={"screenshot": fake_b64},
                    extra={"thought_signature": "sig_msg"},
                ),
            ),
        )
        cleaned_msg = sanitize_message_vision(dirty_msg)
        assert cleaned_msg.image_base64 is None
        assert "screenshot" not in cleaned_msg.tool_calls[0].arguments
        assert cleaned_msg.tool_calls[0].extra == {"thought_signature": "sig_msg"}


# ---------------------------------------------------------------------------
# 5. AutonomousJobRunner Integration Tests (20+ Steps & Multi-Subgoals)
# ---------------------------------------------------------------------------


class TestJobRunnerPlannerIntegration:
    def test_multi_subgoal_task_with_compaction_over_20_steps(self) -> None:
        """Integration test executing a multi-subgoal task with compaction over 20+ steps."""

        async def _test() -> None:
            store = InMemoryAutonomousJobStore()
            runner = AutonomousJobRunner(job_store=store)
            job_id = uuid.uuid4()
            owner_id = uuid.uuid4()

            step_counter = 0

            # Step executor simulating a 25-step task across 4 sub-goals
            async def _custom_stepper(job: AutonomousJob) -> JobStepOutcome:
                nonlocal step_counter
                step_counter += 1
                now = datetime.now(UTC)

                idx = job.active_sub_goal_index

                # Each sub-goal takes ~6 tool steps to complete
                is_sub_goal_done = (step_counter % 6 == 0) or (step_counter >= 24)

                outcome = runner.planner.process_tool_result(
                    job=job,
                    tool_name=f"tool_step_{step_counter}",
                    tool_result={"status": "ok", "step": step_counter},
                    mark_sub_goal_complete=is_sub_goal_done,
                    outcome_summary=f"Sub-goal {idx + 1} finished at step {step_counter}",
                    now=now,
                )
                return outcome

            # Wire the custom multi-step executor
            runner._step_executor = _custom_stepper

            # Define 4 explicit sub-goals
            four_sub_goals = [
                SubGoal(
                    id="sg-1",
                    title="Phase 1: Setup",
                    description="desc 1",
                    status=SubGoalStatus.IN_PROGRESS,
                ),
                SubGoal(
                    id="sg-2",
                    title="Phase 2: Action",
                    description="desc 2",
                    status=SubGoalStatus.PENDING,
                ),
                SubGoal(
                    id="sg-3",
                    title="Phase 3: Processing",
                    description="desc 3",
                    status=SubGoalStatus.PENDING,
                ),
                SubGoal(
                    id="sg-4",
                    title="Phase 4: Verification",
                    description="desc 4",
                    status=SubGoalStatus.PENDING,
                ),
            ]

            await runner.start_job(
                job_id=job_id,
                owner_id=owner_id,
                goal="Execute long multi-phase task",
                sub_goals=four_sub_goals,
                max_steps=40,
            )

            # Wait for execution across 20+ steps to completion
            for _ in range(60):
                await asyncio.sleep(0.02)
                current = await store.get(job_id)
                if current and current.state in {
                    AutonomousJobState.COMPLETED,
                    AutonomousJobState.FAILED,
                }:
                    break

            final = await store.get(job_id)
            assert final is not None
            assert final.state == AutonomousJobState.COMPLETED
            assert final.steps_used >= 20  # Verified executed over 20 steps!

            # Verify context compaction in checkpoint data
            cp = final.checkpoint_data
            assert cp is not None
            compacted_history = cp.get("compacted_history", [])
            assert len(compacted_history) == 4  # All 4 sub-goals compacted!
            for ch in compacted_history:
                assert "Completed" in ch["summaryText"]
                assert ch["toolCallCount"] > 0

            # Verify Consent U5 Zero-Retention
            cp_str = str(cp)
            assert "data:image/" not in cp_str
            assert "screenshot_base64" not in cp_str

            await runner.shutdown()

        asyncio.run(_test())

    def test_mid_job_tool_failure_halts_job_honestly(self) -> None:
        """Integration test verifying that a tool failure mid-job halts honestly (Rule H1)."""

        async def _test() -> None:
            store = InMemoryAutonomousJobStore()
            runner = AutonomousJobRunner(job_store=store)
            job_id = uuid.uuid4()
            owner_id = uuid.uuid4()

            step_count = 0

            async def _failing_midway_stepper(job: AutonomousJob) -> JobStepOutcome:
                nonlocal step_count
                step_count += 1
                if step_count == 3:
                    # Tool 3 fails catastrophically!
                    return runner.planner.process_tool_result(
                        job=job,
                        tool_name="database.commit",
                        tool_result={"status": "failed", "errorCode": "disk_full"},
                        mark_sub_goal_complete=True,
                    )
                return runner.planner.process_tool_result(
                    job=job,
                    tool_name="system.ping",
                    tool_result={"status": "ok"},
                    mark_sub_goal_complete=(step_count == 2),
                )

            runner._step_executor = _failing_midway_stepper

            await runner.start_job(
                job_id=job_id,
                owner_id=owner_id,
                goal="Job that experiences mid-run error",
                max_steps=20,
            )

            for _ in range(40):
                await asyncio.sleep(0.02)
                current = await store.get(job_id)
                if current and current.state in {
                    AutonomousJobState.COMPLETED,
                    AutonomousJobState.FAILED,
                }:
                    break

            final = await store.get(job_id)
            assert final is not None
            assert final.state == AutonomousJobState.FAILED
            assert "disk_full" in str(final.error_message)
            assert final.stop_reason == "tool_failure"

            # Check that the failing sub-goal is marked FAILED
            failed_sg = next(
                (sg for sg in final.sub_goals if sg.status == SubGoalStatus.FAILED), None
            )
            assert failed_sg is not None

            await runner.shutdown()

        asyncio.run(_test())
