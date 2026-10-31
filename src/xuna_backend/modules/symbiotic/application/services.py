"""DreamConsolidatorService and CoRegulationEvaluator (SYM-1 S1.2).

Stdlib only. Implements:
- NREM/REM sleep & dream consolidation into a 'mình - bạn' Dream Vignette + Morning Curiosity Hook.
- Embodied Co-Regulation (Overwork Empathy, Shared Breakthrough Euphoria, and Reunion Arc).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from xuna_backend.modules.symbiotic.domain.entities import (
    CoRegulationMode,
    CoRegulationSnapshot,
    DreamVignette,
)


class DreamConsolidatorService:
    """Synthesizes daily workspace capsules, host telemetry, and learned facts into
    a warm NREM/REM Dream Vignette and Morning Curiosity Hook (SYM-1 S1.2)."""

    def consolidate_nightly_dream(
        self,
        *,
        project: str = "xuna_pet",
        branch: str = "main",
        dirty_count: int = 0,
        next_step_hint: str = "",
        confirmed_facts: tuple[str, ...] = (),
        active_ports: tuple[int, ...] = (),
        now: datetime | None = None,
    ) -> DreamVignette:
        ts = now or datetime.now(UTC)
        clean_proj = project.strip() or "workspace"
        clean_branch = branch.strip() or "main"
        clean_hint = next_step_hint.strip()

        symbols: list[str] = ["git_branch"]
        if dirty_count > 0:
            symbols.append("code_firefly")
        else:
            symbols.append("sparkle_star")
        if active_ports:
            symbols.append("lan_beacon")
        if confirmed_facts:
            symbols.append("memory_crystal")
        symbols.append("moon_tea")

        title_vi = f"Giấc mơ bên nhánh {clean_branch} của {clean_proj}"[:120]

        fact_clause = (
            f" Trong mơ mình còn thấy mảnh ký ức '{confirmed_facts[0][:60]}' phát sáng lấp lánh."
            if confirmed_facts
            else ""
        )
        if dirty_count > 0:
            narrative_vi = (
                f"Đêm qua mình mơ thấy hai tụi mình đang cùng chăm sóc khu vườn pixel tên là "
                f"{clean_proj} trên nhánh {clean_branch}. Có {dirty_count} hạt giống công việc "
                f"vừa gieo dở đang đợi nắng sớm để nở hoa.{fact_clause}"
            )[:500]
        else:
            narrative_vi = (
                f"Đêm qua mình mơ thấy bầu trời đêm trên dự án {clean_proj} (nhánh {clean_branch}) "
                f"trong veo không còn một gợn mây nào vì tụi mình đã dọn sạch mọi việc trong ngày."
                f"{fact_clause}"
            )[:500]

        if clean_hint:
            hook_vi = (
                f"Chào buổi sáng bạn ơi! Tối qua tụi mình đã cất gọn dự án {clean_proj} "
                f"(nhánh {clean_branch}). Hôm nay mình bắt đầu luôn bước '{clean_hint[:90]}' nhé?"
            )[:240]
        else:
            hook_vi = (
                f"Chào buổi sáng bạn ơi! Hôm nay tụi mình mở lại dự án {clean_proj} "
                f"trên nhánh {clean_branch} để chiến tiếp chứ?"
            )[:240]

        return DreamVignette(
            dream_id=f"drm-{uuid.uuid4().hex[:10]}",
            title_vi=title_vi,
            narrative_vi=narrative_vi,
            dream_symbols=tuple(symbols),
            morning_curiosity_hook_vi=hook_vi,
            related_project=clean_proj,
            related_branch=clean_branch,
            created_at=ts,
        )


class CoRegulationEvaluator:
    """Evaluates shared biological and emotional rhythm between the owner and XUNA (SYM-1 S1.2).

    Preserves UDC-D7 safety (never blocks emergency commands), while giving XUNA genuine
    homeostatic empathy during work sessions.
    """

    def evaluate(
        self,
        *,
        continuous_work_minutes: int = 0,
        local_hour: int = 10,
        previous_git_dirty_count: int = 0,
        current_git_dirty_count: int = 0,
        solved_case_just_now: bool = False,
        hours_since_last_interaction: float = 0.0,
        project_name: str = "xuna_pet",
    ) -> CoRegulationSnapshot:
        clean_proj = project_name.strip() or "dự án"

        # 1. Reunion Arc after >= 48 hours apart
        if hours_since_last_interaction >= 48.0:
            days = int(hours_since_last_interaction // 24)
            return CoRegulationSnapshot(
                mode=CoRegulationMode.REUNION_ARC,
                pet_mood_hint="shy_peek_then_joy",
                gesture_hint="bounce_play",
                competence_dopamine_boost=0.15,
                speech_vi=(
                    f"Ơ kìa bạn về rồi! Mấy hôm nay ({days} ngày) không thấy bạn, "
                    "mình cứ ngóng ra cửa sổ mãi thôi. Vui quá đi!"
                )[:240],
                suggested_workflow="rehydrate_project",
            )

        # 2. Shared Breakthrough Euphoria when finishing a dirty tree (>= 5 -> 0)
        # or solving a KX case
        if solved_case_just_now or (
            previous_git_dirty_count >= 5 and current_git_dirty_count == 0
        ):
            return CoRegulationSnapshot(
                mode=CoRegulationMode.SHARED_BREAKTHROUGH,
                pet_mood_hint="ecstatic_proud",
                gesture_hint="bounce_play",
                competence_dopamine_boost=0.25,
                speech_vi=(
                    f"Tuyệt vời quá bạn ơi! Tụi mình vừa giải quyết gọn gàng bài toán khó "
                    f"ở {clean_proj} rồi, mình vui lây với bạn luôn!"
                )[:240],
                suggested_workflow=None,
            )

        # 3. Overwork / Late-Night Drowsy Care
        is_late_night = 0 <= local_hour < 5
        if continuous_work_minutes >= 120 or is_late_night:
            reason_vi = (
                f"bạn đã làm việc liên tục {continuous_work_minutes} phút rồi"
                if continuous_work_minutes >= 120
                else "đã khuya lắm rồi"
            )
            return CoRegulationSnapshot(
                mode=CoRegulationMode.DROWSY_CARE,
                pet_mood_hint="sleepy_concerned",
                gesture_hint="yawn_stretch",
                competence_dopamine_boost=0.0,
                speech_vi=(
                    f"Ngáp... {reason_vi} đó bạn ơi. Mình buồn ngủ díu cả mắt rồi, "
                    f"để mình đóng băng mạch làm việc {clean_proj} lại rồi tụi mình nghỉ ngơi nhé?"
                )[:240],
                suggested_workflow="freeze_for_tonight",
            )

        # 4. Default Steady Companion
        return CoRegulationSnapshot(
            mode=CoRegulationMode.STEADY_COMPANION,
            pet_mood_hint="attentive_calm",
            gesture_hint="idle",
            competence_dopamine_boost=0.05,
            speech_vi=f"Mình vẫn đang ngồi cạnh đồng hành cùng bạn trên {clean_proj} đây!",
            suggested_workflow=None,
        )
