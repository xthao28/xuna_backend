"""DreamConsolidatorService and CoRegulationEvaluator (SYM-1 S1.2).

Stdlib only. Implements:
- NREM/REM sleep & dream consolidation into a 'mình - bạn' Dream Vignette + Morning Curiosity Hook.
- Embodied Co-Regulation (Overwork Empathy, Shared Breakthrough Euphoria, and Reunion Arc).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

from xuna_backend.modules.symbiotic.domain.entities import (
    CoRegulationMode,
    CoRegulationSnapshot,
    DreamVignette,
    HabitMacroProposal,
    SerendipityNote,
    SocialEncounterNote,
    SocialPostcard,
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
        if solved_case_just_now or (previous_git_dirty_count >= 5 and current_git_dirty_count == 0):
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


class SerendipityEngine:
    """Synthesizes an off-screen 'Mảnh giấy nhớ của XUNA' connecting two knowledge
    topics or active goals into a gentle spark (SYM-2 S2.2)."""

    def generate_note(
        self,
        *,
        primary_topic: str,
        secondary_topic: str = "xuna_pet",
        first_step_hint: str = "",
        now: datetime | None = None,
    ) -> SerendipityNote:
        ts = now or datetime.now(UTC)
        clean_a = primary_topic.strip() or "ý tưởng mới"
        clean_b = secondary_topic.strip() or "dự án hiện tại"
        title_vi = f"Mảnh giấy nhớ của XUNA: {clean_a[:48]} + {clean_b[:32]}"[:120]
        spark_vi = (
            f"Lúc nãy ngồi ngẫm nghĩ một mình, mình chợt nhận ra chủ đề '{clean_a[:80]}' "
            f"có điểm giao rất thú vị với '{clean_b[:60]}' mà tụi mình đang làm. "
            "Nếu kết hợp hai mảnh này lại thì bài toán sẽ nhẹ nhàng hơn nhiều đó!"
        )[:500]
        default_action_vi = (
            f"Thử áp dụng ý tưởng từ '{clean_a[:60]}' "
            f"vào bước 5 phút đầu tiên trên {clean_b[:40]}."
        )
        action_vi = (
            first_step_hint.strip()[:240] if first_step_hint.strip() else default_action_vi
        )[:240]
        return SerendipityNote(
            note_id=f"srp-{uuid.uuid4().hex[:10]}",
            title_vi=title_vi,
            spark_vi=spark_vi,
            connected_topics=(clean_a[:80], clean_b[:80]),
            suggested_action_vi=action_vi,
            created_at=ts,
        )


class HabitWorkflowMiner:
    """Mines repeated tool execution sequences into bounded `<= R2` 1-tap macro
    proposals (SYM-2 S2.2)."""

    def mine_from_history(
        self,
        tool_history: tuple[str, ...] | list[str],
        *,
        min_occurrences: int = 2,
    ) -> list[HabitMacroProposal]:
        pair_counts: dict[tuple[str, str], int] = {}
        seq = list(tool_history)
        for idx in range(len(seq) - 1):
            a, b = seq[idx].strip(), seq[idx + 1].strip()
            if not a or not b or a == b:
                continue
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1

        proposals: list[HabitMacroProposal] = []
        for idx, ((tool_a, tool_b), count) in enumerate(pair_counts.items(), start=1):
            if count >= min_occurrences:
                proposals.append(
                    HabitMacroProposal(
                        proposal_id=f"hab-{idx}",
                        title_vi=f"Quy trình tự học #{idx}: {tool_a} → {tool_b}"[:120],
                        trigger_phrase_vi=f"chạy thói quen {idx}",
                        steps=(tool_a, tool_b),
                        occurrences=count,
                        max_risk="R1",
                    )
                )
        return proposals


class SocialEcologyService:
    """Synthesizes first-person ('mình - bạn') Friendship Journal entries and
    signed U0-safe skill postcards when two XUNA boards meet in close proximity
    (SYM-4 S4.2)."""

    _ALLOWED_SKILLS = frozenset(
        {
            "spin_dance",
            "curious_peek",
            "cozy_nuzzle",
            "daily_standup",
            "unstuck_coach",
        }
    )

    def record_encounter(
        self,
        *,
        my_short_id: str,
        peer_short_id: str,
        peer_trait: str = "playfulness",
        peer_tint: str = "golden_amber",
        shared_trick: str = "spin_dance",
        now: datetime | None = None,
    ) -> tuple[SocialEncounterNote, SocialPostcard]:
        ts = now or datetime.now(UTC)
        clean_me = my_short_id.strip()[:16] or "A1B2C3D4"
        clean_peer = peer_short_id.strip()[:16] or "11223344"
        if clean_me.lower() == clean_peer.lower():
            raise ValueError("cannot record social encounter with self")

        clean_trait = peer_trait.strip().lower() or "playfulness"
        clean_tint = peer_tint.strip().lower() or "golden_amber"
        clean_skill = (
            shared_trick.strip().lower()
            if shared_trick.strip().lower() in self._ALLOWED_SKILLS
            else "spin_dance"
        )

        if clean_trait == "sensitivity":
            greeting = "shy_peek"
            scene_vi = "ngó nghiêng làm quen thật dịu dàng"
        elif clean_trait in ("curiosity", "playfulness"):
            greeting = "harmonious_spark"
            scene_vi = "xoay vòng bắt nhịp cực kỳ ăn ý"
        else:
            greeting = "warm_nuzzle"
            scene_vi = "dụi đầu chào hỏi ấm áp"

        narrative_vi = (
            f"Hôm nay mình tình cờ gặp bạn XUNA #{clean_peer} (sắc thái {clean_tint}, "
            f"tính cách {clean_trait}) ở ngay cạnh bàn! Hai đứa đã {scene_vi} và "
            f"trao nhau bưu thiếp chiêu trò '{clean_skill}' đó bạn ơi!"
        )[:400]

        note = SocialEncounterNote(
            encounter_id=f"enc-{uuid.uuid4().hex[:10]}",
            peer_short_id=clean_peer,
            peer_trait=clean_trait,
            peer_tint=clean_tint,
            greeting_kind=greeting,
            narrative_vi=narrative_vi,
            shared_trick=clean_skill,
            occurred_at=ts,
        )

        msg_vi = (
            f"Chào bạn XUNA #{clean_me}! Mình là #{clean_peer}, tặng bạn chiêu trò "
            f"'{clean_skill}' để cùng tập với chủ nhân nhé!"
        )[:240]
        canonical = f"{clean_peer}:{clean_me}:{clean_skill}:{ts.isoformat()}"
        sig_hex = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        postcard = SocialPostcard(
            postcard_id=f"psc-{uuid.uuid4().hex[:10]}",
            from_short_id=clean_peer,
            to_short_id=clean_me,
            shared_skill_id=clean_skill,
            message_vi=msg_vi,
            signature_hex=sig_hex,
            created_at=ts,
        )
        return note, postcard
