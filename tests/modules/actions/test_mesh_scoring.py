"""Unit tests for pure Evidence Confidence Scorer (`actions/domain/mesh_scoring.py`)."""

import uuid
from datetime import UTC, datetime, timedelta

from xuna_backend.modules.actions.domain.habit_learner import (
    DayPart,
    build_context_key,
)
from xuna_backend.modules.actions.domain.habits import HostRoutingHabit
from xuna_backend.modules.actions.domain.mesh_scoring import (
    RISK_THRESHOLDS,
    ActionTargetRequiredReason,
    CandidateUnavailableReason,
    HostScoringCandidateInput,
    MeshScoringContext,
    TargetReasonCode,
    combine_confidence,
    extract_resource_query,
    match_hosts_by_query,
    phonetic_bigram_sim,
    phonetic_fold,
    score_hosts_for_tool,
    score_live_presence,
    score_resource_match,
    score_session_continuity,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostKind,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
)
from xuna_backend.modules.hosts.domain.twin import (
    HostFocusState,
    HostPresence,
    HostResources,
    HostTwin,
)

NOW = datetime(2026, 11, 20, 19, 0, 0, tzinfo=UTC)
PUB = b"\x04" + b"\x11" * 64


def _host(
    owner_id: uuid.UUID,
    alias: str,
    *,
    platform: HostPlatform = HostPlatform.MACOS,
    kind: HostKind = HostKind.DESKTOP,
    place: str | None = None,
    nicknames: tuple[str, ...] = (),
    wol_mac: str | None = None,
) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname=alias.lower().replace(" ", "-"),
        os_version="15.0",
        agent_version="0.2.0",
        public_key=PUB,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        host_kind=kind,
        place=place,
        nicknames=nicknames,
        wol_mac=wol_mac,
    )


def _tool(
    name: str,
    *,
    family: ToolFamily,
    risk: RiskTier,
    observes: bool = False,
    permission: ToolPermissionState = ToolPermissionState.GRANTED,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Tool {name}",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "additionalProperties": False,
        },
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=family,
        risk=risk,
        observes=observes,
        permission=permission,
    )


def test_risk_threshold_strict_ordering() -> None:
    r0, r1, r2, r3 = (
        RISK_THRESHOLDS[RiskTier.R0],
        RISK_THRESHOLDS[RiskTier.R1],
        RISK_THRESHOLDS[RiskTier.R2],
        RISK_THRESHOLDS[RiskTier.R3],
    )
    assert r0[0] < r1[0] < r2[0] < r3[0]
    assert r0[1] < r1[1] < r2[1] < r3[1]


def test_fuzzy_and_phonetic_matching_h3_guarantees() -> None:
    assert phonetic_bigram_sim("", "") == 1.0
    assert phonetic_bigram_sim("a", "b") == 0.0
    assert phonetic_bigram_sim("a", "a") == 1.0
    assert phonetic_fold("Phòng Làm Việc") == phonetic_fold("fong lam viec")

    owner = uuid.uuid4()
    office = _host(owner, "Phòng Làm Việc", place="office")
    living = _host(owner, "LivingStation", platform=HostPlatform.WINDOWS, place="home")
    other_office = _host(owner, "OfficeWin", platform=HostPlatform.WINDOWS, place="office")

    # Phonetic unique match
    m_pho = match_hosts_by_query("fong lam viec", [office, living])
    assert m_pho.status == "unique"
    assert m_pho.host == office

    # < 2 char query never divides by zero and never guesses
    m_short = match_hosts_by_query("m", [office, living])
    assert m_short.status == "no_match"
    assert m_short.candidates == ()

    # Unrelated queries and pronouns return no_match with empty candidates
    m_pronoun = match_hosts_by_query("it", [office, living])
    assert m_pronoun.status == "no_match"
    assert m_pronoun.candidates == ()
    assert match_hosts_by_query("máy kia", [office, living]).status == "no_match"

    # Colliding host names must not fuzzy-match target pronouns
    may_nha = _host(owner, "Máy Nhà")
    bit_host = _host(owner, "Bit", platform=HostPlatform.WINDOWS)
    other_pc_1 = _host(owner, "Other PC 1", platform=HostPlatform.WINDOWS)
    computer_1 = _host(owner, "Computer 1", platform=HostPlatform.WINDOWS)
    macbook = _host(owner, "MacBook")
    for pronoun_q, host_pool in (
        ("máy kia", [may_nha, macbook]),
        ("máy này", [may_nha, macbook]),
        ("it", [bit_host, macbook]),
        ("other pc", [other_pc_1, macbook]),
        ("other computer", [computer_1, macbook]),
    ):
        res_p = match_hosts_by_query(pronoun_q, host_pool)
        assert res_p.status == "no_match", f"{pronoun_q!r} leaked: {res_p}"
        assert res_p.candidates == ()

    # Explicit exact alias "IT" still resolves uniquely for @IT / /may IT (§5.4.4 step 2)
    it_host = _host(owner, "IT")
    res_exact_it = match_hosts_by_query("IT", [it_host, macbook])
    assert res_exact_it.status == "unique"
    assert res_exact_it.host == it_host
    assert res_exact_it.match_source == "alias"

    # Ambiguous @office filtered strictly to office hosts (excludes LivingStation)
    m_place = match_hosts_by_query("office", [office, living, other_office])
    assert m_place.status == "ambiguous"
    assert set(m_place.candidates) == {office, other_office}

    # Fuzzy near-tie (Mac3 against Mac1/Mac2) never guesses (H3)
    mac1 = _host(owner, "Mac1")
    mac2 = _host(owner, "Mac2", platform=HostPlatform.WINDOWS)
    m_tie = match_hosts_by_query("Mac3", [mac1, mac2])
    assert m_tie.status == "ambiguous"


def test_component_scorers_resource_presence_continuity_and_combine() -> None:
    desc_app = _tool("app.launch", family=ToolFamily.APP, risk=RiskTier.R2)
    res_has = HostResources(installed_apps=("editor_x",), workspaces=(), updated_at=NOW)
    res_miss = HostResources(installed_apps=("other_y",), workspaces=(), updated_at=NOW)

    assert extract_resource_query("url.open", {"url": "https://example.com"}) is None
    assert extract_resource_query("app.launch", {"id": "Editor_X"}) == "editor_x"

    assert score_resource_match(desc_app, res_has, None, "editor_x") == 1.00
    assert score_resource_match(desc_app, res_has, None, "editor") == 0.85
    assert score_resource_match(desc_app, res_miss, None, "editor_x") == 0.25
    assert score_resource_match(desc_app, None, None, "editor_x") == 0.50
    assert score_resource_match(desc_app, res_has, None, None) == 0.70

    locked_p = HostPresence(
        idle_seconds=30,
        screen_locked=True,
        media_playing=False,
        updated_at=NOW,
        subnet_hash="a" * 32,
    )
    unlocked_p = HostPresence(
        idle_seconds=5,
        screen_locked=False,
        media_playing=True,
        updated_at=NOW,
        subnet_hash="a" * 32,
    )
    assert (
        score_live_presence(
            HostKind.DESKTOP,
            ToolFamily.APP,
            online=False,
            presence=unlocked_p,
            client_subnet_hash=None,
        )
        == 0.0
    )
    assert (
        score_live_presence(
            HostKind.DESKTOP, ToolFamily.APP, online=True, presence=None, client_subnet_hash=None
        )
        == 0.40
    )

    # Interactive lock applies 0.35x multiplier on desktop/mobile, NOT on server
    desk_locked = score_live_presence(
        HostKind.DESKTOP,
        ToolFamily.APP,
        online=True,
        presence=locked_p,
        client_subnet_hash="a" * 32,
    )
    srv_locked = score_live_presence(
        HostKind.SERVER,
        ToolFamily.APP,
        online=True,
        presence=locked_p,
        client_subnet_hash="a" * 32,
    )
    assert desk_locked < 0.30
    assert srv_locked == 1.0

    # Follow-me shifts weight from continuity (0.15->0.05) to presence (0.30->0.40)
    assert combine_confidence(0.5, 1.0, 0.5, 0.0, follow_me=True) > combine_confidence(
        0.5, 1.0, 0.5, 0.0, follow_me=False
    )

    # MEDIA family: active mediaPlaying auto-routes R1 (delta >= 0.12) over quieter silent host
    playing_media_p = HostPresence(
        idle_seconds=200,
        screen_locked=False,
        media_playing=True,
        updated_at=NOW,
        subnet_hash="a" * 32,
    )
    silent_media_p = HostPresence(
        idle_seconds=10,
        screen_locked=False,
        media_playing=False,
        updated_at=NOW,
        subnet_hash="a" * 32,
    )
    m_play = score_live_presence(
        HostKind.DESKTOP,
        ToolFamily.MEDIA,
        online=True,
        presence=playing_media_p,
        client_subnet_hash="a" * 32,
    )
    m_silent = score_live_presence(
        HostKind.DESKTOP,
        ToolFamily.MEDIA,
        online=True,
        presence=silent_media_p,
        client_subnet_hash="a" * 32,
    )
    assert 0.30 * (m_play - m_silent) >= RISK_THRESHOLDS[RiskTier.R1][1]

    owner = uuid.uuid4()
    h1 = _host(owner, "Alpha")
    assert (
        score_session_continuity(
            h1,
            has_active_cu_lane=False,
            is_focused=False,
            last_target_alias="Alpha",
            last_target_at=NOW - timedelta(minutes=2),
            now=NOW,
            default_host_id=None,
        )
        == 0.80
    )


def test_classification_precedence_no_host_offline_grant_denied_and_locked_r2() -> None:
    owner = uuid.uuid4()
    h_a = _host(owner, "NodeA", wol_mac="AA:BB:CC:DD:EE:01")
    h_b = _host(owner, "NodeB", platform=HostPlatform.WINDOWS)
    app_desc = _tool("app.launch", family=ToolFamily.APP, risk=RiskTier.R2)
    obs_desc = _tool("ui.inspect", family=ToolFamily.UI, risk=RiskTier.R1, observes=True)
    m_a = Manifest(host_id=h_a.host_id, tools=(app_desc, obs_desc), updated_at=NOW)
    m_b = Manifest(host_id=h_b.host_id, tools=(app_desc,), updated_at=NOW)
    g_a = (
        Grant(h_a.host_id, ToolFamily.APP, RiskTier.R2, True, NOW),
        Grant(h_a.host_id, ToolFamily.UI, RiskTier.R1, True, NOW),
    )

    # 1. no_host when zero hosts advertise tool
    dec_no = score_hosts_for_tool(
        [HostScoringCandidateInput(host=h_a, online=True, manifest=m_a, grants=g_a)],
        MeshScoringContext(
            tool_name="media.play_pause",
            arguments={},
            now=NOW,
            host_screen_consent=True,
        ),
    )
    assert dec_no.action_required_reason == ActionTargetRequiredReason.NO_HOST

    # 2. host_offline when all capable hosts are offline (preserves wolAvailable)
    dec_off = score_hosts_for_tool(
        [
            HostScoringCandidateInput(host=h_a, online=False, manifest=m_a, grants=g_a),
            HostScoringCandidateInput(host=h_b, online=False, manifest=m_b, grants=()),
        ],
        MeshScoringContext(
            tool_name="app.launch",
            arguments={"id": "x"},
            now=NOW,
            host_screen_consent=True,
        ),
    )
    assert dec_off.action_required_reason == ActionTargetRequiredReason.HOST_OFFLINE
    assert dec_off.candidates[0].wol_available is True

    # 3. grant_denied when >= 1 online capable host is ungranted (and marks offline sibling)
    dec_gd = score_hosts_for_tool(
        [
            HostScoringCandidateInput(host=h_a, online=False, manifest=m_a, grants=g_a),
            HostScoringCandidateInput(host=h_b, online=True, manifest=m_b, grants=()),
        ],
        MeshScoringContext(
            tool_name="app.launch",
            arguments={"id": "x"},
            now=NOW,
            host_screen_consent=True,
        ),
    )
    assert dec_gd.action_required_reason == ActionTargetRequiredReason.GRANT_DENIED
    reasons = {c.host.host_id: c.unavailable_reason for c in dec_gd.candidates}
    assert reasons[h_a.host_id] == CandidateUnavailableReason.HOST_OFFLINE
    assert reasons[h_b.host_id] == CandidateUnavailableReason.GRANT_DENIED

    # 3b. U5 consent missing -> u5_consent_required
    dec_u5 = score_hosts_for_tool(
        [HostScoringCandidateInput(host=h_a, online=True, manifest=m_a, grants=g_a)],
        MeshScoringContext(
            tool_name="ui.inspect",
            arguments={},
            now=NOW,
            host_screen_consent=False,
        ),
    )
    assert dec_u5.action_required_reason == ActionTargetRequiredReason.GRANT_DENIED
    assert dec_u5.candidates[0].unavailable_reason == CandidateUnavailableReason.U5_CONSENT_REQUIRED

    # 4. Single locked interactive desktop host on R2 -> 1-candidate ask_target
    locked_twin = HostTwin(
        host_id=h_a.host_id,
        owner_id=owner,
        presence=HostPresence(
            idle_seconds=200,
            screen_locked=True,
            media_playing=False,
            updated_at=NOW,
        ),
    )
    dec_lk = score_hosts_for_tool(
        [
            HostScoringCandidateInput(
                host=h_a, online=True, manifest=m_a, grants=g_a, twin=locked_twin
            )
        ],
        MeshScoringContext(
            tool_name="app.launch",
            arguments={"id": "x"},
            now=NOW,
            host_screen_consent=True,
        ),
    )
    assert dec_lk.resolved is False
    assert dec_lk.action_required_reason == ActionTargetRequiredReason.ASK_TARGET
    assert len(dec_lk.candidates) == 1

    # 5. Same host unlocked -> auto-resolves as unique_capable_host
    unlocked_twin = HostTwin(
        host_id=h_a.host_id,
        owner_id=owner,
        presence=HostPresence(
            idle_seconds=10,
            screen_locked=False,
            media_playing=False,
            updated_at=NOW,
        ),
    )
    dec_ok = score_hosts_for_tool(
        [
            HostScoringCandidateInput(
                host=h_a, online=True, manifest=m_a, grants=g_a, twin=unlocked_twin
            )
        ],
        MeshScoringContext(
            tool_name="app.launch",
            arguments={"id": "x"},
            now=NOW,
            host_screen_consent=True,
        ),
    )
    assert dec_ok.resolved is True
    assert dec_ok.target_reason.reason_code == TargetReasonCode.UNIQUE_CAPABLE_HOST


def test_multi_host_scoring_with_habits_and_follow_me() -> None:
    owner = uuid.uuid4()
    h_mac = _host(owner, "Mac")
    h_win = _host(owner, "Win", platform=HostPlatform.WINDOWS)
    media_desc = _tool("media.play_pause", family=ToolFamily.MEDIA, risk=RiskTier.R1)
    m_mac = Manifest(host_id=h_mac.host_id, tools=(media_desc,), updated_at=NOW)
    m_win = Manifest(host_id=h_win.host_id, tools=(media_desc,), updated_at=NOW)
    g_mac = (Grant(h_mac.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),)
    g_win = (Grant(h_win.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),)

    ctx_key = build_context_key(ToolFamily.MEDIA, None, DayPart.EVENING)
    habit_mac = HostRoutingHabit(
        habit_id=uuid.uuid4(),
        owner_id=owner,
        pet_id="pet-1",
        context_key=ctx_key,
        host_id=h_mac.host_id,
        positive_score=5.0,
        negative_score=0.0,
        sample_count=10,
        created_at=NOW,
        updated_at=NOW,
    )
    twin_mac = HostTwin(
        host_id=h_mac.host_id,
        owner_id=owner,
        presence=HostPresence(
            idle_seconds=5,
            screen_locked=False,
            media_playing=True,
            updated_at=NOW,
            subnet_hash="1" * 32,
        ),
    )
    twin_win = HostTwin(
        host_id=h_win.host_id,
        owner_id=owner,
        presence=HostPresence(
            idle_seconds=300,
            screen_locked=False,
            media_playing=False,
            updated_at=NOW,
            subnet_hash="2" * 32,
        ),
    )
    dec = score_hosts_for_tool(
        [
            HostScoringCandidateInput(
                host=h_mac, online=True, manifest=m_mac, grants=g_mac, twin=twin_mac
            ),
            HostScoringCandidateInput(
                host=h_win, online=True, manifest=m_win, grants=g_win, twin=twin_win
            ),
        ],
        MeshScoringContext(
            tool_name="media.play_pause",
            arguments={},
            now=NOW,
            host_screen_consent=True,
            focus_state=HostFocusState(
                owner_id=owner,
                focused_host_id=None,
                follow_me=True,
                updated_at=NOW,
                source="app",
            ),
            client_subnet_hash="1" * 32,
            habits_by_key_host={(ctx_key, h_mac.host_id): habit_mac},
        ),
    )
    assert dec.resolved is True
    assert dec.host == h_mac
    assert dec.target_reason.reason_code == TargetReasonCode.FOLLOW_ME_ACTIVE


def test_mesh_scoring_hardening_control_chars_offline_manifest_and_slicing() -> None:
    owner = uuid.uuid4()
    app_desc = _tool("app.launch", family=ToolFamily.APP, risk=RiskTier.R2)

    # 1. Control chars in resource query return None and do not crash scoring
    assert extract_resource_query("app.launch", {"id": "app\x00evil"}) is None
    assert extract_resource_query("app.launch", {"id": "app\nevil"}) is None
    h_online = _host(owner, "Alpha")
    m_online = Manifest(host_id=h_online.host_id, tools=(app_desc,), updated_at=NOW)
    g_online = (Grant(h_online.host_id, ToolFamily.APP, RiskTier.R2, True, NOW),)
    dec_ctrl = score_hosts_for_tool(
        [HostScoringCandidateInput(host=h_online, online=True, manifest=m_online, grants=g_online)],
        MeshScoringContext(
            tool_name="app.launch",
            arguments={"id": "app\x00evil"},
            now=NOW,
            host_screen_consent=True,
        ),
    )
    assert dec_ctrl.resolved is True

    # 2. Explicit offline host with manifest=None reports HOST_OFFLINE, not UNSUPPORTED
    h_off_no_manifest = _host(owner, "OfflineBox", platform=HostPlatform.WINDOWS)
    dec_off = score_hosts_for_tool(
        [HostScoringCandidateInput(host=h_off_no_manifest, online=False, manifest=None, grants=())],
        MeshScoringContext(
            tool_name="app.launch",
            arguments={"id": "x"},
            now=NOW,
            host_screen_consent=True,
            explicit_requested_host_id=h_off_no_manifest.host_id,
        ),
    )
    assert dec_off.resolved is False
    assert dec_off.action_required_reason == ActionTargetRequiredReason.HOST_OFFLINE
    assert dec_off.candidates[0].unavailable_reason == CandidateUnavailableReason.HOST_OFFLINE

    # 3. Online denied host sorts ahead of 5 offline hosts before [:5] slicing
    offline_hosts = [_host(owner, f"A_Off_{i}") for i in range(5)]
    z_denied = _host(owner, "Z_OnlineDenied", platform=HostPlatform.WINDOWS)
    inputs = [
        HostScoringCandidateInput(
            host=h,
            online=False,
            manifest=Manifest(host_id=h.host_id, tools=(app_desc,), updated_at=NOW),
            grants=(),
        )
        for h in offline_hosts
    ] + [
        HostScoringCandidateInput(
            host=z_denied,
            online=True,
            manifest=Manifest(host_id=z_denied.host_id, tools=(app_desc,), updated_at=NOW),
            grants=(),
        )
    ]
    dec_slice = score_hosts_for_tool(
        inputs,
        MeshScoringContext(
            tool_name="app.launch",
            arguments={"id": "x"},
            now=NOW,
            host_screen_consent=True,
        ),
    )
    assert dec_slice.action_required_reason == ActionTargetRequiredReason.GRANT_DENIED
    assert len(dec_slice.candidates) == 5
    assert dec_slice.candidates[0].host == z_denied
    assert dec_slice.candidates[0].unavailable_reason == CandidateUnavailableReason.GRANT_DENIED
