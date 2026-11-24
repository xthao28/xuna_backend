"""Tool catalog builder for LLM function calling (UDC-4 §5.2, UDC-7 S7.1).

The catalog is deterministic and independent of the utterance: eligible host tools are
ordered by `_PRIORITY_ORDER` (core primitives first). When they all fit, every one is sent
directly. When they do not, the first ones are sent directly and the rest stay reachable
through the `tool_search` / `tool_describe` / `tool_invoke` meta tools
(`tool_discovery`), so no granted tool is ever silently dropped.
"""

import copy
import uuid
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime

from xuna_backend.modules.actions.application.tool_discovery import (
    MESH_META_TOOL_COUNT,
    MESH_PIPELINE_TOOL_COUNT,
    META_TOOL_COUNT,
    EligibleHostTool,
    build_mesh_pipeline_tool_specs,
    build_mesh_tool_specs,
    build_meta_tool_specs,
)
from xuna_backend.modules.actions.domain.schema_validator import (
    SchemaDefinitionError,
    SchemaValidator,
)
from xuna_backend.modules.actions.domain.tool_names import (
    AUTONOMOUS_JOB_DESCRIPTION,
    AUTONOMOUS_JOB_FUNCTION_NAME,
    AUTONOMOUS_JOB_SCHEMA,
    BOARD_FUNCTION_NAME,
    BOARD_MEDIA_KEY_DESCRIPTION,
    BOARD_MEDIA_KEY_SCHEMA,
    WORKFLOW_TEMPLATE_DESCRIPTION,
    WORKFLOW_TEMPLATE_FUNCTION_NAME,
    WORKFLOW_TEMPLATE_SCHEMA,
    is_forbidden_tool_name,
    to_function_name,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.cognition.domain.types import ToolSpec
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostId,
    HostStatus,
    Manifest,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
)

MAX_CATALOG_TOOLS = 36
CAP_HOST_ACTIONS_V1 = "host_actions_v1"
CAP_BOARD_HID_V1 = "board_hid_v1"
CAP_WORKFLOW_TEMPLATES_V1 = "workflow_templates_v1"

_U5_REQUIRED_FAMILIES: frozenset[ToolFamily] = frozenset(
    {
        ToolFamily.CU,
        ToolFamily.SCREEN,
        ToolFamily.UI,
        ToolFamily.INPUT,
    }
)

# Core primitives first: these are the tools sent directly when the catalog must defer.
_PRIORITY_ORDER: tuple[str, ...] = (
    # Slots 0..29: guaranteed direct even at worst-case reserved=3 + defer=True (36 - 3 - 3 = 30)
    # Tier 2 live context & Tier 1 OS Deep Open (0..4)
    "window.active_context",
    "app.launch",
    "app.list_allowed",
    "url.open",
    "finder.reveal",
    # Tier 0 headless OS native, media, workspace, web CDP, MCP, clipboard (5..19)
    "media.play_pause",
    "media.next",
    "media.volume_set",
    "media.now_playing",
    "notes.create",
    "notes.search",
    "reminders.create",
    "calendar.list_events",
    "workspace.open",
    "workspace.git_summary",
    "web.navigate",
    "web.snapshot",
    "mcp.list_servers",
    "mcp.call",
    "clipboard.write",
    # Tier 2/3 core CU, UI, Input (20..29) — adjacency invariants preserved!
    "cu.session_start",
    "ui.inspect",
    "ui.find",
    "ui.observe_hybrid",
    "input.click",
    "input.drag",
    "input.type_text",
    "input.key",
    "input.scroll",
    "cu.session_end",
    # Slots 30..31: protocol_async verification/cancellation (direct when reserved <= 1)
    "mcp.task_result",
    "mcp.task_cancel",
    # Slots 32..40: direct when reserved == 0 or fewer tools, else 1-hop tool_search
    "cu.session_extend",
    "media.previous",
    "media.play_search",
    "clipboard.read",
    "notes.append",
    "calendar.create_event",
    "shortcuts.run",
    "mail.create_draft",
    "selection.read",
    # Remaining web / browser / notifications
    "web.click",
    "web.fill",
    "web.select",
    "web.extract",
    "web.wait_for",
    "browser.read_page",
    "browser.current_url",
    "notifications.list",
    "notifications.action",
    # Mobile UI automation (direct when desktop UI tools are absent)
    "ui.inspect_tree",
    "ui.observe",
    "ui.act",
    "ui.wait_for",
    "ui.tap_text",
    "ui.set_text",
    "ui.scroll",
    "ui.long_press",
    "ui.swipe_drag",
    "app.open_scheme",
    # Workspace dev & Forge
    "workspace.coding_agent",
    "forge.synthesize_skill",
    "forge.run_skill",
    "forge.list_and_manage",
    # Remaining host tools
    "host.status",
    "workspace.watch_task",
    "beam.transfer",
    "workflow.batch_steps",
    "action.undo",
    "tv.control",
    "tv.launch_app",
    "tv.cast_media",
    "tv.list_devices",
    "smarthome.control",
    "smarthome.list_entities",
    "dev.ports",
    "system.vitals",
    "workspace.capture_capsule",
    "workspace.restore_capsule",
    "workspace.list_capsules",
    "pocket.push_item",
    "pocket.list_items",
    "pocket.transform_item",
    "watchdog.watch_process_or_port",
    "watchdog.watch_resource_threshold",
    "watchdog.list_and_clear",
    "coach.decompose_overwhelm",
    "habit.propose_workflow",
    "teach.record_session",
    "forge.extract_invoices_to_sheet",
    "skill.markdown_bridge",
    "skill.record_correction",
    "skill.workshop_sync",
    "skill.chain_and_test",
    "memory.markdown_vault",
    "browser.semantic_snapshot",
    "heartbeat.configure_and_tick",
    "canvas.present",
    "pipeline.execute_universal",
    "host.lock_screen",
    "display.wake",
    "workflow.run",
    "host.notify",
    "workflow.list",
    "ui.back",
    "ui.home",
    "ui.recents",
)


def is_tool_granted(
    descriptor: ToolDescriptor,
    grants: Sequence[Grant],
    now: datetime | None = None,
    *,
    host_screen_consent: bool = True,
) -> bool:
    """Returns True iff the tool is supported on the host, U5 (`host_screen`) consent is
    granted when required (`observes == True` or computer-use families), and the host has
    an active (enabled, non-expired, non-suspended) grant for `descriptor.family` whose
    `max_risk` covers `descriptor.risk`."""
    if descriptor.permission == ToolPermissionState.UNSUPPORTED:
        return False
    if (
        descriptor.observes or descriptor.family in _U5_REQUIRED_FAMILIES
    ) and not host_screen_consent:
        return False
    check_now = now or datetime.now(UTC)
    for grant in grants:
        if grant.family == descriptor.family:
            return grant.is_active(check_now) and grant.max_risk.allows(descriptor.risk)
    return False


USAGE_WINDOW_EVENTS = 300


def tool_usage_counts(events: Iterable[AuditEvent]) -> dict[str, int]:
    """Counts the owner's successful host tool calls from content-free audit events.

    Only `host.tool_invoked` events with `status == "ok"` count, so tools reached through
    `tool_invoke` are promoted into the direct catalog once the owner actually uses them.
    """
    counts: dict[str, int] = {}
    for event in events:
        if event.action != AuditAction.HOST_TOOL_INVOKED:
            continue
        tool = event.metadata.get("tool")
        if isinstance(tool, str) and event.metadata.get("status") == "ok":
            counts[tool] = counts.get(tool, 0) + 1
    return counts


def _ordering_key(tool_name: str, usage: Mapping[str, int] | None) -> tuple[int, int, str]:
    """Owner usage first (most used first), then the default `_PRIORITY_ORDER`."""
    used = usage.get(tool_name, 0) if usage is not None else 0
    try:
        default_idx = _PRIORITY_ORDER.index(tool_name)
    except ValueError:
        default_idx = len(_PRIORITY_ORDER)
    return (-used, default_idx, tool_name)


def collect_eligible_host_tools(
    *,
    hosts: Sequence[Host],
    manifests_by_host: Mapping[HostId, Manifest],
    grants_by_host: Mapping[HostId, Sequence[Grant]],
    online_host_ids: Collection[str],
    host_screen_consent: bool,
    now: datetime | None = None,
    usage: Mapping[str, int] | None = None,
    pinned_host_id: HostId | None = None,
) -> dict[str, EligibleHostTool]:
    """Returns every host tool the owner may use this turn, most used first (`usage`, from
    `tool_usage_counts`), then in the default `_PRIORITY_ORDER`.

    A tool is eligible on a host when the host is active and online, the tool is not
    forbidden, `is_tool_granted` passes (grant, risk, U5 consent) and its schema and
    function name are valid. When `pinned_host_id` is set, only that host is included.
    """
    check_now = now or datetime.now(UTC)
    by_tool: dict[str, tuple[ToolDescriptor, list[str]]] = {}
    active_online_hosts = [
        h
        for h in hosts
        if h.status == HostStatus.ACTIVE
        and str(h.host_id) in online_host_ids
        and (pinned_host_id is None or h.host_id == pinned_host_id)
    ]
    active_online_hosts.sort(key=lambda h: str(h.alias).lower())

    for host in active_online_hosts:
        manifest = manifests_by_host.get(host.host_id)
        if manifest is None:
            continue
        grants = grants_by_host.get(host.host_id, ())
        for descriptor in manifest.tools:
            if is_forbidden_tool_name(descriptor.name):
                continue
            if not is_tool_granted(
                descriptor,
                grants,
                check_now,
                host_screen_consent=host_screen_consent,
            ):
                continue
            try:
                SchemaValidator.validate_schema_definition(descriptor.input_schema)
            except SchemaDefinitionError:
                continue
            try:
                to_function_name(descriptor.name)
            except ValueError:
                continue
            alias_str = str(host.alias)
            if descriptor.name not in by_tool:
                by_tool[descriptor.name] = (descriptor, [alias_str])
            else:
                _, aliases = by_tool[descriptor.name]
                if alias_str not in aliases:
                    aliases.append(alias_str)

    if "ui.inspect" in by_tool:
        by_tool.pop("ui.accessibility_tree", None)

    return {
        name: EligibleHostTool(descriptor=by_tool[name][0], aliases=tuple(by_tool[name][1]))
        for name in sorted(by_tool, key=lambda n: _ordering_key(n, usage))
    }


def _host_tool_spec(tool_name: str, item: EligibleHostTool) -> ToolSpec:
    schema_copy = copy.deepcopy(item.descriptor.input_schema)
    props_raw = schema_copy.get("properties")
    props = dict(props_raw) if isinstance(props_raw, dict) else {}
    props["target"] = {
        "type": "string",
        "enum": list(item.aliases),
    }
    if tool_name == "input.click":
        if "xNorm" not in props:
            props["xNorm"] = {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Normalized X coordinate [0.0, 1.0] relative to target window",
            }
        if "yNorm" not in props:
            props["yNorm"] = {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Normalized Y coordinate [0.0, 1.0] relative to target window",
            }
    schema_copy["properties"] = props
    return ToolSpec(
        name=to_function_name(tool_name),
        description=item.descriptor.description,
        parameters=schema_copy,
    )


def build_tool_catalog(
    *,
    hosts: Sequence[Host],
    manifests_by_host: Mapping[HostId, Manifest],
    grants_by_host: Mapping[HostId, Sequence[Grant]],
    online_host_ids: Collection[str],
    client_caps: Collection[str],
    board_available: bool = False,
    host_screen_consent: bool = False,
    include_workflow_template: bool = False,
    include_autonomous_job: bool = False,
    now: datetime | None = None,
    usage: Mapping[str, int] | None = None,
    pinned_host_id: HostId | None = None,
    include_mesh_tools: bool = False,
) -> tuple[ToolSpec, ...]:
    """Builds at most `MAX_CATALOG_TOOLS` (`36`) `ToolSpec` entries for `ChatRequest.tools`.

    - Gated behind `client_caps` (`host_actions_v1` for host tools, `board_hid_v1` for board HID).
    - Eligibility comes from `collect_eligible_host_tools`; order is the owner's own tool
      usage, then `_PRIORITY_ORDER` as the cold-start default.
    - When `pinned_host_id` is set, only that host's tools are included.
    - When `include_mesh_tools` is enabled with `pinned_host_id is None` and `>= 2` active hosts
      (at least one online), appends `mesh__status` and `mesh__find_resource`.
    """
    include_host_tools = CAP_HOST_ACTIONS_V1 in client_caps
    include_board_tool = CAP_BOARD_HID_V1 in client_caps and board_available
    include_wf_template = include_host_tools and (
        include_workflow_template or CAP_WORKFLOW_TEMPLATES_V1 in client_caps
    )
    if not include_host_tools and not include_board_tool:
        return ()

    eligible: dict[str, EligibleHostTool] = {}
    if include_host_tools:
        eligible = collect_eligible_host_tools(
            hosts=hosts,
            manifests_by_host=manifests_by_host,
            grants_by_host=grants_by_host,
            online_host_ids=online_host_ids,
            host_screen_consent=host_screen_consent,
            now=now,
            usage=usage,
            pinned_host_id=pinned_host_id,
        )

    active_host_count = sum(1 for h in hosts if h.status == HostStatus.ACTIVE)
    effective_mesh_tools = (
        include_host_tools
        and include_mesh_tools
        and pinned_host_id is None
        and bool(online_host_ids)
        and active_host_count >= 2
    )
    effective_mesh_pipelines = (
        include_host_tools
        and include_mesh_tools
        and bool(online_host_ids)
        and active_host_count >= 2
    )
    effective_auto_job = (
        include_host_tools and include_autonomous_job and ("cu.session_start" in eligible)
    )
    reserved = (
        (1 if include_board_tool else 0)
        + (1 if include_wf_template else 0)
        + (1 if effective_auto_job else 0)
        + (MESH_META_TOOL_COUNT if effective_mesh_tools else 0)
        + (MESH_PIPELINE_TOOL_COUNT if effective_mesh_pipelines else 0)
    )
    host_limit = max(MAX_CATALOG_TOOLS - reserved, 0)
    specs: list[ToolSpec] = []

    if include_host_tools and host_limit > 0 and eligible:
        if len(eligible) <= host_limit:
            direct_names = list(eligible)
            defer = False
        else:
            direct_names = list(eligible)[: max(host_limit - META_TOOL_COUNT, 0)]
            defer = True
        for tool_name in direct_names:
            specs.append(_host_tool_spec(tool_name, eligible[tool_name]))
        if defer:
            specs.extend(build_meta_tool_specs(eligible, frozenset(direct_names)))

    if effective_mesh_tools:
        specs.extend(build_mesh_tool_specs())
    if effective_mesh_pipelines:
        specs.extend(build_mesh_pipeline_tool_specs())

    if effective_auto_job:
        specs.append(
            ToolSpec(
                name=AUTONOMOUS_JOB_FUNCTION_NAME,
                description=AUTONOMOUS_JOB_DESCRIPTION,
                parameters=copy.deepcopy(AUTONOMOUS_JOB_SCHEMA),
            )
        )

    if include_wf_template:
        specs.append(
            ToolSpec(
                name=WORKFLOW_TEMPLATE_FUNCTION_NAME,
                description=WORKFLOW_TEMPLATE_DESCRIPTION,
                parameters=copy.deepcopy(WORKFLOW_TEMPLATE_SCHEMA),
            )
        )

    if include_board_tool:
        specs.append(
            ToolSpec(
                name=BOARD_FUNCTION_NAME,
                description=BOARD_MEDIA_KEY_DESCRIPTION,
                parameters=copy.deepcopy(BOARD_MEDIA_KEY_SCHEMA),
            )
        )

    return tuple(specs)


async def load_owner_host_state(
    *,
    owner_id: uuid.UUID,
    host_repo: HostRepository,
    hub: HostSessionHub,
    self_host_id: str | None = None,
) -> tuple[
    Sequence[Host],
    dict[HostId, Manifest],
    dict[HostId, Sequence[Grant]],
    set[str],
]:
    """Loads the owner's hosts plus manifests and grants of the active, online ones."""
    hosts = await host_repo.list_hosts(owner_id)
    manifests_by_host: dict[HostId, Manifest] = {}
    grants_by_host: dict[HostId, Sequence[Grant]] = {}
    online_host_ids: set[str] = set()

    for host in hosts:
        if host.status != HostStatus.ACTIVE:
            continue
        is_self = self_host_id is not None and str(host.host_id) == self_host_id
        if hub.is_online(str(host.host_id)) or is_self:
            online_host_ids.add(str(host.host_id))
            manifest = await host_repo.get_manifest(host.host_id)
            if manifest is not None:
                manifests_by_host[host.host_id] = manifest
            grants = await host_repo.get_grants(host.host_id)
            grants_by_host[host.host_id] = grants
    return hosts, manifests_by_host, grants_by_host, online_host_ids


async def load_tool_catalog_for_owner(
    *,
    owner_id: uuid.UUID,
    client_caps: Collection[str],
    board_available: bool,
    host_repo: HostRepository,
    hub: HostSessionHub,
    self_host_id: str | None = None,
    host_screen_consent: bool = True,
    include_autonomous_job: bool = False,
    now: datetime | None = None,
    audit: AuditLog | None = None,
    pinned_host_id: HostId | None = None,
    include_mesh_tools: bool = False,
) -> tuple[ToolSpec, ...]:
    """Loads the owner's hosts, manifests, and grants and builds the active tool catalog.

    With `audit`, the owner's last `USAGE_WINDOW_EVENTS` audit events order the catalog by
    real usage; a failed read falls back to the default order.
    """
    if CAP_HOST_ACTIONS_V1 not in client_caps and not (
        CAP_BOARD_HID_V1 in client_caps and board_available
    ):
        return ()

    hosts, manifests_by_host, grants_by_host, online_host_ids = await load_owner_host_state(
        owner_id=owner_id,
        host_repo=host_repo,
        hub=hub,
        self_host_id=self_host_id,
    )
    usage: dict[str, int] | None = None
    if audit is not None:
        try:
            usage = tool_usage_counts(await audit.recent(owner_id, USAGE_WINDOW_EVENTS))
        except Exception:  # usage ordering is an optimisation; never fail the turn on it
            usage = None
    return build_tool_catalog(
        hosts=hosts,
        manifests_by_host=manifests_by_host,
        grants_by_host=grants_by_host,
        online_host_ids=online_host_ids,
        client_caps=client_caps,
        board_available=board_available,
        host_screen_consent=host_screen_consent,
        include_autonomous_job=include_autonomous_job,
        now=now,
        usage=usage,
        pinned_host_id=pinned_host_id,
        include_mesh_tools=include_mesh_tools,
    )
