"""Tool catalog builder for LLM function calling (UDC-4 §5.2, UDC-7 S7.1)."""

import copy
import uuid
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime

from xuna_backend.modules.actions.domain.schema_validator import (
    SchemaDefinitionError,
    SchemaValidator,
)
from xuna_backend.modules.actions.domain.tool_names import (
    BOARD_FUNCTION_NAME,
    BOARD_MEDIA_KEY_DESCRIPTION,
    BOARD_MEDIA_KEY_SCHEMA,
    WORKFLOW_TEMPLATE_DESCRIPTION,
    WORKFLOW_TEMPLATE_FUNCTION_NAME,
    WORKFLOW_TEMPLATE_SCHEMA,
    is_forbidden_tool_name,
    to_function_name,
)
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

MAX_CATALOG_TOOLS = 8
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

_PRIORITY_ORDER: tuple[str, ...] = (
    "media.play_pause",
    "media.next",
    "media.previous",
    "media.volume_set",
    "media.now_playing",
    "app.launch",
    "workspace.coding_agent",
    "workspace.open",
    "workspace.git_summary",
    "display.wake",
    "url.open",
    "browser.current_url",
    "workflow.run",
    "host.notify",
    "host.status",
    "app.list_allowed",
    "workflow.list",
    "ui.tap_text",
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
    now: datetime | None = None,
) -> tuple[ToolSpec, ...]:
    """Builds up to `MAX_CATALOG_TOOLS` (`8`) `ToolSpec` entries for `ChatRequest.tools`.

    - Gated behind `client_caps` (`host_actions_v1` for host tools, `board_hid_v1` for board HID).
    - Rejects descriptors failing `SchemaValidator.validate_schema_definition` or forbidden names.
    - Injects optional `target` enum (`[eligible_aliases]`) into each host tool's `parameters`.
    """
    include_host_tools = CAP_HOST_ACTIONS_V1 in client_caps
    include_board_tool = CAP_BOARD_HID_V1 in client_caps and board_available
    include_wf_template = include_host_tools and (
        include_workflow_template or CAP_WORKFLOW_TEMPLATES_V1 in client_caps
    )
    if not include_host_tools and not include_board_tool:
        return ()

    check_now = now or datetime.now(UTC)
    reserved = (1 if include_board_tool else 0) + (1 if include_wf_template else 0)
    host_limit = max(MAX_CATALOG_TOOLS - reserved, 0)
    specs: list[ToolSpec] = []

    if include_host_tools and host_limit > 0:
        by_tool: dict[str, tuple[ToolDescriptor, list[str]]] = {}
        active_online_hosts = [
            h for h in hosts if h.status == HostStatus.ACTIVE and str(h.host_id) in online_host_ids
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

        def _sort_key(tool_name: str) -> tuple[int, str]:
            try:
                idx = _PRIORITY_ORDER.index(tool_name)
            except ValueError:
                idx = len(_PRIORITY_ORDER)
            return (idx, tool_name)

        for tool_name in sorted(by_tool.keys(), key=_sort_key)[:host_limit]:
            descriptor, eligible_aliases = by_tool[tool_name]
            schema_copy = copy.deepcopy(descriptor.input_schema)
            props_raw = schema_copy.get("properties")
            props = dict(props_raw) if isinstance(props_raw, dict) else {}
            props["target"] = {
                "type": "string",
                "enum": list(eligible_aliases),
            }
            schema_copy["properties"] = props
            specs.append(
                ToolSpec(
                    name=to_function_name(tool_name),
                    description=descriptor.description,
                    parameters=schema_copy,
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


async def load_tool_catalog_for_owner(
    *,
    owner_id: uuid.UUID,
    client_caps: Collection[str],
    board_available: bool,
    host_repo: HostRepository,
    hub: HostSessionHub,
    self_host_id: str | None = None,
    host_screen_consent: bool = False,
    now: datetime | None = None,
) -> tuple[ToolSpec, ...]:
    """Loads the owner's hosts, manifests, and grants and builds the active tool catalog."""
    if CAP_HOST_ACTIONS_V1 not in client_caps and not (
        CAP_BOARD_HID_V1 in client_caps and board_available
    ):
        return ()

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

    return build_tool_catalog(
        hosts=hosts,
        manifests_by_host=manifests_by_host,
        grants_by_host=grants_by_host,
        online_host_ids=online_host_ids,
        client_caps=client_caps,
        board_available=board_available,
        host_screen_consent=host_screen_consent,
        now=now,
    )
