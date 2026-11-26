"""Parallel R0/R1 scatter-gather prober and mesh meta-tool executor (Mesh §5.5).

Executes read-only/observing `tools/call` probes concurrently across eligible online
hosts with per-host `HostGrant` + `U5` verification, strict `2,000 ms` deadlines,
fail-closed error propagation (H4), and zero raw OS/PII path leakage (H2/H5).
"""

import asyncio
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from xuna_backend.modules.actions.application.catalog import is_tool_granted
from xuna_backend.modules.actions.application.prompts import truncate_tool_value
from xuna_backend.modules.actions.application.tool_discovery import (
    MESH_FIND_RESOURCE_FUNCTION,
    MESH_STATUS_FUNCTION,
    encode_mesh_meta_payload,
    parse_mesh_find_resource_args,
    parse_mesh_status_args,
)
from xuna_backend.modules.actions.domain.battery_advisor import (
    LOW_BATTERY_THRESHOLD_PCT,
    evaluate_battery_advisory,
    is_low_battery,
)
from xuna_backend.modules.actions.domain.tool_names import is_forbidden_tool_name
from xuna_backend.modules.hosts.application.ports import HostSessionHub
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostId,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
    fold_text,
)
from xuna_backend.modules.hosts.domain.session import structured_content_exceeds_limit
from xuna_backend.modules.hosts.domain.twin import HostFocusState, HostPresence, HostTwin

DEFAULT_PROBE_DEADLINE_MS = 2000

_PROBE_ALLOWED_RISKS: frozenset[RiskTier] = frozenset({RiskTier.R0, RiskTier.R1})
_U5_REQUIRED_FAMILIES: frozenset[ToolFamily] = frozenset(
    {
        ToolFamily.CU,
        ToolFamily.SCREEN,
        ToolFamily.UI,
        ToolFamily.INPUT,
    }
)
_PATH_KEYS: frozenset[str] = frozenset(
    {
        "path",
        "filepath",
        "fullpath",
        "executable",
        "executablepath",
        "target",
        "cwd",
        "workingdir",
        "homedir",
        "rootdir",
        "dir",
        "folder",
    }
)
_OS_PATH_RE = re.compile(
    r"""(?:(?<=^)|(?<=[\s"'(\[=,:]))"""
    r"""(?:(?i:file://)[^\s"'()\]]+|[A-Za-z]:[\\/][^\s"'()\]]+|\\\\[^\s"'()\]]+|~[\\/][^\s"'()\]]+|\.{1,2}[\\/][^\s"'()\]]+|/[A-Za-z0-9._-]+(?:/[^\s"'()\]]*)+)"""
)
_MAX_SANITIZE_DEPTH = 4


@dataclass(frozen=True, slots=True)
class ProbeTarget:
    """Candidate host bundle for `MeshProber` operations."""

    host: Host
    manifest: Manifest | None
    grants: Sequence[Grant]
    online: bool
    is_self: bool = False


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """Bounded fail-closed result of a single host probe."""

    host_id: str
    alias: str
    tool: str
    status: Literal["ok", "timeout", "error", "denied", "unsupported"]
    error_code: str | None = None
    structured_content: dict[str, object] | None = None
    duration_ms: int | None = None


def build_probe_targets(
    hosts: Sequence[Host],
    manifests_by_host: Mapping[HostId, Manifest],
    grants_by_host: Mapping[HostId, Sequence[Grant]],
    online_host_ids: Sequence[str] | set[str] | frozenset[str],
    *,
    self_host_id: str | None = None,
) -> tuple[ProbeTarget, ...]:
    """Builds `ProbeTarget` tuples from loaded owner host state."""
    online_set = set(online_host_ids)
    targets: list[ProbeTarget] = []
    for host in hosts:
        if host.status != HostStatus.ACTIVE:
            continue
        hid_str = str(host.host_id)
        is_self = self_host_id is not None and hid_str == self_host_id
        targets.append(
            ProbeTarget(
                host=host,
                manifest=manifests_by_host.get(host.host_id),
                grants=grants_by_host.get(host.host_id, ()),
                online=(hid_str in online_set) or is_self,
                is_self=is_self,
            )
        )
    return tuple(targets)


def _safe_get_twin(hub: HostSessionHub, host_id: str) -> HostTwin | None:
    fn = getattr(hub, "get_twin", None)
    if callable(fn):
        twin = fn(host_id)
        if isinstance(twin, HostTwin):
            return twin
    return None


def _safe_get_focus(hub: HostSessionHub, owner_id: uuid.UUID) -> HostFocusState | None:
    fn = getattr(hub, "get_focus_state", None)
    if callable(fn):
        state = fn(owner_id)
        if isinstance(state, HostFocusState):
            return state
    return None


def _sanitize_probe_value(value: object, depth: int = 0) -> object:
    """Strips filesystem path keys/values and bounds list/string lengths (H2)."""
    if depth >= _MAX_SANITIZE_DEPTH:
        return "[truncated]"
    if isinstance(value, Mapping):
        clean: dict[str, object] = {}
        for k, v in value.items():
            key_str = str(k)
            if key_str.lower().replace("_", "").replace("-", "") in _PATH_KEYS:
                continue
            clean[key_str[:64]] = _sanitize_probe_value(v, depth + 1)
        return clean
    if isinstance(value, list | tuple):
        return [_sanitize_probe_value(item, depth + 1) for item in list(value)[:64]]
    if isinstance(value, str):
        redacted = _OS_PATH_RE.sub("[redacted_path]", value)
        return redacted[:128]
    return truncate_tool_value(value)


def _extract_normalized_app_ids(structured_content: Mapping[str, object] | None) -> tuple[str, ...]:
    """Extracts sanitized app IDs from `app.list_allowed` `structuredContent.apps`."""
    if not isinstance(structured_content, Mapping):
        return ()
    raw_apps = structured_content.get("apps")
    if not isinstance(raw_apps, list | tuple):
        return ()
    seen: set[str] = set()
    out: list[str] = []
    for entry in raw_apps[:128]:
        raw_id: object = None
        if isinstance(entry, str):
            raw_id = entry
        elif isinstance(entry, Mapping):
            raw_id = entry.get("id") or entry.get("name")
        if not isinstance(raw_id, str):
            continue
        cleaned = raw_id.strip()
        if (
            not cleaned
            or "[redacted_path]" in cleaned
            or "[truncated]" in cleaned
            or len(cleaned) > 64
            or "/" in cleaned
            or "\\" in cleaned
            or ".." in cleaned
            or ":" in cleaned
        ):
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            out.append(cleaned)
    return tuple(out)


def _serialize_presence(presence: HostPresence | None) -> dict[str, object] | None:
    if presence is None:
        return None
    return {
        "idleSeconds": presence.idle_seconds,
        "screenLocked": presence.screen_locked,
        "mediaPlaying": presence.media_playing,
        "batteryPercent": presence.battery_percent,
        "isCharging": presence.is_charging,
        "foregroundAppId": presence.foreground_app_id,
        "subnetHash": presence.subnet_hash,
        "updatedAt": presence.updated_at.astimezone(UTC).isoformat(),
    }


def _short_error_code(raw: object, fallback: str) -> str:
    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned:
            return cleaned[:64]
    return fallback


class MeshProber:
    """Bounded concurrent R0/R1 scatter-gather prober across active owner hosts."""

    def __init__(
        self,
        hub: HostSessionHub,
        *,
        clock: Callable[[], datetime] | None = None,
        default_deadline_ms: int = DEFAULT_PROBE_DEADLINE_MS,
    ) -> None:
        self._hub = hub
        self._clock = clock or (lambda: datetime.now(UTC))
        self._default_deadline_ms = max(100, default_deadline_ms)

    async def probe_one(
        self,
        *,
        host: Host,
        descriptor: ToolDescriptor | None,
        grants: Sequence[Grant],
        arguments: dict[str, object] | None = None,
        host_screen_consent: bool = True,
        deadline_ms: int | None = None,
        now: datetime | None = None,
    ) -> ProbeOutcome:
        """Executes a single R0/R1 read-only probe with strict pre-flight and deadline."""
        hid = str(host.host_id)
        alias = str(host.alias)
        tool_name = descriptor.name if descriptor is not None else ""
        if descriptor is None or descriptor.permission == ToolPermissionState.UNSUPPORTED:
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=tool_name,
                status="unsupported",
                error_code="unsupported",
            )
        if is_forbidden_tool_name(descriptor.name):
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=descriptor.name,
                status="denied",
                error_code="forbidden_tool",
            )
        if descriptor.risk not in _PROBE_ALLOWED_RISKS:
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=descriptor.name,
                status="denied",
                error_code="probe_risk_too_high",
            )
        check_now = now or self._clock()
        if not is_tool_granted(
            descriptor,
            grants,
            check_now,
            host_screen_consent=host_screen_consent,
        ):
            needs_u5 = descriptor.observes or descriptor.family in _U5_REQUIRED_FAMILIES
            deny_code = (
                "u5_consent_required" if (needs_u5 and not host_screen_consent) else "grant_denied"
            )
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=descriptor.name,
                status="denied",
                error_code=deny_code,
            )
        if not self._hub.is_online(hid):
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=descriptor.name,
                status="error",
                error_code="host_offline",
            )

        eff_ms = max(100, deadline_ms if deadline_ms is not None else self._default_deadline_ms)
        timeout_s = eff_ms / 1000.0
        params: dict[str, object] = {
            "name": descriptor.name,
            "arguments": dict(arguments or {}),
            "_meta": {
                "xuna/invocationId": str(uuid.uuid4()),
                "xuna/deadlineMs": eff_ms,
            },
        }
        try:
            raw = await asyncio.wait_for(
                self._hub.send_request(hid, "tools/call", params, timeout_s=timeout_s),
                timeout=timeout_s,
            )
        except TimeoutError:
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=descriptor.name,
                status="timeout",
                error_code="timeout",
                duration_ms=eff_ms,
            )
        except RuntimeError:
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=descriptor.name,
                status="error",
                error_code="rpc_error",
            )
        except Exception:
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=descriptor.name,
                status="error",
                error_code="probe_failed",
            )

        if not isinstance(raw, Mapping):
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=descriptor.name,
                status="error",
                error_code="invalid_probe_response",
            )
        meta_raw = raw.get("_meta")
        meta = meta_raw if isinstance(meta_raw, Mapping) else {}
        sc_raw = raw.get("structuredContent")
        sc_map = sc_raw if isinstance(sc_raw, Mapping) else {}
        xuna_status = meta.get("xuna/status") or sc_map.get("status")
        duration_raw = meta.get("xuna/durationMs")
        duration = (
            duration_raw
            if isinstance(duration_raw, int) and not isinstance(duration_raw, bool)
            else None
        )
        if structured_content_exceeds_limit(sc_map, allow_media=False):
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=descriptor.name,
                status="error",
                error_code="structured_content_too_large",
                duration_ms=duration,
            )
        if xuna_status is not None and xuna_status != "ok":
            status_str = xuna_status.strip() if isinstance(xuna_status, str) else ""
            if status_str in ("denied", "denied_by_local_policy"):
                mapped_status: Literal["denied", "unsupported", "timeout", "error"] = "denied"
                default_err = "probe_denied"
            elif status_str in ("needs_permission", "permission_denied"):
                mapped_status = "denied"
                default_err = "needs_permission"
            elif status_str == "unsupported":
                mapped_status = "unsupported"
                default_err = "unsupported"
            elif status_str == "timeout":
                mapped_status = "timeout"
                default_err = "timeout"
            elif status_str == "invalid_args":
                mapped_status = "error"
                default_err = "invalid_args"
            else:
                mapped_status = "error"
                default_err = "probe_failed"
            err_code = _short_error_code(
                meta.get("xuna/errorCode") or sc_map.get("errorCode"),
                default_err,
            )
            return ProbeOutcome(
                host_id=hid,
                alias=alias,
                tool=descriptor.name,
                status=mapped_status,
                error_code=err_code,
                duration_ms=duration,
            )

        sanitized = _sanitize_probe_value(sc_map)
        clean_sc = sanitized if isinstance(sanitized, dict) else {}
        return ProbeOutcome(
            host_id=hid,
            alias=alias,
            tool=descriptor.name,
            status="ok",
            structured_content=clean_sc,
            duration_ms=duration,
        )

    async def probe_many(
        self,
        requests: Sequence[tuple[ProbeTarget, str, dict[str, object]]],
        *,
        host_screen_consent: bool,
        deadline_ms: int | None = None,
        now: datetime | None = None,
    ) -> list[ProbeOutcome]:
        """Executes multiple probes concurrently; individual failures never abort peers."""
        if not requests:
            return []
        check_now = now or self._clock()

        async def _run_one(
            target: ProbeTarget, tool_name: str, args: dict[str, object]
        ) -> ProbeOutcome:
            desc = target.manifest.find_tool(tool_name) if target.manifest is not None else None
            return await self.probe_one(
                host=target.host,
                descriptor=desc,
                grants=target.grants,
                arguments=args,
                host_screen_consent=host_screen_consent,
                deadline_ms=deadline_ms,
                now=check_now,
            )

        raw_results = await asyncio.gather(
            *(_run_one(target, tname, args) for target, tname, args in requests),
            return_exceptions=True,
        )
        outcomes: list[ProbeOutcome] = []
        for (target, tname, _), res in zip(requests, raw_results, strict=True):
            if isinstance(res, ProbeOutcome):
                outcomes.append(res)
            else:
                outcomes.append(
                    ProbeOutcome(
                        host_id=str(target.host.host_id),
                        alias=str(target.host.alias),
                        tool=tname,
                        status="error",
                        error_code="probe_failed",
                    )
                )
        return outcomes

    async def query_status(
        self,
        *,
        owner_id: uuid.UUID,
        targets: Sequence[ProbeTarget],
        host_ids: tuple[uuid.UUID, ...] | None = None,
        include_vitals: bool = False,
        host_screen_consent: bool = True,
        deadline_ms: int | None = None,
    ) -> dict[str, object]:
        """Returns live `HostTwin` presence + concurrent R0/R1 probes for owner's hosts."""
        allowed_ids = set(host_ids) if host_ids is not None else None
        filtered = [
            t
            for t in targets
            if t.host.owner_id == owner_id
            and t.host.status == HostStatus.ACTIVE
            and (allowed_ids is None or t.host.host_id.value in allowed_ids)
        ]
        filtered.sort(key=lambda t: (t.host.alias.value.lower(), str(t.host.host_id)))

        focus_state = _safe_get_focus(self._hub, owner_id)
        focused_id = focus_state.focused_host_id if focus_state is not None else None
        now = self._clock()

        probe_requests: list[tuple[ProbeTarget, str, dict[str, object]]] = []
        status_req_idx: dict[HostId, int] = {}
        vitals_req_idx: dict[HostId, int] = {}

        for target in filtered:
            if not target.online or target.manifest is None:
                continue
            status_desc = target.manifest.find_tool("host.status") or target.manifest.find_tool(
                "media.now_playing"
            )
            if status_desc is not None:
                status_req_idx[target.host.host_id] = len(probe_requests)
                probe_requests.append((target, status_desc.name, {}))
            if include_vitals:
                vitals_desc = target.manifest.find_tool("system.vitals")
                if vitals_desc is not None:
                    vitals_req_idx[target.host.host_id] = len(probe_requests)
                    probe_requests.append((target, "system.vitals", {}))

        outcomes = await self.probe_many(
            probe_requests,
            host_screen_consent=host_screen_consent,
            deadline_ms=deadline_ms,
            now=now,
        )

        rows: list[dict[str, object]] = []
        twins_by_host: dict[str, HostTwin] = {}
        online_ids: set[str] = set()

        for target in filtered:
            host = target.host
            hid_str = str(host.host_id)
            twin = _safe_get_twin(self._hub, hid_str) if target.online else None
            if target.online:
                online_ids.add(hid_str)
            if twin is not None:
                twins_by_host[hid_str] = twin
            presence = twin.presence if twin is not None else None
            presence_dict = _serialize_presence(presence)

            row: dict[str, object] = {
                "hostId": hid_str,
                "alias": host.alias.value,
                "platform": host.platform.value,
                "hostKind": host.host_kind.value,
                "place": host.place,
                "online": target.online,
                "focused": focused_id == host.host_id,
                "presence": presence_dict,
            }

            if not target.online:
                row["probeStatus"] = "offline"
                if include_vitals:
                    row["vitalsStatus"] = "offline"
            else:
                if host.host_id in status_req_idx:
                    st_out = outcomes[status_req_idx[host.host_id]]
                    row["probeStatus"] = st_out.status
                    if st_out.error_code is not None:
                        row["probeErrorCode"] = st_out.error_code
                    if st_out.status == "ok" and st_out.structured_content is not None:
                        row["probe"] = st_out.structured_content
                else:
                    row["probeStatus"] = "unsupported"
                    row["probeErrorCode"] = "unsupported"

                if include_vitals:
                    if host.host_id in vitals_req_idx:
                        vt_out = outcomes[vitals_req_idx[host.host_id]]
                        row["vitalsStatus"] = vt_out.status
                        if vt_out.error_code is not None:
                            row["vitalsErrorCode"] = vt_out.error_code
                        if vt_out.status == "ok" and vt_out.structured_content is not None:
                            row["vitals"] = vt_out.structured_content
                    else:
                        row["vitalsStatus"] = "unsupported"
                        row["vitalsErrorCode"] = "unsupported"

                if is_low_battery(presence, threshold_pct=LOW_BATTERY_THRESHOLD_PCT):
                    row["lowBattery"] = True

            rows.append(row)

        payload: dict[str, object] = {"status": "ok", "hosts": rows}
        advisory = evaluate_battery_advisory(
            [t.host for t in filtered],
            twins_by_host,
            online_host_ids=online_ids,
            threshold_pct=LOW_BATTERY_THRESHOLD_PCT,
        )
        if advisory is not None and advisory.alternative_hosts:
            payload["batteryAdvisories"] = list(advisory.messages)
        return payload

    async def find_resource(
        self,
        *,
        owner_id: uuid.UUID,
        targets: Sequence[ProbeTarget],
        query: str,
        kind: Literal["app", "workspace", "toolchain", "tool"] | None = None,
        host_screen_consent: bool = True,
        deadline_ms: int | None = None,
    ) -> dict[str, object]:
        """Searches `HostTwin.resources` + manifests across online hosts with R0 fallback."""
        now = self._clock()
        q_folded = fold_text(query)
        online_targets = [
            t
            for t in targets
            if t.host.owner_id == owner_id and t.host.status == HostStatus.ACTIVE and t.online
        ]
        online_targets.sort(key=lambda t: (t.host.alias.value.lower(), str(t.host.host_id)))

        matches: list[dict[str, object]] = []
        probe_errors: list[dict[str, object]] = []
        app_probe_targets: list[tuple[ProbeTarget, str, dict[str, object]]] = []

        for target in online_targets:
            host = target.host
            hid_str = str(host.host_id)
            twin = _safe_get_twin(self._hub, hid_str)
            resources = twin.resources if twin is not None else None
            has_twin_resources = resources is not None and (
                bool(resources.installed_apps)
                or bool(resources.workspaces)
                or bool(resources.toolchains)
                or resources.capabilities_version is not None
            )

            if kind in (None, "app"):
                if has_twin_resources and resources is not None:
                    for app_id in resources.installed_apps:
                        folded_app = fold_text(app_id)
                        if q_folded in folded_app:
                            matches.append(
                                {
                                    "hostId": hid_str,
                                    "alias": host.alias.value,
                                    "platform": host.platform.value,
                                    "kind": "app",
                                    "id": app_id,
                                    "exact": folded_app == q_folded,
                                    "source": "twin",
                                }
                            )
                elif target.manifest is not None and target.manifest.find_tool("app.list_allowed"):
                    app_probe_targets.append((target, "app.list_allowed", {}))

            if kind in (None, "workspace") and resources is not None:
                for ws in resources.workspaces:
                    folded_ws = fold_text(ws)
                    if q_folded in folded_ws:
                        matches.append(
                            {
                                "hostId": hid_str,
                                "alias": host.alias.value,
                                "platform": host.platform.value,
                                "kind": "workspace",
                                "id": ws,
                                "exact": folded_ws == q_folded,
                                "source": "twin",
                            }
                        )

            if kind in (None, "toolchain") and resources is not None:
                for tc in resources.toolchains:
                    folded_tc = fold_text(tc.id)
                    if q_folded in folded_tc:
                        matches.append(
                            {
                                "hostId": hid_str,
                                "alias": host.alias.value,
                                "platform": host.platform.value,
                                "kind": "toolchain",
                                "id": tc.id,
                                "version": tc.version,
                                "exact": folded_tc == q_folded,
                                "source": "twin",
                            }
                        )

            if kind in (None, "tool") and target.manifest is not None:
                for desc in target.manifest.tools:
                    if is_forbidden_tool_name(desc.name):
                        continue
                    if desc.permission == ToolPermissionState.UNSUPPORTED:
                        continue
                    folded_name = fold_text(desc.name)
                    folded_desc = fold_text(desc.description)
                    if q_folded in folded_name or q_folded in folded_desc:
                        matches.append(
                            {
                                "hostId": hid_str,
                                "alias": host.alias.value,
                                "platform": host.platform.value,
                                "kind": "tool",
                                "id": desc.name,
                                "exact": folded_name == q_folded,
                                "risk": desc.risk.value,
                                "granted": is_tool_granted(
                                    desc,
                                    target.grants,
                                    now,
                                    host_screen_consent=host_screen_consent,
                                ),
                                "source": "manifest",
                            }
                        )

        if app_probe_targets:
            probe_outcomes = await self.probe_many(
                app_probe_targets,
                host_screen_consent=host_screen_consent,
                deadline_ms=deadline_ms,
                now=now,
            )
            for (target, _, _), outcome in zip(app_probe_targets, probe_outcomes, strict=True):
                if outcome.status != "ok":
                    probe_errors.append(
                        {
                            "hostId": outcome.host_id,
                            "alias": outcome.alias,
                            "status": outcome.status,
                            "errorCode": outcome.error_code or "probe_failed",
                        }
                    )
                    continue
                for app_id in _extract_normalized_app_ids(outcome.structured_content):
                    folded_app = fold_text(app_id)
                    if q_folded in folded_app:
                        matches.append(
                            {
                                "hostId": str(target.host.host_id),
                                "alias": target.host.alias.value,
                                "platform": target.host.platform.value,
                                "kind": "app",
                                "id": app_id,
                                "exact": folded_app == q_folded,
                                "source": "probe",
                            }
                        )

        matches.sort(
            key=lambda m: (
                not bool(m.get("exact")),
                not bool(m.get("granted", True)),
                str(m.get("alias", "")).lower(),
                str(m.get("id", "")).lower(),
            )
        )
        result: dict[str, object] = {
            "status": "ok",
            "query": query,
            "matches": matches,
        }
        if probe_errors:
            result["probeErrors"] = probe_errors
        return result


async def execute_mesh_meta_call(
    *,
    prober: MeshProber,
    tool_name: str,
    arguments: Mapping[str, object],
    owner_id: uuid.UUID,
    hosts: Sequence[Host],
    manifests: Mapping[HostId, Manifest],
    grants: Mapping[HostId, Sequence[Grant]],
    online: Sequence[str] | set[str] | frozenset[str],
    self_host_id: str | None,
    host_screen_consent: bool,
) -> str:
    """Dispatches `mesh__status` / `mesh__find_resource` and returns encoded JSON."""
    targets = build_probe_targets(hosts, manifests, grants, online, self_host_id=self_host_id)
    if tool_name == MESH_STATUS_FUNCTION:
        host_ids, include_vitals = parse_mesh_status_args(arguments)
        payload = await prober.query_status(
            owner_id=owner_id,
            targets=targets,
            host_ids=host_ids,
            include_vitals=include_vitals,
            host_screen_consent=host_screen_consent,
        )
    elif tool_name == MESH_FIND_RESOURCE_FUNCTION:
        query, kind = parse_mesh_find_resource_args(arguments)
        payload = await prober.find_resource(
            owner_id=owner_id,
            targets=targets,
            query=query,
            kind=kind,
            host_screen_consent=host_screen_consent,
        )
    else:
        raise ValueError(f"Unsupported mesh meta tool: {tool_name}")
    return encode_mesh_meta_payload(payload)
