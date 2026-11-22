"""Autopilot preset resolution and high-impact risk classification (ADR-0015 2026-11-22).

Centralizes `HostTwin.trust_preset` lazy resolution (`xuna/get_policy`) and the
high-impact risk rules shared by `Orchestrator` and `CrossHostPipelineRunner`,
keeping `orchestrator.py` lean and avoiding duplicate policy logic.
"""

from __future__ import annotations

import re

from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseSessionState,
    hit_test_node_at_coordinates,
    is_consequential_element_name,
    is_hard_denied_target,
    is_high_impact_element_name,
    is_password_ui_node,
)
from xuna_backend.modules.hosts.application.ports import HostSessionHub
from xuna_backend.modules.hosts.domain.entities import (
    HostPlatform,
    RiskTier,
    ToolDescriptor,
)

_DESKTOP_PLATFORMS: frozenset[HostPlatform] = frozenset(
    {HostPlatform.WINDOWS, HostPlatform.MACOS, HostPlatform.LINUX}
)
_VALID_TRUST_PRESETS: frozenset[str] = frozenset({"strict", "standard", "autopilot"})

_HIGH_IMPACT_MENU_TOKENS: tuple[str, ...] = (
    "empty trash",
    "sign out",
    "log out",
    "logout",
    "uninstall",
    "format",
    "đăng xuất",
    "dọn sạch thùng rác",
)

_DESTRUCTIVE_STANDARD_MENU_TOKENS: tuple[str, ...] = (
    "quit",
    "exit",
    "thoát",
    *_HIGH_IMPACT_MENU_TOKENS,
)

_SENSITIVE_WEB_WORDS: tuple[str, ...] = (
    "pin",
    "card",
    "credit",
    "token",
    "secret",
    "wire",
    "thẻ",
)

_AUTOPILOT_SENSITIVE_WEB_SUBSTRINGS: tuple[str, ...] = (
    "password",
    "passwd",
    "cvv",
    "cvc",
    "payment",
    "checkout",
    "check out",
    "transfer",
    "chuyen khoan",
    "thanh toan",
    "dat hang",
    "xoa",
    "mua ngay",
    "mật khẩu",
    "mat khau",
    "the tin dung",
    "the visa",
    "the mastercard",
    "the atm",
    "the ngan hang",
)

_STANDARD_SENSITIVE_WEB_SUBSTRINGS: tuple[str, ...] = (
    *_AUTOPILOT_SENSITIVE_WEB_SUBSTRINGS,
    "xac nhan",
)


async def resolve_host_trust_preset(
    hub: HostSessionHub,
    host_id: str,
    platform: HostPlatform,
    *,
    is_self_host: bool = False,
) -> str:
    """Returns the live `trustPreset` for an online desktop host, fail-closed to `'standard'`."""
    if is_self_host or platform not in _DESKTOP_PLATFORMS:
        return "standard"
    get_twin = getattr(hub, "get_twin", None)
    if not callable(get_twin):
        return "standard"
    twin = get_twin(host_id)
    if twin is None:
        return "standard"
    cached = getattr(twin, "trust_preset", None)
    if isinstance(cached, str) and cached in _VALID_TRUST_PRESETS:
        return cached
    preset = "standard"
    try:
        raw = await hub.send_request(host_id, "xuna/get_policy", {}, timeout_s=1.0)
        raw_preset = raw.get("trustPreset") if isinstance(raw, dict) else None
        if isinstance(raw_preset, str) and raw_preset in _VALID_TRUST_PRESETS:
            preset = raw_preset
    except Exception:
        preset = "standard"
    updater = getattr(hub, "update_trust_preset", None)
    if callable(updater):
        updater(host_id, preset)
    return preset


def is_consequential_web_action(args: dict[str, object], *, autopilot: bool = False) -> bool:
    """Checks whether a browser action targets a consequential/high-impact or sensitive element."""
    matcher = is_high_impact_element_name if autopilot else is_consequential_element_name
    substrings = (
        _AUTOPILOT_SENSITIVE_WEB_SUBSTRINGS if autopilot else _STANDARD_SENSITIVE_WEB_SUBSTRINGS
    )
    for key in ("ref", "selector", "targetText", "elementName", "text", "value"):
        val = args.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        decomposed = re.sub(r"([a-z])([A-Z])", r"\1 \2", val)
        if matcher(val) or matcher(decomposed):
            return True
        normalized = decomposed.replace("_", " ").replace("-", " ").lower()
        if matcher(normalized):
            return True
        words = set(re.findall(r"[\w]+", normalized))
        if any(w in words for w in _SENSITIVE_WEB_WORDS):
            return True
        if (
            "checkout" in normalized
            or "check out" in normalized
            or any(tok in normalized for tok in substrings)
        ):
            return True
    return False


def cu_session_risk_for_autopilot(clean_args: dict[str, object]) -> RiskTier:
    """`cu.session_start`/`cu.session_extend` runs at R2 under autopilot unless hard-denied."""
    raw_apps = clean_args.get("apps")
    if isinstance(raw_apps, list) and any(is_hard_denied_target(str(a)) for a in raw_apps):
        return RiskTier.R3
    return RiskTier.R2


def autopilot_input_risk(
    descriptor: ToolDescriptor,
    clean_args: dict[str, object],
    active_cu: ComputerUseSessionState | None,
    *,
    cached_node: dict[str, object] | None = None,
) -> RiskTier:
    """Risk classification for `input.*` and `app.launch` under `trustPreset == 'autopilot'`."""
    if descriptor.name == "input.click":
        explicit_name = clean_args.get("elementName")
        if isinstance(explicit_name, str) and is_high_impact_element_name(explicit_name):
            return RiskTier.R3
        menu_path = clean_args.get("menuPath")
        if isinstance(menu_path, str):
            lower_menu = menu_path.strip().lower()
            if is_high_impact_element_name(menu_path) or any(
                tok in lower_menu for tok in _HIGH_IMPACT_MENU_TOKENS
            ):
                return RiskTier.R3
        node = cached_node
        if node is None and active_cu is not None:
            x_norm = clean_args.get("xNorm")
            y_norm = clean_args.get("yNorm")
            if isinstance(x_norm, int | float) and isinstance(y_norm, int | float):
                node = hit_test_node_at_coordinates(
                    list(active_cu.cached_nodes.values()), float(x_norm), float(y_norm)
                )
        if node is not None:
            if is_password_ui_node(node):
                return RiskTier.R3
            cached_name = node.get("name")
            if isinstance(cached_name, str) and is_high_impact_element_name(cached_name):
                return RiskTier.R3
        return descriptor.risk
    if descriptor.name == "input.drag":
        for k in ("elementName", "fromElementName", "toElementName"):
            name = clean_args.get(k)
            if isinstance(name, str) and is_high_impact_element_name(name):
                return RiskTier.R3
        return descriptor.risk
    return descriptor.risk


def declared_risk_floor(
    descriptor: ToolDescriptor,
    clean_args: dict[str, object],
    *,
    autopilot: bool = False,
) -> RiskTier:
    """Argument-implied risk floor combined with host preflight answers."""
    if descriptor.name == "mail.send":
        return RiskTier.R3
    if descriptor.name == "mail.create_draft":
        return RiskTier.R2
    if descriptor.name in ("cu.session_start", "cu.session_extend") and autopilot:
        return cu_session_risk_for_autopilot(clean_args)
    if descriptor.name in ("web.click", "web.fill"):
        return (
            RiskTier.R3
            if is_consequential_web_action(clean_args, autopilot=autopilot)
            else descriptor.risk
        )
    matcher = is_high_impact_element_name if autopilot else is_consequential_element_name
    menu_tokens = _HIGH_IMPACT_MENU_TOKENS if autopilot else _DESTRUCTIVE_STANDARD_MENU_TOKENS
    if descriptor.name == "input.click":
        name = clean_args.get("elementName")
        if isinstance(name, str) and matcher(name):
            return RiskTier.R3
        menu_path = clean_args.get("menuPath")
        if isinstance(menu_path, str):
            lower_menu = menu_path.strip().lower()
            if matcher(menu_path) or any(tok in lower_menu for tok in menu_tokens):
                return RiskTier.R3
    elif descriptor.name == "input.drag":
        for k in ("elementName", "fromElementName", "toElementName"):
            name = clean_args.get(k)
            if isinstance(name, str) and matcher(name):
                return RiskTier.R3
    return descriptor.risk

