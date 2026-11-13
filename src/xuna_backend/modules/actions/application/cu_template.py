"""Saved computer-use session templates: the per-turn pin (UDC-17 S17.5).

The owner saves a template on their phone (owner decision 2026-11-13: templates live on the
phone, the server keeps nothing new). Tapping "Chạy lại" sends one chat turn with the template's
scope; for that turn only, `cu.session_start` must match it: same computer, the same apps and
domains, budgets no larger than the template's. The goal text is not pinned (the model may
reword it; the approval card shows it and the owner signs it). A template never replaces the
phone-signed R3 approval of `cu.session_start`.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from xuna_backend.modules.actions.application.computer_use import (
    MAX_SESSION_MINUTES,
    MAX_SESSION_STEPS,
)
from xuna_backend.modules.cognition.domain.types import Locale

# Same scope limits the Desktop Agent enforces for a whole session (UDC-17 S17.3).
MAX_TEMPLATE_APPS = 8
MAX_TEMPLATE_DOMAINS = 12
MAX_TEMPLATE_NAME = 40

_APP_ID_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")
_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,63}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def normalize_template_domain(raw: str) -> str:
    """Lower case, without surrounding blanks or a trailing dot."""
    return raw.strip().lower().rstrip(".")


@dataclass(frozen=True, slots=True)
class CuTemplatePin:
    """The scope one turn's `cu.session_start` must match."""

    name: str
    host_id: str
    apps: frozenset[str]
    domains: frozenset[str]
    max_steps: int
    max_minutes: int

    @classmethod
    def create(
        cls,
        *,
        name: str,
        host_id: str,
        apps: list[str],
        domains: list[str],
        max_steps: int,
        max_minutes: int,
    ) -> CuTemplatePin:
        """Validates like `cu.session_start` arguments; raises `ValueError` with a code."""
        clean_name = name.strip()
        if not 1 <= len(clean_name) <= MAX_TEMPLATE_NAME or _CONTROL_RE.search(clean_name):
            raise ValueError("template_name_invalid")
        try:
            clean_host = str(uuid.UUID(host_id))
        except ValueError as error:
            raise ValueError("template_host_invalid") from error
        if not apps or len(apps) > MAX_TEMPLATE_APPS:
            raise ValueError("template_apps_invalid")
        clean_apps = frozenset(a.strip().lower() for a in apps)
        if any(not _APP_ID_RE.match(a) for a in clean_apps):
            raise ValueError("template_apps_invalid")
        if len(domains) > MAX_TEMPLATE_DOMAINS:
            raise ValueError("template_domains_invalid")
        clean_domains = frozenset(normalize_template_domain(d) for d in domains)
        if any(not _DOMAIN_RE.match(d) for d in clean_domains):
            raise ValueError("template_domains_invalid")
        if not 1 <= max_steps <= MAX_SESSION_STEPS or not 1 <= max_minutes <= MAX_SESSION_MINUTES:
            raise ValueError("template_budget_invalid")
        return cls(
            name=clean_name,
            host_id=clean_host,
            apps=clean_apps,
            domains=clean_domains,
            max_steps=max_steps,
            max_minutes=max_minutes,
        )

    def mismatch(self, host_id: str, arguments: dict[str, object]) -> str | None:
        """Why a `cu.session_start` call on `host_id` does not match, or `None`."""
        if host_id != self.host_id:
            return "host"
        raw_apps = arguments.get("apps")
        apps = (
            frozenset(str(a).strip().lower() for a in raw_apps)
            if isinstance(raw_apps, list)
            else None
        )
        if apps != self.apps:
            return "apps"
        raw_domains = arguments.get("domains")
        domains = (
            frozenset(normalize_template_domain(str(d)) for d in raw_domains)
            if isinstance(raw_domains, list)
            else frozenset()
        )
        if domains != self.domains:
            return "domains"
        steps = arguments.get("maxSteps")
        minutes = arguments.get("maxMinutes")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps > self.max_steps:
            return "budget"
        if not isinstance(minutes, int) or isinstance(minutes, bool) or minutes > self.max_minutes:
            return "budget"
        return None

    def prompt_note(self, locale: Locale, target_alias: str | None) -> str:
        """System-prompt line for this turn only, so the model's first `cu.session_start`
        already matches instead of being refused once (UDC-17 S17.5). Carries the owner's
        own template name and validated scope; `target_alias` is the computer's current
        alias read by the backend, or `None` when the owner no longer has that computer."""
        apps = ", ".join(sorted(self.apps))
        domains = ", ".join(sorted(self.domains))
        if locale == Locale.EN:
            where = (
                f"on the computer '{target_alias}'"
                if target_alias
                else "on the template's computer, which is no longer enrolled (tell the owner)"
            )
            return (
                f"\n- This turn reruns the owner's saved template '{self.name}'. Call "
                f"cu__session_start {where} with exactly apps=[{apps}] and "
                f"domains=[{domains or 'none'}], maxSteps at most {self.max_steps} and "
                f"maxMinutes at most {self.max_minutes}; take the goal from the owner's "
                "message. Do not add or drop any app or domain."
            )
        where_vi = (
            f"trên máy '{target_alias}'"
            if target_alias
            else "trên máy của mẫu, nhưng máy đó không còn được ghép (hãy báo chủ)"
        )
        return (
            f"\n- Lượt này chủ chạy lại mẫu đã lưu '{self.name}'. Gọi cu__session_start "
            f"{where_vi} với đúng apps=[{apps}] và domains=[{domains or 'không có'}], "
            f"maxSteps tối đa {self.max_steps} và maxMinutes tối đa {self.max_minutes}; "
            "mục tiêu lấy từ tin nhắn của chủ. Không thêm hay bớt app hoặc tên miền nào."
        )

    def expected(self) -> dict[str, object]:
        """The scope the model must use, for its tool result."""
        return {
            "apps": sorted(self.apps),
            "domains": sorted(self.domains),
            "maxSteps": self.max_steps,
            "maxMinutes": self.max_minutes,
        }
