"""Standing approvals: an owner approval given in advance (ADR-0014 D1).

A standing approval satisfies ADR-0012's `owner approval (R2+)` step for one exact call
(host + tool + the full argument set, identified by `args_sha256`) until it expires, runs out
of uses or is revoked. It never covers R3, never covers tools that observe or act on the screen,
and never covers a free-text argument.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from xuna_backend.modules.hosts.domain.entities import (
    HostId,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
)

STANDING_DEFAULT_LIFETIME = timedelta(days=7)
STANDING_MAX_LIFETIME = timedelta(days=30)
STANDING_MAX_USES = 1000

# Screen-reading and screen-acting families are never eligible (ADR-0014 D1.2). Decided by the
# family the host declares, not by tool names.
_EXCLUDED_FAMILIES = frozenset({ToolFamily.UI, ToolFamily.CU, ToolFamily.INPUT, ToolFamily.SCREEN})
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class StandingApprovalError(ValueError):
    """Raised with a stable `code` when a standing approval is not allowed."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def check_standing_eligibility(descriptor: ToolDescriptor, arguments: dict[str, object]) -> None:
    """Raises `StandingApprovalError` unless this exact call may be pre-approved.

    Eligible only when the tool is R2, does not observe, is not in a screen-control family,
    is supported on the host, and every argument is a boolean, an integer, or a string that
    the tool's own schema pins to a closed `enum`. An unconstrained string is free text and
    is refused.
    """
    if descriptor.risk != RiskTier.R2:
        raise StandingApprovalError("not_eligible_risk")
    if descriptor.observes:
        raise StandingApprovalError("not_eligible_observes")
    if descriptor.family in _EXCLUDED_FAMILIES:
        raise StandingApprovalError("not_eligible_family")
    if descriptor.permission == ToolPermissionState.UNSUPPORTED:
        raise StandingApprovalError("not_eligible_unsupported")
    properties = descriptor.input_schema.get("properties")
    props = properties if isinstance(properties, dict) else {}
    for name, value in arguments.items():
        if isinstance(value, bool | int):
            continue
        if isinstance(value, str):
            schema = props.get(name)
            allowed = schema.get("enum") if isinstance(schema, dict) else None
            if isinstance(allowed, list) and value in allowed:
                continue
            raise StandingApprovalError("not_eligible_free_text")
        raise StandingApprovalError("not_eligible_argument")


def is_standing_eligible(descriptor: ToolDescriptor, arguments: dict[str, object]) -> bool:
    try:
        check_standing_eligibility(descriptor, arguments)
    except StandingApprovalError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class StandingApproval:
    """A signed, expiring, revocable pre-approval of one exact host call."""

    approval_id: uuid.UUID
    owner_id: uuid.UUID
    host_id: HostId
    target_alias: str
    tool: str
    args_sha256: str
    summary: str
    created_at: datetime
    expires_at: datetime
    max_uses: int | None
    signature: bytes
    signer_public_key: bytes
    uses: int = 0
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        if not _SHA256_HEX_RE.match(self.args_sha256):
            raise ValueError("args_sha256 must be 64 lowercase hex characters")
        if not 1 <= len(self.target_alias) <= 32:
            raise ValueError("target_alias must be 1..32 characters")
        if not 1 <= len(self.summary) <= 240:
            raise ValueError("summary must be 1..240 characters")
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        if self.revoked_at is not None and self.revoked_at.tzinfo is None:
            raise ValueError("revoked_at must be timezone-aware")
        if not self.created_at < self.expires_at <= self.created_at + STANDING_MAX_LIFETIME:
            raise ValueError("expiry must be after creation and at most 30 days away")
        if self.max_uses is not None and not 1 <= self.max_uses <= STANDING_MAX_USES:
            raise ValueError("max_uses must be 1..1000")
        if self.uses < 0:
            raise ValueError("uses must not be negative")
        if len(self.signature) != 64:
            raise ValueError("signature must be 64 bytes")
        if len(self.signer_public_key) != 65 or self.signer_public_key[0] != 0x04:
            raise ValueError("signer_public_key must be 65-byte uncompressed P-256 point")

    def is_usable(self, now: datetime) -> bool:
        if self.revoked_at is not None or now >= self.expires_at:
            return False
        return self.max_uses is None or self.uses < self.max_uses
