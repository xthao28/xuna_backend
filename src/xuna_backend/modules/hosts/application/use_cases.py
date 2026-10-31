"""Application use cases for the Host Registry & XTP v1 module (UDC-3 §5.4, UDC-7).

Covers enrollment (start, claim, confirm), authenticated sessions (open, touch,
close), manifest recording, host listing, grant updates (including expiry and
suspension), LAN peer metadata, revocation, and dev/test tool invocation.
"""

import base64
import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.hosts.application.ports import (
    EnrollmentRepository,
    HostRepository,
    HostSessionHub,
)
from xuna_backend.modules.hosts.domain.crypto import (
    SIGNATURE_SIZE,
    HostProofPurpose,
    args_sha256,
    decode_b64url_16b,
    host_proof_message,
    public_key_fingerprint_b64url,
    sas,
    validate_public_key,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    PendingEnrollment,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    default_initial_grants,
    validate_tool_name,
)
from xuna_backend.modules.hosts.domain.session import (
    APPROVAL_TTL_S,
    ENROLLMENT_TTL_S,
    UNAUTHORIZED,
    HostRateLimiter,
    HostSessionClose,
    HostSessionRejected,
    SelfEnrollmentChallengeStore,
)
from xuna_backend.modules.identity.application.ports import SignatureVerifier


class HostError(ValueError):
    """Domain/application error mapped to RFC 9457 problem codes in the API layer."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class StartEnrollmentCommand:
    code: bytes
    public_key: bytes
    nonce: bytes
    signature: bytes
    platform: HostPlatform
    os_version: str
    agent_version: str
    hostname: str


@dataclass(frozen=True, slots=True)
class ClaimEnrollmentCommand:
    enrollment_code: str
    alias: str
    key_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class SelfEnrollHostCommand:
    platform: HostPlatform
    os_version: str
    agent_version: str
    alias: str
    public_key: bytes
    signature: bytes
    hostname: str | None = None
    tools: list[dict[str, object]] | None = None


@dataclass(frozen=True, slots=True)
class OpenHostSessionCommand:
    host_id: HostId
    nonce: bytes
    signature: bytes
    platform: HostPlatform
    os_version: str
    agent_version: str


@dataclass(frozen=True, slots=True)
class GrantUpdateSpec:
    family: ToolFamily
    max_risk: RiskTier
    enabled: bool
    expires_at: datetime | None = None
    suspended: bool | None = None


@dataclass(frozen=True, slots=True)
class HostSummaryView:
    host: Host
    grants: list[Grant]
    manifest: Manifest | None
    online: bool


@dataclass(frozen=True, slots=True)
class ToolInvocationResult:
    invocation_id: str
    tool_name: str
    risk: RiskTier
    status: str
    is_error: bool
    structured_content: dict[str, object]
    text: str
    duration_ms: int
    error_code: str | None = None


async def start_enrollment(
    enrollments: EnrollmentRepository,
    verifier: SignatureVerifier,
    command: StartEnrollmentCommand,
    now: datetime,
    pending_id: HostId | None = None,
) -> PendingEnrollment:
    """Verifies the agent's purpose-1 proof over `sha256(code)` and stores a 5-min record."""
    message = host_proof_message(
        nonce=command.nonce,
        purpose=HostProofPurpose.ENROLL,
        payload=command.code,
    )
    if not verifier.verify(command.public_key, message, command.signature):
        raise HostSessionRejected(UNAUTHORIZED)

    code_sha256 = hashlib.sha256(command.code).hexdigest()
    computed_sas = sas(command.code, command.public_key)
    fingerprint = public_key_fingerprint_b64url(command.public_key)
    resolved_id = pending_id or HostId.generate()

    pending = PendingEnrollment(
        pending_id=resolved_id,
        code_sha256=code_sha256,
        public_key=command.public_key,
        key_fingerprint=fingerprint,
        sas=computed_sas,
        platform=command.platform,
        os_version=command.os_version,
        agent_version=command.agent_version,
        hostname=command.hostname,
        created_at=now,
        expires_at=now + timedelta(seconds=ENROLLMENT_TTL_S),
    )
    await enrollments.save_pending(pending)
    return pending


async def claim_enrollment(
    enrollments: EnrollmentRepository,
    audit: AuditLog,
    claim_limiter: HostRateLimiter,
    owner_id: uuid.UUID,
    command: ClaimEnrollmentCommand,
    now: datetime,
    monotonic_now: float,
    request_id: str = "",
) -> PendingEnrollment:
    """Owner claims a pending enrollment from the mobile app after scanning the QR."""
    if not claim_limiter.allow(str(owner_id), monotonic_now):
        raise HostError("rate_limited")

    try:
        alias = HostAlias(command.alias)
        code_bytes = decode_b64url_16b(command.enrollment_code)
    except ValueError as error:
        raise HostError("validation_failed") from error

    code_sha256 = hashlib.sha256(code_bytes).hexdigest()
    pending = await enrollments.get_pending_by_code_hash(code_sha256)
    if pending is None or pending.is_expired(now):
        if pending is not None:
            await enrollments.delete_pending(pending.pending_id)
        raise HostError("not_found")

    if pending.owner_id is not None and pending.owner_id != owner_id:
        raise HostError("not_found")

    if command.key_fingerprint is not None and command.key_fingerprint != pending.key_fingerprint:
        await enrollments.delete_pending(pending.pending_id)
        await audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=AuditAction.HOST_ENROLL_REJECTED,
                occurred_at=now,
                subject=str(pending.pending_id),
                request_id=request_id,
                metadata={
                    "reason": "fingerprint_mismatch",
                    "platform": pending.platform.value,
                },
            )
        )
        raise HostError("conflict")

    claimed = PendingEnrollment(
        pending_id=pending.pending_id,
        code_sha256=pending.code_sha256,
        public_key=pending.public_key,
        key_fingerprint=pending.key_fingerprint,
        sas=pending.sas,
        platform=pending.platform,
        os_version=pending.os_version,
        agent_version=pending.agent_version,
        hostname=pending.hostname,
        created_at=pending.created_at,
        expires_at=pending.expires_at,
        owner_id=owner_id,
        alias=alias,
    )
    await enrollments.save_pending(claimed)
    return claimed


async def confirm_enrollment(
    enrollments: EnrollmentRepository,
    hosts: HostRepository,
    hub: HostSessionHub,
    audit: AuditLog,
    owner_id: uuid.UUID,
    pending_id: HostId,
    now: datetime,
    request_id: str = "",
) -> tuple[Host, list[Grant]]:
    """Owner confirms the 6-digit SAS match: creates the Host, initial grants,
    audits `host.enrolled`, and signals the waiting enrollment WebSocket."""
    pending = await enrollments.get_pending_by_id(pending_id)
    if pending is None or pending.is_expired(now):
        if pending is not None:
            await enrollments.delete_pending(pending.pending_id)
        existing = await hosts.get_host(pending_id)
        if (
            existing is not None
            and existing.owner_id == owner_id
            and existing.status == HostStatus.ACTIVE
        ):
            grants = await hosts.get_grants(existing.host_id)
            return existing, grants
        raise HostError("not_found")

    if pending.owner_id != owner_id or pending.alias is None:
        raise HostError("not_found")

    host = Host(
        host_id=pending.pending_id,
        owner_id=owner_id,
        alias=pending.alias,
        platform=pending.platform,
        hostname=pending.hostname,
        os_version=pending.os_version,
        agent_version=pending.agent_version,
        public_key=pending.public_key,
        status=HostStatus.ACTIVE,
        created_at=now,
        last_seen_at=now,
    )
    initial_grants = list(default_initial_grants(host.host_id, now))
    await hosts.save_host(host)
    await hosts.set_grants(host.host_id, initial_grants)
    await enrollments.delete_pending(pending.pending_id)

    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.HOST_ENROLLED,
            occurred_at=now,
            subject=str(host.host_id),
            request_id=request_id,
            metadata={"platform": host.platform.value},
        )
    )
    await hub.notify_enrolled(str(host.host_id), host.host_id, host.alias)
    return host, initial_grants


async def open_host_session(
    hosts: HostRepository,
    verifier: SignatureVerifier,
    audit: AuditLog,
    command: OpenHostSessionCommand,
    now: datetime,
) -> Host:
    """Authenticates a purpose-2 session proof for an enrolled host."""
    host = await hosts.get_host(command.host_id)
    if host is None:
        raise HostSessionRejected(UNAUTHORIZED)
    if host.status != HostStatus.ACTIVE:
        await audit.record(
            AuditEvent(
                owner_id=host.owner_id,
                action=AuditAction.HOST_REJECTED,
                occurred_at=now,
                subject=str(host.host_id),
                metadata={"reason": "revoked", "platform": host.platform.value},
            )
        )
        raise HostSessionRejected(UNAUTHORIZED)

    message = host_proof_message(
        nonce=command.nonce,
        purpose=HostProofPurpose.SESSION,
        payload=host.public_key,
        host_id=host.host_id,
    )
    if not verifier.verify(host.public_key, message, command.signature):
        await audit.record(
            AuditEvent(
                owner_id=host.owner_id,
                action=AuditAction.HOST_REJECTED,
                occurred_at=now,
                subject=str(host.host_id),
                metadata={"reason": "invalid_signature", "platform": host.platform.value},
            )
        )
        raise HostSessionRejected(UNAUTHORIZED)

    updated = Host(
        host_id=host.host_id,
        owner_id=host.owner_id,
        alias=host.alias,
        platform=command.platform,
        hostname=host.hostname,
        os_version=command.os_version,
        agent_version=command.agent_version,
        public_key=host.public_key,
        status=HostStatus.ACTIVE,
        created_at=host.created_at,
        last_seen_at=now,
        lan_fingerprint=host.lan_fingerprint,
        lan_port=host.lan_port,
    )
    await hosts.save_host(updated)
    await audit.record(
        AuditEvent(
            owner_id=host.owner_id,
            action=AuditAction.HOST_CONNECTED,
            occurred_at=now,
            subject=str(host.host_id),
            metadata={"platform": updated.platform.value},
        )
    )
    return updated


async def record_duplicate_session_rejection(
    hosts: HostRepository,
    audit: AuditLog,
    host_id: HostId,
    now: datetime,
) -> None:
    host = await hosts.get_host(host_id)
    if host is None:
        return
    await audit.record(
        AuditEvent(
            owner_id=host.owner_id,
            action=AuditAction.HOST_REJECTED,
            occurred_at=now,
            subject=str(host.host_id),
            metadata={"reason": "duplicate_session", "platform": host.platform.value},
        )
    )


async def touch_host_session(
    hosts: HostRepository,
    host_id: HostId,
    now: datetime,
) -> HostSessionClose | None:
    host = await hosts.get_host(host_id)
    if host is None or host.status != HostStatus.ACTIVE:
        return UNAUTHORIZED
    updated = Host(
        host_id=host.host_id,
        owner_id=host.owner_id,
        alias=host.alias,
        platform=host.platform,
        hostname=host.hostname,
        os_version=host.os_version,
        agent_version=host.agent_version,
        public_key=host.public_key,
        status=host.status,
        created_at=host.created_at,
        last_seen_at=now,
        lan_fingerprint=host.lan_fingerprint,
        lan_port=host.lan_port,
    )
    await hosts.save_host(updated)
    return None


async def close_host_session(
    hosts: HostRepository,
    host_id: HostId,
    now: datetime,
) -> None:
    host = await hosts.get_host(host_id)
    if host is None:
        return
    updated = Host(
        host_id=host.host_id,
        owner_id=host.owner_id,
        alias=host.alias,
        platform=host.platform,
        hostname=host.hostname,
        os_version=host.os_version,
        agent_version=host.agent_version,
        public_key=host.public_key,
        status=host.status,
        created_at=host.created_at,
        last_seen_at=now,
        lan_fingerprint=host.lan_fingerprint,
        lan_port=host.lan_port,
    )
    await hosts.save_host(updated)


async def update_host_lan_peer(
    hosts: HostRepository,
    host_id: HostId,
    enabled: bool,
    cert_fingerprint: str | None,
    port: int | None,
    now: datetime,
) -> Host | None:
    """Stores or clears the Desktop Agent's LAN certificate fingerprint & port (`UDC-7` S7.4)."""
    host = await hosts.get_host(host_id)
    if host is None or host.status != HostStatus.ACTIVE:
        return None
    updated = Host(
        host_id=host.host_id,
        owner_id=host.owner_id,
        alias=host.alias,
        platform=host.platform,
        hostname=host.hostname,
        os_version=host.os_version,
        agent_version=host.agent_version,
        public_key=host.public_key,
        status=host.status,
        created_at=host.created_at,
        last_seen_at=now,
        lan_fingerprint=cert_fingerprint if enabled else None,
        lan_port=port if enabled else None,
    )
    await hosts.save_host(updated)
    return updated


async def owner_phone_public_keys_b64(
    hosts: HostRepository,
    owner_id: uuid.UUID,
) -> list[str]:
    """Returns base64-encoded 65-byte uncompressed P-256 public keys of the owner's
    active mobile (`android`/`ios`) hosts (`UDC-7` S7.4, S7.5)."""
    owned = await hosts.list_hosts(owner_id)
    keys: list[str] = []
    for h in owned:
        if h.status == HostStatus.ACTIVE and h.platform in (
            HostPlatform.ANDROID,
            HostPlatform.IOS,
        ):
            encoded = base64.b64encode(h.public_key).decode("ascii")
            if encoded not in keys:
                keys.append(encoded)
    return keys


async def record_manifest(
    hosts: HostRepository,
    host_id: HostId,
    tools: list[ToolDescriptor],
    now: datetime,
) -> Manifest:
    manifest = Manifest(host_id=host_id, tools=tuple(tools), updated_at=now)
    await hosts.save_manifest(manifest)
    return manifest


async def self_enroll_host(
    hosts: HostRepository,
    verifier: SignatureVerifier,
    audit: AuditLog,
    claim_limiter: HostRateLimiter,
    challenge_store: SelfEnrollmentChallengeStore,
    owner_id: uuid.UUID,
    command: SelfEnrollHostCommand,
    now: datetime,
    monotonic_now: float,
    request_id: str = "",
) -> tuple[Host, list[Grant], Manifest | None]:
    """Enrolls (or updates) a phone host directly from an authenticated mobile session
    using a purpose-1 ECDSA P-256 proof over a challenge nonce (`UDC-5` §5.2)."""
    if not claim_limiter.allow(str(owner_id), monotonic_now):
        raise HostError("rate_limited")

    challenge = challenge_store.consume(str(owner_id), monotonic_now)
    if challenge is None:
        raise HostError("unauthorized", "missing or expired self-enrollment challenge")

    try:
        alias = HostAlias(command.alias)
        validate_public_key(command.public_key)
    except ValueError as error:
        raise HostError("validation_failed") from error

    if len(command.signature) != SIGNATURE_SIZE:
        raise HostError("validation_failed", "proof signature must be 64 bytes")

    candidate_payloads = (
        challenge.code,
        bytes(16),
        owner_id.bytes,
    )
    verified = False
    for payload in candidate_payloads:
        msg = host_proof_message(
            nonce=challenge.nonce,
            purpose=HostProofPurpose.ENROLL,
            payload=payload,
        )
        if verifier.verify(command.public_key, msg, command.signature):
            verified = True
            break

    if not verified:
        await audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=AuditAction.HOST_ENROLL_REJECTED,
                occurred_at=now,
                subject="self",
                request_id=request_id,
                metadata={
                    "reason": "invalid_signature",
                    "platform": command.platform.value,
                },
            )
        )
        raise HostError("unauthorized", "invalid self-enrollment proof")

    parsed_tools: list[ToolDescriptor] | None = None
    if command.tools is not None:
        try:
            parsed_tools = [ToolDescriptor.from_wire(t) for t in command.tools]
        except ValueError as error:
            raise HostError("validation_failed", str(error)) from error

    resolved_hostname = (
        (command.hostname.strip() if command.hostname else "")
        or alias.value[:32]
        or f"{command.platform.value}-host"
    )[:32]

    owned = await hosts.list_hosts(owner_id)
    existing = next(
        (
            h
            for h in owned
            if h.status == HostStatus.ACTIVE and h.public_key == command.public_key
        ),
        None,
    )

    if existing is not None:
        host = Host(
            host_id=existing.host_id,
            owner_id=owner_id,
            alias=alias,
            platform=command.platform,
            hostname=resolved_hostname,
            os_version=command.os_version[:64],
            agent_version=command.agent_version[:32],
            public_key=command.public_key,
            status=HostStatus.ACTIVE,
            created_at=existing.created_at,
            last_seen_at=now,
            lan_fingerprint=existing.lan_fingerprint,
            lan_port=existing.lan_port,
        )
        await hosts.save_host(host)
        grants = await hosts.get_grants(host.host_id)
        if not grants:
            grants = list(default_initial_grants(host.host_id, now))
            await hosts.set_grants(host.host_id, grants)
    else:
        host = Host(
            host_id=HostId.generate(),
            owner_id=owner_id,
            alias=alias,
            platform=command.platform,
            hostname=resolved_hostname,
            os_version=command.os_version[:64],
            agent_version=command.agent_version[:32],
            public_key=command.public_key,
            status=HostStatus.ACTIVE,
            created_at=now,
            last_seen_at=now,
        )
        grants = list(default_initial_grants(host.host_id, now))
        await hosts.save_host(host)
        await hosts.set_grants(host.host_id, grants)

    if parsed_tools is not None:
        manifest: Manifest | None = await record_manifest(
            hosts, host.host_id, parsed_tools, now
        )
    else:
        manifest = await hosts.get_manifest(host.host_id)

    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.HOST_ENROLLED,
            occurred_at=now,
            subject=str(host.host_id),
            request_id=request_id,
            metadata={
                "platform": host.platform.value,
                "mode": "self",
            },
        )
    )
    return host, grants, manifest


async def update_host_manifest(
    hosts: HostRepository,
    owner_id: uuid.UUID,
    host_id: HostId,
    raw_tools: list[dict[str, object]],
    now: datetime,
) -> Manifest:
    """Owner-scoped manifest update for a self-enrolled mobile host."""
    host = await hosts.get_host(host_id)
    if host is None or host.owner_id != owner_id or host.status != HostStatus.ACTIVE:
        raise HostError("not_found")
    try:
        parsed_tools = [ToolDescriptor.from_wire(t) for t in raw_tools]
    except ValueError as error:
        raise HostError("validation_failed", str(error)) from error
    return await record_manifest(hosts, host_id, parsed_tools, now)


async def list_hosts(
    hosts: HostRepository,
    hub: HostSessionHub,
    owner_id: uuid.UUID,
) -> list[HostSummaryView]:
    owned = await hosts.list_hosts(owner_id)
    views: list[HostSummaryView] = []
    for host in owned:
        if host.status != HostStatus.ACTIVE:
            continue
        grants = await hosts.get_grants(host.host_id)
        manifest = await hosts.get_manifest(host.host_id)
        views.append(
            HostSummaryView(
                host=host,
                grants=grants,
                manifest=manifest,
                online=hub.is_online(str(host.host_id)),
            )
        )
    return views


async def revoke_host(
    hosts: HostRepository,
    hub: HostSessionHub,
    audit: AuditLog,
    owner_id: uuid.UUID,
    host_id: HostId,
    now: datetime,
    request_id: str = "",
) -> None:
    host = await hosts.get_host(host_id)
    if host is None or host.owner_id != owner_id or host.status != HostStatus.ACTIVE:
        raise HostError("not_found")
    revoked = Host(
        host_id=host.host_id,
        owner_id=host.owner_id,
        alias=host.alias,
        platform=host.platform,
        hostname=host.hostname,
        os_version=host.os_version,
        agent_version=host.agent_version,
        public_key=host.public_key,
        status=HostStatus.REVOKED,
        created_at=host.created_at,
        last_seen_at=now,
    )
    await hosts.save_host(revoked)
    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.HOST_REVOKED,
            occurred_at=now,
            subject=str(host_id),
            request_id=request_id,
            metadata={"platform": host.platform.value},
        )
    )
    await hub.disconnect_host(str(host_id), reason="revoked")


async def set_grants(
    hosts: HostRepository,
    audit: AuditLog,
    owner_id: uuid.UUID,
    host_id: HostId,
    grants_input: list[tuple[ToolFamily, RiskTier, bool] | GrantUpdateSpec],
    now: datetime,
    request_id: str = "",
    *,
    suspend_all: bool | None = None,
) -> list[Grant]:
    host = await hosts.get_host(host_id)
    if host is None or host.owner_id != owner_id or host.status != HostStatus.ACTIVE:
        raise HostError("not_found")
    if not grants_input and suspend_all is None:
        raise HostError("validation_failed", "grants or suspendAll must be provided")

    existing = {g.family: g for g in await hosts.get_grants(host_id)}

    if suspend_all is not None:
        for fam, prev in list(existing.items()):
            existing[fam] = Grant(
                host_id=host_id,
                family=fam,
                max_risk=prev.max_risk,
                enabled=prev.enabled,
                updated_at=now,
                expires_at=prev.expires_at,
                suspended_at=now if suspend_all else None,
            )

    for item in grants_input:
        if isinstance(item, GrantUpdateSpec):
            family = item.family
            max_risk = item.max_risk
            enabled = item.enabled
            expires_at = item.expires_at
            prev_grant = existing.get(family)
            if item.suspended is True:
                suspended_at: datetime | None = (
                    prev_grant.suspended_at
                    if prev_grant is not None and prev_grant.suspended_at is not None
                    else now
                )
            elif item.suspended is False:
                suspended_at = None
            else:
                suspended_at = prev_grant.suspended_at if prev_grant is not None else None
        else:
            family, max_risk, enabled = item
            expires_at = None
            suspended_at = None
        existing[family] = Grant(
            host_id=host_id,
            family=family,
            max_risk=max_risk,
            enabled=enabled,
            updated_at=now,
            expires_at=expires_at,
            suspended_at=suspended_at,
        )
    updated_list = sorted(existing.values(), key=lambda g: g.family.value)
    await hosts.set_grants(host_id, updated_list)
    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.HOST_GRANT_CHANGED,
            occurred_at=now,
            subject=str(host_id),
            request_id=request_id,
            metadata={
                "count": len(grants_input),
                "platform": host.platform.value,
            },
        )
    )
    return updated_list


def _is_tool_granted(
    tool: ToolDescriptor, grants: list[Grant], now: datetime | None = None
) -> bool:
    if tool.risk == RiskTier.R0:
        return True
    for grant in grants:
        if grant.family == tool.family:
            return grant.is_active(now) and grant.max_risk.allows(tool.risk)
    return False


async def invoke_tool(
    hosts: HostRepository,
    hub: HostSessionHub,
    audit: AuditLog,
    owner_id: uuid.UUID,
    host_id: HostId,
    tool_name: str,
    arguments: dict[str, object],
    now: datetime,
    *,
    deadline_ms: int = 10_000,
    include_dev_approval: bool = False,
    request_id: str = "",
) -> ToolInvocationResult:
    """Dev/test tool invocation path (UDC-3 §5.4; UDC-4 wraps with real owner approvals)."""
    try:
        validate_tool_name(tool_name)
    except ValueError as error:
        raise HostError("invalid_argument") from error

    host = await hosts.get_host(host_id)
    if host is None or host.owner_id != owner_id or host.status != HostStatus.ACTIVE:
        raise HostError("not_found")

    manifest = await hosts.get_manifest(host_id)
    descriptor = manifest.find_tool(tool_name) if manifest is not None else None
    if descriptor is None:
        raise HostError("tool_not_found")

    grants = await hosts.get_grants(host_id)
    invocation_id = str(uuid.uuid4())

    if not _is_tool_granted(descriptor, grants, now):
        await audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=AuditAction.HOST_TOOL_DENIED,
                occurred_at=now,
                subject=str(host_id),
                request_id=request_id,
                metadata={
                    "tool": descriptor.name,
                    "risk": descriptor.risk.value,
                    "status": "denied",
                    "reason": "grant_denied",
                    "platform": host.platform.value,
                },
            )
        )
        return ToolInvocationResult(
            invocation_id=invocation_id,
            tool_name=descriptor.name,
            risk=descriptor.risk,
            status="denied",
            is_error=True,
            structured_content={},
            text="denied: grant_denied",
            duration_ms=0,
            error_code="grant_denied",
        )

    if descriptor.risk.level >= RiskTier.R2.level and not include_dev_approval:
        await audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=AuditAction.HOST_TOOL_DENIED,
                occurred_at=now,
                subject=str(host_id),
                request_id=request_id,
                metadata={
                    "tool": descriptor.name,
                    "risk": descriptor.risk.value,
                    "status": "denied",
                    "reason": "approval_required",
                    "platform": host.platform.value,
                },
            )
        )
        return ToolInvocationResult(
            invocation_id=invocation_id,
            tool_name=descriptor.name,
            risk=descriptor.risk,
            status="denied",
            is_error=True,
            structured_content={},
            text="denied: approval_required",
            duration_ms=0,
            error_code="approval_required",
        )

    if not hub.is_online(str(host_id)):
        raise HostError("host_offline")

    meta: dict[str, object] = {
        "xuna/invocationId": invocation_id,
        "xuna/deadlineMs": deadline_ms,
    }
    if descriptor.risk.level >= RiskTier.R2.level and include_dev_approval:
        expires_at = (now + timedelta(seconds=APPROVAL_TTL_S)).astimezone(UTC).isoformat()
        meta["xuna/approval"] = {
            "approvalId": str(uuid.uuid4()),
            "argsSha256": args_sha256(arguments),
            "expiresAt": expires_at,
        }

    call_params: dict[str, object] = {
        "name": descriptor.name,
        "arguments": arguments,
        "_meta": meta,
    }

    try:
        raw_result = await hub.send_request(
            str(host_id),
            "tools/call",
            call_params,
            timeout_s=max(deadline_ms / 1000.0, 0.5),
        )
    except TimeoutError:
        await audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=AuditAction.HOST_TOOL_INVOKED,
                occurred_at=now,
                subject=str(host_id),
                request_id=request_id,
                metadata={
                    "tool": descriptor.name,
                    "risk": descriptor.risk.value,
                    "status": "timeout",
                    "durationMs": deadline_ms,
                    "platform": host.platform.value,
                },
            )
        )
        return ToolInvocationResult(
            invocation_id=invocation_id,
            tool_name=descriptor.name,
            risk=descriptor.risk,
            status="timeout",
            is_error=True,
            structured_content={},
            text="timeout",
            duration_ms=deadline_ms,
            error_code="deadline_exceeded",
        )

    result_meta = raw_result.get("_meta")
    status = "failed"
    duration_ms = 0
    error_code: str | None = None
    if isinstance(result_meta, dict):
        if isinstance(result_meta.get("xuna/status"), str):
            status = str(result_meta["xuna/status"])
        if type(result_meta.get("xuna/durationMs")) is int:
            duration_ms = int(result_meta["xuna/durationMs"])
        if isinstance(result_meta.get("xuna/errorCode"), str):
            error_code = str(result_meta["xuna/errorCode"])

    is_error = bool(raw_result.get("isError", status != "ok"))
    sc_raw = raw_result.get("structuredContent")
    structured_content = dict(sc_raw) if isinstance(sc_raw, dict) else {}

    text = status
    content_raw = raw_result.get("content")
    if isinstance(content_raw, list) and content_raw:
        first = content_raw[0]
        if isinstance(first, dict) and isinstance(first.get("text"), str):
            text = str(first["text"])

    audit_action = (
        AuditAction.HOST_TOOL_DENIED if status == "denied" else AuditAction.HOST_TOOL_INVOKED
    )
    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=audit_action,
            occurred_at=now,
            subject=str(host_id),
            request_id=request_id,
            metadata={
                "tool": descriptor.name,
                "risk": descriptor.risk.value,
                "status": status,
                "durationMs": max(duration_ms, 0),
                "platform": host.platform.value,
            },
        )
    )

    return ToolInvocationResult(
        invocation_id=invocation_id,
        tool_name=descriptor.name,
        risk=descriptor.risk,
        status=status,
        is_error=is_error,
        structured_content=structured_content,
        text=text,
        duration_ms=duration_ms,
        error_code=error_code,
    )
