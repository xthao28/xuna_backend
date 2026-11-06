"""Create, list and revoke standing approvals (ADR-0014 D1).

Creation is fail-closed: the call must be eligible, the expiry within 30 days, and the request
must carry a signature from an enrolled phone key over `xuna.preapproval.v1`. With no signature
verifier configured, nothing can be created, so a backend without that check never accepts one.
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from xuna_backend.modules.actions.application.ports import StandingApprovalRepository
from xuna_backend.modules.actions.domain.entities import ActionSummary
from xuna_backend.modules.actions.domain.schema_validator import (
    ArgumentValidationError,
    SchemaDefinitionError,
    SchemaValidator,
)
from xuna_backend.modules.actions.domain.standing_approval import (
    STANDING_DEFAULT_LIFETIME,
    STANDING_MAX_LIFETIME,
    STANDING_MAX_USES,
    StandingApproval,
    StandingApprovalError,
    check_standing_eligibility,
)
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.cognition.domain.types import Locale
from xuna_backend.modules.hosts.application.ports import HostRepository
from xuna_backend.modules.hosts.domain.crypto import (
    args_sha256,
    preapproval_proof_message,
    validate_public_key,
)
from xuna_backend.modules.hosts.domain.entities import HostId, HostPlatform, HostStatus
from xuna_backend.modules.identity.application.ports import SignatureVerifier


class StandingApprovalService:
    def __init__(
        self,
        *,
        host_repo: HostRepository,
        repository: StandingApprovalRepository,
        audit: AuditLog,
        verifier: SignatureVerifier | None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._host_repo = host_repo
        self._repository = repository
        self._audit = audit
        self._verifier = verifier
        self._clock = clock or (lambda: datetime.now(UTC))

    async def create(
        self,
        *,
        owner_id: uuid.UUID,
        approval_id: uuid.UUID,
        host_id: HostId,
        tool: str,
        arguments: dict[str, object],
        expires_at_epoch_s: int | None,
        max_uses: int | None,
        signature: bytes,
        signer_public_key: bytes,
        locale: Locale = Locale.VI,
        request_id: str = "",
    ) -> StandingApproval:
        now = self._clock()
        if self._verifier is None:
            raise StandingApprovalError("signature_unverifiable")
        if max_uses is not None and not 1 <= max_uses <= STANDING_MAX_USES:
            raise StandingApprovalError("invalid_max_uses")
        expires_at = (
            now + STANDING_DEFAULT_LIFETIME
            if expires_at_epoch_s is None
            else datetime.fromtimestamp(expires_at_epoch_s, UTC)
        )
        if not now < expires_at <= now + STANDING_MAX_LIFETIME:
            raise StandingApprovalError("invalid_expiry")

        host = await self._host_repo.get_host(host_id)
        if host is None or host.owner_id != owner_id or host.status != HostStatus.ACTIVE:
            raise StandingApprovalError("host_not_found")
        manifest = await self._host_repo.get_manifest(host_id)
        descriptor = manifest.find_tool(tool) if manifest is not None else None
        if descriptor is None:
            raise StandingApprovalError("unknown_tool")
        try:
            SchemaValidator.validate_schema_definition(descriptor.input_schema)
            clean_args, _ = SchemaValidator.validate_arguments(
                descriptor.input_schema, arguments, allow_target=False
            )
        except (SchemaDefinitionError, ArgumentValidationError) as error:
            raise StandingApprovalError("invalid_args") from error
        check_standing_eligibility(descriptor, clean_args)

        digest = args_sha256(clean_args)
        await self._verify_phone_signature(
            owner_id=owner_id,
            message=preapproval_proof_message(
                str(approval_id),
                tool,
                digest,
                str(host_id),
                int(expires_at.timestamp()),
                max_uses or 0,
            ),
            signature=signature,
            signer_public_key=signer_public_key,
        )

        existing = await self._repository.get(owner_id, approval_id)
        if existing is not None:
            same = (
                existing.host_id == host_id
                and existing.tool == tool
                and existing.args_sha256 == digest
                and existing.max_uses == max_uses
                and int(existing.expires_at.timestamp()) == int(expires_at.timestamp())
            )
            if not same:
                raise StandingApprovalError("standing_approval_conflict")
            return existing

        approval = StandingApproval(
            approval_id=approval_id,
            owner_id=owner_id,
            host_id=host_id,
            target_alias=str(host.alias),
            tool=tool,
            args_sha256=digest,
            summary=ActionSummary.format_pending(tool, clean_args, str(host.alias), locale),
            created_at=now,
            expires_at=expires_at,
            max_uses=max_uses,
            signature=signature,
            signer_public_key=signer_public_key,
        )
        await self._repository.save(approval)
        await self._audit.record(
            AuditEvent(
                owner_id=owner_id,
                action=AuditAction.HOST_PREAPPROVAL_CREATED,
                occurred_at=now,
                subject=str(host_id),
                request_id=request_id,
                metadata={
                    "approvalId": str(approval_id),
                    "tool": tool,
                    "risk": descriptor.risk.value,
                    "maxUses": max_uses or 0,
                },
            )
        )
        return approval

    async def list_active(self, owner_id: uuid.UUID) -> list[StandingApproval]:
        return await self._repository.list_active(owner_id, self._clock())

    async def revoke(
        self, *, owner_id: uuid.UUID, approval_id: uuid.UUID, request_id: str = ""
    ) -> bool:
        now = self._clock()
        existing = await self._repository.get(owner_id, approval_id)
        if existing is None:
            return False
        revoked = await self._repository.revoke(owner_id, approval_id, now)
        if revoked:
            await self._audit.record(
                AuditEvent(
                    owner_id=owner_id,
                    action=AuditAction.HOST_PREAPPROVAL_REVOKED,
                    occurred_at=now,
                    subject=str(existing.host_id),
                    request_id=request_id,
                    metadata={"approvalId": str(approval_id), "tool": existing.tool},
                )
            )
        return True

    async def _verify_phone_signature(
        self,
        *,
        owner_id: uuid.UUID,
        message: bytes,
        signature: bytes,
        signer_public_key: bytes,
    ) -> None:
        if len(signature) != 64:
            raise StandingApprovalError("signature_invalid")
        try:
            validate_public_key(signer_public_key)
        except ValueError as error:
            raise StandingApprovalError("signature_invalid") from error
        owned = await self._host_repo.list_hosts(owner_id)
        phone_keys = [
            h.public_key
            for h in owned
            if h.status == HostStatus.ACTIVE
            and h.platform in (HostPlatform.ANDROID, HostPlatform.IOS)
        ]
        verifier = self._verifier
        if verifier is None or signer_public_key not in phone_keys:
            raise StandingApprovalError("signature_invalid")
        if not verifier.verify(signer_public_key, message, signature):
            raise StandingApprovalError("signature_invalid")
