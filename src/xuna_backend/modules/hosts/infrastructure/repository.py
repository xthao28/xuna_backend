"""SQLAlchemy repositories for the Host Registry & XTP v1 module (UDC-3 §5.4)."""

import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

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
)
from xuna_backend.modules.hosts.infrastructure.models import (
    HostEnrollmentModel,
    HostGrantModel,
    HostManifestModel,
    HostModel,
)


def _to_pending(row: HostEnrollmentModel) -> PendingEnrollment:
    return PendingEnrollment(
        pending_id=HostId(row.pending_id),
        code_sha256=row.code_sha256,
        public_key=bytes(row.public_key),
        key_fingerprint=row.key_fingerprint,
        sas=row.sas,
        platform=HostPlatform(row.platform),
        os_version=row.os_version,
        agent_version=row.agent_version,
        hostname=row.hostname,
        created_at=row.created_at,
        expires_at=row.expires_at,
        owner_id=row.owner_id,
        alias=HostAlias(row.alias) if row.alias is not None else None,
    )


def _to_host(row: HostModel) -> Host:
    return Host(
        host_id=HostId(row.host_id),
        owner_id=row.owner_id,
        alias=HostAlias(row.alias),
        platform=HostPlatform(row.platform),
        hostname=row.hostname,
        os_version=row.os_version,
        agent_version=row.agent_version,
        public_key=bytes(row.public_key),
        status=HostStatus(row.status),
        created_at=row.created_at,
        last_seen_at=row.last_seen_at,
    )


def _to_grant(row: HostGrantModel) -> Grant:
    return Grant(
        host_id=HostId(row.host_id),
        family=ToolFamily(row.family),
        max_risk=RiskTier(row.max_risk),
        enabled=row.enabled,
        updated_at=row.updated_at,
    )


class SqlEnrollmentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_pending(self, enrollment: PendingEnrollment) -> None:
        row = await self._session.get(HostEnrollmentModel, enrollment.pending_id.value)
        if row is None:
            self._session.add(
                HostEnrollmentModel(
                    pending_id=enrollment.pending_id.value,
                    code_sha256=enrollment.code_sha256,
                    public_key=enrollment.public_key,
                    key_fingerprint=enrollment.key_fingerprint,
                    sas=enrollment.sas,
                    platform=enrollment.platform.value,
                    os_version=enrollment.os_version,
                    agent_version=enrollment.agent_version,
                    hostname=enrollment.hostname,
                    created_at=enrollment.created_at,
                    expires_at=enrollment.expires_at,
                    owner_id=enrollment.owner_id,
                    alias=enrollment.alias.value if enrollment.alias is not None else None,
                )
            )
        else:
            row.code_sha256 = enrollment.code_sha256
            row.public_key = enrollment.public_key
            row.key_fingerprint = enrollment.key_fingerprint
            row.sas = enrollment.sas
            row.platform = enrollment.platform.value
            row.os_version = enrollment.os_version
            row.agent_version = enrollment.agent_version
            row.hostname = enrollment.hostname
            row.created_at = enrollment.created_at
            row.expires_at = enrollment.expires_at
            row.owner_id = enrollment.owner_id
            row.alias = enrollment.alias.value if enrollment.alias is not None else None
        await self._session.flush()

    async def get_pending_by_code_hash(self, code_sha256: str) -> PendingEnrollment | None:
        stmt = select(HostEnrollmentModel).where(HostEnrollmentModel.code_sha256 == code_sha256)
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return _to_pending(row) if row is not None else None

    async def get_pending_by_id(self, pending_id: HostId) -> PendingEnrollment | None:
        row = await self._session.get(HostEnrollmentModel, pending_id.value)
        return _to_pending(row) if row is not None else None

    async def delete_pending(self, pending_id: HostId) -> None:
        await self._session.execute(
            delete(HostEnrollmentModel).where(HostEnrollmentModel.pending_id == pending_id.value)
        )
        await self._session.flush()


class SqlHostRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_host(self, host_id: HostId) -> Host | None:
        row = await self._session.get(HostModel, host_id.value)
        return _to_host(row) if row is not None else None

    async def save_host(self, host: Host) -> None:
        row = await self._session.get(HostModel, host.host_id.value)
        if row is None:
            self._session.add(
                HostModel(
                    host_id=host.host_id.value,
                    owner_id=host.owner_id,
                    alias=host.alias.value,
                    platform=host.platform.value,
                    hostname=host.hostname,
                    os_version=host.os_version,
                    agent_version=host.agent_version,
                    public_key=host.public_key,
                    status=host.status.value,
                    created_at=host.created_at,
                    last_seen_at=host.last_seen_at,
                )
            )
        else:
            row.owner_id = host.owner_id
            row.alias = host.alias.value
            row.platform = host.platform.value
            row.hostname = host.hostname
            row.os_version = host.os_version
            row.agent_version = host.agent_version
            row.public_key = host.public_key
            row.status = host.status.value
            row.last_seen_at = host.last_seen_at
        await self._session.flush()

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        stmt = (
            select(HostModel)
            .where(HostModel.owner_id == owner_id)
            .order_by(HostModel.created_at.asc())
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [_to_host(r) for r in rows]

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        stmt = (
            select(HostGrantModel)
            .where(HostGrantModel.host_id == host_id.value)
            .order_by(HostGrantModel.family.asc())
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [_to_grant(r) for r in rows]

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        await self._session.execute(
            delete(HostGrantModel).where(HostGrantModel.host_id == host_id.value)
        )
        for grant in grants:
            self._session.add(
                HostGrantModel(
                    grant_id=uuid.uuid4(),
                    host_id=host_id.value,
                    family=grant.family.value,
                    max_risk=grant.max_risk.value,
                    enabled=grant.enabled,
                    updated_at=grant.updated_at,
                )
            )
        await self._session.flush()

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        row = await self._session.get(HostManifestModel, host_id.value)
        if row is None:
            return None
        tools = tuple(ToolDescriptor.from_wire(item) for item in row.tools_json)
        return Manifest(
            host_id=HostId(row.host_id),
            tools=tools,
            updated_at=row.updated_at,
            schema_version=row.schema_version,
        )

    async def save_manifest(self, manifest: Manifest) -> None:
        row = await self._session.get(HostManifestModel, manifest.host_id.value)
        tools_payload = [t.to_wire() for t in manifest.tools]
        if row is None:
            self._session.add(
                HostManifestModel(
                    host_id=manifest.host_id.value,
                    schema_version=manifest.schema_version,
                    tools_json=tools_payload,
                    updated_at=manifest.updated_at,
                )
            )
        else:
            row.schema_version = manifest.schema_version
            row.tools_json = tools_payload
            row.updated_at = manifest.updated_at
        await self._session.flush()
