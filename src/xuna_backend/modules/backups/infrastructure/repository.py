"""SQL implementation of BackupRepository (CLD-20)."""

import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.backups.domain.entities import (
    BackupManifest,
    DeletionTombstone,
    EncryptedData,
)
from xuna_backend.modules.backups.infrastructure.models import (
    BackupManifestRow,
    DeletionTombstoneRow,
)
from xuna_backend.modules.identity.domain.pet_id import PetId


class SqlBackupRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_backup(self, manifest: BackupManifest) -> None:
        row = BackupManifestRow(
            backup_id=manifest.backup_id,
            pet_id=manifest.pet_id.value,
            owner_id=manifest.owner_id,
            brain_id=manifest.brain_id,
            snapshot_generation=manifest.snapshot_generation,
            schema_version=manifest.schema_version,
            payload_sha256=manifest.payload_sha256,
            encrypted_data={
                "algorithm": manifest.encrypted_data.algorithm,
                "keyId": manifest.encrypted_data.key_id,
                "nonce": manifest.encrypted_data.nonce,
                "ciphertext": manifest.encrypted_data.ciphertext,
                "tag": manifest.encrypted_data.tag,
            },
            created_at=manifest.created_at,
        )
        self._session.add(row)
        await self._session.flush()

    async def get_backup(self, backup_id: uuid.UUID) -> BackupManifest | None:
        result = await self._session.execute(
            select(BackupManifestRow).where(BackupManifestRow.backup_id == backup_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_manifest(row)

    async def list_backups(self, pet_id: PetId, limit: int = 50) -> list[BackupManifest]:
        result = await self._session.execute(
            select(BackupManifestRow)
            .where(BackupManifestRow.pet_id == pet_id.value)
            .order_by(BackupManifestRow.created_at.desc())
            .limit(limit)
        )
        return [self._to_manifest(row) for row in result.scalars().all()]

    async def delete_backup(self, backup_id: uuid.UUID) -> None:
        await self._session.execute(
            delete(BackupManifestRow).where(BackupManifestRow.backup_id == backup_id)
        )
        await self._session.flush()

    async def add_tombstone(self, tombstone: DeletionTombstone) -> None:
        stmt = (
            insert(DeletionTombstoneRow)
            .values(
                tombstone_id=tombstone.tombstone_id,
                pet_id=tombstone.pet_id.value,
                owner_id=tombstone.owner_id,
                item_type=tombstone.item_type,
                item_id=tombstone.item_id,
                reason=tombstone.reason,
                deleted_at=tombstone.deleted_at,
            )
            .on_conflict_do_nothing(index_elements=["pet_id", "item_type", "item_id"])
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def list_tombstones(self, pet_id: PetId, limit: int = 100) -> list[DeletionTombstone]:
        result = await self._session.execute(
            select(DeletionTombstoneRow)
            .where(DeletionTombstoneRow.pet_id == pet_id.value)
            .order_by(DeletionTombstoneRow.deleted_at.desc())
            .limit(limit)
        )
        return [self._to_tombstone(row) for row in result.scalars().all()]

    async def is_tombstoned(self, pet_id: PetId, item_type: str, item_id: str) -> bool:
        result = await self._session.execute(
            select(DeletionTombstoneRow.tombstone_id)
            .where(
                (DeletionTombstoneRow.pet_id == pet_id.value)
                & (
                    (DeletionTombstoneRow.item_type == item_type)
                    | (DeletionTombstoneRow.item_type == "memory")
                )
                & (
                    (DeletionTombstoneRow.item_id == item_id)
                    | (DeletionTombstoneRow.item_id == "*")
                )
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    def _to_manifest(self, row: BackupManifestRow) -> BackupManifest:
        data: dict[str, Any] = row.encrypted_data
        return BackupManifest(
            backup_id=row.backup_id,
            pet_id=PetId(row.pet_id),
            owner_id=row.owner_id,
            brain_id=row.brain_id,
            snapshot_generation=row.snapshot_generation,
            schema_version=row.schema_version,
            payload_sha256=row.payload_sha256,
            encrypted_data=EncryptedData(
                algorithm=data.get("algorithm", "aes-256-gcm"),
                key_id=data.get("keyId", "primary"),
                nonce=data.get("nonce", ""),
                ciphertext=data.get("ciphertext", ""),
                tag=data.get("tag", ""),
            ),
            created_at=row.created_at,
        )

    def _to_tombstone(self, row: DeletionTombstoneRow) -> DeletionTombstone:
        return DeletionTombstone(
            tombstone_id=row.tombstone_id,
            pet_id=PetId(row.pet_id),
            owner_id=row.owner_id,
            item_type=row.item_type,
            item_id=row.item_id,
            reason=row.reason,
            deleted_at=row.deleted_at,
        )
