"""Opt-in pet cryptographic co-signing (`ProofPurpose.APPROVE`) & signed E-Stop (`STOP`).

Implements ADR-0020 (D6 owner opt-in per category, default OFF):
- Per-owner co-signing policy & single-use challenge/nonce stores (RAM-only).
- Generic R3 category classification without hardcoded application/toolchain names (H5).
- Dual-check ECDSA P-256 verification against the owner's bound active pet key.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock

from xuna_backend.modules.hosts.domain.crypto import validate_public_key
from xuna_backend.modules.hosts.domain.entities import RiskTier
from xuna_backend.modules.identity.domain.device_proof import (
    NONCE_SIZE,
    SIGNATURE_SIZE,
    ProofPurpose,
    approve_proof_payload,
    cosign_short_code,
    proof_message,
    stop_proof_payload,
)
from xuna_backend.modules.identity.domain.entities import Pet
from xuna_backend.modules.identity.domain.pet_id import PetId


class CosignCategory(StrEnum):
    """High-consequence R3 categories eligible for optional pet hardware co-signing."""

    OUTSIDE_WORKSPACE_R3 = "outside_workspace_r3"
    BULK_DELETE = "bulk_delete"
    REMOTE_PUSH = "remote_push"
    DRAFT_APPLY = "draft_apply"


@dataclass(frozen=True, slots=True)
class OwnerCosignPolicy:
    """Owner's co-signing preferences (D6 default OFF across every category)."""

    enabled_categories: frozenset[CosignCategory] = frozenset()
    require_signed_board_stop: bool = False

    def requires_for(self, category: CosignCategory | None) -> bool:
        return category is not None and category in self.enabled_categories


class InMemoryCosignPolicyStore:
    """Thread-safe in-memory store for owner pet co-signing policies."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._policies: dict[uuid.UUID, OwnerCosignPolicy] = {}

    def get_policy(self, owner_id: uuid.UUID) -> OwnerCosignPolicy:
        with self._lock:
            return self._policies.get(owner_id, OwnerCosignPolicy())

    def set_policy(self, owner_id: uuid.UUID, policy: OwnerCosignPolicy) -> OwnerCosignPolicy:
        with self._lock:
            self._policies[owner_id] = policy
            return policy


@dataclass(frozen=True, slots=True)
class PendingCosignChallenge:
    """Single-use co-signing challenge generated alongside a pending R3 approval."""

    approval_id: uuid.UUID
    owner_id: uuid.UUID
    nonce: bytes
    nonce_hex: str
    target_sha256_hex: str
    short_code: str
    category: CosignCategory


class InMemoryCosignChallengeStore:
    """Tracks per-approval pet co-signing challenges in RAM."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._by_approval: dict[uuid.UUID, PendingCosignChallenge] = {}

    def create(
        self,
        *,
        approval_id: uuid.UUID,
        owner_id: uuid.UUID,
        target_sha256_hex: str,
        category: CosignCategory,
        nonce: bytes | None = None,
    ) -> PendingCosignChallenge:
        raw_nonce = nonce if nonce is not None else os.urandom(NONCE_SIZE)
        if len(raw_nonce) != NONCE_SIZE:
            raise ValueError("nonce must be 32 bytes")
        target_bytes = bytes.fromhex(target_sha256_hex)
        code = cosign_short_code(raw_nonce, target_bytes)
        challenge = PendingCosignChallenge(
            approval_id=approval_id,
            owner_id=owner_id,
            nonce=raw_nonce,
            nonce_hex=raw_nonce.hex(),
            target_sha256_hex=target_sha256_hex,
            short_code=code,
            category=category,
        )
        with self._lock:
            self._by_approval[approval_id] = challenge
        return challenge

    def get(self, approval_id: uuid.UUID) -> PendingCosignChallenge | None:
        with self._lock:
            return self._by_approval.get(approval_id)

    def remove(self, approval_id: uuid.UUID) -> None:
        with self._lock:
            self._by_approval.pop(approval_id, None)


@dataclass
class InMemoryProofNonceStore:
    """Single-use `(owner_id, nonce_hex)` replay guard bounded in memory."""

    max_entries: int = 4096
    _seen: dict[tuple[uuid.UUID, str], None] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def try_consume(self, owner_id: uuid.UUID, nonce_hex: str) -> bool:
        key = (owner_id, nonce_hex.lower())
        with self._lock:
            if key in self._seen:
                return False
            if len(self._seen) >= self.max_entries:
                oldest = next(iter(self._seen))
                self._seen.pop(oldest, None)
            self._seen[key] = None
            return True


_DELETE_VERBS = frozenset({"rm", "del", "erase", "rmdir", "rd", "unlink", "remove-item"})
_PUSH_VERBS = frozenset({"push", "publish", "upload", "deploy"})


def classify_cosign_category(
    tool_name: str,
    arguments: Mapping[str, object],
    *,
    risk: RiskTier,
    preflight_reason: str | None = None,
) -> CosignCategory | None:
    """Maps an R3 call + optional preflight reason into a D6 `CosignCategory` or `None`."""
    if risk.level < RiskTier.R3.level:
        return None
    if tool_name == "draft.apply":
        return CosignCategory.DRAFT_APPLY
    reason = (preflight_reason or "").strip().lower()
    if tool_name.startswith("fs.") or "outside_workspace" in reason or "secret" in reason:
        return CosignCategory.OUTSIDE_WORKSPACE_R3
    cmd = str(arguments.get("command") or arguments.get("cmd") or "").strip().lower()
    if cmd:
        tokens = [tok.strip("\"'") for tok in cmd.replace("&&", " ").replace(";", " ").split()]
        if any(tok.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] in _DELETE_VERBS for tok in tokens):
            return CosignCategory.BULK_DELETE
        if any(tok in _PUSH_VERBS for tok in tokens):
            return CosignCategory.REMOTE_PUSH
    if tool_name in ("exec.run", "capability.run") and reason:
        return CosignCategory.OUTSIDE_WORKSPACE_R3
    return None


@dataclass(frozen=True, slots=True)
class PetApproveProofInput:
    pet_id: str
    nonce_hex: str
    generation: int
    short_code: str
    signature: bytes
    public_key: bytes


@dataclass(frozen=True, slots=True)
class PetStopProofInput:
    pet_id: str
    nonce_hex: str
    generation: int
    signature: bytes
    public_key: bytes


def _select_active_pet(
    pets: list[Pet],
    owner_id: uuid.UUID,
    pet_id_hex: str,
    public_key: bytes,
) -> Pet | None:
    try:
        validate_public_key(public_key)
        expected_id = PetId(pet_id_hex.strip().lower())
    except (TypeError, ValueError):
        return None
    for pet in pets:
        if (
            pet.owner_id == owner_id
            and pet.pet_id == expected_id
            and pet.device_revoked_at is None
            and pet.device_public_key == public_key
        ):
            return pet
    return None


def build_verified_approve_message(
    *,
    proof: PetApproveProofInput,
    owner_id: uuid.UUID,
    pets: list[Pet],
    target_sha256_hex: str,
    expected_nonce_hex: str | None = None,
    expected_short_code: str | None = None,
) -> tuple[Pet, bytes] | None:
    """Validates `PetApproveProofInput` fields & returns `(pet, proof_message)` or `None`."""
    if len(proof.signature) != SIGNATURE_SIZE:
        return None
    nonce_hex = proof.nonce_hex.strip().lower()
    if expected_nonce_hex is not None and nonce_hex != expected_nonce_hex.lower():
        return None
    try:
        nonce = bytes.fromhex(nonce_hex)
        target_bytes = bytes.fromhex(target_sha256_hex)
        derived_code = cosign_short_code(nonce, target_bytes)
    except ValueError:
        return None
    if proof.short_code != derived_code:
        return None
    if expected_short_code is not None and proof.short_code != expected_short_code:
        return None
    pet = _select_active_pet(pets, owner_id, proof.pet_id, proof.public_key)
    if pet is None:
        return None
    try:
        payload = approve_proof_payload(target_bytes, proof.short_code)
        msg = proof_message(nonce, pet.pet_id, ProofPurpose.APPROVE, proof.generation, payload)
    except ValueError:
        return None
    return pet, msg


def build_verified_stop_message(
    *,
    proof: PetStopProofInput,
    owner_id: uuid.UUID,
    pets: list[Pet],
) -> tuple[Pet, bytes] | None:
    """Validates `PetStopProofInput` fields & returns `(pet, proof_message)` or `None`."""
    if len(proof.signature) != SIGNATURE_SIZE:
        return None
    try:
        nonce = bytes.fromhex(proof.nonce_hex.strip().lower())
    except ValueError:
        return None
    if len(nonce) != NONCE_SIZE:
        return None
    pet = _select_active_pet(pets, owner_id, proof.pet_id, proof.public_key)
    if pet is None:
        return None
    try:
        msg = proof_message(
            nonce, pet.pet_id, ProofPurpose.STOP, proof.generation, stop_proof_payload()
        )
    except ValueError:
        return None
    return pet, msg
