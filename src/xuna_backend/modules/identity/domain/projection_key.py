"""Projection ordering key `(PetId, snapshotGeneration)` (Proposed, CLD-01).

`snapshotGeneration` is the device's u32 LifeSnapshot v2 generation. It wraps,
so ordering uses serial-number arithmetic exactly like firmware
`SnapshotStore::recover` (`int32_t(a - b) > 0`). Keys of different pets are
never ordered against each other.
"""

from dataclasses import dataclass

from xuna_backend.modules.identity.domain.pet_id import PetId

_U32 = 1 << 32


@dataclass(frozen=True, slots=True)
class ProjectionKey:
    pet_id: PetId
    snapshot_generation: int

    def __post_init__(self) -> None:
        if not 0 <= self.snapshot_generation < _U32:
            raise ValueError("snapshot_generation must be a u32")

    def is_newer_than(self, other: "ProjectionKey") -> bool:
        if self.pet_id != other.pet_id:
            raise ValueError("cannot order projections of different pets")
        delta = (self.snapshot_generation - other.snapshot_generation) % _U32
        return 0 < delta < _U32 // 2
