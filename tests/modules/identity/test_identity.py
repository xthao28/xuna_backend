import pytest

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.identity.domain.projection_key import ProjectionKey

BOARD_PET = "d945c77026fd4f4a8a899bbff4aeed86"  # real board PetId, 2026-10-17


def test_accepts_device_uuid_v4() -> None:
    assert str(PetId(BOARD_PET)) == BOARD_PET


@pytest.mark.parametrize(
    "value",
    [
        "",
        BOARD_PET.upper(),
        BOARD_PET[:-1],
        "d945c770-26fd-4f4a-8a89-9bbff4aeed86",
        BOARD_PET[:12] + "1" + BOARD_PET[13:],  # version nibble
        BOARD_PET[:16] + "c" + BOARD_PET[17:],  # variant
    ],
)
def test_rejects_malformed_pet_ids(value: str) -> None:
    with pytest.raises(ValueError):
        PetId(value)


def key(generation: int, pet: str = BOARD_PET) -> ProjectionKey:
    return ProjectionKey(PetId(pet), generation)


@pytest.mark.parametrize(
    ("newer", "older"),
    [(39, 38), (1, 0), (0, 0xFFFFFFFF), (5, 0xFFFFFFF0), (0x7FFFFFFF, 0)],
)
def test_generation_order_is_wrap_safe(newer: int, older: int) -> None:
    assert key(newer).is_newer_than(key(older))
    assert not key(older).is_newer_than(key(newer))


def test_equal_generation_is_not_newer() -> None:
    assert not key(38).is_newer_than(key(38))


def test_rejects_cross_pet_and_out_of_range() -> None:
    other = "000102030405460788090a0b0c0d0e0f"
    with pytest.raises(ValueError):
        key(1).is_newer_than(key(0, other))
    with pytest.raises(ValueError):
        key(1 << 32)
    with pytest.raises(ValueError):
        key(-1)
