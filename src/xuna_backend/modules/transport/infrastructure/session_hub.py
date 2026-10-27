"""Process-local registry of device-direct sessions (DC-3). One API process
today; several processes would need a shared registry (e.g. PostgreSQL
advisory locks or a pub/sub), decided with public hosting (D19)."""

from collections.abc import Awaitable, Callable


class InMemoryDeviceSessionHub:
    def __init__(self) -> None:
        self._live: dict[str, tuple[object, Callable[[], Awaitable[None]]]] = {}

    async def claim(
        self, pet_id: str, token: object, replace: Callable[[], Awaitable[None]]
    ) -> None:
        previous = self._live.get(pet_id)
        self._live[pet_id] = (token, replace)
        if previous is not None and previous[0] is not token:
            await previous[1]()

    def release(self, pet_id: str, token: object) -> bool:
        current = self._live.get(pet_id)
        if current is None or current[0] is not token:
            return False
        del self._live[pet_id]
        return True

    def live(self) -> int:
        return len(self._live)
