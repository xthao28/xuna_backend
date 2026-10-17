"""Configuration-selected adapters, one per capability.

Like xiaozhi's `selected_module`, but typed and explicit: adapters register a
factory under a name; configuration chooses the name; unknown names fail at
startup, not on the first user request.
"""

from collections.abc import Callable, Mapping
from typing import Any

from xuna_backend.modules.cognition.domain.types import Capability

Factory = Callable[[Mapping[str, Any]], object]


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[tuple[Capability, str], Factory] = {}

    def register(self, capability: Capability, name: str, factory: Factory) -> None:
        key = (capability, name)
        if key in self._factories:
            raise ValueError(f"duplicate adapter {capability}:{name}")
        self._factories[key] = factory

    def names(self, capability: Capability) -> list[str]:
        return sorted(n for c, n in self._factories if c == capability)

    def create(self, capability: Capability, name: str, options: Mapping[str, Any]) -> object:
        """Callers `cast` the result to the capability's port Protocol."""
        factory = self._factories.get((capability, name))
        if factory is None:
            raise LookupError(f"no adapter {capability}:{name}; have {self.names(capability)}")
        return factory(options)
