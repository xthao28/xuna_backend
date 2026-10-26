"""In-memory MQTT broker for testing and local simulation (CLD-40, CLD-41, E14)."""

from collections.abc import Callable
from typing import Any


class InMemoryMqttBroker:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []
        self._handlers: dict[str, list[Callable[[str, str], Any]]] = {}

    async def publish(
        self,
        topic: str,
        payload: str | bytes,
        qos: int = 1,
        retain: bool = False,
    ) -> None:
        payload_str = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        self.published.append(
            {
                "topic": topic,
                "payload": payload_str,
                "qos": qos,
                "retain": retain,
            }
        )
        for sub_topic, callbacks in self._handlers.items():
            if self._matches(sub_topic, topic):
                for cb in callbacks:
                    cb(topic, payload_str)

    def subscribe(self, topic: str, callback: Callable[[str, str], Any]) -> None:
        self._handlers.setdefault(topic, []).append(callback)

    def clear(self) -> None:
        self.published.clear()
        self._handlers.clear()

    @staticmethod
    def _matches(pattern: str, topic: str) -> bool:
        if pattern == "#" or pattern == topic:
            return True
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")
        if len(pattern_parts) != len(topic_parts) and pattern_parts[-1] != "#":
            return False
        for p, t in zip(pattern_parts, topic_parts, strict=False):
            if p == "#":
                return True
            if p != "+" and p != t:
                return False
        return len(pattern_parts) == len(topic_parts)
