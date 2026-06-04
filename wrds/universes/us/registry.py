from __future__ import annotations

from typing import Iterable, Protocol

from .sources import FactSetLinks, StockNames
from .strategies import UniverseBuilder

try:
    from ...core.registry import NamedRegistry
except ImportError:  # pragma: no cover - direct script compatibility
    from core.registry import NamedRegistry


class Named(Protocol):
    name: str


class USRegistry:
    def __init__(self, items: Iterable[Named]) -> None:
        self._registry = NamedRegistry(items, label="US component")

    @classmethod
    def default(cls, client) -> "USRegistry":
        return cls((StockNames(client), FactSetLinks(client), UniverseBuilder()))

    def get(self, name: str):
        return self._registry.get(name)
