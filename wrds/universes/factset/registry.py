from __future__ import annotations

from typing import Iterable

from .sources import FactSetSource
from .strategies import LatestLinkStrategy

try:
    from ...core.registry import Named, NamedRegistry
except ImportError:  # pragma: no cover - direct script compatibility
    from core.registry import Named, NamedRegistry


class UniverseRegistry:
    def __init__(self, items: Iterable[Named]) -> None:
        self._registry = NamedRegistry(items, label="universe component")

    @classmethod
    def default(cls, client) -> "UniverseRegistry":
        return cls((FactSetSource(client), LatestLinkStrategy()))

    def get(self, name: str):
        return self._registry.get(name)
