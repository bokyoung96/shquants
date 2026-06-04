from __future__ import annotations

from typing import Iterable

from .sources import OptionLinks, OptionMeta, OptionPrices

try:
    from ...core.registry import Named, NamedRegistry
except ImportError:  # pragma: no cover - direct script compatibility
    from core.registry import Named, NamedRegistry


class OptionRegistry:
    def __init__(self, items: Iterable[Named]) -> None:
        self._registry = NamedRegistry(items, label="option component")

    @classmethod
    def default(cls, client) -> "OptionRegistry":
        return cls((OptionLinks(client), OptionMeta(client), OptionPrices(client)))

    def get(self, name: str):
        return self._registry.get(name)
