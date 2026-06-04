from __future__ import annotations

import pandas as pd

from .registry import UniverseRegistry
from .sources import LINK_COLS, LinkSource, clean
from .strategies import UNIVERSE_COLS, BuildStrategy


class Universe:
    def __init__(
        self,
        client=None,
        *,
        source: LinkSource | None = None,
        strategy: BuildStrategy | None = None,
    ) -> None:
        registry = UniverseRegistry.default(client) if source is None or strategy is None else None
        self.source = source or registry.get("links")
        self.strategy = strategy or registry.get("latest")

    @classmethod
    def from_registry(cls, registry: UniverseRegistry) -> "Universe":
        return cls(source=registry.get("links"), strategy=registry.get("latest"))

    def latest(self) -> str:
        return self.source.latest()

    def links(self, *, date: str = "latest", limit: int | None = None) -> pd.DataFrame:
        return self.source.links(date=date, limit=limit)

    def build(self, links: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.build(links)


__all__ = (
    "BuildStrategy",
    "LINK_COLS",
    "LinkSource",
    "UNIVERSE_COLS",
    "Universe",
    "UniverseRegistry",
    "clean",
)
