from __future__ import annotations

from backtesting.data import SourceRegistry

from .consensus import sources as consensus_sources
from .fundamentals import sources as fundamental_sources
from .indexes import sources as index_sources
from .prices import sources as price_sources


def source_registry() -> SourceRegistry:
    return SourceRegistry(
        (
            *price_sources(),
            *fundamental_sources(),
            *consensus_sources(),
            *index_sources(),
        )
    )

