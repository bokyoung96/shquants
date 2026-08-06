from __future__ import annotations

import pandas as pd

from backtesting.data import MarketData

from .data import Emp008Config
from .factor_registry import factor_definitions_for_set


def build_raw_factors(market: MarketData, config: Emp008Config) -> dict[str, pd.DataFrame]:
    return {
        definition.id.value: definition.builder(market, config)
        for definition in factor_definitions_for_set(config.factor_set)
    }


__all__ = ["build_raw_factors"]
