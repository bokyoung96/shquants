from __future__ import annotations

import pandas as pd

from backtesting.data import MarketData

from .data import Emp008Config
from .factor_builders import _sector_relative_retail_flow, align_like_close, month_end_observations
from .factor_registry import factor_definitions_for_set


def build_raw_factors(market: MarketData, config: Emp008Config) -> dict[str, pd.DataFrame]:
    return {
        definition.id.value: definition.builder(market, config)
        for definition in factor_definitions_for_set(config.factor_set)
    }


__all__ = ["_sector_relative_retail_flow", "align_like_close", "build_raw_factors", "month_end_observations"]
