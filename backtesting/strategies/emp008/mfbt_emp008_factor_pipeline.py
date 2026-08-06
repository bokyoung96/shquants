from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtesting.data import MarketData

from .mfbt_emp008_data import MfbtEmp008Config, load_mfbt_emp008_market
from .mfbt_emp008_factor_registry import (
    FactorDefinition,
    FactorSetDefinition,
    factor_definitions_for_set,
    get_factor_set_definition,
)
from .mfbt_emp008_factors import build_raw_mfbt_factors
from .mfbt_emp008_preprocess import build_sector_active_exposures, preprocess_factor_frame


@dataclass(frozen=True, slots=True)
class PreparedEmp008Factors:
    config: MfbtEmp008Config
    market: MarketData
    factor_set_definition: FactorSetDefinition
    raw_factors: dict[str, pd.DataFrame]
    alpha_factors: dict[str, pd.DataFrame]
    sector_factors: dict[str, pd.DataFrame]
    close: pd.DataFrame
    market_cap: pd.DataFrame
    float_market_cap: pd.DataFrame
    universe: pd.DataFrame
    sector: pd.DataFrame
    benchmark_weights: pd.DataFrame
    monthly_dates: tuple[pd.Timestamp, ...]


def common_month_end_dates(factors: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    non_empty = [set(frame.dropna(how="all").index) for frame in factors.values()]
    if not non_empty:
        return []
    return sorted(set.intersection(*non_empty))


def complete_benchmark_history(
    bm_weights: pd.DataFrame,
    float_market_cap: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    completed = bm_weights.reindex(index=float_market_cap.index, columns=float_market_cap.columns).astype(float).copy()
    official_rows = completed.fillna(0.0).sum(axis=1).gt(0.0)
    if not official_rows.any():
        return completed

    first_official_date = official_rows.index[official_rows][0]
    warmup_rows = completed.index < first_official_date
    proxy_values = float_market_cap.astype(float).where(universe).clip(lower=0.0)
    proxy_weights = proxy_values.div(proxy_values.sum(axis=1).replace(0.0, float("nan")), axis=0).fillna(0.0)
    completed.loc[warmup_rows] = proxy_weights.loc[warmup_rows]
    return completed


def neutralize_large_benchmark_weight_exposures(
    alpha_factors: dict[str, pd.DataFrame],
    bm_weights: pd.DataFrame,
    factor_definitions: tuple[FactorDefinition, ...],
    *,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    if threshold <= 0.0:
        return alpha_factors

    selected_names = [
        definition.id.value
        for definition in factor_definitions
        if definition.neutralize_large_benchmark_weight
    ]
    if not selected_names:
        return alpha_factors

    neutralized = dict(alpha_factors)
    for name in selected_names:
        if name not in neutralized:
            continue
        frame = neutralized[name]
        large_bm = bm_weights.reindex(index=frame.index, columns=frame.columns).fillna(0.0).ge(threshold)
        neutralized[name] = frame.mask(large_bm, 0.0)
    return neutralized


def prepare_emp008_factors(market: MarketData, config: MfbtEmp008Config) -> PreparedEmp008Factors:
    factor_set_definition = get_factor_set_definition(config.factor_set)
    factor_definitions = factor_definitions_for_set(config.factor_set)
    raw_factors = build_raw_mfbt_factors(market, config)

    close = market.frames["close"].astype(float)
    market_cap = market.frames["market_cap"].reindex(index=close.index, columns=close.columns).astype(float)
    float_market_cap = market.frames["float_market_cap"].reindex(index=close.index, columns=close.columns).astype(float)
    universe = market.frames["k200_yn"].reindex(index=close.index, columns=close.columns).fillna(0).astype(bool)
    sector = market.frames["sector_neutral_big"].reindex(index=close.index, columns=close.columns).ffill()
    benchmark_weights = market.frames["bm_weights"].reindex(index=close.index, columns=close.columns).astype(float)
    benchmark_weights = complete_benchmark_history(benchmark_weights, float_market_cap, universe)

    alpha_factors = {
        definition.id.value: preprocess_factor_frame(
            raw_factors[definition.id.value],
            float_market_cap,
            universe,
            rank_transform=definition.rank_transform,
            winsor_quantile=(
                getattr(config, definition.winsor_config_attr) if definition.winsor_config_attr is not None else None
            ),
            zscore_cap=(
                getattr(config, definition.zscore_cap_config_attr) if definition.zscore_cap_config_attr is not None else None
            ),
        )
        for definition in factor_definitions
    }
    alpha_factors = neutralize_large_benchmark_weight_exposures(
        alpha_factors,
        benchmark_weights,
        factor_definitions,
        threshold=config.large_bm_neutral_weight_threshold,
    )
    sector_factors = build_sector_active_exposures(sector, float_market_cap, universe)
    monthly_dates = tuple(common_month_end_dates(raw_factors))

    return PreparedEmp008Factors(
        config=config,
        market=market,
        factor_set_definition=factor_set_definition,
        raw_factors=raw_factors,
        alpha_factors=alpha_factors,
        sector_factors=sector_factors,
        close=close,
        market_cap=market_cap,
        float_market_cap=float_market_cap,
        universe=universe,
        sector=sector,
        benchmark_weights=benchmark_weights,
        monthly_dates=monthly_dates,
    )


def load_and_prepare_emp008_factors(
    parquet_dir: Path,
    start: str,
    end: str,
    config: MfbtEmp008Config,
) -> PreparedEmp008Factors:
    market = load_mfbt_emp008_market(parquet_dir=parquet_dir, start=start, end=end, config=config)
    return prepare_emp008_factors(market, config)
