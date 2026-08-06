from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

import pandas as pd

from backtesting.data import MarketData

from .data import Emp008Config, load_emp008_market
from .factor_registry import (
    FactorDefinition,
    FactorSetDefinition,
    factor_definitions_for_set,
    get_factor_set_definition,
)
from .factors import build_raw_factors
from .preprocess import build_sector_active_exposures, preprocess_factor_frame


@dataclass(frozen=True, slots=True)
class PreparedEmp008Factors:
    config: Emp008Config
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


def validate_prepared_emp008_factors(
    prepared: PreparedEmp008Factors,
    *,
    config: Emp008Config | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    required_dates: Iterable[str | pd.Timestamp] = (),
) -> None:
    if config is not None and config != prepared.config:
        diff_parts: list[str] = []
        for field in fields(Emp008Config):
            prepared_value = getattr(prepared.config, field.name)
            requested_value = getattr(config, field.name)
            if prepared_value != requested_value:
                diff_parts.append(
                    f"{field.name}: prepared={prepared_value!r} requested={requested_value!r}"
                )
        raise ValueError(
            "prepared/config mismatch: " + "; ".join(diff_parts)
        )

    if prepared.close.empty or not prepared.monthly_dates:
        raise ValueError("prepared bundle is empty and cannot be reused")

    close_index = pd.DatetimeIndex(prepared.close.index)
    available_start = close_index.min()
    available_end = close_index.max()

    monthly_dates = set(pd.to_datetime(prepared.monthly_dates))
    missing_required_dates = sorted(
        {
            pd.Timestamp(date)
            for date in required_dates
            if pd.Timestamp(date) not in monthly_dates
        }
    )
    if missing_required_dates:
        missing_text = ", ".join(date.date().isoformat() for date in missing_required_dates)
        available_text = ", ".join(date.date().isoformat() for date in prepared.monthly_dates)
        raise ValueError(
            "prepared bundle is missing required target dates: "
            f"{missing_text}; available monthly dates: {available_text}"
        )

    if start is not None:
        requested_start = pd.Timestamp(start)
        if requested_start < available_start:
            raise ValueError(
                "prepared data range does not cover requested start: "
                f"requested {requested_start.date().isoformat()} "
                f"but available range is {available_start.date().isoformat()} to {available_end.date().isoformat()}"
            )
    if end is not None:
        requested_end = pd.Timestamp(end)
        if (
            requested_end > available_end
            and requested_end.to_period("M") != available_end.to_period("M")
        ):
            raise ValueError(
                "prepared data range does not cover requested end: "
                f"requested {requested_end.date().isoformat()} "
                f"but available range is {available_start.date().isoformat()} to {available_end.date().isoformat()}"
            )
        available_monthly_end = max(monthly_dates)
        if requested_end.to_period("M") > available_monthly_end.to_period("M"):
            raise ValueError(
                "prepared monthly output range does not cover requested end month: "
                f"requested {requested_end.strftime('%Y-%m')} "
                f"but available monthly range ends at {available_monthly_end.strftime('%Y-%m-%d')}"
            )


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


def prepare_emp008_factors(market: MarketData, config: Emp008Config) -> PreparedEmp008Factors:
    factor_set_definition = get_factor_set_definition(config.factor_set)
    factor_definitions = factor_definitions_for_set(config.factor_set)
    raw_factors = build_raw_factors(market, config)

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
    config: Emp008Config,
) -> PreparedEmp008Factors:
    market = load_emp008_market(parquet_dir=parquet_dir, start=start, end=end, config=config)
    return prepare_emp008_factors(market, config)
