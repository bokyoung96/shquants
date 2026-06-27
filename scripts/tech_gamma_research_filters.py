from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from backtesting.strategies.positivity import positivity_score

if TYPE_CHECKING:
    from scripts.run_tech_gamma_long_only import TechGammaConfig


PositivityBenchmark = Literal[
    "absolute",
    "sector_cap_weighted",
    "sector_equal_weight",
    "index_cap_weighted",
    "index_equal_weight",
]


@dataclass(frozen=True, slots=True)
class ResearchFeatureData:
    sector: pd.DataFrame | None = None
    market_cap: pd.DataFrame | None = None
    op_fwd_12m: pd.DataFrame | None = None
    foreign_flow: pd.DataFrame | None = None
    institution_flow: pd.DataFrame | None = None


FEATURE_COLUMNS = (
    "daily_positivity",
    "positivity_benchmark",
    "positivity_spread",
    "positivity_filter_ok",
    "op_revision",
    "op_sector_rank",
    "foreign_flow_to_cap",
    "institution_flow_to_cap",
    "factor_filter_ok",
    "sector_name",
)


def load_research_feature_data(parquet_dir: Path, tickers: tuple[str, ...]) -> ResearchFeatureData:
    return ResearchFeatureData(
        sector=_read_optional(parquet_dir / "qw_wics_sec_big.parquet", tickers=tickers),
        market_cap=_read_optional(parquet_dir / "qw_mktcap.parquet", tickers=tickers),
        op_fwd_12m=_read_optional(parquet_dir / "qw_op_fwd_12m.parquet", tickers=tickers),
        foreign_flow=_read_optional(parquet_dir / "qw_foreign.parquet", tickers=tickers),
        institution_flow=_read_optional(parquet_dir / "qw_institution.parquet", tickers=tickers),
    )


def apply_research_features(
    frame: pd.DataFrame,
    config: TechGammaConfig,
    data: ResearchFeatureData,
) -> pd.DataFrame:
    if not config.use_positivity and config.factor_filter == "none":
        return frame
    clean = frame.drop(columns=[column for column in FEATURE_COLUMNS if column in frame.columns])
    daily_close = clean.groupby(["ticker", "date"], sort=True)["close"].last().rename("daily_close").reset_index()
    close = daily_close.pivot(index="date", columns="ticker", values="daily_close")
    dates = pd.DatetimeIndex(close.index)
    tickers = tuple(str(column) for column in close.columns)
    positivity = positivity_score(
        close.pct_change(fill_method=None),
        lookback=config.positivity_lookback_days,
        min_periods=config.positivity_lookback_days,
    ).shift(1)
    sector = _aligned(data.sector, dates, tickers)
    cap = _aligned(data.market_cap, dates, tickers).ffill().shift(1)
    benchmark = _positivity_benchmark(positivity, sector, cap, config.positivity_benchmark)
    features = pd.DataFrame(
        {
            "daily_positivity": _stack(positivity),
            "positivity_benchmark": _stack(benchmark),
        }
    ).reset_index(names=["date", "ticker"])
    features["positivity_spread"] = features["daily_positivity"] - features["positivity_benchmark"]
    features["positivity_filter_ok"] = _positivity_filter(features, config)
    features = _add_factor_filter(features, config, data, dates, tickers, sector, cap)
    return clean.merge(features, on=["ticker", "date"], how="left", sort=False)


def _read_optional(path: Path, *, tickers: tuple[str, ...]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path, engine="pyarrow")
    columns = frame.columns.intersection(pd.Index(tickers))
    return frame.loc[:, columns]


def _aligned(source: pd.DataFrame | None, dates: pd.DatetimeIndex, tickers: tuple[str, ...]) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame(index=dates, columns=tickers)
    return source.reindex(index=dates, columns=tickers).ffill()


def _positivity_benchmark(
    positivity: pd.DataFrame,
    sector: pd.DataFrame,
    cap: pd.DataFrame,
    mode: PositivityBenchmark,
) -> pd.DataFrame:
    match mode:
        case "absolute":
            return pd.DataFrame(0.0, index=positivity.index, columns=positivity.columns)
        case "index_equal_weight":
            return _index_average(positivity, None)
        case "index_cap_weighted":
            return _index_average(positivity, cap)
        case "sector_equal_weight":
            return _sector_average(positivity, sector, None)
        case "sector_cap_weighted":
            return _sector_average(positivity, sector, cap)


def _index_average(values: pd.DataFrame, weights: pd.DataFrame | None) -> pd.DataFrame:
    if weights is None or weights.isna().all(axis=None):
        average = values.mean(axis=1)
    else:
        clean_weights = weights.where(values.notna())
        average = values.mul(clean_weights).sum(axis=1).divide(clean_weights.sum(axis=1))
    return pd.DataFrame({column: average for column in values.columns}, index=values.index)


def _sector_average(values: pd.DataFrame, sector: pd.DataFrame, weights: pd.DataFrame | None) -> pd.DataFrame:
    result = pd.DataFrame(float("nan"), index=values.index, columns=values.columns)
    for sector_name in pd.unique(sector.to_numpy().ravel()):
        if pd.isna(sector_name):
            continue
        mask = sector.eq(sector_name)
        sector_values = values.where(mask)
        if weights is None or weights.where(mask).isna().all(axis=None):
            average = sector_values.mean(axis=1)
        else:
            clean_weights = weights.where(mask & values.notna())
            average = sector_values.mul(clean_weights).sum(axis=1).divide(clean_weights.sum(axis=1))
        result = result.where(~mask, pd.DataFrame({column: average for column in values.columns}, index=values.index))
    return result


def _positivity_filter(features: pd.DataFrame, config: TechGammaConfig) -> pd.Series:
    if config.positivity_benchmark == "absolute":
        return features["daily_positivity"].ge(config.min_daily_positivity).fillna(False)
    return features["positivity_spread"].ge(config.positivity_margin).fillna(False)


def _add_factor_filter(
    features: pd.DataFrame,
    config: TechGammaConfig,
    data: ResearchFeatureData,
    dates: pd.DatetimeIndex,
    tickers: tuple[str, ...],
    sector: pd.DataFrame,
    cap: pd.DataFrame,
) -> pd.DataFrame:
    op_revision = _revision(_aligned(data.op_fwd_12m, dates, tickers), config.factor_lookback_days)
    foreign_flow = _flow_to_cap(_aligned(data.foreign_flow, dates, tickers), cap, config.factor_lookback_days)
    institution_flow = _flow_to_cap(_aligned(data.institution_flow, dates, tickers), cap, config.factor_lookback_days)
    op_rank = _sector_rank(op_revision, sector)
    stacked = pd.DataFrame(
        {
            "op_revision": _stack(op_revision),
            "op_sector_rank": _stack(op_rank),
            "foreign_flow_to_cap": _stack(foreign_flow),
            "institution_flow_to_cap": _stack(institution_flow),
            "sector_name": _stack(sector),
        }
    ).reset_index(names=["date", "ticker"])
    merged = features.merge(stacked, on=["date", "ticker"], how="left", sort=False)
    merged["factor_filter_ok"] = _factor_filter(merged, config.factor_filter).fillna(False)
    return merged


def _revision(values: pd.DataFrame, lookback: int) -> pd.DataFrame:
    numeric = values.astype(float)
    base = numeric.shift(lookback)
    return numeric.sub(base).divide(base.abs()).shift(1)


def _flow_to_cap(flow: pd.DataFrame, cap: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return flow.astype(float).rolling(lookback, min_periods=lookback).sum().shift(1).divide(cap.astype(float))


def _sector_rank(values: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    ranks = pd.DataFrame(float("nan"), index=values.index, columns=values.columns)
    for sector_name in pd.unique(sector.to_numpy().ravel()):
        if pd.isna(sector_name):
            continue
        mask = sector.eq(sector_name)
        ranks = ranks.where(~mask, values.where(mask).rank(axis=1, pct=True))
    return ranks


def _factor_filter(features: pd.DataFrame, name: str) -> pd.Series:
    match name:
        case "none":
            return pd.Series(True, index=features.index)
        case "op_revision_positive":
            return features["op_revision"].gt(0.0)
        case "op_sector_rank_positive":
            return features["op_revision"].gt(0.0) & features["op_sector_rank"].gt(0.5)
        case "foreign_flow_positive":
            return features["foreign_flow_to_cap"].gt(0.0)
        case "institution_flow_positive":
            return features["institution_flow_to_cap"].gt(0.0)
        case "op_or_flow_positive":
            return features["op_revision"].gt(0.0) | features["foreign_flow_to_cap"].gt(0.0)
        case _:
            raise KeyError(f"unknown factor filter {name!r}")


def _stack(frame: pd.DataFrame) -> pd.Series:
    return frame.stack(future_stack=True)
