from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import pandas as pd

SECTOR_CONTRIBUTION_METHOD_WEIGHTED_ASSET_RETURNS = "weighted-asset-return-attribution"
ROLLING_WINDOW = 252


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    cumulative_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    final_equity: float
    avg_turnover: float
    alpha: float
    beta: float
    tracking_error: float
    information_ratio: float


@dataclass(frozen=True, slots=True)
class RollingMetrics:
    window: int
    series: dict[str, pd.Series]


@dataclass(frozen=True, slots=True)
class DrawdownStats:
    underwater: pd.Series
    episodes: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ExposureSnapshot:
    holdings_count: pd.Series
    latest_holdings: pd.DataFrame
    latest_holdings_winners: pd.DataFrame = field(default_factory=pd.DataFrame)
    latest_holdings_losers: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True, slots=True)
class SectorSnapshot:
    latest_weighted: pd.Series
    latest_count: pd.Series
    concentration: pd.Series


@dataclass(frozen=True, slots=True)
class ResearchSnapshot:
    monthly_heatmap: pd.DataFrame = field(default_factory=pd.DataFrame)
    return_distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_return_distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    yearly_excess_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    sector_contribution_method: str = ""
    sector_contribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_episodes: pd.DataFrame = field(default_factory=pd.DataFrame)


def annualized_sharpe(returns: pd.Series, periods: int = 252) -> float:
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return 0.0

    volatility = float(clean.std(ddof=0))
    if abs(volatility) < 1e-12:
        return 0.0
    return float(clean.mean() / volatility * sqrt(periods))


def monthly_return_series(returns: pd.Series, monthly_returns: pd.Series | None = None) -> pd.Series:
    if monthly_returns is not None and not monthly_returns.empty:
        return monthly_returns.astype(float).sort_index().rename("monthly_return")
    return (1.0 + returns.fillna(0.0).astype(float)).resample("ME").prod().sub(1.0).rename("monthly_return")


def build_monthly_heatmap(returns: pd.Series, monthly_returns: pd.Series | None = None) -> pd.DataFrame:
    monthly = monthly_return_series(returns, monthly_returns)
    if monthly.empty:
        return pd.DataFrame()

    frame = monthly.to_frame("value")
    frame["year"] = frame.index.year.astype(int)
    frame["month"] = frame.index.month.astype(int)
    heatmap = frame.pivot_table(index="year", columns="month", values="value", aggfunc="last")
    heatmap.index.name = "year"
    heatmap.columns.name = "month"
    return heatmap.sort_index().sort_index(axis=1)


def build_return_distribution(returns: pd.Series, bins: int = 20) -> pd.DataFrame:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return pd.DataFrame(columns=["start", "end", "count", "frequency"])

    lower = float(clean.min())
    upper = float(clean.max())
    if abs(upper - lower) < 1e-12:
        padding = max(abs(lower) * 0.01, 1e-6)
        edges = np.array([lower - padding, upper + padding], dtype=float)
    else:
        effective_bins = max(1, min(bins, len(clean)))
        unique_values = np.sort(clean.unique())
        if len(unique_values) <= effective_bins:
            midpoints = [(left + right) / 2.0 for left, right in zip(unique_values[:-1], unique_values[1:])]
            leading_gap = max(abs(unique_values[0]) * 0.01, 1e-6)
            trailing_gap = max(abs(unique_values[-1]) * 0.01, 1e-6)
            edges = np.array(
                [unique_values[0] - leading_gap, *midpoints, unique_values[-1] + trailing_gap],
                dtype=float,
            )
        else:
            edges = np.linspace(lower, upper, num=effective_bins + 1, dtype=float)

    buckets = pd.cut(clean, bins=edges, include_lowest=True, duplicates="drop")
    counts = buckets.value_counts(sort=False)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "start": [float(interval.left) for interval in counts.index],
            "end": [float(interval.right) for interval in counts.index],
            "count": counts.astype(int).values,
            "frequency": [float(count) / total for count in counts.values],
        }
    )


def build_yearly_excess_returns(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    strategy_yearly = (1.0 + strategy_returns.fillna(0.0).astype(float)).resample("YE").prod().sub(1.0)
    benchmark_yearly = (1.0 + benchmark_returns.fillna(0.0).astype(float)).resample("YE").prod().sub(1.0)
    yearly_excess = strategy_yearly.sub(benchmark_yearly.reindex(strategy_yearly.index).fillna(0.0), fill_value=0.0)
    return yearly_excess.rename("yearly_excess_return")
