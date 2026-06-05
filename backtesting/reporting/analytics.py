from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import pandas as pd

SECTOR_CONTRIBUTION_METHOD_WEIGHTED_ASSET_RETURNS = "weighted-asset-return-attribution"
ROLLING_WINDOW = 252
_TRADING_PERIODS = 252
_EPSILON = 1e-12


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
    alpha: float | None = None
    beta: float | None = None
    tracking_error: float | None = None
    information_ratio: float | None = None
    downside_deviation: float = 0.0
    value_at_risk_95: float = 0.0
    conditional_value_at_risk_95: float = 0.0
    win_rate: float = 0.0
    payoff_ratio: float = 0.0
    profit_factor: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    best_year: float = 0.0
    worst_year: float = 0.0
    longest_drawdown_days: float = 0.0
    recovery_days: float = 0.0
    current_drawdown: float = 0.0
    month_hit_ratio: float = 0.0
    year_hit_ratio: float = 0.0
    correlation: float | None = None
    upside_capture: float | None = None
    downside_capture: float | None = None
    active_return: float | None = None
    active_risk: float | None = None


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
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, name="turnover"))
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
    yearly_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    yearly_excess_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    sector_contribution_method: str = ""
    sector_contribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_episodes: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_ohlc: pd.DataFrame = field(default_factory=pd.DataFrame)


def annualized_sharpe(returns: pd.Series, periods: int = _TRADING_PERIODS) -> float:
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return 0.0

    volatility = float(clean.std(ddof=0))
    if abs(volatility) < _EPSILON:
        return 0.0
    return float(clean.mean() / volatility * sqrt(periods))


def annualized_volatility(returns: pd.Series, periods: int = _TRADING_PERIODS) -> float:
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=0) * sqrt(periods))


def annualized_downside_deviation(returns: pd.Series, periods: int = _TRADING_PERIODS) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    downside = clean.clip(upper=0.0)
    return float((downside.pow(2).mean() ** 0.5) * sqrt(periods))


def monthly_return_series(returns: pd.Series, monthly_returns: pd.Series | None = None) -> pd.Series:
    if monthly_returns is not None and not monthly_returns.empty:
        return monthly_returns.astype(float).sort_index().rename("monthly_return")
    return (1.0 + returns.fillna(0.0).astype(float)).resample("ME").prod().sub(1.0).rename("monthly_return")


def yearly_return_series(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, name="yearly_return")
    return (1.0 + returns.fillna(0.0).astype(float)).resample("YE").prod().sub(1.0).rename("yearly_return")


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
    if abs(upper - lower) < _EPSILON:
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
    if benchmark_returns.empty:
        return pd.Series(dtype=float, name="yearly_excess_return")
    strategy_yearly = yearly_return_series(strategy_returns)
    benchmark_yearly = yearly_return_series(benchmark_returns)
    yearly_excess = strategy_yearly.sub(benchmark_yearly.reindex(strategy_yearly.index).fillna(0.0), fill_value=0.0)
    return yearly_excess.rename("yearly_excess_return")


def win_rate(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    return float(clean.gt(0.0).mean())


def payoff_ratio(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    gains = clean.loc[clean.gt(0.0)]
    losses = clean.loc[clean.lt(0.0)]
    if gains.empty or losses.empty:
        return 0.0
    denominator = abs(float(losses.mean()))
    if denominator < _EPSILON:
        return 0.0
    return float(gains.mean() / denominator)


def profit_factor(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    gross_profit = float(clean.loc[clean.gt(0.0)].sum())
    gross_loss = abs(float(clean.loc[clean.lt(0.0)].sum()))
    if gross_loss < _EPSILON:
        return 0.0
    return gross_profit / gross_loss


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    return float(clean.quantile(1.0 - level))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    threshold = value_at_risk(clean, level=level)
    tail = clean.loc[clean.le(threshold)]
    if tail.empty:
        return threshold
    return float(tail.mean())


def skewness(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    value = float(clean.skew()) if len(clean) >= 3 else 0.0
    return 0.0 if pd.isna(value) else value


def kurtosis(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    value = float(clean.kurt()) if len(clean) >= 4 else 0.0
    return 0.0 if pd.isna(value) else value


def hit_ratio(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    return float(clean.gt(0.0).mean())


def capture_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series, *, upside: bool) -> float | None:
    aligned = pd.concat(
        [strategy_returns.astype(float).rename("strategy"), benchmark_returns.astype(float).rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return None
    if upside:
        window = aligned.loc[aligned["benchmark"].gt(0.0)]
    else:
        window = aligned.loc[aligned["benchmark"].lt(0.0)]
    if window.empty:
        return None
    benchmark_mean = float(window["benchmark"].mean())
    if abs(benchmark_mean) < _EPSILON:
        return None
    return float(window["strategy"].mean() / benchmark_mean)


def rolling_return(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    return returns.rolling(window=window, min_periods=window).apply(
        lambda values: float((1.0 + pd.Series(values)).prod() - 1.0),
        raw=False,
    ).rename("rolling_return")


def rolling_volatility(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    return returns.rolling(window=window, min_periods=window).std(ddof=0).mul(sqrt(_TRADING_PERIODS)).rename(
        "rolling_volatility"
    )


def rolling_downside_deviation(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    return returns.rolling(window=window, min_periods=window).apply(
        lambda values: annualized_downside_deviation(pd.Series(values)),
        raw=False,
    ).rename("rolling_downside_deviation")


def max_duration(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).max())
