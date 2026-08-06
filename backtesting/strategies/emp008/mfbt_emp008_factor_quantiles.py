from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .mfbt_emp008_factor_pipeline import PreparedEmp008Factors
from .mfbt_emp008_factor_registry import FactorDirection, factor_definitions_for_set


@unique
class QuantileWeighting(StrEnum):
    EQUAL = "equal_weight"
    MARKET_CAP = "market_cap_weight"


@dataclass(frozen=True, slots=True)
class Emp008FactorQuantileResult:
    monthly_returns: pd.DataFrame
    portfolio_weights: pd.DataFrame
    rank_ic: pd.DataFrame
    cumulative_returns: pd.DataFrame
    summary: pd.DataFrame


def run_emp008_factor_quantiles(
    *,
    prepared: PreparedEmp008Factors,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    q: int = 5,
) -> Emp008FactorQuantileResult:
    directions = {
        definition.id.value: definition.direction
        for definition in factor_definitions_for_set(prepared.factor_set_definition.id)
    }
    return evaluate_factor_quantiles(
        factors=prepared.alpha_factors,
        directions=directions,
        close=prepared.close,
        market_cap=prepared.market_cap,
        universe=prepared.universe,
        monthly_dates=prepared.monthly_dates,
        start=start,
        end=end,
        q=q,
    )


def evaluate_factor_quantiles(
    *,
    factors: Mapping[str, pd.DataFrame],
    directions: Mapping[str, FactorDirection],
    close: pd.DataFrame,
    market_cap: pd.DataFrame,
    universe: pd.DataFrame,
    monthly_dates: Sequence[pd.Timestamp],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    q: int,
) -> Emp008FactorQuantileResult:
    if q < 2:
        raise ValueError("q must be at least 2")

    if not factors:
        raise ValueError("at least one factor is required")

    normalized_monthly_dates = tuple(pd.Timestamp(date) for date in monthly_dates)
    if len(normalized_monthly_dates) < 2:
        raise ValueError("at least two monthly dates are required")

    factor_names = list(factors)
    missing_directions = sorted(name for name in factor_names if name not in directions)
    if missing_directions:
        missing_text = ", ".join(missing_directions)
        raise ValueError(f"missing directions for factor(s): {missing_text}")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    aligned_close = close.astype(float)
    aligned_market_cap = market_cap.reindex(index=aligned_close.index, columns=aligned_close.columns).astype(float)
    aligned_universe = universe.reindex(index=aligned_close.index, columns=aligned_close.columns).fillna(False).astype(bool)
    aligned_factors = {
        name: frame.reindex(index=aligned_close.index, columns=aligned_close.columns).astype(float)
        for name, frame in factors.items()
    }

    monthly_return_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    rank_ic_rows: list[dict[str, object]] = []

    for signal_date, return_date in zip(normalized_monthly_dates[:-1], normalized_monthly_dates[1:]):
        if return_date < start_ts or return_date > end_ts:
            continue
        signal_prices = aligned_close.loc[signal_date]
        return_prices = aligned_close.loc[return_date]
        signal_market_cap = aligned_market_cap.loc[signal_date]
        signal_universe = aligned_universe.loc[signal_date]
        for factor_name, frame in aligned_factors.items():
            direction = FactorDirection(directions[factor_name])
            signal_values = frame.loc[signal_date]
            eligibility = (
                signal_universe
                & signal_values.map(np.isfinite)
                & signal_prices.map(np.isfinite)
                & return_prices.map(np.isfinite)
                & signal_prices.gt(0.0)
                & signal_market_cap.map(np.isfinite)
                & signal_market_cap.gt(0.0)
            )
            eligible_names = signal_values.index[eligibility]
            if len(eligible_names) == 0:
                continue

            eligible_signals = signal_values.loc[eligible_names]
            eligible_signal_prices = signal_prices.loc[eligible_names]
            eligible_return_prices = return_prices.loc[eligible_names]
            next_returns = eligible_return_prices.divide(eligible_signal_prices).sub(1.0)
            memberships = _build_quantile_memberships(eligible_signals, q=q)
            if not memberships:
                continue

            for weighting in QuantileWeighting:
                quantile_returns: dict[str, float] = {}
                quantile_counts: dict[str, int] = {}
                for quantile, names in memberships.items():
                    quantile_weights = _quantile_weights(
                        names,
                        weighting=weighting,
                        signal_market_cap=signal_market_cap,
                    )
                    quantile_return = float(next_returns.loc[names].mul(quantile_weights).sum())
                    quantile_returns[quantile] = quantile_return
                    quantile_counts[quantile] = len(names)
                    monthly_return_rows.append(
                        {
                            "signal_date": signal_date,
                            "return_date": return_date,
                            "factor": factor_name,
                            "weighting": weighting,
                            "portfolio": quantile,
                            "return": quantile_return,
                            "constituent_count": len(names),
                        }
                    )
                    for ticker, weight in quantile_weights.items():
                        weight_rows.append(
                            {
                                "signal_date": signal_date,
                                "return_date": return_date,
                                "factor": factor_name,
                                "weighting": weighting,
                                "quantile": quantile,
                                "ticker": ticker,
                                "weight": float(weight),
                            }
                        )

                low_quantile = "Q1"
                high_quantile = f"Q{q}"
                if low_quantile in quantile_returns and high_quantile in quantile_returns:
                    high_minus_low = quantile_returns[high_quantile] - quantile_returns[low_quantile]
                    monthly_return_rows.append(
                        {
                            "signal_date": signal_date,
                            "return_date": return_date,
                            "factor": factor_name,
                            "weighting": weighting,
                            "portfolio": "high_minus_low",
                            "return": high_minus_low,
                            "constituent_count": quantile_counts[low_quantile] + quantile_counts[high_quantile],
                        }
                    )
                    preferred_minus_avoided = (
                        high_minus_low
                        if direction is FactorDirection.HIGH
                        else quantile_returns[low_quantile] - quantile_returns[high_quantile]
                    )
                    monthly_return_rows.append(
                        {
                            "signal_date": signal_date,
                            "return_date": return_date,
                            "factor": factor_name,
                            "weighting": weighting,
                            "portfolio": "preferred_minus_avoided",
                            "return": preferred_minus_avoided,
                            "constituent_count": quantile_counts[low_quantile] + quantile_counts[high_quantile],
                        }
                    )

            rank_ic = _spearman_rank_ic(eligible_signals, next_returns)
            rank_ic_rows.append(
                {
                    "signal_date": signal_date,
                    "return_date": return_date,
                    "factor": factor_name,
                    "rank_ic": rank_ic,
                    "directional_rank_ic": rank_ic if direction is FactorDirection.HIGH else -rank_ic,
                    "n_obs": len(eligible_names),
                }
            )

    if not monthly_return_rows:
        factor_text = ", ".join(factor_names)
        raise ValueError(
            "no factor quantile observations for "
            f"{factor_text} in requested range {start_ts.date().isoformat()} to {end_ts.date().isoformat()}"
        )

    monthly_returns = pd.DataFrame(monthly_return_rows).sort_values(
        ["signal_date", "return_date", "factor", "weighting", "portfolio"],
        kind="mergesort",
    ).reset_index(drop=True)
    portfolio_weights = pd.DataFrame(weight_rows).sort_values(
        ["signal_date", "return_date", "factor", "weighting", "quantile", "ticker"],
        kind="mergesort",
    ).reset_index(drop=True)
    rank_ic = pd.DataFrame(rank_ic_rows).sort_values(
        ["signal_date", "return_date", "factor"],
        kind="mergesort",
    ).reset_index(drop=True)

    return Emp008FactorQuantileResult(
        monthly_returns=monthly_returns,
        portfolio_weights=portfolio_weights,
        rank_ic=rank_ic,
        cumulative_returns=_empty_cumulative_returns_frame(),
        summary=_empty_summary_frame(),
    )


def _build_quantile_memberships(signals: pd.Series, *, q: int) -> dict[str, list[str]]:
    ordered = pd.DataFrame(
        {
            "signal": signals.astype(float),
            "ticker_key": signals.index.map(str),
        },
        index=signals.index,
    ).sort_values(["signal", "ticker_key"], kind="mergesort")
    ordered_names = ordered.index.tolist()
    if not ordered_names:
        return {}
    bucket_count = min(q, len(ordered_names))
    buckets = np.array_split(np.array(ordered_names, dtype=object), bucket_count)
    return {
        f"Q{bucket_index}": [str(name) if not isinstance(name, str) else name for name in bucket.tolist()]
        for bucket_index, bucket in enumerate(buckets, start=1)
        if len(bucket) > 0
    }


def _quantile_weights(
    names: list[str],
    *,
    weighting: QuantileWeighting,
    signal_market_cap: pd.Series,
) -> pd.Series:
    index = pd.Index(names, dtype=object)
    if weighting is QuantileWeighting.EQUAL:
        return pd.Series(1.0 / len(names), index=index, dtype=float)
    if weighting is QuantileWeighting.MARKET_CAP:
        values = signal_market_cap.loc[names].astype(float)
        total = float(values.sum())
        return values.divide(total)
    raise ValueError(f"unsupported weighting: {weighting}")


def _spearman_rank_ic(signals: pd.Series, next_returns: pd.Series) -> float:
    if len(signals) < 2 or signals.nunique(dropna=True) < 2 or next_returns.nunique(dropna=True) < 2:
        return float("nan")
    return float(signals.corr(next_returns, method="spearman"))


def _empty_cumulative_returns_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": pd.Series(dtype="datetime64[ns]"),
            "return_date": pd.Series(dtype="datetime64[ns]"),
            "factor": pd.Series(dtype="object"),
            "weighting": pd.Series(dtype="object"),
            "portfolio": pd.Series(dtype="object"),
            "cumulative_return": pd.Series(dtype="float64"),
        }
    )


def _empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor": pd.Series(dtype="object"),
            "weighting": pd.Series(dtype="object"),
            "portfolio": pd.Series(dtype="object"),
            "metric": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )


__all__ = [
    "Emp008FactorQuantileResult",
    "QuantileWeighting",
    "evaluate_factor_quantiles",
    "run_emp008_factor_quantiles",
]
