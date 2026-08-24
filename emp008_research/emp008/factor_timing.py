from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factor_registry import FactorDirection


TIMING_DIAGNOSTIC_COLUMNS = (
    "rebalance_date",
    "factor",
    "direction",
    "base_weight",
    "fast_return",
    "slow_return",
    "state",
    "multiplier",
    "timed_weight",
    "last_signal_date",
)


@dataclass(frozen=True, slots=True)
class FactorTimingConfig:
    policy: str = "momentum"
    fast_lookback: int = 6
    slow_lookback: int = 12
    strong_multiplier: float = 1.25
    neutral_multiplier: float = 1.0
    weak_multiplier: float = 0.75

    def __post_init__(self) -> None:
        if self.policy != "momentum":
            raise ValueError("factor timing policy must be 'momentum'")
        if self.fast_lookback <= 0 or self.slow_lookback <= 0:
            raise ValueError("factor timing lookbacks must be positive")
        if self.fast_lookback > self.slow_lookback:
            raise ValueError("fast_lookback must not exceed slow_lookback")
        multipliers = (
            self.strong_multiplier,
            self.neutral_multiplier,
            self.weak_multiplier,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in multipliers):
            raise ValueError("factor timing multipliers must be finite and positive")


@dataclass(frozen=True, slots=True)
class FactorTimingDecision:
    weights: pd.Series
    diagnostics: pd.DataFrame


def decide_factor_timing(
    *,
    factor_returns: pd.DataFrame,
    factor_directions: Mapping[str, FactorDirection],
    base_weights: pd.Series,
    rebalance_date: pd.Timestamp,
    config: FactorTimingConfig | None,
) -> FactorTimingDecision:
    weights = _normalized_base_weights(base_weights)
    if config is None:
        return FactorTimingDecision(
            weights=weights,
            diagnostics=pd.DataFrame(columns=TIMING_DIAGNOSTIC_COLUMNS),
        )

    factors = list(weights.index.astype(str))
    missing_returns = sorted(set(factors).difference(map(str, factor_returns.columns)))
    if missing_returns:
        raise ValueError(f"missing factor return columns: {', '.join(missing_returns)}")
    missing_directions = sorted(set(factors).difference(map(str, factor_directions)))
    if missing_directions:
        raise ValueError(f"missing factor directions: {', '.join(missing_directions)}")

    history = factor_returns.copy()
    history.index = pd.to_datetime(history.index)
    if history.index.has_duplicates:
        raise ValueError("factor return dates must be unique")
    history = history.sort_index().loc[lambda frame: frame.index < pd.Timestamp(rebalance_date), factors]
    if not history.empty and not np.isfinite(history.to_numpy(dtype=float)).all():
        raise ValueError("factor returns must be finite")

    last_signal_date = history.index[-1] if not history.empty else pd.NaT
    if len(history) < config.slow_lookback:
        diagnostics = _fallback_diagnostics(
            weights=weights,
            factor_directions=factor_directions,
            rebalance_date=pd.Timestamp(rebalance_date),
            last_signal_date=last_signal_date,
        )
        return FactorTimingDecision(weights=weights, diagnostics=diagnostics)

    multipliers: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for factor in factors:
        direction = factor_directions[factor]
        sign = -1.0 if direction is FactorDirection.LOW else 1.0
        directional = history[factor].astype(float).mul(sign)
        fast_return = _compounded_return(directional.tail(config.fast_lookback))
        slow_return = _compounded_return(directional.tail(config.slow_lookback))
        state, multiplier = _state_and_multiplier(fast_return, slow_return, config)
        multipliers[factor] = multiplier
        rows.append(
            {
                "rebalance_date": pd.Timestamp(rebalance_date),
                "factor": factor,
                "direction": direction.value,
                "base_weight": float(weights.loc[factor]),
                "fast_return": fast_return,
                "slow_return": slow_return,
                "state": state,
                "multiplier": multiplier,
                "timed_weight": float("nan"),
                "last_signal_date": last_signal_date,
            }
        )

    timed = weights.mul(pd.Series(multipliers, index=weights.index, dtype=float))
    timed = timed.div(float(timed.sum()))
    diagnostics = pd.DataFrame(rows, columns=TIMING_DIAGNOSTIC_COLUMNS)
    diagnostics["timed_weight"] = diagnostics["factor"].map(timed).astype(float)
    return FactorTimingDecision(weights=timed, diagnostics=diagnostics)


def _normalized_base_weights(base_weights: pd.Series) -> pd.Series:
    weights = base_weights.astype(float).copy()
    weights.index = weights.index.astype(str)
    if weights.empty:
        raise ValueError("base factor weights must not be empty")
    if weights.index.has_duplicates:
        raise ValueError("base factor weights must have unique factors")
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("base factor weights must be finite")
    if weights.lt(0.0).any():
        raise ValueError("base factor weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("at least one base factor weight must be positive")
    return weights.div(total)


def _compounded_return(returns: pd.Series) -> float:
    return float(returns.add(1.0).prod() - 1.0)


def _state_and_multiplier(
    fast_return: float,
    slow_return: float,
    config: FactorTimingConfig,
) -> tuple[str, float]:
    if fast_return > 0.0 and slow_return > 0.0:
        return "strong", config.strong_multiplier
    if fast_return < 0.0 and slow_return < 0.0:
        return "weak", config.weak_multiplier
    return "neutral", config.neutral_multiplier


def _fallback_diagnostics(
    *,
    weights: pd.Series,
    factor_directions: Mapping[str, FactorDirection],
    rebalance_date: pd.Timestamp,
    last_signal_date: object,
) -> pd.DataFrame:
    rows = [
        {
            "rebalance_date": rebalance_date,
            "factor": factor,
            "direction": factor_directions[factor].value,
            "base_weight": float(weight),
            "fast_return": float("nan"),
            "slow_return": float("nan"),
            "state": "insufficient_history",
            "multiplier": 1.0,
            "timed_weight": float(weight),
            "last_signal_date": last_signal_date,
        }
        for factor, weight in weights.items()
    ]
    return pd.DataFrame(rows, columns=TIMING_DIAGNOSTIC_COLUMNS)


__all__ = [
    "FactorTimingConfig",
    "FactorTimingDecision",
    "TIMING_DIAGNOSTIC_COLUMNS",
    "decide_factor_timing",
]
