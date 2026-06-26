from __future__ import annotations

# Strat1 shooting-star sell filters:
# 1) (high - low) / prev_close > 15%  [intraday volatility]
# 2) upper_wick > body               [sell-off pressure]
# 3) high / closest-3M-close > 1.5x  [prior run-up]
# 4) last-day volume rank >= 90%     [volume confirmation]
# Selected names are held as an equal-weight short basket.

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from ..base import RegisteredStrategy


@dataclass(slots=True)
class Strat1(RegisteredStrategy):
    """Shooting-star sell filter adapted to the registered strategy contract."""

    rng_min: float = 0.15
    ref_min: float = 1.5
    vol_pct: float = 0.9
    vol_days: int = 252
    ref_m: int = 3
    min_hits: int = 4
    gross_short: float = 1.0

    def __post_init__(self) -> None:
        if self.rng_min < 0:
            raise ValueError("rng_min must be non-negative")
        if self.ref_min <= 0:
            raise ValueError("ref_min must be positive")
        if not 0.0 <= self.vol_pct <= 1.0:
            raise ValueError("vol_pct must be between 0 and 1")
        if self.vol_days <= 0:
            raise ValueError("vol_days must be positive")
        if self.ref_m <= 0:
            raise ValueError("ref_m must be positive")
        if not 1 <= self.min_hits <= 4:
            raise ValueError("min_hits must be between 1 and 4")
        if self.gross_short < 0:
            raise ValueError("gross_short must be non-negative")

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_O,
            DatasetId.QW_ADJ_H,
            DatasetId.QW_ADJ_L,
            DatasetId.QW_ADJ_C,
            DatasetId.QW_V,
        )

    def build_signal(self, market: MarketData) -> pd.DataFrame:
        close = _require_frame(market, "close").astype(float)
        open_ = _require_frame(market, "open").reindex_like(close).astype(float)
        high = _require_frame(market, "high").reindex_like(close).astype(float)
        low = _require_frame(market, "low").reindex_like(close).astype(float)
        volume = _require_frame(market, "volume").reindex_like(close).astype(float)

        index = pd.DatetimeIndex(close.index)
        signal = pd.DataFrame(float("nan"), index=close.index, columns=close.columns, dtype=float)
        if len(index) < 2:
            return signal

        for offset, ts in enumerate(index[1:], start=1):
            prev = index[offset - 1]
            ref = _nearest_reference_date(index[: offset + 1], ts=ts, months=self.ref_m)

            body = pd.concat([close.loc[ts], open_.loc[ts]], axis=1)
            body_top = body.max(axis=1)
            body_bottom = body.min(axis=1)
            vol_rank = volume.loc[:ts].tail(self.vol_days).rank(pct=True).iloc[-1]

            conditions = pd.concat(
                {
                    "range_ok": (high.loc[ts] - low.loc[ts]).divide(close.loc[prev]).gt(self.rng_min),
                    "wick_ok": (high.loc[ts] - body_top).gt(body_top - body_bottom),
                    "ref_ok": high.loc[ts].divide(close.loc[ref]).gt(self.ref_min),
                    "vol_ok": vol_rank.ge(self.vol_pct),
                },
                axis=1,
            ).fillna(False)
            hits = conditions.sum(axis=1).astype(float)
            selected = hits.ge(self.min_hits)
            signal.loc[ts, selected] = -hits.loc[selected]

        return signal

    def target_weights(self, signal: pd.Series) -> pd.Series:
        weights = pd.Series(0.0, index=signal.index, dtype=float)
        selected = signal.dropna()
        if selected.empty or self.gross_short == 0.0:
            return weights
        weights.loc[selected.index] = -float(self.gross_short) / len(selected)
        return weights


def _require_frame(market: MarketData, key: str) -> pd.DataFrame:
    try:
        return market.frames[key]
    except KeyError as exc:
        available = ", ".join(sorted(market.frames)) or "<none>"
        raise KeyError(f"team strat1 requires market frame '{key}'. Available: {available}") from exc


def _nearest_reference_date(index: pd.DatetimeIndex, *, ts: pd.Timestamp, months: int) -> pd.Timestamp:
    target = ts - pd.DateOffset(months=months)
    candidates = index[index <= ts]
    diffs = (pd.Series(candidates, index=candidates) - target).abs()
    return diffs.idxmin()


__all__ = ("Strat1",)
