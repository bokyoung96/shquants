from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class RevisionPullbackReclaimSignalProducer:
    revision_lookback: int = 20
    pullback_lookback: int = 20
    trend_lookback: int = 60
    reclaim_lookback: int = 3
    max_hold_days: int = 12
    revision_threshold: float = 0.03

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_ADJ_H,
            DatasetId.QW_ADJ_L,
            DatasetId.QW_V,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFY1,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        high = market.frames["high"]
        low = market.frames["low"]
        volume = market.frames["volume"]
        op_fwd_q1 = market.frames["op_fwd_q1"]
        op_fwd_y1 = market.frames["op_fwd"]

        sma20 = close.rolling(20, min_periods=10).mean()
        sma60 = close.rolling(self.trend_lookback, min_periods=30).mean()
        recent_high = close.shift(1).rolling(self.pullback_lookback, min_periods=10).max()
        recent_low = low.shift(1).rolling(self.pullback_lookback, min_periods=10).min()
        volume_ratio = volume.divide(volume.rolling(20, min_periods=10).mean().replace(0.0, pd.NA))

        op_q1_rev = op_fwd_q1.pct_change(self.revision_lookback, fill_method=None)
        op_y1_rev = op_fwd_y1.pct_change(self.revision_lookback, fill_method=None)
        revision_raw = 0.65 * op_q1_rev + 0.35 * op_y1_rev

        pullback_depth = close.shift(1).divide(recent_high).sub(1.0)
        reclaim_trigger = close.gt(high.shift(1).rolling(self.reclaim_lookback, min_periods=2).max())
        sma_reclaim = close.gt(sma20) & close.shift(1).le(sma20.shift(1))
        trend_ok = close.gt(sma60)
        short_term_washout = close.pct_change(5, fill_method=None).lt(-0.02)

        price_range = recent_high.sub(recent_low).replace(0.0, pd.NA)
        range_recovery = close.shift(1).sub(recent_low).divide(price_range)

        event_now = (
            close.notna()
            & revision_raw.gt(self.revision_threshold)
            & op_q1_rev.gt(0.0)
            & op_y1_rev.gt(-0.02)
            & trend_ok
            & pullback_depth.le(-0.03)
            & pullback_depth.ge(-0.12)
            & range_recovery.gt(0.25)
            & short_term_washout
            & sma_reclaim
            & reclaim_trigger
            & volume_ratio.gt(0.8)
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_exit = (
            revision_raw.le(0.0)
            | close.lt(sma20 * 0.97)
            | close.lt(recent_low * 0.99)
            | close.isna()
        )

        alpha = (
            revision_raw
            + close.divide(sma20.replace(0.0, pd.NA)).sub(1.0) * 0.5
            - pullback_depth.abs() * 0.15
        ).replace([float("inf"), float("-inf")], pd.NA)
        tradable = close.notna() & alpha.notna()

        return SignalBundle(
            alpha=alpha.where(event_now & tradable),
            context={
                "tradable": tradable,
                "eligible_entry": eligible_entry.fillna(False),
                "eligible_exit": eligible_exit.fillna(False),
                "max_hold_days": self.max_hold_days,
            },
            meta={
                "revision_raw": revision_raw,
                "pullback_depth": pullback_depth,
                "range_recovery": range_recovery,
                "volume_ratio": volume_ratio,
            },
        )
