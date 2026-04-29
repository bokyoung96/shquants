from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class ConsensusBandReclaimSignalProducer:
    revision_lookback: int = 20
    trend_lookback: int = 60
    pullback_lookback: int = 5
    max_hold_days: int = 12
    revision_threshold: float = 0.015
    volume_threshold: float = 1.0
    reclaim_buffer: float = 0.0

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_O,
            DatasetId.QW_ADJ_C,
            DatasetId.QW_ADJ_H,
            DatasetId.QW_ADJ_L,
            DatasetId.QW_V,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFY1,
        )

    def build(self, market: MarketData) -> SignalBundle:
        open_ = market.frames['open']
        close = market.frames['close']
        high = market.frames['high']
        low = market.frames['low']
        volume = market.frames['volume']
        op_fwd_q1 = market.frames['op_fwd_q1']
        op_fwd_y1 = market.frames['op_fwd']

        sma5 = close.rolling(5, min_periods=3).mean()
        sma20 = close.rolling(20, min_periods=10).mean()
        sma60 = close.rolling(self.trend_lookback, min_periods=30).mean()
        avg_vol20 = volume.rolling(20, min_periods=10).mean().replace(0.0, pd.NA)
        volume_ratio = volume.divide(avg_vol20)

        op_q1_rev = op_fwd_q1.pct_change(self.revision_lookback, fill_method=None)
        op_y1_rev = op_fwd_y1.pct_change(self.revision_lookback, fill_method=None)
        revision_gate = op_q1_rev.gt(self.revision_threshold) & op_y1_rev.gt(0.0)

        prior_20d = close.shift(1).divide(close.shift(21)).sub(1.0)
        trend_ok = close.gt(sma60 * 0.98) & sma20.gt(sma60 * 0.99) & prior_20d.ge(-0.05) & prior_20d.le(0.30)

        pullback_depth = close.shift(1).divide(high.shift(1).rolling(self.pullback_lookback, min_periods=3).max()).sub(1.0)
        pullback_ok = pullback_depth.le(-0.02) & pullback_depth.ge(-0.12)
        support_ok = low.shift(1).rolling(3, min_periods=2).min().gt(sma60.shift(1) * 0.95)

        reclaim_ref = pd.concat([
            high.shift(1).rolling(3, min_periods=2).max(),
            sma5.shift(1),
        ], axis=0).groupby(level=0).max()
        reclaim = close.gt(reclaim_ref * (1.0 + self.reclaim_buffer)) & open_.le(reclaim_ref * 1.015)
        close_strength = close.gt(open_) & close.ge((high + low) / 2.0)

        event_now = (
            close.notna()
            & revision_gate
            & trend_ok
            & pullback_ok
            & support_ok
            & reclaim
            & close_strength
            & volume_ratio.gt(self.volume_threshold)
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        trailing_support = pd.concat([
            sma20 * 0.985,
            low.shift(1).rolling(3, min_periods=2).min() * 0.985,
        ], axis=0).groupby(level=0).max()
        eligible_exit = (
            op_q1_rev.le(-0.02)
            | op_y1_rev.le(-0.05)
            | close.lt(trailing_support)
            | close.isna()
        )

        alpha = (
            0.45 * op_q1_rev
            + 0.20 * op_y1_rev
            + 0.15 * close.divide(reclaim_ref.replace(0.0, pd.NA)).sub(1.0)
            - 0.10 * pullback_depth.abs()
            + 0.10 * volume_ratio
            + 0.10 * close.divide(sma20.replace(0.0, pd.NA)).sub(1.0)
        ).replace([float('inf'), float('-inf')], pd.NA)
        tradable = close.notna() & alpha.notna()

        return SignalBundle(
            alpha=alpha.where(event_now & tradable),
            context={
                'tradable': tradable,
                'eligible_entry': eligible_entry.fillna(False),
                'eligible_exit': eligible_exit.fillna(False),
                'max_hold_days': self.max_hold_days,
            },
            meta={
                'op_q1_rev': op_q1_rev,
                'op_y1_rev': op_y1_rev,
                'volume_ratio': volume_ratio,
                'pullback_depth': pullback_depth,
                'prior_20d': prior_20d,
                'reclaim_ref': reclaim_ref,
            },
        )
