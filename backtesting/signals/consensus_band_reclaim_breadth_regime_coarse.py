from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class ConsensusBandReclaimBreadthRegimeCoarseSignalProducer:
    revision_lookback: int = 20
    trend_lookback: int = 60
    pullback_lookback: int = 7
    max_hold_days: int = 10
    revision_threshold: float = 0.005
    breadth_threshold: float = 0.45

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_O,
            DatasetId.QW_ADJ_C,
            DatasetId.QW_ADJ_H,
            DatasetId.QW_ADJ_L,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFY1,
        )

    def build(self, market: MarketData) -> SignalBundle:
        open_ = market.frames['open']
        close = market.frames['close']
        high = market.frames['high']
        low = market.frames['low']
        op_fwd_q1 = market.frames['op_fwd_q1']
        op_fwd_y1 = market.frames['op_fwd']

        sma10 = close.rolling(10, min_periods=5).mean()
        sma60 = close.rolling(self.trend_lookback, min_periods=30).mean()

        if market.universe is not None:
            universe = market.universe.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        else:
            universe = close.notna()

        op_q1_rev = op_fwd_q1.pct_change(self.revision_lookback, fill_method=None)
        op_y1_rev = op_fwd_y1.pct_change(self.revision_lookback, fill_method=None)
        revision_ok = op_q1_rev.gt(self.revision_threshold) & op_y1_rev.gt(-0.01)

        trend_ok = close.gt(sma60 * 0.98) & sma10.gt(sma60 * 0.99)

        recent_high = high.shift(1).rolling(self.pullback_lookback, min_periods=3).max()
        recent_low = low.shift(1).rolling(self.pullback_lookback, min_periods=3).min()
        pullback_depth = close.shift(1).divide(recent_high.replace(0.0, pd.NA)).sub(1.0)
        pullback_ok = pullback_depth.le(-0.01) & pullback_depth.ge(-0.12) & close.shift(1).gt(recent_low)

        reclaim_ref = recent_high
        reclaim_ok = close.gt(reclaim_ref * 0.995) & open_.le(reclaim_ref * 1.02)

        breadth_numer = (close.gt(sma60) & universe).sum(axis=1)
        breadth_denom = universe.sum(axis=1).replace(0, pd.NA)
        market_breadth = breadth_numer.divide(breadth_denom)
        breadth_ok = market_breadth.shift(1).ge(self.breadth_threshold)
        breadth_frame = pd.DataFrame({col: breadth_ok for col in close.columns}, index=close.index).fillna(False)

        event_now = close.notna() & revision_ok & trend_ok & pullback_ok & reclaim_ok & breadth_frame
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev

        breadth_exit = market_breadth.lt(0.38)
        breadth_exit_frame = pd.DataFrame({col: breadth_exit for col in close.columns}, index=close.index).fillna(False)
        trailing_support = sma10
        eligible_exit = (
            op_q1_rev.le(0.0)
            | op_y1_rev.le(-0.02)
            | close.lt(trailing_support)
            | breadth_exit_frame
            | close.isna()
        )

        alpha = (
            0.5 * op_q1_rev
            + 0.2 * op_y1_rev
            + 0.15 * close.divide(reclaim_ref.replace(0.0, pd.NA)).sub(1.0)
            + 0.15 * pd.DataFrame({col: market_breadth for col in close.columns}, index=close.index)
        ).replace([float('inf'), float('-inf')], pd.NA)
        tradable = close.notna() & alpha.notna() & universe

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
                'pullback_depth': pullback_depth,
                'reclaim_ref': reclaim_ref,
                'market_breadth': market_breadth,
                'trailing_support': trailing_support,
            },
        )
