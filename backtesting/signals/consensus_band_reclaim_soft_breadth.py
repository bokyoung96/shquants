from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class ConsensusBandReclaimSoftBreadthSignalProducer:
    revision_lookback: int = 20
    trend_lookback: int = 60
    pullback_lookback: int = 7
    max_hold_days: int = 10
    revision_threshold: float = 0.0115
    volume_threshold: float = 0.85
    reclaim_buffer: float = -0.0015
    breadth_threshold: float = 0.44

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
        sma10 = close.rolling(10, min_periods=5).mean()
        sma20 = close.rolling(20, min_periods=10).mean()
        sma60 = close.rolling(self.trend_lookback, min_periods=30).mean()
        sma120 = close.rolling(120, min_periods=60).mean()
        avg_vol20 = volume.rolling(20, min_periods=10).mean().replace(0.0, pd.NA)
        volume_ratio = volume.divide(avg_vol20)

        if market.universe is not None:
            universe = market.universe.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        else:
            universe = close.notna()

        op_q1_rev = op_fwd_q1.pct_change(self.revision_lookback, fill_method=None)
        op_y1_rev = op_fwd_y1.pct_change(self.revision_lookback, fill_method=None)
        revision_gate = op_q1_rev.gt(self.revision_threshold) & op_y1_rev.gt(-0.004)

        prior_20d = close.shift(1).divide(close.shift(21)).sub(1.0)
        trend_ok = close.gt(sma60 * 0.972) & sma20.gt(sma60 * 0.988) & prior_20d.ge(-0.08) & prior_20d.le(0.32)

        pullback_depth = close.shift(1).divide(high.shift(1).rolling(self.pullback_lookback, min_periods=4).max()).sub(1.0)
        pullback_ok = pullback_depth.le(-0.013) & pullback_depth.ge(-0.125)
        support_ok = low.shift(1).rolling(3, min_periods=2).min().gt(sma60.shift(1) * 0.942)

        range_10 = high.shift(1).rolling(10, min_periods=5).max().divide(low.shift(1).rolling(10, min_periods=5).min().replace(0.0, pd.NA)).sub(1.0)
        range_40 = high.shift(1).rolling(40, min_periods=20).max().divide(low.shift(1).rolling(40, min_periods=20).min().replace(0.0, pd.NA)).sub(1.0)
        contraction_ratio = range_10.divide(range_40.replace(0.0, pd.NA))
        contraction_ok = contraction_ratio.le(0.79)

        breadth_numer = (close.gt(sma60) & universe).sum(axis=1)
        breadth_denom = universe.sum(axis=1).replace(0, pd.NA)
        market_breadth = breadth_numer.divide(breadth_denom)
        breadth_ma5 = market_breadth.rolling(5, min_periods=3).mean()
        breadth_regime_ok = market_breadth.shift(1).ge(self.breadth_threshold) & breadth_ma5.shift(1).ge(0.445)
        breadth_regime_frame = pd.DataFrame({col: breadth_regime_ok for col in close.columns}, index=close.index).fillna(False)

        reclaim_ref = pd.concat([
            high.shift(1).rolling(3, min_periods=2).max(),
            sma5.shift(1),
            sma10.shift(1),
        ], axis=0).groupby(level=0).max()
        reclaim = close.gt(reclaim_ref * (1.0 + self.reclaim_buffer)) & open_.le(reclaim_ref * 1.02)
        close_strength = close.ge(open_ * 0.997) & close.ge((high + low) / 2.0)
        medium_trend_ok = close.ge(sma120 * 0.972)

        event_now = (
            close.notna()
            & revision_gate
            & trend_ok
            & pullback_ok
            & support_ok
            & contraction_ok
            & breadth_regime_frame
            & reclaim
            & close_strength
            & medium_trend_ok
            & volume_ratio.gt(self.volume_threshold)
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev

        trailing_support = pd.concat([
            sma10 * 0.986,
            sma20 * 0.981,
            low.shift(1).rolling(4, min_periods=2).min() * 0.986,
        ], axis=0).groupby(level=0).max()
        breadth_exit = market_breadth.lt(0.39) | market_breadth.diff(3).lt(-0.11)
        breadth_exit_frame = pd.DataFrame({col: breadth_exit for col in close.columns}, index=close.index).fillna(False)
        eligible_exit = (
            op_q1_rev.le(-0.013)
            | op_y1_rev.le(-0.038)
            | close.lt(trailing_support)
            | close.lt(sma120 * 0.962)
            | breadth_exit_frame
            | close.isna()
        )

        alpha = (
            0.40 * op_q1_rev
            + 0.16 * op_y1_rev
            + 0.12 * close.divide(reclaim_ref.replace(0.0, pd.NA)).sub(1.0)
            - 0.06 * pullback_depth.abs()
            + 0.08 * volume_ratio
            + 0.07 * close.divide(sma10.replace(0.0, pd.NA)).sub(1.0)
            + 0.07 * close.divide(sma20.replace(0.0, pd.NA)).sub(1.0)
            + 0.08 * contraction_ratio.rsub(1.0)
            + 0.08 * pd.DataFrame({col: market_breadth for col in close.columns}, index=close.index)
            + 0.08 * close.divide(sma120.replace(0.0, pd.NA)).sub(1.0)
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
                'volume_ratio': volume_ratio,
                'pullback_depth': pullback_depth,
                'prior_20d': prior_20d,
                'reclaim_ref': reclaim_ref,
                'contraction_ratio': contraction_ratio,
                'market_breadth': market_breadth,
                'breadth_ma5': breadth_ma5,
                'trailing_support': trailing_support,
            },
        )
