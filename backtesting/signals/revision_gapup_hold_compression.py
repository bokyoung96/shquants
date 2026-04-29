from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class RevisionGapupHoldCompressionSignalProducer:
    revision_lookback: int = 20
    compression_lookback: int = 20
    trend_lookback: int = 60
    exit_lookback: int = 10
    max_hold_days: int = 7
    op_revision_threshold: float = 0.03
    eps_revision_threshold: float = 0.02
    gap_threshold: float = 0.02

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
            DatasetId.QW_EPS_NFQ1,
        )

    def build(self, market: MarketData) -> SignalBundle:
        open_ = market.frames['open']
        close = market.frames['close']
        high = market.frames['high']
        low = market.frames['low']
        volume = market.frames['volume']
        op_fwd_q1 = market.frames['op_fwd_q1']
        op_fwd_y1 = market.frames['op_fwd']
        eps_fwd_q1 = market.frames['eps_fwd_q1']

        prev_close = close.shift(1)
        sma20 = close.rolling(20, min_periods=10).mean()
        sma60 = close.rolling(self.trend_lookback, min_periods=30).mean()
        exit_floor = low.shift(1).rolling(self.exit_lookback, min_periods=5).min()

        tr_components = pd.concat(
            [
                (high - low).stack().rename('hl'),
                (high - prev_close).abs().stack().rename('hc'),
                (low - prev_close).abs().stack().rename('lc'),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1).unstack()
        atr10 = true_range.rolling(10, min_periods=5).mean()
        atr40 = true_range.rolling(40, min_periods=20).mean()
        atr_ratio = atr10.divide(atr40.replace(0.0, pd.NA))

        range_short = high.shift(1).rolling(self.compression_lookback, min_periods=10).max().sub(
            low.shift(1).rolling(self.compression_lookback, min_periods=10).min()
        )
        range_long = high.shift(1).rolling(60, min_periods=30).max().sub(
            low.shift(1).rolling(60, min_periods=30).min()
        )
        range_ratio = range_short.divide(range_long.replace(0.0, pd.NA))
        volume_ratio = volume.divide(volume.rolling(20, min_periods=10).mean().replace(0.0, pd.NA))

        op_q1_rev = op_fwd_q1.pct_change(self.revision_lookback, fill_method=None)
        op_y1_rev = op_fwd_y1.pct_change(self.revision_lookback, fill_method=None)
        eps_q1_rev = eps_fwd_q1.pct_change(self.revision_lookback, fill_method=None)
        revision_strength = 0.5 * op_q1_rev + 0.25 * op_y1_rev + 0.25 * eps_q1_rev

        gap_return = open_.divide(prev_close).sub(1.0)
        close_location = close.sub(low).divide(high.sub(low).replace(0.0, pd.NA))
        intraday_return = close.divide(open_).sub(1.0)
        breakout_level = high.shift(1).rolling(15, min_periods=8).max()
        breakout_ok = close.ge(breakout_level * 0.995)

        compression_ready = atr_ratio.lt(0.85) & range_ratio.lt(0.75)
        revision_gate = (
            op_q1_rev.gt(self.op_revision_threshold)
            & op_y1_rev.gt(0.0)
            & eps_q1_rev.gt(self.eps_revision_threshold)
        )
        event_now = (
            close.notna()
            & compression_ready.shift(1).fillna(False)
            & revision_gate
            & gap_return.ge(self.gap_threshold)
            & intraday_return.ge(-0.01)
            & close_location.ge(0.6)
            & breakout_ok
            & close.gt(sma20)
            & sma20.gt(sma60)
            & volume_ratio.gt(1.2)
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_exit = (
            revision_strength.le(0.0)
            | close.lt(sma20)
            | close.lt(exit_floor * 0.99)
            | close_location.lt(0.35)
            | close.isna()
        )

        alpha = (
            0.45 * revision_strength
            + 0.20 * gap_return
            + 0.15 * close_location
            + 0.10 * intraday_return
            + 0.10 * volume_ratio
            - 0.10 * atr_ratio
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
                'eps_q1_rev': eps_q1_rev,
                'revision_strength': revision_strength,
                'gap_return': gap_return,
                'close_location': close_location,
                'volume_ratio': volume_ratio,
                'atr_ratio': atr_ratio,
            },
        )
