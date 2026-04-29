from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class RevisionSqueezeBreakoutSignalProducer:
    revision_lookback: int = 20
    squeeze_short: int = 10
    squeeze_long: int = 40
    breakout_lookback: int = 15
    trend_lookback: int = 60
    max_hold_days: int = 15
    revision_threshold: float = 0.02

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
        open_ = market.frames["open"]
        close = market.frames["close"]
        high = market.frames["high"]
        low = market.frames["low"]
        volume = market.frames["volume"]
        op_fwd_q1 = market.frames["op_fwd_q1"]
        op_fwd_y1 = market.frames["op_fwd"]

        sma20 = close.rolling(20, min_periods=10).mean()
        sma60 = close.rolling(self.trend_lookback, min_periods=30).mean()
        prior_breakout = high.shift(1).rolling(self.breakout_lookback, min_periods=8).max()
        support_low = low.shift(1).rolling(10, min_periods=5).min()

        tr_components = pd.concat(
            [
                (high - low).stack().rename("hl"),
                (high - close.shift(1)).abs().stack().rename("hc"),
                (low - close.shift(1)).abs().stack().rename("lc"),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1).unstack()
        atr10 = true_range.rolling(self.squeeze_short, min_periods=5).mean()
        atr40 = true_range.rolling(self.squeeze_long, min_periods=20).mean()
        atr_ratio = atr10.divide(atr40.replace(0.0, pd.NA))

        range10 = high.shift(1).rolling(10, min_periods=5).max().sub(low.shift(1).rolling(10, min_periods=5).min())
        range40 = high.shift(1).rolling(40, min_periods=20).max().sub(low.shift(1).rolling(40, min_periods=20).min())
        range_ratio = range10.divide(range40.replace(0.0, pd.NA))
        volume_ratio = volume.divide(volume.rolling(20, min_periods=10).mean().replace(0.0, pd.NA))

        op_q1_rev = op_fwd_q1.pct_change(self.revision_lookback, fill_method=None)
        op_y1_rev = op_fwd_y1.pct_change(self.revision_lookback, fill_method=None)
        revision_accel = op_q1_rev.sub(op_y1_rev)
        revision_gate = (
            op_q1_rev.gt(self.revision_threshold)
            & op_y1_rev.gt(0.0)
            & revision_accel.gt(-0.01)
        )

        squeeze_ready = atr_ratio.lt(0.8) & range_ratio.lt(0.65)
        breakout = close.gt(prior_breakout) & open_.le(prior_breakout * 1.01)
        trend_ok = close.gt(sma60) & sma20.gt(sma60)
        momentum_ok = close.pct_change(5, fill_method=None).gt(-0.02)

        event_now = (
            close.notna()
            & revision_gate
            & squeeze_ready.shift(1).fillna(False)
            & breakout
            & trend_ok
            & momentum_ok
            & volume_ratio.gt(1.0)
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_exit = (
            ~revision_gate
            | close.lt(sma20)
            | close.lt(support_low * 0.99)
            | close.isna()
        )

        alpha = (
            0.55 * op_q1_rev
            + 0.25 * op_y1_rev
            + 0.15 * revision_accel
            + 0.05 * volume_ratio
            - 0.10 * atr_ratio
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
                "op_q1_rev": op_q1_rev,
                "op_y1_rev": op_y1_rev,
                "revision_accel": revision_accel,
                "atr_ratio": atr_ratio,
                "range_ratio": range_ratio,
                "volume_ratio": volume_ratio,
            },
        )
