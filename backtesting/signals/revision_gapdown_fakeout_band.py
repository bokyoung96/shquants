from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class RevisionGapdownFakeoutBandSignalProducer:
    revision_lookback: int = 20
    high_lookback: int = 252
    pullback_lookback: int = 20
    boll_lookback: int = 20
    max_hold_days: int = 8
    revision_threshold: float = 0.02
    near_high_ratio: float = 0.88
    gap_threshold: float = -0.01

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

        prev_close = close.shift(1)
        sma20 = close.rolling(self.boll_lookback, min_periods=10).mean()
        std20 = close.rolling(self.boll_lookback, min_periods=10).std()
        lower_bb = sma20 - 2.0 * std20
        recent_high = high.shift(1).rolling(self.pullback_lookback, min_periods=10).max()
        trailing_high = high.shift(1).rolling(self.high_lookback, min_periods=126).max()
        trailing_low = low.shift(1).rolling(self.pullback_lookback, min_periods=10).min()
        volume_ratio = volume.divide(volume.rolling(20, min_periods=10).mean().replace(0.0, pd.NA))

        op_q1_rev = op_fwd_q1.pct_change(self.revision_lookback, fill_method=None)
        op_y1_rev = op_fwd_y1.pct_change(self.revision_lookback, fill_method=None)
        revision_raw = 0.7 * op_q1_rev + 0.3 * op_y1_rev

        gap_return = open_.divide(prev_close).sub(1.0)
        close_vs_open = close.divide(open_).sub(1.0)
        close_vs_prev = close.divide(prev_close).sub(1.0)
        near_high = prev_close.divide(trailing_high).ge(self.near_high_ratio)
        pullback_depth = prev_close.divide(recent_high).sub(1.0)
        fakeout_break = low.le(lower_bb.shift(1).fillna(lower_bb)) | low.le(trailing_low)
        reclaim = close.gt(open_) & close.divide(low.replace(0.0, pd.NA)).gt(1.015) & close.ge(prev_close * 0.97)
        trend_ok = prev_close.notna()

        event_now = (
            close.notna()
            & revision_raw.gt(self.revision_threshold)
            & op_q1_rev.gt(0.0)
            & op_y1_rev.gt(-0.02)
            & near_high
            & pullback_depth.le(-0.02)
            & pullback_depth.ge(-0.15)
            & gap_return.le(self.gap_threshold)
            & fakeout_break
            & reclaim
            & trend_ok
            & volume_ratio.gt(1.0)
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_exit = (
            revision_raw.le(0.0)
            | close.lt(sma20 * 0.97)
            | close.gt(recent_high * 1.01)
            | close.isna()
        )

        alpha = (
            0.55 * revision_raw
            + 0.20 * (-gap_return)
            + 0.15 * close_vs_open
            + 0.10 * close_vs_prev
            + 0.05 * volume_ratio
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
                "gap_return": gap_return,
                "close_vs_open": close_vs_open,
                "pullback_depth": pullback_depth,
                "volume_ratio": volume_ratio,
            },
        )
