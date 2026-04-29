from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class ConsensusInsideBarBreakoutSignalProducer:
    revision_lookback: int = 20
    mother_lookback: int = 3
    trend_lookback: int = 60
    max_hold_days: int = 10
    revision_threshold: float = 0.03
    volume_threshold: float = 1.2
    compression_threshold: float = 0.06

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
        avg_vol20 = volume.rolling(20, min_periods=10).mean().replace(0.0, pd.NA)
        volume_ratio = volume.divide(avg_vol20)

        op_q1_rev = op_fwd_q1.pct_change(self.revision_lookback, fill_method=None)
        op_y1_rev = op_fwd_y1.pct_change(self.revision_lookback, fill_method=None)
        revision_gate = op_q1_rev.gt(self.revision_threshold) & op_y1_rev.gt(0.0)

        inside_day = high.lt(high.shift(1)) & low.gt(low.shift(1))
        inside_cluster = inside_day.shift(1).fillna(False).astype(bool) & inside_day.shift(2).fillna(False).astype(bool)
        mother_high = high.shift(1).rolling(self.mother_lookback, min_periods=self.mother_lookback).max()
        mother_low = low.shift(1).rolling(self.mother_lookback, min_periods=self.mother_lookback).min()
        compression = mother_high.sub(mother_low).divide(close.shift(1).replace(0.0, pd.NA))

        prior_advance = close.shift(1).divide(close.shift(6)).sub(1.0)
        breakout = close.gt(mother_high) & open_.le(mother_high * 1.01)
        close_strength = close.divide(low.replace(0.0, pd.NA)).gt(1.02) & close.gt(open_)
        trend_ok = close.gt(sma60) & sma20.gt(sma60) & prior_advance.ge(0.03) & prior_advance.le(0.25)

        event_now = (
            close.notna()
            & revision_gate
            & inside_cluster
            & compression.lt(self.compression_threshold)
            & breakout
            & close_strength
            & trend_ok
            & volume_ratio.gt(self.volume_threshold)
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        trailing_support = mother_low
        eligible_exit = (
            op_q1_rev.le(0.0)
            | close.lt(sma20 * 0.99)
            | close.lt(trailing_support * 0.99)
            | close.isna()
        )

        alpha = (
            0.45 * op_q1_rev
            + 0.20 * op_y1_rev
            + 0.15 * prior_advance
            + 0.10 * close.divide(mother_high.replace(0.0, pd.NA)).sub(1.0)
            + 0.10 * volume_ratio
            - 0.10 * compression
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
                "volume_ratio": volume_ratio,
                "compression": compression,
                "prior_advance": prior_advance,
            },
        )
