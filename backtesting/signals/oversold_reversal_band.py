from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class OversoldReversalBandSignalProducer:
    boll_lookback: int = 20
    rebound_lookback: int = 3
    max_hold_days: int = 10

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_ADJ_H,
            DatasetId.QW_ADJ_L,
            DatasetId.QW_ADJ_O,
            DatasetId.QW_V,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        high = market.frames["high"]
        low = market.frames["low"]
        open_ = market.frames["open"]
        volume = market.frames["volume"]

        sma20 = close.rolling(self.boll_lookback).mean()
        std20 = close.rolling(self.boll_lookback).std()
        lower_bb = sma20 - 2.0 * std20
        zscore = close.sub(sma20).divide(std20.replace(0.0, pd.NA))
        gap_return = open_.divide(close.shift(1)).sub(1.0)
        intraday_reversal = close.divide(low.replace(0.0, pd.NA)).sub(1.0)
        volume_ratio = volume.divide(volume.rolling(self.boll_lookback).mean().replace(0.0, pd.NA))
        short_rebound = close.pct_change(self.rebound_lookback, fill_method=None)
        upper_exit = close.gt(sma20) | close.gt(close.rolling(5).max().shift(1))

        event_now = (
            close.lt(lower_bb)
            & zscore.lt(-2.0)
            & gap_return.lt(-0.015)
            & intraday_reversal.gt(0.01)
            & volume_ratio.gt(1.1)
            & close.notna()
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_exit = upper_exit | short_rebound.gt(0.08) | close.lt(low.rolling(3).min().shift(1)) | close.isna()

        alpha_raw = (-zscore + intraday_reversal + (-gap_return)).replace([float("inf"), float("-inf")], pd.NA)
        tradable = close.notna() & alpha_raw.notna()
        return SignalBundle(
            alpha=alpha_raw.where(event_now & tradable),
            context={
                "tradable": tradable,
                "eligible_entry": eligible_entry.fillna(False),
                "eligible_exit": eligible_exit.fillna(False),
                "max_hold_days": self.max_hold_days,
            },
            meta={
                "zscore": zscore,
                "gap_return": gap_return,
                "intraday_reversal": intraday_reversal,
            },
        )
