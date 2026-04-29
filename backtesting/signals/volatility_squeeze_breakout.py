from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class VolatilitySqueezeBreakoutSignalProducer:
    squeeze_lookback: int = 20
    trend_lookback: int = 60
    breakout_lookback: int = 20
    max_hold_days: int = 20

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_ADJ_H,
            DatasetId.QW_ADJ_L,
            DatasetId.QW_V,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        high = market.frames["high"]
        low = market.frames["low"]
        volume = market.frames["volume"]

        sma20 = close.rolling(self.squeeze_lookback).mean()
        std20 = close.rolling(self.squeeze_lookback).std()
        upper_bb = sma20 + 2.0 * std20
        lower_bb = sma20 - 2.0 * std20
        bb_width = upper_bb.sub(lower_bb).divide(sma20.replace(0.0, pd.NA))

        tr_components = pd.concat(
            [
                (high - low).stack().rename("hl"),
                (high - close.shift(1)).abs().stack().rename("hc"),
                (low - close.shift(1)).abs().stack().rename("lc"),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1).unstack()
        atr20 = true_range.rolling(self.squeeze_lookback).mean()
        ema20 = close.ewm(span=self.squeeze_lookback, adjust=False).mean()
        upper_kc = ema20 + 1.5 * atr20
        lower_kc = ema20 - 1.5 * atr20
        squeeze_on = upper_bb.lt(upper_kc) & lower_bb.gt(lower_kc)

        breakout_level = close.shift(1).rolling(self.breakout_lookback).max()
        trend_filter = close.gt(close.rolling(self.trend_lookback).mean())
        volume_ratio = volume.divide(volume.rolling(self.squeeze_lookback).mean().replace(0.0, pd.NA))
        momentum_5d = close.pct_change(5, fill_method=None)

        event_now = (
            squeeze_on.shift(1).fillna(False)
            & close.gt(breakout_level)
            & trend_filter
            & volume_ratio.gt(1.2)
            & momentum_5d.gt(0.0)
            & close.notna()
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_exit = (
            close.lt(sma20)
            | close.lt(breakout_level * 0.97)
            | bb_width.gt(bb_width.rolling(40).mean() * 1.8)
            | close.isna()
        )

        alpha_raw = (momentum_5d + volume_ratio.rank(axis=1, pct=True) * 0.1 - bb_width).replace(
            [float("inf"), float("-inf")], pd.NA
        )
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
                "bb_width": bb_width,
                "breakout_level": breakout_level,
                "volume_ratio": volume_ratio,
            },
        )
