from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class ShareholderReturnEventCoreBandSignalProducer:
    buyback_threshold: float = -0.01
    flow_lookback: int = 20

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_V,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_SHA_OUT,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        volume = market.frames["volume"]
        foreign_flow = market.frames["foreign_flow"]
        shares_out = market.frames["shares_out"]

        adv = (close.abs() * volume).rolling(self.flow_lookback).mean()
        foreign_intensity = foreign_flow.divide(adv)
        share_change = shares_out.pct_change(fill_method=None)
        buyback_event = share_change.le(self.buyback_threshold)
        price_support = close.pct_change(20, fill_method=None).gt(-0.05)
        flow_support = foreign_intensity.rolling(5).sum().gt(0.0)

        event_now = buyback_event & price_support & flow_support & close.notna()
        alpha_raw = (-share_change.clip(upper=0.0)) + 0.5 * foreign_intensity.rolling(5).sum()
        alpha = self._cross_sectional_zscore(alpha_raw)
        tradable = close.notna() & alpha.notna()

        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_add_1 = event_now & event_prev
        eligible_add_2 = eligible_add_1 & close.pct_change(5, fill_method=None).gt(0.0)
        eligible_exit = (~event_now) | alpha.isna()

        return SignalBundle(
            alpha=alpha.where(event_now & tradable),
            context={
                "close": close,
                "tradable": tradable,
                "eligible_entry": eligible_entry.fillna(False),
                "eligible_add_1": eligible_add_1.fillna(False),
                "eligible_add_2": eligible_add_2.fillna(False),
                "eligible_exit": eligible_exit.fillna(False),
            },
            meta={
                "share_change": share_change,
                "foreign_intensity": foreign_intensity,
                "alpha_raw": alpha_raw,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
