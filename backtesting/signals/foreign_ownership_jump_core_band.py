from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class ForeignOwnershipJumpCoreBandSignalProducer:
    change_lookback: int = 5
    flow_lookback: int = 20
    min_ratio_change: float = 0.002
    support_momentum_lookback: int = 20

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_FOREIGN_RATIO,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        foreign_flow = market.frames["foreign_flow"]
        foreign_ratio = market.frames["foreign_ratio"]

        ratio_change = foreign_ratio.diff(self.change_lookback)
        price_support = close.pct_change(self.support_momentum_lookback, fill_method=None)
        flow_support = foreign_flow.rolling(self.flow_lookback).sum()

        ratio_score = self._cross_sectional_zscore(ratio_change)
        flow_score = self._cross_sectional_zscore(flow_support)
        alpha = ratio_score + 0.35 * flow_score

        event_now = (
            ratio_change.gt(self.min_ratio_change)
            & flow_support.gt(0.0)
            & price_support.gt(-0.03)
            & close.notna()
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_add_1 = event_now & event_prev
        eligible_add_2 = eligible_add_1 & price_support.gt(0.0)
        eligible_exit = (~event_now) | alpha.isna()
        tradable = close.notna() & alpha.notna()

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
                "ratio_change": ratio_change,
                "ratio_score": ratio_score,
                "flow_score": flow_score,
                "price_support": price_support,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
