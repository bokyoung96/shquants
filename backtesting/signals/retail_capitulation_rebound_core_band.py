from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class RetailCapitulationReboundCoreBandSignalProducer:
    flow_lookback: int = 10
    rebound_lookback: int = 5
    retail_threshold: float = -0.02

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_V,
            DatasetId.QW_RETAIL,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_INSTITUTION,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames['close']
        volume = market.frames['volume']
        retail_flow = market.frames['retail_flow']
        foreign_flow = market.frames['foreign_flow']
        inst_flow = market.frames['inst_flow']

        adv = (close.abs() * volume).rolling(20).mean()
        retail_intensity = retail_flow.rolling(self.flow_lookback).sum().divide(adv)
        foreign_intensity = foreign_flow.rolling(self.flow_lookback).sum().divide(adv)
        inst_intensity = inst_flow.rolling(self.flow_lookback).sum().divide(adv)
        rebound = close.pct_change(self.rebound_lookback, fill_method=None)

        alpha_raw = (-retail_intensity) + 0.5 * foreign_intensity + 0.5 * inst_intensity + 0.2 * rebound
        alpha = self._cross_sectional_zscore(alpha_raw)

        event_now = (
            retail_intensity.lt(self.retail_threshold)
            & foreign_intensity.gt(0.0)
            & inst_intensity.gt(0.0)
            & rebound.gt(-0.02)
            & close.notna()
        )
        event_prev = event_now.shift(1, fill_value=False)
        eligible_entry = event_now & ~event_prev
        eligible_add_1 = event_now & event_prev
        eligible_add_2 = eligible_add_1 & rebound.gt(0.0)
        eligible_exit = (~event_now) | alpha.isna()
        tradable = close.notna() & alpha.notna()

        return SignalBundle(
            alpha=alpha.where(event_now & tradable),
            context={
                'close': close,
                'tradable': tradable,
                'eligible_entry': eligible_entry.fillna(False),
                'eligible_add_1': eligible_add_1.fillna(False),
                'eligible_add_2': eligible_add_2.fillna(False),
                'eligible_exit': eligible_exit.fillna(False),
            },
            meta={
                'retail_intensity': retail_intensity,
                'foreign_intensity': foreign_intensity,
                'inst_intensity': inst_intensity,
                'rebound': rebound,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
