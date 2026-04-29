from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class InstitutionAccumulationCoreBandSignalProducer:
    flow_lookback: int = 15
    support_momentum_lookback: int = 20
    min_inst_intensity: float = 0.015

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_V,
            DatasetId.QW_INSTITUTION,
            DatasetId.QW_FOREIGN,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames['close']
        volume = market.frames['volume']
        inst_flow = market.frames['inst_flow']
        foreign_flow = market.frames['foreign_flow']

        adv = (close.abs() * volume).rolling(self.flow_lookback).mean()
        inst_intensity = inst_flow.rolling(self.flow_lookback).sum().divide(adv)
        foreign_intensity = foreign_flow.rolling(self.flow_lookback).sum().divide(adv)
        price_support = close.pct_change(self.support_momentum_lookback, fill_method=None)

        alpha_raw = inst_intensity + 0.35 * foreign_intensity + 0.15 * price_support
        alpha = self._cross_sectional_zscore(alpha_raw)

        event_now = (
            inst_intensity.gt(self.min_inst_intensity)
            & foreign_intensity.gt(-0.01)
            & price_support.gt(-0.04)
            & close.notna()
        )
        event_prev = event_now.shift(1, fill_value=False)
        eligible_entry = event_now & ~event_prev
        eligible_add_1 = event_now & event_prev
        eligible_add_2 = eligible_add_1 & inst_intensity.gt(inst_intensity.rolling(5).mean())
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
                'inst_intensity': inst_intensity,
                'foreign_intensity': foreign_intensity,
                'price_support': price_support,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
