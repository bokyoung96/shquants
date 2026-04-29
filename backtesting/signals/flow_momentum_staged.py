from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class FlowMomentumStagedSignalProducer:
    flow_lookback: int = 20
    momentum_lookback: int = 120

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_FOREIGN,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        foreign_flow = market.frames["foreign_flow"]

        traded_value = close.abs().rolling(self.flow_lookback).mean()
        foreign_intensity = foreign_flow.rolling(self.flow_lookback).sum().divide(traded_value)
        momentum = close.pct_change(self.momentum_lookback, fill_method=None)

        foreign_score = self._cross_sectional_zscore(foreign_intensity)
        momentum_score = self._cross_sectional_zscore(momentum)
        alpha = foreign_score + 0.5 * momentum_score
        tradable = alpha.notna() & close.notna()

        eligible_entry = alpha.notna() & alpha.shift(1).isna()
        eligible_add_1 = alpha.notna() & alpha.shift(1).notna()
        eligible_add_2 = eligible_add_1
        eligible_exit = alpha.isna()

        return SignalBundle(
            alpha=alpha.where(tradable),
            context={
                "close": close,
                "foreign_flow": foreign_flow,
                "foreign_intensity": foreign_intensity,
                "momentum": momentum,
                "tradable": tradable,
                "eligible_entry": eligible_entry.fillna(False),
                "eligible_add_1": eligible_add_1.fillna(False),
                "eligible_add_2": eligible_add_2.fillna(False),
                "eligible_exit": eligible_exit.fillna(False),
            },
            meta={
                "foreign_score": foreign_score,
                "momentum_score": momentum_score,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
