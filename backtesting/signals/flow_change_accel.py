from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class FlowChangeAccelSignalProducer:
    short_lookback: int = 5
    long_lookback: int = 20
    liquidity_lookback: int = 20

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_V,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_INSTITUTION,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        volume = market.frames["volume"]
        foreign_flow = market.frames["foreign_flow"]
        inst_flow = market.frames["inst_flow"]

        adv = (close.abs() * volume).rolling(self.liquidity_lookback).mean()
        foreign_intensity = foreign_flow.divide(adv)
        inst_intensity = inst_flow.divide(adv)

        foreign_short = foreign_intensity.rolling(self.short_lookback).sum()
        foreign_long = foreign_intensity.rolling(self.long_lookback).sum()
        inst_short = inst_intensity.rolling(self.short_lookback).sum()
        inst_long = inst_intensity.rolling(self.long_lookback).sum()

        foreign_accel = foreign_short - foreign_long.divide(max(self.long_lookback / self.short_lookback, 1.0))
        inst_accel = inst_short - inst_long.divide(max(self.long_lookback / self.short_lookback, 1.0))

        alpha = self._cross_sectional_zscore(foreign_accel) + self._cross_sectional_zscore(inst_accel)
        tradable = alpha.notna() & adv.notna() & close.notna()

        return SignalBundle(
            alpha=alpha.where(tradable),
            context={
                "close": close,
                "volume": volume,
                "adv": adv,
                "foreign_flow": foreign_flow,
                "inst_flow": inst_flow,
                "foreign_intensity": foreign_intensity,
                "inst_intensity": inst_intensity,
                "foreign_accel": foreign_accel,
                "inst_accel": inst_accel,
                "tradable": tradable,
            },
            meta={
                "foreign_score": self._cross_sectional_zscore(foreign_accel),
                "inst_score": self._cross_sectional_zscore(inst_accel),
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
