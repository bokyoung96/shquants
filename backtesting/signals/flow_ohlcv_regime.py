from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class FlowOhlcvRegimeSignalProducer:
    flow_lookback: int = 20
    momentum_lookback: int = 60
    liquidity_lookback: int = 20
    momentum_weight: float = 0.3
    market_filter_lookback: int = 120
    market_filter_threshold: float = -0.03
    min_breadth: float = 0.5

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_V,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_RETAIL,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        volume = market.frames["volume"]
        foreign_flow = market.frames["foreign_flow"]
        retail_flow = market.frames["retail_flow"]

        adv20 = (close * volume).rolling(self.liquidity_lookback).mean()
        foreign_intensity = foreign_flow.rolling(self.flow_lookback).sum().divide(adv20)
        retail_intensity = retail_flow.rolling(self.flow_lookback).sum().divide(adv20)
        momentum = close.pct_change(self.momentum_lookback, fill_method=None)

        alpha = (
            self._cross_sectional_zscore(foreign_intensity)
            - self._cross_sectional_zscore(retail_intensity)
            + self.momentum_weight * self._cross_sectional_zscore(momentum)
        )

        market_momentum = close.pct_change(self.market_filter_lookback, fill_method=None).mean(axis=1)
        above_ma = close.divide(close.rolling(self.market_filter_lookback).mean()) > 1.0
        breadth = above_ma.mean(axis=1)
        regime_pass = (market_momentum > self.market_filter_threshold) & (breadth >= self.min_breadth)
        regime_frame = pd.DataFrame(
            {col: regime_pass for col in close.columns},
            index=close.index,
            columns=close.columns,
        )

        tradable = alpha.notna() & adv20.notna() & close.notna()
        return SignalBundle(
            alpha=alpha.where(tradable),
            context={
                "close": close,
                "volume": volume,
                "adv20": adv20,
                "foreign_flow": foreign_flow,
                "retail_flow": retail_flow,
                "foreign_intensity": foreign_intensity,
                "retail_intensity": retail_intensity,
                "momentum": momentum,
                "market_momentum": pd.DataFrame({col: market_momentum for col in close.columns}, index=close.index, columns=close.columns),
                "breadth": pd.DataFrame({col: breadth for col in close.columns}, index=close.index, columns=close.columns),
                "market_filter_pass": regime_frame,
                "tradable": tradable,
            },
            meta={
                "foreign_score": self._cross_sectional_zscore(foreign_intensity),
                "retail_score": self._cross_sectional_zscore(retail_intensity),
                "momentum_score": self._cross_sectional_zscore(momentum),
                "market_momentum": market_momentum,
                "breadth": breadth,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
