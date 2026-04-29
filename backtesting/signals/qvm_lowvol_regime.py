from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class QvmLowvolRegimeSignalProducer:
    momentum_lookback: int = 120
    vol_lookback: int = 60
    value_weight: float = 0.35
    quality_weight: float = 0.35
    momentum_weight: float = 0.30
    market_filter_lookback: int = 120
    market_filter_threshold: float = -0.03
    min_breadth: float = 0.5

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_MKTCAP,
            DatasetId.QW_EQUITY_LFQ0,
            DatasetId.QW_OP_LFQ0,
            DatasetId.QW_OCF_LFQ0,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        market_cap = market.frames["market_cap"]
        equity = market.frames["equity"]
        op = market.frames["op"]
        ocf = market.frames["oper_cash_flow"]

        momentum = close.pct_change(self.momentum_lookback, fill_method=None)
        volatility = close.pct_change(fill_method=None).rolling(self.vol_lookback).std()
        book_to_market = equity.divide(market_cap).replace([float("inf"), float("-inf")], pd.NA)
        op_to_equity = op.divide(equity).replace([float("inf"), float("-inf")], pd.NA)
        ocf_to_equity = ocf.divide(equity).replace([float("inf"), float("-inf")], pd.NA)

        value_score = self._cross_sectional_zscore(book_to_market)
        quality_score = 0.5 * self._cross_sectional_zscore(op_to_equity) + 0.5 * self._cross_sectional_zscore(ocf_to_equity)
        momentum_score = self._cross_sectional_zscore(momentum)
        lowvol_score = -self._cross_sectional_zscore(volatility)

        alpha = (
            self.value_weight * value_score
            + self.quality_weight * quality_score
            + self.momentum_weight * momentum_score
            + lowvol_score
        )

        market_momentum = close.pct_change(self.market_filter_lookback, fill_method=None).mean(axis=1)
        above_ma = close.divide(close.rolling(self.market_filter_lookback).mean()) > 1.0
        breadth = above_ma.mean(axis=1)
        regime_pass = (market_momentum > self.market_filter_threshold) & (breadth >= self.min_breadth)
        regime_frame = pd.DataFrame({col: regime_pass for col in close.columns}, index=close.index, columns=close.columns)

        tradable = alpha.notna() & close.notna() & market_cap.notna() & equity.notna()
        return SignalBundle(
            alpha=alpha.where(tradable),
            context={
                "close": close,
                "momentum": momentum,
                "volatility": volatility,
                "book_to_market": book_to_market,
                "op_to_equity": op_to_equity,
                "ocf_to_equity": ocf_to_equity,
                "market_momentum": pd.DataFrame({col: market_momentum for col in close.columns}, index=close.index, columns=close.columns),
                "breadth": pd.DataFrame({col: breadth for col in close.columns}, index=close.index, columns=close.columns),
                "market_filter_pass": regime_frame,
                "tradable": tradable,
            },
            meta={
                "value_score": value_score,
                "quality_score": quality_score,
                "momentum_score": momentum_score,
                "lowvol_score": lowvol_score,
                "market_momentum": market_momentum,
                "breadth": breadth,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
