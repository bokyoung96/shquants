from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class FlowFundamentalSignalProducer:
    flow_lookback: int = 20
    liquidity_lookback: int = 20
    value_weight: float = 0.5
    quality_weight: float = 0.5

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_V,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_RETAIL,
            DatasetId.QW_MKTCAP,
            DatasetId.QW_EQUITY_LFQ0,
            DatasetId.QW_OP_LFQ0,
            DatasetId.QW_OCF_LFQ0,
            DatasetId.QW_ASSET_LFQ0,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        volume = market.frames["volume"]
        foreign_flow = market.frames["foreign_flow"]
        retail_flow = market.frames["retail_flow"]
        market_cap = market.frames["market_cap"]
        equity = market.frames["equity"]
        op = market.frames["op"]
        ocf = market.frames["oper_cash_flow"]
        asset = market.frames["asset"]

        adv20 = (close * volume).rolling(self.liquidity_lookback).mean()
        foreign_intensity = foreign_flow.rolling(self.flow_lookback).sum().divide(adv20)
        retail_intensity = retail_flow.rolling(self.flow_lookback).sum().divide(adv20)

        book_to_market = equity.divide(market_cap).replace([float("inf"), float("-inf")], pd.NA)
        op_to_equity = op.divide(equity).replace([float("inf"), float("-inf")], pd.NA)
        ocf_to_asset = ocf.divide(asset).replace([float("inf"), float("-inf")], pd.NA)

        flow_score = self._cross_sectional_zscore(foreign_intensity) - self._cross_sectional_zscore(retail_intensity)
        value_score = self._cross_sectional_zscore(book_to_market)
        quality_score = 0.5 * self._cross_sectional_zscore(op_to_equity) + 0.5 * self._cross_sectional_zscore(ocf_to_asset)
        fundamental_score = self.value_weight * value_score + self.quality_weight * quality_score

        alpha = flow_score + fundamental_score

        tradable = (
            alpha.notna()
            & adv20.notna()
            & close.notna()
            & market_cap.notna()
            & equity.notna()
        )
        context = {
            "close": close,
            "volume": volume,
            "adv20": adv20,
            "foreign_flow": foreign_flow,
            "retail_flow": retail_flow,
            "foreign_intensity": foreign_intensity,
            "retail_intensity": retail_intensity,
            "book_to_market": book_to_market,
            "op_to_equity": op_to_equity,
            "ocf_to_asset": ocf_to_asset,
            "tradable": tradable,
        }
        meta = {
            "flow_score": flow_score,
            "value_score": value_score,
            "quality_score": quality_score,
            "fundamental_score": fundamental_score,
        }
        return SignalBundle(alpha=alpha.where(tradable), context=context, meta=meta)

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
