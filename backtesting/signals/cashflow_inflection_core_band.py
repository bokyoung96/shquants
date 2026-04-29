from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class CashflowInflectionCoreBandSignalProducer:
    fundamental_lookback: int = 63
    support_momentum_lookback: int = 20

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_OCF_LFQ0,
            DatasetId.QW_OP_LFQ0,
            DatasetId.QW_ASSET_LFQ0,
            DatasetId.QW_EQUITY_LFQ0,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        ocf = market.frames["oper_cash_flow"]
        op = market.frames["op"]
        asset = market.frames["asset"]
        equity = market.frames["equity"]

        ocf_to_asset = ocf.divide(asset).replace([float("inf"), float("-inf")], pd.NA)
        op_to_equity = op.divide(equity).replace([float("inf"), float("-inf")], pd.NA)
        ocf_delta = ocf_to_asset.diff(self.fundamental_lookback)
        margin_delta = op_to_equity.diff(self.fundamental_lookback)
        price_support = close.pct_change(self.support_momentum_lookback, fill_method=None)

        alpha_raw = (ocf_delta + 0.7 * margin_delta + 0.15 * price_support).replace(
            [float("inf"), float("-inf")], pd.NA
        )
        alpha = self._cross_sectional_zscore(alpha_raw)

        event_now = (
            ocf_delta.gt(0.003)
            & margin_delta.gt(0.001)
            & price_support.gt(-0.05)
            & close.notna()
        )
        event_prev = event_now.shift(1, fill_value=False)
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
                "ocf_delta": ocf_delta,
                "margin_delta": margin_delta,
                "price_support": price_support,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
