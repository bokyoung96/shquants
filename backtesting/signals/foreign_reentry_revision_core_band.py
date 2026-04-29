from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class ForeignReentryRevisionCoreBandSignalProducer:
    ratio_lookback: int = 20
    revision_threshold: float = 0.05
    support_momentum_lookback: int = 20

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_FOREIGN_RATIO,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFY1,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        foreign_flow = market.frames["foreign_flow"]
        foreign_ratio = market.frames["foreign_ratio"]
        op_fwd_q1 = market.frames["op_fwd_q1"]
        op_fwd = market.frames["op_fwd"]

        ratio_change = foreign_ratio.diff(self.ratio_lookback)
        foreign_support = foreign_flow.rolling(20).sum()
        q1_revision = op_fwd_q1.pct_change(fill_method=None)
        fy1_revision = op_fwd.pct_change(fill_method=None)
        revision_raw = 0.6 * q1_revision + 0.4 * fy1_revision
        price_support = close.pct_change(self.support_momentum_lookback, fill_method=None)

        alpha_raw = (ratio_change + 0.5 * revision_raw + 0.15 * price_support).replace(
            [float("inf"), float("-inf")], pd.NA
        )
        alpha = self._cross_sectional_zscore(alpha_raw)

        event_now = (
            ratio_change.gt(0.003)
            & foreign_support.gt(0.0)
            & revision_raw.gt(self.revision_threshold)
            & price_support.gt(-0.04)
            & close.notna()
        )
        event_prev = event_now.shift(1, fill_value=False)
        eligible_entry = event_now & ~event_prev
        eligible_add_1 = event_now & event_prev
        eligible_add_2 = eligible_add_1 & revision_raw.gt(revision_raw.rolling(3).mean())
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
                "revision_raw": revision_raw,
                "foreign_support": foreign_support,
                "price_support": price_support,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
