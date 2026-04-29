from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class RevisionEventCoreBandSignalProducer:
    revision_threshold: float = 0.08
    momentum_lookback: int = 20

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFY1,
            DatasetId.QW_EPS_NFY1,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        op_fwd_q1 = market.frames["op_fwd_q1"]
        op_fwd = market.frames["op_fwd"]
        eps_fwd = market.frames["eps_fwd"]

        op_q1_rev = op_fwd_q1.pct_change(fill_method=None)
        op_fy1_rev = op_fwd.pct_change(fill_method=None)
        eps_fy1_rev = eps_fwd.pct_change(fill_method=None)
        revision_raw = 0.5 * op_q1_rev + 0.3 * op_fy1_rev + 0.2 * eps_fy1_rev
        momentum = close.pct_change(self.momentum_lookback, fill_method=None)

        revision_score = self._cross_sectional_zscore(revision_raw)
        momentum_score = self._cross_sectional_zscore(momentum)
        alpha = revision_score + 0.25 * momentum_score

        event_now = (
            revision_raw.gt(self.revision_threshold)
            & op_q1_rev.gt(0.0)
            & op_fy1_rev.gt(-0.02)
            & close.notna()
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_add_1 = event_now & event_prev
        eligible_add_2 = eligible_add_1 & momentum.gt(0.0)
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
                "revision_raw": revision_raw,
                "revision_score": revision_score,
                "momentum_score": momentum_score,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
