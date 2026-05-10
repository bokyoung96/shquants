from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.long_only import LongOnlyTopN
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(slots=True)
class RevisionMinParamV02(ComposableStrategy):
    top_n: int = 30
    lookback: int = 20
    pth: float = 0.6

    def __post_init__(self) -> None:
        self.signal_producer = _RevisionMinParamV02SignalProducer(
            lookback=self.lookback,
            pth=self.pth,
        )
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)


@dataclass(slots=True)
class _RevisionMinParamV02SignalProducer:
    lookback: int = 20
    pth: float = 0.6

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_OP_NFQ1,
        )

    def build(self, market: MarketData) -> SignalBundle:
        eps = market.frames['eps_fwd_q1']
        op = market.frames['op_fwd_q1']

        eps_rev = eps.pct_change(self.lookback, fill_method=None)
        op_rev = op.pct_change(self.lookback, fill_method=None)
        eps_pos = eps_rev.gt(0)
        op_pos = op_rev.gt(0)
        pth = 0.5 * eps_pos.astype(float) + 0.5 * op_pos.astype(float)

        eps_rank = eps_rev.rank(axis=1, pct=True)
        op_rank = op_rev.rank(axis=1, pct=True)
        alpha = 0.5 * eps_rank + 0.5 * op_rank
        alpha = alpha.where(eps_pos & op_pos & (pth >= self.pth))

        tradable = alpha.notna()
        return SignalBundle(alpha=alpha, context={'tradable': tradable}, meta={})
