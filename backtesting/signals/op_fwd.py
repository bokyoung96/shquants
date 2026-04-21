from __future__ import annotations

from dataclasses import dataclass

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class OpFwdYieldSignalProducer:
    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_OP_NFY1, DatasetId.QW_MKTCAP)

    def build(self, market: MarketData) -> SignalBundle:
        op_fwd = market.frames["op_fwd"]
        market_cap = market.frames["market_cap"].where(market.frames["market_cap"].ne(0.0))
        alpha = op_fwd.div(market_cap)
        return SignalBundle(
            alpha=alpha,
            context={"op_fwd": op_fwd, "market_cap": market_cap, "tradable": alpha.notna()},
        )
