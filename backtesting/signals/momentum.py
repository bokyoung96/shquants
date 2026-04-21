from __future__ import annotations

from dataclasses import dataclass

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class MomentumSignalProducer:
    lookback: int = 20

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_ADJ_C,)

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        alpha = close.pct_change(self.lookback, fill_method=None)
        return SignalBundle(
            alpha=alpha,
            context={"close": close, "tradable": alpha.notna()},
        )
