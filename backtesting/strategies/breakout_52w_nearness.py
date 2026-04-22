from __future__ import annotations

from dataclasses import dataclass

from backtesting.construction.long_only import LongOnlyTopN
from backtesting.signals.base import SignalBundle
from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .composable import ComposableStrategy


@dataclass(frozen=True, slots=True)
class Breakout52WeekNearnessSignalProducer:
    breakout_window: int = 252

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_ADJ_C,)

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"].astype(float)
        prior_high = close.rolling(self.breakout_window, min_periods=self.breakout_window).max().shift(1)
        alpha = close.div(prior_high)
        return SignalBundle(
            alpha=alpha,
            context={
                "close": close,
                "prior_high": prior_high,
                "tradable": alpha.notna(),
            },
        )


@dataclass(slots=True)
class Breakout52WeekNearnessTopN(ComposableStrategy):
    top_n: int = 20
    breakout_window: int = 252

    def __post_init__(self) -> None:
        self.signal_producer = Breakout52WeekNearnessSignalProducer(breakout_window=self.breakout_window)
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)
