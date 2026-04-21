from __future__ import annotations

from dataclasses import dataclass

from backtesting.construction.long_only import LongOnlyTopN
from backtesting.signals.momentum import MomentumSignalProducer

from .composable import ComposableStrategy


@dataclass(slots=True)
class MomentumTopN(ComposableStrategy):
    top_n: int = 20
    lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = MomentumSignalProducer(lookback=self.lookback)
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)
