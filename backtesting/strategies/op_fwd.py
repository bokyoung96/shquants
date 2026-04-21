from __future__ import annotations

from dataclasses import dataclass

from backtesting.construction.long_only import LongOnlyTopN
from backtesting.signals.op_fwd import OpFwdYieldSignalProducer

from .composable import ComposableStrategy


@dataclass(slots=True)
class OpFwdYieldTopN(ComposableStrategy):
    top_n: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = OpFwdYieldSignalProducer()
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)
