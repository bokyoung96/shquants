from __future__ import annotations

from dataclasses import dataclass

from backtesting.construction.long_only import LongOnlyTopN
from backtesting.signals.flow_change_accel import FlowChangeAccelSignalProducer

from .composable import ComposableStrategy


@dataclass(slots=True)
class FlowChangeAccelTopN(ComposableStrategy):
    top_n: int = 20
    short_lookback: int = 5
    long_lookback: int = 20
    liquidity_lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = FlowChangeAccelSignalProducer(
            short_lookback=self.short_lookback,
            long_lookback=self.long_lookback,
            liquidity_lookback=self.liquidity_lookback,
        )
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)
