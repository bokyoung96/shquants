from __future__ import annotations

from dataclasses import dataclass

from backtesting.construction.long_only import LongOnlyTopN
from backtesting.signals.flow_ohlcv import FlowOhlcvSignalProducer

from .composable import ComposableStrategy


@dataclass(slots=True)
class FlowOhlcvTopN(ComposableStrategy):
    top_n: int = 20
    flow_lookback: int = 20
    momentum_lookback: int = 60
    liquidity_lookback: int = 20
    momentum_weight: float = 0.5

    def __post_init__(self) -> None:
        self.signal_producer = FlowOhlcvSignalProducer(
            flow_lookback=self.flow_lookback,
            momentum_lookback=self.momentum_lookback,
            liquidity_lookback=self.liquidity_lookback,
            momentum_weight=self.momentum_weight,
        )
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)
