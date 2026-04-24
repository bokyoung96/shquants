from __future__ import annotations

from dataclasses import dataclass

from backtesting.construction.long_only import LongOnlyTopN
from backtesting.signals.flow_fundamental import FlowFundamentalSignalProducer

from .composable import ComposableStrategy


@dataclass(slots=True)
class FlowFundamentalTopN(ComposableStrategy):
    top_n: int = 20
    flow_lookback: int = 20
    liquidity_lookback: int = 20
    value_weight: float = 0.5
    quality_weight: float = 0.5

    def __post_init__(self) -> None:
        self.signal_producer = FlowFundamentalSignalProducer(
            flow_lookback=self.flow_lookback,
            liquidity_lookback=self.liquidity_lookback,
            value_weight=self.value_weight,
            quality_weight=self.quality_weight,
        )
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)
