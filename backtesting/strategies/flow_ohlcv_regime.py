from __future__ import annotations

from dataclasses import dataclass

from backtesting.construction.long_only import LongOnlyTopN
from backtesting.policy.risk import RegimeFilterPolicy
from backtesting.signals.flow_ohlcv_regime import FlowOhlcvRegimeSignalProducer

from .composable import ComposableStrategy


@dataclass(slots=True)
class FlowOhlcvRegimeTopN(ComposableStrategy):
    top_n: int = 20
    flow_lookback: int = 20
    momentum_lookback: int = 60
    liquidity_lookback: int = 20
    momentum_weight: float = 0.3
    market_filter_lookback: int = 120
    market_filter_threshold: float = -0.03
    min_breadth: float = 0.5
    off_exposure_multiplier: float = 0.0

    def __post_init__(self) -> None:
        self.signal_producer = FlowOhlcvRegimeSignalProducer(
            flow_lookback=self.flow_lookback,
            momentum_lookback=self.momentum_lookback,
            liquidity_lookback=self.liquidity_lookback,
            momentum_weight=self.momentum_weight,
            market_filter_lookback=self.market_filter_lookback,
            market_filter_threshold=self.market_filter_threshold,
            min_breadth=self.min_breadth,
        )
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)
        self.position_policy = RegimeFilterPolicy(
            regime_key="market_filter_pass",
            off_exposure_multiplier=self.off_exposure_multiplier,
        )
