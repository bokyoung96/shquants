from __future__ import annotations

from dataclasses import dataclass

from backtesting.construction.long_only import LongOnlyTopN
from backtesting.policy.risk import RegimeFilterPolicy
from backtesting.signals.flow_fundamental_regime import FlowFundamentalRegimeSignalProducer

from .composable import ComposableStrategy


@dataclass(slots=True)
class FlowFundamentalRegimeTopN(ComposableStrategy):
    top_n: int = 20
    flow_lookback: int = 20
    liquidity_lookback: int = 20
    value_weight: float = 0.5
    quality_weight: float = 0.5
    market_filter_lookback: int = 120
    market_filter_threshold: float = -0.03
    min_breadth: float = 0.5
    off_exposure_multiplier: float = 0.35

    def __post_init__(self) -> None:
        self.signal_producer = FlowFundamentalRegimeSignalProducer(
            flow_lookback=self.flow_lookback,
            liquidity_lookback=self.liquidity_lookback,
            value_weight=self.value_weight,
            quality_weight=self.quality_weight,
            market_filter_lookback=self.market_filter_lookback,
            market_filter_threshold=self.market_filter_threshold,
            min_breadth=self.min_breadth,
        )
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)
        self.position_policy = RegimeFilterPolicy(
            regime_key="market_filter_pass",
            off_exposure_multiplier=self.off_exposure_multiplier,
        )
