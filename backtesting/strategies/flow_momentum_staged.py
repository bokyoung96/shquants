from __future__ import annotations

from dataclasses import dataclass

from backtesting.construction.long_only import LongOnlyTopN
from backtesting.policy.staged import BudgetPreservingStagedPolicy, BucketDefinition, StagedRuleSet
from backtesting.signals.flow_momentum_staged import FlowMomentumStagedSignalProducer

from .composable import ComposableStrategy


@dataclass(slots=True)
class FlowMomentumStagedTopN(ComposableStrategy):
    top_n: int = 20
    flow_lookback: int = 20
    momentum_lookback: int = 120

    def __post_init__(self) -> None:
        self.signal_producer = FlowMomentumStagedSignalProducer(
            flow_lookback=self.flow_lookback,
            momentum_lookback=self.momentum_lookback,
        )
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)
        self.position_policy = BudgetPreservingStagedPolicy(
            buckets=(
                BucketDefinition("entry", 1 / 3),
                BucketDefinition("add_1", 1 / 3),
                BucketDefinition("add_2", 1 / 3),
            ),
            rules=StagedRuleSet(
                entry_key="eligible_entry",
                add_keys=("eligible_add_1", "eligible_add_2"),
                exit_key="eligible_exit",
            ),
        )
