from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtesting.signals.base import SignalBundle
from backtesting.strategy.cross import RankLongOnly

from .base import ConstructionResult


@dataclass(slots=True)
class LongOnlyTopN:
    top_n: int = 20
    ranker: RankLongOnly = field(init=False)

    def __post_init__(self) -> None:
        self.ranker = RankLongOnly(top_n=self.top_n)

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.astype(float)
        valid_count = alpha.notna().sum(axis=1)
        selected_count = valid_count.clip(upper=self.top_n)
        ranks = alpha.rank(axis=1, ascending=False, method="first", na_option="bottom")
        selection_mask = ranks.le(self.top_n) & alpha.notna()

        denominator = selected_count.astype(float).where(selected_count.ne(0), float("nan"))
        base_target_weights = selection_mask.astype(float).div(denominator, axis=0)
        base_target_weights = base_target_weights.fillna(0.0).astype(float)
        return ConstructionResult(
            base_target_weights=base_target_weights,
            selection_mask=selection_mask,
            group_long_budget=None,
            group_short_budget=None,
            meta={},
        )
