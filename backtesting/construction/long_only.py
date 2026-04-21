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
        rows: dict[pd.Timestamp, pd.Series] = {}
        selected: dict[pd.Timestamp, pd.Series] = {}

        for timestamp in bundle.alpha.index:
            weights = self.ranker.target_weights(bundle.alpha.loc[timestamp])
            rows[timestamp] = weights
            selected[timestamp] = weights.ne(0.0)

        base_target_weights = (
            pd.DataFrame.from_dict(rows, orient="index")
            .reindex(index=bundle.alpha.index, columns=bundle.alpha.columns)
            .fillna(0.0)
            .astype(float)
        )
        selection_mask = (
            pd.DataFrame.from_dict(selected, orient="index")
            .reindex(index=bundle.alpha.index, columns=bundle.alpha.columns)
            .fillna(False)
            .astype(bool)
        )
        return ConstructionResult(
            base_target_weights=base_target_weights,
            selection_mask=selection_mask,
            group_long_budget=None,
            group_short_budget=None,
            meta={},
        )
