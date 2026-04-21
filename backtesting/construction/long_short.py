from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.signals.base import SignalBundle
from backtesting.strategy.base import validate_positive

from .base import ConstructionResult


@dataclass(slots=True)
class LongShortTopBottom:
    top_n: int
    bottom_n: int
    gross_long: float = 1.0
    gross_short: float = 1.0

    def __post_init__(self) -> None:
        validate_positive("top_n", self.top_n)
        validate_positive("bottom_n", self.bottom_n)

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha
        weights_by_date: dict[pd.Timestamp, pd.Series] = {}
        selected_long_by_date: dict[pd.Timestamp, pd.Series] = {}
        selected_short_by_date: dict[pd.Timestamp, pd.Series] = {}

        for timestamp in alpha.index:
            signal = alpha.loc[timestamp].dropna()
            weights = pd.Series(0.0, index=alpha.columns, dtype=float)
            long_count, short_count = _leg_sizes(
                available_count=len(signal),
                top_n=self.top_n,
                bottom_n=self.bottom_n,
            )
            if long_count > 0 and short_count > 0:
                longs = signal.sort_values(ascending=False).head(long_count)
                short_pool = signal.drop(index=longs.index, errors="ignore")
                shorts = short_pool.sort_values(ascending=True).head(short_count)

                weights.loc[longs.index] = self.gross_long / len(longs)
                weights.loc[shorts.index] = -self.gross_short / len(shorts)

            weights_by_date[timestamp] = weights
            selected_long_by_date[timestamp] = weights.gt(0.0)
            selected_short_by_date[timestamp] = weights.lt(0.0)

        base_target_weights = (
            pd.DataFrame.from_dict(weights_by_date, orient="index")
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(0.0)
            .astype(float)
        )
        selected_long = (
            pd.DataFrame.from_dict(selected_long_by_date, orient="index")
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(False)
            .astype(bool)
        )
        selected_short = (
            pd.DataFrame.from_dict(selected_short_by_date, orient="index")
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(False)
            .astype(bool)
        )
        return ConstructionResult(
            base_target_weights=base_target_weights,
            selection_mask=base_target_weights.ne(0.0),
            group_long_budget=None,
            group_short_budget=None,
            meta={
                "selected_long": selected_long,
                "selected_short": selected_short,
            },
        )


def _leg_sizes(available_count: int, top_n: int, bottom_n: int) -> tuple[int, int]:
    if available_count < 2:
        return 0, 0

    short_count = min(bottom_n, available_count - 1)
    long_count = min(top_n, available_count - short_count)
    if long_count <= 0 or short_count <= 0:
        return 0, 0
    return long_count, short_count
