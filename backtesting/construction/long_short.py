from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
        alpha = bundle.alpha.astype(float)
        valid = alpha.notna()
        valid_count = valid.sum(axis=1).astype(int)
        short_count = (valid_count - 1).clip(lower=0, upper=self.bottom_n).astype(int)
        long_count = (valid_count - short_count).clip(lower=0, upper=self.top_n).astype(int)
        qualified = (long_count > 0) & (short_count > 0)
        long_count = long_count.where(qualified, 0)
        short_count = short_count.where(qualified, 0)

        long_rank = alpha.rank(axis=1, ascending=False, method="first", na_option="bottom")
        short_rank = alpha.rank(axis=1, ascending=True, method="first", na_option="bottom")
        selected_long = long_rank.le(long_count, axis=0) & valid
        selected_short = short_rank.le(short_count, axis=0) & valid & ~selected_long

        long_denominator = long_count.astype(float).replace(0.0, np.nan)
        short_denominator = short_count.astype(float).replace(0.0, np.nan)
        base_target_weights = (
            selected_long.astype(float).mul(float(self.gross_long), axis=0).div(long_denominator, axis=0)
            - selected_short.astype(float).mul(float(self.gross_short), axis=0).div(short_denominator, axis=0)
        )
        base_target_weights = base_target_weights.fillna(0.0).astype(float)
        selected_long = selected_long.astype(bool)
        selected_short = selected_short.astype(bool)
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
