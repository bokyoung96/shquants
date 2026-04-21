from dataclasses import dataclass

import pandas as pd

from .base import CrossSectionalStrategy, validate_positive


@dataclass(slots=True)
class RankLongOnly(CrossSectionalStrategy):
    top_n: int

    def __post_init__(self) -> None:
        validate_positive("top_n", self.top_n)

    def target_weights(self, signal: pd.Series) -> pd.Series:
        weights = self.zeros_like(signal)
        winners = signal.dropna().sort_values(ascending=False).head(self.top_n)
        if winners.empty:
            return weights

        weights.loc[winners.index] = 1.0 / len(winners)
        return weights


@dataclass(slots=True)
class RankLongShort(CrossSectionalStrategy):
    top_n: int
    bottom_n: int

    def __post_init__(self) -> None:
        validate_positive("top_n", self.top_n)
        validate_positive("bottom_n", self.bottom_n)

    def target_weights(self, signal: pd.Series) -> pd.Series:
        weights = self.zeros_like(signal)
        valid_signal = signal.dropna()
        long_leg = valid_signal.sort_values(ascending=False).head(self.top_n)
        short_pool = valid_signal.drop(index=long_leg.index, errors="ignore")
        short_leg = short_pool.sort_values(ascending=True).head(self.bottom_n)

        if not long_leg.empty:
            weights.loc[long_leg.index] = 1.0 / len(long_leg)
        if not short_leg.empty:
            weights.loc[short_leg.index] = -1.0 / len(short_leg)
        return weights
