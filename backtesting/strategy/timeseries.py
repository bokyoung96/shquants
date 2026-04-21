from dataclasses import dataclass

import pandas as pd

from .base import TimeSeriesStrategy


@dataclass(slots=True)
class ThresholdTrend(TimeSeriesStrategy):
    threshold: float = 0.0

    def target_weights(self, signal: pd.Series) -> pd.Series:
        return signal.gt(self.threshold).astype(float)
