from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    def zeros_like(self, signal: pd.Series) -> pd.Series:
        return pd.Series(0.0, index=signal.index, dtype=float)

    @abstractmethod
    def target_weights(self, signal: pd.Series) -> pd.Series:
        raise NotImplementedError


class CrossSectionalStrategy(BaseStrategy):
    pass


class TimeSeriesStrategy(BaseStrategy):
    pass


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
