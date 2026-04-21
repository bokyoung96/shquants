import pandas as pd

from backtesting.strategy import TimeSeriesStrategy
from backtesting.strategy.timeseries import ThresholdTrend


def test_threshold_trend_goes_long_above_threshold() -> None:
    signal = pd.Series({"A": -1.0, "B": 0.0, "C": 0.2, "D": 0.8})
    strategy = ThresholdTrend(threshold=0.1)

    weights = strategy.target_weights(signal)

    assert weights["A"] == 0.0
    assert weights["B"] == 0.0
    assert weights["C"] == 1.0
    assert weights["D"] == 1.0


def test_threshold_trend_exposes_time_series_extension_point() -> None:
    assert isinstance(ThresholdTrend(), TimeSeriesStrategy)
