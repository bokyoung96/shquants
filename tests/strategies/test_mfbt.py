import pandas as pd
from pandas.testing import assert_frame_equal

from backtesting.data import MarketData
from backtesting.strategies import build_strategy


def test_mfbt_price_momentum_emits_binary_signal_from_252_day_close_high() -> None:
    index = pd.date_range("2024-01-02", periods=253, freq="D")
    close = pd.DataFrame(
        {
            "A": [100.0] * 252 + [81.0],
            "B": [100.0] * 252 + [80.0],
            "C": [100.0] * 252 + [79.0],
        },
        index=index,
    )
    market = MarketData(frames={"close": close}, universe=None, benchmark=None)

    signal = build_strategy("mfbt").build_signal(market)

    expected = pd.DataFrame(0.0, index=index, columns=close.columns)
    expected.loc[index[-2], ["A", "B", "C"]] = 1.0
    expected.loc[index[-1], "A"] = 1.0
    assert_frame_equal(signal, expected)


def test_mfbt_builds_equal_weight_plan_for_price_momentum_names() -> None:
    index = pd.date_range("2024-01-02", periods=253, freq="D")
    close = pd.DataFrame(
        {
            "A": [100.0] * 252 + [90.0],
            "B": [100.0] * 252 + [85.0],
            "C": [100.0] * 252 + [75.0],
        },
        index=index,
    )
    market = MarketData(frames={"close": close}, universe=None, benchmark=None)

    plan = build_strategy("mfbt", top_n=2).build_plan(market)
    last = plan.target_weights.iloc[-1]

    assert last["A"] == 0.5
    assert last["B"] == 0.5
    assert last["C"] == 0.0
