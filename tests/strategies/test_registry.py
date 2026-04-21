from pathlib import Path
import warnings

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backtesting.data import MarketData
from backtesting.strategies import RegisteredStrategy, build_strategy, list_strategies


def test_registry_lists_default_strategies() -> None:
    assert set(list_strategies()) == {
        "momentum",
        "op_fwd_yield",
        "breakout_52w_simple",
        "breakout_52w_staged",
    }


def test_momentum_strategy_builds_weights() -> None:
    strategy = build_strategy("momentum", top_n=1, lookback=1)
    close = pd.DataFrame(
        {
            "A": [10.0, 11.0, 12.0],
            "B": [10.0, 10.0, 9.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    market = MarketData(frames={"close": close}, universe=None, benchmark=None)

    plan = strategy.build_plan(market)
    weights = strategy.build_weights(market)

    assert plan.bucket_ledger["bucket_id"].eq("base").all()
    assert plan.target_weights.loc["2024-01-04", "A"] == 1.0
    assert plan.target_weights.loc["2024-01-04", "B"] == 0.0
    assert weights.equals(plan.target_weights)
    assert weights.loc["2024-01-04", "A"] == 1.0
    assert weights.loc["2024-01-04", "B"] == 0.0


def test_op_fwd_yield_strategy_builds_weights() -> None:
    strategy = build_strategy("op_fwd_yield", top_n=1)
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    market = MarketData(
        frames={
            "op_fwd": pd.DataFrame({"A": [20.0, 20.0], "B": [5.0, 5.0]}, index=index),
            "market_cap": pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0]}, index=index),
        },
        universe=None,
        benchmark=None,
    )

    weights = strategy.build_weights(market)

    assert weights.loc["2024-01-03", "A"] == 1.0
    assert weights.loc["2024-01-03", "B"] == 0.0


def test_op_fwd_yield_strategy_builds_plan() -> None:
    strategy = build_strategy("op_fwd_yield", top_n=1)
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    market = MarketData(
        frames={
            "op_fwd": pd.DataFrame({"A": [20.0, 20.0], "B": [5.0, 5.0]}, index=index),
            "market_cap": pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0]}, index=index),
        },
        universe=None,
        benchmark=None,
    )

    plan = strategy.build_plan(market)

    assert_frame_equal(plan.target_weights, strategy.build_weights(market))
    assert plan.bucket_ledger["bucket_id"].eq("base").all()


def test_breakout_52w_simple_enters_on_prior_252_day_high_break_and_exits_on_20_day_low_break() -> None:
    strategy = build_strategy("breakout_52w_simple")
    index = pd.date_range("2024-01-01", periods=274, freq="B")
    close = pd.DataFrame(
        {
            "A": [
                *([100.0] * 252),
                101.0,
                *([102.0] * 20),
                99.0,
            ]
        },
        index=index,
    )
    market = MarketData(frames={"close": close}, universe=None, benchmark=None)

    plan = strategy.build_plan(market)

    assert plan.target_weights.loc[close.index[251], "A"] == 0.0
    assert plan.target_weights.loc[close.index[252], "A"] == 1.0
    assert plan.target_weights.loc[close.index[272], "A"] == 1.0
    assert plan.target_weights.loc[close.index[273], "A"] == 0.0


def test_breakout_52w_staged_adds_second_and_third_buckets_after_10dma_pullback_and_rebreak() -> None:
    strategy = build_strategy("breakout_52w_staged")
    index = pd.date_range("2024-01-01", periods=270, freq="B")
    close = pd.DataFrame(
        {
            "A": [
                *([100.0] * 252),
                101.0,
                103.0,
                105.0,
                104.0,
                103.0,
                102.0,
                104.5,
                106.0,
                104.0,
                103.0,
                102.0,
                105.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
            ]
        },
        index=index,
    )
    market = MarketData(frames={"close": close}, universe=None, benchmark=None)

    plan = strategy.build_plan(market)

    first_breakout = close.index[252]
    first_rebreak = close.index[258]
    second_rebreak = close.index[263]
    assert plan.target_weights.loc[first_breakout, "A"] == pytest.approx(1 / 3)
    assert plan.target_weights.loc[first_rebreak, "A"] == pytest.approx(2 / 3)
    assert plan.target_weights.loc[second_rebreak, "A"] == pytest.approx(1.0)


def test_breakout_52w_staged_exits_in_three_steps_after_20_day_low_break() -> None:
    strategy = build_strategy("breakout_52w_staged")
    index = pd.date_range("2024-01-01", periods=290, freq="B")
    close = pd.DataFrame(
        {
            "A": [
                *([100.0] * 252),
                101.0,
                103.0,
                105.0,
                104.0,
                103.0,
                102.0,
                104.5,
                106.0,
                104.0,
                103.0,
                102.0,
                105.0,
                107.0,
                90.0,
                89.0,
                88.0,
            ]
        },
        index=index[:268],
    )
    market = MarketData(frames={"close": close}, universe=None, benchmark=None)

    plan = strategy.build_plan(market)

    assert plan.target_weights.iloc[-3, 0] == pytest.approx(2 / 3)
    assert plan.target_weights.iloc[-2, 0] == pytest.approx(1 / 3)
    assert plan.target_weights.iloc[-1, 0] == 0.0


def test_registered_strategy_preserves_legacy_extension_path() -> None:
    class LegacyStrategy(RegisteredStrategy):
        @property
        def datasets(self) -> tuple:
            return ()

        def build_signal(self, market: MarketData) -> pd.DataFrame:
            return market.frames["close"].pct_change(fill_method=None)

        def target_weights(self, signal: pd.Series) -> pd.Series:
            weights = pd.Series(0.0, index=signal.index, dtype=float)
            winner = signal.dropna().sort_values(ascending=False).head(1)
            if not winner.empty:
                weights.loc[winner.index] = 1.0
            return weights

    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 10.0, 9.0]}, index=index)
    market = MarketData(frames={"close": close}, universe=None, benchmark=None)

    strategy = LegacyStrategy()

    plan = strategy.build_plan(market)

    assert plan.target_weights.loc["2024-01-04", "A"] == 1.0
    assert plan.bucket_ledger["bucket_id"].eq("base").all()


def test_momentum_strategy_avoids_future_warning_on_pct_change() -> None:
    strategy = build_strategy("momentum", top_n=1, lookback=1)
    close = pd.DataFrame(
        {
            "A": [10.0, 11.0, 12.0],
            "B": [10.0, 10.0, 9.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    market = MarketData(frames={"close": close}, universe=None, benchmark=None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        strategy.build_signal(market)

    assert not any(item.category is FutureWarning for item in caught)


def test_registered_strategy_avoids_future_warning_when_masking_universe() -> None:
    strategy = build_strategy("momentum", top_n=1, lookback=1)
    close = pd.DataFrame(
        {
            "A": [10.0, 11.0, 12.0],
            "B": [10.0, 10.0, 9.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    universe = pd.DataFrame(
        {
            "A": [True, True, True],
            "B": [True, None, True],
        },
        index=close.index,
    )
    market = MarketData(frames={"close": close}, universe=universe, benchmark=None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        strategy.build_weights(market)

    assert not any(item.category is FutureWarning for item in caught)
