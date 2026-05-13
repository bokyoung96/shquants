import warnings

import pandas as pd
from pandas.testing import assert_frame_equal

from backtesting.data import MarketData
from backtesting.strategies import RegisteredStrategy, build_strategy, list_strategies


def test_registry_lists_default_strategies() -> None:
    strategies = list_strategies()

    assert "momentum" in strategies
    assert "index_alpha_tilt_consensus_revision_oi_beta" in strategies
    assert "consensus_beta_persistence_concentrated_longonly" not in strategies
    assert "revision_asymmetric_relay_hedge_ls" not in strategies
    assert "revision_minparam_v02" not in strategies
    assert "revision_oi_beta_momo_gate_ls" not in strategies
    assert "revision_oi_high_beta_momentum_ls" not in strategies
    assert "revision_oi_soft_beta_tilt_momentum_ls" not in strategies
    assert "revision_oi_state_conditioned_beta_gate_ls" not in strategies
    assert "revision_oi_state_conditioned_short_squeeze_beta_cap_ls" not in strategies
    assert "revision_oi_state_conditioned_short_squeeze_beta_exclusion_ls" not in strategies


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
    assert_frame_equal(weights, plan.target_weights)
    assert weights.loc["2024-01-04", "A"] == 1.0
    assert weights.loc["2024-01-04", "B"] == 0.0


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


def test_index_alpha_tilt_uses_available_market_cap_dataset() -> None:
    strategy = build_strategy("index_alpha_tilt_consensus_revision_oi_beta")

    dataset_values = {dataset.value for dataset in strategy.datasets}

    assert "qw_mktcap" in dataset_values
    assert "qw_mktcap_flt" not in dataset_values


def test_soft_participation_index_overlay_uses_available_market_cap_dataset() -> None:
    strategy = build_strategy("consensus_beta_soft_participation_index_overlay")

    dataset_values = {dataset.value for dataset in strategy.datasets}

    assert "qw_mktcap" in dataset_values
    assert "qw_mktcap_flt" not in dataset_values


def test_index_alpha_tilt_overlays_active_weights_inside_k200_universe() -> None:
    index = pd.date_range("2024-01-02", periods=8, freq="D")
    close = pd.DataFrame(
        {
            "A": [100, 101, 103, 105, 108, 111, 114, 118],
            "B": [100, 100, 99, 98, 97, 96, 95, 94],
            "C": [50, 52, 54, 56, 58, 60, 62, 64],
            "D": [100, 100, 101, 101, 102, 102, 103, 103],
        },
        index=index,
        dtype=float,
    )
    eps = pd.DataFrame(
        {
            "A": [10, 10.2, 10.5, 10.9, 11.4, 11.9, 12.4, 13.0],
            "B": [10, 9.9, 9.7, 9.4, 9.1, 8.8, 8.5, 8.2],
            "C": [8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7],
            "D": [9, 9.0, 9.1, 9.1, 9.2, 9.2, 9.3, 9.3],
        },
        index=index,
        dtype=float,
    )
    op = eps * 2.0
    mktcap = pd.DataFrame(100.0, index=index, columns=close.columns)
    k200 = pd.DataFrame(
        {
            "A": [1] * len(index),
            "B": [1] * len(index),
            "C": [0] * len(index),
            "D": [1] * len(index),
        },
        index=index,
    )
    sector = pd.DataFrame(
        {
            "A": ["tech"] * len(index),
            "B": ["tech"] * len(index),
            "C": ["health"] * len(index),
            "D": ["industrial"] * len(index),
        },
        index=index,
    )
    foreign = pd.DataFrame(
        {"A": [3] * len(index), "B": [-3] * len(index), "C": [3] * len(index), "D": [0] * len(index)},
        index=index,
        dtype=float,
    )
    inst = pd.DataFrame(
        {"A": [2] * len(index), "B": [-2] * len(index), "C": [2] * len(index), "D": [0] * len(index)},
        index=index,
        dtype=float,
    )
    retail = pd.DataFrame(
        {"A": [-1] * len(index), "B": [3] * len(index), "C": [-1] * len(index), "D": [0] * len(index)},
        index=index,
        dtype=float,
    )
    benchmark = pd.DataFrame({"IKS200": [100, 101, 102, 103, 104, 105, 106, 107]}, index=index, dtype=float)

    market = MarketData(
        frames={
            "close": close,
            "benchmark": benchmark,
            "eps_fwd_q1": eps,
            "op_fwd_q1": op,
            "foreign_flow": foreign,
            "inst_flow": inst,
            "retail_flow": retail,
            "sector_big": sector,
            "market_cap": mktcap,
            "k200_yn": k200,
        },
        universe=None,
        benchmark=None,
    )
    strategy = build_strategy(
        "index_alpha_tilt_consensus_revision_oi_beta",
        lookback=2,
        flow_lookback=2,
        momentum_lookback=2,
        active_share_target=0.30,
        max_stock_active=0.20,
        max_sector_active=0.30,
        min_names=3,
    )

    plan = strategy.build_plan(market)
    last = plan.target_weights.iloc[-1].reindex(close.columns, fill_value=0.0)

    assert last["C"] == 0.0
    assert last.sum() == 1.0
    assert last["A"] > (1.0 / 3.0)
    assert last["B"] < (1.0 / 3.0)
