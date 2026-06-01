import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from backtesting.data import MarketData
from backtesting.strategies import RegisteredStrategy, build_strategy, list_strategies


def test_strategy_modules_export_simple_class_names() -> None:
    from backtesting.strategies.benchmark_overlay import BenchmarkOverlay
    from backtesting.strategies.benchmark_tilt import BenchmarkTilt
    from backtesting.strategies.earnings_revision import EarningsRevision
    from backtesting.strategies.mfbt import Mfbt
    from backtesting.strategies.revision_signal import RevisionSignal
    from backtesting.strategies.rrg_sector_rotation import RrgSectorRotation

    assert BenchmarkOverlay.__name__ == "BenchmarkOverlay"
    assert BenchmarkTilt.__name__ == "BenchmarkTilt"
    assert EarningsRevision.__name__ == "EarningsRevision"
    assert Mfbt.__name__ == "Mfbt"
    assert RevisionSignal.__name__ == "RevisionSignal"
    assert RrgSectorRotation.__name__ == "RrgSectorRotation"


def test_registry_lists_default_strategies() -> None:
    strategies = list_strategies()

    assert "trend_rank" in strategies
    assert "earnings_revision" in strategies
    assert "revision_signal" in strategies
    assert "benchmark_tilt" in strategies
    assert "benchmark_overlay" in strategies
    assert "mfbt" in strategies
    assert "rrg_sector_rotation" in strategies
    assert "index_alpha_tilt_consensus_revision_oi_beta" not in strategies
    assert "q1q5_ls" not in strategies
    assert "squeeze_ls" not in strategies
    assert "beta_boost_ls" not in strategies
    assert "regime_ls" not in strategies
    assert "sector_tilt" not in strategies
    assert "breadth_long" not in strategies
    assert "soft_long" not in strategies
    assert "consensus_beta_persistence_concentrated_longonly" not in strategies
    assert "revision_asymmetric_relay_hedge_ls" not in strategies
    assert "revision_minparam_v02" not in strategies
    assert "revision_oi_beta_momo_gate_ls" not in strategies
    assert "revision_oi_high_beta_momentum_ls" not in strategies
    assert "revision_oi_soft_beta_tilt_momentum_ls" not in strategies
    assert "revision_oi_state_conditioned_beta_gate_ls" not in strategies
    assert "revision_oi_state_conditioned_short_squeeze_beta_cap_ls" not in strategies
    assert "revision_oi_state_conditioned_short_squeeze_beta_exclusion_ls" not in strategies
    assert "rrg-fwd-flow1-ls" not in strategies
    assert "rrg-fwd-flow1-ls-gs0.5-listed-exit-validated" not in strategies
    assert "rrg-fwd-flow1-ls03-change10-etf-shortoff-research" not in strategies
    assert "rrg-fwd-flow1-ls-lag31-monthly-gs0.0-l5-validated" not in strategies


def test_registry_rejects_old_long_strategy_names() -> None:
    with pytest.raises(KeyError):
        build_strategy("index_alpha_tilt_consensus_revision_oi_beta")


def test_registry_lists_screened_strategy_names_only() -> None:
    strategies = set(list_strategies())

    assert strategies == {
        "trend_rank",
        "earnings_revision",
        "revision_signal",
        "benchmark_overlay",
        "benchmark_tilt",
        "mfbt",
        "rrg_sector_rotation",
    }
    assert "consensus_beta_soft_participation_benchmark_overlay" not in strategies
    assert "rrg-fwd-flow1-ls" not in strategies


def test_rrg_sector_rotation_uses_wi26_big_sector_dataset() -> None:
    from backtesting.catalog import DatasetId

    strategy = build_strategy("rrg_sector_rotation")
    dataset_values = {dataset.value for dataset in strategy.datasets}

    assert DatasetId.QW_WI_SEC_26_BIG.value in dataset_values
    assert DatasetId.QW_WICS_SEC_BIG.value not in dataset_values


def test_rrg_sector_rotation_ignores_flow_when_op_revision_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from backtesting.strategies import rrg_sector_rotation as module

    index = pd.date_range("2024-01-02", periods=60, freq="D")
    columns = ["A", "B"]
    close = pd.DataFrame({"A": np.linspace(100.0, 130.0, len(index)), "B": np.linspace(100.0, 125.0, len(index))}, index=index)
    k200 = pd.DataFrame(True, index=index, columns=columns)
    sector = pd.DataFrame({"A": "Tech", "B": "Tech"}, index=index)
    market_cap = pd.DataFrame(100.0, index=index, columns=columns)
    benchmark = pd.DataFrame({"IKS200": np.linspace(100.0, 125.0, len(index))}, index=index)
    empty = pd.DataFrame(np.nan, index=index, columns=columns)
    zeros = pd.DataFrame(0.0, index=index, columns=columns)
    market = MarketData(
        frames={
            "close": close,
            "k200_yn": k200,
            "sector_big": sector,
            "market_cap": market_cap,
            "benchmark": benchmark,
            "eps_fwd_q1": empty,
            "eps_fwd_q2": empty,
            "eps_fwd": empty,
            "op_fwd_q1": empty,
            "op_fwd_q2": empty,
            "op_fwd": empty,
            "trading_value": pd.DataFrame(1_000_000.0, index=index, columns=columns),
            "foreign_flow": zeros,
            "inst_flow": zeros,
            "retail_flow": zeros,
        },
        universe=None,
        benchmark=None,
    )

    def fake_rrg_context(**_: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        state = pd.DataFrame({"Tech": "Leading"}, index=index)
        return state, state.eq("Leading"), state.eq("Lagging")

    stock_op = pd.DataFrame(np.nan, index=index, columns=columns)
    sector_op = pd.DataFrame({"Tech": np.nan}, index=index)

    monkeypatch.setattr(module, "_build_rrg_context", fake_rrg_context)
    monkeypatch.setattr(module, "_build_stock_op_revision", lambda **_: stock_op)
    monkeypatch.setattr(module, "_build_sector_op_revision", lambda **_: sector_op)

    strategy = build_strategy("rrg_sector_rotation", gross_short=0.0)
    weights = strategy.build_weights(market)
    last = weights.iloc[-1]

    assert last["A"] == 0.0
    assert last["B"] == 0.0


def test_rrg_op_revision_uses_prior_based_change() -> None:
    from backtesting.strategies.rrg_sector_rotation import _build_stock_op_revision

    index = pd.date_range("2024-01-02", periods=3, freq="D")
    columns = ["A", "B", "C", "D"]
    sector = pd.DataFrame("Tech", index=index, columns=columns)
    prior = pd.Series({"A": 100.0, "B": -100.0, "C": 0.0, "D": 1.0})
    current = pd.Series({"A": 120.0, "B": -80.0, "C": 10.0, "D": 100.0})
    op = pd.DataFrame([prior, prior, current], index=index)

    actual = _build_stock_op_revision(
        frames={"op_fwd_q1": op, "op_fwd_q2": op, "op_fwd": op},
        index=index,
        columns=pd.Index(columns),
        sector=sector,
        lookback=1,
    )

    last = actual.iloc[-1]
    assert last["A"] == pytest.approx(0.20)
    assert last["B"] == pytest.approx(0.20)
    assert pd.isna(last["C"])
    assert last["D"] == pytest.approx(1.0)


def test_rrg_sector_rotation_weights_op_revision_scores_and_keeps_weakening_off_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backtesting.strategies import rrg_sector_rotation as module

    index = pd.date_range("2024-01-02", periods=60, freq="D")
    columns = ["L1", "L2", "S1", "S2", "N"]
    close = pd.DataFrame(
        {symbol: np.linspace(100.0, 120.0, len(index)) for symbol in columns},
        index=index,
    )
    k200 = pd.DataFrame(True, index=index, columns=columns)
    sector = pd.DataFrame(
        {
            "L1": "Strong",
            "L2": "Strong",
            "S1": "Weak",
            "S2": "WeakeningSector",
            "N": "Neutral",
        },
        index=index,
    )
    market_cap = pd.DataFrame(100.0, index=index, columns=columns)
    benchmark = pd.DataFrame({"IKS200": np.linspace(100.0, 115.0, len(index))}, index=index)
    empty = pd.DataFrame(np.nan, index=index, columns=columns)
    zeros = pd.DataFrame(0.0, index=index, columns=columns)
    market = MarketData(
        frames={
            "close": close,
            "k200_yn": k200,
            "sector_big": sector,
            "market_cap": market_cap,
            "benchmark": benchmark,
            "eps_fwd_q1": empty,
            "eps_fwd_q2": empty,
            "eps_fwd": empty,
            "op_fwd_q1": empty,
            "op_fwd_q2": empty,
            "op_fwd": empty,
            "trading_value": pd.DataFrame(1_000_000.0, index=index, columns=columns),
            "foreign_flow": zeros,
            "inst_flow": zeros,
            "retail_flow": zeros,
        },
        universe=None,
        benchmark=None,
    )

    def fake_rrg_context(**_: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        state = pd.DataFrame(
            {"Strong": "Leading", "Weak": "Lagging", "WeakeningSector": "Weakening", "Neutral": "Improving"},
            index=index,
        )
        long_sector = state.isin(("Leading", "Improving", "Weakening"))
        short_sector = state.eq("Lagging")
        return state, long_sector, short_sector

    stock_op = pd.DataFrame(
        {
            "L1": 0.4,
            "L2": 0.3,
            "S1": -0.4,
            "S2": -0.2,
            "N": 0.2,
        },
        index=index,
    )
    sector_op = pd.DataFrame(
        {"Strong": 0.5, "Weak": -0.5, "WeakeningSector": -0.5, "Neutral": 0.2},
        index=index,
    )

    monkeypatch.setattr(module, "_build_rrg_context", fake_rrg_context)
    monkeypatch.setattr(module, "_build_stock_op_revision", lambda **_: stock_op)
    monkeypatch.setattr(module, "_build_sector_op_revision", lambda **_: sector_op)

    strategy = build_strategy("rrg_sector_rotation", max_long_names=2, max_short_names=2, gross_short=0.5)
    weights = strategy.build_weights(market)
    last = weights.iloc[-1]

    assert last["L1"] == pytest.approx(0.5)
    assert last["L2"] == pytest.approx(1.0 / 3.0)
    assert last["N"] == pytest.approx(1.0 / 6.0)
    assert last["S1"] == pytest.approx(-0.5)
    assert last["S2"] == 0.0
    assert last[last.gt(0.0)].sum() == pytest.approx(1.0)
    assert last[last.lt(0.0)].sum() == pytest.approx(-0.5)


def test_momentum_strategy_builds_weights() -> None:
    strategy = build_strategy("trend_rank", top_n=1, lookback=1)
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
    strategy = build_strategy("trend_rank", top_n=1, lookback=1)
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
    strategy = build_strategy("trend_rank", top_n=1, lookback=1)
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


def test_benchmark_tilt_uses_available_market_cap_dataset() -> None:
    strategy = build_strategy("benchmark_tilt")

    dataset_values = {dataset.value for dataset in strategy.datasets}

    assert "qw_mktcap" in dataset_values
    assert "qw_mktcap_flt" not in dataset_values


def test_revision_signal_holds_all_positive_revision_names_without_top_n() -> None:
    index = pd.date_range("2024-01-02", periods=125, freq="D")
    close = pd.DataFrame(
        {
            "A": [100.0 + i for i in range(len(index))],
            "B": [100.0 + i for i in range(len(index))],
            "C": [100.0 + i for i in range(len(index))],
        },
        index=index,
    )
    eps = pd.DataFrame(
        {
            "A": [10.0 + 0.1 * i for i in range(len(index))],
            "B": [11.0 + 0.1 * i for i in range(len(index))],
            "C": [12.0 - 0.1 * i for i in range(len(index))],
        },
        index=index,
    )
    op = eps * 2.0
    benchmark = pd.DataFrame({"IKS200": [100.0 + i for i in range(len(index))]}, index=index)
    market = MarketData(
        frames={
            "close": close,
            "eps_fwd_q1": eps,
            "op_fwd_q1": op,
            "benchmark": benchmark,
        },
        universe=None,
        benchmark=None,
    )
    strategy = build_strategy("revision_signal", lookback=20)

    plan = strategy.build_plan(market)
    last = plan.target_weights.iloc[-1]

    assert last["A"] == 0.5
    assert last["B"] == 0.5
    assert last["C"] == 0.0


def test_revision_signal_moves_to_cash_when_benchmark_trend_fails() -> None:
    index = pd.date_range("2024-01-02", periods=125, freq="D")
    close = pd.DataFrame({"A": [100.0 + i for i in range(len(index))]}, index=index)
    eps = pd.DataFrame({"A": [10.0 + 0.1 * i for i in range(len(index))]}, index=index)
    op = eps * 2.0
    benchmark = pd.DataFrame({"IKS200": [200.0 - i for i in range(len(index))]}, index=index)
    market = MarketData(
        frames={
            "close": close,
            "eps_fwd_q1": eps,
            "op_fwd_q1": op,
            "benchmark": benchmark,
        },
        universe=None,
        benchmark=None,
    )
    strategy = build_strategy("revision_signal", lookback=20)

    plan = strategy.build_plan(market)

    assert plan.target_weights.iloc[-1].sum() == 0.0


def test_soft_participation_benchmark_overlay_uses_available_market_cap_dataset() -> None:
    strategy = build_strategy("benchmark_overlay")

    dataset_values = {dataset.value for dataset in strategy.datasets}

    assert "qw_mktcap" in dataset_values
    assert "qw_mktcap_flt" not in dataset_values


def test_benchmark_tilt_overlays_active_weights_inside_k200_universe() -> None:
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
        "benchmark_tilt",
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


def test_benchmark_overlay_fast_active_helper_matches_reference() -> None:
    from backtesting.strategies.benchmark_overlay import _BenchmarkOverlayConstruction

    construction = _BenchmarkOverlayConstruction(
        active_share_target=0.30,
        max_stock_active=0.08,
        max_sector_active=0.12,
        min_names=5,
    )
    index = pd.Index(["A", "B", "C", "D", "E", "F"])
    signal = pd.Series([1.4, -0.8, 0.4, -0.2, 0.1, -0.05], index=index, dtype=float)
    base = pd.Series([0.30, 0.22, 0.18, 0.14, 0.10, 0.06], index=index, dtype=float)
    sector = pd.Series(["tech", "tech", "finance", "finance", "industrial", "industrial"], index=index)

    expected = construction._build_active_overlay(signal=signal, base=base, sector_row=sector, scale=0.75)
    actual_values = construction._build_active_overlay_values(
        signal=signal.to_numpy(dtype=float),
        base=base.to_numpy(dtype=float),
        sector=sector.to_numpy(dtype=object),
        scale=0.75,
    )
    actual = pd.Series(actual_values, index=index, dtype=float)

    assert_series_equal(actual, expected, check_exact=False, atol=1e-12, rtol=1e-12)
    assert np.isfinite(actual_values).all()


def test_benchmark_overlay_cross_sectional_zscore_matches_reference() -> None:
    from backtesting.strategies.benchmark_overlay import _BenchmarkOverlaySignal

    frame = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [3.0, None],
            "C": [5.0, 4.0],
            "D": [7.0, 6.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    membership = pd.DataFrame(
        {
            "A": [True, True],
            "B": [True, True],
            "C": [False, True],
            "D": [False, False],
        },
        index=frame.index,
    )

    expected_rows = []
    for timestamp in frame.index:
        members = membership.loc[timestamp][membership.loc[timestamp]].index
        expected_rows.append(_BenchmarkOverlaySignal._zscore(frame.loc[timestamp].reindex(members).fillna(0.0)))
    expected = pd.DataFrame(expected_rows, index=frame.index).reindex(columns=frame.columns).fillna(0.0)

    actual = _BenchmarkOverlaySignal._cross_sectional_zscore(frame, membership)

    assert_frame_equal(actual, expected, check_exact=False, atol=1e-12, rtol=1e-12)
