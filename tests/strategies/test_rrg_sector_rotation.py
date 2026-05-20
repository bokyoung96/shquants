from __future__ import annotations

import pandas as pd
import pytest

from backtesting.data.loader import MarketData
from backtesting.signals.base import SignalBundle
from backtesting.strategies.rrg_sector_rotation import (
    RrgFwdBenchmarkTilt,
    RrgPureSectorRotation,
    RrgSectorRotation,
    _RrgSectorRotationSignal,
    _RrgPureSectorConstruction,
    _SparseBenchmarkOverlayConstruction,
    _bounded_delta,
    _build_forward_entry_mask,
    _build_pure_sector_budget,
    _build_state_equal_sector_weight_basis,
    _classify_rrg_states,
    _sector_rank,
)


def test_bounded_delta_clips_negative_to_positive_explosion() -> None:
    current = pd.DataFrame({"A": [10.0], "B": [12.0]}, index=pd.to_datetime(["2024-01-02"]))
    prior = pd.DataFrame({"A": [-90.0], "B": [10.0]}, index=current.index)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"]}, index=current.index)

    actual = _bounded_delta(current=current, prior=prior, sector=sector)

    assert actual.loc[current.index[0], "A"] == pytest.approx(1.0)
    assert actual.loc[current.index[0], "B"] == pytest.approx(2.0 / 51.0)


def test_sector_rank_scores_within_each_sector() -> None:
    index = pd.to_datetime(["2024-01-02"])
    values = pd.DataFrame({"A": [3.0], "B": [1.0], "C": [8.0], "D": [2.0]}, index=index)
    sector = pd.DataFrame(
        {"A": ["Tech"], "B": ["Tech"], "C": ["Finance"], "D": ["Finance"]},
        index=index,
    )

    actual = _sector_rank(values, sector=sector, ascending=True)

    assert actual.loc[index[0], "A"] == pytest.approx(1.0)
    assert actual.loc[index[0], "B"] == pytest.approx(0.5)
    assert actual.loc[index[0], "C"] == pytest.approx(1.0)
    assert actual.loc[index[0], "D"] == pytest.approx(0.5)


def test_classify_rrg_states_maps_quadrants_to_sector_legs() -> None:
    index = pd.to_datetime(["2024-01-02"])
    relative_strength = pd.DataFrame(
        {
            "Leading": [0.20],
            "Improving": [-0.10],
            "Lagging": [-0.20],
            "Weakening": [0.10],
        },
        index=index,
    )
    momentum = pd.DataFrame(
        {
            "Leading": [0.10],
            "Improving": [0.10],
            "Lagging": [-0.10],
            "Weakening": [-0.10],
        },
        index=index,
    )

    states, long_sector, short_sector = _classify_rrg_states(
        relative_strength=relative_strength,
        momentum=momentum,
    )

    assert states.loc[index[0], "Leading"] == "Leading"
    assert states.loc[index[0], "Improving"] == "Improving"
    assert states.loc[index[0], "Lagging"] == "Lagging"
    assert states.loc[index[0], "Weakening"] == "Weakening"
    assert bool(long_sector.loc[index[0], "Leading"])
    assert bool(long_sector.loc[index[0], "Improving"])
    assert not bool(long_sector.loc[index[0], "Lagging"])
    assert bool(short_sector.loc[index[0], "Lagging"])
    assert bool(short_sector.loc[index[0], "Weakening"])
    assert not bool(short_sector.loc[index[0], "Leading"])


def test_rrg_sector_rotation_declares_required_datasets() -> None:
    strategy = RrgSectorRotation()

    dataset_values = {dataset.value for dataset in strategy.datasets}

    assert dataset_values == {
        "qw_adj_c",
        "qw_BM",
        "qw_k200_yn",
        "qw_wics_sec_big",
        "qw_mktcap",
        "qw_v",
        "qw_eps_nfq1",
        "qw_eps_nfq2",
        "qw_eps_nfy1",
        "qw_op_nfq1",
        "qw_op_nfq2",
        "qw_op_nfy1",
        "qw_foreign",
        "qw_institution",
        "qw_retail",
    }


def test_rrg_sector_rotation_flow_only_excludes_forward_datasets() -> None:
    strategy = RrgSectorRotation(alpha_mode="flow_only")

    dataset_values = {dataset.value for dataset in strategy.datasets}

    assert "qw_eps_nfq1" not in dataset_values
    assert "qw_eps_nfq2" not in dataset_values
    assert "qw_eps_nfy1" not in dataset_values
    assert "qw_op_nfq1" not in dataset_values
    assert "qw_op_nfq2" not in dataset_values
    assert "qw_op_nfy1" not in dataset_values
    assert {"qw_foreign", "qw_institution", "qw_retail", "qw_v"}.issubset(dataset_values)


def test_rrg_sector_rotation_flow_only_builds_without_forward_frames() -> None:
    market = _rrg_market()
    frames = {
        key: value
        for key, value in market.frames.items()
        if key not in {"eps_fwd_q1", "eps_fwd_q2", "eps_fwd", "op_fwd_q1", "op_fwd_q2", "op_fwd"}
    }
    market = MarketData(frames=frames, universe=market.universe, benchmark=market.benchmark)
    strategy = RrgSectorRotation(
        top_n=2,
        bottom_n=2,
        alpha_mode="flow_only",
        gross_short=0.0,
    )

    bundle = strategy.signal_producer.build(market)
    weights = strategy.build_weights(market)

    assert "flow_score_20d" in bundle.meta
    assert "fwd_score" not in bundle.meta
    assert weights.max().max() > 0.0
    assert weights.min().min() == pytest.approx(0.0)


def test_rrg_sector_rotation_fwd_only_excludes_flow_datasets() -> None:
    strategy = RrgSectorRotation(alpha_mode="fwd_only")

    dataset_values = {dataset.value for dataset in strategy.datasets}

    assert "qw_v" not in dataset_values
    assert "qw_foreign" not in dataset_values
    assert "qw_institution" not in dataset_values
    assert "qw_retail" not in dataset_values
    assert {"qw_eps_nfq1", "qw_eps_nfq2", "qw_eps_nfy1", "qw_op_nfq1", "qw_op_nfq2", "qw_op_nfy1"}.issubset(
        dataset_values
    )


def test_rrg_sector_rotation_fwd_only_builds_without_flow_frames() -> None:
    market = _rrg_market()
    frames = {
        key: value
        for key, value in market.frames.items()
        if key not in {"volume", "foreign_flow", "inst_flow", "retail_flow"}
    }
    market = MarketData(frames=frames, universe=market.universe, benchmark=market.benchmark)
    strategy = RrgSectorRotation(
        top_n=2,
        bottom_n=2,
        alpha_mode="fwd_only",
        gross_short=0.0,
    )

    bundle = strategy.signal_producer.build(market)
    weights = strategy.build_weights(market)

    assert "fwd_score" in bundle.meta
    assert "flow_score_20d" not in bundle.meta
    assert weights.max().max() > 0.0
    assert weights.min().min() == pytest.approx(0.0)


def test_rrg_sector_rotation_fwd_only_does_not_force_fill_top_n() -> None:
    market = _rrg_market()
    index = market.frames["close"].index
    frames = dict(market.frames)
    for key in ("op_fwd_q1", "op_fwd_q2", "op_fwd"):
        frame = frames[key].copy()
        frame["B"] = [40.0 - i * 0.04 for i in range(len(index))]
        frames[key] = frame
    market = MarketData(frames=frames, universe=market.universe, benchmark=market.benchmark)
    strategy = RrgSectorRotation(
        top_n=2,
        bottom_n=2,
        alpha_mode="fwd_only",
        gross_short=0.0,
    )

    bundle = strategy.signal_producer.build(market)
    weights = strategy.build_weights(market)
    last_date = weights.index[-1]
    last = weights.loc[last_date]

    assert bool(bundle.context["long_entry"].loc[last_date, "A"])
    assert not bool(bundle.context["long_entry"].loc[last_date, "B"])
    assert int(last.gt(0.0).sum()) == 1
    assert last["A"] == pytest.approx(1.0)
    assert last["B"] == pytest.approx(0.0)


def test_rrg_sector_rotation_can_disable_name_cap() -> None:
    market = _rrg_market()
    strategy = RrgSectorRotation(
        top_n=1,
        bottom_n=2,
        alpha_mode="fwd_only",
        gross_short=0.0,
        use_name_cap=False,
    )

    weights = strategy.build_weights(market)
    last = weights.iloc[-1]

    assert last["A"] > 0.0
    assert last["B"] > 0.0
    assert int(last.gt(0.0).sum()) == 2


def test_sparse_benchmark_overlay_uses_only_nonzero_signal_names_for_active_book() -> None:
    index = pd.to_datetime(["2024-01-02"])
    columns = ["A", "B", "C", "D"]
    alpha = pd.DataFrame({"A": [1.0], "B": [-1.0], "C": [0.0], "D": [0.0]}, index=index)
    benchmark_weights = pd.DataFrame(0.25, index=index, columns=columns)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"], "C": ["Finance"], "D": ["Finance"]}, index=index)
    bundle = SignalBundle(
        alpha=alpha,
        context={
            "sector": sector,
            "benchmark_weights": benchmark_weights,
            "benchmark_membership": benchmark_weights.gt(0.0),
            "overlay_scale": pd.Series(1.0, index=index),
            "inclusion": alpha.ne(0.0),
        },
    )

    result = _SparseBenchmarkOverlayConstruction(
        active_share_target=0.04,
        max_stock_active=0.03,
        max_sector_active=0.05,
        min_names=1,
    ).build(bundle)

    active = result.base_target_weights.loc[index[0]].sub(benchmark_weights.loc[index[0]])
    assert active["A"] == pytest.approx(0.02)
    assert active["B"] == pytest.approx(-0.02)
    assert active["C"] == pytest.approx(0.0)
    assert active["D"] == pytest.approx(0.0)


def test_rrg_fwd_benchmark_tilt_builds_index_core_with_sparse_active_tilts() -> None:
    strategy = RrgFwdBenchmarkTilt(
        tilt_rule="dual_family",
        active_share_target=0.04,
        max_stock_active=0.01,
        max_sector_active=0.03,
    )

    bundle = strategy.signal_producer.build(_rrg_market())
    weights = strategy.build_weights(_rrg_market())
    last = weights.iloc[-1]
    benchmark_weights = bundle.context["benchmark_weights"].iloc[-1]
    active = last.sub(benchmark_weights.reindex(last.index).fillna(0.0))

    assert last.sum() == pytest.approx(1.0)
    assert last.gt(0.0).sum() == 4
    assert active.abs().sum() > 0.0
    assert active.abs().max() <= 0.011


def test_pure_sector_construction_holds_selected_sectors_cap_weighted() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [0.0], "D": [0.0]}, index=index)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"], "C": ["Finance"], "D": ["Finance"]}, index=index)
    market_cap = pd.DataFrame({"A": [90.0], "B": [10.0], "C": [70.0], "D": [30.0]}, index=index)
    sector_budget = pd.DataFrame({"Tech": [1.0], "Finance": [0.0]}, index=index)
    bundle = SignalBundle(
        alpha=alpha,
        context={
            "sector": sector,
            "sector_weight_basis": market_cap,
            "sector_budget": sector_budget,
            "tradable": pd.DataFrame(True, index=index, columns=alpha.columns),
        },
    )

    result = _RrgPureSectorConstruction().build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == pytest.approx(0.9)
    assert weights["B"] == pytest.approx(0.1)
    assert weights["C"] == pytest.approx(0.0)
    assert weights["D"] == pytest.approx(0.0)
    assert weights.sum() == pytest.approx(1.0)


def test_rrg_pure_sector_rotation_builds_without_forward_frames() -> None:
    market = _rrg_market()
    frames = {
        key: value
        for key, value in market.frames.items()
        if not key.startswith("eps_") and not key.startswith("op_")
    }
    market = MarketData(frames=frames, universe=market.universe, benchmark=market.benchmark)
    strategy = RrgPureSectorRotation(
        selection_rule="leading_improving",
        weighting_rule="equal",
    )

    weights = strategy.build_weights(market)

    assert weights.max().max() > 0.0
    assert weights.min().min() == pytest.approx(0.0)
    assert float(weights.sum(axis=1).max()) <= 1.000001


def test_pure_sector_budget_can_include_weakening_with_state_priority() -> None:
    index = pd.to_datetime(["2024-01-02"])
    rrg_state = pd.DataFrame(
        {
            "Leading": ["Leading"],
            "Improving": ["Improving"],
            "Weakening": ["Weakening"],
            "Lagging": ["Lagging"],
        },
        index=index,
    )
    relative_strength = pd.DataFrame(
        {"Leading": [0.20], "Improving": [-0.05], "Weakening": [0.10], "Lagging": [-0.10]},
        index=index,
    )
    momentum = pd.DataFrame(
        {"Leading": [0.10], "Improving": [0.08], "Weakening": [-0.03], "Lagging": [-0.08]},
        index=index,
    )

    budget = _build_pure_sector_budget(
        rrg_state=rrg_state,
        relative_strength=relative_strength,
        momentum=momentum,
        selection_rule="leading_improving_weakening",
        weighting_rule="state_rank",
    )

    row = budget.loc[index[0]]
    assert row["Leading"] > row["Improving"] > row["Weakening"] > 0.0
    assert row["Lagging"] == pytest.approx(0.0)
    assert row.sum() == pytest.approx(1.0)


def test_pure_sector_budget_can_select_only_resilient_weakening_sectors() -> None:
    index = pd.to_datetime(["2024-01-02"])
    rrg_state = pd.DataFrame(
        {
            "Leading": ["Leading"],
            "Improving": ["Improving"],
            "ResilientWeakening": ["Weakening"],
            "BrokenWeakening": ["Weakening"],
        },
        index=index,
    )
    relative_strength = pd.DataFrame(
        {
            "Leading": [0.20],
            "Improving": [-0.05],
            "ResilientWeakening": [0.08],
            "BrokenWeakening": [0.01],
        },
        index=index,
    )
    momentum = pd.DataFrame(
        {
            "Leading": [0.10],
            "Improving": [0.08],
            "ResilientWeakening": [-0.01],
            "BrokenWeakening": [-0.09],
        },
        index=index,
    )

    budget = _build_pure_sector_budget(
        rrg_state=rrg_state,
        relative_strength=relative_strength,
        momentum=momentum,
        selection_rule="leading_improving_resilient_weakening",
        weighting_rule="score",
    )

    row = budget.loc[index[0]]
    assert row["ResilientWeakening"] > 0.0
    assert row["BrokenWeakening"] == pytest.approx(0.0)
    assert row.sum() == pytest.approx(1.0)


def test_pure_sector_score_weighting_excludes_nonpositive_scores_when_any_positive_score_exists() -> None:
    index = pd.to_datetime(["2024-01-02"])
    rrg_state = pd.DataFrame(
        {
            "Positive": ["Improving"],
            "Negative": ["Improving"],
            "Lagging": ["Lagging"],
        },
        index=index,
    )
    relative_strength = pd.DataFrame(
        {"Positive": [0.03], "Negative": [-0.10], "Lagging": [-0.20]},
        index=index,
    )
    momentum = pd.DataFrame(
        {"Positive": [0.04], "Negative": [0.02], "Lagging": [-0.03]},
        index=index,
    )

    budget = _build_pure_sector_budget(
        rrg_state=rrg_state,
        relative_strength=relative_strength,
        momentum=momentum,
        selection_rule="leading_improving",
        weighting_rule="score",
    )

    row = budget.loc[index[0]]
    assert row["Positive"] == pytest.approx(1.0)
    assert row["Negative"] == pytest.approx(0.0)
    assert row["Lagging"] == pytest.approx(0.0)


def test_pure_sector_score_weighting_falls_back_to_equal_only_when_all_selected_scores_are_nonpositive() -> None:
    index = pd.to_datetime(["2024-01-02"])
    rrg_state = pd.DataFrame(
        {
            "A": ["Improving"],
            "B": ["Improving"],
            "Lagging": ["Lagging"],
        },
        index=index,
    )
    relative_strength = pd.DataFrame(
        {"A": [-0.10], "B": [-0.20], "Lagging": [-0.30]},
        index=index,
    )
    momentum = pd.DataFrame(
        {"A": [0.03], "B": [0.04], "Lagging": [-0.02]},
        index=index,
    )

    budget = _build_pure_sector_budget(
        rrg_state=rrg_state,
        relative_strength=relative_strength,
        momentum=momentum,
        selection_rule="leading_improving",
        weighting_rule="score",
    )

    row = budget.loc[index[0]]
    assert row["A"] == pytest.approx(0.5)
    assert row["B"] == pytest.approx(0.5)
    assert row["Lagging"] == pytest.approx(0.0)


def test_forward_entry_rules_use_different_revision_confirmations() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    columns = ["A", "B", "C"]
    sector = pd.DataFrame("Tech", index=index, columns=columns)
    rrg_state = pd.DataFrame({"Tech": ["Improving", "Improving"]}, index=index)
    frames = {
        "eps_fwd_q1": pd.DataFrame({"A": [10.0, 11.0], "B": [10.0, 11.0], "C": [10.0, 9.0]}, index=index),
        "eps_fwd_q2": pd.DataFrame({"A": [10.0, 11.0], "B": [10.0, 11.0], "C": [10.0, 9.0]}, index=index),
        "eps_fwd": pd.DataFrame({"A": [10.0, 11.0], "B": [10.0, 11.0], "C": [10.0, 9.0]}, index=index),
        "op_fwd_q1": pd.DataFrame({"A": [20.0, 21.0], "B": [20.0, 19.0], "C": [20.0, 19.0]}, index=index),
        "op_fwd_q2": pd.DataFrame({"A": [20.0, 21.0], "B": [20.0, 19.0], "C": [20.0, 19.0]}, index=index),
        "op_fwd": pd.DataFrame({"A": [20.0, 21.0], "B": [20.0, 19.0], "C": [20.0, 19.0]}, index=index),
    }

    state_conditioned = _build_forward_entry_mask(
        frames=frames,
        index=index,
        columns=pd.Index(columns),
        sector=sector,
        rrg_state=rrg_state,
        lookback=1,
        entry_rule="state_conditioned",
    )
    dual_family = _build_forward_entry_mask(
        frames=frames,
        index=index,
        columns=pd.Index(columns),
        sector=sector,
        rrg_state=rrg_state,
        lookback=1,
        entry_rule="dual_family",
    )
    majority_horizons = _build_forward_entry_mask(
        frames=frames,
        index=index,
        columns=pd.Index(columns),
        sector=sector,
        rrg_state=rrg_state,
        lookback=1,
        entry_rule="majority_horizons",
    )

    last_date = index[-1]
    assert bool(state_conditioned.loc[last_date, "A"])
    assert bool(state_conditioned.loc[last_date, "B"])
    assert not bool(state_conditioned.loc[last_date, "C"])
    assert bool(dual_family.loc[last_date, "A"])
    assert not bool(dual_family.loc[last_date, "B"])
    assert bool(majority_horizons.loc[last_date, "A"])
    assert not bool(majority_horizons.loc[last_date, "B"])


def test_state_equal_sector_budget_basis_ignores_sector_market_cap_size() -> None:
    index = pd.to_datetime(["2024-01-02"])
    columns = ["A", "B", "C", "D", "E"]
    sector = pd.DataFrame(
        {
            "A": ["Tech"],
            "B": ["Tech"],
            "C": ["Industrials"],
            "D": ["Materials"],
            "E": ["Materials"],
        },
        index=index,
    )
    membership = pd.DataFrame(True, index=index, columns=columns)
    rrg_state = pd.DataFrame(
        {"Tech": ["Weakening"], "Industrials": ["Improving"], "Materials": ["Leading"]},
        index=index,
    )

    basis = _build_state_equal_sector_weight_basis(
        sector=sector,
        membership=membership,
        rrg_state=rrg_state,
    )

    row_basis = basis.loc[index[0]]
    assert row_basis.loc[["A", "B"]].sum() == pytest.approx(0.0)
    assert row_basis.loc[["C"]].sum() == pytest.approx(0.5)
    assert row_basis.loc[["D", "E"]].sum() == pytest.approx(0.5)


def test_rrg_sector_rotation_builds_signed_weights_from_market_data() -> None:
    market = _rrg_market()
    strategy = RrgSectorRotation(
        top_n=2,
        bottom_n=2,
        lookback=20,
        flow_lookback=20,
        flow_impulse_lookback=5,
        rrg_medium_lookback=126,
        rrg_momentum_lookback=21,
        rrg_short_lookback=42,
    )

    bundle = strategy.signal_producer.build(market)
    plan = strategy.build_plan(market)
    last = plan.target_weights.iloc[-1]

    assert set(bundle.context) == {
        "tradable",
        "long_entry",
        "sector",
        "long_sector",
        "short_sector",
        "sector_weight_basis",
    }
    assert set(bundle.meta) == {
        "rrg_state",
        "fwd_score",
        "fwd_confidence",
        "fwd_coverage",
        "flow_score_20d",
        "flow_score_5d",
    }
    assert bundle.context["long_sector"].loc[market.frames["close"].index[-1], "Tech"]
    assert bundle.context["short_sector"].loc[market.frames["close"].index[-1], "Finance"]
    assert last["A"] > 0.0
    assert last["B"] > 0.0
    assert last["C"] < 0.0
    assert last["D"] < 0.0
    assert last.clip(lower=0.0).sum() == pytest.approx(1.0)
    assert (-last.clip(upper=0.0)).sum() == pytest.approx(1.0)
    assert last.sum() == pytest.approx(0.0)


def test_rrg_sector_rotation_has_no_sector_legs_before_rrg_warmup() -> None:
    market = _rrg_market()
    strategy = RrgSectorRotation(
        top_n=2,
        bottom_n=2,
        lookback=20,
        flow_lookback=20,
        flow_impulse_lookback=5,
        rrg_medium_lookback=126,
        rrg_momentum_lookback=21,
        rrg_short_lookback=42,
    )

    bundle = strategy.signal_producer.build(market)
    first_date = market.frames["close"].index[0]

    assert not bool(bundle.context["long_sector"].loc[first_date].any())
    assert not bool(bundle.context["short_sector"].loc[first_date].any())
    assert set(bundle.meta["rrg_state"].loc[first_date]) == {"Unclassified"}


def test_rrg_sector_rotation_emits_negative_weights_for_short_leg() -> None:
    strategy = RrgSectorRotation(top_n=2, bottom_n=2)

    weights = strategy.build_weights(_rrg_market())

    assert weights.min().min() < 0.0
    assert weights.max().max() > 0.0


def _rrg_market() -> MarketData:
    index = pd.date_range("2023-01-02", periods=220, freq="B")
    columns = ["A", "B", "C", "D"]
    accel = [max(i - 180, 0) for i in range(len(index))]
    close = pd.DataFrame(index=index, columns=columns, dtype=float)
    close["A"] = [100.0 + i * 0.55 + accel[i] * 0.18 for i in range(len(index))]
    close["B"] = [98.0 + i * 0.45 + accel[i] * 0.15 for i in range(len(index))]
    close["C"] = [120.0 - i * 0.25 - accel[i] * 0.10 for i in range(len(index))]
    close["D"] = [118.0 - i * 0.35 - accel[i] * 0.13 for i in range(len(index))]

    benchmark = pd.DataFrame({"IKS200": [100.0 + i * 0.10 for i in range(len(index))]}, index=index)
    k200 = pd.DataFrame(True, index=index, columns=columns)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"], "C": ["Finance"], "D": ["Finance"]}, index=index).ffill()
    market_cap = pd.DataFrame({"A": [65.0], "B": [35.0], "C": [60.0], "D": [40.0]}, index=index).ffill()
    volume = pd.DataFrame(1_000_000.0, index=index, columns=columns)

    eps_q1 = pd.DataFrame(index=index, columns=columns, dtype=float)
    eps_q1["A"] = [10.0 + i * 0.06 for i in range(len(index))]
    eps_q1["B"] = [9.0 + i * 0.05 for i in range(len(index))]
    eps_q1["C"] = [9.5 - i * 0.03 for i in range(len(index))]
    eps_q1["D"] = [10.0 - i * 0.04 for i in range(len(index))]
    eps_q2 = eps_q1 + 1.0
    eps_y1 = eps_q1 + 2.0
    op_q1 = eps_q1 * 1.8
    op_q2 = eps_q2 * 1.8
    op_y1 = eps_y1 * 1.8

    foreign = pd.DataFrame({"A": [900_000.0], "B": [700_000.0], "C": [-600_000.0], "D": [-800_000.0]}, index=index).ffill()
    inst = pd.DataFrame({"A": [700_000.0], "B": [500_000.0], "C": [-500_000.0], "D": [-700_000.0]}, index=index).ffill()
    retail = pd.DataFrame({"A": [-400_000.0], "B": [-300_000.0], "C": [450_000.0], "D": [550_000.0]}, index=index).ffill()

    return MarketData(
        frames={
            "close": close,
            "benchmark": benchmark,
            "k200_yn": k200,
            "sector_big": sector,
            "market_cap": market_cap,
            "volume": volume,
            "eps_fwd_q1": eps_q1,
            "eps_fwd_q2": eps_q2,
            "eps_fwd": eps_y1,
            "op_fwd_q1": op_q1,
            "op_fwd_q2": op_q2,
            "op_fwd": op_y1,
            "foreign_flow": foreign,
            "inst_flow": inst,
            "retail_flow": retail,
        },
        universe=k200,
        benchmark=None,
    )
