from __future__ import annotations

import pandas as pd
import pytest

from backtesting.data.loader import MarketData
from backtesting.strategies.rrg_sector_rotation import (
    RrgSectorRotation,
    _RrgSectorRotationSignal,
    _bounded_delta,
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
