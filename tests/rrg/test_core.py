from __future__ import annotations

import math

import pandas as pd
import pytest

from rrg.core import (
    HorizonSpec,
    RrgConfig,
    classify_rrg_state,
    classify_turning_point,
    compute_horizon_rrg,
    compute_multi_horizon_rrg,
)


def test_compute_horizon_rrg_uses_log_relative_strength_derivatives() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    sector_prices = pd.DataFrame({"Tech": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=index)
    benchmark = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=index)

    result = compute_horizon_rrg(
        sector_prices=sector_prices,
        benchmark=benchmark,
        horizon=HorizonSpec(name="short", periods=1),
        config=RrgConfig(trend_window=1, acceleration_z_window=2),
    )

    row = result[(result["date"].eq(index[2])) & (result["sector"].eq("Tech"))].iloc[0]
    lrs_t = math.log(121.0 / 100.0)
    lrs_t_1 = math.log(110.0 / 100.0)
    lrs_t_2 = math.log(100.0 / 100.0)

    assert row["rs"] == pytest.approx(1.21)
    assert row["log_rs"] == pytest.approx(lrs_t)
    assert row["mom"] == pytest.approx(lrs_t - lrs_t_1)
    assert row["acc"] == pytest.approx(lrs_t - 2.0 * lrs_t_1 + lrs_t_2)


def test_state_and_turning_labels_cover_rrg_quadrants_and_acceleration() -> None:
    assert classify_rrg_state(0.10, 0.20) == "Leading"
    assert classify_rrg_state(-0.10, 0.20) == "Improving"
    assert classify_rrg_state(-0.10, -0.20) == "Lagging"
    assert classify_rrg_state(0.10, -0.20) == "Weakening"
    assert classify_rrg_state(float("nan"), 0.20) == "Unclassified"

    assert classify_turning_point(mom=0.20, acc_z=-1.5, threshold=1.0) == "Exhaustion risk"
    assert classify_turning_point(mom=-0.20, acc_z=1.5, threshold=1.0) == "Recovery candidate"
    assert classify_turning_point(mom=0.20, acc_z=1.5, threshold=1.0) == "Trend strengthening"
    assert classify_turning_point(mom=-0.20, acc_z=-1.5, threshold=1.0) == "Breakdown pressure"
    assert classify_turning_point(mom=0.20, acc_z=0.2, threshold=1.0) == "Neutral"


def test_compute_multi_horizon_rrg_returns_tidy_rows_with_persistence() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    sector_prices = pd.DataFrame(
        {
            "Tech": [100.0, 101.0, 103.0, 106.0, 110.0, 115.0, 121.0, 128.0],
            "Energy": [100.0, 99.0, 97.0, 94.0, 90.0, 85.0, 79.0, 72.0],
        },
        index=index,
    )
    benchmark = pd.Series([100.0] * len(index), index=index)

    result = compute_multi_horizon_rrg(
        sector_prices=sector_prices,
        benchmark=benchmark,
        config=RrgConfig(
            horizons=(HorizonSpec("short", 1), HorizonSpec("medium", 2)),
            trend_window=1,
            acceleration_z_window=3,
            turning_threshold=0.5,
        ),
    )

    assert {
        "date",
        "sector",
        "horizon",
        "rs",
        "log_rs",
        "rs_centered",
        "mom",
        "acc",
        "acc_z",
        "state",
        "turning_label",
        "persistence",
        "confidence",
    }.issubset(result.columns)
    assert set(result["horizon"].dropna()) == {"short", "medium"}

    tech_short = result[(result["sector"].eq("Tech")) & (result["horizon"].eq("short"))].dropna(subset=["state"])
    latest = tech_short.iloc[-1]
    assert latest["state"] == "Leading"
    assert int(latest["persistence"]) >= 2
