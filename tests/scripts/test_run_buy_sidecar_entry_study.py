from __future__ import annotations

import pandas as pd
import pytest
from matplotlib import image as mpimg

from scripts.run_buy_sidecar_entry_study import (
    build_candidate_rules,
    build_execution_sensitivity,
    build_holding_regime_summary,
    plot_study,
    select_robust_entry_window,
    summarize_candidates,
)


def test_build_candidate_rules_covers_entry_release_grid_and_close_variants() -> None:
    rules = build_candidate_rules(entry_delays=(0, 1), release_delays=(0, 3))

    assert [rule.name for rule in rules] == [
        "a0_r0",
        "a0_r3",
        "a0_close",
        "a1_r0",
        "a1_r3",
        "a1_close",
    ]


def test_execution_sensitivity_respects_end_labeled_minute_bars() -> None:
    pairs = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2026-01-02").date(),
                "activation_dt": pd.Timestamp("2026-01-02 09:06:20"),
                "release_dt": pd.Timestamp("2026-01-02 09:11:20"),
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"dt": "2026-01-02 09:10:00", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0},
            {"dt": "2026-01-02 09:11:00", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.0},
            {"dt": "2026-01-02 09:15:00", "open": 103.0, "high": 105.0, "low": 102.0, "close": 104.0},
            {"dt": "2026-01-02 09:16:00", "open": 104.5, "high": 106.0, "low": 104.0, "close": 105.0},
        ]
    )

    sensitivity = build_execution_sensitivity(
        pairs,
        prices,
        entry_delays=(3,),
        release_delays=(3,),
    )

    complete_close = sensitivity[sensitivity["fill_method"].eq("first_complete_close")].iloc[0]
    boundary_open = sensitivity[sensitivity["fill_method"].eq("first_boundary_open")].iloc[0]
    bar_worst = sensitivity[sensitivity["fill_method"].eq("containing_bar_worst")].iloc[0]
    assert complete_close["entry_bar_dt"] == pd.Timestamp("2026-01-02 09:10:00")
    assert complete_close["exit_bar_dt"] == pd.Timestamp("2026-01-02 09:15:00")
    assert complete_close["ret"] == pytest.approx(104.0 / 100.0 - 1.0)
    assert boundary_open["entry_bar_dt"] == pd.Timestamp("2026-01-02 09:11:00")
    assert boundary_open["exit_bar_dt"] == pd.Timestamp("2026-01-02 09:16:00")
    assert boundary_open["ret"] == pytest.approx(104.5 / 100.5 - 1.0)
    assert bar_worst["ret"] == pytest.approx(102.0 / 101.0 - 1.0)


def test_summarize_candidates_reports_cost_and_leave_one_out_stability() -> None:
    trades = pd.DataFrame(
        [
            {"rule": "a3_r3", "trade_date": "2026-01-02", "ret": 0.01},
            {"rule": "a3_r3", "trade_date": "2026-02-02", "ret": 0.02},
            {"rule": "a3_r3", "trade_date": "2026-03-02", "ret": -0.005},
        ]
    )

    summary = summarize_candidates(
        trades,
        cost_bps=(10,),
        bootstrap_samples=500,
        seed=7,
    ).iloc[0]

    assert summary["mean_ret"] == pytest.approx(0.025 / 3.0)
    assert summary["net_10bps_mean_ret"] == pytest.approx(0.025 / 3.0 - 0.001)
    assert summary["net_10bps_win_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["loo_mean_min"] == pytest.approx((0.01 - 0.005) / 2.0)
    assert summary["loo_mean_max"] == pytest.approx((0.01 + 0.02) / 2.0)
    assert summary["bootstrap_mean_ci_low"] <= summary["mean_ret"]
    assert summary["bootstrap_mean_ci_high"] >= summary["mean_ret"]


def test_select_robust_entry_window_requires_execution_and_cost_stability() -> None:
    candidate_summary = pd.DataFrame(
        [
            {"entry_delay_m": 0, "exit_delay_m": 3, "exit_kind": "release", "win_rate": 0.80, "bootstrap_mean_ci_low": 0.001, "net_10bps_mean_ret": 0.003},
            {"entry_delay_m": 1, "exit_delay_m": 3, "exit_kind": "release", "win_rate": 0.93, "bootstrap_mean_ci_low": 0.002, "net_10bps_mean_ret": 0.004},
            {"entry_delay_m": 2, "exit_delay_m": 3, "exit_kind": "release", "win_rate": 1.00, "bootstrap_mean_ci_low": 0.003, "net_10bps_mean_ret": 0.005},
            {"entry_delay_m": 3, "exit_delay_m": 3, "exit_kind": "release", "win_rate": 1.00, "bootstrap_mean_ci_low": 0.003, "net_10bps_mean_ret": 0.005},
            {"entry_delay_m": 4, "exit_delay_m": 3, "exit_kind": "release", "win_rate": 1.00, "bootstrap_mean_ci_low": 0.002, "net_10bps_mean_ret": 0.004},
        ]
    )
    execution_summary = pd.DataFrame(
        [
            {"entry_delay_m": delay, "exit_delay_m": 3, "fill_method": method, "mean_ret": mean}
            for delay, means in {
                0: (0.002, -0.001),
                1: (0.004, 0.001),
                2: (0.005, 0.002),
                3: (0.005, 0.002),
                4: (0.004, 0.001),
            }.items()
            for method, mean in zip(("first_boundary_open", "containing_bar_worst"), means)
        ]
    )

    window = select_robust_entry_window(candidate_summary, execution_summary, exit_delay_minutes=3)

    assert window["entry_delay_m"].tolist() == [1, 2, 3, 4]


def test_holding_regime_summary_uses_regime_names_as_summary_rules() -> None:
    pairs = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2026-01-02").date(),
                "activation_dt": pd.Timestamp("2026-01-02 09:06:20"),
                "release_dt": pd.Timestamp("2026-01-02 09:11:20"),
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"dt": "2026-01-02 09:11:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10},
            {"dt": "2026-01-02 15:30:00", "open": 109.0, "high": 111.0, "low": 108.0, "close": 110.0, "volume": 10},
            {"dt": "2026-01-05 09:01:00", "open": 108.0, "high": 109.0, "low": 107.0, "close": 108.5, "volume": 10},
            {"dt": "2026-01-05 15:30:00", "open": 104.0, "high": 106.0, "low": 103.0, "close": 105.0, "volume": 10},
        ]
    )

    summary = build_holding_regime_summary(pairs, prices, entry_delay_minutes=3)

    assert set(summary["rule"]) == {
        "entry_to_close",
        "close_to_next_open",
        "close_to_next_close",
        "entry_to_next_close",
    }


def test_plot_study_writes_nonblank_event_journey_canvas(tmp_path) -> None:
    entry_rows = [
        {
            "rule": f"a{delay}_r3",
            "entry_delay_m": delay,
            "exit_kind": "release",
            "exit_delay_m": 3,
            "mean_ret": mean,
            "median_ret": mean - 0.0005,
            "win_rate": win_rate,
            "min_ret": min_ret,
            "net_10bps_mean_ret": mean - 0.001,
            "bootstrap_mean_ci_low": mean - 0.002,
            "bootstrap_mean_ci_high": mean + 0.002,
        }
        for delay, mean, win_rate, min_ret in [
            (0, 0.00815, 0.93, -0.00356),
            (1, 0.00861, 0.93, -0.00214),
            (2, 0.00901, 0.93, -0.00038),
            (3, 0.00872, 1.00, 0.00043),
            (4, 0.00862, 1.00, 0.00262),
            (5, 0.00250, 0.73, -0.00432),
        ]
    ]
    exit_rows = [
        {
            "rule": f"a3_r{delay}",
            "entry_delay_m": 3,
            "exit_kind": "release",
            "exit_delay_m": delay,
            "mean_ret": mean,
            "median_ret": mean,
            "win_rate": 1.0 if delay in (1, 2, 3) else 0.87,
            "min_ret": -0.001,
            "net_10bps_mean_ret": mean - 0.001,
            "bootstrap_mean_ci_low": mean - 0.002,
            "bootstrap_mean_ci_high": mean + 0.002,
        }
        for delay, mean in enumerate([0.00621, 0.00676, 0.00841, 0.00872, 0.00735, 0.00820])
        if delay != 3
    ]
    canonical = pd.DataFrame([*entry_rows, *exit_rows])
    holding = pd.DataFrame(
        [
            {"rule": "entry_to_close", "mean_ret": 0.02439, "win_rate": 0.87, "min_ret": -0.04974},
            {"rule": "close_to_next_close", "mean_ret": -0.01234, "win_rate": 0.36, "min_ret": -0.10106},
        ]
    )

    output = plot_study(tmp_path, canonical, pd.DataFrame(), holding)

    pixels = mpimg.imread(output)
    height, width = pixels.shape[:2]
    assert width >= 1600
    assert height >= 900
    assert 1.6 <= width / height <= 1.9
    assert float(pixels.std()) > 0.02
