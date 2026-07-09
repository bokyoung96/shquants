from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_winner_holding_regime_comparison import (
    WinnerRegimeConfig,
    compare_regimes,
    simulate_exit_regime,
)


def test_breakeven_regime_moves_stop_after_1r_reached() -> None:
    entries = pd.DataFrame(
        {
            "ticker": ["A"],
            "date": pd.to_datetime(["2024-01-02"]),
            "signal_time": pd.to_datetime(["2024-01-02 09:30"]),
            "entry_time": pd.to_datetime(["2024-01-02 09:40"]),
            "entry_price": [100.0],
            "atr": [5.0],
            "signal_score": [1.0],
        }
    )
    daily = pd.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "daily_high": [102.0, 106.0, 101.0],
            "daily_low": [99.0, 101.0, 99.0],
            "close": [102.0, 104.0, 97.0],
            "prior_52w_close_high": [98.0, 98.0, 98.0],
        }
    )

    baseline = simulate_exit_regime(entries, daily, WinnerRegimeConfig(round_trip_cost_bps=0.0), regime="baseline")
    breakeven = simulate_exit_regime(entries, daily, WinnerRegimeConfig(round_trip_cost_bps=0.0), regime="breakeven_1r")

    assert baseline.iloc[0]["exit_reason"] == "new_high_lost"
    assert baseline.iloc[0]["exit_price"] == pytest.approx(97.0)
    assert breakeven.iloc[0]["exit_reason"] == "breakeven_stop"
    assert breakeven.iloc[0]["exit_price"] == pytest.approx(100.0)


def test_breakeven_regime_matches_baseline_before_1r_reached() -> None:
    entries = pd.DataFrame(
        {
            "ticker": ["A"],
            "date": pd.to_datetime(["2024-01-02"]),
            "signal_time": pd.to_datetime(["2024-01-02 09:30"]),
            "entry_time": pd.to_datetime(["2024-01-02 09:40"]),
            "entry_price": [100.0],
            "atr": [5.0],
            "signal_score": [1.0],
        }
    )
    daily = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "daily_high": [102.0, 103.0],
            "daily_low": [99.0, 94.0],
            "close": [102.0, 97.0],
            "prior_52w_close_high": [98.0, 98.0],
        }
    )

    baseline = simulate_exit_regime(entries, daily, WinnerRegimeConfig(round_trip_cost_bps=0.0), regime="baseline")
    breakeven = simulate_exit_regime(entries, daily, WinnerRegimeConfig(round_trip_cost_bps=0.0), regime="breakeven_1r")

    assert baseline.iloc[0]["exit_reason"] == "atr_stop"
    assert breakeven.iloc[0]["exit_reason"] == "atr_stop"
    assert baseline.iloc[0]["exit_price"] == breakeven.iloc[0]["exit_price"] == 95.0


def test_compare_regimes_returns_both_variants() -> None:
    entries = pd.DataFrame(
        {
            "ticker": ["A"],
            "date": pd.to_datetime(["2024-01-02"]),
            "signal_time": pd.to_datetime(["2024-01-02 09:30"]),
            "entry_time": pd.to_datetime(["2024-01-02 09:40"]),
            "entry_price": [100.0],
            "atr": [5.0],
            "signal_score": [1.0],
        }
    )
    daily = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "daily_high": [102.0, 103.0],
            "daily_low": [99.0, 94.0],
            "close": [102.0, 97.0],
            "prior_52w_close_high": [98.0, 98.0],
        }
    )

    result = compare_regimes(entries, daily, WinnerRegimeConfig(round_trip_cost_bps=0.0))

    assert set(result) == {"baseline", "breakeven_1r"}
    assert len(result["baseline"]) == 1
    assert len(result["breakeven_1r"]) == 1
