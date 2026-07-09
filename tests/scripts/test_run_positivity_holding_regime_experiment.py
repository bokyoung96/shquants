from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_positivity_holding_regime_experiment import (
    PositivityExitRule,
    apply_positivity_exit_overlay,
)


def _trade(*, exit_time: pd.Timestamp = pd.Timestamp("2024-01-08 15:30")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A001"],
            "side": ["long"],
            "signal_time": [pd.Timestamp("2024-01-02 10:00")],
            "entry_time": [pd.Timestamp("2024-01-02 10:10")],
            "exit_time": [exit_time],
            "entry_price": [100.0],
            "exit_price": [112.0],
            "signal_score": [1.0],
            "gross_return": [0.12],
            "net_return": [0.1165],
            "exit_reason": ["new_high_lost"],
        }
    )


def test_absolute_spread_exit_uses_next_trading_day_open() -> None:
    daily = pd.DataFrame(
        {
            "ticker": ["A001"] * 4,
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "open": [100.0, 103.0, 101.0, 99.0],
            "positivity_spread": [0.05, -0.01, 0.03, 0.04],
        }
    )

    result = apply_positivity_exit_overlay(_trade(), daily, PositivityExitRule.absolute_nonpositive())

    assert result.iloc[0]["exit_time"] == pd.Timestamp("2024-01-04 09:00")
    assert result.iloc[0]["exit_price"] == 101.0
    assert result.iloc[0]["exit_reason"] == "positivity_absolute_nonpositive"
    assert result.iloc[0]["gross_return"] == pytest.approx(0.01)


def test_relative_decay_exit_uses_entry_day_spread_as_anchor() -> None:
    daily = pd.DataFrame(
        {
            "ticker": ["A001"] * 5,
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]),
            "open": [100.0, 103.0, 104.0, 105.0, 106.0],
            "positivity_spread": [0.08, 0.05, 0.035, 0.06, 0.07],
        }
    )

    result = apply_positivity_exit_overlay(_trade(), daily, PositivityExitRule.relative_decay(0.5))

    assert result.iloc[0]["exit_time"] == pd.Timestamp("2024-01-05 09:00")
    assert result.iloc[0]["exit_price"] == 105.0
    assert result.iloc[0]["exit_reason"] == "positivity_relative_decay_50"


def test_consecutive_weakness_requires_two_completed_weak_days() -> None:
    daily = pd.DataFrame(
        {
            "ticker": ["A001"] * 6,
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]),
            "open": [100.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "positivity_spread": [0.08, 0.019, 0.021, 0.015, 0.014, 0.03],
        }
    )

    trade = _trade(exit_time=pd.Timestamp("2024-01-10 15:30"))

    result = apply_positivity_exit_overlay(trade, daily, PositivityExitRule.consecutive_weakness(threshold=0.02, days=2))

    assert result.iloc[0]["exit_time"] == pd.Timestamp("2024-01-09 09:00")
    assert result.iloc[0]["exit_price"] == 107.0
    assert result.iloc[0]["exit_reason"] == "positivity_weak_2d_le_002"


def test_overlay_does_not_extend_original_exit() -> None:
    daily = pd.DataFrame(
        {
            "ticker": ["A001"] * 4,
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "open": [100.0, 103.0, 104.0, 105.0],
            "positivity_spread": [0.08, 0.07, 0.06, -0.01],
        }
    )

    result = apply_positivity_exit_overlay(_trade(), daily, PositivityExitRule.absolute_nonpositive())

    assert result.iloc[0]["exit_time"] == pd.Timestamp("2024-01-08 15:30")
    assert result.iloc[0]["exit_reason"] == "new_high_lost"
