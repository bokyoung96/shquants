from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_exit_regime_comparison_experiment import ExitRegime, simulate_exit_regime_trade
from scripts.run_tech_gamma_long_only import TechGammaConfig


def _config() -> TechGammaConfig:
    return TechGammaConfig(start="2024-01-01", min_holding_days=1, atr_stop_multiplier=1.0)


def _signal() -> pd.Series:
    return pd.Series(
        {
            "ticker": "A001",
            "date": pd.Timestamp("2024-01-02"),
            "ts": pd.Timestamp("2024-01-02 10:00"),
            "next_ts": pd.Timestamp("2024-01-02 10:05"),
            "next_open": 100.0,
            "atr": 5.0,
            "signal_score": 3.0,
        }
    )


def _daily(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["date", "open", "close", "daily_low", "prior_52w_close_high", "positivity_spread"],
    ).assign(ticker="A001", date=lambda frame: pd.to_datetime(frame["date"]))


def test_current_regime_exits_when_intraday_low_touches_atr_stop() -> None:
    daily = _daily(
        [
            ("2024-01-02", 100.0, 100.0, 99.0, 90.0, 0.08),
            ("2024-01-03", 101.0, 101.0, 94.0, 90.0, 0.07),
        ]
    )

    trade = simulate_exit_regime_trade(_signal(), daily, ExitRegime.current(), _config())

    assert trade is not None
    assert trade["exit_time"] == pd.Timestamp("2024-01-03 15:30")
    assert trade["exit_price"] == 95.0
    assert trade["exit_reason"] == "atr_stop"
    assert trade["gross_return"] == pytest.approx(-0.05)


def test_atr_close_confirmed_ignores_wick_touch_until_close_breaks_stop() -> None:
    daily = _daily(
        [
            ("2024-01-02", 100.0, 100.0, 99.0, 90.0, 0.08),
            ("2024-01-03", 101.0, 101.0, 94.0, 90.0, 0.07),
            ("2024-01-04", 96.0, 94.0, 93.0, 90.0, 0.07),
        ]
    )

    trade = simulate_exit_regime_trade(_signal(), daily, ExitRegime.atr_close_confirmed(), _config())

    assert trade is not None
    assert trade["exit_time"] == pd.Timestamp("2024-01-04 15:30")
    assert trade["exit_price"] == 94.0
    assert trade["exit_reason"] == "atr_close_stop"


def test_positivity_relaxed_atr_uses_close_stop_while_supportive_then_touch_stop_when_weak() -> None:
    daily = _daily(
        [
            ("2024-01-02", 100.0, 100.0, 99.0, 90.0, 0.08),
            ("2024-01-03", 101.0, 101.0, 94.0, 90.0, 0.07),
            ("2024-01-04", 96.0, 101.0, 94.0, 90.0, 0.03),
        ]
    )

    trade = simulate_exit_regime_trade(_signal(), daily, ExitRegime.positivity_relaxed_atr(), _config())

    assert trade is not None
    assert trade["exit_time"] == pd.Timestamp("2024-01-04 15:30")
    assert trade["exit_price"] == 95.0
    assert trade["exit_reason"] == "positivity_weak_atr_stop"


def test_positivity_relative_decay_exits_at_same_day_open_before_intraday_exit_checks() -> None:
    daily = _daily(
        [
            ("2024-01-02", 100.0, 100.0, 99.0, 90.0, 0.08),
            ("2024-01-03", 98.0, 101.0, 94.0, 90.0, 0.03),
        ]
    )

    trade = simulate_exit_regime_trade(_signal(), daily, ExitRegime.positivity_relative_open_exit(), _config())

    assert trade is not None
    assert trade["exit_time"] == pd.Timestamp("2024-01-03 09:00")
    assert trade["exit_price"] == 98.0
    assert trade["exit_reason"] == "positivity_relative_decay_open"
