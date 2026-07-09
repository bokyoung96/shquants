from __future__ import annotations

import pandas as pd
import pytest

from backtesting.strategies.fifty_two_week_breakout_atr import (
    BreakoutAtrConfig,
    confirmed_breakout_entries,
    run_breakout_atr_strategy,
    simulate_atr_continuation,
)


def test_confirmed_breakout_enters_after_next_close_confirmation() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A"] * 4,
            "date": pd.to_datetime(["2024-01-02"] * 4),
            "ts": pd.to_datetime(
                [
                    "2024-01-02 09:25",
                    "2024-01-02 09:30",
                    "2024-01-02 09:35",
                    "2024-01-02 09:40",
                ]
            ),
            "hhmm": ["0925", "0930", "0935", "0940"],
            "close": [100.0, 101.2, 101.4, 101.1],
            "previous_intraday_close": [99.5, 100.0, 101.2, 101.4],
            "next_ts": pd.to_datetime(
                [
                    "2024-01-02 09:30",
                    "2024-01-02 09:35",
                    "2024-01-02 09:40",
                    "2024-01-02 09:45",
                ]
            ),
            "next_open": [101.0, 101.3, 101.5, 101.0],
            "atr": [2.0, 2.0, 2.0, 2.0],
            "prior_52w_close_high": [100.5, 100.5, 100.5, 100.5],
        }
    )
    config = BreakoutAtrConfig(range_end_hhmm="0920", exit_hhmm="1455", range_buffer_bps=0.0)

    entries = confirmed_breakout_entries(frame, config)

    assert entries[["ticker", "signal_time", "entry_time", "entry_price"]].to_dict("records") == [
        {
            "ticker": "A",
            "signal_time": pd.Timestamp("2024-01-02 09:30"),
            "entry_time": pd.Timestamp("2024-01-02 09:40"),
            "entry_price": 101.5,
        }
    ]


def test_confirmed_breakout_rejects_failed_next_close_confirmation() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "ts": pd.to_datetime(["2024-01-02 09:25", "2024-01-02 09:30", "2024-01-02 09:35"]),
            "hhmm": ["0925", "0930", "0935"],
            "close": [101.0, 99.9, 101.2],
            "previous_intraday_close": [100.0, 101.0, 99.9],
            "next_ts": pd.to_datetime(["2024-01-02 09:30", "2024-01-02 09:35", "2024-01-02 09:40"]),
            "next_open": [101.0, 101.1, 101.2],
            "atr": [1.0, 1.0, 1.0],
            "prior_52w_close_high": [100.5, 100.5, 100.5],
        }
    )

    entries = confirmed_breakout_entries(frame, BreakoutAtrConfig(range_end_hhmm="0920", range_buffer_bps=0.0))

    assert entries.empty


def test_simulate_atr_continuation_exits_at_stop_price_after_min_holding_days() -> None:
    entries = pd.DataFrame(
        {
            "ticker": ["A"],
            "date": pd.to_datetime(["2024-01-02"]),
            "signal_time": pd.to_datetime(["2024-01-02 09:30"]),
            "entry_time": pd.to_datetime(["2024-01-02 09:40"]),
            "entry_price": [100.0],
            "atr": [3.0],
            "signal_score": [1.0],
        }
    )
    daily = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [105.0, 99.0],
            "daily_low": [99.0, 96.5],
            "prior_52w_close_high": [98.0, 98.0],
        }
    )

    trades = simulate_atr_continuation(
        entries,
        daily,
        BreakoutAtrConfig(atr_stop_multiplier=1.0, min_holding_days=1, round_trip_cost_bps=0.0),
    )

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "atr_stop"
    assert trade["exit_price"] == pytest.approx(97.0)
    assert trade["gross_return"] == pytest.approx(-0.03)


def test_run_breakout_atr_strategy_combines_entries_and_daily_exit() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A"] * 4,
            "date": pd.to_datetime(["2024-01-02"] * 4),
            "ts": pd.to_datetime(["2024-01-02 09:25", "2024-01-02 09:30", "2024-01-02 09:35", "2024-01-02 09:40"]),
            "hhmm": ["0925", "0930", "0935", "0940"],
            "close": [100.0, 101.2, 101.4, 101.1],
            "previous_intraday_close": [99.5, 100.0, 101.2, 101.4],
            "next_ts": pd.to_datetime(["2024-01-02 09:30", "2024-01-02 09:35", "2024-01-02 09:40", "2024-01-02 09:45"]),
            "next_open": [101.0, 101.3, 101.5, 101.0],
            "atr": [2.0, 2.0, 2.0, 2.0],
            "prior_52w_close_high": [100.5, 100.5, 100.5, 100.5],
        }
    )
    daily = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [101.0, 102.0],
            "daily_low": [100.0, 101.0],
            "prior_52w_close_high": [100.5, 100.5],
        }
    )

    result = run_breakout_atr_strategy(frame, daily, BreakoutAtrConfig(range_buffer_bps=0.0, round_trip_cost_bps=0.0))

    assert len(result.entries) == 1
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "end_of_data"
