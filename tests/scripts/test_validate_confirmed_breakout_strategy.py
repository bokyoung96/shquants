from __future__ import annotations

import pandas as pd

from scripts.validate_confirmed_breakout_strategy import (
    audit_return_accounting,
    audit_trade_log_integrity,
    plot_entry_exit_case,
)


def test_audit_return_accounting_recomputes_net_return_and_fixed_pnl() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "entry_time": pd.to_datetime(["2024-01-02 09:10", "2024-01-03 09:10"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30", "2024-01-04 15:30"]),
            "entry_price": [100.0, 200.0],
            "exit_price": [110.0, 190.0],
            "gross_return": [0.10, -0.05],
            "net_return": [0.0965, -0.0535],
        }
    )
    ledger = pd.DataFrame({"equity": [1.0, 1.00215]}, index=pd.to_datetime(["2024-01-03", "2024-01-04"]))

    audit = audit_return_accounting(trades, ledger, slots=20)

    assert audit["gross_return_mismatches"] == 0
    assert audit["net_return_mismatches"] == 0
    assert audit["fixed_notional_final_return_from_trades"] == 0.00215
    assert audit["fixed_notional_final_return_delta"] == 0.0


def test_audit_trade_log_integrity_flags_bad_temporal_order() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "signal_time": pd.to_datetime(["2024-01-02 09:30", "2024-01-03 09:30"]),
            "entry_time": pd.to_datetime(["2024-01-02 09:35", "2024-01-03 09:25"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30", "2024-01-02 15:30"]),
            "entry_price": [100.0, 100.0],
            "exit_price": [101.0, 99.0],
            "net_return": [0.0065, -0.0135],
        }
    )

    audit = audit_trade_log_integrity(trades)

    assert audit["entry_before_or_at_signal_violations"] == 1
    assert audit["exit_before_entry_violations"] == 1


def test_plot_entry_exit_case_creates_png(tmp_path) -> None:
    trade = pd.Series(
        {
            "ticker": "A000001",
            "signal_time": pd.Timestamp("2024-01-02 09:30"),
            "entry_time": pd.Timestamp("2024-01-02 09:40"),
            "exit_time": pd.Timestamp("2024-01-03 15:30"),
            "entry_price": 102.0,
            "exit_price": 110.0,
            "gross_return": 0.0784313725,
            "net_return": 0.0760313725,
            "exit_reason": "new_high_lost",
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "close": [99.0, 112.0, 110.0],
            "daily_low": [97.0, 100.0, 109.0],
            "prior_52w_close_high": [100.0, 100.0, 111.0],
        }
    )
    entry_bars = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-02 09:30", "2024-01-02 09:35", "2024-01-02 09:40"]),
            "open": [99.0, 101.0, 102.0],
            "high": [101.0, 103.0, 104.0],
            "low": [98.0, 100.0, 101.0],
            "close": [101.0, 103.0, 103.5],
            "volume": [1, 2, 3],
        }
    )
    exit_bars = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-03 15:20", "2024-01-03 15:25", "2024-01-03 15:30"]),
            "open": [111.0, 110.5, 110.2],
            "high": [112.0, 111.0, 110.5],
            "low": [110.0, 109.8, 109.5],
            "close": [110.5, 110.2, 110.0],
            "volume": [1, 2, 3],
        }
    )

    plot_entry_exit_case(trade, daily, entry_bars, exit_bars, tmp_path / "case.png")

    assert (tmp_path / "case.png").stat().st_size > 0
