from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_52w_5m_breakout_atr_report import (
    build_report,
    compute_metrics,
    central_return_window,
    profit_factor,
    yearly_returns,
)


def test_profit_factor_sums_positive_returns_over_absolute_losses() -> None:
    returns = pd.Series([0.10, -0.04, 0.02, -0.01])

    assert profit_factor(returns) == 2.4


def test_compute_metrics_uses_fixed_notional_ledger_and_trade_distribution() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "entry_time": pd.to_datetime(["2024-01-02 09:40", "2024-01-03 09:40"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30", "2024-01-04 15:30"]),
            "net_return": [0.10, -0.02],
            "exit_reason": ["new_high_lost", "atr_stop"],
        }
    )
    ledger = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "equity": [1.00, 1.04, 1.08],
            "drawdown": [0.0, -0.01, 0.0],
            "active_positions": [1, 2, 0],
        }
    ).set_index("date")

    metrics = compute_metrics(trades, ledger)

    assert metrics["trades"] == 2
    assert metrics["final_return"] == pytest.approx(0.08)
    assert metrics["mdd"] == -0.01
    assert metrics["avg_trade_return"] == 0.04
    assert metrics["hit_rate"] == 0.5
    assert metrics["profit_factor"] == 5.0
    assert metrics["max_active_positions"] == 2


def test_yearly_returns_uses_exit_year_fixed_slot_contribution() -> None:
    trades = pd.DataFrame(
        {
            "exit_time": pd.to_datetime(["2024-01-03", "2024-02-01", "2025-01-02"]),
            "net_return": [0.10, -0.02, 0.04],
        }
    )

    result = yearly_returns(trades, slots=20)

    assert result[["year", "trades", "year_return"]].to_dict("records") == [
        {"year": 2024, "trades": 2, "year_return": 0.004},
        {"year": 2025, "trades": 1, "year_return": 0.002},
    ]


def test_central_return_window_clips_extreme_tails_for_readable_histogram() -> None:
    returns = pd.Series([-1000.0, -50.0, 0.0, 50.0, 10000.0])

    low, high = central_return_window(returns, lower_q=0.2, upper_q=0.8)

    assert low == pytest.approx(-50.0)
    assert high == pytest.approx(50.0)


def test_build_report_writes_markdown_and_png(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    trades = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "signal_time": pd.to_datetime(["2024-01-02 09:30", "2024-01-03 09:30"]),
            "entry_time": pd.to_datetime(["2024-01-02 09:40", "2024-01-03 09:40"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30", "2024-01-04 15:30"]),
            "entry_price": [100.0, 100.0],
            "exit_price": [110.0, 98.0],
            "net_return": [0.10, -0.02],
            "exit_reason": ["new_high_lost", "atr_stop"],
        }
    )
    ledger = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "equity": [1.00, 1.04, 1.08],
            "drawdown": [0.0, -0.01, 0.0],
            "active_positions": [1, 2, 0],
        }
    )
    trades.to_csv(source / "selected_trades.csv", index=False)
    ledger.to_csv(source / "fixed_notional_ledger.csv", index=False)

    output = tmp_path / "report"
    build_report(source, output)

    text = (output / "report.md").read_text(encoding="utf-8")
    assert "52W High 5M Breakout + ATR Strategy" in text
    assert "No positivity filter" in text
    assert "No foreign/institution flow filter" in text
    assert (output / "performance.png").exists()
