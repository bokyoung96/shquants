from __future__ import annotations

import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from scripts.verified_flow_backtest import (
    audit_backtest,
    fixed_notional_mtm_ledger,
    position_slots,
    rebalanced_mtm_ledger,
    plot_verified_ledgers,
    select_fixed_slot_trades,
    same_ticker_overlap_violations,
    write_audit_outputs,
    write_trade_return_distribution,
)


def test_position_slots_uses_intraday_entry_exit_order() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "entry_time": pd.to_datetime(["2024-01-02 09:05", "2024-01-02 09:10"]),
            "exit_time": pd.to_datetime(["2024-01-02 15:20", "2024-01-02 15:30"]),
        }
    )

    assert position_slots(trades) == 2


def test_same_ticker_overlap_detector_flags_overlapping_positions() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "entry_time": pd.to_datetime(["2024-01-02 09:05", "2024-01-02 10:00", "2024-01-02 09:05"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30", "2024-01-02 15:30", "2024-01-02 15:30"]),
        }
    )

    assert same_ticker_overlap_violations(trades) == 1


def test_fixed_notional_ledger_marks_open_positions_and_persists_realized_pnl() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A"],
            "entry_time": pd.to_datetime(["2024-01-02 09:05"]),
            "exit_time": pd.to_datetime(["2024-01-04 15:30"]),
            "entry_price": [100.0],
            "net_return": [0.10],
        }
    )
    close = pd.DataFrame({"A": [95.0, 90.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

    ledger, missing = fixed_notional_mtm_ledger(trades, close, slots=1)

    assert missing == 0
    assert ledger.loc[pd.Timestamp("2024-01-02"), "equity"] == pytest.approx(0.95)
    assert ledger.loc[pd.Timestamp("2024-01-03"), "equity"] == pytest.approx(0.90)
    assert ledger.loc[pd.Timestamp("2024-01-04"), "equity"] == pytest.approx(1.10)


def test_rebalanced_ledger_preserves_trade_level_exit_return() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A"],
            "entry_time": pd.to_datetime(["2024-01-02 09:05"]),
            "exit_time": pd.to_datetime(["2024-01-04 15:30"]),
            "entry_price": [100.0],
            "net_return": [0.10],
        }
    )
    close = pd.DataFrame({"A": [95.0, 90.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

    ledger, missing = rebalanced_mtm_ledger(trades, close, slots=1)

    assert missing == 0
    assert ledger["equity"].iloc[-1] == pytest.approx(1.10)


def test_audit_backtest_reports_core_metrics() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "entry_time": pd.to_datetime(["2024-01-02 09:05", "2024-01-02 09:10"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30", "2024-01-03 15:30"]),
            "entry_price": [100.0, 100.0],
            "net_return": [0.10, -0.05],
        }
    )
    close = pd.DataFrame(
        {"A": [105.0, 110.0], "B": [98.0, 95.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    audit, fixed, rebalanced = audit_backtest(trades, close)

    assert audit.trades == 2
    assert audit.position_slots == 2
    assert audit.same_ticker_overlap_violations == 0
    assert audit.raw_trade_return_sum == pytest.approx(0.05)
    assert not fixed.empty
    assert not rebalanced.empty


def test_select_fixed_slot_trades_skips_when_full_and_reuses_slots_after_exit() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "entry_time": pd.to_datetime(
                [
                    "2024-01-02 09:05",
                    "2024-01-02 09:10",
                    "2024-01-02 09:15",
                    "2024-01-03 09:05",
                ]
            ),
            "exit_time": pd.to_datetime(
                [
                    "2024-01-03 15:30",
                    "2024-01-02 15:30",
                    "2024-01-03 15:30",
                    "2024-01-04 15:30",
                ]
            ),
            "net_return": [0.01, 0.02, 0.03, 0.04],
        }
    )

    selected, skipped = select_fixed_slot_trades(trades, max_positions=2)

    assert selected["ticker"].tolist() == ["A", "B", "D"]
    assert skipped["ticker"].tolist() == ["C"]
    assert selected["portfolio_skip_reason"].isna().all()
    assert skipped["portfolio_skip_reason"].tolist() == ["max_positions"]


def test_write_trade_return_distribution_creates_png(tmp_path) -> None:
    trades = pd.DataFrame(
        {
            "net_return": [-0.02, -0.01, 0.0, 0.03, 0.10],
            "exit_reason": ["atr_stop", "atr_stop", "new_high_lost", "new_high_lost", "new_high_lost"],
        }
    )

    write_trade_return_distribution(trades, tmp_path / "distribution.png", title="Distribution")

    assert (tmp_path / "distribution.png").stat().st_size > 0


def test_trade_return_distribution_uses_diagnostic_panels_without_cdf(tmp_path, monkeypatch) -> None:
    trades = pd.DataFrame(
        {
            "net_return": [-0.081, -0.043, -0.011, -0.006, -0.003, 0.002, 0.019, 0.074, 0.28],
            "exit_reason": [
                "atr_stop",
                "atr_stop",
                "new_high_lost",
                "new_high_lost",
                "new_high_lost",
                "time_exit",
                "time_exit",
                "time_exit",
                "time_exit",
            ],
        }
    )
    saved: list[Figure] = []

    def capture_savefig(self, *_args, **_kwargs) -> None:
        saved.append(self)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    write_trade_return_distribution(trades, tmp_path / "diagnostic_distribution.png", title="Distribution")

    assert len(saved) == 1
    titles = [axis.get_title(loc="left") for axis in saved[0].axes]
    assert not any("CDF" in title for title in titles)
    assert any("Central return shape" in title for title in titles)
    assert any("Loss cluster" in title for title in titles)
    assert any("Right-tail winners" in title for title in titles)
    assert any("Exit reason profile" in title for title in titles)
    plt.close("all")


def test_verified_ledger_plot_uses_strategy_dashboard_panels(tmp_path, monkeypatch) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-03-01"])
    fixed = pd.DataFrame(
        {
            "equity": [1.0, 1.03, 1.01, 1.08],
            "drawdown": [0.0, 0.0, -0.0194, 0.0],
            "active_positions": [1, 2, 1, 0],
        },
        index=dates,
    )
    rebalanced = pd.DataFrame(
        {
            "equity": [1.0, 1.04, 1.02, 1.11],
            "drawdown": [0.0, 0.0, -0.0192, 0.0],
            "active_positions": [1, 2, 1, 0],
        },
        index=dates,
    )
    saved: list[Figure] = []

    def capture_savefig(self, *_args, **_kwargs) -> None:
        saved.append(self)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    plot_verified_ledgers(fixed, rebalanced, tmp_path / "curves.png")

    assert len(saved) == 1
    titles = [axis.get_title(loc="left") for axis in saved[0].axes]
    assert "Cumulative return path" in titles
    assert "Drawdown pressure" in titles
    assert "Exposure / active positions" in titles
    assert "Monthly return tape" in titles
    assert "Yearly scorecard" in titles
    assert "Portfolio snapshot" in titles
    plt.close("all")


def test_write_audit_outputs_groups_files_by_result_type(tmp_path) -> None:
    trades_path = tmp_path / "intraday_trades.csv"
    close_path = tmp_path / "close.parquet"
    output_dir = tmp_path / "result"
    trades = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "signal_time": pd.to_datetime(["2024-01-02 09:00", "2024-01-02 09:05"]),
            "entry_time": pd.to_datetime(["2024-01-02 09:10", "2024-01-02 09:15"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30", "2024-01-03 15:30"]),
            "entry_price": [100.0, 50.0],
            "net_return": [0.02, -0.01],
            "exit_reason": ["test", "test"],
        }
    )
    close = pd.DataFrame(
        {"A": [100.0, 102.0], "B": [50.0, 49.5]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    trades.to_csv(trades_path, index=False)
    close.to_parquet(close_path)

    write_audit_outputs(trades_path=trades_path, close_path=close_path, output_dir=output_dir, slots=20)

    assert (output_dir / "verified" / "backtest_report.md").exists()
    assert (output_dir / "verified" / "backtest_curves.png").exists()
    assert (output_dir / "fixed20" / "report.md").exists()
    assert (output_dir / "fixed20" / "selected_trades.csv").exists()
    assert (output_dir / "distributions" / "all_trade_return_distribution.png").exists()
    assert (output_dir / "distributions" / "fixed20_selected_return_distribution.png").exists()
    assert not (output_dir / "verified_backtest_report.md").exists()
