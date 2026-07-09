from __future__ import annotations

import pandas as pd

from scripts.run_slot_priority_mtf_breakout import (
    default_variants,
    mtf_tier,
    select_mtf_priority_fixed_slot_trades,
    selected_key_difference_count,
    slot_selection_audit,
)


def test_mtf_tier_is_simple_boolean_priority() -> None:
    trades = pd.DataFrame(
        {
            "weekly_sector_rs_ok": [True, True, False, False],
            "daily_vol_compression_ok": [True, False, True, False],
        }
    )

    assert mtf_tier(trades).tolist() == [1, 2, 3, 4]


def test_mtf_tier_priority_prefers_sector_and_vol_without_scores() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["T4", "T2", "T3", "T1"],
            "signal_time": pd.to_datetime(["2024-01-03 10:00"] * 4),
            "entry_time": pd.to_datetime(["2024-01-03 10:05"] * 4),
            "exit_time": pd.to_datetime(["2024-01-05 15:30"] * 4),
            "weekly_sector_rs_ok": [False, True, False, True],
            "daily_vol_compression_ok": [False, False, True, True],
            "net_return": [0.04, 0.02, 0.03, 0.01],
        }
    )

    selected, skipped = select_mtf_priority_fixed_slot_trades(trades, max_positions=2)

    assert selected["ticker"].tolist() == ["T1", "T2"]
    assert skipped["ticker"].tolist() == ["T3", "T4"]
    assert skipped["portfolio_skip_reason"].tolist() == ["max_positions", "max_positions"]


def test_selection_audit_uses_accounting_slots_separate_from_position_cap() -> None:
    close = pd.DataFrame({"A": [100.0, 110.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    trades = pd.DataFrame(
        {
            "ticker": ["A"],
            "signal_time": pd.to_datetime(["2024-01-02 10:00"]),
            "entry_time": pd.to_datetime(["2024-01-02 10:05"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30"]),
            "entry_price": [100.0],
            "net_return": [0.10],
        }
    )

    audit, selected, _skipped, fixed, _rebalanced = slot_selection_audit(
        trades,
        close,
        max_positions=15,
        accounting_slots=20,
        priority=False,
    )

    assert audit.slot_weight == 0.05
    assert selected["ticker"].tolist() == ["A"]
    assert fixed["equity"].iloc[-1] == 1.005


def test_default_variants_keep_anti_overfit_comparison_set() -> None:
    assert [variant.name for variant in default_variants()] == [
        "5m_only_max20",
        "flow_confirmed_max20",
        "flow_confirmed_max15",
        "slot_priority_mtf_max15",
        "hard_filter_weekly_sector_daily_vol",
    ]


def test_selected_key_difference_count_counts_replacements_once() -> None:
    left = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "signal_time": pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:05"]),
            "entry_time": pd.to_datetime(["2024-01-02 10:05", "2024-01-02 10:10"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30", "2024-01-03 15:30"]),
        }
    )
    right = pd.DataFrame(
        {
            "ticker": ["A", "C"],
            "signal_time": pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:15"]),
            "entry_time": pd.to_datetime(["2024-01-02 10:05", "2024-01-02 10:20"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30", "2024-01-03 15:30"]),
        }
    )

    assert selected_key_difference_count(left, right) == 1
