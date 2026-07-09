from __future__ import annotations

import pandas as pd

from scripts.run_positivity_utilization_experiment import (
    attach_positivity_features,
    select_priority_fixed_slot_trades,
)


def test_attach_positivity_features_joins_signal_date_and_ticker() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A001", "A002"],
            "signal_time": pd.to_datetime(["2024-01-03 10:00", "2024-01-03 10:00"]),
            "entry_time": pd.to_datetime(["2024-01-03 10:05", "2024-01-03 10:05"]),
        }
    )
    candidates = pd.DataFrame(
        {
            "ticker": ["A001", "A002"],
            "date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "positivity_spread": [0.03, 0.09],
            "daily_positivity": [0.54, 0.60],
            "positivity_benchmark": [0.51, 0.51],
        }
    )

    result = attach_positivity_features(trades, candidates)

    assert result["positivity_spread"].tolist() == [0.03, 0.09]
    assert result["positivity_rank_pct"].tolist() == [0.5, 1.0]


def test_priority_selection_prefers_higher_positivity_when_slots_are_full() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["LOW", "HIGH"],
            "signal_time": pd.to_datetime(["2024-01-03 10:00", "2024-01-03 10:00"]),
            "entry_time": pd.to_datetime(["2024-01-03 10:05", "2024-01-03 10:05"]),
            "exit_time": pd.to_datetime(["2024-01-05 15:30", "2024-01-05 15:30"]),
            "positivity_rank_pct": [0.2, 0.9],
            "net_return": [0.01, 0.02],
        }
    )

    selected, skipped = select_priority_fixed_slot_trades(trades, max_positions=1, priority_column="positivity_rank_pct")

    assert selected["ticker"].tolist() == ["HIGH"]
    assert skipped["ticker"].tolist() == ["LOW"]
    assert skipped.iloc[0]["portfolio_skip_reason"] == "max_positions"
