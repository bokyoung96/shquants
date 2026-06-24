from __future__ import annotations

import pandas as pd
import pytest

from backtesting.strategies.positivity_event import (
    EventQueueConfig,
    build_positivity_event_queue_strategy,
    true_range_atr,
)


def test_true_range_atr_uses_high_low_and_previous_close() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    high = pd.DataFrame({"A": [10.0, 13.0, 12.0]}, index=idx)
    low = pd.DataFrame({"A": [9.0, 11.0, 9.0]}, index=idx)
    close = pd.DataFrame({"A": [9.5, 12.0, 10.0]}, index=idx)

    atr = true_range_atr(high=high, low=low, close=close, lookback=2)

    assert pd.isna(atr.loc[idx[0], "A"])
    assert atr.loc[idx[1], "A"] == pytest.approx(2.25)
    assert atr.loc[idx[2], "A"] == pytest.approx(3.25)


def test_event_queue_enters_near_high_and_exits_on_atr_stop() -> None:
    idx = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    close = pd.DataFrame({"A": [100.0, 102.0, 104.0, 103.0, 96.0]}, index=idx)
    high = close.add(1.0)
    low = close.sub(1.0)
    membership = pd.DataFrame(True, index=idx, columns=close.columns)

    result = build_positivity_event_queue_strategy(
        close=close,
        high=high,
        low=low,
        membership=membership,
        config=EventQueueConfig(
            max_positions=1,
            positivity_lookback=2,
            min_periods=2,
            high_lookback=2,
            atr_lookback=2,
            atr_multiplier=2.0,
            relative_signal_groups=1,
            entry_high_ratio=0.95,
            exit_high_ratio=0.90,
        ),
    )

    assert result.weights.loc[idx[2], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[3], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[4], "A"] == pytest.approx(0.0)
    assert result.trades["exit_reason"].tolist() == ["atr_stop"]


def test_event_queue_replaces_weakest_active_name_when_score_margin_is_met() -> None:
    idx = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 101.0, 103.0, 103.0, 103.0],
            "B": [100.0, 101.0, 100.0, 103.0, 104.0],
        },
        index=idx,
    )
    high = close.add(1.0)
    low = close.sub(1.0)
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    bonus = pd.DataFrame(0.0, index=idx, columns=close.columns)
    bonus.loc[idx[3], "B"] = 5.0

    result = build_positivity_event_queue_strategy(
        close=close,
        high=high,
        low=low,
        membership=membership,
        score_bonus=bonus,
        config=EventQueueConfig(
            max_positions=1,
            positivity_lookback=2,
            min_periods=2,
            high_lookback=2,
            atr_lookback=2,
            atr_multiplier=3.0,
            relative_signal_groups=1,
            replacement_margin=1.0,
        ),
    )

    assert result.weights.loc[idx[2], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[3], "A"] == pytest.approx(0.0)
    assert result.weights.loc[idx[3], "B"] == pytest.approx(1.0)
    assert result.trades["exit_reason"].tolist() == ["replacement"]
