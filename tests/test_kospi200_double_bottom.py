import numpy as np
import pandas as pd
import pytest

from scripts.kospi200_double_bottom import (
    DoubleBottomConfig,
    aggregate_daily_ohlc,
    build_fills,
    detect_double_bottoms,
    find_pivot_lows,
    select_non_overlapping_signals,
    summarize_equity,
)


def test_aggregate_daily_ohlc_uses_seoul_date_and_ohlc_extremes() -> None:
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2024-01-01 23:55", "2024-01-02 00:05", "2024-01-02 00:10"],
                utc=True,
            ),
            "open": [10.0, 11.0, 12.0],
            "high": [12.0, 14.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [11.0, 13.0, 12.5],
        }
    )

    daily = aggregate_daily_ohlc(frame)

    assert list(daily.index.strftime("%Y-%m-%d")) == ["2024-01-02"]
    assert daily.iloc[0].to_dict() == {
        "open": 10.0,
        "high": 14.0,
        "low": 9.0,
        "close": 12.5,
    }


def test_pivot_low_is_not_usable_until_two_following_sessions_exist() -> None:
    index = pd.date_range("2024-01-01", periods=7, freq="B")
    daily = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": [9, 8, 7, 8, 9, 10, 11], "close": 10.0},
        index=index,
    )

    pivots = find_pivot_lows(daily, window=2)

    assert bool(pivots.loc[index[2], "is_pivot_low"])
    assert pivots.loc[index[2], "confirmed_at"] == index[4]
    assert not bool(pivots.iloc[-1]["is_pivot_low"])


def test_double_bottom_filters_and_next_open_entry() -> None:
    index = pd.date_range("2024-01-01", periods=36, freq="B")
    low = np.linspace(110.0, 120.0, 36)
    high = low + 5.0
    close = low + 2.0
    low[5], high[5], close[5] = 100.0, 102.0, 101.0
    high[10:20] = 135.0
    low[20], high[20], close[20] = 101.5, 104.0, 102.0
    close[25] = 136.0
    open_ = np.full(36, 103.0)
    open_[26], open_[31], open_[36 - 1] = 137.0, 140.0, 145.0
    daily = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=index
    )

    signals = detect_double_bottoms(daily, DoubleBottomConfig())

    assert len(signals) == 1
    assert signals.iloc[0]["first_pivot_date"] == index[5]
    assert signals.iloc[0]["second_pivot_date"] == index[20]
    assert signals.iloc[0]["breakout_date"] == index[25]
    assert signals.iloc[0]["entry_date"] == index[26]
    assert signals.iloc[0]["entry_price"] == pytest.approx(137.0)


def test_staged_fills_use_open_offsets_and_do_not_backfill_missing_fills() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    daily = pd.DataFrame({"open": np.arange(100.0, 108.0), "close": 100.0}, index=index)

    fills = build_fills(
        daily, entry_pos=2, weights=(0.5, 0.25, 0.25), offsets=(0, 2, 10)
    )

    assert list(fills["fill_date"]) == [index[2], index[4]]
    assert list(fills["weight"]) == [0.5, 0.25]
    assert list(fills["price"]) == [102.0, 104.0]


def test_overlap_selection_and_equity_summary() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    signals = pd.DataFrame({"entry_pos": [1, 2, 4], "exit_pos": [3, 4, 5]})

    selected = select_non_overlapping_signals(signals)
    assert list(selected["entry_pos"]) == [1, 4]

    equity = pd.Series([1.0, 1.1, 1.05, 1.2, 1.0], index=index)
    summary = summarize_equity(equity, trade_returns=[0.2, -1 / 6])
    assert summary["total_return"] == pytest.approx(0.0)
    assert summary["max_drawdown"] == pytest.approx(-1 / 6)
    assert summary["trade_count"] == 2
