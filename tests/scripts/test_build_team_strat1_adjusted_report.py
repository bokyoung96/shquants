import pandas as pd
import pytest

from scripts.build_team_strat1_adjusted_report import deduplicate_events, summarize_periods


def test_adjusted_event_returns_exit_on_fixed_signal_high_touch_after_entry() -> None:
    index = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    )
    symbol = "A000001"
    mask = pd.DataFrame({symbol: [False, True, False, False, False, False, False]}, index=index)
    open_ = pd.DataFrame({symbol: [100.0, 100.0, 104.0, 98.0, 97.0, 96.0, 95.0]}, index=index)
    high = pd.DataFrame({symbol: [101.0, 110.0, 109.0, 108.0, 109.0, 111.0, 101.0]}, index=index)
    close = pd.DataFrame({symbol: [100.0, 102.0, 100.0, 96.0, 94.0, 93.0, 92.0]}, index=index)

    events = deduplicate_events(mask, open_, high, close)

    event = events.iloc[0]
    assert bool(event["gap_over_signal_high"]) is False
    assert bool(event["prior_high_touch_T1"]) is False
    assert bool(event["prior_high_touch_T2"]) is False
    assert bool(event["prior_high_touch_T3"]) is False
    assert bool(event["prior_high_touch_T4"]) is True
    assert event["adjusted_exit_date_T4"] == index[5]
    assert event["adjusted_exit_price_T4"] == pytest.approx(110.0)
    assert event["first_prior_high_touch_price"] == pytest.approx(110.0)
    assert event["T+4"] == pytest.approx((104.0 - 93.0) / 104.0)
    assert event["adjusted_T+4"] == pytest.approx((104.0 - 110.0) / 104.0)


def test_adjusted_event_returns_zero_when_entry_open_above_signal_high() -> None:
    index = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    )
    symbol = "A000001"
    mask = pd.DataFrame({symbol: [False, True, False, False, False, False, False]}, index=index)
    open_ = pd.DataFrame({symbol: [100.0, 100.0, 112.0, 98.0, 97.0, 96.0, 95.0]}, index=index)
    high = pd.DataFrame({symbol: [101.0, 110.0, 113.0, 108.0, 109.0, 105.0, 101.0]}, index=index)
    close = pd.DataFrame({symbol: [100.0, 102.0, 100.0, 96.0, 94.0, 93.0, 92.0]}, index=index)

    events = deduplicate_events(mask, open_, high, close)

    event = events.iloc[0]
    assert bool(event["gap_over_signal_high"]) is True
    assert event["entry_status"] == "not_entered"
    for horizon in range(1, 6):
        assert event[f"adjusted_T+{horizon}"] == 0.0


def test_adjusted_event_returns_exit_at_gap_open_after_entry() -> None:
    index = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    )
    symbol = "A000001"
    mask = pd.DataFrame({symbol: [False, True, False, False, False, False, False]}, index=index)
    open_ = pd.DataFrame({symbol: [100.0, 100.0, 104.0, 112.0, 97.0, 96.0, 95.0]}, index=index)
    high = pd.DataFrame({symbol: [101.0, 110.0, 109.0, 113.0, 109.0, 105.0, 101.0]}, index=index)
    close = pd.DataFrame({symbol: [100.0, 102.0, 100.0, 111.0, 94.0, 93.0, 92.0]}, index=index)

    events = deduplicate_events(mask, open_, high, close)

    event = events.iloc[0]
    assert bool(event["gap_over_signal_high"]) is False
    assert bool(event["prior_high_touch_T1"]) is False
    assert bool(event["prior_high_touch_T2"]) is True
    assert event["adjusted_exit_date_T2"] == index[3]
    assert event["adjusted_exit_price_T2"] == pytest.approx(112.0)
    assert event["adjusted_exit_reason_T2"] == "gap_open_stop"
    assert event["adjusted_T+2"] == pytest.approx((104.0 - 112.0) / 104.0)


def test_summary_keeps_baseline_returns_for_not_entered_events() -> None:
    events = pd.DataFrame(
        [
            {
                "signal_date": pd.Timestamp("2024-01-03"),
                "symbol": "A000001",
                "gap_over_signal_high": False,
                "first_prior_high_touch_date": pd.Timestamp("2024-01-05"),
                **{f"T+{horizon}": 0.10 for horizon in range(1, 6)},
                **{f"adjusted_T+{horizon}": 0.20 for horizon in range(1, 6)},
            },
            {
                "signal_date": pd.Timestamp("2024-01-04"),
                "symbol": "A000002",
                "gap_over_signal_high": True,
                "first_prior_high_touch_date": pd.NaT,
                **{f"T+{horizon}": 0.50 for horizon in range(1, 6)},
                **{f"adjusted_T+{horizon}": 0.0 for horizon in range(1, 6)},
            },
        ]
    )

    summary = summarize_periods(events, pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-10"))

    row = summary.iloc[0]
    assert row["events"] == 2
    assert row["not_entered_events"] == 1
    assert row["T+1_return"] == pytest.approx(0.30)
    assert row["T+1_adjusted_return"] == pytest.approx(0.10)
    assert row["T+1_active_return"] == pytest.approx(0.10)
