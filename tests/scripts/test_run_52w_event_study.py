from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_52w_event_study import compute_event_forward_returns, compute_event_path_returns, summarize_horizons


def test_compute_event_forward_returns_excludes_event_ticker_from_benchmark() -> None:
    events = pd.DataFrame(
        {
            "ticker": ["A"],
            "signal_time": pd.to_datetime(["2024-01-02 10:00"]),
            "entry_time": pd.to_datetime(["2024-01-02 10:10"]),
            "entry_price": [105.0],
        }
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 110.0, 120.0],
            "B": [200.0, 220.0, 210.0],
            "C": [50.0, 45.0, 55.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    result = compute_event_forward_returns(events, close, horizons=(1, 2), round_trip_cost_bps=35.0)

    row = result.iloc[0]
    assert row["event_close_return_1d"] == pytest.approx(0.10)
    assert row["event_entry_return_1d"] == pytest.approx(110.0 / 105.0 - 1.0)
    assert row["event_entry_net_return_1d"] == pytest.approx(110.0 / 105.0 - 1.0 - 0.0035)
    assert row["benchmark_return_1d"] == pytest.approx(((220.0 / 200.0 - 1.0) + (45.0 / 50.0 - 1.0)) / 2.0)
    assert row["excess_return_1d"] == pytest.approx(row["event_close_return_1d"] - row["benchmark_return_1d"])
    assert row["event_close_return_2d"] == pytest.approx(0.20)


def test_compute_event_forward_returns_drops_missing_future_horizon() -> None:
    events = pd.DataFrame(
        {
            "ticker": ["A"],
            "signal_time": pd.to_datetime(["2024-01-03 10:00"]),
            "entry_time": pd.to_datetime(["2024-01-03 10:10"]),
            "entry_price": [110.0],
        }
    )
    close = pd.DataFrame({"A": [100.0, 110.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

    result = compute_event_forward_returns(events, close, horizons=(1,), round_trip_cost_bps=35.0)

    assert pd.isna(result.iloc[0]["event_close_return_1d"])
    assert pd.isna(result.iloc[0]["benchmark_return_1d"])


def test_summarize_horizons_reports_excess_statistics() -> None:
    events = pd.DataFrame(
        {
            "event_entry_net_return_1d": [0.01, -0.02, 0.03],
            "event_close_return_1d": [0.02, -0.01, 0.04],
            "benchmark_return_1d": [0.01, 0.00, 0.01],
            "excess_return_1d": [0.01, -0.01, 0.03],
        }
    )

    summary = summarize_horizons(events, horizons=(1,))

    row = summary.iloc[0]
    assert row["horizon_days"] == 1
    assert row["events"] == 3
    assert row["entry_net_mean"] == pytest.approx((0.01 - 0.02 + 0.03) / 3.0)
    assert row["excess_mean"] == pytest.approx(0.01)
    assert row["excess_hit_rate"] == pytest.approx(2.0 / 3.0)


def test_compute_event_path_returns_starts_from_confirmed_entry_and_excludes_event_ticker() -> None:
    events = pd.DataFrame(
        {
            "ticker": ["A"],
            "signal_time": pd.to_datetime(["2024-01-02 10:00"]),
            "entry_time": pd.to_datetime(["2024-01-02 10:10"]),
            "entry_price": [105.0],
        }
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 110.0, 120.0],
            "B": [200.0, 220.0, 210.0],
            "C": [50.0, 45.0, 55.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    result = compute_event_path_returns(events, close, max_horizon=2)

    assert list(result["event_day"]) == [0, 1, 2]
    assert result.loc[result["event_day"].eq(0), "signal_entry_mean"].iloc[0] == pytest.approx(0.0)
    assert result.loc[result["event_day"].eq(1), "signal_entry_mean"].iloc[0] == pytest.approx(110.0 / 105.0 - 1.0)
    assert result.loc[result["event_day"].eq(2), "signal_entry_mean"].iloc[0] == pytest.approx(120.0 / 105.0 - 1.0)
    assert result.loc[result["event_day"].eq(1), "benchmark_mean"].iloc[0] == pytest.approx(
        ((220.0 / 200.0 - 1.0) + (45.0 / 50.0 - 1.0)) / 2.0
    )
