from __future__ import annotations

import pandas as pd

from scripts.run_multi_timeframe_filter_experiment import (
    apply_mtf_filter,
    attach_completed_weekly_features,
    daily_volatility_compression,
)


def test_attach_completed_weekly_features_uses_prior_completed_week_only() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A001", "A001"],
            "signal_time": pd.to_datetime(["2024-01-08 10:00", "2024-01-12 10:00"]),
            "entry_time": pd.to_datetime(["2024-01-08 10:05", "2024-01-12 10:05"]),
        }
    )
    weekly = pd.DataFrame(
        {
            "ticker": ["A001", "A001"],
            "week_date": pd.to_datetime(["2024-01-05", "2024-01-12"]),
            "weekly_market_rs_ok": [True, False],
            "weekly_sector_rs_ok": [False, True],
        }
    )

    result = attach_completed_weekly_features(trades, weekly)

    assert result["weekly_market_rs_ok"].tolist() == [True, True]
    assert result["weekly_sector_rs_ok"].tolist() == [False, False]


def test_daily_volatility_compression_uses_only_completed_prior_days() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    smooth = pd.Series(range(len(dates)), index=dates, dtype=float).mul(0.1).add(100.0)
    noisy_recent = smooth.copy()
    noisy_recent.iloc[-1] = noisy_recent.iloc[-2] * 1.2
    close = pd.DataFrame({"A001": smooth, "A002": noisy_recent})

    result = daily_volatility_compression(close, short_window=20, long_window=60)

    assert bool(result.loc[dates[-1], "A001"])
    assert bool(result.loc[dates[-1], "A002"])
    assert not bool(result.loc[dates[-1] + pd.offsets.BDay(1), "A002"])


def test_apply_mtf_filter_keeps_only_requested_true_rows() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["A001", "A002", "A003"],
            "signal_time": pd.to_datetime(["2024-01-08 10:00"] * 3),
            "weekly_sector_rs_ok": [True, False, True],
            "daily_vol_compression_ok": [True, True, False],
        }
    )

    sector_only = apply_mtf_filter(trades, ["weekly_sector_rs_ok"])
    combined = apply_mtf_filter(trades, ["weekly_sector_rs_ok", "daily_vol_compression_ok"])

    assert sector_only["ticker"].tolist() == ["A001", "A003"]
    assert combined["ticker"].tolist() == ["A001"]
