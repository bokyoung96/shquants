from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_adr_regime_positivity_experiment import (
    apply_variant_flags,
    compute_adr_regime,
    summarize_event_variants,
)


def test_compute_adr_regime_uses_prior_days_only() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    close = pd.DataFrame(
        {
            "A": [100, 101, 102, 103, 104, 103],
            "B": [100, 99, 98, 99, 100, 101],
            "C": [100, 101, 102, 101, 102, 103],
            "D": [100, 99, 98, 97, 96, 97],
        },
        index=dates,
        dtype=float,
    )

    result = compute_adr_regime(close, lookback=2, threshold=1.0, min_periods=2)

    day = pd.Timestamp("2024-01-05")
    prior_adr = result.loc[pd.Timestamp("2024-01-04"), "adr"]
    older_adr = result.loc[pd.Timestamp("2024-01-03"), "adr"]
    assert result.loc[day, "adr20"] == pytest.approx((older_adr + prior_adr) / 2.0)
    assert result.loc[day, "adr20"] != pytest.approx(
        (prior_adr + result.loc[day, "adr"]) / 2.0
    )


def test_apply_variant_flags_keeps_all_events_in_broad_regime_and_filters_narrow_regime() -> None:
    events = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "event_date": pd.to_datetime(["2024-01-05"] * 4),
            "positivity_spread": [0.03, -0.01, 0.01, -0.02],
            "adr_regime": ["broad", "broad", "narrow", "narrow"],
        }
    )

    result = apply_variant_flags(events, positivity_margin=0.02)

    assert result["keep_baseline"].tolist() == [True, True, True, True]
    assert result["keep_always_pos_gt0"].tolist() == [True, False, True, False]
    assert result["keep_always_pos_margin"].tolist() == [True, False, False, False]
    assert result["keep_adr_pos_gt0"].tolist() == [True, True, True, False]
    assert result["keep_adr_pos_margin"].tolist() == [True, True, False, False]


def test_summarize_event_variants_reports_winner_capture_and_forward_metrics() -> None:
    events = pd.DataFrame(
        {
            "event_entry_return_20d": [0.50, 0.20, -0.10, 0.05],
            "event_entry_net_return_20d": [0.4965, 0.1965, -0.1035, 0.0465],
            "excess_return_20d": [0.30, 0.10, -0.20, 0.00],
            "keep_baseline": [True, True, True, True],
            "keep_adr_pos_gt0": [True, False, False, True],
        }
    )

    summary = summarize_event_variants(
        events,
        variants={"baseline": "keep_baseline", "adr_pos_gt0": "keep_adr_pos_gt0"},
        top_winner_fraction=0.25,
        horizon=20,
    )

    baseline = summary.loc[summary["variant"].eq("baseline")].iloc[0]
    filtered = summary.loc[summary["variant"].eq("adr_pos_gt0")].iloc[0]
    assert baseline["events"] == 4
    assert baseline["top_winner_capture"] == pytest.approx(1.0)
    assert filtered["events"] == 2
    assert filtered["top_winner_capture"] == pytest.approx(1.0)
    assert filtered["entry_net_mean_20d"] == pytest.approx((0.4965 + 0.0465) / 2.0)
