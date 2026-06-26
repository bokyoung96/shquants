from __future__ import annotations

import pandas as pd
import pytest

from etc.next_day_down_event_study import (
    build_iks200_daily_frame,
    build_intraday_next_day_paths,
    build_next_day_reactions,
    mark_down_day_events,
    summarize_intraday_paths,
    summarize_overall,
    summarize_yearly,
)


def _qw_bm() -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([["IKS200"], ["open", "high", "low", "close"]], names=["code", "field"])
    return pd.DataFrame(
        [
            [100.0, 101.0, 99.0, 100.0],
            [99.0, 100.0, 96.0, 97.0],
            [98.0, 101.0, 95.0, 100.0],
            [100.0, 101.0, 98.0, 99.0],
            [98.0, 99.0, 96.0, 96.03],
            [97.0, 99.0, 96.0, 98.0],
        ],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2025-02-03", "2025-02-04", "2025-02-05"]),
        columns=columns,
    )


def test_build_iks200_daily_frame_extracts_qw_bm_multiindex_ohlc() -> None:
    daily = build_iks200_daily_frame(_qw_bm())

    assert daily.index.name == "date"
    assert daily.loc[pd.Timestamp("2024-01-02"), "close"] == 97.0
    assert daily.loc[pd.Timestamp("2024-01-02"), "ret_cc"] == pytest.approx(-0.03)
    assert daily.loc[pd.Timestamp("2025-02-04"), "ret_cc"] == pytest.approx(-0.03)


def test_mark_down_day_events_assigns_one_non_overlapping_down_bucket_without_short_logic() -> None:
    daily = build_iks200_daily_frame(_qw_bm())

    events = mark_down_day_events(daily, thresholds_pct=[1, 2, 3, 4, 8])

    event = events[events["event_date"].eq(pd.Timestamp("2024-01-02"))]
    assert event["threshold_pct"].tolist() == [3]
    assert event["bucket_label"].tolist() == ["-3%~-4% 미만"]
    assert "short" not in " ".join(events.columns).lower()


def test_next_day_reactions_measure_buy_close_gap_and_next_day_ohlc_path() -> None:
    daily = build_iks200_daily_frame(_qw_bm())
    events = mark_down_day_events(daily, thresholds_pct=[3])

    reactions = build_next_day_reactions(daily, events)

    trade = reactions[reactions["event_date"].eq(pd.Timestamp("2024-01-02"))].iloc[0]
    assert trade["next_date"] == pd.Timestamp("2024-01-03")
    assert trade["gap_ret"] == pytest.approx(98.0 / 97.0 - 1.0)
    assert trade["next_high_ret"] == pytest.approx(101.0 / 97.0 - 1.0)
    assert trade["next_low_ret"] == pytest.approx(95.0 / 97.0 - 1.0)
    assert trade["next_close_ret"] == pytest.approx(100.0 / 97.0 - 1.0)
    assert trade["open_to_close_ret"] == pytest.approx(100.0 / 98.0 - 1.0)
    assert trade["gap_up"]


def test_summaries_group_overall_and_yearly_by_threshold() -> None:
    daily = build_iks200_daily_frame(_qw_bm())
    events = mark_down_day_events(daily, thresholds_pct=[1, 2, 3])
    reactions = build_next_day_reactions(daily, events)

    overall = summarize_overall(reactions)
    yearly = summarize_yearly(reactions)

    row = overall[overall["threshold_pct"].eq(3)].iloc[0]
    assert row["n"] == 2
    assert row["gap_up_rate"] == pytest.approx(1.0)
    assert row["mean_gap_ret"] == pytest.approx(((98.0 / 97.0 - 1.0) + (97.0 / 96.03 - 1.0)) / 2.0)

    yearly_row = yearly[(yearly["event_year"].eq(2024)) & (yearly["threshold_pct"].eq(3))].iloc[0]
    assert yearly_row["n"] == 1
    assert yearly_row["mean_next_close_ret"] == pytest.approx(100.0 / 97.0 - 1.0)


def test_intraday_paths_use_futures_minutes_for_next_day_event_study() -> None:
    daily = build_iks200_daily_frame(_qw_bm())
    events = mark_down_day_events(daily, thresholds_pct=[3])
    reactions = build_next_day_reactions(daily, events)
    minutes = pd.DataFrame(
        [
            {"ts": "2024-01-02 06:30:00Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "1530", "close": 200.0},
            {"ts": "2024-01-03 00:00:00Z", "trade_date_kst": "2024-01-03", "hhmm_kst": "0900", "close": 202.0},
            {"ts": "2024-01-03 00:01:00Z", "trade_date_kst": "2024-01-03", "hhmm_kst": "0901", "close": 204.0},
            {"ts": "2025-02-04 06:30:00Z", "trade_date_kst": "2025-02-04", "hhmm_kst": "1530", "close": 300.0},
            {"ts": "2025-02-05 00:00:00Z", "trade_date_kst": "2025-02-05", "hhmm_kst": "0900", "close": 303.0},
        ]
    )

    paths = build_intraday_next_day_paths(minutes, reactions)

    row = paths[(paths["event_date"].eq(pd.Timestamp("2024-01-02"))) & (paths["minute_from_open"].eq(1))].iloc[0]
    assert row["threshold_pct"] == 3
    assert row["futures_event_close"] == 200.0
    assert row["ret_from_futures_event_close"] == pytest.approx(204.0 / 200.0 - 1.0)
    assert row["ret_from_next_open"] == pytest.approx(204.0 / 202.0 - 1.0)

    summary = summarize_intraday_paths(paths)
    summary_row = summary[
        (summary["event_year"].eq(2024)) & (summary["threshold_pct"].eq(3)) & (summary["minute_from_open"].eq(1))
    ].iloc[0]
    assert summary_row["n"] == 1
    assert summary_row["mean_ret_from_futures_event_close"] == pytest.approx(204.0 / 200.0 - 1.0)
