from __future__ import annotations

import pandas as pd


def _sample_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.DataFrame(
        {
            "open": [100.0, 99.0, 98.0],
            "high": [101.0, 100.0, 101.0],
            "low": [99.0, 96.0, 95.0],
            "close": [100.0, 97.0, 100.0],
            "ret_cc": [None, -0.03, 100.0 / 97.0 - 1.0],
            "ret_oc": [0.0, 97.0 / 99.0 - 1.0, 100.0 / 98.0 - 1.0],
            "date_index": [0, 1, 2],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).rename("date"),
    )
    events = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-02"),
                "event_year": 2024,
                "event_idx": 1,
                "threshold_pct": 3,
                "threshold_ret": -0.03,
                "bucket_floor_pct": 3,
                "bucket_ceiling_pct": 4,
                "bucket_label": "-3%~-4% 미만",
                "event_ret_cc": -0.03,
                "event_ret_oc": 97.0 / 99.0 - 1.0,
                "event_open": 99.0,
                "event_high": 100.0,
                "event_low": 96.0,
                "event_close": 97.0,
            }
        ]
    )
    reactions = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-02"),
                "next_date": pd.Timestamp("2024-01-03"),
                "event_year": 2024,
                "next_year": 2024,
                "threshold_pct": 3,
                "threshold_ret": -0.03,
                "bucket_floor_pct": 3,
                "bucket_ceiling_pct": 4,
                "bucket_label": "-3%~-4% 미만",
                "event_ret_cc": -0.03,
                "event_ret_oc": 97.0 / 99.0 - 1.0,
                "event_close": 97.0,
                "next_open": 98.0,
                "next_high": 101.0,
                "next_low": 95.0,
                "next_close": 100.0,
                "gap_ret": 98.0 / 97.0 - 1.0,
                "gap_up": True,
                "next_high_ret": 101.0 / 97.0 - 1.0,
                "next_low_ret": 95.0 / 97.0 - 1.0,
                "next_close_ret": 100.0 / 97.0 - 1.0,
                "open_to_close_ret": 100.0 / 98.0 - 1.0,
                "gap_to_close_change": 100.0 / 98.0 - 1.0,
                "overnight_gain_kept": 3.0,
            },
            {
                "event_date": pd.Timestamp("2025-01-02"),
                "next_date": pd.Timestamp("2025-01-03"),
                "event_year": 2025,
                "next_year": 2025,
                "threshold_pct": 3,
                "threshold_ret": -0.03,
                "bucket_floor_pct": 3,
                "bucket_ceiling_pct": 4,
                "bucket_label": "-3%~-4% 미만",
                "event_ret_cc": -0.04,
                "event_ret_oc": -0.03,
                "event_close": 100.0,
                "next_open": 101.0,
                "next_high": 102.0,
                "next_low": 99.0,
                "next_close": 101.5,
                "gap_ret": 0.01,
                "gap_up": True,
                "next_high_ret": 0.02,
                "next_low_ret": -0.01,
                "next_close_ret": 0.015,
                "open_to_close_ret": 101.5 / 101.0 - 1.0,
                "gap_to_close_change": 101.5 / 101.0 - 1.0,
                "overnight_gain_kept": 1.5,
            },
        ]
    )
    overall = pd.DataFrame(
        [
            {
                "threshold_pct": 3,
                "bucket_label": "-3%~-4% 미만",
                "bucket_floor_pct": 3,
                "bucket_ceiling_pct": 4,
                "n": 1,
                "mean_event_ret_cc": -0.03,
                "median_event_ret_cc": -0.03,
                "gap_up_count": 1,
                "gap_up_rate": 1.0,
                "mean_gap_ret": 98.0 / 97.0 - 1.0,
                "median_gap_ret": 98.0 / 97.0 - 1.0,
                "mean_next_high_ret": 101.0 / 97.0 - 1.0,
                "mean_next_low_ret": 95.0 / 97.0 - 1.0,
                "mean_next_close_ret": 100.0 / 97.0 - 1.0,
                "median_next_close_ret": 100.0 / 97.0 - 1.0,
                "next_close_win_rate": 1.0,
                "mean_open_to_close_ret": 100.0 / 98.0 - 1.0,
                "median_open_to_close_ret": 100.0 / 98.0 - 1.0,
                "open_to_close_win_rate": 1.0,
                "mean_overnight_gain_kept": 3.0,
                "compound_next_close_ret": 100.0 / 97.0 - 1.0,
            }
        ]
    )
    yearly = overall.assign(event_year=2024)[["event_year", *overall.columns]]
    intraday_paths = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-02"),
                "next_date": pd.Timestamp("2024-01-03"),
                "event_year": 2024,
                "threshold_pct": 3,
                "bucket_label": "-3%~-4% 미만",
                "event_ret_cc": -0.03,
                "hhmm_kst": "0900",
                "minute_from_open": 0,
                "futures_event_close": 200.0,
                "futures_next_open": 202.0,
                "futures_price": 202.0,
                "ret_from_futures_event_close": 0.01,
                "ret_from_next_open": 0.0,
            },
            {
                "event_date": pd.Timestamp("2024-01-02"),
                "next_date": pd.Timestamp("2024-01-03"),
                "event_year": 2024,
                "threshold_pct": 3,
                "bucket_label": "-3%~-4% 미만",
                "event_ret_cc": -0.03,
                "hhmm_kst": "0901",
                "minute_from_open": 1,
                "futures_event_close": 200.0,
                "futures_next_open": 202.0,
                "futures_price": 204.0,
                "ret_from_futures_event_close": 0.02,
                "ret_from_next_open": 204.0 / 202.0 - 1.0,
            },
        ]
    )
    intraday_summary = pd.DataFrame(
        [
            {
                "event_year": 2024,
                "threshold_pct": 3,
                "minute_from_open": 0,
                "n": 1,
                "mean_ret_from_futures_event_close": 0.01,
                "median_ret_from_futures_event_close": 0.01,
                "mean_ret_from_next_open": 0.0,
                "median_ret_from_next_open": 0.0,
            },
            {
                "event_year": 2024,
                "threshold_pct": 3,
                "minute_from_open": 1,
                "n": 1,
                "mean_ret_from_futures_event_close": 0.02,
                "median_ret_from_futures_event_close": 0.02,
                "mean_ret_from_next_open": 204.0 / 202.0 - 1.0,
                "median_ret_from_next_open": 204.0 / 202.0 - 1.0,
            },
        ]
    )
    return daily, events, reactions, overall, yearly, intraday_paths, intraday_summary


def test_write_excel_report_creates_numeric_workbook(tmp_path) -> None:
    from scripts.run_next_day_down_event_study import write_excel_report

    daily, events, reactions, overall, yearly, intraday_paths, intraday_summary = _sample_outputs()
    path = tmp_path / "report.xlsx"

    write_excel_report(path, daily, events, reactions, overall, yearly, intraday_paths, intraday_summary)

    assert path.exists()
    sheets = pd.ExcelFile(path).sheet_names
    assert {
        "overall_summary",
        "yearly_summary",
        "event_reactions",
        "matrix_gap_up_probability",
        "intraday_1m_summary",
    }.issubset(sheets)


def test_write_markdown_is_korean_and_includes_2026_summary(tmp_path) -> None:
    from scripts.run_next_day_down_event_study import write_markdown

    daily, events, reactions, overall, yearly, intraday_paths, _ = _sample_outputs()
    yearly_2026 = overall.assign(event_year=2026)[["event_year", *overall.columns]]
    path = tmp_path / "report.md"

    write_markdown(
        path,
        code="IKS200",
        daily=daily,
        events=events,
        reactions=reactions,
        overall=overall,
        yearly=yearly_2026,
        intraday_paths=intraday_paths,
        excel_filename="next_day_down_event_study.xlsx",
    )

    text = path.read_text(encoding="utf-8")
    assert "## 2026년 요약" in text
    assert "## 전체 요약" in text
    assert "## Overall" not in text


def test_plot_yearly_subplots_writes_png(tmp_path) -> None:
    from scripts.run_next_day_down_event_study import plot_yearly_subplots

    _, _, _, _, yearly, _, _ = _sample_outputs()
    path = tmp_path / "yearly.png"

    plot_yearly_subplots(path, yearly)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_yearly_intraday_event_study_subplots_writes_png(tmp_path) -> None:
    from scripts.run_next_day_down_event_study import plot_yearly_intraday_event_study_subplots

    _, _, _, _, _, _, intraday_summary = _sample_outputs()
    path = tmp_path / "intraday.png"

    plot_yearly_intraday_event_study_subplots(path, intraday_summary)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_2026_extreme_intraday_examples_writes_png(tmp_path) -> None:
    from scripts.run_next_day_down_event_study import plot_2026_extreme_intraday_examples

    reactions = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2026-01-02"),
                "next_date": pd.Timestamp("2026-01-03"),
                "event_year": 2026,
                "threshold_pct": 5,
                "bucket_label": "-5%~-6% 미만",
                "event_ret_cc": -0.055,
            },
            {
                "event_date": pd.Timestamp("2026-01-04"),
                "next_date": pd.Timestamp("2026-01-05"),
                "event_year": 2026,
                "threshold_pct": 6,
                "bucket_label": "-6%~-7% 미만",
                "event_ret_cc": -0.065,
            },
        ]
    )
    minutes = pd.DataFrame(
        [
            {"trade_date_kst": "2026-01-02", "hhmm_kst": "0900", "open": 100.0, "close": 99.0},
            {"trade_date_kst": "2026-01-02", "hhmm_kst": "1530", "open": 96.0, "close": 95.0},
            {"trade_date_kst": "2026-01-03", "hhmm_kst": "0900", "open": 98.0, "close": 98.5},
            {"trade_date_kst": "2026-01-03", "hhmm_kst": "1530", "open": 101.0, "close": 102.0},
            {"trade_date_kst": "2026-01-04", "hhmm_kst": "0900", "open": 102.0, "close": 101.5},
            {"trade_date_kst": "2026-01-04", "hhmm_kst": "1530", "open": 99.0, "close": 99.5},
            {"trade_date_kst": "2026-01-05", "hhmm_kst": "0900", "open": 101.0, "close": 101.5},
            {"trade_date_kst": "2026-01-05", "hhmm_kst": "1530", "open": 103.0, "close": 104.0},
            {"trade_date_kst": "2026-01-06", "hhmm_kst": "0900", "open": 104.0, "close": 104.5},
            {"trade_date_kst": "2026-01-06", "hhmm_kst": "1530", "open": 105.0, "close": 106.0},
        ]
    )
    path = tmp_path / "examples.png"

    plot_2026_extreme_intraday_examples(path, minutes, reactions)

    assert path.exists()
    assert path.stat().st_size > 0


def test_three_day_example_path_extends_to_t_plus_two_close() -> None:
    from scripts.run_next_day_down_event_study import _three_day_example_path

    event_day = pd.DataFrame(
        [
            {"open": 100.0, "close": 99.0},
            {"open": 96.0, "close": 95.0},
        ]
    )
    next_day = pd.DataFrame(
        [
            {"open": 98.0, "close": 98.5},
            {"open": 101.0, "close": 102.0},
        ]
    )
    t_plus_two = pd.DataFrame(
        [
            {"open": 103.0, "close": 103.5},
            {"open": 104.0, "close": 105.0},
        ]
    )

    path = _three_day_example_path(event_day, next_day, t_plus_two)

    assert path["x"].tolist() == [0, 1, 13, 14, 26, 27]
    assert path["ret_pct"].iloc[-1] == 5.0


def test_load_futures_minutes_derives_kst_date_and_hhmm_when_missing(tmp_path) -> None:
    from scripts.run_next_day_down_event_study import load_futures_minutes

    path = tmp_path / "minutes.parquet"
    pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-03 00:00:00Z", "2024-01-03 00:01:00Z"]),
            "close": [202.0, 204.0],
        }
    ).to_parquet(path)

    loaded = load_futures_minutes(path)

    assert loaded["trade_date_kst"].astype(str).tolist() == ["2024-01-03", "2024-01-03"]
    assert loaded["hhmm_kst"].tolist() == ["0900", "0901"]
