from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from etc.sidecar import (
    build_parquet_bundle,
    collect_next_day_reaction_files,
    run_event_study,
    run_pipeline,
    write_excel_report,
)


def _write_price_excel(path, label: str, rows: list[tuple[date, time, int, int, int, int]]) -> None:
    frame = pd.DataFrame(
        rows,
        columns=[f"[일시]{label}", "시간", "시가", "고가", "저가", "종가"],
    )
    frame.to_excel(path, index=False)


def _write_event_excel(path) -> None:
    frame = pd.DataFrame(
        [
            [None, None, None, None, None, None, "현재가", "대비", "등락률", "현재가", "대비", "등락률", "현재가", "대비", "등락률"],
            [1, date(2026, 1, 2), time(9, 6, 2), "유가", "사이드카", "발동", 100, 4, 0.04, 100, 4, 0.04, 100, 4, 0.05],
            [2, date(2026, 1, 2), time(9, 11, 2), "유가", "사이드카", "발동해제", 101, 4, 0.04, 101, 4, 0.04, 101, 4, 0.04],
            [3, date(2026, 1, 3), time(9, 6, 2), "유가", "사이드카", "발동", 100, -4, -0.04, 100, -4, -0.04, 100, -4, -0.05],
            [4, date(2026, 1, 3), time(9, 11, 2), "유가", "사이드카", "발동해제", 101, -4, -0.04, 101, -4, -0.04, 101, -4, -0.04],
            [5, date(2026, 1, 4), time(9, 3, 42), "유가", "CB", "발동", 100, -8, -0.08, 100, -8, -0.08, 100, -8, -0.08],
            [6, date(2026, 1, 4), time(9, 23, 42), "유가", "CB", "발동해제", 100, -8, -0.08, 100, -8, -0.08, 100, -8, -0.08],
        ],
        columns=["No", "일자", "시간", "시장", "구분", "시장조치", "KOSPI", "Unnamed: 7", "Unnamed: 8", "KOSPI200", "Unnamed: 10", "Unnamed: 11", "K200선물", "Unnamed: 13", "Unnamed: 14"],
    )
    frame.to_excel(path, index=False)


def _write_next_day_reaction_csv(path) -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "next_date": "2026-01-03",
                "kind": "사이드카",
                "activation_time": "09:06:02",
                "release_time": "09:11:02",
                "direction": "Up trigger",
                "kospi200_next_return_pct": -1.0,
            },
            {
                "date": "2026-01-03",
                "next_date": "2026-01-04",
                "kind": "사이드카",
                "activation_time": "09:06:02",
                "release_time": "09:11:02",
                "direction": "Down trigger",
                "kospi200_next_return_pct": 2.0,
            },
        ]
    )
    frame.to_csv(path, index=False)


def _write_sidecar_sources(source) -> None:
    _write_event_excel(source / "사이드카 이력.xlsx")
    _write_price_excel(
        source / "KODEX 레버리지(1분).xlsx",
        "KODEX 레버리지",
        [
            (date(2026, 1, 2), time(9, 6), 100, 100, 100, 100),
            (date(2026, 1, 2), time(9, 9), 100, 100, 100, 100),
            (date(2026, 1, 2), time(9, 14), 103, 103, 103, 103),
        ],
    )
    _write_price_excel(
        source / "KODEX 인버스(1분).xlsx",
        "KODEX 인버스",
        [
            (date(2026, 1, 3), time(9, 6), 1000, 1000, 1000, 1000),
            (date(2026, 1, 3), time(9, 9), 1000, 1000, 1000, 1000),
            (date(2026, 1, 3), time(9, 14), 990, 990, 990, 990),
        ],
    )


def test_build_parquet_bundle_normalizes_price_and_sidecar_history(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "parquet"
    source.mkdir()
    _write_sidecar_sources(source)
    (source / "~$KODEX 레버리지(1분).xlsx").write_bytes(b"locked")

    paths = build_parquet_bundle(source, output)

    assert set(paths) == {"events", "leverage_prices", "inverse_prices"}
    events = pd.read_parquet(paths["events"], engine="pyarrow")
    leverage = pd.read_parquet(paths["leverage_prices"], engine="pyarrow")
    inverse = pd.read_parquet(paths["inverse_prices"], engine="pyarrow")
    assert events["kind"].tolist() == ["사이드카", "사이드카", "사이드카", "사이드카"]
    assert events["event_dt"].tolist()[0] == pd.Timestamp("2026-01-02 09:06:02")
    assert leverage.loc[0, "dt"] == pd.Timestamp("2026-01-02 09:06:00")
    assert leverage.loc[0, "close"] == 100
    assert inverse.loc[0, "close"] == 1000


def test_run_event_study_defaults_to_entry3_exit3_strategy(tmp_path):
    source = tmp_path / "source"
    parquet_dir = tmp_path / "parquet"
    source.mkdir()
    _write_sidecar_sources(source)
    build_parquet_bundle(source, parquet_dir)

    trades, summary = run_event_study(parquet_dir)

    assert trades[["direction", "etf", "entry_delay_m", "exit_delay_m"]].to_dict("records") == [
        {"direction": "buy_sidecar", "etf": "KODEX leverage", "entry_delay_m": 3, "exit_delay_m": 3},
        {"direction": "sell_sidecar", "etf": "KODEX inverse", "entry_delay_m": 3, "exit_delay_m": 3},
    ]
    assert trades["entry_dt"].tolist() == [pd.Timestamp("2026-01-02 09:09:00"), pd.Timestamp("2026-01-03 09:09:00")]
    assert trades["exit_dt"].tolist() == [pd.Timestamp("2026-01-02 09:14:00"), pd.Timestamp("2026-01-03 09:14:00")]
    assert trades["ret_pct"].round(4).tolist() == [3.0, -1.0]
    all_summary = summary[(summary["scope"] == "all_years") & (summary["group"] == "all")].iloc[0]
    assert all_summary["win_rate_pct"] == pytest.approx(50.0)


def test_collect_next_day_reaction_files_copies_legacy_summaries_to_results(tmp_path):
    source = tmp_path / "source"
    results = tmp_path / "results"
    source.mkdir()
    _write_next_day_reaction_csv(source / "sidecar_event_summary_no_cb.csv")
    _write_next_day_reaction_csv(source / "sidecar_event_summary_with_cb.csv")

    written = collect_next_day_reaction_files(source, results)

    assert written["sidecar_only"] == results / "next_day_reaction_sidecar_only.csv"
    assert written["with_cb"] == results / "next_day_reaction_with_cb.csv"
    summary = pd.read_csv(results / "next_day_reaction_summary.csv")
    assert set(summary["file"]) == {"next_day_reaction_sidecar_only.csv", "next_day_reaction_with_cb.csv"}
    assert summary.loc[summary["group"].eq("all"), "opposite_trigger_pct"].tolist() == [100.0, 100.0]


def test_run_pipeline_writes_clean_result_names(tmp_path):
    source = tmp_path / "source"
    parquet_dir = tmp_path / "parquet"
    results = tmp_path / "results"
    source.mkdir()
    _write_sidecar_sources(source)
    _write_next_day_reaction_csv(source / "sidecar_event_summary_no_cb.csv")

    run_pipeline(source, parquet_dir, results)

    assert (results / "event_study_entry3_exit3_trades.csv").exists()
    assert (results / "event_study_entry3_exit3_summary.csv").exists()
    assert (results / "event_study_entry3_exit3_report.xlsx").exists()
    assert (results / "next_day_reaction_sidecar_only.csv").exists()
    assert (results / "next_day_reaction_summary.csv").exists()


def test_write_excel_report_includes_trade_summary_and_year_sheets(tmp_path):
    trades = pd.DataFrame(
        [
            {
                "date": date(2026, 1, 2),
                "year": 2026,
                "activation_dt": pd.Timestamp("2026-01-02 09:06:02"),
                "release_dt": pd.Timestamp("2026-01-02 09:11:02"),
                "direction": "buy_sidecar",
                "etf": "KODEX leverage",
                "futures_return_at_trigger_pct": 5.0,
                "minutes_to_release": 5.0,
                "entry_delay_m": 3,
                "entry_dt": pd.Timestamp("2026-01-02 09:09:00"),
                "entry_price": 100.0,
                "exit_delay_m": 3,
                "exit_dt": pd.Timestamp("2026-01-02 09:14:00"),
                "exit_price": 103.0,
                "ret_pct": 3.0,
            },
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "scope": "all_years",
                "group": "all",
                "entry_delay_m": 3,
                "exit_delay_m": 3,
                "n": 1,
                "wins": 1,
                "losses": 0,
                "mean_pct": 3.0,
                "median_pct": 3.0,
                "win_rate_pct": 100.0,
                "min_pct": 3.0,
                "max_pct": 3.0,
                "sum_pct": 3.0,
            },
        ]
    )

    output = write_excel_report(trades, summary, tmp_path / "report.xlsx")

    workbook = pd.ExcelFile(output)
    assert workbook.sheet_names == ["trades", "summary", "trades_2026"]
    detail = pd.read_excel(output, sheet_name="trades")
    assert detail.loc[0, "entry_delay_m"] == 3
    assert detail.loc[0, "exit_delay_m"] == 3
