from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from sidecar.pipeline import build_parquet_bundle, run_event_study, write_excel_report


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
            [5, date(2026, 1, 3), time(9, 11, 2), "유가", "사이드카", "발동해제", 101, -4, -0.04, 101, -4, -0.04, 101, -4, -0.04],
            [6, date(2026, 1, 4), time(9, 3, 42), "유가", "CB", "발동", 100, -8, -0.08, 100, -8, -0.08, 100, -8, -0.08],
            [7, date(2026, 1, 4), time(9, 23, 42), "유가", "CB", "발동해제", 100, -8, -0.08, 100, -8, -0.08, 100, -8, -0.08],
        ],
        columns=["No", "일자", "시간", "시장", "구분", "시장조치", "KOSPI", "Unnamed: 7", "Unnamed: 8", "KOSPI200", "Unnamed: 10", "Unnamed: 11", "K200선물", "Unnamed: 13", "Unnamed: 14"],
    )
    frame.to_excel(path, index=False)


def test_build_parquet_bundle_normalizes_price_and_sidecar_history(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "parquet"
    source.mkdir()
    _write_event_excel(source / "사이드카 이력.xlsx")
    _write_price_excel(
        source / "KODEX 레버리지(1분).xlsx",
        "KODEX 레버리지",
        [(date(2026, 1, 2), time(9, 6), 100, 101, 99, 100)],
    )
    _write_price_excel(
        source / "KODEX 인버스(1분).xlsx",
        "KODEX 인버스",
        [(date(2026, 1, 3), time(9, 6), 1000, 1001, 999, 1000)],
    )
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


def test_run_event_study_pairs_events_and_uses_inverse_for_sell_sidecars(tmp_path):
    source = tmp_path / "source"
    parquet_dir = tmp_path / "parquet"
    source.mkdir()
    _write_event_excel(source / "사이드카 이력.xlsx")
    _write_price_excel(
        source / "KODEX 레버리지(1분).xlsx",
        "KODEX 레버리지",
        [
            (date(2026, 1, 2), time(9, 6), 100, 100, 100, 100),
            (date(2026, 1, 2), time(9, 12), 101, 101, 101, 101),
            (date(2026, 1, 2), time(9, 14), 103, 103, 103, 103),
            (date(2026, 1, 2), time(9, 16), 105, 105, 105, 105),
        ],
    )
    _write_price_excel(
        source / "KODEX 인버스(1분).xlsx",
        "KODEX 인버스",
        [
            (date(2026, 1, 3), time(9, 6), 1000, 1000, 1000, 1000),
            (date(2026, 1, 3), time(9, 12), 1010, 1010, 1010, 1010),
            (date(2026, 1, 3), time(9, 14), 990, 990, 990, 990),
            (date(2026, 1, 3), time(9, 16), 980, 980, 980, 980),
        ],
    )
    build_parquet_bundle(source, parquet_dir)

    trades, summary = run_event_study(parquet_dir)

    assert trades[["direction", "etf"]].to_dict("records") == [
        {"direction": "buy_sidecar", "etf": "KODEX leverage"},
        {"direction": "sell_sidecar", "etf": "KODEX inverse"},
    ]
    assert trades["ret_1m_pct"].round(4).tolist() == [1.0, 1.0]
    assert trades["ret_3m_pct"].round(4).tolist() == [3.0, -1.0]
    assert trades["ret_5m_pct"].round(4).tolist() == [5.0, -2.0]
    all_summary = summary[summary["group"] == "all"].set_index("horizon")
    assert all_summary.loc["1m", "mean_pct"] == pytest.approx(1.0)
    assert all_summary.loc["3m", "win_rate_pct"] == pytest.approx(50.0)


def test_write_excel_report_includes_total_and_yearly_sheets(tmp_path):
    trades = pd.DataFrame(
        [
            {
                "date": date(2025, 4, 7),
                "activation_dt": pd.Timestamp("2025-04-07 09:12:11"),
                "release_dt": pd.Timestamp("2025-04-07 09:17:11"),
                "direction": "sell_sidecar",
                "etf": "KODEX inverse",
                "futures_return_at_trigger_pct": -5.18,
                "minutes_to_release": 5.0,
                "entry_dt": pd.Timestamp("2025-04-07 09:12:00"),
                "entry_price": 4860.0,
                "exit_1m_dt": pd.Timestamp("2025-04-07 09:18:00"),
                "exit_1m_price": 4845.0,
                "ret_1m_pct": -0.308642,
                "exit_3m_dt": pd.Timestamp("2025-04-07 09:20:00"),
                "exit_3m_price": 4860.0,
                "ret_3m_pct": 0.0,
                "exit_5m_dt": pd.Timestamp("2025-04-07 09:22:00"),
                "exit_5m_price": 4865.0,
                "ret_5m_pct": 0.102881,
            },
            {
                "date": date(2026, 1, 2),
                "activation_dt": pd.Timestamp("2026-01-02 09:06:02"),
                "release_dt": pd.Timestamp("2026-01-02 09:11:02"),
                "direction": "buy_sidecar",
                "etf": "KODEX leverage",
                "futures_return_at_trigger_pct": 5.0,
                "minutes_to_release": 5.0,
                "entry_dt": pd.Timestamp("2026-01-02 09:06:00"),
                "entry_price": 100.0,
                "exit_1m_dt": pd.Timestamp("2026-01-02 09:12:00"),
                "exit_1m_price": 101.0,
                "ret_1m_pct": 1.0,
                "exit_3m_dt": pd.Timestamp("2026-01-02 09:14:00"),
                "exit_3m_price": 103.0,
                "ret_3m_pct": 3.0,
                "exit_5m_dt": pd.Timestamp("2026-01-02 09:16:00"),
                "exit_5m_price": 105.0,
                "ret_5m_pct": 5.0,
            },
        ]
    )

    output = write_excel_report(trades, tmp_path / "report.xlsx")

    workbook = pd.ExcelFile(output)
    assert workbook.sheet_names == [
        "거래내역_전체",
        "요약_전체",
        "요약_연도별",
        "거래내역_2025",
        "요약_2025",
        "거래내역_2026",
        "요약_2026",
    ]
    detail_2025 = pd.read_excel(output, sheet_name="거래내역_2025")
    assert detail_2025.loc[0, "발동시간"] == pd.Timestamp("2025-04-07 09:12:11")
    assert detail_2025.loc[0, "종료시간"] == pd.Timestamp("2025-04-07 09:17:11")
    assert detail_2025.loc[0, "사이드카방향"] == "매도 사이드카"
