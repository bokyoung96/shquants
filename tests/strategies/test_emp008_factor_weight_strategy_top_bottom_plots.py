from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtesting.strategies.emp008.reports.factor_weight_strategy_top_bottom_plots import (
    build_strategy_state_matrix,
    calculate_replacement_counts,
    generate_strategy_top_bottom_report,
    validate_paired_rows,
)


def _rows(
    candidate: str,
    date: str,
    tickers: list[str],
    *,
    names: dict[str, str] | None = None,
) -> pd.DataFrame:
    names = names or {}
    return pd.DataFrame(
        {
            "date": pd.Timestamp(date),
            "candidate_id": candidate,
            "rank": range(1, len(tickers) + 1),
            "ticker": tickers,
            "stock_name": [names.get(ticker, ticker) for ticker in tickers],
        }
    )


def _paired_sample() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate = "s50_m30_e10_v10"
    wi26_top = _rows(candidate, "2020-01-31", [f"A{i:03d}" for i in range(1, 11)])
    wi26_bottom = _rows(candidate, "2020-01-31", [f"B{i:03d}" for i in range(1, 11)])
    wics_top = _rows(candidate, "2020-01-31", [f"A{i:03d}" for i in range(1, 10)] + ["A999"])
    wics_bottom = _rows(candidate, "2020-01-31", [f"B{i:03d}" for i in range(1, 9)] + ["B998", "B999"])
    return wi26_top, wi26_bottom, wics_top, wics_bottom


def test_strategy_state_and_replacement_counts_are_not_aggregated() -> None:
    wi26_top, wi26_bottom, wics_top, wics_bottom = _paired_sample()

    state = build_strategy_state_matrix(
        pd.concat([wi26_top, _rows("equal_25", "2020-01-31", ["A777"])]),
        wi26_bottom,
        candidate_id="s50_m30_e10_v10",
    )
    counts = calculate_replacement_counts(wi26_top, wi26_bottom, wics_top, wics_bottom)

    assert state.loc[pd.Timestamp("2020-01-31"), "A001"] == 1
    assert state.loc[pd.Timestamp("2020-01-31"), "B001"] == -1
    assert "A777" not in state.columns
    assert counts.iloc[0][["top_replacements", "bottom_replacements", "total_replacements"]].tolist() == [1, 2, 3]


def test_pair_validation_rejects_missing_strategy() -> None:
    wi26_top, wi26_bottom, wics_top, wics_bottom = _paired_sample()
    extra = _rows("equal_25", "2020-01-31", [f"C{i:03d}" for i in range(1, 11)])

    with pytest.raises(ValueError, match="전략 집합 불일치"):
        validate_paired_rows(pd.concat([wi26_top, extra]), wi26_bottom, wics_top, wics_bottom)


def _write_workbook(path: Path, candidate_ids: tuple[str, ...]) -> None:
    top_parts: list[pd.DataFrame] = []
    bottom_parts: list[pd.DataFrame] = []
    for candidate_idx, candidate in enumerate(candidate_ids):
        for date_idx, date in enumerate(("2020-01-31", "2023-01-31", "2024-01-31")):
            suffix = candidate_idx * 100 + date_idx * 20
            top_parts.append(_rows(candidate, date, [f"A{suffix + i:05d}" for i in range(1, 11)]))
            bottom_parts.append(_rows(candidate, date, [f"B{suffix + i:05d}" for i in range(1, 11)]))

    def workbook_frame(parts: list[pd.DataFrame]) -> pd.DataFrame:
        frame = pd.concat(parts, ignore_index=True)
        return pd.DataFrame(
            {
                "리밸런싱일": frame["date"],
                "후보 순위": 1,
                "후보 ID": frame["candidate_id"],
                "Size": 0.5,
                "12개월 모멘텀": 0.3,
                "영업이익 컨센서스": 0.1,
                "FCF/TEV": 0.1,
                "종목 순위": frame["rank"],
                "종목코드": frame["ticker"],
                "종목명": frame["stock_name"],
            }
        )

    with pd.ExcelWriter(path) as writer:
        workbook_frame(top_parts).to_excel(writer, sheet_name="Top 10", index=False)
        workbook_frame(bottom_parts).to_excel(writer, sheet_name="Bottom 10", index=False)


def test_report_writes_summary_details_and_source_csvs(tmp_path: Path) -> None:
    candidates = ("s50_m30_e10_v10", "equal_25")
    wi26 = tmp_path / "wi26.xlsx"
    wics = tmp_path / "wics.xlsx"
    _write_workbook(wi26, candidates)
    _write_workbook(wics, candidates)

    outputs = generate_strategy_top_bottom_report(
        wi26_workbook=wi26,
        wics_workbook=wics,
        output_dir=tmp_path / "output",
        start="2020-01-01",
    )

    assert outputs["summary_png"].stat().st_size > 0
    assert outputs["replacement_csv"].stat().st_size > 0
    assert len(outputs["detail_pngs"]) == 2
    assert len(outputs["state_csvs"]) == 4
    assert outputs["manifest_json"].stat().st_size > 0
