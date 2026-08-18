from __future__ import annotations

import pandas as pd

from backtesting.strategies.emp008.reports import factor_weight_top_bottom_plots as plots
from backtesting.strategies.emp008.reports.factor_weight_top_bottom_plots import (
    build_consensus_scores,
)


def test_consensus_score_is_top_count_minus_bottom_count_from_2020() -> None:
    top = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-30", "2020-01-31", "2020-01-31"]),
            "candidate_id": ["a", "a", "b"],
            "ticker": ["OLD", "AAA", "AAA"],
        }
    )
    bottom = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "candidate_id": ["a", "b"],
            "ticker": ["AAA", "BBB"],
        }
    )

    result = build_consensus_scores(top=top, bottom=bottom, start="2020-01-01")

    assert result.loc[pd.Timestamp("2020-01-31"), "AAA"] == 1
    assert result.loc[pd.Timestamp("2020-01-31"), "BBB"] == -1
    assert "OLD" not in result.columns


def _write_workbook(path, *, top_ticker: str, bottom_ticker: str) -> None:
    columns = [f"column_{position}" for position in range(13)]
    dates = pd.to_datetime(["2020-01-31", "2023-01-31", "2023-02-28", "2024-01-31"])

    def rows(ticker: str) -> pd.DataFrame:
        values = []
        for date in dates:
            for candidate in ("a", "b"):
                row = [None] * 13
                row[0] = date
                row[2] = candidate
                row[7] = 1
                row[8] = ticker
                values.append(row)
        return pd.DataFrame(values, columns=columns)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        rows(top_ticker).to_excel(writer, sheet_name="Top 10", index=False)
        rows(bottom_ticker).to_excel(writer, sheet_name="Bottom 10", index=False)


def test_generate_report_writes_full_yearly_and_2023_heatmaps(tmp_path) -> None:
    wi26 = tmp_path / "wi26.xlsx"
    wics = tmp_path / "wics.xlsx"
    _write_workbook(wi26, top_ticker="AAA", bottom_ticker="BBB")
    _write_workbook(wics, top_ticker="CCC", bottom_ticker="DDD")

    outputs = plots.generate_top_bottom_visual_report(
        wi26_workbook=wi26,
        wics_workbook=wics,
        output_dir=tmp_path / "report",
        stock_names={
            "AAA": "알파",
            "BBB": "베타",
            "CCC": "감마",
            "DDD": "델타",
        },
    )

    assert set(outputs) == {
        "timeline_png",
        "yearly_frequency_png",
        "detail_2023_png",
        "wi26_scores_csv",
        "wics_scores_csv",
        "manifest_json",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
