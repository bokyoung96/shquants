from __future__ import annotations

import pandas as pd
import pytest

from backtesting.strategies.emp008.reports import factor_weight_grid_comparison as comparison
from backtesting.strategies.emp008.reports.factor_weight_grid_comparison import (
    build_cumulative_pair_gap_bp,
    build_yearly_pair_gap_bp,
    generate_pair_comparison_report,
)


def test_plot_labels_are_korean() -> None:
    assert comparison.PLOT_LABELS == {
        "candidate_title": "동일 후보별 누적수익률 차이: WICS - WI26",
        "yearly_title": "연도별 누적수익률 차이: WICS - WI26",
        "direction": "양수: WICS 우위 / 음수: WI26 우위",
        "paired_note": "매년 동일한 팩터 가중치 후보끼리 비교",
        "y_axis": "누적수익률 차이 (bp)",
        "x_axis": "날짜",
        "median": "후보 중앙값",
    }


def _return_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.to_datetime(
        ["2022-12-30", "2023-01-02", "2023-01-03", "2024-01-02"]
    )
    wi26 = pd.DataFrame(
        {
            "IKS200": [0.0, 0.01, 0.00, 0.02],
            "equal_25": [0.0, 0.02, 0.01, 0.01],
            "s30_m40_e20_v10": [0.0, 0.00, 0.01, 0.03],
        },
        index=index,
    )
    wics = pd.DataFrame(
        {
            "IKS200": [0.0, 0.01, 0.00, 0.02],
            "equal_25": [0.0, 0.01, 0.00, 0.03],
            "s30_m40_e20_v10": [0.0, 0.02, 0.00, 0.02],
        },
        index=index,
    )
    return wi26, wics


def test_cumulative_pair_gap_pairs_common_candidates_and_uses_wics_minus_wi26() -> None:
    wi26, wics = _return_panels()

    result = build_cumulative_pair_gap_bp(wi26, wics)

    assert result.columns.tolist() == ["equal_25", "s30_m40_e20_v10"]
    assert result.loc[pd.Timestamp("2023-01-02"), "equal_25"] == pytest.approx(
        (1.01 - 1.02) * 10_000.0
    )
    assert result.loc[pd.Timestamp("2023-01-02"), "s30_m40_e20_v10"] > 0.0


def test_yearly_pair_gap_resets_each_year_and_drops_prior_year_zero_baseline() -> None:
    wi26, wics = _return_panels()

    result = build_yearly_pair_gap_bp(wi26, wics)

    assert list(result) == [2023, 2024]
    for frame in result.values():
        assert frame.iloc[0].eq(0.0).all()
    assert result[2023].iloc[-1]["equal_25"] < 0.0
    assert result[2024].iloc[-1]["equal_25"] > 0.0


def test_pair_gap_rejects_different_benchmark_returns() -> None:
    wi26, wics = _return_panels()
    wics.loc[pd.Timestamp("2023-01-03"), "IKS200"] = 0.05

    with pytest.raises(ValueError, match="benchmark returns differ"):
        build_cumulative_pair_gap_bp(wi26, wics)


def test_pair_gap_rejects_missing_common_candidate() -> None:
    wi26, wics = _return_panels()
    wics = wics.rename(
        columns={
            "equal_25": "other_a",
            "s30_m40_e20_v10": "other_b",
        }
    )

    with pytest.raises(ValueError, match="no common candidate"):
        build_cumulative_pair_gap_bp(wi26, wics)


def test_generate_report_writes_both_subplots_and_year_end_table(tmp_path) -> None:
    wi26, wics = _return_panels()
    wi26_csv = tmp_path / "wi26.csv"
    wics_csv = tmp_path / "wics.csv"
    output_dir = tmp_path / "comparison"
    wi26.to_csv(wi26_csv, index_label="date")
    wics.to_csv(wics_csv, index_label="date")

    outputs = generate_pair_comparison_report(
        wi26_csv=wi26_csv,
        wics_csv=wics_csv,
        output_dir=output_dir,
    )

    assert set(outputs) == {
        "candidate_pair_cumulative_gap_png",
        "yearly_pair_cumulative_gap_png",
        "yearly_pair_end_gap_csv",
        "manifest_json",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    year_end = pd.read_csv(outputs["yearly_pair_end_gap_csv"], index_col="year")
    assert year_end.index.tolist() == [2023, 2024]
    assert year_end.loc[2023, "equal_25"] < 0.0
