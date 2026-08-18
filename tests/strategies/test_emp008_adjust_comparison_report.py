from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backtesting.strategies.emp008.reports import adjust_comparison as report_module
from backtesting.strategies.emp008.reports.adjust_comparison import (
    ComparisonRun,
    build_adjust_comparison_report,
)


def _write_run(
    root: Path,
    *,
    factor_set: str,
    returns: list[float],
    weights: list[list[float]],
    sector_neutral_dataset: str | None = None,
) -> None:
    gross_dir = root / "backtests" / "gross"
    net_dir = root / "backtests" / "net"
    (gross_dir / "series").mkdir(parents=True)
    (net_dir / "series").mkdir(parents=True)
    (root / "weights").mkdir(parents=True)
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    pd.DataFrame({"date": dates, "returns": returns}).to_csv(gross_dir / "series" / "returns.csv", index=False)
    pd.DataFrame({"date": dates, "returns": [value - 0.0001 for value in returns]}).to_csv(
        net_dir / "series" / "returns.csv", index=False
    )
    pd.DataFrame(weights, index=dates, columns=["A", "B"]).to_csv(root / "weights" / "target_weights.csv")
    (root / "run_summary.json").write_text(
        json.dumps(
            {
                "factor_set": factor_set,
                "risk_model": "factor_idio",
                "tracking_error_annual": 0.007,
                "sector_neutral_dataset": sector_neutral_dataset,
                "backtest": {"output_dir": str(gross_dir)},
                "costed_backtest": {"output_dir": str(net_dir)},
            }
        ),
        encoding="utf-8",
    )


def test_yearly_cumulative_excess_starts_at_zero_before_first_year_return() -> None:
    returns = pd.DataFrame(
        {
            "Model": [0.20, 0.01, -0.005],
            "KOSPI200 BM": [0.10, 0.002, 0.001],
        },
        index=pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"]),
    )

    excess = report_module._yearly_cumulative_excess_frame(returns, ("Model",), 2024)

    assert excess.index[0] == pd.Timestamp("2023-12-29")
    assert excess.iloc[0].eq(0.0).all()
    expected_final = ((1.01 * 0.995) / (1.002 * 1.001) - 1.0) * 10_000.0
    assert excess.loc[pd.Timestamp("2024-01-03"), "Model"] == pytest.approx(expected_final)


def test_build_adjust_comparison_report_writes_three_models_and_benchmark(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    mfbt = tmp_path / "mfbt"
    adjust = tmp_path / "adjust"
    _write_run(origin, factor_set="origin", returns=[0.0, 0.01, -0.005], weights=[[0.5, 0.5]] * 3)
    _write_run(mfbt, factor_set="mfbt", returns=[0.0, 0.012, -0.004], weights=[[0.6, 0.4]] * 3)
    _write_run(adjust, factor_set="adjust", returns=[0.0, 0.015, -0.003], weights=[[0.7, 0.3]] * 3)
    benchmark_path = tmp_path / "benchmark.parquet"
    pd.DataFrame({"IKS200": [100.0, 101.0, 100.5]}, index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])).to_parquet(
        benchmark_path
    )

    payload = build_adjust_comparison_report(
        runs=(
            ComparisonRun("기존 EMP008", "origin", origin),
            ComparisonRun("1차수정 EMP008", "mfbt", mfbt),
            ComparisonRun("2차수정 EMP008", "adjust", adjust),
        ),
        benchmark_path=benchmark_path,
        output_dir=tmp_path / "report",
    )

    assert payload["period"] == {"start": "2024-01-31", "end": "2024-03-29", "days": 3}
    assert set(payload["net_metrics"]) == {"기존 EMP008", "1차수정 EMP008", "2차수정 EMP008", "KOSPI200 BM"}
    assert payload["factor_sets"]["2차수정 EMP008"] == [
        "momentum_12_1m",
        "earnings_momentum",
        "dividend_yield_ttm",
        "value",
        "ln_market_cap",
    ]
    assert payload["latest_position_gap"]["adjust_vs_mfbt_active_share_pct"] == pytest.approx(10.0)
    for key in (
        "report_html",
        "report_json",
        "report_xlsx",
        "cumulative_mdd_png",
        "cumulative_difference_png",
        "cumulative_difference_csv",
        "yearly_cumulative_png",
        "return_distribution_png",
        "latest_positions_png",
    ):
        assert Path(payload[key]).exists()

    cumulative_difference = pd.read_csv(payload["cumulative_difference_csv"])
    assert set(cumulative_difference.columns) == {
        "date",
        "기존 EMP008 vs BM",
        "1차수정 EMP008 vs BM",
        "2차수정 EMP008 vs BM",
    }
    assert payload["yearly_excess_bp"]["2024"]["2차수정 EMP008"] == pytest.approx(
        payload["cumulative_difference_final_bp"]["2차수정 EMP008 vs BM"]
    )
    report_html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "기존 EMP008 · 1차수정 EMP008 · 2차수정 EMP008 · KOSPI200 BM" in report_html
    assert "KOSPI200 대비 누적 초과성과" in report_html
    assert "1차수정 EMP008 대비 누적 성과차이" not in report_html
    for column in ("기존", "1차", "2차", "2차-1차", "절대차이"):
        assert f"<th>{column}</th>" in report_html
    assert "<th>Adjust-MFBT</th>" not in report_html
    assert "<th>abs_Adjust-MFBT</th>" not in report_html
    assert "최근 2차와 1차의 active share" in report_html


def test_adjust_comparison_report_allows_model_specific_sector_taxonomies(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    mfbt = tmp_path / "mfbt"
    adjust = tmp_path / "adjust"
    _write_run(origin, factor_set="origin", returns=[0.0, 0.01, -0.005], weights=[[0.5, 0.5]] * 3)
    _write_run(
        mfbt,
        factor_set="mfbt",
        returns=[0.0, 0.012, -0.004],
        weights=[[0.6, 0.4]] * 3,
        sector_neutral_dataset="qw_wics_sec_big",
    )
    _write_run(
        adjust,
        factor_set="adjust",
        returns=[0.0, 0.015, -0.003],
        weights=[[0.7, 0.3]] * 3,
        sector_neutral_dataset="qw_wics_sec_big",
    )
    benchmark_path = tmp_path / "benchmark.parquet"
    pd.DataFrame(
        {"IKS200": [100.0, 101.0, 100.5]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"]),
    ).to_parquet(benchmark_path)

    payload = build_adjust_comparison_report(
        runs=(
            ComparisonRun("기존 EMP008", "origin", origin),
            ComparisonRun("1차수정 EMP008", "mfbt", mfbt),
            ComparisonRun("2차수정 EMP008", "adjust", adjust),
        ),
        benchmark_path=benchmark_path,
        output_dir=tmp_path / "report",
    )

    assert payload["sector_neutral_datasets"] == {
        "기존 EMP008": "WI26",
        "1차수정 EMP008": "WICS",
        "2차수정 EMP008": "WICS",
    }
    report_html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "기존 EMP008</td>" in report_html
    assert "WI26</td>" in report_html
    assert report_html.count("WICS</td>") == 2


def test_adjust_comparison_report_rejects_mixed_risk_conditions(tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("origin", "mfbt", "adjust")]
    for root, factor_set in zip(roots, ("origin", "mfbt", "adjust"), strict=True):
        _write_run(root, factor_set=factor_set, returns=[0.0, 0.01, 0.02], weights=[[0.5, 0.5]] * 3)
    summary_path = roots[-1] / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["risk_model"] = "direct_covariance"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.parquet"
    pd.DataFrame({"IKS200": [100.0, 101.0, 102.0]}, index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])).to_parquet(
        benchmark_path
    )

    with pytest.raises(ValueError, match="same risk conditions"):
        build_adjust_comparison_report(
            runs=tuple(
                ComparisonRun(label, factor_set, root)
                for label, factor_set, root in zip(("Origin", "MFBT", "Adjust"), ("origin", "mfbt", "adjust"), roots, strict=True)
            ),
            benchmark_path=benchmark_path,
            output_dir=tmp_path / "report",
        )
