from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtesting.strategies.emp008.model_comparison_report import (
    ModelReportInput,
    _report_years,
    build_emp008_model_comparison_report,
    performance_metrics,
)


def test_performance_metrics_reports_compounded_return_and_drawdown() -> None:
    returns = pd.Series(
        [0.10, -0.20, 0.10],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    metrics = performance_metrics(returns)

    assert metrics["total_return"] == pytest.approx((1.10 * 0.80 * 1.10) - 1.0)
    assert metrics["max_drawdown"] == pytest.approx(-0.20)
    assert metrics["positive_day_rate"] == pytest.approx(2.0 / 3.0)


def test_performance_metrics_uses_252_trading_days_for_annualization() -> None:
    dates = pd.bdate_range("2023-01-02", periods=252)
    returns = pd.Series(0.0, index=dates)
    returns.iloc[-1] = 0.10

    assert performance_metrics(returns)["cagr"] == pytest.approx(0.10)


def test_report_years_drops_single_observation_stub_year() -> None:
    dates = pd.to_datetime(["2022-12-29", "2023-01-02", "2023-01-03"])

    assert _report_years(dates) == [2023]


def test_build_emp008_model_comparison_report_writes_requested_artifacts(tmp_path: Path) -> None:
    dates = pd.bdate_range("2023-01-02", "2024-12-31")
    modified = _write_run(
        tmp_path / "modified",
        dates,
        returns=pd.Series(0.0008, index=dates),
        latest={"A": 0.65, "B": 0.35},
    )
    original = _write_run(
        tmp_path / "original",
        dates,
        returns=pd.Series(0.0005, index=dates),
        latest={"A": 0.30, "C": 0.70},
    )
    close = pd.DataFrame(
        {
            "A": 100.0 * (1.001 ** pd.RangeIndex(len(dates))),
            "B": 100.0 * (1.002 ** pd.RangeIndex(len(dates))),
            "C": 100.0 * (0.999 ** pd.RangeIndex(len(dates))),
        },
        index=dates,
    )
    close_path = tmp_path / "adj_close.parquet"
    close.to_parquet(close_path)
    sector = pd.DataFrame({"A": "반도체", "B": "산업재", "C": "금융"}, index=dates)
    sector_path = tmp_path / "sector.parquet"
    sector.to_parquet(sector_path)
    output_dir = tmp_path / "report"

    payload = build_emp008_model_comparison_report(
        modified=ModelReportInput(
            label="수정EMP008",
            factor_set="mfbt",
            risk_model="direct_covariance",
            tracking_error_annual=0.10,
            gross_run_dir=modified,
            net_run_dir=modified,
        ),
        original=ModelReportInput(
            label="기존EMP008",
            factor_set="origin",
            risk_model="factor_idio",
            tracking_error_annual=0.007,
            gross_run_dir=original,
            net_run_dir=original,
        ),
        adjusted_close_path=close_path,
        sector_path=sector_path,
        output_dir=output_dir,
        cost_assumptions={"fee": 0.0002, "sell_tax": 0.0015, "slippage": 0.0005},
    )

    required = {
        "report_html",
        "report_json",
        "metrics_csv",
        "yearly_returns_csv",
        "latest_positions_csv",
        "latest_sector_exposure_csv",
        "performance_gap_contributors_csv",
        "cumulative_mdd_png",
        "yearly_cumulative_subplots_png",
        "return_distribution_png",
        "latest_positions_png",
    }
    assert required <= payload.keys()
    for key in required:
        assert Path(payload[key]).exists(), key
    html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "수정EMP008" in html
    assert "기존EMP008" in html
    assert "최근 포지션과 성과 차이의 이유" in html
    positions = pd.read_csv(payload["latest_positions_csv"])
    assert set(positions["status"]) == {"공통편입", "수정EMP008 전용", "기존EMP008 전용"}


def _write_run(
    root: Path,
    dates: pd.DatetimeIndex,
    *,
    returns: pd.Series,
    latest: dict[str, float],
) -> Path:
    (root / "series").mkdir(parents=True)
    (root / "positions").mkdir(parents=True)
    pd.DataFrame({"date": dates, "returns": returns.to_numpy()}).to_csv(root / "series" / "returns.csv", index=False)
    pd.DataFrame({"date": dates, "turnover": [0.0] * (len(dates) - 1) + [0.2]}).to_csv(
        root / "series" / "turnover.csv", index=False
    )
    latest_frame = pd.DataFrame(
        {"symbol": list(latest), "target_weight": list(latest.values())}
    )
    latest_frame["abs_weight"] = latest_frame["target_weight"].abs()
    latest_frame.to_csv(root / "positions" / "latest_weights.csv", index=False)
    weights = pd.DataFrame(0.0, index=dates, columns=["A", "B", "C"])
    for symbol, weight in latest.items():
        weights[symbol] = weight
    weights.to_parquet(root / "positions" / "weights.parquet")
    return root
