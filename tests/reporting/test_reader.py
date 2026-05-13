from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from backtesting.engine import BacktestResult
from backtesting.policy.base import BUCKET_LEDGER_COLUMNS, PositionPlan
from backtesting.reporting.reader import RunReader
from backtesting.reporting.writer import RunWriter, _EMPTY_PNG
from backtesting.run import RunConfig, RunReport


def _write_placeholder_plot(path: Path, series: pd.Series, title: str, ylabel: str) -> None:
    path.write_bytes(_EMPTY_PNG)


pytestmark = pytest.mark.usefixtures("_stub_plot_generation")


@pytest.fixture
def _stub_plot_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RunWriter, "_plot_series", staticmethod(_write_placeholder_plot))


def test_run_reader_returns_none_for_missing_optional_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "sample-run"
    _write_required_bundle(run_dir)

    run = RunReader().read(run_dir)

    assert run.monthly_returns is None
    assert run.latest_qty is None
    assert run.latest_weights is None
    assert run.bucket_ledger is None
    assert run.validation is None
    assert run.split is None
    assert run.factor is None
    assert run.timing is None


def test_run_reader_loads_optional_artifacts_when_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "sample-run"
    _write_required_bundle(run_dir)

    monthly_returns = pd.Series([0.05], index=pd.to_datetime(["2024-01-31"]), name="monthly_returns").rename_axis("date")
    monthly_returns.to_csv(run_dir / "series" / "monthly_returns.csv", index_label="date")

    latest_qty = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "qty": [10.0, -5.0],
            "abs_qty": [10.0, 5.0],
        }
    )
    latest_qty.to_csv(run_dir / "positions" / "latest_qty.csv", index=False)

    latest_weights = pd.DataFrame(
        {
            "symbol": ["B", "A"],
            "target_weight": [0.75, -0.25],
            "abs_weight": [0.75, 0.25],
        }
    )
    latest_weights.to_csv(run_dir / "positions" / "latest_weights.csv", index=False)

    validation = {"warnings": ["low_liquidity"], "status": "pass"}
    split = {"is": {"start": "2024-01-01", "end": "2024-01-15"}, "oos": None}
    factor = {"metrics": {"ic": 0.12}}
    timing = {"data_load": 0.01, "plan_build": 0.02, "engine_run": 0.03, "write_artifacts": 0.04, "total": 0.10}
    (run_dir / "validation.json").write_text(
        '{"warnings":["low_liquidity"],"status":"pass"}',
        encoding="utf-8",
    )
    (run_dir / "split.json").write_text(
        '{"is":{"start":"2024-01-01","end":"2024-01-15"},"oos":null}',
        encoding="utf-8",
    )
    (run_dir / "factor.json").write_text(
        '{"metrics":{"ic":0.12}}',
        encoding="utf-8",
    )
    (run_dir / "timing.json").write_text(
        '{"data_load":0.01,"plan_build":0.02,"engine_run":0.03,"write_artifacts":0.04,"total":0.1}',
        encoding="utf-8",
    )

    run = RunReader().read(run_dir)

    assert_series_equal(run.monthly_returns, monthly_returns)
    assert_frame_equal(run.latest_qty, latest_qty)
    assert_frame_equal(run.latest_weights, latest_weights)
    assert run.validation == validation
    assert run.split == split
    assert run.factor == factor
    assert run.timing == timing


def test_run_reader_round_trips_writer_bundle_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "results"
    report = _build_report()

    written_dir = RunWriter(run_dir).write(report)
    run = RunReader().read(written_dir)

    assert run.run_id == written_dir.name
    assert run.path == written_dir
    assert run.summary == report.summary
    assert_series_equal(run.equity, report.result.equity.rename("equity").rename_axis("date"))
    assert_series_equal(run.returns, report.result.returns.rename("returns").rename_axis("date"))
    assert_series_equal(run.turnover, report.result.turnover.rename("turnover").rename_axis("date"))
    assert_frame_equal(run.weights, report.result.weights)
    assert_frame_equal(run.qty, report.result.qty)
    assert_series_equal(
        run.monthly_returns,
        RunWriter._monthly_returns(report.result.returns).rename("monthly_returns").rename_axis("date"),
    )
    assert_frame_equal(
        run.latest_qty,
        pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "qty": [2.0, -5.0],
                "abs_qty": [2.0, 5.0],
            }
        ).sort_values(["abs_qty", "symbol"], ascending=[False, True]).reset_index(drop=True),
    )
    assert_frame_equal(
        run.latest_weights,
        pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "target_weight": [-0.25, 0.75],
                "abs_weight": [0.25, 0.75],
            }
        ).sort_values(["abs_weight", "symbol"], ascending=[False, True]).reset_index(drop=True),
    )
    assert_frame_equal(run.bucket_ledger, report.position_plan.bucket_ledger)
    assert run.validation == {"warnings": []}
    assert run.split == {"is": None, "oos": None}
    assert run.factor == {"metrics": {}}
    assert run.timing is None


def _write_required_bundle(run_dir: Path) -> None:
    series_dir = run_dir / "series"
    positions_dir = run_dir / "positions"
    run_dir.mkdir()
    series_dir.mkdir()
    positions_dir.mkdir()

    (run_dir / "config.json").write_text(
        '{"strategy":"trend_rank","start":"2024-01-01","end":"2024-01-31"}',
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        '{"cagr":0.1,"mdd":-0.2,"sharpe":1.2,"final_equity":110.0,"avg_turnover":0.05}',
        encoding="utf-8",
    )
    pd.Series([100.0, 110.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]), name="equity").to_csv(
        series_dir / "equity.csv",
        index_label="date",
    )
    pd.Series([0.0, 0.1], index=pd.to_datetime(["2024-01-02", "2024-01-03"]), name="returns").to_csv(
        series_dir / "returns.csv",
        index_label="date",
    )
    pd.Series([0.0, 0.05], index=pd.to_datetime(["2024-01-02", "2024-01-03"]), name="turnover").to_csv(
        series_dir / "turnover.csv",
        index_label="date",
    )
    pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-03"])).to_parquet(positions_dir / "weights.parquet")
    pd.DataFrame({"A": [10.0]}, index=pd.to_datetime(["2024-01-03"])).to_parquet(positions_dir / "qty.parquet")


def _build_report() -> RunReport:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    result = BacktestResult(
        equity=pd.Series([100.0, 110.0], index=index, name="equity"),
        returns=pd.Series([0.0, 0.1], index=index, name="returns"),
        weights=pd.DataFrame({"A": [-0.25, -0.25], "B": [0.75, 0.75]}, index=index),
        qty=pd.DataFrame({"A": [0.0, 2.0], "B": [0.0, -5.0]}, index=index),
        turnover=pd.Series([0.0, 0.05], index=index, name="turnover"),
    )
    config = RunConfig(start="2024-01-02", end="2024-01-03", strategy="trend_rank", name="writer-roundtrip")
    summary = {"cagr": 0.1, "mdd": -0.2, "sharpe": 1.2, "final_equity": 110.0, "avg_turnover": 0.05}
    position_plan = PositionPlan(
        target_weights=result.weights.copy(),
        bucket_ledger=pd.DataFrame.from_records(
            [
                {
                    "date": index[0],
                    "symbol": "A",
                    "side": "short",
                    "bucket_id": "short",
                    "stage_index": 0,
                    "target_weight": -0.25,
                    "actual_weight": -0.25,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "writer_test",
                    "construction_group": "short",
                    "budget_id": "short",
                },
                {
                    "date": index[0],
                    "symbol": "B",
                    "side": "long",
                    "bucket_id": "long",
                    "stage_index": 0,
                    "target_weight": 0.75,
                    "actual_weight": 0.75,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "writer_test",
                    "construction_group": "long",
                    "budget_id": "long",
                },
                {
                    "date": index[1],
                    "symbol": "A",
                    "side": "short",
                    "bucket_id": "short",
                    "stage_index": 0,
                    "target_weight": -0.25,
                    "actual_weight": -0.25,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "writer_test",
                    "construction_group": "short",
                    "budget_id": "short",
                },
                {
                    "date": index[1],
                    "symbol": "B",
                    "side": "long",
                    "bucket_id": "long",
                    "stage_index": 0,
                    "target_weight": 0.75,
                    "actual_weight": 0.75,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "writer_test",
                    "construction_group": "long",
                    "budget_id": "long",
                },
            ],
            columns=BUCKET_LEDGER_COLUMNS,
        ),
    )
    return RunReport(config=config, summary=summary, result=result, position_plan=position_plan)
