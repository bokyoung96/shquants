import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import run as root_run
from backtesting.catalog import DataCatalog
from backtesting.data import ParquetStore
from backtesting.engine import BacktestResult
from backtesting.policy.base import BUCKET_LEDGER_COLUMNS, PositionPlan
from backtesting.reporting.writer import RunWriter, _EMPTY_PNG
from backtesting.run import BacktestRunner, RunConfig, RunReport, main as backtesting_main
from backtesting.specs import (
    ConditionSpec,
    DataPolicySpec,
    ExecutionSpec,
    HookPlan,
    PortfolioShapeSpec,
    PositionBucketSpec,
    PositionPolicySpec,
    PositionRuleSpec,
    ResolvedExecutionSpec,
    ScheduleEvaluationSpec,
    ScheduleSpec,
    SelectionSpec,
    ShortingSpec,
    WeightSourceSpec,
    WeightingSpec,
)

def test_runner_executes_filter_equal_weight_spec_without_using_top_n(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 10.0], "B": [11.0, 11.0], "C": [9.0, 9.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1], "B": [1, 1], "C": [1, 1]}, index=index))

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-03",
        top_n=1,
        schedule=ScheduleSpec(kind="named", name="daily"),
        fill_mode="close",
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="close", op=">=", value=10.0),),
        ),
        weighting=WeightingSpec(kind="equal_weight"),
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    expected = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5], "C": [0.0, 0.0]}, index=index)
    assert_frame_equal(report.result.weights, expected)

def test_runner_executes_staged_spec_with_bucket_ledger(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 10.0], "B": [20.0, 20.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1], "B": [1, 1]}, index=index))

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-03",
        schedule=ScheduleSpec(kind="named", name="daily"),
        fill_mode="close",
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="close", op=">=", value=1.0),),
        ),
        weighting=WeightingSpec(kind="equal_weight"),
        position_policy=PositionPolicySpec(
            kind="staged",
            buckets=(PositionBucketSpec("b0", 0.25), PositionBucketSpec("b1", 0.75)),
            adds=(PositionRuleSpec("still_passes_after_rebalances", count=1),),
        ),
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    expected = pd.DataFrame({"A": [0.125, 0.5], "B": [0.125, 0.5]}, index=index)
    assert report.position_plan is not None
    assert_frame_equal(report.position_plan.target_weights, expected)
    assert not report.position_plan.bucket_ledger.empty
    assert report.output_dir is not None
    bucket_ledger = pd.read_parquet(report.output_dir / "positions" / "bucket_ledger.parquet")
    assert_frame_equal(
        bucket_ledger.reset_index(drop=True),
        report.position_plan.bucket_ledger.reset_index(drop=True),
        check_dtype=False,
    )

def test_runner_signal_dates_rebalances_only_when_target_weights_change(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    store.write(
        "qw_adj_c",
        pd.DataFrame(
            {
                "A": [10.0, 10.0, 10.0, 10.0],
                "B": [10.0, 10.0, 10.0, 10.0],
            },
            index=index,
        ),
    )
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1, 1], "B": [1, 1, 1, 1]}, index=index))

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-05",
        schedule=ScheduleSpec(kind="signal_dates", weight_change_tolerance=1e-8),
        fill_mode="close",
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="close", op=">=", value=10.0),),
        ),
        weighting=WeightingSpec(kind="explicit", path=str(tmp_path / "weights.csv")),
    )
    (tmp_path / "weights.csv").write_text(
        "date,A,B\n"
        "2024-01-02,0,0\n"
        "2024-01-03,1,0\n"
        "2024-01-04,1,0\n"
        "2024-01-05,0,1\n",
        encoding="utf-8",
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    assert list(report.result.turnover.round(8)) == [0.0, 1.0, 0.0, 2.0]

def test_runner_signal_dates_ignores_weight_changes_within_tolerance(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    store.write(
        "qw_adj_c",
        pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=index),
    )
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1]}, index=index))

    weights_path = tmp_path / "weights.csv"
    weights_path.write_text(
        "date,A\n"
        "2024-01-02,0\n"
        "2024-01-03,1\n"
        "2024-01-04,1.000000001\n",
        encoding="utf-8",
    )
    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-04",
        schedule=ScheduleSpec(kind="signal_dates", weight_change_tolerance=1e-6),
        fill_mode="close",
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="close", op=">=", value=10.0),),
        ),
        weighting=WeightingSpec(kind="explicit", path=str(weights_path)),
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    assert list(report.result.turnover.round(8)) == [0.0, 1.0, 0.0]

def test_runner_signal_dates_respects_nested_weekly_evaluation_schedule(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0] * 5}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1] * 5}, index=index))
    weights_path = tmp_path / "weights.csv"
    weights_path.write_text(
        "date,A\n"
        "2024-01-01,0\n"
        "2024-01-02,1\n"
        "2024-01-03,1\n"
        "2024-01-04,1\n"
        "2024-01-05,1\n",
        encoding="utf-8",
    )
    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-01-05",
        schedule=ScheduleSpec(
            kind="signal_dates",
            evaluation=ScheduleEvaluationSpec(kind="named", name="weekly"),
        ),
        fill_mode="close",
        selection=SelectionSpec(kind="filter", conditions=(ConditionSpec(field="close", op=">=", value=10.0),)),
        weighting=WeightingSpec(kind="explicit", path=str(weights_path)),
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    assert list(report.result.turnover.round(8)) == [0.0, 0.0, 0.0, 0.0, 1.0]

def test_runner_executes_rank_top_bottom_long_short_portfolio_shape(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    store.write(
        "qw_adj_c",
        pd.DataFrame(
            {
                "A": [5.0, 1.0],
                "B": [4.0, 0.0],
                "C": [1.0, 3.0],
                "D": [0.0, 2.0],
            },
            index=index,
        ),
    )
    store.write("qw_k200_yn", pd.DataFrame(1, index=index, columns=["A", "B", "C", "D"]))

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-03",
        schedule=ScheduleSpec(kind="named", name="daily"),
        fill_mode="close",
        selection=SelectionSpec(kind="rank_top_bottom", field="close", top_n=2, bottom_n=1),
        portfolio_shape=PortfolioShapeSpec(kind="long_short"),
        shorting=ShortingSpec(enabled=True),
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    expected = pd.DataFrame(
        {
            "A": [0.5, 0.0],
            "B": [0.5, -1.0],
            "C": [0.0, 0.5],
            "D": [-1.0, 0.5],
        },
        index=index,
    )
    assert_frame_equal(report.result.weights, expected)

def test_runner_executes_rank_top_bottom_sector_neutral_portfolio_shape(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    store.write(
        "qw_adj_c",
        pd.DataFrame(
            {
                "A": [5.0, 1.0],
                "B": [1.0, 5.0],
                "C": [4.0, 0.0],
                "D": [0.0, 4.0],
            },
            index=index,
        ),
    )
    store.write("qw_k200_yn", pd.DataFrame(1, index=index, columns=["A", "B", "C", "D"]))
    store.write(
        "qw_wics_sec_big",
        pd.DataFrame(
            {"A": ["Tech"], "B": ["Tech"], "C": ["Finance"], "D": ["Finance"]},
            index=pd.to_datetime(["2024-01-31"]),
        ),
    )

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-03",
        schedule=ScheduleSpec(kind="named", name="daily"),
        fill_mode="close",
        selection=SelectionSpec(kind="rank_top_bottom", field="close", top_n=1, bottom_n=1),
        portfolio_shape=PortfolioShapeSpec(kind="sector_neutral", group_field="sector"),
        shorting=ShortingSpec(enabled=True),
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    expected = pd.DataFrame(
        {
            "A": [0.5, -0.5],
            "B": [-0.5, 0.5],
            "C": [0.5, -0.5],
            "D": [-0.5, 0.5],
        },
        index=index,
    )
    assert_frame_equal(report.result.weights, expected)

def test_runner_applies_shorting_borrow_fee_from_execution_spec(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02"])
    store.write("qw_adj_c", pd.DataFrame({"A": [100.0], "B": [90.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1], "B": [1]}, index=index))

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-02",
        schedule=ScheduleSpec(kind="named", name="daily"),
        fill_mode="close",
        selection=SelectionSpec(kind="rank_top_bottom", field="close", top_n=1, bottom_n=1),
        portfolio_shape=PortfolioShapeSpec(kind="long_short"),
        shorting=ShortingSpec(enabled=True, borrow_fee_annual=0.252),
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    assert report.result.equity.loc["2024-01-02"] == pytest.approx(99_900_000.0)
