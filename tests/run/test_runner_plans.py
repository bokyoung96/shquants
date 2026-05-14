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

def test_runner_executes_strategy_plan_and_stores_position_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    close = pd.DataFrame({"A": [10.0, 10.0], "B": [20.0, 20.0]}, index=index)
    store.write("qw_adj_c", close)
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1], "B": [1, 1]}, index=index))

    plan = PositionPlan(
        target_weights=pd.DataFrame({"A": [0.75, 0.25], "B": [0.25, 0.75]}, index=index),
        bucket_ledger=pd.DataFrame.from_records(
            [
                {
                    "date": index[0],
                    "symbol": "A",
                    "side": "long",
                    "bucket_id": "base",
                    "stage_index": 0,
                    "target_weight": 0.75,
                    "actual_weight": 0.75,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "manual_plan",
                    "construction_group": None,
                    "budget_id": "base",
                },
                {
                    "date": index[0],
                    "symbol": "B",
                    "side": "long",
                    "bucket_id": "base",
                    "stage_index": 0,
                    "target_weight": 0.25,
                    "actual_weight": 0.25,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "manual_plan",
                    "construction_group": None,
                    "budget_id": "base",
                },
                {
                    "date": index[1],
                    "symbol": "A",
                    "side": "long",
                    "bucket_id": "base",
                    "stage_index": 0,
                    "target_weight": 0.25,
                    "actual_weight": 0.25,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "manual_plan",
                    "construction_group": None,
                    "budget_id": "base",
                },
                {
                    "date": index[1],
                    "symbol": "B",
                    "side": "long",
                    "bucket_id": "base",
                    "stage_index": 0,
                    "target_weight": 0.75,
                    "actual_weight": 0.75,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "manual_plan",
                    "construction_group": None,
                    "budget_id": "base",
                },
            ],
            columns=BUCKET_LEDGER_COLUMNS,
        ),
    )

    class StrategyStub:
        datasets: tuple = ()

        def build_plan(self, market) -> PositionPlan:
            return plan

        def build_weights(self, market) -> pd.DataFrame:
            raise AssertionError("runner should execute the position plan, not build_weights")

    monkeypatch.setattr("backtesting.run.build_strategy", lambda *args, **kwargs: StrategyStub())

    runner = BacktestRunner(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
        result_dir=result_dir,
    )
    report = runner.run(
        RunConfig(
            strategy="trend_rank",
            start="2024-01-02",
            end="2024-01-03",
            schedule="daily",
            fill_mode="close",
        )
    )

    assert report.position_plan is plan
    assert_frame_equal(report.result.weights, plan.target_weights)

def test_runner_rejects_invalid_position_plan_before_engine_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02"])
    close = pd.DataFrame({"A": [10.0], "B": [20.0]}, index=index)
    store.write("qw_adj_c", close)
    store.write("qw_k200_yn", pd.DataFrame({"A": [1], "B": [1]}, index=index))

    invalid_plan = PositionPlan(
        target_weights=pd.DataFrame({"A": [0.60], "B": [0.40]}, index=index),
        bucket_ledger=pd.DataFrame.from_records(
            [
                {
                    "date": index[0],
                    "symbol": "A",
                    "side": "long",
                    "bucket_id": "base",
                    "stage_index": 0,
                    "target_weight": 0.90,
                    "actual_weight": 0.90,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "manual_plan",
                    "construction_group": None,
                    "budget_id": "base",
                }
            ],
            columns=BUCKET_LEDGER_COLUMNS,
        ),
    )

    class StrategyStub:
        datasets: tuple = ()

        def build_plan(self, market) -> PositionPlan:
            return invalid_plan

    monkeypatch.setattr("backtesting.run.build_strategy", lambda *args, **kwargs: StrategyStub())

    def _fail_if_engine_runs(*args, **kwargs):
        raise AssertionError("engine should not run when the position plan is invalid")

    monkeypatch.setattr("backtesting.run.BacktestEngine.run", _fail_if_engine_runs)

    runner = BacktestRunner(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
        result_dir=result_dir,
    )

    with pytest.raises(ValueError, match="bucket target_weight values do not match plan target_weights"):
        runner.run(
            RunConfig(
                strategy="trend_rank",
                start="2024-01-02",
                end="2024-01-02",
                schedule="daily",
                fill_mode="close",
            )
        )

def test_runner_prefers_hook_plan_over_composable_fields_when_both_are_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-06-06", "2024-06-12", "2024-06-13", "2024-06-14"])
    close = pd.DataFrame({"A": [9.0, 10.0, 11.0, 12.0]}, index=index)
    store.write("qw_adj_c", close)
    store.write("qw_mktcap", pd.DataFrame({"A": [90.0, 100.0, 120.0, 130.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1, 1]}, index=index))

    hook_plan = PositionPlan(
        target_weights=pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0]}, index=index),
        bucket_ledger=pd.DataFrame.from_records(
            [
                {
                    "date": date,
                    "symbol": "A",
                    "side": "long",
                    "bucket_id": "hook_bucket",
                    "stage_index": 0,
                    "target_weight": weight,
                    "actual_weight": weight,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "manual_plan",
                    "construction_group": None,
                    "budget_id": "base",
                }
                for date, weight in zip(index, [1.0, 0.0, 1.0, 0.0], strict=True)
            ],
            columns=BUCKET_LEDGER_COLUMNS,
        ),
    )

    class HookStub:
        def build_plan(self, *, market, resolved_spec, universe_spec):
            return HookPlan(position_plan=hook_plan, schedule=None, tradable=None, metadata={"plan_source": "hook_stub"})

    monkeypatch.setattr("backtesting.run.get_hook", lambda hook_id: HookStub())

    def _fail_if_composable_builder_runs(*args, **kwargs):
        raise AssertionError("composable builder should not run when weight_source.kind='hook'")

    monkeypatch.setattr("backtesting.run.build_position_plan_from_execution_spec", _fail_if_composable_builder_runs)

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(
        start="2024-06-06",
        end="2024-06-14",
        schedule=ScheduleSpec(kind="named", name="daily"),
        fill_mode="close",
        weight_source=WeightSourceSpec(kind="hook", hook_id="kospi200_semiannual_floatcap"),
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="close", op=">=", value=11.0),),
        ),
        weighting=WeightingSpec(kind="equal_weight"),
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    assert report.position_plan is hook_plan
    assert_frame_equal(report.result.weights, hook_plan.target_weights.loc[spec.start:spec.end])
    assert report.execution_resolution is not None
    assert report.execution_resolution["plan_source"] == "hook_stub"

def test_runner_legacy_inputs_do_not_invoke_composable_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=index))
    store.write("qw_adj_o", pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1]}, index=index))

    def _fail_if_composable_builder_runs(*args, **kwargs):
        raise AssertionError("legacy strategy inputs should not invoke composable builder")

    monkeypatch.setattr("backtesting.run.build_position_plan_from_execution_spec", _fail_if_composable_builder_runs)

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    config = RunConfig(strategy="trend_rank", start="2024-01-02", end="2024-01-04", lookback=1, schedule="daily", fill_mode="close")

    report = runner.run(config)

    assert report.position_plan is not None
    assert report.execution_resolution is not None
    assert "plan_source" not in report.execution_resolution
