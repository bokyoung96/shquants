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

def test_runner_run_and_run_spec_produce_identical_results_for_legacy_inputs(tmp_path: Path) -> None:
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

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    config = RunConfig(strategy="trend_rank", start="2024-01-02", end="2024-01-04", lookback=1, schedule="daily", fill_mode="close")

    legacy = runner.run(config)
    resolved = runner.run_spec(runner.resolve_spec_from_config(config))

    pd.testing.assert_series_equal(legacy.result.equity, resolved.result.equity)
    pd.testing.assert_series_equal(legacy.result.returns, resolved.result.returns)
    pd.testing.assert_series_equal(legacy.result.turnover, resolved.result.turnover)
    pd.testing.assert_frame_equal(legacy.result.qty, resolved.result.qty)
    assert legacy.summary == resolved.summary

def test_run_spec_persists_resolution_artifacts(tmp_path: Path) -> None:
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

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(start="2024-01-02", end="2024-01-04", strategy="trend_rank", lookback=1, schedule=ScheduleSpec(kind="named", name="daily"), fill_mode="close")

    report = runner.run_spec(runner.resolve_spec(spec))

    assert report.output_dir is not None
    assert (report.output_dir / "resolved_execution_spec.json").exists()
    assert (report.output_dir / "execution_resolution.json").exists()
    resolved_payload = pd.read_json(report.output_dir / "resolved_execution_spec.json", typ="series")
    resolution_payload = pd.read_json(report.output_dir / "execution_resolution.json", typ="series")
    assert resolved_payload["execution"]["spec_source"] == "cli"
    assert resolution_payload["spec_source"] == "cli"

def test_hook_runs_persist_resolved_rebalance_dates(tmp_path: Path) -> None:
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

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    resolved = ResolvedExecutionSpec(
        execution=ExecutionSpec(
            start="2024-06-12",
            end="2024-06-14",
            fill_mode="close",
            schedule=ScheduleSpec(kind="custom_dates"),
            weight_source=WeightSourceSpec(kind="hook", hook_id="kospi200_semiannual_floatcap"),
            data_policy=DataPolicySpec(
                requested_weight_basis="float_market_cap",
                resolved_weight_basis="market_cap",
                fallback_order=("market_cap",),
                fallbacks_applied=(
                    {"from": "float_market_cap", "to": "market_cap", "reason": "missing qw_mktcap_flt source"},
                ),
            ),
            spec_source="preset",
            preset_id="kospi200_semiannual_floatcap",
        ),
        dataset_ids=(),
        schedule=ScheduleSpec(kind="custom_dates"),
        hook_id="kospi200_semiannual_floatcap",
        resolution_notes=("float_market_cap unavailable; resolved to market_cap",),
    )
    resolved = ResolvedExecutionSpec(
        execution=resolved.execution,
        dataset_ids=tuple(runner.resolve_spec(resolved.execution).dataset_ids),
        schedule=resolved.schedule,
        hook_id=resolved.hook_id,
        resolution_notes=resolved.resolution_notes,
    )

    report = runner.run_spec(resolved)

    assert report.output_dir is not None
    assert report.resolved_spec is not None
    assert report.resolved_spec.schedule.dates == ("2024-06-13",)
    resolved_payload = pd.read_json(report.output_dir / "resolved_execution_spec.json", typ="series")
    resolution_payload = pd.read_json(report.output_dir / "execution_resolution.json", typ="series")
    assert tuple(resolved_payload["schedule"]["dates"]) == ("2024-06-13",)
    assert tuple(resolution_payload["rebalance_dates"]) == ("2024-06-13",)
