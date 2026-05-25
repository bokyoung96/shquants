import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import run as root_run
from backtesting.calculation import _strategy_kwargs, _validate_shorting_enabled_for_plan
from backtesting.catalog import DataCatalog
from backtesting.data import ParquetStore
from backtesting.engine import BacktestResult
from backtesting.policy.base import BUCKET_LEDGER_COLUMNS, PositionPlan
from backtesting.reporting.writer import RunWriter, _EMPTY_PNG
from backtesting.run import BacktestRunner, RunConfig, RunReport, _parse_strategy_params, main as backtesting_main
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
    TargetWeightsSpec,
    WeightSourceSpec,
    WeightingSpec,
)

def test_runner_maps_run_config_shorting_costs_to_execution_spec(tmp_path: Path) -> None:
    runner = BacktestRunner(parquet_dir=tmp_path / "parquet", raw_dir=tmp_path / "raw", result_dir=tmp_path / "results")

    resolved_spec = runner.resolve_spec_from_config(
        RunConfig(
            start="2024-01-02",
            end="2024-01-04",
            borrow_fee_annual=0.252,
            short_cash_collateral_ratio=1.25,
        )
    )

    assert resolved_spec.execution.shorting.enabled is True
    assert resolved_spec.execution.shorting.borrow_fee_annual == 0.252
    assert resolved_spec.execution.shorting.cash_collateral_ratio == 1.25

def test_runner_maps_run_config_strategy_params_to_execution_spec(tmp_path: Path) -> None:
    runner = BacktestRunner(parquet_dir=tmp_path / "parquet", raw_dir=tmp_path / "raw", result_dir=tmp_path / "results")

    resolved_spec = runner.resolve_spec_from_config(
        RunConfig(
            strategy="benchmark_tilt",
            strategy_params={
                "active_share_target": 0.25,
                "max_stock_active": 0.08,
                "min_names": 10,
            },
            start="2024-01-02",
            end="2024-01-04",
        )
    )

    assert resolved_spec.execution.strategy_params == {
        "active_share_target": 0.25,
        "max_stock_active": 0.08,
        "min_names": 10,
    }

def test_parse_strategy_params_parses_json_values() -> None:
    assert _parse_strategy_params(
        [
            "bottom_n=9",
            "gross_short=0.75",
            "label=\"core\"",
            "flag=true",
        ]
    ) == {
        "bottom_n": 9,
        "gross_short": 0.75,
        "label": "core",
        "flag": True,
    }

def test_calculation_strategy_kwargs_include_strategy_params() -> None:
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-04",
        strategy="benchmark_tilt",
        top_n=7,
        flow_lookback=11,
        strategy_params={
            "active_share_target": 0.25,
            "max_stock_active": 0.08,
            "min_names": 10,
        },
    )

    assert _strategy_kwargs(spec) == {
        "top_n": 7,
        "lookback": 20,
        "flow_lookback": 11,
        "momentum_lookback": 60,
        "liquidity_lookback": 20,
        "momentum_weight": 0.5,
        "active_share_target": 0.25,
        "max_stock_active": 0.08,
        "min_names": 10,
    }

def test_position_plan_shorting_gate_rejects_negative_weights_without_shorting_enabled() -> None:
    index = pd.to_datetime(["2024-01-02"])
    plan = PositionPlan(
        target_weights=pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=index),
        bucket_ledger=pd.DataFrame(),
        bucket_meta=pd.DataFrame(),
        validation={},
    )
    spec = ExecutionSpec(start="2024-01-02", end="2024-01-02", shorting=ShortingSpec(enabled=False))

    with pytest.raises(ValueError, match="shorting.enabled"):
        _validate_shorting_enabled_for_plan(spec, plan)

def test_position_plan_shorting_gate_allows_negative_weights_when_shorting_enabled() -> None:
    index = pd.to_datetime(["2024-01-02"])
    plan = PositionPlan(
        target_weights=pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=index),
        bucket_ledger=pd.DataFrame(),
        bucket_meta=pd.DataFrame(),
        validation={},
    )
    spec = ExecutionSpec(start="2024-01-02", end="2024-01-02", shorting=ShortingSpec(enabled=True))

    _validate_shorting_enabled_for_plan(spec, plan)

def test_runner_executes_momentum_strategy(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 10.0, 9.0]}, index=index))
    store.write("qw_adj_o", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 10.0, 9.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1], "B": [1, 1, 1]}, index=index))

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
            end="2024-01-04",
            top_n=1,
            lookback=1,
            schedule="daily",
            fill_mode="close",
        )
    )

    assert report.summary["final_equity"] > 0.0
    assert report.config.use_k200 is True
    assert report.config.universe_id is None
    assert report.config.benchmark_name == "KOSPI200"
    assert report.result.weights.loc["2024-01-04", "A"] == 1.0
    assert report.output_dir is not None
    assert (report.output_dir / "config.json").exists()
    assert (report.output_dir / "summary.json").exists()
    assert (report.output_dir / "series" / "equity.csv").exists()
    assert (report.output_dir / "series" / "monthly_returns.csv").exists()
    assert (report.output_dir / "positions" / "latest_qty.csv").exists()
    assert (report.output_dir / "plots" / "equity.png").exists()
    assert (report.output_dir / "plots" / "drawdown.png").exists()
    assert (report.output_dir / "pages" / "performance.png").exists()
    assert (report.output_dir / "validation.json").exists()
    assert (report.output_dir / "split.json").exists()
    assert (report.output_dir / "factor.json").exists()


def test_runner_allows_exits_after_symbol_leaves_universe(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 10.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 0]}, index=index))
    weights_path = tmp_path / "weights.csv"
    weights_path.write_text(
        "date,A\n"
        "2024-01-02,1\n"
        "2024-01-03,0\n",
        encoding="utf-8",
    )

    runner = BacktestRunner(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
        result_dir=result_dir,
        write_report_assets=False,
    )
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-03",
        capital=100.0,
        target_weights=TargetWeightsSpec(kind="file", path=str(weights_path)),
        schedule=ScheduleSpec(kind="named", name="daily"),
        fill_mode="close",
        use_k200=True,
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    assert report.result.qty.loc["2024-01-02", "A"] == 10.0
    assert report.result.qty.loc["2024-01-03", "A"] == 0.0
    assert report.result.turnover.loc["2024-01-03"] == 1.0


def test_runner_can_skip_heavy_report_assets_for_dashboard_launch(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 10.0, 9.0]}, index=index))
    store.write("qw_adj_o", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 10.0, 9.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1], "B": [1, 1, 1]}, index=index))

    runner = BacktestRunner(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
        result_dir=result_dir,
        write_report_assets=False,
    )
    report = runner.run(
        RunConfig(
            strategy="trend_rank",
            start="2024-01-02",
            end="2024-01-04",
            top_n=1,
            lookback=1,
            schedule="daily",
            fill_mode="close",
        )
    )

    assert report.output_dir is not None
    assert (report.output_dir / "config.json").exists()
    assert (report.output_dir / "summary.json").exists()
    assert (report.output_dir / "series" / "equity.csv").exists()
    assert (report.output_dir / "series" / "returns.csv").exists()
    assert (report.output_dir / "series" / "turnover.csv").exists()
    assert (report.output_dir / "positions" / "weights.parquet").exists()
    assert (report.output_dir / "positions" / "qty.parquet").exists()
    assert not (report.output_dir / "plots").exists()
    assert not (report.output_dir / "pages").exists()

def test_runner_profiles_and_persists_backtest_timing(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 10.0, 9.0]}, index=index))
    store.write("qw_adj_o", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 10.0, 9.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1], "B": [1, 1, 1]}, index=index))

    runner = BacktestRunner(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
        result_dir=result_dir,
        write_report_assets=False,
        profile=True,
    )
    report = runner.run(
        RunConfig(
            strategy="trend_rank",
            start="2024-01-02",
            end="2024-01-04",
            top_n=1,
            lookback=1,
            schedule="daily",
            fill_mode="close",
        )
    )

    assert report.output_dir is not None
    assert report.timing is not None
    assert set(report.timing) == {"data_load", "plan_build", "engine_run", "write_artifacts", "total"}
    assert all(value >= 0.0 for value in report.timing.values())
    assert report.timing["total"] >= report.timing["write_artifacts"]
    assert json.loads((report.output_dir / "timing.json").read_text(encoding="utf-8")) == report.timing

def test_runner_persists_implicit_legacy_universe_as_none(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 10.0, 9.0]}, index=index))
    store.write("qw_adj_o", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 10.0, 9.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1], "B": [1, 1, 1]}, index=index))

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
            end="2024-01-04",
            top_n=1,
            lookback=1,
            schedule="daily",
            fill_mode="close",
        )
    )

    assert report.config.universe_id is None
    assert report.output_dir is not None
    persisted_config = pd.read_json(report.output_dir / "config.json", typ="series")
    assert persisted_config["universe_id"] is None

def test_runner_uses_warmup_history_but_trims_persisted_outputs(
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
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 9.0, 8.0]}, index=index))
    store.write("qw_adj_o", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 9.0, 8.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1], "B": [1, 1, 1]}, index=index))

    plan = PositionPlan(
        target_weights=pd.DataFrame({"A": [0.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0]}, index=index),
        bucket_ledger=pd.DataFrame.from_records(
            [
                {
                    "date": date,
                    "symbol": "A",
                    "side": "long",
                    "bucket_id": "base",
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
                for date, weight in zip(index, [0.0, 1.0, 1.0], strict=True)
            ],
            columns=BUCKET_LEDGER_COLUMNS,
        ),
    )

    class StrategyStub:
        datasets: tuple = ()

        def build_plan(self, market) -> PositionPlan:
            return plan

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
            start="2024-01-03",
            end="2024-01-04",
            schedule="daily",
            fill_mode="close",
            warmup_days=1,
        )
    )
    assert report.result.weights.index.min() == pd.Timestamp("2024-01-03")
    assert report.result.weights.loc["2024-01-03", "A"] == 1.0
    assert report.output_dir is not None
    assert report.position_plan is not None
    assert report.position_plan.target_weights.index.tolist() == list(pd.to_datetime(["2024-01-03", "2024-01-04"]))
    assert report.position_plan.bucket_ledger["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-03", "2024-01-04"]

    equity = pd.read_csv(report.output_dir / "series" / "equity.csv")
    assert equity["date"].tolist() == ["2024-01-03", "2024-01-04"]

    bucket_ledger = pd.read_parquet(report.output_dir / "positions" / "bucket_ledger.parquet")
    assert bucket_ledger["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-03", "2024-01-04"]

def test_runner_raises_clear_error_when_trimmed_display_range_is_empty(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 9.0, 8.0]}, index=index))
    store.write("qw_adj_o", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [10.0, 9.0, 8.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1], "B": [1, 1, 1]}, index=index))

    runner = BacktestRunner(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
        result_dir=result_dir,
    )

    with pytest.raises(ValueError, match="no backtest rows remain after trimming to display range"):
        runner.run(
            RunConfig(
                strategy="trend_rank",
                start="2024-01-05",
                end="2024-01-06",
                top_n=1,
                lookback=1,
                schedule="daily",
                fill_mode="close",
                warmup_days=3,
            )
        )

def test_runner_uses_kosdaq_universe_specific_datasets(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    store.write("qw_adj_c", pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=index))
    store.write("qw_ksdq_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=index))
    store.write("qw_ksdq150_yn", pd.DataFrame({"A": [1, 1, 1], "B": [0, 0, 0]}, index=index))

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
            end="2024-01-04",
            lookback=1,
            schedule="daily",
            fill_mode="close",
            universe_id="kosdaq150",
        )
    )

    assert report.config.universe_id == "kosdaq150"
    assert report.config.use_k200 is False
    assert report.config.benchmark_name == "KOSDAQ150"
    assert report.result.equity.index[-1].isoformat() == "2024-01-04T00:00:00"

def test_runner_uses_kosdaq_default_next_open_path(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    store.write("qw_ksdq_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [9.0, 9.0, 9.0]}, index=index))
    store.write("qw_ksdq_adj_o", pd.DataFrame({"A": [9.5, 10.5, 11.5], "B": [9.0, 9.0, 9.0]}, index=index))
    store.write("qw_ksdq150_yn", pd.DataFrame({"A": [1, 1, 1], "B": [0, 0, 0]}, index=index))

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
            end="2024-01-04",
            lookback=1,
            schedule="daily",
            universe_id="kosdaq150",
        )
    )

    assert report.config.universe_id == "kosdaq150"
    assert report.config.use_k200 is False
    assert report.config.benchmark_name == "KOSDAQ150"
    assert report.result.equity.index[-1].isoformat() == "2024-01-04T00:00:00"

def test_runner_uses_etf_universe_specific_datasets(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    store.write("qw_adj_c", pd.DataFrame({"A069500": [1.0, 1.0, 1.0]}, index=index))
    store.write("qw_etf_adj_c", pd.DataFrame({"A069500": [35000.0, 35100.0, 35200.0]}, index=index))

    runner = BacktestRunner(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
        result_dir=result_dir,
        write_report_assets=False,
    )
    report = runner.run(
        RunConfig(
            strategy="trend_rank",
            start="2024-01-02",
            end="2024-01-04",
            lookback=1,
            schedule="daily",
            fill_mode="close",
            universe_id="etf",
        )
    )

    assert report.config.universe_id == "etf"
    assert report.config.use_k200 is False
    assert report.config.benchmark_name == "KOSPI200"
    assert report.result.equity.index[-1].isoformat() == "2024-01-04T00:00:00"
