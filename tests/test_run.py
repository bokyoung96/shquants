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
    PositionBucketSpec,
    PositionPolicySpec,
    PositionRuleSpec,
    ResolvedExecutionSpec,
    ScheduleSpec,
    SelectionSpec,
    WeightSourceSpec,
    WeightingSpec,
)


@pytest.fixture(autouse=True)
def _stub_plot_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    def _write_placeholder_plot(path: Path, series: pd.Series, title: str, ylabel: str) -> None:
        path.write_bytes(_EMPTY_PNG)

    monkeypatch.setattr(RunWriter, "_plot_series", staticmethod(_write_placeholder_plot))


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
            strategy="momentum",
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
            strategy="momentum",
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
            strategy="momentum",
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
            strategy="momentum",
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
                strategy="momentum",
                start="2024-01-02",
                end="2024-01-02",
                schedule="daily",
                fill_mode="close",
            )
        )


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
                strategy="momentum",
                start="2024-01-05",
                end="2024-01-06",
                top_n=1,
                lookback=1,
                schedule="daily",
                fill_mode="close",
                warmup_days=3,
            )
        )


def test_root_run_delegates_to_backtesting_main() -> None:
    assert root_run.main is backtesting_main


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
            strategy="momentum",
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
            strategy="momentum",
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
    config = RunConfig(strategy="momentum", start="2024-01-02", end="2024-01-04", lookback=1, schedule="daily", fill_mode="close")

    report = runner.run(config)

    assert report.position_plan is not None
    assert report.execution_resolution is not None
    assert "plan_source" not in report.execution_resolution


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
    config = RunConfig(strategy="momentum", start="2024-01-02", end="2024-01-04", lookback=1, schedule="daily", fill_mode="close")

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
    spec = ExecutionSpec(start="2024-01-02", end="2024-01-04", strategy="momentum", lookback=1, schedule=ScheduleSpec(kind="named", name="daily"), fill_mode="close")

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


def test_run_parser_accepts_universe_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class StubRunner:
        def __init__(self, result_dir=None):
            pass

        def run_resolved_cli(self, *, preset_id=None, spec_path=None, config=None):
            observed["config"] = config
            observed["preset_id"] = preset_id
            observed["spec_path"] = spec_path
            index = pd.to_datetime(["2024-01-02"])
            result = BacktestResult(
                equity=pd.Series([1.0], index=index),
                returns=pd.Series([0.0], index=index),
                weights=pd.DataFrame({"A": [1.0]}, index=index),
                qty=pd.DataFrame({"A": [1.0]}, index=index),
                turnover=pd.Series([0.0], index=index),
            )
            return RunReport(config=config, summary={"final_equity": 1.0, "avg_turnover": 0.0}, result=result)

    monkeypatch.setattr("backtesting.run.BacktestRunner", StubRunner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--strategy",
            "momentum",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-02",
            "--universe",
            "kosdaq150",
        ],
    )

    backtesting_main()

    assert observed["config"].universe_id == "kosdaq150"


def test_run_parser_accepts_preset_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class StubRunner:
        def __init__(self, result_dir=None):
            pass

        def run_resolved_cli(self, *, preset_id=None, spec_path=None, config=None):
            observed["config"] = config
            observed["preset_id"] = preset_id
            observed["spec_path"] = spec_path
            index = pd.to_datetime(["2024-01-02"])
            result = BacktestResult(
                equity=pd.Series([1.0], index=index),
                returns=pd.Series([0.0], index=index),
                weights=pd.DataFrame({"A": [1.0]}, index=index),
                qty=pd.DataFrame({"A": [1.0]}, index=index),
                turnover=pd.Series([0.0], index=index),
            )
            return RunReport(config=RunConfig(start="2024-01-02", end="2024-01-02"), summary={"final_equity": 1.0, "avg_turnover": 0.0}, result=result)

    monkeypatch.setattr("backtesting.run.BacktestRunner", StubRunner)
    monkeypatch.setattr("sys.argv", ["run.py", "--preset", "kospi200_semiannual_floatcap"])

    backtesting_main()

    assert observed["preset_id"] == "kospi200_semiannual_floatcap"
