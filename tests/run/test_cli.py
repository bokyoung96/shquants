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

def test_root_run_delegates_to_backtesting_main() -> None:
    assert root_run.main is backtesting_main

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
            "trend_rank",
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

def test_run_parser_accepts_etf_universe_argument(monkeypatch: pytest.MonkeyPatch) -> None:
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
                weights=pd.DataFrame({"A069500": [1.0]}, index=index),
                qty=pd.DataFrame({"A069500": [1.0]}, index=index),
                turnover=pd.Series([0.0], index=index),
            )
            return RunReport(config=config, summary={"final_equity": 1.0, "avg_turnover": 0.0}, result=result)

    monkeypatch.setattr("backtesting.run.BacktestRunner", StubRunner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--strategy",
            "trend_rank",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-02",
            "--universe",
            "etf",
        ],
    )

    backtesting_main()

    assert observed["config"].universe_id == "etf"

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
