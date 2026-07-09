from pathlib import Path

import pandas as pd
import pytest

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data import MarketData, ParquetStore
from backtesting.run import BacktestRunner
from backtesting.specs import ExecutionSpec, ScheduleSpec, ShortingSpec
from backtesting.strategies import build_strategy, list_strategies


def test_team_strat1_registers_ohlcv_sell_filter_strategy() -> None:
    strategy = build_strategy("team_strat1", vol_days=3, ref_m=3)

    assert "team_strat1" in list_strategies()
    assert strategy.datasets == (
        DatasetId.QW_ADJ_O,
        DatasetId.QW_ADJ_H,
        DatasetId.QW_ADJ_L,
        DatasetId.QW_ADJ_C,
        DatasetId.QW_V,
    )


def test_team_strat1_defaults_to_requested_filter_thresholds() -> None:
    strategy = build_strategy("team_strat1")

    assert strategy.rng_min == pytest.approx(0.10)
    assert strategy.ref_min == pytest.approx(1.5)
    assert strategy.vol_pct == pytest.approx(0.9)
    assert strategy.vol_days == 250
    assert strategy.ref_m == 3
    assert strategy.min_hits == 4


def test_team_strat1_builds_short_weights_from_shooting_star_filters() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-03-28", "2024-03-29", "2024-04-01"])
    close = pd.DataFrame(
        {
            "A": [100.0, 100.0, 100.0, 145.0],
            "B": [100.0, 100.0, 100.0, 102.0],
        },
        index=index,
    )
    market = MarketData(
        frames={
            "open": pd.DataFrame({"A": [100.0, 100.0, 100.0, 150.0], "B": [100.0, 100.0, 100.0, 101.0]}, index=index),
            "high": pd.DataFrame({"A": [100.0, 101.0, 102.0, 160.0], "B": [100.0, 101.0, 102.0, 104.0]}, index=index),
            "low": pd.DataFrame({"A": [100.0, 99.0, 98.0, 140.0], "B": [100.0, 99.0, 98.0, 100.0]}, index=index),
            "close": close,
            "volume": pd.DataFrame({"A": [10.0, 20.0, 30.0, 100.0], "B": [100.0, 100.0, 100.0, 50.0]}, index=index),
        },
        universe=None,
        benchmark=None,
    )
    strategy = build_strategy("team_strat1", vol_days=3, ref_m=3, gross_short=1.0)

    signal = strategy.build_signal(market)
    weights = strategy.build_weights(market)

    assert signal.loc[index[-1], "A"] == pytest.approx(-4.0)
    assert pd.isna(signal.loc[index[-1], "B"])
    assert weights.loc[index[-1], "A"] == pytest.approx(-1.0)
    assert weights.loc[index[-1], "B"] == 0.0
    assert weights.iloc[:-1].eq(0.0).all().all()


def test_backtest_runner_resolves_and_runs_team_strat1(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-03-28", "2024-03-29", "2024-04-01"])

    store.write("qw_adj_o", pd.DataFrame({"A": [100.0, 100.0, 100.0, 150.0], "B": [100.0, 100.0, 100.0, 101.0]}, index=index))
    store.write("qw_adj_h", pd.DataFrame({"A": [100.0, 101.0, 102.0, 160.0], "B": [100.0, 101.0, 102.0, 104.0]}, index=index))
    store.write("qw_adj_l", pd.DataFrame({"A": [100.0, 99.0, 98.0, 140.0], "B": [100.0, 99.0, 98.0, 100.0]}, index=index))
    store.write("qw_adj_c", pd.DataFrame({"A": [100.0, 100.0, 100.0, 145.0], "B": [100.0, 100.0, 100.0, 102.0]}, index=index))
    store.write("qw_v", pd.DataFrame({"A": [10.0, 20.0, 30.0, 100.0], "B": [100.0, 100.0, 100.0, 50.0]}, index=index))

    runner = BacktestRunner(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
        result_dir=result_dir,
        write_report_assets=False,
    )
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-04-01",
        strategy="team_strat1",
        strategy_params={"vol_days": 3, "ref_m": 3, "gross_short": 1.0},
        schedule=ScheduleSpec(kind="named", name="daily"),
        fill_mode="close",
        use_k200=False,
        shorting=ShortingSpec(enabled=True),
    )

    resolved = runner.resolve_spec(spec)
    report = runner.run_spec(resolved)

    assert resolved.dataset_ids == (
        DatasetId.QW_ADJ_C,
        DatasetId.QW_ADJ_O,
        DatasetId.QW_ADJ_H,
        DatasetId.QW_ADJ_L,
        DatasetId.QW_V,
    )
    assert report.position_plan is not None
    assert report.position_plan.target_weights.loc["2024-04-01", "A"] == pytest.approx(-1.0)
    assert report.position_plan.target_weights.loc["2024-04-01", "B"] == 0.0
