from __future__ import annotations

import json

import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from backtest.run_backtest import run_backtest
from backtest.spec import BacktestSpec, CostModel


def test_engine_uses_next_open_fill_and_schedule_dates() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    close = pd.DataFrame({"A": [100.0, 110.0, 120.0, 120.0]}, index=index)
    open_ = pd.DataFrame({"A": [95.0, 100.0, 115.0, 120.0]}, index=index)
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0]}, index=index)
    schedule = pd.Series([True, False, True, False], index=index, dtype=bool)

    result = BacktestEngine(cost=CostModel()).run(
        close=close,
        open_=open_,
        weights=weights,
        capital=1_000.0,
        schedule=schedule,
        fill_mode="next_open",
    )

    assert result.equity.tolist() == pytest.approx([1_000.0, 1_100.0, 1_200.0, 1_200.0])
    assert result.qty["A"].tolist() == pytest.approx([0.0, 10.0, 10.0, 0.0])
    assert result.turnover.tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_engine_applies_costs_and_disables_fractional_shares() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    close = pd.DataFrame({"A": [42.0, 42.0]}, index=index)
    weights = pd.DataFrame({"A": [1.0, 0.0]}, index=index)

    result = BacktestEngine(
        cost=CostModel(fee=0.01, sell_tax=0.02, slippage=0.005),
    ).run(
        close=close,
        weights=weights,
        capital=100.0,
        fill_mode="close",
        allow_fractional=False,
    )

    assert result.qty.loc["2024-01-02", "A"] == 2.0
    assert result.equity.loc["2024-01-02"] == pytest.approx(98.74)
    assert result.equity.loc["2024-01-03"] == pytest.approx(95.8)
    assert result.turnover.tolist() == pytest.approx([0.84, 0.850719])


def test_run_backtest_writes_local_artifacts(tmp_path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "results"
    data_dir.mkdir()

    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    close = pd.DataFrame(
        {
            "A": [100.0, 110.0, 110.0],
            "B": [100.0, 100.0, 100.0],
        },
        index=index,
    )
    close.to_parquet(data_dir / "qw_adj_c.parquet")

    weights = pd.DataFrame(
        {
            "A": [0.5, 0.0],
            "B": [0.5, 1.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    weights_path = tmp_path / "target_weights.csv"
    weights.to_csv(weights_path)

    report = run_backtest(
        BacktestSpec(
            name="emp008_unit",
            data_dir=data_dir,
            output_dir=output_dir,
            weights_csv=weights_path,
            start="2024-01-02",
            end="2024-01-04",
            fill_mode="close",
            capital=1_000.0,
        )
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    returns = pd.read_csv(output_dir / "series" / "returns.csv", parse_dates=["date"]).set_index("date")

    assert report.result.equity.tolist() == pytest.approx([1_000.0, 1_050.0, 1_050.0])
    assert summary["config"]["fill_mode"] == "close"
    assert summary["rows"] == 3
    assert returns["returns"].tolist() == pytest.approx([0.0, 0.05, 0.0])
    assert (output_dir / "series" / "equity.csv").exists()
    assert (output_dir / "series" / "qty.parquet").exists()
