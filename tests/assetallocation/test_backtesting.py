import json
from pathlib import Path

import pandas as pd

from assetallocation.backtesting.engine import TwoAssetBacktester
from assetallocation.backtesting.report import BacktestReportWriter


def _weights(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY US Equity": [1.0, 0.0, 1.0, 0.5],
            "IEF US Equity": [0.0, 1.0, 0.0, 0.5],
        },
        index=index,
    )


def _daily_returns(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY US Equity": [0.00, 0.10, -0.10, 0.20],
            "IEF US Equity": [0.00, 0.01, 0.02, 0.03],
        },
        index=index,
    )


def test_two_asset_backtester_applies_weights_to_next_day_returns() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D", name="date")

    result = TwoAssetBacktester().run(_weights(index), _daily_returns(index))

    expected_returns = pd.Series([0.0995, 0.019, 0.199], index=index[:3], name="portfolio_return")
    pd.testing.assert_series_equal(result.returns, expected_returns)
    gross_returns = pd.Series([0.10, 0.02, 0.20], index=index[:3], name="gross_portfolio_return")
    pd.testing.assert_series_equal(result.gross_returns, gross_returns)
    assert result.costs.iloc[0] == 0.0005
    assert result.costs.iloc[1] == 0.001
    assert result.equity.iloc[-1] == 1.0995 * 1.019 * 1.199
    assert result.drawdown.iloc[0] == 0.0
    assert result.benchmark_returns.iloc[0] == 0.75 * 0.10 + 0.25 * 0.01 - 0.0005
    assert result.benchmark_weights.to_dict() == {"SPY US Equity": 0.75, "IEF US Equity": 0.25}
    assert result.metrics["transaction_cost_bps"] == 5.0
    assert result.metrics["total_transaction_cost"] > 0.0
    assert result.metrics["observations"] == 3
    assert "benchmark_total_return" in result.metrics
    assert "active_total_return" in result.metrics
    assert result.metrics["turnover"] > 0.0


def test_backtest_report_writer_outputs_performance_and_graphs(tmp_path: Path) -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D", name="date")
    result = TwoAssetBacktester().run(_weights(index), _daily_returns(index))

    output = BacktestReportWriter(tmp_path).write("ridge", result)

    assert output.metrics_path.exists()
    assert output.returns_path.exists()
    assert output.equity_path.exists()
    assert output.drawdown_path.exists()
    assert output.weights_path.exists()
    assert output.spec_path.exists()
    assert output.summary_plot_path.exists()
    assert output.equity_plot_path.exists()
    assert output.drawdown_plot_path.exists()
    assert output.weights_plot_path.exists()
    assert json.loads(output.metrics_path.read_text())["observations"] == 3
    spec = json.loads(output.spec_path.read_text())
    assert spec["benchmark_weights"] == {"SPY US Equity": 0.75, "IEF US Equity": 0.25}
    assert spec["transaction_cost_bps"] == 5.0
    assert spec["rebalance"] == "weekly W-FRI model weight, forward-filled daily and applied to next-day returns"
    assert spec["strategy_weight_bounds"] == {"SPY US Equity": [0.0, 1.0], "IEF US Equity": [0.0, 1.0]}
    equity = pd.read_parquet(output.equity_path, engine="pyarrow")
    drawdown = pd.read_parquet(output.drawdown_path, engine="pyarrow")
    assert equity.columns.tolist() == ["strategy_net", "strategy_gross", "benchmark_net"]
    assert drawdown.columns.tolist() == ["strategy_net", "strategy_gross", "benchmark_net"]
