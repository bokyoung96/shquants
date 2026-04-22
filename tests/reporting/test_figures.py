from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from backtesting.reporting.analytics import (
    DrawdownStats,
    ExposureSnapshot,
    PerformanceMetrics,
    RollingMetrics,
    SectorSnapshot,
)
from backtesting.reporting.comparison_figures import ComparisonFigureBuilder
from backtesting.reporting.figures import TearsheetFigureBuilder
from backtesting.reporting.snapshots import PerformanceSnapshot


def _write_image_success(self, path, *args, **kwargs):  # type: ignore[no-untyped-def]
    Path(path).write_bytes(b"png")


def _sample_snapshot(run_id: str, scale: float = 1.0) -> PerformanceSnapshot:
    index = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
            "2024-03-31",
            "2024-04-30",
            "2024-05-31",
            "2024-06-30",
        ]
    )
    strategy_returns = pd.Series([0.02, -0.01, 0.03, 0.01, -0.02, 0.015], index=index, name="strategy_returns")
    benchmark_returns = pd.Series([0.01, -0.005, 0.02, 0.008, -0.01, 0.01], index=index, name="benchmark_returns")
    strategy_equity = pd.Series([100.0, 99.0, 101.97, 102.9897, 100.93, 102.444], index=index).mul(scale)
    benchmark_equity = pd.Series([100.0, 99.5, 101.49, 102.30192, 101.2789008, 102.291689808], index=index).mul(scale)
    underwater = strategy_equity.div(strategy_equity.cummax()).sub(1.0).rename("underwater")
    latest_holdings = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "target_weight": [0.5 * scale, -0.2 * scale, 0.1 * scale],
            "abs_weight": [0.5 * scale, 0.2 * scale, 0.1 * scale],
        }
    )
    rolling_sharpe = pd.Series([0.6, 0.4, 0.8, 0.9, 0.3, 0.7], index=index, name="rolling_sharpe")
    rolling_beta = pd.Series([1.0, 0.95, 1.05, 1.02, 0.98, 1.01], index=index, name="rolling_beta")
    holdings_count = pd.Series([3.0, 4.0, 4.0, 5.0, 4.0, 3.0], index=index, name="holdings_count")
    monthly_returns = strategy_returns.resample("ME").sum()

    return PerformanceSnapshot(
        run_id=run_id,
        display_name=f"Strategy {run_id}",
        metrics=PerformanceMetrics(
            cumulative_return=float(strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1.0),
            cagr=0.11 * scale,
            annual_volatility=0.18,
            sharpe=1.1,
            sortino=1.4,
            calmar=0.8,
            max_drawdown=float(underwater.min()),
            final_equity=float(strategy_equity.iloc[-1]),
            avg_turnover=0.12,
            alpha=0.03,
            beta=1.01,
            tracking_error=0.09,
            information_ratio=0.5,
        ),
        rolling=RollingMetrics(
            window=252,
            series={"rolling_sharpe": rolling_sharpe, "rolling_beta": rolling_beta},
        ),
        drawdowns=DrawdownStats(
            underwater=underwater,
            episodes=pd.DataFrame(
                {
                    "start": [index[1]],
                    "trough": [index[1]],
                    "end": [index[2]],
                    "drawdown": [float(underwater.iloc[1])],
                }
            ),
        ),
        exposure=ExposureSnapshot(holdings_count=holdings_count, latest_holdings=latest_holdings),
        sectors=SectorSnapshot(
            latest_weighted=pd.Series({"Tech": 0.5 * scale, "Utilities": 0.3 * scale, "Energy": -0.2 * scale}),
            latest_count=pd.Series({"Tech": 2.0, "Utilities": 1.0, "Energy": 1.0}),
            concentration=pd.Series({"Tech": 0.5 * scale, "Utilities": 0.3 * scale, "Energy": 0.2 * scale}),
        ),
        strategy_equity=strategy_equity.rename("strategy_equity"),
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        benchmark_equity=benchmark_equity.rename("benchmark_equity"),
    )


def test_tearsheet_figure_builder_writes_expected_page_assets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(go.Figure, "write_image", _write_image_success)

    assets = TearsheetFigureBuilder(tmp_path).build(_sample_snapshot("alpha"))

    assert assets.keys() == {"executive", "rolling", "calendar", "exposure"}
    for path in assets.values():
        assert path.exists()
        assert path.suffix == ".png"


def test_comparison_figure_builder_writes_expected_page_assets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(go.Figure, "write_image", _write_image_success)

    snapshots = [_sample_snapshot("alpha", 1.0), _sample_snapshot("beta", 1.2)]
    assets = ComparisonFigureBuilder(tmp_path).build(snapshots)

    assert assets.keys() == {"executive", "performance", "rolling", "exposure"}
    for path in assets.values():
        assert path.exists()
        assert path.suffix == ".png"
