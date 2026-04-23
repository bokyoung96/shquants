from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from backtesting.reporting.analytics import (
    DrawdownStats,
    ExposureSnapshot,
    PerformanceMetrics,
    ResearchSnapshot,
    RollingMetrics,
    SectorSnapshot,
)
from backtesting.reporting.comparison_figures import ComparisonFigureBuilder
from backtesting.reporting.figures import TearsheetFigureBuilder, write_figure_asset
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
    monthly_heatmap = pd.DataFrame({1: [0.02], 2: [-0.01], 3: [0.03], 4: [0.01], 5: [-0.02], 6: [0.015]}, index=[2024])
    yearly_returns = pd.Series([float((1.0 + strategy_returns).prod() - 1.0)], index=pd.to_datetime(["2024-12-31"]))

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
            latest_weighted=pd.Series({"G20": 0.5 * scale, "G45": 0.3 * scale, "G10": -0.2 * scale}),
            latest_count=pd.Series({"G20": 2.0, "G45": 1.0, "G10": 1.0}),
            concentration=pd.Series({"G20": 0.5 * scale, "G45": 0.3 * scale, "G10": 0.2 * scale}),
        ),
        strategy_equity=strategy_equity.rename("strategy_equity"),
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        benchmark_equity=benchmark_equity.rename("benchmark_equity"),
        research=ResearchSnapshot(monthly_heatmap=monthly_heatmap, yearly_returns=yearly_returns),
    )


def _sample_snapshot_without_benchmark(run_id: str) -> PerformanceSnapshot:
    snapshot = _sample_snapshot(run_id)
    return PerformanceSnapshot(
        run_id=snapshot.run_id,
        display_name=snapshot.display_name,
        metrics=PerformanceMetrics(
            cumulative_return=snapshot.metrics.cumulative_return,
            cagr=snapshot.metrics.cagr,
            annual_volatility=snapshot.metrics.annual_volatility,
            sharpe=snapshot.metrics.sharpe,
            sortino=snapshot.metrics.sortino,
            calmar=snapshot.metrics.calmar,
            max_drawdown=snapshot.metrics.max_drawdown,
            final_equity=snapshot.metrics.final_equity,
            avg_turnover=snapshot.metrics.avg_turnover,
            win_rate=snapshot.metrics.win_rate,
            payoff_ratio=snapshot.metrics.payoff_ratio,
            profit_factor=snapshot.metrics.profit_factor,
            current_drawdown=snapshot.metrics.current_drawdown,
            best_month=snapshot.metrics.best_month,
            worst_month=snapshot.metrics.worst_month,
        ),
        rolling=RollingMetrics(
            window=snapshot.rolling.window,
            series={
                "rolling_sharpe": snapshot.rolling.series["rolling_sharpe"],
                "rolling_return": pd.Series(
                    [0.02, 0.01, 0.04, 0.05, 0.01, 0.03],
                    index=snapshot.rolling.series["rolling_sharpe"].index,
                    name="rolling_return",
                ),
                "rolling_volatility": pd.Series(
                    [0.15, 0.16, 0.14, 0.13, 0.17, 0.15],
                    index=snapshot.rolling.series["rolling_sharpe"].index,
                    name="rolling_volatility",
                ),
            },
        ),
        drawdowns=snapshot.drawdowns,
        exposure=snapshot.exposure,
        sectors=snapshot.sectors,
        strategy_equity=snapshot.strategy_equity,
        strategy_returns=snapshot.strategy_returns,
        benchmark_returns=pd.Series(dtype=float, name="benchmark_returns"),
        benchmark_equity=pd.Series(dtype=float, name="benchmark_equity"),
        strategy_name=snapshot.strategy_name,
        benchmark=None,
        profile=snapshot.profile,
        has_benchmark=False,
        research=snapshot.research,
    )


def test_tearsheet_figure_builder_writes_expected_page_assets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(go.Figure, "write_image", _write_image_success)

    assets = TearsheetFigureBuilder(tmp_path).build(_sample_snapshot("alpha"))

    assert assets.keys() == {"performance"}
    for path in assets.values():
        assert path.exists()
        assert path.suffix == ".png"


def test_tearsheet_dashboard_does_not_render_latest_holdings_panel(tmp_path: Path, monkeypatch) -> None:
    def _fail_if_called(self, ax, snapshot):  # type: ignore[no-untyped-def]
        raise AssertionError("latest holdings panel should not be rendered")

    monkeypatch.setattr(TearsheetFigureBuilder, "_plot_holdings", _fail_if_called)

    assets = TearsheetFigureBuilder(tmp_path).build(_sample_snapshot("alpha"))

    assert assets["performance"].exists()


def test_tearsheet_dashboard_overlays_benchmark_panels_and_maps_sector_names(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    builder = TearsheetFigureBuilder(tmp_path)
    snapshot = _sample_snapshot("alpha")

    fig, axes = plt.subplots(2, 3)
    try:
        builder._plot_underwater(axes[0, 0], snapshot)
        builder._plot_yearly_returns(axes[0, 1], snapshot)
        builder._plot_monthly_heatmap(axes[0, 2], snapshot)
        builder._plot_distribution(axes[1, 0], snapshot.strategy_returns, "Daily Return Distribution", benchmark_series=snapshot.benchmark_returns)
        builder._plot_sector_weights(axes[1, 1], snapshot)

        drawdown_labels = {line.get_label() for line in axes[0, 0].lines}
        assert snapshot.display_name in drawdown_labels
        assert "KOSPI200" in drawdown_labels
        assert len(axes[0, 1].patches) >= 2
        heatmap_labels = {tick.get_text() for tick in axes[0, 2].get_yticklabels()}
        assert "2024 · S" in heatmap_labels
        assert "2024 · BM" in heatmap_labels
        assert len(axes[1, 0].patches) > 0
        sector_labels = {tick.get_text() for tick in axes[1, 1].get_xticklabels()}
        assert "Industrials" in sector_labels
        assert "Information Technology" in sector_labels
    finally:
        plt.close(fig)


def test_tearsheet_dashboard_supports_no_benchmark_panels(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    builder = TearsheetFigureBuilder(tmp_path)
    snapshot = _sample_snapshot_without_benchmark("absolute")

    fig, axes = plt.subplots(1, 4)
    try:
        builder._plot_underwater(axes[0], snapshot)
        builder._plot_yearly_returns(axes[1], snapshot)
        builder._plot_monthly_heatmap(axes[2], snapshot)
        builder._plot_distribution(axes[3], snapshot.strategy_returns, "Daily Return Distribution", benchmark_series=snapshot.benchmark_returns)

        drawdown_labels = {line.get_label() for line in axes[0].lines}
        assert snapshot.display_name in drawdown_labels
        assert "KOSPI200" not in drawdown_labels
        assert len(axes[1].patches) >= 1
        heatmap_labels = {tick.get_text() for tick in axes[2].get_yticklabels()}
        assert "2024" in heatmap_labels
        assert not any("BM" in label for label in heatmap_labels)
        assert len(axes[3].patches) > 0
    finally:
        plt.close(fig)


def test_comparison_figure_builder_writes_expected_page_assets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(go.Figure, "write_image", _write_image_success)

    snapshots = [_sample_snapshot("alpha", 1.0), _sample_snapshot("beta", 1.2)]
    assets = ComparisonFigureBuilder(tmp_path).build(snapshots)

    assert assets.keys() == {"executive", "performance", "rolling", "exposure"}
    for path in assets.values():
        assert path.exists()
        assert path.suffix == ".png"


def test_comparison_figure_builder_handles_benchmarkless_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(go.Figure, "write_image", _write_image_success)

    snapshots = [_sample_snapshot("alpha", 1.0), _sample_snapshot_without_benchmark("absolute")]
    assets = ComparisonFigureBuilder(tmp_path).build(snapshots)

    assert assets.keys() == {"executive", "performance", "rolling", "exposure"}
    for path in assets.values():
        assert path.exists()
        assert path.suffix == ".png"


def test_write_figure_asset_uses_browser_screenshot_fallback(tmp_path: Path, monkeypatch) -> None:
    def _write_image_fail(self, path, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("kaleido missing")

    def _screenshot_success(fig, html_path, png_path):  # type: ignore[no-untyped-def]
        html_path.write_text("<html></html>", encoding="utf-8")
        png_path.write_bytes(b"png")
        return None

    monkeypatch.setattr(go.Figure, "write_image", _write_image_fail)
    monkeypatch.setattr("backtesting.reporting.figures._write_browser_screenshot", _screenshot_success)

    path = write_figure_asset(go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4])), tmp_path / "fallback.png")

    assert path.exists()
    assert path.suffix == ".png"
