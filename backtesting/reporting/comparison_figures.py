from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .figures import write_figure_asset
from .snapshots import PerformanceSnapshot

__all__ = ("ComparisonFigureBuilder",)


class ComparisonFigureBuilder:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def build(self, snapshots: list[PerformanceSnapshot], *, require_png: bool = False) -> dict[str, Path]:
        if not snapshots:
            raise ValueError("snapshots must not be empty")

        assets = {
            "executive": write_figure_asset(
                self._build_executive(snapshots), self.out_dir / "executive.png", require_png=require_png
            ),
            "performance": write_figure_asset(
                self._build_performance(snapshots), self.out_dir / "performance.png", require_png=require_png
            ),
            "rolling": write_figure_asset(
                self._build_rolling(snapshots), self.out_dir / "rolling.png", require_png=require_png
            ),
            "exposure": write_figure_asset(
                self._build_exposure(snapshots), self.out_dir / "exposure.png", require_png=require_png
            ),
        }
        return assets

    def _build_executive(self, snapshots: list[PerformanceSnapshot]) -> go.Figure:
        has_any_benchmark = any(snapshot.has_benchmark for snapshot in snapshots)
        figure = make_subplots(
            rows=3,
            cols=2,
            shared_xaxes=False,
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            subplot_titles=(
                "Cumulative Return",
                "Underwater",
                "CAGR",
                "Sharpe / Sortino",
                "Win Rate / Profit Factor",
                "Run Summary",
            ),
        )
        for snapshot in snapshots:
            figure.add_trace(
                go.Scatter(
                    x=snapshot.strategy_equity.index,
                    y=_cumulative_returns(snapshot.strategy_equity).values,
                    mode="lines",
                    name=snapshot.display_name,
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=snapshot.drawdowns.underwater.index,
                    y=snapshot.drawdowns.underwater.values,
                    mode="lines",
                    name=f"{snapshot.display_name} DD",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        labels = [snapshot.display_name for snapshot in snapshots]
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.cagr for snapshot in snapshots], name="CAGR"),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.sharpe for snapshot in snapshots], name="Sharpe"),
            row=2,
            col=2,
        )
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.sortino for snapshot in snapshots], name="Sortino"),
            row=2,
            col=2,
        )
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.win_rate for snapshot in snapshots], name="Win Rate"),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.profit_factor for snapshot in snapshots], name="Profit Factor"),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="text",
                text=[_comparison_summary_lines(snapshots)],
                textposition="top left",
                textfont={"size": 12},
                showlegend=False,
            ),
            row=3,
            col=2,
        )
        figure.update_xaxes(visible=False, row=3, col=2)
        figure.update_yaxes(visible=False, row=3, col=2)
        figure.update_yaxes(title_text="Return", tickformat=".0%", row=1, col=1)
        figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=1, col=2)
        figure.update_yaxes(tickformat=".0%", row=2, col=1)
        figure.update_yaxes(title_text="Ratio", row=2, col=2)
        figure.update_yaxes(title_text="Level", row=3, col=1)
        figure.update_layout(
            title="Comparison Executive Summary",
            legend_title_text="Run",
            barmode="group",
            height=1200,
        )
        if has_any_benchmark:
            figure.add_annotation(
                x=1.0,
                y=1.08,
                xref="paper",
                yref="paper",
                text="Benchmark-aware metrics shown when available; benchmarkless runs degrade gracefully.",
                showarrow=False,
                align="right",
                font={"size": 11},
            )
        return figure

    def _build_performance(self, snapshots: list[PerformanceSnapshot]) -> go.Figure:
        labels = [snapshot.display_name for snapshot in snapshots]
        has_any_benchmark = any(snapshot.has_benchmark for snapshot in snapshots)
        figure = make_subplots(
            rows=3,
            cols=2,
            vertical_spacing=0.16,
            subplot_titles=(
                "Cumulative Return",
                "Annual Volatility",
                "Max Drawdown",
                "Current Drawdown",
                "Information Ratio / Alpha" if has_any_benchmark else "Win Rate / Payoff Ratio",
                "Active Risk / Tracking Error" if has_any_benchmark else "Best / Worst Month",
            ),
        )
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.cumulative_return for snapshot in snapshots], name="Cumulative Return"),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.annual_volatility for snapshot in snapshots], name="Volatility"),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.max_drawdown for snapshot in snapshots], name="Max Drawdown"),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.current_drawdown for snapshot in snapshots], name="Current Drawdown"),
            row=2,
            col=2,
        )
        if has_any_benchmark:
            figure.add_trace(
                go.Bar(x=labels, y=[_metric_value(snapshot, "information_ratio") for snapshot in snapshots], name="Information Ratio"),
                row=3,
                col=1,
            )
            figure.add_trace(
                go.Bar(x=labels, y=[_metric_value(snapshot, "alpha") for snapshot in snapshots], name="Alpha"),
                row=3,
                col=1,
            )
            figure.add_trace(
                go.Bar(x=labels, y=[_metric_value(snapshot, "active_risk") for snapshot in snapshots], name="Active Risk"),
                row=3,
                col=2,
            )
            figure.add_trace(
                go.Bar(x=labels, y=[_metric_value(snapshot, "tracking_error") for snapshot in snapshots], name="Tracking Error"),
                row=3,
                col=2,
            )
        else:
            figure.add_trace(
                go.Bar(x=labels, y=[snapshot.metrics.win_rate for snapshot in snapshots], name="Win Rate"),
                row=3,
                col=1,
            )
            figure.add_trace(
                go.Bar(x=labels, y=[snapshot.metrics.payoff_ratio for snapshot in snapshots], name="Payoff Ratio"),
                row=3,
                col=1,
            )
            figure.add_trace(
                go.Bar(x=labels, y=[snapshot.metrics.best_month for snapshot in snapshots], name="Best Month"),
                row=3,
                col=2,
            )
            figure.add_trace(
                go.Bar(x=labels, y=[snapshot.metrics.worst_month for snapshot in snapshots], name="Worst Month"),
                row=3,
                col=2,
            )
        figure.update_yaxes(tickformat=".0%", row=1, col=1)
        figure.update_yaxes(tickformat=".0%", row=1, col=2)
        figure.update_yaxes(tickformat=".0%", row=2, col=1)
        figure.update_yaxes(tickformat=".0%", row=2, col=2)
        if has_any_benchmark:
            figure.update_yaxes(tickformat=".0%", row=3, col=1)
            figure.update_yaxes(tickformat=".0%", row=3, col=2)
        else:
            figure.update_yaxes(title_text="Ratio", row=3, col=1)
            figure.update_yaxes(tickformat=".0%", row=3, col=2)
        figure.update_layout(title="Performance Comparison", barmode="group", height=1200)
        return figure

    def _build_rolling(self, snapshots: list[PerformanceSnapshot]) -> go.Figure:
        has_any_benchmark = any(snapshot.has_benchmark for snapshot in snapshots)
        figure = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(
                "Rolling Sharpe",
                "Rolling Return",
                "Rolling Beta / Correlation" if has_any_benchmark else "Rolling Volatility",
                "Turnover / Holdings Count",
            ),
        )
        for snapshot in snapshots:
            figure.add_trace(self._line(snapshot.rolling.series.get("rolling_sharpe"), snapshot.display_name), row=1, col=1)
            figure.add_trace(self._line(snapshot.rolling.series.get("rolling_return"), snapshot.display_name), row=2, col=1)
            if has_any_benchmark:
                metric_name = snapshot.display_name
                figure.add_trace(
                    self._line(
                        _preferred_series(snapshot, "rolling_beta", "rolling_volatility"),
                        f"{metric_name} Beta/Vol",
                    ),
                    row=3,
                    col=1,
                )
                figure.add_trace(
                    self._line(
                        _preferred_series(snapshot, "rolling_correlation", "rolling_return"),
                        f"{metric_name} Corr/Ret",
                    ),
                    row=3,
                    col=1,
                )
            else:
                figure.add_trace(self._line(snapshot.rolling.series.get("rolling_volatility"), snapshot.display_name), row=3, col=1)
            figure.add_trace(self._line(snapshot.exposure.turnover, f"{snapshot.display_name} Turnover"), row=4, col=1)
            figure.add_trace(self._line(snapshot.exposure.holdings_count, f"{snapshot.display_name} Holdings"), row=4, col=1)
        figure.update_yaxes(title_text="Sharpe", row=1, col=1)
        figure.update_yaxes(title_text="Return", tickformat=".0%", row=2, col=1)
        figure.update_yaxes(title_text="Level", row=3, col=1)
        figure.update_yaxes(title_text="Level", row=4, col=1)
        figure.update_layout(title="Rolling Comparison", legend_title_text="Run", height=1200)
        return figure

    def _build_exposure(self, snapshots: list[PerformanceSnapshot]) -> go.Figure:
        figure = make_subplots(
            rows=2,
            cols=2,
            vertical_spacing=0.16,
            subplot_titles=("Holdings Count", "Average Turnover", "Latest Sector Weights", "Largest Holding"),
        )
        for snapshot in snapshots:
            figure.add_trace(self._line(snapshot.exposure.holdings_count, snapshot.display_name), row=1, col=1)
        figure.add_trace(
            go.Bar(
                x=[snapshot.display_name for snapshot in snapshots],
                y=[snapshot.metrics.avg_turnover for snapshot in snapshots],
                name="Avg Turnover",
            ),
            row=1,
            col=2,
        )
        for trace in _sector_weight_traces(snapshots):
            figure.add_trace(trace, row=2, col=1)
        largest_holding_rows = [_largest_holding(snapshot.exposure.latest_holdings) for snapshot in snapshots]
        figure.add_trace(
            go.Bar(
                x=[snapshot.display_name for snapshot in snapshots],
                y=[row["target_weight"] for row in largest_holding_rows],
                text=[row["symbol"] for row in largest_holding_rows],
                textposition="outside",
                name="Largest Holding",
            ),
            row=2,
            col=2,
        )
        figure.update_yaxes(title_text="Count", row=1, col=1)
        figure.update_yaxes(tickformat=".0%", row=1, col=2)
        figure.update_yaxes(tickformat=".0%", row=2, col=1)
        figure.update_yaxes(tickformat=".0%", row=2, col=2)
        figure.update_layout(title="Exposure Comparison", legend_title_text="Run", barmode="group", height=900)
        return figure

    @staticmethod
    def _line(series: pd.Series | None, name: str) -> go.Scatter:
        clean = pd.Series(dtype=float) if series is None else series.astype(float)
        return go.Scatter(x=clean.index, y=clean.values, mode="lines", name=name)


def _largest_holding(latest_holdings: pd.DataFrame) -> dict[str, float | str]:
    if latest_holdings.empty:
        return {"symbol": "No Holdings", "target_weight": 0.0}

    ranked = latest_holdings.copy()
    if "abs_weight" not in ranked.columns and "target_weight" in ranked.columns:
        ranked["abs_weight"] = ranked["target_weight"].abs()
    top_row = ranked.sort_values(["abs_weight", "symbol"], ascending=[False, True]).iloc[0]
    return {"symbol": str(top_row["symbol"]), "target_weight": float(top_row["target_weight"])}


def _cumulative_returns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float, name="cumulative_return")
    return equity.div(float(equity.iloc[0])).sub(1.0).rename("cumulative_return")


def _metric_value(snapshot: PerformanceSnapshot, name: str) -> float:
    value = getattr(snapshot.metrics, name)
    return np.nan if value is None else float(value)


def _preferred_series(snapshot: PerformanceSnapshot, primary: str, fallback: str) -> pd.Series | None:
    series = snapshot.rolling.series.get(primary)
    if series is not None:
        return series
    return snapshot.rolling.series.get(fallback)


def _sector_weight_traces(snapshots: list[PerformanceSnapshot], *, limit: int = 5) -> list[go.Bar]:
    sectors: list[str] = []
    for snapshot in snapshots:
        for sector in snapshot.sectors.latest_weighted.abs().sort_values(ascending=False).head(limit).index:
            sector_name = str(sector)
            if sector_name not in sectors:
                sectors.append(sector_name)
    if not sectors:
        return [go.Bar(x=[], y=[], name="Sector Weight")]
    traces: list[go.Bar] = []
    for sector in sectors[:limit]:
        traces.append(
            go.Bar(
                x=[snapshot.display_name for snapshot in snapshots],
                y=[float(snapshot.sectors.latest_weighted.get(sector, 0.0)) for snapshot in snapshots],
                name=sector,
            )
        )
    return traces


def _comparison_summary_lines(snapshots: list[PerformanceSnapshot]) -> str:
    lines: list[str] = []
    for snapshot in snapshots:
        benchmark_label = "BM" if snapshot.has_benchmark else "No BM"
        lines.append(
            " | ".join(
                [
                    snapshot.display_name,
                    snapshot.profile.value,
                    benchmark_label,
                    f"CAGR {snapshot.metrics.cagr:.1%}",
                    f"Sharpe {snapshot.metrics.sharpe:.2f}",
                    f"MDD {snapshot.metrics.max_drawdown:.1%}",
                ]
            )
        )
    return "<br>".join(lines)
