from __future__ import annotations

from pathlib import Path

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
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Strategy Equity", "Underwater"),
        )
        for snapshot in snapshots:
            figure.add_trace(
                go.Scatter(
                    x=snapshot.strategy_equity.index,
                    y=snapshot.strategy_equity.values,
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
                row=2,
                col=1,
            )
        figure.update_yaxes(title_text="Equity", row=1, col=1)
        figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
        figure.update_layout(title="Comparison Executive Summary", legend_title_text="Run")
        return figure

    def _build_performance(self, snapshots: list[PerformanceSnapshot]) -> go.Figure:
        labels = [snapshot.display_name for snapshot in snapshots]
        figure = make_subplots(
            rows=2,
            cols=2,
            vertical_spacing=0.16,
            subplot_titles=("Cumulative Return", "Sharpe Ratio", "Annual Volatility", "Max Drawdown"),
        )
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.cumulative_return for snapshot in snapshots], name="Cumulative Return"),
            row=1,
            col=1,
        )
        figure.add_trace(go.Bar(x=labels, y=[snapshot.metrics.sharpe for snapshot in snapshots], name="Sharpe"), row=1, col=2)
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.annual_volatility for snapshot in snapshots], name="Volatility"),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Bar(x=labels, y=[snapshot.metrics.max_drawdown for snapshot in snapshots], name="Max Drawdown"),
            row=2,
            col=2,
        )
        figure.update_yaxes(tickformat=".0%", row=1, col=1)
        figure.update_yaxes(tickformat=".0%", row=2, col=1)
        figure.update_yaxes(tickformat=".0%", row=2, col=2)
        figure.update_layout(title="Performance Comparison", showlegend=False)
        return figure

    def _build_rolling(self, snapshots: list[PerformanceSnapshot]) -> go.Figure:
        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Rolling Sharpe", "Rolling Beta", "Holdings Count"),
        )
        for snapshot in snapshots:
            figure.add_trace(self._line(snapshot.rolling.series.get("rolling_sharpe"), snapshot.display_name), row=1, col=1)
            figure.add_trace(
                self._line(snapshot.rolling.series.get("rolling_beta"), snapshot.display_name),
                row=2,
                col=1,
            )
            figure.add_trace(self._line(snapshot.exposure.holdings_count, snapshot.display_name), row=3, col=1)
        figure.update_yaxes(title_text="Sharpe", row=1, col=1)
        figure.update_yaxes(title_text="Beta", row=2, col=1)
        figure.update_yaxes(title_text="Count", row=3, col=1)
        figure.update_layout(title="Rolling Comparison", legend_title_text="Run")
        return figure

    def _build_exposure(self, snapshots: list[PerformanceSnapshot]) -> go.Figure:
        figure = make_subplots(
            rows=2,
            cols=2,
            vertical_spacing=0.16,
            subplot_titles=("Holdings Count", "Latest Sector Weights", "Latest Sector Counts", "Largest Holding"),
        )
        for snapshot in snapshots:
            figure.add_trace(self._line(snapshot.exposure.holdings_count, snapshot.display_name), row=1, col=1)
            figure.add_trace(
                go.Bar(
                    x=snapshot.sectors.latest_weighted.index,
                    y=snapshot.sectors.latest_weighted.values,
                    name=snapshot.display_name,
                ),
                row=1,
                col=2,
            )
            figure.add_trace(
                go.Bar(
                    x=snapshot.sectors.latest_count.index,
                    y=snapshot.sectors.latest_count.values,
                    name=snapshot.display_name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            top_holding = _largest_holding(snapshot.exposure.latest_holdings)
            figure.add_trace(
                go.Bar(
                    x=[snapshot.display_name],
                    y=[top_holding["target_weight"]],
                    name=top_holding["symbol"],
                    showlegend=False,
                ),
                row=2,
                col=2,
            )
        figure.update_yaxes(title_text="Count", row=1, col=1)
        figure.update_yaxes(tickformat=".0%", row=1, col=2)
        figure.update_layout(title="Exposure Comparison", legend_title_text="Run", barmode="group")
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
