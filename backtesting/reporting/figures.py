from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import PlotExportError
from .snapshots import PerformanceSnapshot

__all__ = ("TearsheetFigureBuilder", "write_figure_asset")


class TearsheetFigureBuilder:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def build(self, snapshot: PerformanceSnapshot, *, require_png: bool = False) -> dict[str, Path]:
        assets = {
            "executive": write_figure_asset(
                self._build_executive(snapshot), self.out_dir / "executive.png", require_png=require_png
            ),
            "rolling": write_figure_asset(
                self._build_rolling(snapshot), self.out_dir / "rolling.png", require_png=require_png
            ),
            "calendar": write_figure_asset(
                self._build_calendar(snapshot), self.out_dir / "calendar.png", require_png=require_png
            ),
            "exposure": write_figure_asset(
                self._build_exposure(snapshot), self.out_dir / "exposure.png", require_png=require_png
            ),
        }
        return assets

    def _build_executive(self, snapshot: PerformanceSnapshot) -> go.Figure:
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Equity Curve", "Underwater"),
        )
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
                x=snapshot.benchmark_equity.index,
                y=snapshot.benchmark_equity.values,
                mode="lines",
                name="Benchmark",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=snapshot.drawdowns.underwater.index,
                y=snapshot.drawdowns.underwater.values,
                mode="lines",
                fill="tozeroy",
                name="Underwater",
            ),
            row=2,
            col=1,
        )
        figure.update_yaxes(title_text="Equity", row=1, col=1)
        figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
        figure.update_layout(title=f"{snapshot.display_name} Executive Summary", legend_title_text="Series")
        return figure

    def _build_rolling(self, snapshot: PerformanceSnapshot) -> go.Figure:
        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Rolling Sharpe", "Rolling Beta", "Holdings Count"),
        )
        figure.add_trace(self._line(snapshot.rolling.series.get("rolling_sharpe"), "Rolling Sharpe"), row=1, col=1)
        figure.add_trace(self._line(snapshot.rolling.series.get("rolling_beta"), "Rolling Beta"), row=2, col=1)
        figure.add_trace(self._line(snapshot.exposure.holdings_count, "Holdings Count"), row=3, col=1)
        figure.update_yaxes(title_text="Sharpe", row=1, col=1)
        figure.update_yaxes(title_text="Beta", row=2, col=1)
        figure.update_yaxes(title_text="Count", row=3, col=1)
        figure.update_layout(title=f"{snapshot.display_name} Rolling Diagnostics", showlegend=False)
        return figure

    def _build_calendar(self, snapshot: PerformanceSnapshot) -> go.Figure:
        monthly_returns = _monthly_returns(snapshot.strategy_returns)
        figure = make_subplots(
            rows=2,
            cols=2,
            vertical_spacing=0.16,
            subplot_titles=("Monthly Returns", "Return Distribution", "Quarterly Returns", "Cumulative Returns"),
        )
        figure.add_trace(
            go.Bar(x=monthly_returns.index, y=monthly_returns.values, name="Monthly Returns"),
            row=1,
            col=1,
        )
        figure.add_trace(go.Histogram(x=snapshot.strategy_returns.values, nbinsx=12, name="Daily Returns"), row=1, col=2)
        quarterly_returns = (1.0 + snapshot.strategy_returns).resample("QE").prod().sub(1.0)
        figure.add_trace(
            go.Bar(x=quarterly_returns.index, y=quarterly_returns.values, name="Quarterly Returns"),
            row=2,
            col=1,
        )
        cumulative_returns = snapshot.strategy_equity.div(float(snapshot.strategy_equity.iloc[0])).sub(1.0)
        figure.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, mode="lines", name="Cumulative Return"),
            row=2,
            col=2,
        )
        figure.update_yaxes(tickformat=".0%", row=1, col=1)
        figure.update_yaxes(tickformat=".0%", row=2, col=1)
        figure.update_yaxes(tickformat=".0%", row=2, col=2)
        figure.update_layout(title=f"{snapshot.display_name} Calendar And Return Profile", showlegend=False)
        return figure

    def _build_exposure(self, snapshot: PerformanceSnapshot) -> go.Figure:
        latest_holdings = _top_holdings(snapshot.exposure.latest_holdings)
        sector_weights = snapshot.sectors.latest_weighted.sort_values(ascending=False)
        sector_counts = snapshot.sectors.latest_count.sort_values(ascending=False)
        figure = make_subplots(
            rows=2,
            cols=2,
            vertical_spacing=0.16,
            subplot_titles=("Holdings Count", "Latest Holdings", "Sector Weights", "Sector Counts"),
        )
        figure.add_trace(
            go.Scatter(
                x=snapshot.exposure.holdings_count.index,
                y=snapshot.exposure.holdings_count.values,
                mode="lines",
                name="Holdings Count",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=latest_holdings["target_weight"] if not latest_holdings.empty else [],
                y=latest_holdings["symbol"] if not latest_holdings.empty else [],
                orientation="h",
                name="Latest Holdings",
            ),
            row=1,
            col=2,
        )
        figure.add_trace(go.Bar(x=sector_weights.index, y=sector_weights.values, name="Sector Weight"), row=2, col=1)
        figure.add_trace(go.Bar(x=sector_counts.index, y=sector_counts.values, name="Sector Count"), row=2, col=2)
        figure.update_yaxes(title_text="Count", row=1, col=1)
        figure.update_xaxes(title_text="Weight", row=1, col=2)
        figure.update_yaxes(tickformat=".0%", row=2, col=1)
        figure.update_layout(title=f"{snapshot.display_name} Exposure Profile", showlegend=False)
        return figure

    @staticmethod
    def _line(series: pd.Series | None, name: str) -> go.Scatter:
        clean = pd.Series(dtype=float) if series is None else series.astype(float)
        return go.Scatter(x=clean.index, y=clean.values, mode="lines", name=name)


def write_figure_asset(fig: go.Figure, png_path: Path, *, require_png: bool = False) -> Path:
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = png_path.with_suffix(".html")
    try:
        fig.write_image(png_path)
        return png_path
    except Exception as exc:  # pragma: no cover - depends on environment image export support
        fig.write_html(html_path)
        if require_png:
            raise PlotExportError(png_path, html_path, exc) from exc
        return html_path


def _monthly_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return returns
    monthly = (1.0 + returns.astype(float)).resample("ME").prod().sub(1.0)
    monthly.index = monthly.index.normalize()
    return monthly.rename("monthly_returns")


def _top_holdings(latest_holdings: pd.DataFrame, *, limit: int = 10) -> pd.DataFrame:
    if latest_holdings.empty:
        return latest_holdings

    ranked = latest_holdings.copy()
    if "abs_weight" not in ranked.columns and "target_weight" in ranked.columns:
        ranked["abs_weight"] = ranked["target_weight"].abs()
    return ranked.sort_values(["abs_weight", "symbol"], ascending=[False, True]).head(limit)
