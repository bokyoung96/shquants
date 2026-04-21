from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import SavedRun
from .tables import build_latest_weights_table

__all__ = ("PlotExportError", "PlotLibrary")


class PlotExportError(RuntimeError):
    def __init__(self, png_path: Path, html_path: Path, original: Exception) -> None:
        super().__init__(f"Failed to write Plotly image to {png_path}; wrote HTML fallback to {html_path}")
        self.png_path = png_path
        self.html_path = html_path
        self.original = original


class PlotLibrary:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def equity(self, runs: list[SavedRun], *, require_png: bool = False) -> Path:
        fig = go.Figure()
        for run in runs:
            fig.add_trace(self._line_trace(run.equity, run.run_id))
        fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity", legend_title_text="Run")
        return self._write_png(fig, "equity.png", require_png=require_png)

    def drawdown(self, runs: list[SavedRun], *, require_png: bool = False) -> Path:
        fig = go.Figure()
        for run in runs:
            dd = run.equity.div(run.equity.cummax()).sub(1.0)
            fig.add_trace(self._line_trace(dd, run.run_id))
        fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown", legend_title_text="Run")
        return self._write_png(fig, "drawdown.png", require_png=require_png)

    def turnover(self, runs: list[SavedRun], *, require_png: bool = False) -> Path:
        fig = go.Figure()
        for run in runs:
            fig.add_trace(self._line_trace(run.turnover, run.run_id))
        fig.update_layout(title="Turnover", xaxis_title="Date", yaxis_title="Turnover", legend_title_text="Run")
        return self._write_png(fig, "turnover.png", require_png=require_png)

    def top_weights(self, runs: list[SavedRun], *, require_png: bool = False) -> Path:
        fig = go.Figure()
        for run in runs:
            table = build_latest_weights_table(run)
            if table.empty:
                continue
            fig.add_trace(
                go.Bar(
                    x=table["target_weight"],
                    y=table["symbol"],
                    orientation="h",
                    name=run.run_id,
                )
            )
        fig.update_layout(
            title="Top Weights",
            xaxis_title="Target Weight",
            yaxis_title="Symbol",
            barmode="group",
            legend_title_text="Run",
        )
        return self._write_png(fig, "top_weights.png", require_png=require_png)

    def monthly_heatmap(self, runs: list[SavedRun], *, require_png: bool = False) -> Path:
        rows = max(len(runs), 1)
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=self._vertical_spacing(rows),
            subplot_titles=[run.run_id for run in runs] or ["Monthly Returns"],
        )
        for row, run in enumerate(runs, start=1):
            monthly = self._monthly_returns(run)
            heatmap = self._monthly_heatmap_trace(monthly)
            fig.add_trace(heatmap, row=row, col=1)
        fig.update_layout(title="Monthly Return Heatmap", coloraxis_colorscale="RdYlGn")
        return self._write_png(fig, "monthly_heatmap.png", require_png=require_png)

    @staticmethod
    def _line_trace(series: pd.Series, name: str) -> go.Scatter:
        return go.Scatter(x=series.index, y=series.values, mode="lines", name=name)

    @staticmethod
    def _monthly_returns(run: SavedRun) -> pd.Series:
        if run.monthly_returns is not None:
            return run.monthly_returns
        monthly = (1.0 + run.returns).resample("ME").prod().sub(1.0)
        monthly.index = monthly.index.normalize()
        return monthly.astype(float)

    @staticmethod
    def _monthly_heatmap_trace(monthly: pd.Series) -> go.Heatmap:
        if monthly.empty:
            return go.Heatmap(z=[[0.0]], x=["Jan"], y=[""], showscale=False)

        frame = monthly.rename("return").to_frame()
        frame["year"] = frame.index.year
        frame["month"] = frame.index.month
        matrix = frame.pivot(index="year", columns="month", values="return").sort_index()
        matrix = matrix.reindex(columns=range(1, 13))
        return go.Heatmap(z=matrix.values, x=[_MONTH_LABELS[m - 1] for m in matrix.columns], y=matrix.index.astype(str), colorscale="RdYlGn", coloraxis="coloraxis")

    @staticmethod
    def _vertical_spacing(rows: int) -> float:
        if rows <= 1:
            return 0.08
        return min(0.08, 1.0 / (rows - 1) - 1e-6)

    def _write_png(self, fig: go.Figure, filename: str, *, require_png: bool = False) -> Path:
        png_path = self.out_dir / filename
        html_path = png_path.with_suffix(".html")
        try:
            fig.write_image(png_path)
            return png_path
        except Exception as exc:  # pragma: no cover - exercised in fallback environments
            fig.write_html(html_path)
            if require_png:
                raise PlotExportError(png_path, html_path, exc) from exc
            return html_path


_MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
