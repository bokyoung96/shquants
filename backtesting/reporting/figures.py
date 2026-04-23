from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib import colors as mcolors
from matplotlib import dates as mdates
from matplotlib import patches
from matplotlib.ticker import FuncFormatter

from .plots import PlotExportError
from .snapshots import PerformanceSnapshot

__all__ = ("TearsheetFigureBuilder", "write_figure_asset")

_BG = "#ffffff"
_PANEL = "#ffffff"
_PANEL_ALT = "#f7f7f7"
_TEXT = "#181512"
_MUTED = "#7f766c"
_GRID = "#dedede"
_STRATEGY = "#cf2636"
_BENCHMARK = "#111111"
_POSITIVE = "#cf2636"
_NEGATIVE = "#3d3832"
_CARD_COLORS = ("#fff5f6", "#fff8f1", "#f7f3ef", "#f4ede7")
_HEATMAP = mcolors.LinearSegmentedColormap.from_list(
    "shquants_dashboard",
    ["#2f2924", "#f8f4ee", "#cf2636"],
)


class TearsheetFigureBuilder:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def build(self, snapshot: PerformanceSnapshot, *, require_png: bool = False) -> dict[str, Path]:
        path = self.out_dir / "performance.png"
        self._write_performance_dashboard(snapshot, path)
        if require_png and not path.exists():
            raise PlotExportError(path, path.with_suffix(".html"), RuntimeError("failed to render matplotlib dashboard"))
        return {"performance": path}

    def _write_performance_dashboard(self, snapshot: PerformanceSnapshot, png_path: Path) -> None:
        figure = plt.figure(figsize=(22, 16), facecolor=_BG)
        grid = figure.add_gridspec(4, 4, hspace=0.32, wspace=0.22, height_ratios=[1.15, 1.0, 1.0, 1.0])

        axes = {
            "equity": figure.add_subplot(grid[0, 0:2]),
            "underwater": figure.add_subplot(grid[0, 2]),
            "metrics": figure.add_subplot(grid[0, 3]),
            "rolling_sharpe": figure.add_subplot(grid[1, 0]),
            "rolling_relative": figure.add_subplot(grid[1, 1]),
            "yearly": figure.add_subplot(grid[1, 2]),
            "turnover": figure.add_subplot(grid[1, 3]),
            "heatmap": figure.add_subplot(grid[2, 0:2]),
            "daily_dist": figure.add_subplot(grid[2, 2]),
            "monthly_dist": figure.add_subplot(grid[2, 3]),
            "sector_weights": figure.add_subplot(grid[3, 0:2]),
            "holdings_count": figure.add_subplot(grid[3, 2]),
            "yearly_strip": figure.add_subplot(grid[3, 3]),
        }

        self._plot_equity(axes["equity"], snapshot)
        self._plot_underwater(axes["underwater"], snapshot)
        self._plot_metric_cards(axes["metrics"], snapshot)
        self._plot_series_panel(
            axes["rolling_sharpe"],
            snapshot.rolling.series.get("rolling_sharpe"),
            title=f"Rolling Sharpe ({snapshot.rolling.window}d)",
            color=_STRATEGY,
            zero_line=True,
        )
        self._plot_series_panel(
            axes["rolling_relative"],
            snapshot.rolling.series.get("rolling_beta") if snapshot.has_benchmark else snapshot.rolling.series.get("rolling_return"),
            title="Rolling Beta" if snapshot.has_benchmark else "Rolling Return",
            color=_BENCHMARK if snapshot.has_benchmark else _STRATEGY,
            zero_line=not snapshot.has_benchmark,
            one_line=snapshot.has_benchmark,
        )
        self._plot_yearly_returns(axes["yearly"], snapshot)
        self._plot_turnover(axes["turnover"], snapshot)
        self._plot_monthly_heatmap(axes["heatmap"], snapshot)
        self._plot_distribution(
            axes["daily_dist"],
            snapshot.strategy_returns,
            "Daily Return Distribution",
            benchmark_series=snapshot.benchmark_returns if snapshot.has_benchmark else None,
        )
        self._plot_distribution(
            axes["monthly_dist"],
            _monthly_returns(snapshot.strategy_returns),
            "Monthly Return Distribution",
            benchmark_series=_monthly_returns(snapshot.benchmark_returns) if snapshot.has_benchmark else None,
        )
        self._plot_sector_weights(axes["sector_weights"], snapshot)
        self._plot_holdings_count(axes["holdings_count"], snapshot)
        self._plot_yearly_strip(axes["yearly_strip"], snapshot)

        figure.suptitle(snapshot.display_name, x=0.055, y=0.985, ha="left", fontsize=26, fontweight="bold", color=_TEXT)
        figure.text(
            0.055,
            0.962,
            _subtitle(snapshot),
            ha="left",
            va="top",
            fontsize=11,
            color=_MUTED,
        )
        figure.text(
            0.945,
            0.982,
            "Strategy in red · benchmark in black" if snapshot.has_benchmark else "Strategy in red",
            ha="right",
            va="top",
            fontsize=11,
            color=_MUTED,
        )
        figure.savefig(png_path, dpi=180, facecolor=figure.get_facecolor(), bbox_inches="tight")
        plt.close(figure)

    def _plot_equity(self, ax, snapshot: PerformanceSnapshot) -> None:  # type: ignore[no-untyped-def]
        _style_axis(ax, "Cumulative Performance")
        strategy = _cumulative_returns(snapshot.strategy_equity)
        ax.plot(strategy.index, strategy.values, color=_STRATEGY, linewidth=2.8, label=snapshot.display_name)
        ax.fill_between(strategy.index, strategy.values, 0.0, color=_STRATEGY, alpha=0.08)
        if snapshot.has_benchmark and not snapshot.benchmark_equity.empty:
            benchmark = _cumulative_returns(snapshot.benchmark_equity)
            ax.plot(
                benchmark.index,
                benchmark.values,
                color=_BENCHMARK,
                linewidth=2.2,
                label=snapshot.benchmark.name if snapshot.benchmark is not None else "Benchmark",
            )
        ax.yaxis.set_major_formatter(FuncFormatter(_percent_label))
        ax.legend(loc="upper left", frameon=False, fontsize=10)
        _format_date_axis(ax)

    def _plot_underwater(self, ax, snapshot: PerformanceSnapshot) -> None:  # type: ignore[no-untyped-def]
        _style_axis(ax, "Drawdown")
        underwater = snapshot.drawdowns.underwater.astype(float)
        if underwater.empty:
            ax.text(0.5, 0.5, "No drawdown data", ha="center", va="center", color=_MUTED, transform=ax.transAxes)
            return
        ax.fill_between(underwater.index, underwater.values, 0.0, color=_STRATEGY, alpha=0.14)
        ax.plot(underwater.index, underwater.values, color=_STRATEGY, linewidth=2.0, label=snapshot.display_name)
        if snapshot.has_benchmark and not snapshot.benchmark_equity.empty:
            benchmark_underwater = _drawdown(snapshot.benchmark_equity)
            ax.plot(
                benchmark_underwater.index,
                benchmark_underwater.values,
                color=_BENCHMARK,
                linewidth=1.8,
                label=snapshot.benchmark.name if snapshot.benchmark is not None else "Benchmark",
            )
        ax.axhline(0.0, color=_GRID, linewidth=1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(_percent_label))
        ax.legend(loc="lower left", frameon=False, fontsize=8)
        _format_date_axis(ax)

    def _plot_metric_cards(self, ax, snapshot: PerformanceSnapshot) -> None:  # type: ignore[no-untyped-def]
        ax.set_facecolor(_PANEL)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        metrics = [
            ("CAGR", f"{snapshot.metrics.cagr:.1%}"),
            ("Sharpe", f"{snapshot.metrics.sharpe:.2f}"),
            ("Max DD", f"{snapshot.metrics.max_drawdown:.1%}"),
            ("Win Rate", f"{snapshot.metrics.win_rate:.1%}"),
            ("Turnover", f"{snapshot.metrics.avg_turnover:.1%}"),
            ("Final Equity", _compact_number(snapshot.metrics.final_equity)),
        ]
        if snapshot.has_benchmark and snapshot.metrics.beta is not None:
            metrics.extend(
                [
                    ("Beta", f"{snapshot.metrics.beta:.2f}"),
                    ("Information Ratio", f"{(snapshot.metrics.information_ratio or 0.0):.2f}"),
                ]
            )
        else:
            metrics.extend(
                [
                    ("Sortino", f"{snapshot.metrics.sortino:.2f}"),
                    ("Best Month", f"{snapshot.metrics.best_month:.1%}"),
                ]
            )

        ax.text(0.05, 0.96, "Performance Snapshot", fontsize=13, fontweight="bold", color=_TEXT, va="top", transform=ax.transAxes)
        ax.text(0.05, 0.905, f"Profile · {snapshot.profile.value.title()}", fontsize=9, color=_MUTED, va="top", transform=ax.transAxes)
        ax.text(
            0.05,
            0.855,
            _annualized_metric_note(snapshot),
            fontsize=8,
            color=_MUTED,
            va="top",
            transform=ax.transAxes,
            clip_on=True,
        )
        rows = 4
        cols = 2
        left = 0.05
        top = 0.795
        gap_x = 0.045
        gap_y = 0.03
        card_width = (0.90 - gap_x) / cols
        card_height = 0.145
        for idx, (label, value) in enumerate(metrics[: rows * cols]):
            row = idx // cols
            column = idx % cols
            x = left + column * (card_width + gap_x)
            y = top - row * (card_height + gap_y)
            card = patches.FancyBboxPatch(
                (x, y - card_height),
                card_width,
                card_height,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                linewidth=0.0,
                facecolor=_CARD_COLORS[idx % len(_CARD_COLORS)],
                transform=ax.transAxes,
                clip_on=True,
            )
            ax.add_patch(card)
            ax.text(x + 0.025, y - 0.045, label, fontsize=8, color=_MUTED, transform=ax.transAxes, clip_on=True)
            ax.text(
                x + 0.025,
                y - 0.113,
                value,
                fontsize=11,
                fontweight="bold",
                color=_TEXT,
                transform=ax.transAxes,
                clip_on=True,
            )

    def _plot_series_panel(
        self,
        ax,
        series: pd.Series | None,
        *,
        title: str,
        color: str,
        zero_line: bool = False,
        one_line: bool = False,
    ) -> None:  # type: ignore[no-untyped-def]
        _style_axis(ax, title)
        clean = pd.Series(dtype=float) if series is None else series.dropna().astype(float)
        if clean.empty:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", color=_MUTED, transform=ax.transAxes)
            return
        ax.plot(clean.index, clean.values, color=color, linewidth=2.2)
        if zero_line:
            ax.axhline(0.0, color=_GRID, linewidth=1.0, linestyle="--")
        if one_line:
            ax.axhline(1.0, color=_GRID, linewidth=1.0, linestyle="--")
        _format_date_axis(ax)

    def _plot_yearly_returns(self, ax, snapshot: PerformanceSnapshot) -> None:  # type: ignore[no-untyped-def]
        _style_axis(ax, "Yearly Returns")
        strategy_yearly = snapshot.research.yearly_returns.astype(float)
        if strategy_yearly.empty:
            strategy_yearly = _yearly_returns(snapshot.strategy_returns)
        if strategy_yearly.empty:
            ax.text(0.5, 0.5, "Not enough yearly history", ha="center", va="center", color=_MUTED, transform=ax.transAxes)
            return

        strategy_by_year = pd.Series(strategy_yearly.values, index=[pd.Timestamp(index).year for index in strategy_yearly.index])
        frame = pd.DataFrame({"strategy": strategy_by_year})
        if snapshot.has_benchmark and not snapshot.benchmark_returns.empty:
            benchmark_yearly = _yearly_returns(snapshot.benchmark_returns)
            frame["benchmark"] = pd.Series(benchmark_yearly.values, index=[pd.Timestamp(index).year for index in benchmark_yearly.index])
        frame = frame.sort_index().fillna(0.0)
        x = list(range(len(frame.index)))
        if "benchmark" in frame:
            width = 0.38
            ax.bar([value - width / 2 for value in x], frame["strategy"].values, width=width, color=_STRATEGY, alpha=0.92, label=snapshot.display_name)
            ax.bar([value + width / 2 for value in x], frame["benchmark"].values, width=width, color=_BENCHMARK, alpha=0.78, label=snapshot.benchmark.name if snapshot.benchmark is not None else "Benchmark")
        else:
            colors = [_POSITIVE if value >= 0 else _NEGATIVE for value in frame["strategy"].values]
            ax.bar(x, frame["strategy"].values, color=colors, alpha=0.95, label=snapshot.display_name)
        ax.set_xticks(x, [str(year) for year in frame.index])
        ax.axhline(0.0, color=_GRID, linewidth=1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(_percent_label))
        ax.tick_params(axis="x", rotation=35)
        ax.legend(loc="upper left", frameon=False, fontsize=8)

    def _plot_turnover(self, ax, snapshot: PerformanceSnapshot) -> None:  # type: ignore[no-untyped-def]
        _style_axis(ax, "Turnover")
        turnover = snapshot.exposure.turnover.dropna().astype(float)
        if turnover.empty:
            ax.text(0.5, 0.5, "No turnover data", ha="center", va="center", color=_MUTED, transform=ax.transAxes)
            return
        ax.fill_between(turnover.index, turnover.values, 0.0, color=_STRATEGY, alpha=0.12)
        ax.plot(turnover.index, turnover.values, color=_STRATEGY, linewidth=2.1)
        ax.yaxis.set_major_formatter(FuncFormatter(_percent_label))
        _format_date_axis(ax)

    def _plot_monthly_heatmap(self, ax, snapshot: PerformanceSnapshot) -> None:  # type: ignore[no-untyped-def]
        title = "Monthly Heatmap (Strategy / BM)" if snapshot.has_benchmark else "Monthly Heatmap"
        _style_axis(ax, title)
        strategy_heatmap = snapshot.research.monthly_heatmap
        if strategy_heatmap.empty:
            strategy_heatmap = _monthly_heatmap(snapshot.strategy_returns)
        if strategy_heatmap.empty:
            ax.text(0.5, 0.5, "No monthly heatmap", ha="center", va="center", color=_MUTED, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        matrix = strategy_heatmap.reindex(columns=range(1, 13)).fillna(0.0)
        row_labels = [str(year) for year in matrix.index]
        if snapshot.has_benchmark and not snapshot.benchmark_returns.empty:
            benchmark_heatmap = _monthly_heatmap(snapshot.benchmark_returns)
            years = sorted(set(matrix.index).union(set(benchmark_heatmap.index)))
            strategy_matrix = matrix.reindex(index=years, columns=range(1, 13)).fillna(0.0)
            benchmark_matrix = benchmark_heatmap.reindex(index=years, columns=range(1, 13)).fillna(0.0)
            rows = []
            row_labels = []
            for year in years:
                rows.append(strategy_matrix.loc[year])
                row_labels.append(f"{year} · S")
                rows.append(benchmark_matrix.loc[year])
                row_labels.append(f"{year} · BM")
            matrix = pd.DataFrame(rows, index=row_labels, columns=range(1, 13))

        vmax = max(abs(float(matrix.min().min())), abs(float(matrix.max().max())), 0.01)
        ax.imshow(matrix.values, aspect="auto", cmap=_HEATMAP, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(12), _MONTH_LABELS)
        ax.set_yticks(range(len(matrix.index)), row_labels)
        ax.tick_params(axis="x", labelrotation=0)
        for row_idx, year in enumerate(matrix.index):
            for col_idx, month in enumerate(matrix.columns):
                value = float(matrix.loc[year, month])
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6 if len(matrix.index) > 10 else 8,
                    color="#ffffff" if abs(value) > vmax * 0.55 else _TEXT,
                )

    def _plot_distribution(
        self,
        ax,
        series: pd.Series,
        title: str,
        *,
        benchmark_series: pd.Series | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        _style_axis(ax, title)
        clean = series.dropna().astype(float)
        benchmark_clean = pd.Series(dtype=float) if benchmark_series is None else benchmark_series.dropna().astype(float)
        if clean.empty:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", color=_MUTED, transform=ax.transAxes)
            return
        combined = pd.concat([clean, benchmark_clean]) if not benchmark_clean.empty else clean
        bins = min(24, max(8, len(clean) // 8))
        if not combined.empty and float(combined.max()) > float(combined.min()):
            bin_edges = pd.Series(combined).pipe(lambda values: pd.interval_range(start=float(values.min()), end=float(values.max()), periods=bins)).to_tuples()
            edges = [edge[0] for edge in bin_edges] + [bin_edges[-1][1]]
        else:
            spread = max(abs(float(clean.iloc[0])) * 0.05, 1e-4)
            edges = [float(clean.iloc[0]) - spread, float(clean.iloc[0]) + spread]
        ax.hist(clean.values, bins=edges, color=_STRATEGY, alpha=0.58, edgecolor="white", label="Strategy")
        if not benchmark_clean.empty:
            ax.hist(benchmark_clean.values, bins=edges, color=_BENCHMARK, alpha=0.36, edgecolor="white", label="Benchmark")
        ax.axvline(clean.mean(), color=_STRATEGY, linewidth=2.0, linestyle="--")
        if not benchmark_clean.empty:
            ax.axvline(benchmark_clean.mean(), color=_BENCHMARK, linewidth=1.7, linestyle="--")
        ax.xaxis.set_major_formatter(FuncFormatter(_percent_label))
        ax.set_ylabel("Count", color=_MUTED)
        ax.tick_params(axis="y", colors=_MUTED)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    def _plot_holdings(self, ax, snapshot: PerformanceSnapshot) -> None:  # type: ignore[no-untyped-def]
        _style_axis(ax, "Latest Holdings")
        holdings = _top_holdings(snapshot.exposure.latest_holdings, limit=8)
        if holdings.empty:
            ax.text(0.5, 0.5, "No holdings snapshot", ha="center", va="center", color=_MUTED, transform=ax.transAxes)
            return
        weights = holdings["target_weight"].astype(float)
        colors = [_POSITIVE if value >= 0 else _NEGATIVE for value in weights]
        labels = [_safe_label(value) for value in holdings["symbol"]]
        ax.barh(labels, weights, color=colors, alpha=0.95)
        ax.axvline(0.0, color=_GRID, linewidth=1.0)
        ax.xaxis.set_major_formatter(FuncFormatter(_percent_label))
        ax.invert_yaxis()

    def _plot_sector_weights(self, ax, snapshot: PerformanceSnapshot) -> None:  # type: ignore[no-untyped-def]
        _style_axis(ax, "Sector Weights")
        sector_weights = snapshot.sectors.latest_weighted.sort_values(ascending=False).head(8)
        if sector_weights.empty:
            ax.text(0.5, 0.5, "No sector map", ha="center", va="center", color=_MUTED, transform=ax.transAxes)
            return
        labels = [_sector_label(index) for index in sector_weights.index]
        colors = [_POSITIVE if value >= 0 else _NEGATIVE for value in sector_weights.values]
        ax.bar(labels, sector_weights.values, color=colors, alpha=0.95)
        ax.axhline(0.0, color=_GRID, linewidth=1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(_percent_label))
        ax.tick_params(axis="x", rotation=30)

    def _plot_holdings_count(self, ax, snapshot: PerformanceSnapshot) -> None:  # type: ignore[no-untyped-def]
        _style_axis(ax, "Holdings Count")
        count = snapshot.exposure.holdings_count.dropna().astype(float)
        if count.empty:
            ax.text(0.5, 0.5, "No holdings count", ha="center", va="center", color=_MUTED, transform=ax.transAxes)
            return
        ax.plot(count.index, count.values, color=_BENCHMARK, linewidth=2.2)
        ax.fill_between(count.index, count.values, count.min(), color=_BENCHMARK, alpha=0.08)
        _format_date_axis(ax)

    def _plot_yearly_strip(self, ax, snapshot: PerformanceSnapshot) -> None:  # type: ignore[no-untyped-def]
        ax.set_facecolor(_PANEL)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("Return Range", loc="left", fontsize=12, fontweight="bold", color=_TEXT, pad=10)
        details = [
            ("Best Day", f"{snapshot.metrics.best_day:.1%}"),
            ("Worst Day", f"{snapshot.metrics.worst_day:.1%}"),
            ("Best Year", f"{snapshot.metrics.best_year:.1%}"),
            ("Worst Year", f"{snapshot.metrics.worst_year:.1%}"),
            ("VaR 95", f"{snapshot.metrics.value_at_risk_95:.1%}"),
            ("CVaR 95", f"{snapshot.metrics.conditional_value_at_risk_95:.1%}"),
            ("Longest DD", f"{snapshot.metrics.longest_drawdown_days:.0f}d"),
            ("Current DD", f"{snapshot.metrics.current_drawdown:.1%}"),
        ]
        rows = 4
        cols = 2
        left = 0.035
        top = 0.91
        gap_x = 0.04
        gap_y = 0.035
        card_width = (0.93 - gap_x) / cols
        card_height = 0.17
        for idx, (label, value) in enumerate(details):
            row = idx // cols
            column = idx % cols
            x = left + column * (card_width + gap_x)
            y = top - row * (card_height + gap_y)
            card = patches.FancyBboxPatch(
                (x, y - card_height),
                card_width,
                card_height,
                boxstyle="round,pad=0.01,rounding_size=0.025",
                linewidth=0.0,
                facecolor=_CARD_COLORS[idx % len(_CARD_COLORS)],
                transform=ax.transAxes,
                clip_on=True,
            )
            ax.add_patch(card)
            ax.text(x + 0.025, y - 0.052, label, fontsize=8, color=_MUTED, transform=ax.transAxes, clip_on=True)
            ax.text(x + 0.025, y - 0.125, value, fontsize=11, fontweight="bold", color=_TEXT, transform=ax.transAxes, clip_on=True)


def write_figure_asset(fig: go.Figure, png_path: Path, *, require_png: bool = False) -> Path:
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = png_path.with_suffix(".html")
    try:
        fig.write_image(png_path)
        return png_path
    except Exception as exc:  # pragma: no cover - depends on environment image export support
        screenshot_error = _write_browser_screenshot(fig, html_path, png_path)
        if screenshot_error is None and png_path.exists():
            return png_path
        fig.write_html(html_path)
        if require_png:
            raise PlotExportError(png_path, html_path, screenshot_error or exc) from (screenshot_error or exc)
        return html_path


def _style_axis(ax, title: str) -> None:  # type: ignore[no-untyped-def]
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
        spine.set_linewidth(1.0)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=_TEXT, pad=10)


def _format_date_axis(ax) -> None:  # type: ignore[no-untyped-def]
    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _percent_label(value: float, _pos: float) -> str:
    return f"{value:.0%}"


def _subtitle(snapshot: PerformanceSnapshot) -> str:
    benchmark = snapshot.benchmark.name if snapshot.benchmark is not None else "Strategy Only"
    start = snapshot.strategy_equity.index.min()
    end = snapshot.strategy_equity.index.max()
    if start is not None and end is not None and not snapshot.strategy_equity.empty:
        window = f"{pd.Timestamp(start).date()} → {pd.Timestamp(end).date()}"
    else:
        window = "No date range"
    return f"{window}   ·   Benchmark: {benchmark}   ·   Profile: {snapshot.profile.value.title()}"


def _annualized_metric_note(snapshot: PerformanceSnapshot) -> str:
    if snapshot.has_benchmark:
        return "Annualized: CAGR, Sharpe, Information Ratio"
    return "Annualized: CAGR, Sharpe, Sortino"


def _monthly_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return returns
    monthly = (1.0 + returns.astype(float)).resample("ME").prod().sub(1.0)
    monthly.index = monthly.index.normalize()
    return monthly.rename("monthly_returns")


def _monthly_heatmap(returns: pd.Series) -> pd.DataFrame:
    monthly = _monthly_returns(returns)
    if monthly.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        }
    )
    return frame.pivot_table(index="year", columns="month", values="return", aggfunc="sum")


def _cumulative_returns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float, name="cumulative_return")
    return equity.div(float(equity.iloc[0])).sub(1.0).rename("cumulative_return")


def _drawdown(equity: pd.Series) -> pd.Series:
    clean = equity.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float, name="drawdown")
    return clean.div(clean.cummax()).sub(1.0).rename("drawdown")


def _yearly_returns(returns: pd.Series) -> pd.Series:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float, name="yearly_return")
    return (1.0 + clean).resample("YE").prod().sub(1.0).rename("yearly_return")


def _compact_number(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def _top_holdings(latest_holdings: pd.DataFrame, *, limit: int = 10) -> pd.DataFrame:
    if latest_holdings.empty:
        return latest_holdings

    ranked = latest_holdings.copy()
    if "abs_weight" not in ranked.columns and "target_weight" in ranked.columns:
        ranked["abs_weight"] = ranked["target_weight"].abs()
    return ranked.sort_values(["abs_weight", "symbol"], ascending=[False, True]).head(limit)


_SECTOR_LABELS = {
    "G10": "Energy",
    "G15": "Materials",
    "G20": "Industrials",
    "G25": "Consumer Discretionary",
    "G30": "Consumer Staples",
    "G35": "Health Care",
    "G40": "Financials",
    "G45": "Information Technology",
    "G50": "Communication Services",
    "G55": "Utilities",
    "G60": "Real Estate",
}


def _sector_label(value: object) -> str:
    text = str(value).strip()
    return _SECTOR_LABELS.get(text, text)


def _safe_label(value: object) -> str:
    text = str(value).strip()
    if text.isascii():
        return text
    match = re.search(r"\(([^)]+)\)", text)
    if match is not None and match.group(1).strip():
        return match.group(1).strip()
    digits = "".join(character for character in text if character.isdigit())
    if digits:
        return digits
    return "Holding"


_MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _write_browser_screenshot(fig: go.Figure, html_path: Path, png_path: Path) -> Exception | None:
    try:
        fig.write_html(html_path)
        from playwright.sync_api import sync_playwright

        width = max(1280, int(fig.layout.width or 1600))
        height = max(720, int(fig.layout.height or 900))

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.wait_for_selector(".plotly-graph-div", state="visible", timeout=15_000)
            page.wait_for_timeout(500)
            plot = page.locator(".plotly-graph-div").first
            plot.screenshot(path=str(png_path))
            browser.close()
        return None
    except Exception as exc:  # pragma: no cover - runtime/browser dependent
        return exc
