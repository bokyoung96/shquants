from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def daily_returns_from_trades(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float, name="daily_return")
    exits = pd.to_datetime(trades["exit_time"]).dt.normalize()
    return trades.groupby(exits)["net_return"].mean().sort_index().rename("daily_return")


def metrics_table(returns: pd.Series, splits: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = [_metrics_row("full", returns)]
    for name, (start, end) in splits.items():
        rows.append(_metrics_row(name, returns.loc[pd.Timestamp(start) : pd.Timestamp(end)]))
    return pd.DataFrame(rows)


ROLLING_COLUMNS = (
    "segment",
    "start",
    "end",
    "observations",
    "total_return",
    "cagr",
    "avg_daily_bps",
    "sharpe",
    "max_drawdown",
    "hit_rate",
    "window_days",
)


def rolling_metrics(returns: pd.Series, windows: tuple[int, ...] = (63, 126)) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    clean = returns.sort_index().fillna(0.0)
    for window in windows:
        for end_index in range(window, len(clean) + 1):
            window_returns = clean.iloc[end_index - window : end_index]
            row = _metrics_row(f"rolling_{window}", window_returns)
            row["window_days"] = window
            row["start"] = str(window_returns.index.min().date())
            row["end"] = str(window_returns.index.max().date())
            rows.append(row)
    return pd.DataFrame(rows, columns=ROLLING_COLUMNS)


def monthly_heatmap(returns: pd.Series) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    monthly = returns.fillna(0.0).resample("ME").apply(lambda item: (1.0 + item).prod() - 1.0)
    frame = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "return": monthly.to_numpy()})
    return frame.pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))


def write_report_plots(returns: pd.Series, metrics: pd.DataFrame, rolling: pd.DataFrame, heatmap: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _plot_equity_dashboard(returns, metrics, rolling, output / "is_oos_equity_dashboard.png")
    _plot_monthly_heatmap(heatmap, output / "monthly_return_heatmap.png")


def _metrics_row(name: str, returns: pd.Series) -> dict[str, float | int | str]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return _empty_metrics(name)
    equity = (1.0 + clean).cumprod()
    years = max((clean.index.max() - clean.index.min()).days / 365.25, 1.0 / 365.25)
    total = float(equity.iloc[-1] - 1.0)
    daily_std = float(clean.std(ddof=0))
    return {
        "segment": name,
        "start": str(clean.index.min().date()),
        "end": str(clean.index.max().date()),
        "observations": int(len(clean)),
        "total_return": total,
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "avg_daily_bps": float(clean.mean() * 10_000.0),
        "sharpe": float(clean.mean() / daily_std * np.sqrt(252.0)) if daily_std > 0.0 else 0.0,
        "max_drawdown": float(equity.div(equity.cummax()).sub(1.0).min()),
        "hit_rate": float(clean.gt(0.0).mean()),
    }


def _empty_metrics(name: str) -> dict[str, float | int | str]:
    return {
        "segment": name,
        "start": "",
        "end": "",
        "observations": 0,
        "total_return": 0.0,
        "cagr": 0.0,
        "avg_daily_bps": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "hit_rate": 0.0,
    }


def _plot_equity_dashboard(returns: pd.Series, metrics: pd.DataFrame, rolling: pd.DataFrame, path: Path) -> None:
    clean = returns.sort_index().fillna(0.0)
    equity = (1.0 + clean).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=160)
    _plot_equity(axes[0, 0], equity)
    _plot_drawdown(axes[1, 0], drawdown)
    _plot_segments(axes[0, 1], metrics)
    _plot_rolling(axes[1, 1], rolling)
    fig.suptitle("Best Breakout Strategy IS/OOS Robustness", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_equity(ax: plt.Axes, equity: pd.Series) -> None:
    if not equity.empty:
        equity.sub(1.0).mul(100.0).plot(ax=ax, linewidth=1.8, color="#20639b")
    ax.axhline(0.0, color="#222", linewidth=0.8)
    ax.set_title("Cumulative return", loc="left", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.25)


def _plot_drawdown(ax: plt.Axes, drawdown: pd.Series) -> None:
    if not drawdown.empty:
        drawdown.mul(100.0).plot(ax=ax, linewidth=1.4, color="#b23a48")
    ax.axhline(0.0, color="#222", linewidth=0.8)
    ax.set_title("Drawdown", loc="left", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.25)


def _plot_segments(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    display = metrics.set_index("segment")[["total_return", "cagr", "max_drawdown"]].astype(float).mul(100.0)
    display.plot(kind="bar", ax=ax, width=0.78)
    ax.axhline(0.0, color="#222", linewidth=0.8)
    ax.set_title("Segment performance", loc="left", fontweight="bold")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)


def _plot_rolling(ax: plt.Axes, rolling: pd.DataFrame) -> None:
    if not rolling.empty:
        for window, group in rolling.groupby("window_days", sort=True):
            ax.plot(pd.to_datetime(group["end"]), group["cagr"].astype(float) * 100.0, label=f"{int(window)}d")
        ax.legend()
    ax.axhline(0.0, color="#222", linewidth=0.8)
    ax.set_title("Rolling CAGR", loc="left", fontweight="bold")
    ax.set_ylabel("CAGR (%)")
    ax.grid(alpha=0.25)


def _plot_monthly_heatmap(heatmap: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=160)
    matrix = heatmap.mul(100.0).astype(float)
    values = matrix.to_numpy()
    ax.set_title("Monthly return heatmap", loc="left", fontweight="bold")
    if matrix.empty:
        ax.text(0.5, 0.5, "No monthly returns", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return
    limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values)))) if np.isfinite(values).any() else 1.0
    image = ax.imshow(values, cmap="RdYlGn", aspect="auto", vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([str(year) for year in matrix.index])
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([str(month) for month in range(1, 13)])
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, pad=0.02, label="Monthly return (%)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
