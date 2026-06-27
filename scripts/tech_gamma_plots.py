from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_performance_outputs(intraday: pd.DataFrame, overnight: pd.DataFrame, output: Path, title: str = "Scheme Performance") -> None:
    curves = equity_curves(intraday, overnight)
    curves.to_csv(output / "equity_curves.csv", index_label="date")
    plot_performance_subplots(curves, intraday, overnight, output / "performance_subplots.png", title)


def equity_curves(intraday: pd.DataFrame, overnight: pd.DataFrame) -> pd.DataFrame:
    returns = pd.concat(
        [
            _daily_returns(intraday, "intraday"),
            _daily_returns(overnight, "overnight"),
        ],
        axis=1,
    ).fillna(0.0)
    returns["combined"] = returns.sum(axis=1)
    if returns.empty:
        return pd.DataFrame(columns=["intraday", "overnight", "combined"])
    return (1.0 + returns).cumprod()


def plot_performance_subplots(
    curves: pd.DataFrame,
    intraday: pd.DataFrame,
    overnight: pd.DataFrame,
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    _plot_cumulative_return(axes[0, 0], curves)
    _plot_drawdown(axes[1, 0], curves)
    _plot_yearly_returns(axes[0, 1], intraday, overnight)
    _plot_trade_distribution(axes[1, 1], intraday, overnight)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _daily_returns(frame: pd.DataFrame, name: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, name=name)
    exits = pd.to_datetime(frame["exit_time"]).dt.normalize()
    return frame.groupby(exits)["net_return"].mean().sort_index().rename(name)


def _plot_cumulative_return(ax: plt.Axes, curves: pd.DataFrame) -> None:
    if not curves.empty:
        ((curves - 1.0) * 100.0).plot(ax=ax, linewidth=1.6)
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.7)
    ax.set_title("Cumulative return", loc="left", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)


def _plot_drawdown(ax: plt.Axes, curves: pd.DataFrame) -> None:
    if not curves.empty:
        drawdown = curves.div(curves.cummax()).sub(1.0) * 100.0
        drawdown.plot(ax=ax, linewidth=1.4)
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.7)
    ax.set_title("Drawdown / MDD", loc="left", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)


def _plot_yearly_returns(ax: plt.Axes, intraday: pd.DataFrame, overnight: pd.DataFrame) -> None:
    yearly = pd.concat(
        [
            _yearly_returns(intraday, "intraday"),
            _yearly_returns(overnight, "overnight"),
        ],
        axis=1,
    ).fillna(0.0)
    if not yearly.empty:
        yearly["combined"] = yearly.sum(axis=1)
        (yearly * 100.0).plot(kind="bar", ax=ax, width=0.82)
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.7)
    ax.set_title("Yearly return contribution", loc="left", fontweight="bold")
    ax.set_ylabel("Return sum (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)


def _plot_trade_distribution(ax: plt.Axes, intraday: pd.DataFrame, overnight: pd.DataFrame) -> None:
    parts = []
    if not intraday.empty:
        parts.append(intraday.assign(strategy="intraday"))
    if not overnight.empty:
        parts.append(overnight.assign(strategy="overnight"))
    if parts:
        trades = pd.concat(parts, ignore_index=True)
        grouped = [group["net_return"].to_numpy(dtype=float) for _, group in trades.groupby("strategy", sort=True)]
        labels = [str(strategy) for strategy in sorted(trades["strategy"].unique())]
        ax.boxplot(grouped, tick_labels=labels, showfliers=True)
        ax.set_ylabel("Net return")
    ax.set_title("Trade net return distribution", loc="left", fontweight="bold")
    ax.set_xlabel("")
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", alpha=0.25)


def _yearly_returns(frame: pd.DataFrame, name: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, name=name)
    years = pd.to_datetime(frame["exit_time"]).dt.year
    return frame.groupby(years)["net_return"].sum().sort_index().rename(name)
