from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(equity: pd.Series | pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    equity.plot(ax=ax, linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drawdown(drawdown: pd.Series | pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    drawdown.plot(ax=ax, linewidth=1.2)
    ax.set_title("Drawdown")
    ax.set_xlabel("")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights(weights: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    weights.plot(ax=ax, linewidth=1.2)
    ax.set_title("Portfolio Weights")
    ax.set_xlabel("")
    ax.set_ylabel("Weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(equity: pd.DataFrame, drawdown: pd.DataFrame, weights: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.4]},
    )

    equity.plot(ax=axes[0], linewidth=1.5)
    axes[0].set_title("Strategy vs Benchmark")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)

    drawdown.plot(ax=axes[1], linewidth=1.2)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)

    weights.plot(ax=axes[2], linewidth=1.2)
    axes[2].set_title("Portfolio Weights")
    axes[2].set_ylabel("Weight")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, alpha=0.25)
    axes[2].set_xlabel("")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
