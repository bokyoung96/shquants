from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULT_DIR = Path("results/flow_filtered_breakout_single/sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode")
DEFAULT_STRATEGY_NAME = "multi timeframe various momentum indicators strategy"


def load_fixed_notional_inputs(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    fixed20_dir = result_dir / "fixed20"
    ledger = pd.read_csv(fixed20_dir / "fixed_notional_ledger.csv", parse_dates=["date"]).set_index("date")
    trades = pd.read_csv(fixed20_dir / "selected_trades.csv", parse_dates=["signal_time", "entry_time", "exit_time"])
    audit = json.loads((fixed20_dir / "audit.json").read_text(encoding="utf-8"))
    return ledger, trades, audit


def plot_strategy_paper(
    *,
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    audit: dict[str, float],
    path: Path,
    strategy_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 11.2), dpi=190, facecolor="#fbfaf7")

    _plot_performance_path(axes[0, 0], ledger)
    _plot_drawdown(axes[0, 1], ledger)
    _plot_position_exposure(axes[1, 0], ledger)
    _plot_return_distribution(axes[1, 1], trades)

    fig.tight_layout(pad=2.0, h_pad=2.2, w_pad=2.0)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_strategy_paper(result_dir: Path, output_dir: Path, strategy_name: str = DEFAULT_STRATEGY_NAME) -> tuple[Path, Path]:
    ledger, trades, audit = load_fixed_notional_inputs(result_dir)
    slug = strategy_name.lower().replace(" ", "_").replace("-", "_")
    png_path = output_dir / f"{slug}_paper.png"
    pdf_path = output_dir / f"{slug}_paper.pdf"
    plot_strategy_paper(ledger=ledger, trades=trades, audit=audit, path=png_path, strategy_name=strategy_name)
    plot_strategy_paper(ledger=ledger, trades=trades, audit=audit, path=pdf_path, strategy_name=strategy_name)
    return png_path, pdf_path


def _plot_performance_path(ax: plt.Axes, ledger: pd.DataFrame) -> None:
    returns = (ledger["equity"] - 1.0) * 100.0
    ax.plot(returns.index, returns, color="#1f3a5f", linewidth=2.4)
    ax.fill_between(returns.index, returns.to_numpy(dtype=float), 0.0, color="#1f3a5f", alpha=0.09)
    ax.axhline(0.0, color="#242424", linewidth=0.8, alpha=0.7)
    ax.set_title("Performance", loc="left", fontweight="bold")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("")
    _style_axis(ax)


def _plot_drawdown(ax: plt.Axes, ledger: pd.DataFrame) -> None:
    drawdown = ledger["drawdown"] * 100.0
    ax.plot(drawdown.index, drawdown, color="#873343", linewidth=1.6)
    ax.fill_between(drawdown.index, drawdown.to_numpy(dtype=float), 0.0, color="#b44b4b", alpha=0.15)
    ax.axhline(0.0, color="#242424", linewidth=0.8, alpha=0.7)
    ax.set_title("Drawdown", loc="left", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("")
    _style_axis(ax)


def _plot_position_exposure(ax: plt.Axes, ledger: pd.DataFrame) -> None:
    active = ledger["active_positions"].astype(float)
    ax.fill_between(active.index, active.to_numpy(dtype=float), step="mid", color="#4f86c6", alpha=0.20)
    ax.plot(active.index, active, color="#365f8c", linewidth=1.6)
    ax.set_title("Position", loc="left", fontweight="bold")
    ax.set_ylabel("Active positions")
    ax.set_xlabel("")
    _style_axis(ax)


def _plot_return_distribution(ax: plt.Axes, trades: pd.DataFrame) -> None:
    bps = trades["net_return"].dropna().astype(float) * 10_000.0
    q01, q25, q50, q75, q95 = bps.quantile([0.01, 0.25, 0.50, 0.75, 0.95])
    low = max(-350.0, min(-100.0, q01))
    high = min(450.0, max(120.0, q95))
    central = bps[(bps >= low) & (bps <= high)]
    ax.hist(central, bins=70, color="#365f8c", edgecolor="#fbfaf7", linewidth=0.65, alpha=0.92)
    ax.axvspan(q25, q75, color="#f2b84b", alpha=0.18, label="IQR")
    ax.axvline(0.0, color="#242424", linewidth=1.0)
    ax.axvline(bps.mean(), color="#1b8f6a", linewidth=1.8)
    ax.axvline(q50, color="#d55e00", linewidth=1.8)
    ax.set_title("Return distribution", loc="left", fontweight="bold")
    ax.set_xlabel("Trade return (bps)")
    ax.set_ylabel("Trades")
    _style_axis(ax)


def _profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0.0].sum()
    losses = returns[returns < 0.0].sum()
    return float(wins / abs(losses)) if losses < 0.0 else 0.0


def _pct(value: float) -> str:
    return f"{float(value) * 100.0:.1f}%"


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#fbfaf7")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d7d0c6")
    ax.tick_params(colors="#333333", labelsize=8.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a one-page fixed-notional strategy paper.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strategy-name", default=DEFAULT_STRATEGY_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.result_dir / "paper")
    png_path, pdf_path = build_strategy_paper(args.result_dir, output_dir, args.strategy_name)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
