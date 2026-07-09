from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from root import ROOT


DEFAULT_INPUT_DIR = (
    ROOT.results_path
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
    / "multi_timeframe_filter_comparison"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "combo"
CURRENT = "current"
COMBO = "weekly_sector_rs_plus_daily_vol_compression"
LABELS = {
    CURRENT: "Current",
    COMBO: "Weekly sector RS + daily vol compression",
}


def load_combo_inputs(input_dir: Path = DEFAULT_INPUT_DIR) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    metrics = pd.read_csv(input_dir / "multi_timeframe_filter_metrics.csv")
    ledgers: dict[str, pd.DataFrame] = {}
    trades: dict[str, pd.DataFrame] = {}
    for name in (CURRENT, COMBO):
        ledgers[name] = pd.read_csv(input_dir / f"{name}_fixed_notional_ledger.csv", parse_dates=["date"]).set_index("date")
        trades[name] = pd.read_csv(input_dir / f"{name}_selected_trades.csv", parse_dates=["signal_time", "entry_time", "exit_time"])
    return ledgers, trades, metrics.loc[metrics["strategy"].isin([CURRENT, COMBO])].copy()


def build_combo_comparison_paper(input_dir: Path = DEFAULT_INPUT_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledgers, trades, metrics = load_combo_inputs(input_dir)
    png_path = output_dir / "paper.png"
    pdf_path = output_dir / "paper.pdf"
    report_path = output_dir / "summary.md"
    plot_combo_comparison_paper(ledgers=ledgers, trades=trades, metrics=metrics, path=png_path)
    plot_combo_comparison_paper(ledgers=ledgers, trades=trades, metrics=metrics, path=pdf_path)
    write_summary_report(metrics, report_path)
    return png_path, pdf_path, report_path


def plot_combo_comparison_paper(
    *,
    ledgers: dict[str, pd.DataFrame],
    trades: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.8), dpi=190, facecolor="#fbfaf7")
    colors = {CURRENT: "#1f3a5f", COMBO: "#b94e2f"}

    _plot_performance(axes[0, 0], ledgers, metrics, colors)
    _plot_drawdown(axes[0, 1], ledgers, metrics, colors)
    _plot_position(axes[1, 0], ledgers, colors)
    _plot_distribution(axes[1, 1], trades, colors)

    fig.tight_layout(pad=2.0, h_pad=2.1, w_pad=2.0)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary_report(metrics: pd.DataFrame, path: Path) -> None:
    display = metrics.copy()
    display["fixed_return_pct"] = display["fixed_return"] * 100.0
    display["mdd_pct"] = display["mdd"] * 100.0
    display["avg_trade_bps"] = display["avg_trade_return"] * 10_000.0
    display["compression_pct"] = display["compression_vs_current"] * 100.0
    lines = [
        "# Current vs Weekly Sector + Daily Vol",
        "",
        "| strategy | input | selected | compression_pct | fixed_return_pct | mdd_pct | avg_trade_bps | hit_rate | profit_factor | max_active |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in display.sort_values("strategy").itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.input_trades} | {row.selected_trades} | {row.compression_pct:.2f} | {row.fixed_return_pct:.4f} | {row.mdd_pct:.4f} | {row.avg_trade_bps:.4f} | {row.hit_rate:.4f} | {row.profit_factor:.4f} | {row.max_active_positions} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_performance(ax: plt.Axes, ledgers: dict[str, pd.DataFrame], metrics: pd.DataFrame, colors: dict[str, str]) -> None:
    for name in (CURRENT, COMBO):
        ledger = ledgers[name]
        returns = (ledger["equity"] - 1.0) * 100.0
        ax.plot(returns.index, returns, color=colors[name], linewidth=2.2, label=_legend(name, metrics))
    ax.axhline(0.0, color="#242424", linewidth=0.8, alpha=0.7)
    ax.set_title("Performance", loc="left", fontweight="bold")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(frameon=False, fontsize=8.2)
    _style_axis(ax)


def _plot_drawdown(ax: plt.Axes, ledgers: dict[str, pd.DataFrame], metrics: pd.DataFrame, colors: dict[str, str]) -> None:
    for name in (CURRENT, COMBO):
        drawdown = ledgers[name]["drawdown"] * 100.0
        ax.plot(drawdown.index, drawdown, color=colors[name], linewidth=1.5, label=_legend(name, metrics))
    ax.axhline(0.0, color="#242424", linewidth=0.8, alpha=0.7)
    ax.set_title("Drawdown", loc="left", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    _style_axis(ax)


def _plot_position(ax: plt.Axes, ledgers: dict[str, pd.DataFrame], colors: dict[str, str]) -> None:
    for name in (CURRENT, COMBO):
        active = ledgers[name]["active_positions"].astype(float)
        ax.plot(active.index, active, color=colors[name], linewidth=1.4, label=LABELS[name])
    ax.set_title("Position", loc="left", fontweight="bold")
    ax.set_ylabel("Active positions")
    _style_axis(ax)


def _plot_distribution(ax: plt.Axes, trades: dict[str, pd.DataFrame], colors: dict[str, str]) -> None:
    all_bps = pd.concat([trades[name]["net_return"].dropna().astype(float) * 10_000.0 for name in (CURRENT, COMBO)], ignore_index=True)
    q01, q95 = all_bps.quantile([0.01, 0.95])
    lower = max(-350.0, float(q01))
    upper = min(500.0, float(q95))
    for name in (CURRENT, COMBO):
        bps = trades[name]["net_return"].dropna().astype(float) * 10_000.0
        central = bps[(bps >= lower) & (bps <= upper)]
        ax.hist(central, bins=55, density=True, histtype="step", linewidth=1.8, color=colors[name], label=LABELS[name])
        ax.axvline(bps.mean(), color=colors[name], linewidth=1.2, linestyle="--", alpha=0.9)
    ax.axvline(0.0, color="#242424", linewidth=0.9)
    ax.set_title("Return distribution", loc="left", fontweight="bold")
    ax.set_xlabel("Trade return (bps)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8.0)
    _style_axis(ax)


def _legend(name: str, metrics: pd.DataFrame) -> str:
    row = metrics.loc[metrics["strategy"].eq(name)]
    if row.empty:
        return LABELS[name]
    item = row.iloc[0]
    return f"{LABELS[name]}: {float(item['fixed_return']) * 100.0:.1f}%, MDD {float(item['mdd']) * 100.0:.1f}%"


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#fbfaf7")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d7d0c6")
    ax.tick_params(colors="#333333", labelsize=8.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current vs MTF combo comparison paper.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png_path, pdf_path, report_path = build_combo_comparison_paper(args.input_dir, args.output_dir)
    print(png_path)
    print(pdf_path)
    print(report_path)


if __name__ == "__main__":
    main()
