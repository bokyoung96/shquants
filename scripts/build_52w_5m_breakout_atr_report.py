from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from root import ROOT


DEFAULT_RESEARCH_DIR = (
    ROOT.results_path
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
)
DEFAULT_SOURCE_DIR = DEFAULT_RESEARCH_DIR / "variants" / "5m_new_high_only" / "fixed20"
DEFAULT_OUTPUT_DIR = DEFAULT_RESEARCH_DIR / "52w_5m_breakout_atr_final"


def profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0.0].sum())
    losses = float(returns[returns < 0.0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / abs(losses)


def compute_metrics(trades: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, Any]:
    returns = trades["net_return"] if not trades.empty else pd.Series(dtype=float)
    return {
        "trades": int(len(trades)),
        "tickers": int(trades["ticker"].nunique()) if "ticker" in trades and not trades.empty else 0,
        "final_return": float(ledger["equity"].iloc[-1] - 1.0) if not ledger.empty else 0.0,
        "mdd": float(ledger["drawdown"].min()) if not ledger.empty else 0.0,
        "avg_trade_return": float(returns.mean()) if not returns.empty else 0.0,
        "median_trade_return": float(returns.median()) if not returns.empty else 0.0,
        "hit_rate": float(returns.gt(0.0).mean()) if not returns.empty else 0.0,
        "profit_factor": profit_factor(returns),
        "max_active_positions": int(ledger["active_positions"].max()) if "active_positions" in ledger and not ledger.empty else 0,
        "avg_active_positions": float(ledger["active_positions"].mean()) if "active_positions" in ledger and not ledger.empty else 0.0,
        "worst_trade": float(returns.min()) if not returns.empty else 0.0,
        "best_trade": float(returns.max()) if not returns.empty else 0.0,
    }


def yearly_returns(trades: pd.DataFrame, *, slots: int = 20) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["year", "trades", "year_return", "avg_trade_return", "hit_rate"])
    working = trades.copy()
    working["year"] = pd.to_datetime(working["exit_time"]).dt.year
    grouped = working.groupby("year")["net_return"]
    return pd.DataFrame(
        {
            "year": grouped.size().index.astype(int),
            "trades": grouped.size().to_numpy(dtype=int),
            "year_return": (grouped.sum() / float(slots)).to_numpy(dtype=float),
            "avg_trade_return": grouped.mean().to_numpy(dtype=float),
            "hit_rate": grouped.apply(lambda item: item.gt(0.0).mean()).to_numpy(dtype=float),
        }
    ).reset_index(drop=True)


def central_return_window(
    returns_bps: pd.Series,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[float, float]:
    clean = returns_bps.dropna().astype(float)
    if clean.empty:
        return -1.0, 1.0
    low = float(clean.quantile(lower_q, interpolation="higher"))
    high = float(clean.quantile(upper_q, interpolation="lower"))
    if low == high:
        pad = max(1.0, abs(low) * 0.05)
        return low - pad, high + pad
    return low, high


def build_report(source_dir: Path = DEFAULT_SOURCE_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(source_dir / "selected_trades.csv", parse_dates=["signal_time", "entry_time", "exit_time"])
    ledger = pd.read_csv(source_dir / "fixed_notional_ledger.csv", parse_dates=["date"])
    ledger = ledger.set_index("date")
    metrics = compute_metrics(trades, ledger)
    yearly = yearly_returns(trades, slots=20)
    exits = trades["exit_reason"].value_counts().rename_axis("exit_reason").reset_index(name="trades") if "exit_reason" in trades else pd.DataFrame()

    metrics_frame = pd.DataFrame([metrics])
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    yearly.to_csv(output_dir / "yearly_returns.csv", index=False)
    exits.to_csv(output_dir / "exit_reasons.csv", index=False)
    trades.to_csv(output_dir / "selected_trades.csv", index=False)
    ledger.to_csv(output_dir / "fixed_notional_ledger.csv", index_label="date")
    write_performance_png(trades, ledger, output_dir / "performance.png")
    write_markdown(metrics, yearly, exits, output_dir / "report.md")
    return {
        "metrics": metrics,
        "yearly": yearly,
        "exit_reasons": exits,
        "output_dir": output_dir,
    }


def write_performance_png(trades: pd.DataFrame, ledger: pd.DataFrame, path: Path) -> None:
    returns_bps = trades["net_return"].astype(float) * 10_000.0 if not trades.empty else pd.Series(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=160, facecolor="#fbfaf7")
    axes[0, 0].plot(ledger.index, (ledger["equity"] - 1.0) * 100.0, color="#2f4f4f", linewidth=1.7)
    axes[0, 0].set_title("Fixed 20-slot cumulative return", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Return (%)")

    axes[1, 0].plot(ledger.index, ledger["drawdown"] * 100.0, color="#8f3d2f", linewidth=1.1)
    axes[1, 0].set_title("Drawdown", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("DD (%)")

    axes[0, 1].plot(ledger.index, ledger["active_positions"], color="#315f8c", linewidth=1.1)
    axes[0, 1].set_title("Active positions", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Count")

    bins = min(45, max(12, int(len(returns_bps) ** 0.5))) if len(returns_bps) else 12
    low, high = central_return_window(returns_bps)
    central = returns_bps.clip(lower=low, upper=high)
    axes[1, 1].hist(central, bins=bins, color="#b56a3a", alpha=0.76, edgecolor="white")
    axes[1, 1].axvline(0.0, color="#333333", linewidth=1.0)
    axes[1, 1].set_title("Trade net return distribution (1%-99% clipped)", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Net return (bps)")

    for ax in axes.ravel():
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(metrics: dict[str, Any], yearly: pd.DataFrame, exits: pd.DataFrame, path: Path) -> None:
    lines = [
        "# 52W High 5M Breakout + ATR Strategy",
        "",
        "## Strategy Schema",
        "",
        "- Universe: KOSPI200 historical members.",
        "- Entry: 52-week close-high breakout after 09:20 KST.",
        "- Confirmation: next 5-minute close must remain above the prior 52-week close high.",
        "- Fill: enter at the following 5-minute open after confirmation.",
        "- Exit: ATR touch stop at stop price, or daily close losing the prior 52-week close high.",
        "- Portfolio: fixed 20-slot notional, 5% per selected position.",
        "- Cost: 35bp round-trip cost already included in net returns.",
        "- No positivity filter.",
        "- No foreign/institution flow filter.",
        "- No weekly sector RS or daily volatility compression filter.",
        "",
        "## Performance",
        "",
        f"- Selected trades: {metrics['trades']:,}",
        f"- Unique tickers: {metrics['tickers']:,}",
        f"- Final fixed-notional return: {metrics['final_return'] * 100.0:.2f}%",
        f"- MDD: {metrics['mdd'] * 100.0:.2f}%",
        f"- Average trade return: {metrics['avg_trade_return'] * 10_000.0:.2f} bps",
        f"- Median trade return: {metrics['median_trade_return'] * 10_000.0:.2f} bps",
        f"- Hit rate: {metrics['hit_rate'] * 100.0:.2f}%",
        f"- Profit factor: {metrics['profit_factor']:.3f}",
        f"- Max active positions: {metrics['max_active_positions']}",
        f"- Average active positions: {metrics['avg_active_positions']:.2f}",
        f"- Worst trade: {metrics['worst_trade'] * 10_000.0:.2f} bps",
        f"- Best trade: {metrics['best_trade'] * 10_000.0:.2f} bps",
        "",
        "## Yearly Stability",
        "",
        "| year | trades | fixed_return_pct | avg_trade_bps | hit_rate_pct |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in yearly.itertuples(index=False):
        lines.append(f"| {row.year} | {row.trades} | {row.year_return * 100.0:.2f} | {row.avg_trade_return * 10_000.0:.2f} | {row.hit_rate * 100.0:.2f} |")

    lines.extend(["", "## Exit Reasons", "", "| reason | trades |", "| --- | ---: |"])
    for row in exits.itertuples(index=False):
        lines.append(f"| {row.exit_reason} | {row.trades} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Current evidence indicates the robust signal is the simple 52-week high plus 5-minute confirmation breakout. Previously tested positivity, flow, and multi-timeframe filters are excluded from this canonical report because they did not improve the selected trade set or capital efficiency enough to justify the added fitting risk.",
            "",
            "Next discussion item: whether candle body size or body-to-range ratio should be studied as a pre-declared diagnostic of breakout bar quality, not retrofitted after seeing results.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final report for the 52W high 5M breakout + ATR strategy.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_report(args.source_dir, args.output_dir)
    print(result["output_dir"])


if __name__ == "__main__":
    main()
