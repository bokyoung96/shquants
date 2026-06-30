from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_comparison_outputs(
    trades_by_strategy: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.DataFrame({name: _return_curve(trades) for name, trades in trades_by_strategy.items()}).ffill().fillna(1.0)
    drawdowns = curves.div(curves.cummax()).sub(1.0)
    baseline_count = len(next(iter(trades_by_strategy.values()))) if trades_by_strategy else 0
    baseline_returns = next(iter(trades_by_strategy.values()))["net_return"] if trades_by_strategy else pd.Series(dtype=float)
    baseline_top5_threshold = float(baseline_returns.quantile(0.95)) if not baseline_returns.empty else 0.0
    metrics = pd.DataFrame(
        [
            _metrics_row(name, trades, curves[name], drawdowns[name], baseline_count=baseline_count)
            for name, trades in trades_by_strategy.items()
        ]
    )
    yearly_entries = _yearly_entries(trades_by_strategy)
    right_tail = pd.DataFrame(
        [
            _right_tail_row(name, trades, baseline_top5_threshold=baseline_top5_threshold)
            for name, trades in trades_by_strategy.items()
        ]
    )
    curves.to_csv(output_dir / "comparison_equity_curves.csv", index_label="date")
    curves.to_csv(output_dir / "comparison_return_curves.csv", index_label="date")
    drawdowns.to_csv(output_dir / "comparison_drawdowns.csv", index_label="date")
    metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    yearly_entries.to_csv(output_dir / "yearly_entries.csv", index=False)
    right_tail.to_csv(output_dir / "right_tail_preservation.csv", index=False)
    _plot_comparison(curves, drawdowns, metrics, output_dir / "cumulative_mdd_comparison.png", title)


def load_trades(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[column for column in ("entry_time", "exit_time") if column in pd.read_csv(path, nrows=0).columns])


def _return_curve(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    exits = pd.to_datetime(trades["exit_time"]).dt.normalize()
    daily = trades.groupby(exits)["net_return"].sum().div(_position_slots(trades)).sort_index()
    return (1.0 + daily).cumprod()


def _metrics_row(name: str, trades: pd.DataFrame, equity: pd.Series, drawdown: pd.Series, *, baseline_count: int) -> dict[str, object]:
    returns = trades["net_return"] if not trades.empty else pd.Series(dtype=float)
    return {
        "strategy": name,
        "trades": int(len(trades)),
        "entry_reduction_vs_baseline": float(1.0 - len(trades) / baseline_count) if baseline_count else 0.0,
        "first_exit": str(pd.to_datetime(trades["exit_time"]).min()) if not trades.empty else "",
        "last_exit": str(pd.to_datetime(trades["exit_time"]).max()) if not trades.empty else "",
        "final_return_index": float(equity.iloc[-1]) if not equity.empty else 1.0,
        "cumulative_net_return": float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0,
        "position_slots": _position_slots(trades),
        "net_return_sum": float(returns.sum()) if not returns.empty else 0.0,
        "avg_net_bps": float(returns.mean() * 10_000.0) if not returns.empty else 0.0,
        "median_net_bps": float(returns.median() * 10_000.0) if not returns.empty else 0.0,
        "hit_rate": float(returns.gt(0.0).mean()) if not returns.empty else 0.0,
        "profit_factor": _profit_factor(returns),
        "mdd": float(drawdown.min()) if not drawdown.empty else 0.0,
        "avg_holding_days": _avg_holding_days(trades),
        "avg_concurrent_positions": _avg_concurrent_positions(trades),
        "max_concurrent_positions": _max_concurrent_positions(trades),
    }


def _profit_factor(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    gains = float(returns[returns > 0.0].sum())
    losses = float(returns[returns < 0.0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / abs(losses)


def _avg_holding_days(trades: pd.DataFrame) -> float:
    if trades.empty or "entry_time" not in trades.columns:
        return 0.0
    entry = pd.to_datetime(trades["entry_time"]).dt.normalize()
    exit_ = pd.to_datetime(trades["exit_time"]).dt.normalize()
    return float((exit_ - entry).dt.days.mean())


def _position_slots(trades: pd.DataFrame) -> int:
    return max(1, _max_concurrent_positions(trades))


def _position_events(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "entry_time" not in trades.columns:
        return pd.DataFrame(columns=["ts", "delta"])
    entries = pd.DataFrame({"ts": pd.to_datetime(trades["entry_time"]), "delta": 1})
    exits = pd.DataFrame({"ts": pd.to_datetime(trades["exit_time"]), "delta": -1})
    return pd.concat([entries, exits], ignore_index=True).sort_values(["ts", "delta"], ascending=[True, False])


def _avg_concurrent_positions(trades: pd.DataFrame) -> float:
    events = _position_events(trades)
    if events.empty:
        return 0.0
    concurrent = events["delta"].cumsum()
    return float(concurrent.mean())


def _max_concurrent_positions(trades: pd.DataFrame) -> int:
    events = _position_events(trades)
    if events.empty:
        return 0
    return int(events["delta"].cumsum().max())


def _yearly_entries(trades_by_strategy: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, trades in trades_by_strategy.items():
        if trades.empty:
            continue
        time_column = "signal_time" if "signal_time" in trades.columns else "exit_time"
        years = pd.to_datetime(trades[time_column]).dt.year
        for year, count in years.value_counts().sort_index().items():
            rows.append({"strategy": name, "year": int(year), "entries": int(count)})
    return pd.DataFrame(rows, columns=["strategy", "year", "entries"])


def _right_tail_row(name: str, trades: pd.DataFrame, *, baseline_top5_threshold: float) -> dict[str, object]:
    returns = trades["net_return"] if not trades.empty else pd.Series(dtype=float)
    above = returns[returns >= baseline_top5_threshold]
    return {
        "strategy": name,
        "baseline_top5_threshold": baseline_top5_threshold,
        "trades_at_or_above_baseline_top5": int(len(above)),
        "share_at_or_above_baseline_top5": float(len(above) / len(returns)) if len(returns) else 0.0,
        "net_return_sum_at_or_above_baseline_top5": float(above.sum()) if not above.empty else 0.0,
    }


def _plot_comparison(curves: pd.DataFrame, drawdowns: pd.DataFrame, metrics: pd.DataFrame, path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), dpi=160, sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]})
    colors = {"baseline": "#1d3557", "strengthened": "#c44e52"}

    for column in curves.columns:
        color = colors.get(column, None)
        axes[0].plot(curves.index, (curves[column] - 1.0) * 100.0, label=_label(column, metrics), linewidth=2.1, color=color)
    axes[0].set_title("Realized equal-slot portfolio return", loc="left", fontweight="bold")
    axes[0].set_ylabel("Portfolio return (%)")
    axes[0].legend(frameon=False, loc="upper left")
    _style_axis(axes[0])

    for column in drawdowns.columns:
        color = colors.get(column, None)
        axes[1].plot(drawdowns.index, drawdowns[column] * 100.0, label=column, linewidth=1.8, color=color)
    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_title("Drawdown / MDD", loc="left", fontweight="bold")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("")
    _style_axis(axes[1])

    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _label(column: str, metrics: pd.DataFrame) -> str:
    row = metrics.loc[metrics["strategy"].eq(column)]
    if row.empty:
        return column
    item = row.iloc[0]
    return f"{column}: return {item['cumulative_net_return'] * 100.0:.1f}%, MDD {item['mdd'] * 100.0:.1f}%, trades {int(item['trades']):,}"


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare cumulative equity and MDD for two flow-filtered breakout runs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--strengthened", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Flow Filtered Breakout Comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_comparison_outputs(
        {
            "baseline": load_trades(args.baseline),
            "strengthened": load_trades(args.strengthened),
        },
        output_dir=args.output_dir,
        title=args.title,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
