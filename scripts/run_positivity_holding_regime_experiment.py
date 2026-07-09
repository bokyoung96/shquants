from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtesting.data.kr_stock_5m import KrStock5mDataset
from root import ROOT
from scripts.run_flow_filtered_breakout_single import (
    _apply_daily_membership,
    _daily_research_features_from_close,
    config_from_json,
    load_daily_5m_matrices,
)
from scripts.tech_gamma_costs import net_return_after_costs
from scripts.tech_gamma_research_filters import load_research_feature_data
from scripts.verified_flow_backtest import fixed_slot_selection_audit, load_close_prices, load_trades


DEFAULT_RESEARCH_DIR = ROOT.results_path / "flow_filtered_breakout_single" / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode" / "research"
DEFAULT_CURRENT_DIR = DEFAULT_RESEARCH_DIR / "variants" / "current"
DEFAULT_OUTPUT_DIR = DEFAULT_RESEARCH_DIR / "positivity_holding_regime"


@dataclass(frozen=True, slots=True)
class PositivityExitRule:
    name: str
    mode: str
    threshold: float = 0.0
    fraction: float = 0.5
    days: int = 1

    @staticmethod
    def absolute_nonpositive() -> "PositivityExitRule":
        return PositivityExitRule(name="absolute_nonpositive", mode="absolute", threshold=0.0)

    @staticmethod
    def relative_decay(fraction: float) -> "PositivityExitRule":
        return PositivityExitRule(name=f"relative_decay_{int(fraction * 100)}", mode="relative", fraction=fraction)

    @staticmethod
    def consecutive_weakness(*, threshold: float, days: int) -> "PositivityExitRule":
        threshold_label = str(int(round(threshold * 100))).zfill(3)
        return PositivityExitRule(name=f"weak_{days}d_le_{threshold_label}", mode="consecutive", threshold=threshold, days=days)


def apply_positivity_exit_overlay(trades: pd.DataFrame, daily: pd.DataFrame, rule: PositivityExitRule) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    daily_groups = {
        str(ticker): group.assign(date=pd.to_datetime(group["date"]).dt.normalize()).sort_values("date").reset_index(drop=True)
        for ticker, group in daily.groupby("ticker", sort=True)
    }
    rows: list[pd.Series] = []
    for _, trade in trades.iterrows():
        updated = trade.copy()
        ticker_daily = daily_groups.get(str(trade["ticker"]))
        if ticker_daily is not None:
            exit_row = _first_positivity_exit_row(trade, ticker_daily, rule)
            if exit_row is not None:
                exit_price = float(exit_row["open"])
                entry_price = float(trade["entry_price"])
                gross = exit_price / entry_price - 1.0
                updated["exit_time"] = pd.Timestamp(exit_row["date"]) + pd.Timedelta(hours=9)
                updated["exit_price"] = exit_price
                updated["gross_return"] = gross
                updated["net_return"] = net_return_after_costs(gross)
                updated["exit_reason"] = f"positivity_{rule.name}"
        rows.append(updated)
    return pd.DataFrame(rows).sort_values(["entry_time", "ticker", "signal_time"]).reset_index(drop=True)


def run_experiment(current_dir: Path = DEFAULT_CURRENT_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = load_trades(current_dir / "base" / "intraday_trades.csv")
    config = config_from_json(current_dir / "base" / "config.json", start="2019-01-01")
    tickers = tuple(sorted(trades["ticker"].astype(str).unique()))
    dataset = KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    daily = build_daily_positivity_exit_frame(dataset, tickers, config)

    rules = [
        ("current", None),
        ("absolute_nonpositive", PositivityExitRule.absolute_nonpositive()),
        ("relative_decay_50", PositivityExitRule.relative_decay(0.5)),
        ("weak_2d_le_002", PositivityExitRule.consecutive_weakness(threshold=0.02, days=2)),
    ]
    close = load_close_prices(ROOT.parquet_path / "qw_adj_c.parquet", trades)
    ledgers: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for name, rule in rules:
        variant_trades = trades.copy() if rule is None else apply_positivity_exit_overlay(trades, daily, rule)
        audit, selected, skipped, fixed, _rebalanced = fixed_slot_selection_audit(variant_trades, close, max_positions=20)
        selected.to_csv(output_dir / f"{name}_selected_trades.csv", index=False)
        fixed.to_csv(output_dir / f"{name}_fixed_notional_ledger.csv", index_label="date")
        ledgers[name] = fixed
        rows.append(
            {
                "strategy": name,
                "input_trades": int(len(variant_trades)),
                "selected_trades": int(len(selected)),
                "skipped_trades": int(len(skipped)),
                "positivity_exits": int(selected["exit_reason"].astype(str).str.startswith("positivity_").sum()) if not selected.empty else 0,
                "fixed_return": float(audit.fixed_notional_final_return),
                "mdd": float(audit.fixed_notional_mdd),
                "avg_trade_return": float(audit.selected_avg_trade_return),
                "hit_rate": float(audit.selected_hit_rate),
                "profit_factor": float(audit.selected_profit_factor),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "positivity_holding_regime_metrics.csv", index=False)
    (output_dir / "positivity_holding_regime_config.json").write_text(
        json.dumps([asdict(rule) if rule else {"name": "current"} for _, rule in rules], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_dashboard(ledgers, metrics, output_dir / "positivity_holding_regime_comparison.png")
    write_report(metrics, output_dir / "positivity_holding_regime_report.md")
    return metrics


def build_daily_positivity_exit_frame(dataset: KrStock5mDataset, tickers: tuple[str, ...], config: Any) -> pd.DataFrame:
    start = pd.Timestamp(config.start) - pd.Timedelta(days=config.high_lookback_days)
    close, _high, _low = load_daily_5m_matrices(dataset, tickers, start=start, end=config.end)
    open_ = load_daily_5m_open_matrix(dataset, tickers, start=start, end=config.end)
    data = load_research_feature_data(dataset.root.parent, tickers)
    features = _daily_research_features_from_close(close=close, config=config, data=data, tickers=tickers)
    open_long = open_.stack(future_stack=True).rename("open").reset_index()
    open_long.columns = ["date", "ticker", "open"]
    return features.merge(open_long, on=["date", "ticker"], how="left", sort=False).dropna(subset=["open", "positivity_spread"])


def load_daily_5m_open_matrix(
    dataset: KrStock5mDataset,
    tickers: tuple[str, ...],
    *,
    start: pd.Timestamp,
    end: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"):
        path = dataset.field_path(str(month), "o")
        if not path.exists():
            continue
        available = set(pq.read_schema(path).names)
        selected = [ticker for ticker in tickers if ticker in available]
        if not selected:
            continue
        monthly = pd.read_parquet(path, columns=selected, engine="pyarrow")
        monthly.index = pd.to_datetime(monthly.index).normalize()
        parts.append(monthly.groupby(level=0).first())
    if not parts:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers)
    open_ = pd.concat(parts).sort_index().loc[pd.Timestamp(start).normalize() : pd.Timestamp(end).normalize()].reindex(columns=tickers)
    return _apply_daily_membership(open_, dataset.root.parent)


def write_dashboard(ledgers: dict[str, pd.DataFrame], metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), dpi=160, facecolor="#fbfaf7")
    fig.patch.set_facecolor("#fbfaf7")
    for name, ledger in ledgers.items():
        axes[0, 0].plot(ledger.index, (ledger["equity"] - 1.0) * 100.0, label=_label(name, metrics), linewidth=2.0)
        axes[1, 0].plot(ledger.index, ledger["drawdown"] * 100.0, label=name, linewidth=1.5)
        axes[1, 1].plot(ledger.index, ledger["active_positions"], label=name, linewidth=1.1, alpha=0.75)
    bars = metrics.set_index("strategy")
    axes[0, 1].bar(range(len(bars)), bars["positivity_exits"])
    axes[0, 1].set_xticks(range(len(bars)), bars.index, rotation=25, ha="right")
    axes[0, 1].set_title("Positivity exits in selected trades", loc="left", fontweight="bold")
    axes[0, 0].set_title("Fixed-notional cumulative return", loc="left", fontweight="bold")
    axes[1, 0].set_title("Drawdown", loc="left", fontweight="bold")
    axes[1, 1].set_title("Active positions", loc="left", fontweight="bold")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, path: Path) -> None:
    display = metrics.copy()
    display["fixed_return_pct"] = display["fixed_return"] * 100.0
    display["mdd_pct"] = display["mdd"] * 100.0
    display["avg_trade_bps"] = display["avg_trade_return"] * 10_000.0
    display["hit_rate_pct"] = display["hit_rate"] * 100.0
    lines = [
        "# Positivity Holding Regime Experiment",
        "",
        "Entry universe is unchanged from the current confirmed breakout strategy. Positivity is tested only as an earlier holding/exit regime.",
        "All variants use 35bp round-trip costs and fixed 20-slot notional accounting.",
        "",
        "| strategy | selected_trades | positivity_exits | fixed_return_pct | mdd_pct | avg_trade_bps | hit_rate_pct | profit_factor |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.selected_trades} | {row.positivity_exits} | {row.fixed_return_pct:.4f} | {row.mdd_pct:.4f} | {row.avg_trade_bps:.4f} | {row.hit_rate_pct:.4f} | {row.profit_factor:.4f} |"
        )
    lines.extend(
        [
            "",
            "Rules:",
            "- `absolute_nonpositive`: if the completed previous trading day's positivity spread is <= 0, exit next trading day open.",
            "- `relative_decay_50`: if the completed previous trading day's positivity spread is <= 50% of entry-day spread, exit next trading day open.",
            "- `weak_2d_le_002`: if positivity spread is <= 0.02 for two completed trading days in a row, exit next trading day open.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _first_positivity_exit_row(trade: pd.Series, daily: pd.DataFrame, rule: PositivityExitRule) -> pd.Series | None:
    entry_date = pd.Timestamp(trade["entry_time"]).normalize()
    original_exit_time = pd.Timestamp(trade["exit_time"])
    entry_rows = daily.loc[daily["date"].eq(entry_date)]
    if entry_rows.empty:
        return None
    entry_spread = float(entry_rows.iloc[0]["positivity_spread"])
    candidate_rows = daily.loc[daily["date"].gt(entry_date) & daily["date"].le(original_exit_time.normalize())].reset_index(drop=True)
    for index in range(1, len(candidate_rows)):
        decision_rows = candidate_rows.iloc[:index]
        execution_row = candidate_rows.iloc[index]
        execution_time = pd.Timestamp(execution_row["date"]) + pd.Timedelta(hours=9)
        if execution_time >= original_exit_time:
            continue
        if _rule_triggered(decision_rows, entry_spread, rule):
            return execution_row
    return None


def _rule_triggered(decision_rows: pd.DataFrame, entry_spread: float, rule: PositivityExitRule) -> bool:
    latest = float(decision_rows.iloc[-1]["positivity_spread"])
    if rule.mode == "absolute":
        return latest <= rule.threshold
    if rule.mode == "relative":
        return latest <= entry_spread * rule.fraction
    if rule.mode == "consecutive":
        if len(decision_rows) < rule.days:
            return False
        recent = decision_rows.tail(rule.days)["positivity_spread"].astype(float)
        return bool(recent.le(rule.threshold).all())
    raise ValueError(f"unknown positivity exit mode: {rule.mode}")


def _label(name: str, metrics: pd.DataFrame) -> str:
    row = metrics.loc[metrics["strategy"].eq(name)]
    if row.empty:
        return name
    item = row.iloc[0]
    return f"{name}: {float(item['fixed_return']) * 100.0:.1f}%, MDD {float(item['mdd']) * 100.0:.1f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test positivity as a holding/exit regime overlay.")
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.current_dir, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
