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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from root import ROOT
from scripts.run_multi_timeframe_filter_experiment import (
    apply_mtf_filter,
)
from scripts.verified_flow_backtest import (
    FixedSlotSelectionAudit,
    fixed_notional_mtm_ledger,
    fixed_slot_selection_audit,
    load_close_prices,
    load_trades,
    profit_factor,
    rebalanced_mtm_ledger,
)


DEFAULT_RESEARCH_DIR = (
    ROOT.results_path
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
)
DEFAULT_VARIANTS_DIR = DEFAULT_RESEARCH_DIR / "variants"
DEFAULT_MTF_DIR = DEFAULT_RESEARCH_DIR / "multi_timeframe_filter_comparison"
DEFAULT_OUTPUT_DIR = DEFAULT_RESEARCH_DIR / "positivity_utilization_comparison"
MTF_COLUMNS = ("weekly_sector_rs_ok", "daily_vol_compression_ok")


@dataclass(frozen=True, slots=True)
class PositivityVariant:
    name: str
    source: str
    use_mtf_combo: bool = False
    priority: bool = False


def default_variants() -> list[PositivityVariant]:
    return [
        PositivityVariant("current_hard_filter", "current"),
        PositivityVariant("no_positivity_flow_only", "flow_only"),
        PositivityVariant("no_positivity_pos_rank_priority", "flow_only", priority=True),
        PositivityVariant("mtf_combo_with_positivity", "current", use_mtf_combo=True),
        PositivityVariant("mtf_combo_without_positivity", "flow_only", use_mtf_combo=True),
        PositivityVariant("mtf_combo_without_positivity_pos_rank_priority", "flow_only", use_mtf_combo=True, priority=True),
    ]


def attach_positivity_features(trades: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    working = trades.copy()
    working["signal_date"] = pd.to_datetime(working["signal_time"]).dt.normalize()
    candidate_cols = [
        column
        for column in ("date", "ticker", "daily_positivity", "positivity_benchmark", "positivity_spread", "sector_name")
        if column in candidates.columns
    ]
    features = candidates[candidate_cols].copy()
    features["signal_date"] = pd.to_datetime(features["date"]).dt.normalize()
    features = features.drop(columns=["date"])
    merged = working.merge(features, on=["signal_date", "ticker"], how="left", sort=False)
    merged["positivity_rank_pct"] = merged.groupby("signal_date")["positivity_spread"].rank(pct=True, method="average")
    return merged


def select_priority_fixed_slot_trades(
    trades: pd.DataFrame,
    *,
    max_positions: int,
    priority_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if trades.empty:
        selected = trades.copy()
        selected["portfolio_skip_reason"] = pd.NA
        return selected, selected.copy()
    priority = trades[priority_column] if priority_column in trades.columns else pd.Series(0.0, index=trades.index)
    ordered = trades.assign(_priority=priority.fillna(float("-inf")).astype(float))
    ordered = ordered.sort_values(["entry_time", "_priority", "ticker", "signal_time"], ascending=[True, False, True, True]).reset_index(drop=True)
    selected_rows: list[pd.Series] = []
    skipped_rows: list[pd.Series] = []
    open_exits: list[pd.Timestamp] = []
    for _, trade in ordered.iterrows():
        entry_time = pd.Timestamp(trade["entry_time"])
        open_exits = [exit_time for exit_time in open_exits if exit_time > entry_time]
        cleaned = trade.drop(labels=["_priority"])
        if len(open_exits) >= max_positions:
            skipped = cleaned.copy()
            skipped["portfolio_skip_reason"] = "max_positions"
            skipped_rows.append(skipped)
            continue
        accepted = cleaned.copy()
        accepted["portfolio_skip_reason"] = pd.NA
        selected_rows.append(accepted)
        open_exits.append(pd.Timestamp(trade["exit_time"]))
    columns = [column for column in ordered.columns if column != "_priority"] + ["portfolio_skip_reason"]
    selected = pd.DataFrame(selected_rows, columns=columns)
    skipped = pd.DataFrame(skipped_rows, columns=columns)
    return selected.reset_index(drop=True), skipped.reset_index(drop=True)


def priority_fixed_slot_selection_audit(
    trades: pd.DataFrame,
    close: pd.DataFrame,
    *,
    max_positions: int,
    priority_column: str,
) -> tuple[FixedSlotSelectionAudit, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected, skipped = select_priority_fixed_slot_trades(trades, max_positions=max_positions, priority_column=priority_column)
    fixed, _fixed_missing = fixed_notional_mtm_ledger(selected, close, slots=max_positions)
    rebalanced, _rebalanced_missing = rebalanced_mtm_ledger(selected, close, slots=max_positions)
    selected_returns = selected["net_return"] if not selected.empty else pd.Series(dtype=float)
    skipped_returns = skipped["net_return"] if not skipped.empty else pd.Series(dtype=float)
    audit = FixedSlotSelectionAudit(
        max_positions=int(max_positions),
        slot_weight=float(1.0 / max_positions),
        input_trades=int(len(trades)),
        selected_trades=int(len(selected)),
        skipped_trades=int(len(skipped)),
        max_active_positions=_position_slots(selected),
        fixed_notional_final_return=float(fixed["equity"].iloc[-1] - 1.0) if not fixed.empty else 0.0,
        fixed_notional_mdd=float(fixed["drawdown"].min()) if not fixed.empty else 0.0,
        rebalanced_final_return=float(rebalanced["equity"].iloc[-1] - 1.0) if not rebalanced.empty else 0.0,
        rebalanced_mdd=float(rebalanced["drawdown"].min()) if not rebalanced.empty else 0.0,
        selected_avg_trade_return=float(selected_returns.mean()) if not selected_returns.empty else 0.0,
        selected_hit_rate=float(selected_returns.gt(0.0).mean()) if not selected_returns.empty else 0.0,
        selected_profit_factor=profit_factor(selected_returns),
        skipped_avg_trade_return=float(skipped_returns.mean()) if not skipped_returns.empty else 0.0,
        skipped_hit_rate=float(skipped_returns.gt(0.0).mean()) if not skipped_returns.empty else 0.0,
    )
    return audit, selected, skipped, fixed, rebalanced


def run_experiment(
    variants_dir: Path = DEFAULT_VARIANTS_DIR,
    mtf_dir: Path = DEFAULT_MTF_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        name: _load_source(name, variants_dir, mtf_dir)
        for name in ("current", "flow_only")
    }
    all_trades = pd.concat([frame for frame in sources.values()], ignore_index=True)
    close = load_close_prices(ROOT.parquet_path / "qw_adj_c.parquet", all_trades)

    ledgers: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for spec in default_variants():
        trades = sources[spec.source].copy()
        if spec.use_mtf_combo:
            trades = apply_mtf_filter(trades, MTF_COLUMNS)
        if spec.priority:
            audit, selected, skipped, fixed, _rebalanced = priority_fixed_slot_selection_audit(
                trades,
                close,
                max_positions=20,
                priority_column="positivity_rank_pct",
            )
        else:
            audit, selected, skipped, fixed, _rebalanced = fixed_slot_selection_audit(trades, close, max_positions=20)
        selected.to_csv(output_dir / f"{spec.name}_selected_trades.csv", index=False)
        skipped.to_csv(output_dir / f"{spec.name}_skipped_trades.csv", index=False)
        fixed.to_csv(output_dir / f"{spec.name}_fixed_notional_ledger.csv", index_label="date")
        ledgers[spec.name] = fixed
        rows.append(
            {
                "strategy": spec.name,
                "source": spec.source,
                "input_trades": int(len(trades)),
                "selected_trades": int(len(selected)),
                "skipped_trades": int(len(skipped)),
                "fixed_return": float(audit.fixed_notional_final_return),
                "mdd": float(audit.fixed_notional_mdd),
                "avg_trade_return": float(audit.selected_avg_trade_return),
                "hit_rate": float(audit.selected_hit_rate),
                "profit_factor": float(audit.selected_profit_factor),
                "max_active_positions": int(audit.max_active_positions),
                "avg_selected_positivity_rank": float(selected["positivity_rank_pct"].mean()) if "positivity_rank_pct" in selected and not selected.empty else 0.0,
                "avg_skipped_positivity_rank": float(skipped["positivity_rank_pct"].mean()) if "positivity_rank_pct" in skipped and not skipped.empty else 0.0,
            }
        )
    metrics = pd.DataFrame(rows).sort_values("fixed_return", ascending=False).reset_index(drop=True)
    metrics.to_csv(output_dir / "positivity_utilization_metrics.csv", index=False)
    (output_dir / "positivity_utilization_config.json").write_text(
        json.dumps([asdict(spec) for spec in default_variants()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_dashboard(ledgers, metrics, output_dir / "positivity_utilization_comparison.png")
    write_report(metrics, output_dir / "positivity_utilization_report.md")
    return metrics


def write_dashboard(ledgers: dict[str, pd.DataFrame], metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=160, facecolor="#fbfaf7")
    for name, ledger in ledgers.items():
        if ledger.empty:
            continue
        axes[0, 0].plot(ledger.index, (ledger["equity"] - 1.0) * 100.0, linewidth=1.5, label=_label(name, metrics))
        axes[1, 0].plot(ledger.index, ledger["drawdown"] * 100.0, linewidth=1.0)
        axes[1, 1].plot(ledger.index, ledger["active_positions"], linewidth=0.9, alpha=0.65)
    ordered = metrics.sort_values("fixed_return", ascending=True)
    axes[0, 1].barh(ordered["strategy"], ordered["fixed_return"] * 100.0, color="#2f7ebc")
    axes[0, 0].set_title("Fixed-notional cumulative return", loc="left", fontweight="bold")
    axes[0, 1].set_title("Final return", loc="left", fontweight="bold")
    axes[1, 0].set_title("Drawdown", loc="left", fontweight="bold")
    axes[1, 1].set_title("Active positions", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, path: Path) -> None:
    display = metrics.copy()
    display["fixed_return_pct"] = display["fixed_return"] * 100.0
    display["mdd_pct"] = display["mdd"] * 100.0
    display["avg_trade_bps"] = display["avg_trade_return"] * 10_000.0
    lines = [
        "# Positivity Utilization Comparison",
        "",
        "| strategy | input | selected | fixed_return_pct | mdd_pct | avg_trade_bps | hit_rate | profit_factor | avg_selected_pos_rank | avg_skipped_pos_rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.input_trades} | {row.selected_trades} | {row.fixed_return_pct:.4f} | {row.mdd_pct:.4f} | {row.avg_trade_bps:.4f} | {row.hit_rate:.4f} | {row.profit_factor:.4f} | {row.avg_selected_positivity_rank:.4f} | {row.avg_skipped_positivity_rank:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_source(name: str, variants_dir: Path, mtf_dir: Path) -> pd.DataFrame:
    trades = load_trades(variants_dir / name / "base" / "intraday_trades.csv")
    candidates = pd.read_csv(variants_dir / name / "base" / "prefilter_candidates.csv", parse_dates=["date"])
    enriched = attach_positivity_features(trades, candidates)
    mtf = pd.read_csv(mtf_dir / "current_trades_with_mtf_features.csv", parse_dates=["signal_time", "entry_time", "exit_time"])
    mtf_columns = ["ticker", "signal_time", "entry_time", *MTF_COLUMNS]
    return (
        enriched.merge(mtf[mtf_columns].drop_duplicates(["ticker", "signal_time", "entry_time"]), on=["ticker", "signal_time", "entry_time"], how="left", sort=False)
        .sort_values(["entry_time", "ticker", "signal_time"])
        .reset_index(drop=True)
    )


def _position_slots(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 1
    entries = pd.DataFrame({"ts": pd.to_datetime(trades["entry_time"]), "delta": 1})
    exits = pd.DataFrame({"ts": pd.to_datetime(trades["exit_time"]), "delta": -1})
    events = pd.concat([entries, exits], ignore_index=True).sort_values(["ts", "delta"], ascending=[True, False])
    return max(1, int(events["delta"].cumsum().max()))


def _label(name: str, metrics: pd.DataFrame) -> str:
    row = metrics.loc[metrics["strategy"].eq(name)]
    if row.empty:
        return name
    item = row.iloc[0]
    return f"{name}: {float(item['fixed_return']) * 100.0:.1f}%, MDD {float(item['mdd']) * 100.0:.1f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare positivity as hard filter, removal, and priority ranking.")
    parser.add_argument("--variants-dir", type=Path, default=DEFAULT_VARIANTS_DIR)
    parser.add_argument("--mtf-dir", type=Path, default=DEFAULT_MTF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.variants_dir, args.mtf_dir, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
