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
from scripts.run_multi_timeframe_filter_experiment import apply_mtf_filter
from scripts.verified_flow_backtest import (
    FixedSlotSelectionAudit,
    fixed_notional_mtm_ledger,
    load_close_prices,
    load_trades,
    position_slots,
    profit_factor,
    rebalanced_mtm_ledger,
    select_fixed_slot_trades,
)


DEFAULT_RESEARCH_DIR = (
    ROOT.results_path
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
)
DEFAULT_VARIANTS_DIR = DEFAULT_RESEARCH_DIR / "variants"
DEFAULT_MTF_DIR = DEFAULT_RESEARCH_DIR / "multi_timeframe_filter_comparison"
DEFAULT_OUTPUT_DIR = DEFAULT_RESEARCH_DIR / "slot_priority_mtf_breakout"
MTF_COLUMNS = ("weekly_sector_rs_ok", "daily_vol_compression_ok")


@dataclass(frozen=True, slots=True)
class SlotPriorityVariant:
    name: str
    source: str
    max_positions: int
    accounting_slots: int = 20
    priority: bool = False
    required_mtf_columns: tuple[str, ...] = ()


def default_variants() -> list[SlotPriorityVariant]:
    return [
        SlotPriorityVariant("5m_only_max20", "5m_new_high_only", 20),
        SlotPriorityVariant("flow_confirmed_max20", "flow_only", 20),
        SlotPriorityVariant("flow_confirmed_max15", "flow_only", 15),
        SlotPriorityVariant("slot_priority_mtf_max15", "flow_only", 15, priority=True),
        SlotPriorityVariant("hard_filter_weekly_sector_daily_vol", "flow_only", 20, required_mtf_columns=MTF_COLUMNS),
    ]


def mtf_tier(trades: pd.DataFrame) -> pd.Series:
    sector = _bool_column(trades, "weekly_sector_rs_ok")
    vol = _bool_column(trades, "daily_vol_compression_ok")
    tier = pd.Series(4, index=trades.index, dtype="int64")
    tier.loc[vol] = 3
    tier.loc[sector] = 2
    tier.loc[sector & vol] = 1
    return tier


def select_mtf_priority_fixed_slot_trades(
    trades: pd.DataFrame,
    *,
    max_positions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if trades.empty:
        selected = trades.copy()
        selected["portfolio_skip_reason"] = pd.NA
        return selected, selected.copy()

    ordered = trades.assign(_mtf_tier=mtf_tier(trades))
    sort_columns = ["entry_time", "_mtf_tier", "ticker", "signal_time" if "signal_time" in ordered.columns else "exit_time"]
    ordered = ordered.sort_values(sort_columns, ascending=[True, True, True, True]).reset_index(drop=True)

    selected_rows: list[pd.Series] = []
    skipped_rows: list[pd.Series] = []
    open_exits: list[pd.Timestamp] = []
    for _, trade in ordered.iterrows():
        entry_time = pd.Timestamp(trade["entry_time"])
        open_exits = [exit_time for exit_time in open_exits if exit_time > entry_time]
        cleaned = trade.drop(labels=["_mtf_tier"])
        if len(open_exits) >= max_positions:
            skipped = cleaned.copy()
            skipped["portfolio_skip_reason"] = "max_positions"
            skipped_rows.append(skipped)
            continue
        accepted = cleaned.copy()
        accepted["portfolio_skip_reason"] = pd.NA
        selected_rows.append(accepted)
        open_exits.append(pd.Timestamp(trade["exit_time"]))

    columns = [column for column in ordered.columns if column != "_mtf_tier"] + ["portfolio_skip_reason"]
    selected = pd.DataFrame(selected_rows, columns=columns)
    skipped = pd.DataFrame(skipped_rows, columns=columns)
    return selected.reset_index(drop=True), skipped.reset_index(drop=True)


def slot_selection_audit(
    trades: pd.DataFrame,
    close: pd.DataFrame,
    *,
    max_positions: int,
    accounting_slots: int = 20,
    priority: bool = False,
) -> tuple[FixedSlotSelectionAudit, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if priority:
        selected, skipped = select_mtf_priority_fixed_slot_trades(trades, max_positions=max_positions)
    else:
        selected, skipped = select_fixed_slot_trades(trades, max_positions=max_positions)
    fixed, _fixed_missing = fixed_notional_mtm_ledger(selected, close, slots=accounting_slots)
    rebalanced, _rebalanced_missing = rebalanced_mtm_ledger(selected, close, slots=accounting_slots)
    selected_returns = selected["net_return"] if not selected.empty else pd.Series(dtype=float)
    skipped_returns = skipped["net_return"] if not skipped.empty else pd.Series(dtype=float)
    audit = FixedSlotSelectionAudit(
        max_positions=int(max_positions),
        slot_weight=float(1.0 / accounting_slots),
        input_trades=int(len(trades)),
        selected_trades=int(len(selected)),
        skipped_trades=int(len(skipped)),
        max_active_positions=position_slots(selected),
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
    sources = {source: load_trades(variants_dir / source / "base" / "intraday_trades.csv") for source in _variant_sources()}
    sources["flow_only"] = attach_mtf_features(sources["flow_only"], mtf_dir)
    all_trades = pd.concat([frame for frame in sources.values()], ignore_index=True)
    close = load_close_prices(ROOT.parquet_path / "qw_adj_c.parquet", all_trades)

    ledgers: dict[str, pd.DataFrame] = {}
    selected_by_name: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for spec in default_variants():
        trades = sources[spec.source].copy()
        if spec.required_mtf_columns:
            trades = apply_mtf_filter(trades, spec.required_mtf_columns)
        audit, selected, skipped, fixed, _rebalanced = slot_selection_audit(
            trades,
            close,
            max_positions=spec.max_positions,
            accounting_slots=spec.accounting_slots,
            priority=spec.priority,
        )
        selected.to_csv(output_dir / f"{spec.name}_selected_trades.csv", index=False)
        skipped.to_csv(output_dir / f"{spec.name}_skipped_trades.csv", index=False)
        fixed.to_csv(output_dir / f"{spec.name}_fixed_notional_ledger.csv", index_label="date")
        ledgers[spec.name] = fixed
        selected_by_name[spec.name] = selected
        avg_active = float(fixed["active_positions"].mean()) if not fixed.empty else 0.0
        avg_exposure = avg_active / float(spec.accounting_slots)
        rows.append(
            {
                "strategy": spec.name,
                "source": spec.source,
                "max_positions": int(spec.max_positions),
                "accounting_slots": int(spec.accounting_slots),
                "slot_weight": float(audit.slot_weight),
                "input_trades": int(audit.input_trades),
                "selected_trades": int(audit.selected_trades),
                "skipped_trades": int(audit.skipped_trades),
                "fixed_return": float(audit.fixed_notional_final_return),
                "mdd": float(audit.fixed_notional_mdd),
                "avg_trade_return": float(audit.selected_avg_trade_return),
                "hit_rate": float(audit.selected_hit_rate),
                "profit_factor": float(audit.selected_profit_factor),
                "max_active_positions": int(audit.max_active_positions),
                "avg_active_positions": avg_active,
                "avg_exposure": avg_exposure,
                "return_per_avg_exposure": float(audit.fixed_notional_final_return / avg_exposure) if avg_exposure > 0.0 else 0.0,
                "mdd_per_avg_exposure": float(audit.fixed_notional_mdd / avg_exposure) if avg_exposure > 0.0 else 0.0,
            }
        )
    metrics = pd.DataFrame(rows)
    baseline_selected = selected_by_name.get("flow_confirmed_max15")
    if baseline_selected is not None:
        metrics["selected_diff_vs_flow_max15"] = [
            selected_key_difference_count(baseline_selected, selected_by_name[strategy]) for strategy in metrics["strategy"]
        ]
    else:
        metrics["selected_diff_vs_flow_max15"] = 0
    metrics.to_csv(output_dir / "slot_priority_mtf_metrics.csv", index=False)
    (output_dir / "slot_priority_mtf_config.json").write_text(
        json.dumps([asdict(spec) for spec in default_variants()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_dashboard(ledgers, metrics, output_dir / "slot_priority_mtf_comparison.png")
    write_report(metrics, output_dir / "slot_priority_mtf_report.md")
    return metrics


def attach_mtf_features(trades: pd.DataFrame, mtf_dir: Path) -> pd.DataFrame:
    mtf = pd.read_csv(mtf_dir / "current_trades_with_mtf_features.csv", parse_dates=["signal_time", "entry_time", "exit_time"])
    columns = ["ticker", "signal_time", "entry_time", *MTF_COLUMNS]
    return (
        trades.merge(mtf[columns].drop_duplicates(["ticker", "signal_time", "entry_time"]), on=["ticker", "signal_time", "entry_time"], how="left", sort=False)
        .sort_values(["entry_time", "ticker", "signal_time"])
        .reset_index(drop=True)
    )


def write_dashboard(ledgers: dict[str, pd.DataFrame], metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=160, facecolor="#fbfaf7")
    colors = {
        "flow_confirmed_max20": "#244c66",
        "flow_confirmed_max15": "#6d8aa0",
        "slot_priority_mtf_max15": "#c4512d",
        "hard_filter_weekly_sector_daily_vol": "#628b48",
        "5m_only_max20": "#8b6f47",
    }
    for name, ledger in ledgers.items():
        if ledger.empty:
            continue
        color = colors.get(name)
        axes[0, 0].plot(ledger.index, (ledger["equity"] - 1.0) * 100.0, linewidth=1.6, label=_label(name, metrics), color=color)
        axes[1, 0].plot(ledger.index, ledger["drawdown"] * 100.0, linewidth=1.0, color=color)
        axes[1, 1].plot(ledger.index, ledger["active_positions"], linewidth=0.9, alpha=0.7, color=color)
    ordered = metrics.sort_values("return_per_avg_exposure", ascending=True)
    axes[0, 1].barh(ordered["strategy"], ordered["return_per_avg_exposure"] * 100.0, color="#c4512d")
    axes[0, 0].set_title("Fixed 5% cumulative return", loc="left", fontweight="bold")
    axes[0, 1].set_title("Return per average exposure", loc="left", fontweight="bold")
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
    display["avg_exposure_pct"] = display["avg_exposure"] * 100.0
    display["return_per_avg_exposure_pct"] = display["return_per_avg_exposure"] * 100.0
    lines = [
        "# Slot Priority MTF Breakout Comparison",
        "",
        "Anti-overfitting constraints:",
        "",
        "- No weighted factor score.",
        "- Positivity is excluded from the priority rule.",
        "- ATR exit, transaction cost, and 5% per-position fixed-notional accounting are unchanged.",
        "- Max-15 variants cap selection at 15 positions but keep accounting slots at 20, leaving unused capital as cash.",
        "",
        "| strategy | max_pos | input | selected | skipped | diff_vs_flow15 | fixed_return_pct | mdd_pct | avg_trade_bps | hit_rate | profit_factor | avg_exposure_pct | return_per_avg_exposure_pct | max_active |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.max_positions} | {row.input_trades} | {row.selected_trades} | {row.skipped_trades} | {row.selected_diff_vs_flow_max15} | {row.fixed_return_pct:.4f} | {row.mdd_pct:.4f} | {row.avg_trade_bps:.4f} | {row.hit_rate:.4f} | {row.profit_factor:.4f} | {row.avg_exposure_pct:.4f} | {row.return_per_avg_exposure_pct:.4f} | {row.max_active_positions} |"
        )
    lines.extend(
        [
            "",
            "Primary comparison: `slot_priority_mtf_max15` should be judged against `flow_confirmed_max15`, not against the higher-exposure max-20 strategy.",
            "",
            _interpretation(metrics),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def selected_key_difference_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
    key_columns = ["ticker", "signal_time", "entry_time", "exit_time"]
    missing = [column for column in key_columns if column not in left.columns or column not in right.columns]
    if missing:
        raise ValueError(f"missing selected key columns: {missing}")
    left_keys = set(map(tuple, left[key_columns].astype(str).to_numpy()))
    right_keys = set(map(tuple, right[key_columns].astype(str).to_numpy()))
    only_left = len(left_keys - right_keys)
    only_right = len(right_keys - left_keys)
    return max(only_left, only_right)


def _variant_sources() -> tuple[str, ...]:
    return tuple(dict.fromkeys(spec.source for spec in default_variants()))


def _bool_column(trades: pd.DataFrame, column: str) -> pd.Series:
    if column not in trades.columns:
        return pd.Series(False, index=trades.index)
    return trades[column].where(trades[column].notna(), False).astype(bool)


def _label(name: str, metrics: pd.DataFrame) -> str:
    row = metrics.loc[metrics["strategy"].eq(name)]
    if row.empty:
        return name
    item = row.iloc[0]
    return f"{name}: {float(item['fixed_return']) * 100.0:.1f}%, MDD {float(item['mdd']) * 100.0:.1f}%"


def _interpretation(metrics: pd.DataFrame) -> str:
    base = metrics.loc[metrics["strategy"].eq("flow_confirmed_max15")]
    priority = metrics.loc[metrics["strategy"].eq("slot_priority_mtf_max15")]
    if base.empty or priority.empty:
        return "Interpretation: primary comparison rows are missing."
    base_row = base.iloc[0]
    priority_row = priority.iloc[0]
    diff = int(priority_row["selected_diff_vs_flow_max15"])
    pf_delta = float(priority_row["profit_factor"] - base_row["profit_factor"])
    avg_delta_bps = float((priority_row["avg_trade_return"] - base_row["avg_trade_return"]) * 10_000.0)
    mdd_delta_pct = float((priority_row["mdd"] - base_row["mdd"]) * 100.0)
    return (
        "Interpretation: `slot_priority_mtf_max15` changed "
        f"{diff} selected trades versus `flow_confirmed_max15`. "
        f"Profit-factor delta is {pf_delta:.4f}, average-trade delta is {avg_delta_bps:.2f} bps, "
        f"and MDD delta is {mdd_delta_pct:.2f} percentage points. "
        "A very small selected-set difference means this simple priority rule is not yet a meaningful allocator improvement."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare simple MTF slot priority for the confirmed breakout strategy.")
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
