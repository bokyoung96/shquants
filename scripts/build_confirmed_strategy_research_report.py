from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import shutil
import sys
from dataclasses import asdict, replace
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
from scripts.run_flow_filtered_breakout_single import config_from_json, run_fast_prefiltered_single_strategy
from scripts.validate_confirmed_breakout_strategy import run_validation
from scripts.verified_flow_backtest import fixed_slot_selection_audit, load_close_prices, load_trades, profit_factor
from backtesting.data.kr_stock_5m import KrStock5mDataset


DEFAULT_RESULT_DIR = ROOT.results_path / "flow_filtered_breakout_single" / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_DIR / "research"
DEFAULT_CONFIG_PATH = DEFAULT_RESULT_DIR / "base" / "config.json"
VARIANT_ORDER = ("5m_new_high_only", "positivity_only", "flow_only", "current")
CANONICAL_FULL = "flow_only"
LEGACY_POSITIVITY_FLOW = "current"
COMPRESSED_STRATEGY = "weekly_sector_rs_plus_daily_vol_compression"


def fixed20_metrics(strategy: str, trades: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, Any]:
    returns = trades["net_return"].astype(float) if not trades.empty else pd.Series(dtype=float)
    return {
        "strategy": strategy,
        "trades": int(len(trades)),
        "tickers": int(trades["ticker"].nunique()) if "ticker" in trades.columns and not trades.empty else 0,
        "final_return": round(float(ledger["equity"].iloc[-1] - 1.0), 12) if not ledger.empty else 0.0,
        "mdd": float(ledger["drawdown"].min()) if not ledger.empty else 0.0,
        "avg_net_return": float(returns.mean()) if not returns.empty else 0.0,
        "median_net_return": float(returns.median()) if not returns.empty else 0.0,
        "hit_rate": float(returns.gt(0.0).mean()) if not returns.empty else 0.0,
        "profit_factor": profit_factor(returns),
        "max_active_positions": int(ledger["active_positions"].max()) if "active_positions" in ledger.columns and not ledger.empty else 0,
    }


def factor_impact_table(metrics: pd.DataFrame) -> pd.DataFrame:
    ordered = metrics.set_index("strategy").reindex([name for name in VARIANT_ORDER if name in set(metrics["strategy"])]).reset_index()
    if ordered.empty:
        return ordered
    baseline = ordered.iloc[0]
    ordered["return_delta_vs_baseline"] = (ordered["final_return"].astype(float) - float(baseline["final_return"])).round(12)
    ordered["mdd_delta_vs_baseline"] = (ordered["mdd"].astype(float) - float(baseline["mdd"])).round(12)
    ordered["trade_reduction_vs_baseline"] = 1.0 - ordered["trades"].astype(float).divide(float(baseline["trades"]) if float(baseline["trades"]) else 1.0)
    if "prefilter_candidates" in ordered.columns:
        ordered["candidate_reduction_vs_baseline"] = 1.0 - ordered["prefilter_candidates"].astype(float).divide(
            float(baseline["prefilter_candidates"]) if float(baseline["prefilter_candidates"]) else 1.0
        )
    ordered["return_per_trade_bps"] = ordered["final_return"].astype(float).divide(ordered["trades"].replace(0, pd.NA).astype(float)).fillna(0.0) * 10_000.0
    return ordered


def write_comparison_dashboard(ledgers: dict[str, pd.DataFrame], metrics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), dpi=160, facecolor="#fbfaf7")
    fig.patch.set_facecolor("#fbfaf7")
    colors = {
        "5m_new_high_only": "#1d3557",
        "positivity_only": "#4c956c",
        "flow_only": "#f4a261",
        "current": "#c44e52",
    }

    for name, ledger in ledgers.items():
        if ledger.empty:
            continue
        axes[0, 0].plot(ledger.index, (ledger["equity"] - 1.0) * 100.0, label=_metric_label(name, metrics), linewidth=2.0, color=colors.get(name))
        axes[1, 0].plot(ledger.index, ledger["drawdown"] * 100.0, label=name, linewidth=1.7, color=colors.get(name))
        axes[1, 1].plot(ledger.index, ledger["active_positions"], label=name, linewidth=1.25, alpha=0.8, color=colors.get(name))

    axes[0, 0].set_title("Fixed-notional cumulative return", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Return (%)")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8.5)
    _style_axis(axes[0, 0])

    bars = metrics.set_index("strategy").reindex([name for name in VARIANT_ORDER if name in set(metrics["strategy"])])
    x = range(len(bars))
    axes[0, 1].bar(x, bars["trades"], color=[colors.get(name, "#888888") for name in bars.index])
    axes[0, 1].set_xticks(list(x), [_short_name(name) for name in bars.index], rotation=25, ha="right")
    axes[0, 1].set_title("Accepted trades after 20-slot cap", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Trades")
    _style_axis(axes[0, 1])

    axes[1, 0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1, 0].set_title("Drawdown", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Drawdown (%)")
    _style_axis(axes[1, 0])

    axes[1, 1].set_title("Active positions", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("Positions")
    _style_axis(axes[1, 1])

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_factor_impact_dashboard(metrics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    impact = factor_impact_table(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5), dpi=160, facecolor="#fbfaf7")
    fig.patch.set_facecolor("#fbfaf7")
    labels = [_short_name(name) for name in impact["strategy"]]
    x = range(len(impact))
    colors = ["#1d3557", "#4c956c", "#f4a261", "#c44e52"][: len(impact)]

    axes[0, 0].bar(x, impact["final_return"] * 100.0, color=colors)
    axes[0, 0].set_title("Final return by filter set", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Return (%)")
    axes[0, 0].set_xticks(list(x), labels, rotation=25, ha="right")
    _style_axis(axes[0, 0])

    axes[0, 1].bar(x, impact["trades"], color=colors)
    axes[0, 1].set_title("Entry compression", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Trades")
    axes[0, 1].set_xticks(list(x), labels, rotation=25, ha="right")
    _style_axis(axes[0, 1])

    axes[1, 0].bar(x, impact["return_delta_vs_baseline"] * 100.0, color=colors)
    axes[1, 0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1, 0].set_title("Return delta vs 5m-only baseline", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Delta (%)")
    axes[1, 0].set_xticks(list(x), labels, rotation=25, ha="right")
    _style_axis(axes[1, 0])

    axes[1, 1].bar(x, impact["trade_reduction_vs_baseline"] * 100.0, color=colors)
    axes[1, 1].set_title("Trade reduction vs baseline", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("Reduction (%)")
    axes[1, 1].set_xticks(list(x), labels, rotation=25, ha="right")
    _style_axis(axes[1, 1])

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_research(config_path: Path, output_dir: Path, *, force: bool = False, batch_size: int = 20) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_root = output_dir / "variants"
    variant_root.mkdir(exist_ok=True)
    config = config_from_json(config_path, start="2019-01-01")
    dataset = KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    close_path = ROOT.parquet_path / "qw_adj_c.parquet"

    ledgers: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    variant_audits: dict[str, Any] = {}
    for name, variant_config in variant_configs(config).items():
        variant_dir = variant_root / name
        if force and variant_dir.exists():
            shutil.rmtree(variant_dir)
        run_fast_prefiltered_single_strategy(variant_config, output_dir=variant_dir, batch_size=batch_size, dataset=dataset)
        trades = load_trades(variant_dir / "base" / "intraday_trades.csv")
        close = load_close_prices(close_path, trades)
        audit, selected, skipped, fixed, rebalanced = fixed_slot_selection_audit(trades, close, max_positions=20)
        fixed20_dir = variant_dir / "fixed20"
        fixed20_dir.mkdir(exist_ok=True)
        selected.to_csv(fixed20_dir / "selected_trades.csv", index=False)
        skipped.to_csv(fixed20_dir / "skipped_trades.csv", index=False)
        fixed.to_csv(fixed20_dir / "fixed_notional_ledger.csv", index_label="date")
        rebalanced.to_csv(fixed20_dir / "rebalanced_ledger.csv", index_label="date")
        (fixed20_dir / "audit.json").write_text(json.dumps(asdict(audit), ensure_ascii=False, indent=2), encoding="utf-8")
        ledgers[name] = fixed
        metrics_row = fixed20_metrics(name, selected, fixed)
        metrics_row["prefilter_candidates"] = int(len(pd.read_csv(variant_dir / "base" / "prefilter_candidates.csv")))
        metrics_row["input_trades"] = int(audit.input_trades)
        metrics_row["skipped_trades"] = int(audit.skipped_trades)
        metric_rows.append(metrics_row)
        variant_audits[name] = asdict(audit)

    metrics = pd.DataFrame(metric_rows)
    impact = factor_impact_table(metrics)
    yearly = yearly_walk_forward_table(ledgers)
    metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    impact.to_csv(output_dir / "factor_impact.csv", index=False)
    yearly.to_csv(output_dir / "walk_forward_yearly_returns.csv", index=False)
    pd.DataFrame({name: ledger["equity"] for name, ledger in ledgers.items()}).ffill().to_csv(output_dir / "comparison_equity.csv", index_label="date")
    pd.DataFrame({name: ledger["drawdown"] for name, ledger in ledgers.items()}).ffill().to_csv(output_dir / "comparison_drawdown.csv", index_label="date")
    write_comparison_dashboard(ledgers, metrics, output_dir / "baseline_vs_current_comparison.png")
    write_factor_impact_dashboard(metrics, output_dir / "factor_impact.png")
    validation = run_validation(variant_root / "current", output_dir / "bias_audit", dataset)
    report = write_markdown_report(output_dir, metrics, impact, yearly, validation, variant_audits)
    return {"metrics": metrics, "impact": impact, "validation": validation, "report": report}


def rewrite_existing_report(output_dir: Path) -> Path:
    metrics = pd.read_csv(output_dir / "comparison_metrics.csv")
    impact = pd.read_csv(output_dir / "factor_impact.csv")
    yearly = pd.read_csv(output_dir / "walk_forward_yearly_returns.csv")
    validation = json.loads((output_dir / "bias_audit" / "strategy_integrity_audit.json").read_text(encoding="utf-8"))
    variant_audits: dict[str, Any] = {}
    for name in VARIANT_ORDER:
        audit_path = output_dir / "variants" / name / "fixed20" / "audit.json"
        if audit_path.exists():
            variant_audits[name] = json.loads(audit_path.read_text(encoding="utf-8"))
    return write_markdown_report(output_dir, metrics, impact, yearly, validation, variant_audits)


def variant_configs(config: Any) -> dict[str, Any]:
    return {
        "5m_new_high_only": replace(config, use_positivity=False, factor_filter="none"),
        "positivity_only": replace(config, use_positivity=True, factor_filter="none"),
        "flow_only": replace(config, use_positivity=False, factor_filter="foreign_or_institution_positive"),
        "current": replace(config, use_positivity=True, factor_filter="foreign_or_institution_positive"),
    }


def write_markdown_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    impact: pd.DataFrame,
    yearly: pd.DataFrame,
    validation: dict[str, Any],
    variant_audits: dict[str, Any],
) -> Path:
    path = output_dir / "strategy_research_report.md"
    canonical = metrics.loc[metrics["strategy"].eq(CANONICAL_FULL)].iloc[0]
    legacy = metrics.loc[metrics["strategy"].eq(LEGACY_POSITIVITY_FLOW)].iloc[0]
    baseline = metrics.loc[metrics["strategy"].eq("5m_new_high_only")].iloc[0]
    compressed = _load_compressed_strategy_metric(output_dir)
    lines = [
        "# Confirmed Episode Strategy Research Report",
        "",
        "## Scope",
        "",
        "This report compares a 5-minute 52-week-new-high-only baseline, factor-filter variants, the selected canonical full strategy, and the compressed multi-timeframe variant.",
        "All variants use the same confirmed 5-minute breakout entry, daily continuation exit, ATR stop rule, transaction cost model, episode compression, and fixed 20-slot notional portfolio accounting.",
        "The original filter variants differ by positivity and foreign/institution flow filters. Follow-up experiments showed that positivity does not change the final selected trade set, so the selected canonical full schema now uses flow confirmation without positivity.",
        "",
        "## Selected Strategy Schema",
        "",
        "### Canonical Full Strategy",
        "",
        "- Name: `Flow-Confirmed 52-Week High Breakout Strategy`",
        "- Universe: KOSPI200 historical members.",
        "- Entry setup: 52-week high breakout after 09:20 KST, with next 5-minute close confirmation and next 5-minute open entry.",
        "- Filters: foreign or institution 60-day flow-to-cap confirmation.",
        "- Excluded from canonical schema: positivity hard filter, positivity rank priority, and positivity exit/regime overlays.",
        "- Exit: entry ATR touch stop at stop price, or daily close losing the prior 52-week close high.",
        "- Portfolio accounting: fixed 20-slot notional, 5% per selected position, 35bp round-trip costs.",
        "",
        "### Compressed Strategy Variant",
        "",
        "- Name: `Sector-Relative Volatility-Compressed Breakout Strategy`",
        "- Starts from the canonical full strategy mechanics and keeps the same entry/exit execution rules.",
        "- Adds `weekly_sector_rs_ok`: prior completed week 12-week stock return must exceed same-sector 12-week average return.",
        "- Adds `daily_vol_compression_ok`: prior-day 20-day realized volatility must be less than or equal to prior-day 60-day realized volatility.",
        "- Purpose: reduce trade count and operating complexity, not maximize total fixed-notional return.",
        "",
        _compressed_summary_block(compressed),
        "",
        "## Performance Summary",
        "",
        _markdown_table(
            metrics.assign(final_return=metrics["final_return"] * 100.0, mdd=metrics["mdd"] * 100.0, hit_rate=metrics["hit_rate"] * 100.0),
            ["strategy", "prefilter_candidates", "input_trades", "trades", "skipped_trades", "final_return", "mdd", "hit_rate", "profit_factor"],
        ),
        "",
        "## Factor Impact",
        "",
        _markdown_table(
            impact.assign(
                return_delta_vs_baseline=impact["return_delta_vs_baseline"] * 100.0,
                trade_reduction_vs_baseline=impact["trade_reduction_vs_baseline"] * 100.0,
                candidate_reduction_vs_baseline=impact.get("candidate_reduction_vs_baseline", pd.Series(0.0, index=impact.index)) * 100.0,
            ),
            ["strategy", "prefilter_candidates", "trades", "return_delta_vs_baseline", "candidate_reduction_vs_baseline", "trade_reduction_vs_baseline", "return_per_trade_bps"],
        ),
        "",
        "Interpretation:",
        "",
        "- Positivity materially reduces daily prefilter candidates, but in this data it does not reduce final confirmed entries after the 5-minute confirmation and episode compression layers.",
        "- Removing positivity from the flow-confirmed strategy leaves the same selected trade set and the same fixed-notional result in the current evidence set.",
        "- Foreign/institution flow removes 108 selected trades versus the 5m-only baseline and reduces MDD slightly, but it also lowers final fixed-notional return by about 4.67 percentage points.",
        "- The canonical full strategy is therefore the flow-confirmed 52-week high breakout, not the legacy positivity+flow combination.",
        "- The compressed strategy is an optional operating-simplicity version: it reduces selected trades materially and improves trade quality/profit factor, while sacrificing total fixed-notional return.",
        "",
        "## Next Improvement Candidates",
        "",
        "1. Convert daily volatility compression from a hard filter into a selection priority when slots are scarce. The hard filter improves trade quality but cuts too much notional deployment.",
        "2. Test ATR stop multiplier and re-entry rules, while keeping ATR touch execution. ATR removal and close-confirmed ATR materially worsened drawdown in follow-up experiments.",
        "3. Add execution realism checks for ATR gaps and stop slippage, because the current stop-price fill is optimistic when the market gaps through the stop.",
        "4. Evaluate a separate positivity sleeve instead of forcing positivity into this breakout schema. Positivity did not alter selected trades as a hard filter or rank priority here, but may still be useful in a standalone cross-sectional momentum/rebalance design.",
        "5. Run a true forward/paper-trading holdout after schema selection. Further in-sample feature additions should be treated as exploratory until validated out of sample.",
        "",
        "## Walk-Forward Style Yearly Stability",
        "",
        "These are chronological yearly fixed-notional equity increments, not a parameter re-optimization walk-forward. They test whether the fixed rule survives across calendar regimes without changing parameters.",
        "",
        _markdown_table(yearly, ["strategy", "year", "year_return_pct", "year_end_equity"]),
        "",
        "## Bias And Integrity Audit",
        "",
        f"- Canonical full final fixed-notional return: {float(canonical['final_return']) * 100.0:.2f}%",
        f"- Legacy positivity+flow final fixed-notional return: {float(legacy['final_return']) * 100.0:.2f}%",
        f"- 5m-only baseline final fixed-notional return: {float(baseline['final_return']) * 100.0:.2f}%",
        f"- Canonical full trades: {int(canonical['trades']):,}",
        f"- Legacy positivity+flow trades: {int(legacy['trades']):,}",
        f"- 5m-only baseline trades: {int(baseline['trades']):,}",
        f"- Return accounting mismatches: {validation['return_accounting']['net_return_mismatches']}",
        f"- Entry price mismatches: {validation['source_entry_exit']['entry_price_mismatches']}",
        f"- Signal confirmation violations: {validation['source_entry_exit']['signal_confirmation_violations']}",
        f"- Exit condition violations: {validation['source_entry_exit']['exit_condition_violations']}",
        f"- KOSPI200 membership violations: {validation['universe_membership']['kospi200_membership_violations']}",
        "",
        "## Live Readiness Notes",
        "",
        "- Positivity is retained as rejected research evidence, not as a selected canonical filter.",
        "- Market cap and 60-day foreign/institution flow-to-cap are shifted by one trading day before signal use.",
        "- Weekly sector-relative strength in the compressed variant uses only the prior completed weekly bar.",
        "- Daily volatility compression in the compressed variant uses only returns available through the prior completed trading day.",
        "- The daily prefilter uses same-day 5-minute max close only to decide which candidate days require intraday loading; final entry is revalidated at 5-minute signal/confirmation bars.",
        "- ATR stop is modelled as stop-price fill once daily low breaches the stop. This is practical but optimistic if the market gaps through the stop or trades below the stop without available liquidity.",
        "- Walk-forward parameter retraining is not applicable because this is a fixed-rule strategy. The stronger live-readiness test is an untouched holdout or paper-trading period after strategy selection.",
        "- Backtest-overfitting remains a material research-process risk because the strategy was selected after multiple comparisons. Bailey, Borwein, Lopez de Prado, and Zhu propose PBO specifically for this type of investment-simulation selection risk.",
        "- Based on this audit, the code path is internally consistent, but I would not approve immediate full-size live deployment. The safer gate is paper trading or tiny notional live shadowing with real stop execution and slippage logging.",
        "",
        "## External Bias References",
        "",
        "- Bailey, Borwein, Lopez de Prado, and Zhu, The Probability of Backtest Overfitting: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253",
        "- Bailey et al., Statistical Overfitting and Backtest Performance: https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf",
        "- Lopez de Prado, Backtesting: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2606462",
        "- Risk.net summary of The Probability of Backtest Overfitting: https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting",
        "",
        "## Artifacts",
        "",
        "- `baseline_vs_current_comparison.png`",
        "- `factor_impact.png`",
        "- `multi_timeframe_filter_comparison/combo/paper.png`",
        "- `positivity_utilization_comparison/positivity_utilization_report.md`",
        "- `bias_audit/strategy_integrity_report.md`",
        "",
        "## Variant Audit Snapshot",
        "",
        "```json",
        json.dumps(variant_audits, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load_compressed_strategy_metric(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "multi_timeframe_filter_comparison" / "multi_timeframe_filter_metrics.csv"
    if not path.exists():
        return None
    metrics = pd.read_csv(path)
    row = metrics.loc[metrics["strategy"].eq(COMPRESSED_STRATEGY)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _compressed_summary_block(row: dict[str, Any] | None) -> str:
    if row is None:
        return "Compressed strategy metrics are not available in this output directory."
    return "\n".join(
        [
            "Compressed variant evidence:",
            "",
            f"- Selected trades: {int(row['selected_trades']):,}",
            f"- Compression vs canonical universe: {float(row['compression_vs_current']) * 100.0:.2f}%",
            f"- Fixed-notional final return: {float(row['fixed_return']) * 100.0:.2f}%",
            f"- Fixed-notional MDD: {float(row['mdd']) * 100.0:.2f}%",
            f"- Average selected trade return: {float(row['avg_trade_return']) * 10_000.0:.1f} bps",
            f"- Profit factor: {float(row['profit_factor']):.3f}",
            f"- Max active positions: {int(row['max_active_positions'])}",
        ]
    )


def yearly_walk_forward_table(ledgers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, ledger in ledgers.items():
        if ledger.empty:
            continue
        annual = ledger.copy()
        annual.index = pd.to_datetime(annual.index)
        previous_equity = 1.0
        for year, group in annual.groupby(annual.index.year, sort=True):
            year_end = float(group["equity"].iloc[-1])
            rows.append(
                {
                    "strategy": strategy,
                    "year": int(year),
                    "year_return_pct": round((year_end - previous_equity) * 100.0, 4),
                    "year_end_equity": round(year_end, 6),
                }
            )
            previous_equity = year_end
    return pd.DataFrame(rows, columns=["strategy", "year", "year_return_pct", "year_end_equity"])


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: f"{value:.4f}")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in selected.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _metric_label(name: str, metrics: pd.DataFrame) -> str:
    row = metrics.loc[metrics["strategy"].eq(name)]
    if row.empty:
        return _short_name(name)
    item = row.iloc[0]
    return f"{_short_name(name)}: {float(item['final_return']) * 100.0:.1f}%, MDD {float(item['mdd']) * 100.0:.1f}%, N={int(item['trades']):,}"


def _short_name(name: str) -> str:
    return {
        "5m_new_high_only": "5m only",
        "positivity_only": "positivity",
        "flow_only": "flow",
        "current": "current",
    }.get(name, name)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build baseline/current comparison, factor impact, and bias audit report.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rewrite markdown report from existing CSV/JSON artifacts without rerunning backtests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report_only:
        rewrite_existing_report(args.output_dir)
    else:
        run_research(args.config, args.output_dir, force=args.force, batch_size=args.batch_size)
    print(args.output_dir)


if __name__ == "__main__":
    main()
