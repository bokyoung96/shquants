from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtesting.data.kr_stock_5m import KrStock5mDataset
from root import ROOT
from scripts.run_flow_filtered_breakout_single import (
    _daily_research_features_from_close,
    config_from_json,
    load_daily_5m_matrices,
)
from scripts.tech_gamma_research_filters import load_research_feature_data
from scripts.tech_gamma_universe import kospi200_tickers
from scripts.verified_flow_backtest import fixed_slot_selection_audit, profit_factor


DEFAULT_RESEARCH_DIR = (
    ROOT.results_path
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
)
DEFAULT_EVENT_FORWARD = DEFAULT_RESEARCH_DIR / "52w_event_study" / "event_forward_returns.csv"
DEFAULT_EVENT_TRADES = DEFAULT_RESEARCH_DIR / "variants" / "5m_new_high_only" / "base" / "intraday_trades.csv"
DEFAULT_CONFIG = DEFAULT_RESEARCH_DIR / "variants" / "5m_new_high_only" / "base" / "config.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RESEARCH_DIR / "adr_regime_positivity_experiment"
DEFAULT_VARIANTS = {
    "baseline": "keep_baseline",
    "always_pos_gt0": "keep_always_pos_gt0",
    "always_pos_margin": "keep_always_pos_margin",
    "adr_pos_gt0": "keep_adr_pos_gt0",
    "adr_pos_margin": "keep_adr_pos_margin",
}


def compute_adr_regime(
    close: pd.DataFrame,
    *,
    lookback: int = 20,
    threshold: float = 1.0,
    min_periods: int | None = None,
) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if close.empty:
        return pd.DataFrame(columns=["advancers", "decliners", "adr", "adr20", "advance_ratio20", "adr_regime"])

    resolved_min_periods = min_periods if min_periods is not None else max(5, lookback // 2)
    returns = close.pct_change(fill_method=None)
    valid = returns.notna()
    advancers = returns.gt(0.0).where(valid).sum(axis=1)
    decliners = returns.lt(0.0).where(valid).sum(axis=1)
    active = valid.sum(axis=1)
    adr = advancers / decliners.mask(decliners.eq(0), np.nan)
    adr20 = adr.rolling(lookback, min_periods=resolved_min_periods).mean().shift(1)
    advance_ratio20 = (
        advancers.rolling(lookback, min_periods=resolved_min_periods).sum()
        / active.rolling(lookback, min_periods=resolved_min_periods).sum().mask(lambda value: value.eq(0), np.nan)
    ).shift(1)
    regime = pd.Series("unknown", index=close.index, dtype="object")
    regime.loc[adr20.gt(threshold)] = "broad"
    regime.loc[adr20.le(threshold)] = "narrow"
    return pd.DataFrame(
        {
            "advancers": advancers.astype(float),
            "decliners": decliners.astype(float),
            "adr": adr,
            "adr20": adr20,
            "advance_ratio20": advance_ratio20,
            "adr_regime": regime,
        },
        index=pd.DatetimeIndex(close.index).normalize(),
    )


def apply_variant_flags(events: pd.DataFrame, *, positivity_margin: float = 0.02) -> pd.DataFrame:
    if "positivity_spread" not in events.columns:
        raise ValueError("events must include positivity_spread")
    if "adr_regime" not in events.columns:
        raise ValueError("events must include adr_regime")
    result = events.copy()
    spread = pd.to_numeric(result["positivity_spread"], errors="coerce")
    pos_gt0 = spread.gt(0.0)
    pos_margin = spread.ge(float(positivity_margin))
    non_narrow_regime = result["adr_regime"].fillna("unknown").ne("narrow")

    result["keep_baseline"] = True
    result["keep_always_pos_gt0"] = pos_gt0.fillna(False)
    result["keep_always_pos_margin"] = pos_margin.fillna(False)
    result["keep_adr_pos_gt0"] = (non_narrow_regime | pos_gt0).fillna(False)
    result["keep_adr_pos_margin"] = (non_narrow_regime | pos_margin).fillna(False)
    return result


def summarize_event_variants(
    events: pd.DataFrame,
    *,
    variants: dict[str, str] = DEFAULT_VARIANTS,
    top_winner_fraction: float = 0.05,
    horizon: int = 20,
) -> pd.DataFrame:
    forward_column = f"event_entry_return_{horizon}d"
    net_column = f"event_entry_net_return_{horizon}d"
    excess_column = f"excess_return_{horizon}d"
    valid = events.dropna(subset=[forward_column]).copy()
    if valid.empty:
        return pd.DataFrame()

    winner_count = max(1, int(np.ceil(len(valid) * top_winner_fraction)))
    top_winner_index = set(valid.nlargest(winner_count, forward_column).index)
    rows: list[dict[str, float | int | str]] = []
    for variant, flag in variants.items():
        if flag not in valid.columns:
            continue
        selected = valid.loc[valid[flag].astype(bool)].copy()
        entry_net = pd.to_numeric(selected[net_column], errors="coerce").dropna()
        excess = pd.to_numeric(selected[excess_column], errors="coerce").dropna()
        broad_share = _share(selected, "adr_regime", "broad")
        narrow_share = _share(selected, "adr_regime", "narrow")
        rows.append(
            {
                "variant": variant,
                "events": int(len(selected)),
                "kept_ratio": float(len(selected) / len(valid)),
                "top_winner_capture": float(selected.index.isin(top_winner_index).sum() / winner_count),
                f"entry_net_mean_{horizon}d": _mean(entry_net),
                f"entry_net_median_{horizon}d": _median(entry_net),
                f"entry_net_hit_rate_{horizon}d": _hit_rate(entry_net),
                f"excess_mean_{horizon}d": _mean(excess),
                f"excess_median_{horizon}d": _median(excess),
                f"excess_hit_rate_{horizon}d": _hit_rate(excess),
                "broad_regime_share": broad_share,
                "narrow_regime_share": narrow_share,
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    *,
    event_forward_path: Path = DEFAULT_EVENT_FORWARD,
    event_trades_path: Path = DEFAULT_EVENT_TRADES,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    adr_lookback: int = 20,
    adr_threshold: float = 1.0,
    max_positions: int = 20,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    forward = pd.read_csv(event_forward_path, parse_dates=["signal_time", "entry_time", "event_date"])
    trades = pd.read_csv(event_trades_path, parse_dates=["signal_time", "entry_time", "exit_time"])
    config = config_from_json(config_path, start="2019-01-01")
    dataset = KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    tickers = kospi200_tickers(ROOT.parquet_path, config)
    load_days = max(260, int(config.positivity_lookback_days) * 3, adr_lookback * 6)
    start = pd.Timestamp(forward["entry_time"].min()).normalize() - pd.Timedelta(days=load_days)
    end = max(
        pd.Timestamp(forward["entry_time"].max()).normalize() + pd.Timedelta(days=35),
        pd.Timestamp(trades["exit_time"].max()).normalize() + pd.Timedelta(days=5),
    )
    close, _high, _low = load_daily_5m_matrices(dataset, tickers, start=start, end=str(end))

    data = load_research_feature_data(ROOT.parquet_path, tickers)
    daily_features = _daily_research_features_from_close(close=close, config=config, data=data, tickers=tickers)
    adr = compute_adr_regime(close, lookback=adr_lookback, threshold=adr_threshold)
    enriched_forward = _attach_daily_features(forward, daily_features, adr)
    enriched_forward = apply_variant_flags(enriched_forward, positivity_margin=float(config.positivity_margin))

    event_metrics = summarize_event_variants(enriched_forward, horizon=20)
    strategy_metrics, variant_ledgers, variant_trades = summarize_strategy_variants(
        trades,
        enriched_forward,
        close,
        max_positions=max_positions,
    )

    enriched_forward.to_csv(output_dir / "adr_regime_filtered_events.csv", index=False)
    event_metrics.to_csv(output_dir / "adr_regime_event_metrics.csv", index=False)
    strategy_metrics.to_csv(output_dir / "adr_regime_strategy_metrics.csv", index=False)
    _write_variant_ledgers(output_dir, variant_ledgers, variant_trades)
    write_comparison_png(event_metrics, strategy_metrics, output_dir / "adr_regime_comparison.png")
    write_report(
        event_metrics,
        strategy_metrics,
        output_dir / "adr_regime_report.md",
        adr_lookback=adr_lookback,
        adr_threshold=adr_threshold,
        positivity_margin=float(config.positivity_margin),
    )
    return {
        "events": enriched_forward,
        "event_metrics": event_metrics,
        "strategy_metrics": strategy_metrics,
        "output_dir": output_dir,
    }


def summarize_strategy_variants(
    trades: pd.DataFrame,
    enriched_forward: pd.DataFrame,
    close: pd.DataFrame,
    *,
    max_positions: int = 20,
    variants: dict[str, str] = DEFAULT_VARIANTS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    flags = enriched_forward[
        ["ticker", "signal_time", "entry_time", *[flag for flag in variants.values() if flag in enriched_forward.columns]]
    ].copy()
    working = trades.merge(flags, on=["ticker", "signal_time", "entry_time"], how="left", validate="one_to_one")
    rows: list[dict[str, float | int | str]] = []
    ledgers: dict[str, pd.DataFrame] = {}
    selected_trades: dict[str, pd.DataFrame] = {}
    for variant, flag in variants.items():
        if flag not in working.columns:
            continue
        filtered = working.loc[working[flag].fillna(False).astype(bool)].copy()
        audit, selected, skipped, fixed, _rebalanced = fixed_slot_selection_audit(filtered, close, max_positions=max_positions)
        returns = selected["net_return"] if not selected.empty else pd.Series(dtype=float)
        ledgers[variant] = fixed
        selected_trades[variant] = selected
        rows.append(
            {
                "variant": variant,
                "input_trades": int(len(filtered)),
                "selected_trades": int(len(selected)),
                "skipped_trades": int(len(skipped)),
                "fixed20_return": float(audit.fixed_notional_final_return),
                "fixed20_mdd": float(audit.fixed_notional_mdd),
                "avg_trade_return": _mean(returns),
                "median_trade_return": _median(returns),
                "hit_rate": _hit_rate(returns),
                "profit_factor": profit_factor(returns),
                "max_active_positions": int(audit.max_active_positions),
            }
        )
    return pd.DataFrame(rows), ledgers, selected_trades


def write_comparison_png(event_metrics: pd.DataFrame, strategy_metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=160, facecolor="#fbfaf7")
    event = event_metrics.set_index("variant")
    strategy = strategy_metrics.set_index("variant")
    order = [variant for variant in DEFAULT_VARIANTS if variant in event.index]
    labels = [_short_label(variant) for variant in order]
    colors = ["#4e6e73", "#b1764a", "#7d8f56", "#2f5f8c", "#8a5a7a"][: len(order)]

    axes[0, 0].bar(labels, event.loc[order, "events"], color=colors)
    axes[0, 0].set_title("Event count after filter", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Events")

    axes[0, 1].bar(labels, event.loc[order, "top_winner_capture"] * 100.0, color=colors)
    axes[0, 1].axhline(100.0, color="#222222", linewidth=0.8, linestyle="--")
    axes[0, 1].set_title("Top 5% winner capture", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Capture (%)")
    axes[0, 1].set_ylim(0, 110)

    axes[1, 0].bar(labels, event.loc[order, "entry_net_mean_20d"] * 100.0, color=colors)
    axes[1, 0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1, 0].set_title("20D event net mean", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Return (%)")

    strategy_order = [variant for variant in order if variant in strategy.index]
    axes[1, 1].bar([_short_label(v) for v in strategy_order], strategy.loc[strategy_order, "fixed20_return"] * 100.0, color=colors[: len(strategy_order)])
    axes[1, 1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1, 1].set_title("Fixed 20-slot ATR strategy return", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("Return (%)")

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=18)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(
    event_metrics: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    path: Path,
    *,
    adr_lookback: int,
    adr_threshold: float,
    positivity_margin: float,
) -> None:
    lines = [
        "# ADR Regime Positivity Experiment",
        "",
        "Purpose: test whether positivity spread should be used only when market breadth is narrow.",
        "",
        "## Rules",
        "",
        "- Baseline: keep all confirmed 52-week high + 5-minute confirmation events.",
        "- Always positivity > 0: keep only events whose stock positivity is above its sector benchmark.",
        f"- Always positivity margin: keep only events whose positivity spread is at least {positivity_margin:.2f}.",
        f"- ADR regime variants: if prior {adr_lookback}D ADR is above {adr_threshold:.1f}, keep all events; otherwise require the same positivity condition.",
        "- ADR is shifted by one trading day, so the event day close is not used for the regime decision.",
        "",
        "## Event-Level 20D Forward Return",
        "",
        "| variant | events | kept | top5 capture | 20D net mean | 20D net median | 20D hit | 20D excess mean | broad share | narrow share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in event_metrics.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.events} | {row.kept_ratio * 100:.1f}% | {row.top_winner_capture * 100:.1f}% | "
            f"{getattr(row, 'entry_net_mean_20d') * 100:.2f}% | {getattr(row, 'entry_net_median_20d') * 100:.2f}% | "
            f"{getattr(row, 'entry_net_hit_rate_20d') * 100:.1f}% | {getattr(row, 'excess_mean_20d') * 100:.2f}% | "
            f"{row.broad_regime_share * 100:.1f}% | {row.narrow_regime_share * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Fixed 20-Slot ATR Strategy",
            "",
            "| variant | input | selected | skipped | return | mdd | avg trade | median trade | hit | pf | max active |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in strategy_metrics.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.input_trades} | {row.selected_trades} | {row.skipped_trades} | "
            f"{row.fixed20_return * 100:.2f}% | {row.fixed20_mdd * 100:.2f}% | {row.avg_trade_return * 100:.2f}% | "
            f"{row.median_trade_return * 100:.2f}% | {row.hit_rate * 100:.1f}% | {row.profit_factor:.2f} | {row.max_active_positions} |"
        )
    lines.extend(["", "![ADR regime comparison](adr_regime_comparison.png)", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _attach_daily_features(forward: pd.DataFrame, daily_features: pd.DataFrame, adr: pd.DataFrame) -> pd.DataFrame:
    working = forward.copy()
    working["event_date"] = pd.to_datetime(working["event_date"]).dt.normalize()
    feature_columns = ["date", "ticker", "daily_positivity", "positivity_benchmark", "positivity_spread"]
    features = daily_features[[column for column in feature_columns if column in daily_features.columns]].copy()
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    merged = working.merge(features, left_on=["event_date", "ticker"], right_on=["date", "ticker"], how="left", validate="many_to_one")
    merged = merged.drop(columns=["date"])
    adr_features = adr.reset_index(names="event_date")
    adr_features["event_date"] = pd.to_datetime(adr_features["event_date"]).dt.normalize()
    return merged.merge(adr_features, on="event_date", how="left", validate="many_to_one")


def _write_variant_ledgers(output_dir: Path, ledgers: dict[str, pd.DataFrame], selected_trades: dict[str, pd.DataFrame]) -> None:
    ledger_dir = output_dir / "variant_ledgers"
    trade_dir = output_dir / "variant_selected_trades"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    trade_dir.mkdir(parents=True, exist_ok=True)
    for variant, ledger in ledgers.items():
        ledger.to_csv(ledger_dir / f"{variant}.csv", index_label="date")
    for variant, trades in selected_trades.items():
        trades.to_csv(trade_dir / f"{variant}.csv", index=False)


def _share(frame: pd.DataFrame, column: str, value: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(frame[column].eq(value).mean())


def _mean(series: pd.Series) -> float:
    return float(series.mean()) if not series.empty else 0.0


def _median(series: pd.Series) -> float:
    return float(series.median()) if not series.empty else 0.0


def _hit_rate(series: pd.Series) -> float:
    return float(series.gt(0.0).mean()) if not series.empty else 0.0


def _short_label(variant: str) -> str:
    return {
        "baseline": "base",
        "always_pos_gt0": "pos>0",
        "always_pos_margin": "pos>=.02",
        "adr_pos_gt0": "ADR pos>0",
        "adr_pos_margin": "ADR pos>=.02",
    }.get(variant, variant)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test ADR-regime positivity filters on 52W 5m confirmed events.")
    parser.add_argument("--event-forward", type=Path, default=DEFAULT_EVENT_FORWARD)
    parser.add_argument("--event-trades", type=Path, default=DEFAULT_EVENT_TRADES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adr-lookback", type=int, default=20)
    parser.add_argument("--adr-threshold", type=float, default=1.0)
    parser.add_argument("--max-positions", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        event_forward_path=args.event_forward,
        event_trades_path=args.event_trades,
        config_path=args.config,
        output_dir=args.output_dir,
        adr_lookback=args.adr_lookback,
        adr_threshold=args.adr_threshold,
        max_positions=args.max_positions,
    )
    print(f"output_dir={result['output_dir']}")
    print(result["event_metrics"].to_string(index=False))
    print(result["strategy_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
