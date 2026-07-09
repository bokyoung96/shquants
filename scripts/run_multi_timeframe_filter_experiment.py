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

from backtesting.data.kr_stock_5m import KrStock5mDataset
from root import ROOT
from scripts.run_flow_filtered_breakout_single import _apply_daily_membership, config_from_json, load_daily_5m_matrices
from scripts.tech_gamma_research_filters import _aligned, load_research_feature_data
from scripts.verified_flow_backtest import fixed_slot_selection_audit, load_close_prices, load_trades


DEFAULT_RESEARCH_DIR = (
    ROOT.results_path
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
)
DEFAULT_CURRENT_DIR = DEFAULT_RESEARCH_DIR / "variants" / "current"
DEFAULT_OUTPUT_DIR = DEFAULT_RESEARCH_DIR / "multi_timeframe_filter_comparison"


@dataclass(frozen=True, slots=True)
class MultiTimeframeFilter:
    name: str
    required_columns: tuple[str, ...]


def default_filters() -> list[MultiTimeframeFilter]:
    return [
        MultiTimeframeFilter("current", ()),
        MultiTimeframeFilter("weekly_market_rs", ("weekly_market_rs_ok",)),
        MultiTimeframeFilter("weekly_sector_rs", ("weekly_sector_rs_ok",)),
        MultiTimeframeFilter("daily_vol_compression", ("daily_vol_compression_ok",)),
        MultiTimeframeFilter("weekly_sector_rs_plus_daily_vol_compression", ("weekly_sector_rs_ok", "daily_vol_compression_ok")),
    ]


def attach_completed_weekly_features(trades: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    working = trades.copy()
    working["signal_date"] = pd.to_datetime(working["signal_time"]).dt.normalize()
    weekly_working = weekly.copy()
    weekly_working["week_date"] = pd.to_datetime(weekly_working["week_date"]).dt.normalize()
    parts: list[pd.DataFrame] = []
    for ticker, ticker_trades in working.groupby("ticker", sort=False):
        ticker_weekly = weekly_working.loc[weekly_working["ticker"].astype(str).eq(str(ticker))].sort_values("week_date")
        if ticker_weekly.empty:
            enriched = ticker_trades.copy()
            for column in weekly_working.columns:
                if column not in {"ticker", "week_date"}:
                    enriched[column] = pd.NA
            parts.append(enriched)
            continue
        parts.append(
            pd.merge_asof(
                ticker_trades.sort_values("signal_date"),
                ticker_weekly.sort_values("week_date"),
                left_on="signal_date",
                right_on="week_date",
                by="ticker",
                direction="backward",
                allow_exact_matches=False,
            )
        )
    return pd.concat(parts, ignore_index=True).sort_values(["entry_time", "ticker", "signal_time"]).reset_index(drop=True)


def daily_volatility_compression(close: pd.DataFrame, *, short_window: int = 20, long_window: int = 60) -> pd.DataFrame:
    returns = close.pct_change(fill_method=None)
    prior_returns = returns.shift(1)
    short_vol = prior_returns.rolling(short_window, min_periods=short_window).std()
    long_vol = prior_returns.rolling(long_window, min_periods=long_window).std()
    compressed = short_vol.le(long_vol)
    next_index = close.index.max() + pd.offsets.BDay(1) if len(close.index) else pd.Timestamp("1900-01-01")
    if len(close.index):
        next_row = returns.rolling(short_window, min_periods=short_window).std().iloc[[-1]].set_axis([next_index])
        next_long = returns.rolling(long_window, min_periods=long_window).std().iloc[[-1]].set_axis([next_index])
        compressed = pd.concat([compressed, next_row.le(next_long)])
    return compressed.fillna(False)


def apply_mtf_filter(trades: pd.DataFrame, required_columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    if not required_columns:
        return trades.copy()
    mask = pd.Series(True, index=trades.index)
    for column in required_columns:
        if column not in trades.columns:
            mask &= False
        else:
            mask &= trades[column].where(trades[column].notna(), False).astype(bool)
    return trades.loc[mask].copy().reset_index(drop=True)


def run_experiment(current_dir: Path = DEFAULT_CURRENT_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = load_trades(current_dir / "base" / "intraday_trades.csv")
    config = config_from_json(current_dir / "base" / "config.json", start="2019-01-01")
    tickers = tuple(sorted(trades["ticker"].astype(str).unique()))
    close = load_close_prices(ROOT.parquet_path / "qw_adj_c.parquet", trades)
    close = _apply_daily_membership(close.reindex(columns=tickers), ROOT.parquet_path)
    dataset = KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    feature_start = pd.Timestamp(config.start) - pd.Timedelta(days=config.high_lookback_days)
    feature_close, _feature_high, _feature_low = load_daily_5m_matrices(dataset, tickers, start=feature_start, end=config.end)
    features = build_multi_timeframe_features(feature_close, tickers, config)
    enriched = attach_completed_weekly_features(trades, features["weekly"])
    enriched = attach_daily_features(enriched, features["daily"])
    enriched.to_csv(output_dir / "current_trades_with_mtf_features.csv", index=False)

    ledgers: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for spec in default_filters():
        variant_trades = apply_mtf_filter(enriched, spec.required_columns)
        audit, selected, skipped, fixed, _rebalanced = fixed_slot_selection_audit(variant_trades, close, max_positions=20)
        selected.to_csv(output_dir / f"{spec.name}_selected_trades.csv", index=False)
        skipped.to_csv(output_dir / f"{spec.name}_skipped_trades.csv", index=False)
        fixed.to_csv(output_dir / f"{spec.name}_fixed_notional_ledger.csv", index_label="date")
        ledgers[spec.name] = fixed
        rows.append(
            {
                "strategy": spec.name,
                "input_trades": int(len(variant_trades)),
                "selected_trades": int(len(selected)),
                "skipped_trades": int(len(skipped)),
                "compression_vs_current": float(1.0 - len(variant_trades) / len(enriched)) if len(enriched) else 0.0,
                "fixed_return": float(audit.fixed_notional_final_return),
                "mdd": float(audit.fixed_notional_mdd),
                "avg_trade_return": float(audit.selected_avg_trade_return),
                "hit_rate": float(audit.selected_hit_rate),
                "profit_factor": float(audit.selected_profit_factor),
                "max_active_positions": int(audit.max_active_positions),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("fixed_return", ascending=False).reset_index(drop=True)
    metrics.to_csv(output_dir / "multi_timeframe_filter_metrics.csv", index=False)
    (output_dir / "multi_timeframe_filter_config.json").write_text(
        json.dumps([asdict(item) for item in default_filters()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_dashboard(ledgers, metrics, output_dir / "multi_timeframe_filter_comparison.png")
    write_report(metrics, output_dir / "multi_timeframe_filter_report.md")
    return metrics


def build_multi_timeframe_features(close: pd.DataFrame, tickers: tuple[str, ...], config: Any) -> dict[str, pd.DataFrame]:
    data = load_research_feature_data(ROOT.parquet_path, tickers)
    weekly = weekly_relative_strength_features(close, data.sector, tickers)
    daily = daily_filter_features(close)
    return {"weekly": weekly, "daily": daily}


def weekly_relative_strength_features(close: pd.DataFrame, sector: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    weekly_close = close.resample("W-FRI").last().dropna(how="all")
    weekly_return = weekly_close / weekly_close.shift(12) - 1.0
    market_return = weekly_return.mean(axis=1, skipna=True)
    dates = pd.DatetimeIndex(close.index)
    sector_daily = _aligned(sector, dates, tickers).ffill()
    weekly_sector = sector_daily.resample("W-FRI").last().reindex(weekly_return.index).ffill()
    stacked_return = weekly_return.stack(future_stack=True).rename("weekly_12w_return").reset_index()
    stacked_return.columns = ["week_date", "ticker", "weekly_12w_return"]
    stacked_sector = weekly_sector.stack(future_stack=True).rename("sector_name").reset_index()
    stacked_sector.columns = ["week_date", "ticker", "sector_name"]
    frame = stacked_return.merge(stacked_sector, on=["week_date", "ticker"], how="left", sort=False)
    market_frame = market_return.rename("weekly_market_12w_return").reset_index()
    market_frame.columns = ["week_date", "weekly_market_12w_return"]
    frame = frame.merge(market_frame, on="week_date", how="left", sort=False)
    sector_benchmark = (
        frame.groupby(["week_date", "sector_name"], dropna=False)["weekly_12w_return"]
        .mean()
        .rename("weekly_sector_12w_return")
        .reset_index()
    )
    frame = frame.merge(sector_benchmark, on=["week_date", "sector_name"], how="left", sort=False)
    frame["weekly_market_rs"] = frame["weekly_12w_return"] - frame["weekly_market_12w_return"]
    frame["weekly_sector_rs"] = frame["weekly_12w_return"] - frame["weekly_sector_12w_return"]
    frame["weekly_market_rs_ok"] = frame["weekly_market_rs"].gt(0.0)
    frame["weekly_sector_rs_ok"] = frame["weekly_sector_rs"].gt(0.0)
    return frame.dropna(subset=["weekly_12w_return"]).sort_values(["ticker", "week_date"]).reset_index(drop=True)


def daily_filter_features(close: pd.DataFrame) -> pd.DataFrame:
    compressed = daily_volatility_compression(close)
    frame = compressed.stack(future_stack=True).rename("daily_vol_compression_ok").reset_index()
    frame.columns = ["signal_date", "ticker", "daily_vol_compression_ok"]
    return frame.sort_values(["ticker", "signal_date"]).reset_index(drop=True)


def attach_daily_features(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    working = trades.copy()
    working["signal_date"] = pd.to_datetime(working["signal_time"]).dt.normalize()
    return working.merge(daily, on=["signal_date", "ticker"], how="left", sort=False)


def write_dashboard(ledgers: dict[str, pd.DataFrame], metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=160, facecolor="#fbfaf7")
    for name, ledger in ledgers.items():
        if ledger.empty:
            continue
        axes[0, 0].plot(ledger.index, (ledger["equity"] - 1.0) * 100.0, label=_label(name, metrics), linewidth=1.6)
        axes[1, 0].plot(ledger.index, ledger["drawdown"] * 100.0, label=name, linewidth=1.1)
        axes[1, 1].plot(ledger.index, ledger["active_positions"], linewidth=0.9, alpha=0.65)
    ordered = metrics.sort_values("fixed_return", ascending=True)
    axes[0, 1].barh(ordered["strategy"], ordered["fixed_return"] * 100.0, color="#2f7ebc")
    axes[0, 0].set_title("Fixed-notional cumulative return", loc="left", fontweight="bold")
    axes[0, 1].set_title("Final return by MTF filter", loc="left", fontweight="bold")
    axes[1, 0].set_title("Drawdown", loc="left", fontweight="bold")
    axes[1, 1].set_title("Active positions", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)
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
    display["compression_pct"] = display["compression_vs_current"] * 100.0
    display["avg_trade_bps"] = display["avg_trade_return"] * 10_000.0
    lines = [
        "# Multi Timeframe Filter Comparison",
        "",
        "All variants keep the current confirmed-breakout entry/exit mechanics and use 35bp round-trip costs with fixed 20-slot notional accounting.",
        "",
        "| strategy | input | selected | compression_pct | fixed_return_pct | mdd_pct | avg_trade_bps | hit_rate | profit_factor | max_active |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.input_trades} | {row.selected_trades} | {row.compression_pct:.2f} | {row.fixed_return_pct:.4f} | {row.mdd_pct:.4f} | {row.avg_trade_bps:.4f} | {row.hit_rate:.4f} | {row.profit_factor:.4f} | {row.max_active_positions} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _label(name: str, metrics: pd.DataFrame) -> str:
    row = metrics.loc[metrics["strategy"].eq(name)]
    if row.empty:
        return name
    item = row.iloc[0]
    return f"{name}: {float(item['fixed_return']) * 100.0:.1f}%, MDD {float(item['mdd']) * 100.0:.1f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multi-timeframe entry compression filters.")
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.current_dir, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
