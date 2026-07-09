from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtesting.data.kr_stock_5m import KrStock5mDataset, read_tickers_bars
from root import ROOT
from scripts.run_flow_filtered_breakout_single import (
    _apply_daily_membership,
    _combine_trades,
    _daily_exit_frame,
    _daily_research_features_from_close,
    _entry_candidates,
    _load_start,
    _ticker_batches,
    compress_breakout_episodes,
    config_from_json,
    load_daily_5m_matrices,
    remove_overlapping_trades,
)
from scripts.run_tech_gamma_long_only import TechGammaConfig, build_features
from scripts.tech_gamma_costs import net_return_after_costs
from scripts.tech_gamma_research_filters import load_research_feature_data
from scripts.verified_flow_backtest import fixed_slot_selection_audit, load_close_prices


DEFAULT_RESEARCH_DIR = (
    ROOT.results_path
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
)
DEFAULT_CURRENT_DIR = DEFAULT_RESEARCH_DIR / "variants" / "current"
DEFAULT_OUTPUT_DIR = DEFAULT_RESEARCH_DIR / "exit_regime_comparison"

AtrMode = Literal["touch", "close", "off", "entry_low_close", "positivity_relaxed"]
PositivityExitMode = Literal["none", "absolute_open", "relative_open", "weak2_open"]


@dataclass(frozen=True, slots=True)
class ExitRegime:
    name: str
    atr_mode: AtrMode
    positivity_exit_mode: PositivityExitMode = "none"
    positivity_fraction: float = 0.5
    positivity_threshold: float = 0.02
    positivity_days: int = 2

    @staticmethod
    def current() -> "ExitRegime":
        return ExitRegime(name="current_atr_touch", atr_mode="touch")

    @staticmethod
    def atr_close_confirmed() -> "ExitRegime":
        return ExitRegime(name="atr_close_confirmed", atr_mode="close")

    @staticmethod
    def no_atr_breakout_only() -> "ExitRegime":
        return ExitRegime(name="no_atr_breakout_only", atr_mode="off")

    @staticmethod
    def entry_day_low_close_confirmed() -> "ExitRegime":
        return ExitRegime(name="entry_day_low_close_confirmed", atr_mode="entry_low_close")

    @staticmethod
    def positivity_relaxed_atr() -> "ExitRegime":
        return ExitRegime(name="positivity_relaxed_atr", atr_mode="positivity_relaxed")

    @staticmethod
    def positivity_absolute_open_exit() -> "ExitRegime":
        return ExitRegime(name="positivity_absolute_open_exit", atr_mode="close", positivity_exit_mode="absolute_open")

    @staticmethod
    def positivity_relative_open_exit() -> "ExitRegime":
        return ExitRegime(name="positivity_relative50_open_exit", atr_mode="close", positivity_exit_mode="relative_open")

    @staticmethod
    def positivity_weak2_open_exit() -> "ExitRegime":
        return ExitRegime(name="positivity_weak2_open_exit", atr_mode="close", positivity_exit_mode="weak2_open")


def default_regimes() -> list[ExitRegime]:
    return [
        ExitRegime.current(),
        ExitRegime.atr_close_confirmed(),
        ExitRegime.no_atr_breakout_only(),
        ExitRegime.entry_day_low_close_confirmed(),
        ExitRegime.positivity_relaxed_atr(),
        ExitRegime.positivity_absolute_open_exit(),
        ExitRegime.positivity_relative_open_exit(),
        ExitRegime.positivity_weak2_open_exit(),
    ]


def simulate_exit_regime_trade(
    signal: pd.Series,
    daily: pd.DataFrame,
    regime: ExitRegime,
    config: TechGammaConfig,
) -> dict[str, object] | None:
    entry_date = pd.Timestamp(signal["date"]).normalize()
    entry_price = float(signal["next_open"])
    stop_price = entry_price - float(signal["atr"]) * config.atr_stop_multiplier
    ticker_daily = daily.assign(date=pd.to_datetime(daily["date"]).dt.normalize()).sort_values("date").reset_index(drop=True)
    if ticker_daily.empty:
        return None
    holding_days = (ticker_daily["date"] - entry_date).dt.days
    entry_rows = ticker_daily.loc[ticker_daily["date"].eq(entry_date)]
    entry_low = float(entry_rows.iloc[0]["daily_low"]) if not entry_rows.empty and pd.notna(entry_rows.iloc[0]["daily_low"]) else float("nan")
    entry_spread = (
        float(entry_rows.iloc[0]["positivity_spread"])
        if not entry_rows.empty and pd.notna(entry_rows.iloc[0].get("positivity_spread"))
        else float("nan")
    )
    eligible = ticker_daily.loc[holding_days.ge(config.min_holding_days)].reset_index(drop=True)
    for index, row in eligible.iterrows():
        if _positivity_open_exit(row, eligible.iloc[: index + 1], entry_spread, regime):
            return _trade(signal, row, entry_price, float(row["open"]), _positivity_exit_reason(regime), at_open=True)
        exit_reason, exit_price = _close_exit(row, regime, stop_price, entry_low, entry_spread)
        if exit_reason is not None:
            return _trade(signal, row, entry_price, exit_price, exit_reason, at_open=False)
    if eligible.empty:
        return None
    exit_row = eligible.iloc[-1]
    return _trade(signal, exit_row, entry_price, float(exit_row["close"]), "end_of_data", at_open=False)


def simulate_exit_regime_continuation(entries: pd.DataFrame, daily: pd.DataFrame, regime: ExitRegime, config: TechGammaConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    daily_groups = {
        str(ticker): group.sort_values("date").reset_index(drop=True)
        for ticker, group in daily.groupby("ticker", sort=True)
    }
    for ticker, ticker_entries in entries.groupby("ticker", sort=True):
        available_date = pd.Timestamp.min
        ticker_daily = daily_groups.get(str(ticker))
        if ticker_daily is None:
            continue
        for _, signal in ticker_entries.sort_values("ts").iterrows():
            signal_date = pd.Timestamp(signal["date"]).normalize()
            if signal_date <= available_date:
                continue
            trade = simulate_exit_regime_trade(signal, ticker_daily, regime, config)
            if trade is None:
                continue
            rows.append(trade)
            available_date = pd.Timestamp(trade["exit_time"]).normalize()
    return pd.DataFrame(rows)


def run_experiment(
    current_dir: Path = DEFAULT_CURRENT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    batch_size: int = 20,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config_from_json(current_dir / "base" / "config.json", start="2019-01-01")
    dataset = KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    candidates = pd.read_csv(current_dir / "base" / "prefilter_candidates.csv", parse_dates=["date"])
    tickers = tuple(sorted(candidates["ticker"].astype(str).unique()))
    daily = build_exit_regime_daily_frame(dataset, tickers, config)
    entries = build_regime_entry_candidates(dataset, candidates, config, batch_size=batch_size)
    entries.to_csv(output_dir / "entry_candidates.csv", index=False)

    all_variant_trades: dict[str, pd.DataFrame] = {}
    for regime in default_regimes():
        trades = simulate_exit_regime_continuation(entries, daily, regime, config)
        trades = remove_overlapping_trades(_combine_trades([trades]))
        if config.episode_compression:
            trades = compress_breakout_episodes(trades, daily)
        trades.to_csv(output_dir / f"{regime.name}_all_trades.csv", index=False)
        all_variant_trades[regime.name] = trades

    close_source = _combine_trades([frame for frame in all_variant_trades.values() if not frame.empty])
    close = load_close_prices(ROOT.parquet_path / "qw_adj_c.parquet", close_source)
    ledgers: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for regime in default_regimes():
        trades = all_variant_trades[regime.name]
        audit, selected, skipped, fixed, _rebalanced = fixed_slot_selection_audit(trades, close, max_positions=20)
        selected.to_csv(output_dir / f"{regime.name}_selected_trades.csv", index=False)
        fixed.to_csv(output_dir / f"{regime.name}_fixed_notional_ledger.csv", index_label="date")
        ledgers[regime.name] = fixed
        reasons = selected["exit_reason"].astype(str) if not selected.empty else pd.Series(dtype=str)
        rows.append(
            {
                "strategy": regime.name,
                "input_trades": int(len(trades)),
                "selected_trades": int(len(selected)),
                "skipped_trades": int(len(skipped)),
                "fixed_return": float(audit.fixed_notional_final_return),
                "mdd": float(audit.fixed_notional_mdd),
                "avg_trade_return": float(audit.selected_avg_trade_return),
                "hit_rate": float(audit.selected_hit_rate),
                "profit_factor": float(audit.selected_profit_factor),
                "max_active_positions": int(audit.max_active_positions),
                "atr_exits": int(reasons.str.contains("atr").sum()),
                "positivity_open_exits": int(
                    reasons.isin(["positivity_absolute_open", "positivity_relative_decay_open", "positivity_weak2_open"]).sum()
                ),
                "positivity_regime_atr_exits": int((reasons.str.startswith("positivity_") & reasons.str.contains("atr")).sum()),
                "new_high_lost_exits": int(reasons.eq("new_high_lost").sum()),
                "end_of_data_exits": int(reasons.eq("end_of_data").sum()),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("fixed_return", ascending=False).reset_index(drop=True)
    metrics.to_csv(output_dir / "exit_regime_comparison_metrics.csv", index=False)
    (output_dir / "exit_regime_comparison_config.json").write_text(
        json.dumps([asdict(regime) for regime in default_regimes()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_dashboard(ledgers, metrics, output_dir / "exit_regime_comparison.png")
    write_report(metrics, output_dir / "exit_regime_comparison_report.md")
    return metrics


def build_exit_regime_daily_frame(dataset: KrStock5mDataset, tickers: tuple[str, ...], config: TechGammaConfig) -> pd.DataFrame:
    start = pd.Timestamp(config.start) - pd.Timedelta(days=config.high_lookback_days)
    close, _high, low = load_daily_5m_matrices(dataset, tickers, start=start, end=config.end)
    daily = _daily_exit_frame(close=close, low=low)
    open_ = load_daily_5m_open_matrix(dataset, tickers, start=start, end=config.end)
    open_long = open_.stack(future_stack=True).rename("open").reset_index()
    open_long.columns = ["date", "ticker", "open"]
    data = load_research_feature_data(dataset.root.parent, tickers)
    features = _daily_research_features_from_close(close=close, config=config, data=data, tickers=tickers)
    return (
        daily.merge(open_long, on=["date", "ticker"], how="left", sort=False)
        .merge(features[["date", "ticker", "positivity_spread"]], on=["date", "ticker"], how="left", sort=False)
        .dropna(subset=["open", "close", "daily_low", "prior_52w_close_high"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


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


def build_regime_entry_candidates(
    dataset: KrStock5mDataset,
    candidates: pd.DataFrame,
    config: TechGammaConfig,
    *,
    batch_size: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month, month_candidates in candidates.groupby(candidates["date"].dt.to_period("M"), sort=True):
        month_start = pd.Period(month, freq="M").to_timestamp()
        read_start = max(_load_start(config), month_start - pd.Timedelta(days=10))
        read_end = pd.Period(month, freq="M").end_time
        tickers = tuple(sorted(month_candidates["ticker"].astype(str).drop_duplicates()))
        for batch in _ticker_batches(tickers, batch_size):
            raw = read_tickers_bars(dataset, batch, start=read_start, end=read_end)
            if raw.empty:
                continue
            usable = raw.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
            frame = build_features(usable, config)
            frame = frame.loc[frame["date"].isin(month_candidates["date"].unique())]
            if frame.empty:
                continue
            candidate_columns = [
                column
                for column in (
                    "date",
                    "ticker",
                    "daily_close",
                    "prior_52w_close_high",
                    "daily_positivity",
                    "positivity_benchmark",
                    "positivity_spread",
                    "positivity_filter_ok",
                    "factor_filter_ok",
                    "foreign_flow_to_cap",
                    "institution_flow_to_cap",
                    "sector_name",
                )
                if column in month_candidates.columns
            ]
            frame = frame.drop(columns=[column for column in candidate_columns if column in frame.columns and column not in ("date", "ticker")])
            frame = frame.merge(month_candidates[candidate_columns], on=["date", "ticker"], how="inner", sort=False)
            frame["breakout_52w_bps"] = (frame["close"] / frame["prior_52w_close_high"] - 1.0) * 10_000.0
            frame["high_52w_breakout_score"] = frame["breakout_52w_bps"].clip(lower=0.0).divide(10.0) + frame["volume_spike"].clip(upper=5.0).sub(1.0).clip(lower=0.0)
            frame["signal_score"] = frame["high_52w_breakout_score"]
            entries = _entry_candidates(frame, config)
            if not entries.empty:
                frames.append(entries)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "ts", "next_ts", "next_open", "atr", "signal_score"])
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date", "ts"]).reset_index(drop=True)


def write_dashboard(ledgers: dict[str, pd.DataFrame], metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=160, facecolor="#fbfaf7")
    for name, ledger in ledgers.items():
        if ledger.empty:
            continue
        label = _label(name, metrics)
        axes[0, 0].plot(ledger.index, (ledger["equity"] - 1.0) * 100.0, linewidth=1.5, label=label)
        axes[1, 0].plot(ledger.index, ledger["drawdown"] * 100.0, linewidth=1.1, label=name)
        axes[1, 1].plot(ledger.index, ledger["active_positions"], linewidth=0.9, alpha=0.6)
    ordered = metrics.sort_values("fixed_return", ascending=True)
    axes[0, 1].barh(ordered["strategy"], ordered["fixed_return"] * 100.0, color="#2f7ebc")
    axes[0, 0].set_title("Fixed-notional cumulative return", loc="left", fontweight="bold")
    axes[0, 1].set_title("Final return by exit regime", loc="left", fontweight="bold")
    axes[1, 0].set_title("Drawdown", loc="left", fontweight="bold")
    axes[1, 1].set_title("Active positions", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=1)
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
        "# Exit Regime Comparison",
        "",
        "All variants use the same confirmed-breakout entry generation, 35bp round-trip costs, and fixed 20-slot notional accounting.",
        "",
        "| strategy | selected | fixed_return_pct | mdd_pct | avg_trade_bps | hit_rate | profit_factor | atr_exits | positivity_open | positivity_regime_atr | new_high_lost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.selected_trades} | {row.fixed_return_pct:.4f} | {row.mdd_pct:.4f} | {row.avg_trade_bps:.4f} | {row.hit_rate:.4f} | {row.profit_factor:.4f} | {row.atr_exits} | {row.positivity_open_exits} | {row.positivity_regime_atr_exits} | {row.new_high_lost_exits} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _positivity_open_exit(row: pd.Series, decision_rows: pd.DataFrame, entry_spread: float, regime: ExitRegime) -> bool:
    if regime.positivity_exit_mode == "none" or pd.isna(entry_spread) or pd.isna(row.get("positivity_spread")):
        return False
    latest = float(row["positivity_spread"])
    if regime.positivity_exit_mode == "absolute_open":
        return latest <= 0.0
    if regime.positivity_exit_mode == "relative_open":
        return latest <= entry_spread * regime.positivity_fraction
    if regime.positivity_exit_mode == "weak2_open":
        if len(decision_rows) < regime.positivity_days:
            return False
        recent = decision_rows.tail(regime.positivity_days)["positivity_spread"].astype(float)
        return bool(recent.le(regime.positivity_threshold).all())
    raise ValueError(f"unknown positivity exit mode: {regime.positivity_exit_mode}")


def _positivity_exit_reason(regime: ExitRegime) -> str:
    if regime.positivity_exit_mode == "absolute_open":
        return "positivity_absolute_open"
    if regime.positivity_exit_mode == "relative_open":
        return "positivity_relative_decay_open"
    if regime.positivity_exit_mode == "weak2_open":
        return "positivity_weak2_open"
    raise ValueError(f"unknown positivity exit mode: {regime.positivity_exit_mode}")


def _close_exit(row: pd.Series, regime: ExitRegime, stop_price: float, entry_low: float, entry_spread: float) -> tuple[str | None, float]:
    close = float(row["close"])
    low = float(row["daily_low"])
    prior_high = float(row["prior_52w_close_high"])
    if regime.atr_mode == "touch" and low <= stop_price:
        return "atr_stop", stop_price
    if regime.atr_mode == "close" and close <= stop_price:
        return "atr_close_stop", close
    if regime.atr_mode == "entry_low_close" and pd.notna(entry_low) and close <= entry_low:
        return "entry_day_low_close_stop", close
    if regime.atr_mode == "positivity_relaxed":
        supportive = pd.notna(entry_spread) and pd.notna(row.get("positivity_spread")) and float(row["positivity_spread"]) > entry_spread * regime.positivity_fraction
        if supportive and close <= stop_price:
            return "positivity_supportive_atr_close_stop", close
        if not supportive and low <= stop_price:
            return "positivity_weak_atr_stop", stop_price
    if close <= prior_high:
        return "new_high_lost", close
    return None, close


def _trade(
    signal: pd.Series,
    row: pd.Series,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    *,
    at_open: bool,
) -> dict[str, object]:
    gross = exit_price / entry_price - 1.0
    exit_time = pd.Timestamp(row["date"]) + (pd.Timedelta(hours=9) if at_open else pd.Timedelta(hours=15, minutes=30))
    return {
        "ticker": str(signal["ticker"]),
        "side": "long",
        "signal_time": pd.Timestamp(signal["ts"]),
        "entry_time": pd.Timestamp(signal["next_ts"]),
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "signal_score": float(signal["signal_score"]),
        "gross_return": gross,
        "net_return": net_return_after_costs(gross),
        "exit_reason": exit_reason,
    }


def _label(name: str, metrics: pd.DataFrame) -> str:
    row = metrics.loc[metrics["strategy"].eq(name)]
    if row.empty:
        return name
    item = row.iloc[0]
    return f"{name}: {float(item['fixed_return']) * 100.0:.1f}%, MDD {float(item['mdd']) * 100.0:.1f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ATR and positivity-driven exit regimes.")
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.current_dir, args.output_dir, batch_size=args.batch_size)
    print(args.output_dir)


if __name__ == "__main__":
    main()
