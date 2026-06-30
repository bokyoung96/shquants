from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtesting.data.kr_stock_5m import KrStock5mDataset, read_ticker_bars, read_tickers_bars
from root import ROOT
from scripts.tech_gamma_costs import ROUND_TRIP_COST_BPS, net_return_after_costs
from scripts.verified_flow_backtest import same_ticker_overlap_violations
from scripts.run_flow_filtered_breakout_single import _daily_exit_frame, config_from_json, load_daily_5m_matrices


DEFAULT_RESULT_DIR = ROOT.results_path / "flow_filtered_breakout_single" / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"


def audit_trade_log_integrity(trades: pd.DataFrame) -> dict[str, Any]:
    signal_time = pd.to_datetime(trades["signal_time"])
    entry_time = pd.to_datetime(trades["entry_time"])
    exit_time = pd.to_datetime(trades["exit_time"])
    required_columns = ["ticker", "signal_time", "entry_time", "exit_time", "entry_price", "exit_price", "gross_return", "net_return"]
    missing_columns = [column for column in required_columns if column not in trades.columns]
    present_required = [column for column in required_columns if column in trades.columns]
    return {
        "trades": int(len(trades)),
        "tickers": int(trades["ticker"].nunique()),
        "entry_before_or_at_signal_violations": int(entry_time.le(signal_time).sum()),
        "exit_before_entry_violations": int(exit_time.lt(entry_time).sum()),
        "same_ticker_overlap_violations": int(same_ticker_overlap_violations(trades)),
        "duplicate_trade_rows": int(trades.duplicated(subset=["ticker", "signal_time", "entry_time", "exit_time"]).sum()),
        "missing_required_columns": missing_columns,
        "missing_required_values": int(trades[present_required].isna().sum().sum()),
        "nonpositive_entry_prices": int(pd.to_numeric(trades["entry_price"], errors="coerce").le(0.0).sum()),
        "nonpositive_exit_prices": int(pd.to_numeric(trades["exit_price"], errors="coerce").le(0.0).sum()),
    }


def audit_return_accounting(trades: pd.DataFrame, ledger: pd.DataFrame, *, slots: int) -> dict[str, Any]:
    gross = trades["exit_price"].astype(float).divide(trades["entry_price"].astype(float)).sub(1.0)
    net = gross.map(net_return_after_costs)
    gross_delta = gross.sub(trades["gross_return"].astype(float)).abs()
    net_delta = net.sub(trades["net_return"].astype(float)).abs()
    final_from_trades = round(float(trades["net_return"].astype(float).sum() / slots), 12)
    final_from_ledger = round(float(ledger["equity"].iloc[-1] - 1.0), 12)
    return {
        "slots": int(slots),
        "round_trip_cost_bps": float(ROUND_TRIP_COST_BPS),
        "gross_return_mismatches": int(gross_delta.gt(1e-10).sum()),
        "max_gross_return_abs_delta": float(gross_delta.max()) if not gross_delta.empty else 0.0,
        "net_return_mismatches": int(net_delta.gt(1e-10).sum()),
        "max_net_return_abs_delta": float(net_delta.max()) if not net_delta.empty else 0.0,
        "fixed_notional_final_return_from_trades": final_from_trades,
        "fixed_notional_final_return_from_ledger": final_from_ledger,
        "fixed_notional_final_return_delta": float(abs(final_from_trades - final_from_ledger)),
    }


def audit_entry_exit_against_source(
    trades: pd.DataFrame,
    candidates: pd.DataFrame,
    daily_exit: pd.DataFrame,
    dataset: KrStock5mDataset,
) -> dict[str, Any]:
    candidate_key = candidates.assign(date=pd.to_datetime(candidates["date"]).dt.normalize()).set_index(["ticker", "date"])
    daily_key = daily_exit.assign(date=pd.to_datetime(daily_exit["date"]).dt.normalize()).set_index(["ticker", "date"])
    entry_price_mismatches = 0
    signal_confirmation_violations = 0
    signal_confirmation_examples: list[dict[str, Any]] = []
    missing_entry_bars = 0
    exit_condition_violations = 0
    missing_signal_candidate_rows = 0
    missing_daily_exit_rows = 0
    min_holding_violations = 0

    def audit_trade(trade: Any, bars: pd.DataFrame | None) -> None:
        nonlocal entry_price_mismatches
        nonlocal signal_confirmation_violations
        nonlocal missing_entry_bars
        nonlocal exit_condition_violations
        nonlocal missing_signal_candidate_rows
        nonlocal missing_daily_exit_rows
        nonlocal min_holding_violations

        signal_time = pd.Timestamp(trade.signal_time)
        entry_time = pd.Timestamp(trade.entry_time)
        exit_time = pd.Timestamp(trade.exit_time)
        date = signal_time.normalize()
        ticker = str(trade.ticker)
        if (ticker, date) not in candidate_key.index:
            missing_signal_candidate_rows += 1
            return
        candidate = candidate_key.loc[(ticker, date)]
        prior_high = float(candidate["prior_52w_close_high"])
        if bars is None or signal_time not in bars.index or entry_time not in bars.index:
            missing_entry_bars += 1
            return
        signal_bar = bars.loc[signal_time]
        signal_position = bars.index.get_loc(signal_time)
        confirmation_time = bars.index[signal_position + 1] if signal_position + 1 < len(bars.index) else pd.NaT
        confirmation_bar = bars.loc[confirmation_time] if pd.notna(confirmation_time) else None
        previous_close = bars["close"].shift(1).loc[signal_time]
        entry_open = float(bars.loc[entry_time, "open"])
        if abs(entry_open - float(trade.entry_price)) > 1e-8:
            entry_price_mismatches += 1
        if (
            not float(signal_bar["close"]) > prior_high
            or not (pd.isna(previous_close) or float(previous_close) <= prior_high)
            or confirmation_bar is None
            or not float(confirmation_bar["close"]) > prior_high
            or not entry_time > confirmation_time
        ):
            signal_confirmation_violations += 1
            if len(signal_confirmation_examples) < 5:
                signal_confirmation_examples.append(
                    {
                        "ticker": ticker,
                        "signal_time": str(signal_time),
                        "entry_time": str(entry_time),
                        "prior_52w_close_high": prior_high,
                        "previous_close": None if pd.isna(previous_close) else float(previous_close),
                        "signal_close": float(signal_bar["close"]),
                        "confirmation_time": None if pd.isna(confirmation_time) else str(confirmation_time),
                        "confirmation_close": None if confirmation_bar is None else float(confirmation_bar["close"]),
                    }
                )

        if (exit_time.normalize() - entry_time.normalize()).days < 1:
            min_holding_violations += 1
        exit_daily_key = (ticker, exit_time.normalize())
        if exit_daily_key not in daily_key.index:
            missing_daily_exit_rows += 1
            return
        exit_candidate = daily_key.loc[exit_daily_key]
        if str(trade.exit_reason) == "new_high_lost":
            if not float(exit_candidate["close"]) <= float(exit_candidate["prior_52w_close_high"]):
                exit_condition_violations += 1
            if abs(float(exit_candidate["close"]) - float(trade.exit_price)) > 1e-8:
                exit_condition_violations += 1
        elif str(trade.exit_reason) == "atr_stop":
            if not float(exit_candidate["daily_low"]) <= float(trade.exit_price):
                exit_condition_violations += 1

    working_trades = trades.copy()
    working_trades["signal_time"] = pd.to_datetime(working_trades["signal_time"])
    working_trades["entry_time"] = pd.to_datetime(working_trades["entry_time"])
    working_trades["exit_time"] = pd.to_datetime(working_trades["exit_time"])
    working_trades["signal_month"] = working_trades["signal_time"].dt.to_period("M")

    for month, month_trades in working_trades.groupby("signal_month", sort=True):
        tickers = tuple(sorted(month_trades["ticker"].astype(str).unique()))
        month_bars = read_tickers_bars(
            dataset,
            tickers,
            start=month.to_timestamp(),
            end=month.end_time,
        ).dropna(subset=["open", "high", "low", "close"])
        month_bars["ts"] = pd.to_datetime(month_bars["ts"])
        month_bars["date"] = month_bars["ts"].dt.normalize()
        day_bars = {
            (str(ticker), pd.Timestamp(date)): group.drop(columns=["date"]).set_index("ts").sort_index()
            for (ticker, date), group in month_bars.groupby(["ticker", "date"], sort=False)
        }

        for trade in month_trades.itertuples(index=False):
            signal_date = pd.Timestamp(trade.signal_time).normalize()
            audit_trade(trade, day_bars.get((str(trade.ticker), signal_date)))

    return {
        "entry_price_mismatches": int(entry_price_mismatches),
        "signal_confirmation_violations": int(signal_confirmation_violations),
        "signal_confirmation_examples": signal_confirmation_examples,
        "missing_entry_bars": int(missing_entry_bars),
        "exit_condition_violations": int(exit_condition_violations),
        "missing_signal_candidate_rows": int(missing_signal_candidate_rows),
        "missing_daily_exit_rows": int(missing_daily_exit_rows),
        "min_holding_violations": int(min_holding_violations),
    }


def audit_universe_membership(trades: pd.DataFrame, membership_path: Path) -> dict[str, Any]:
    membership = pd.read_parquet(membership_path, engine="pyarrow")
    membership.index = pd.to_datetime(membership.index).normalize()
    source_end = membership.index.max()
    trade_dates = pd.DatetimeIndex(pd.to_datetime(trades["signal_time"]).dt.normalize().unique())
    active = membership.reindex(index=membership.index.union(trade_dates)).ffill().reindex(trade_dates).fillna(0).gt(0)
    violations = 0
    missing_rows = 0
    for trade in trades.itertuples(index=False):
        date = pd.Timestamp(trade.signal_time).normalize()
        ticker = str(trade.ticker)
        if date not in active.index or ticker not in active.columns:
            missing_rows += 1
            continue
        if not bool(active.at[date, ticker]):
            violations += 1
    return {
        "kospi200_membership_violations": int(violations),
        "missing_membership_checks": int(missing_rows),
        "membership_source_end": str(source_end.date()),
        "signals_after_membership_source_end": int(pd.to_datetime(trades["signal_time"]).dt.normalize().gt(source_end).sum()),
    }


def select_case_trade(trades: pd.DataFrame) -> pd.Series:
    candidates = trades.loc[trades["exit_reason"].eq("new_high_lost")].copy()
    if candidates.empty:
        candidates = trades.copy()
    candidates["abs_distance_from_p95"] = (candidates["net_return"] - candidates["net_return"].quantile(0.95)).abs()
    return candidates.sort_values(["abs_distance_from_p95", "entry_time"]).iloc[0]


def load_case_frames(result_dir: Path, dataset: KrStock5mDataset, trade: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(result_dir / "base" / "prefilter_candidates.csv", parse_dates=["date"])
    ticker = str(trade["ticker"])
    entry_date = pd.Timestamp(trade["entry_time"]).normalize()
    exit_date = pd.Timestamp(trade["exit_time"]).normalize()
    start = entry_date - pd.Timedelta(days=6)
    end = exit_date + pd.Timedelta(days=2)
    daily = candidates.loc[candidates["ticker"].eq(ticker) & candidates["date"].between(start, end)].sort_values("date")
    entry_bars = read_ticker_bars(dataset, ticker, start=entry_date, end=entry_date + pd.Timedelta(hours=23, minutes=59))
    exit_bars = read_ticker_bars(dataset, ticker, start=exit_date, end=exit_date + pd.Timedelta(hours=23, minutes=59))
    return daily, entry_bars, exit_bars


def build_daily_exit_from_strategy_inputs(result_dir: Path, dataset: KrStock5mDataset, trades: pd.DataFrame) -> pd.DataFrame:
    config = config_from_json(result_dir / "base" / "config.json", start="2019-01-01")
    tickers = tuple(sorted(trades["ticker"].astype(str).unique()))
    close, _high, low = load_daily_5m_matrices(dataset, tickers, start=pd.Timestamp(config.start) - pd.Timedelta(days=config.high_lookback_days), end=config.end)
    return _daily_exit_frame(close=close, low=low)


def plot_entry_exit_case(
    trade: pd.Series,
    daily: pd.DataFrame,
    entry_bars: pd.DataFrame,
    exit_bars: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ticker = str(trade["ticker"])
    fig, axes = plt.subplots(3, 1, figsize=(17.5, 10.2), dpi=160, facecolor="white", gridspec_kw={"height_ratios": [1.05, 1.25, 1.25]})
    _plot_daily_case(axes[0], daily, trade)
    _plot_intraday_case(axes[1], entry_bars, trade, "Entry day 5-minute candles")
    _plot_intraday_case(axes[2], exit_bars, trade, "Exit day 5-minute candles")
    gross = float(trade["gross_return"]) * 100.0
    net = float(trade["net_return"]) * 100.0
    fig.suptitle(f"{ticker}: verified entry-to-exit case | gross {gross:.2f}% | net {net:.2f}% | {trade['exit_reason']}", x=0.01, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_validation(result_dir: Path, output_dir: Path, dataset: KrStock5mDataset) -> dict[str, Any]:
    trades = pd.read_csv(result_dir / "fixed20" / "selected_trades.csv", parse_dates=["signal_time", "entry_time", "exit_time"])
    ledger = pd.read_csv(result_dir / "fixed20" / "fixed_notional_ledger.csv", parse_dates=["date"]).set_index("date")
    candidates = pd.read_csv(result_dir / "base" / "prefilter_candidates.csv", parse_dates=["date"])
    daily_exit = build_daily_exit_from_strategy_inputs(result_dir, dataset, trades)
    audit = {
        "trade_log_integrity": audit_trade_log_integrity(trades),
        "return_accounting": audit_return_accounting(trades, ledger, slots=20),
        "source_entry_exit": audit_entry_exit_against_source(trades, candidates, daily_exit, dataset),
        "universe_membership": audit_universe_membership(trades, dataset.root.parent / "qw_k200_yn.parquet"),
        "known_limitations": [
            "ATR stop exits are simulated at the stop price when the daily low breaches the stop; the exact intraday breach time is not modelled.",
            "Prefilter candidates use same-day daily high to reduce 5-minute loading, but final entry validation is checked against intraday signal/confirmation bars.",
            "Historical KOSPI200 membership is applied, but this audit does not independently validate the upstream membership file against exchange announcements.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strategy_integrity_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_validation_report(audit, output_dir / "strategy_integrity_report.md")
    case_trade = select_case_trade(trades)
    daily, entry_bars, exit_bars = load_case_frames(result_dir, dataset, case_trade)
    plot_entry_exit_case(case_trade, daily, entry_bars, exit_bars, output_dir / f"{case_trade['ticker'].lower()}_entry_exit_case.png")
    return audit


def write_validation_report(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Confirmed Breakout Strategy Integrity Audit",
        "",
        "## Findings",
    ]
    blocking = []
    for section, values in audit.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"### {section}")
        for key, value in values.items():
            lines.append(f"- {key}: {value}")
            if key.endswith("violations") and value:
                blocking.append(f"{section}.{key}={value}")
            if key.endswith("mismatches") and value:
                blocking.append(f"{section}.{key}={value}")
        lines.append("")
    lines.extend(["## Limitations", *[f"- {item}" for item in audit.get("known_limitations", [])]])
    lines.extend(["", "## Conclusion", "No blocking accounting/integrity issue was detected." if not blocking else f"Blocking issues detected: {', '.join(blocking)}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_daily_case(ax: plt.Axes, daily: pd.DataFrame, trade: pd.Series) -> None:
    if daily.empty:
        ax.text(0.5, 0.5, "No daily candidate rows available", transform=ax.transAxes, ha="center", va="center")
        return
    close_column = "daily_close" if "daily_close" in daily.columns else "close"
    ax.fill_between(daily["date"], daily["daily_low"], daily[close_column], color="#6c8ebf", alpha=0.15, label="daily low-close range")
    ax.plot(daily["date"], daily[close_column], marker="o", color="#22577a", label="daily close")
    ax.plot(daily["date"], daily["prior_52w_close_high"], linestyle="--", color="#9d1730", label="prior 52w close high")
    ax.scatter(pd.Timestamp(trade["entry_time"]).normalize(), float(trade["entry_price"]), color="#2ca25f", s=70, zorder=5, label="entry")
    ax.scatter(pd.Timestamp(trade["exit_time"]).normalize(), float(trade["exit_price"]), color="#6a3d9a", s=70, zorder=5, label="exit")
    ax.set_title("Daily continuation path", loc="left", fontweight="bold")
    ax.set_ylabel("Price")
    ax.legend(frameon=False, ncols=5, loc="upper left")
    ax.grid(alpha=0.18)


def _plot_intraday_case(ax: plt.Axes, bars: pd.DataFrame, trade: pd.Series, title: str) -> None:
    if bars.empty:
        ax.text(0.5, 0.5, "No intraday bars available", transform=ax.transAxes, ha="center", va="center")
        return
    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars["ts"])
    width = 3.0 / (24 * 60)
    for row in bars.itertuples(index=False):
        color = "#2b83ba" if float(row.close) >= float(row.open) else "#d7191c"
        ax.vlines(row.ts, float(row.low), float(row.high), color=color, linewidth=1.0, alpha=0.85)
        lower = min(float(row.open), float(row.close))
        height = abs(float(row.close) - float(row.open))
        ax.add_patch(plt.Rectangle((mdates.date2num(row.ts) - width / 2, lower), width, height if height > 0 else 1e-9, color=color, alpha=0.72))
    signal_time = pd.Timestamp(trade["signal_time"])
    entry_time = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    if signal_time.normalize() == bars["ts"].iloc[0].normalize():
        ax.scatter(signal_time, _bar_price_at(bars, signal_time, "close"), color="#d7191c", s=55, label="signal close", zorder=6)
    if entry_time.normalize() == bars["ts"].iloc[0].normalize():
        ax.scatter(entry_time, float(trade["entry_price"]), color="#2ca25f", s=55, label="entry open", zorder=6)
    if exit_time.normalize() == bars["ts"].iloc[0].normalize():
        ax.scatter(exit_time, float(trade["exit_price"]), color="#6a3d9a", s=55, label="exit close/stop", zorder=6)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.18)


def _bar_price_at(bars: pd.DataFrame, ts: pd.Timestamp, column: str) -> float:
    matched = bars.loc[pd.to_datetime(bars["ts"]).eq(ts), column]
    return float(matched.iloc[0]) if not matched.empty else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate confirmed breakout strategy integrity and create an entry/exit case PNG.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.result_dir / "validation")
    audit = run_validation(args.result_dir, output_dir, KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m"))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
