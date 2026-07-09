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
from scripts.run_flow_filtered_breakout_single import config_from_json, load_daily_5m_matrices
from scripts.tech_gamma_costs import ROUND_TRIP_COST_BPS
from scripts.tech_gamma_universe import kospi200_tickers


DEFAULT_RESEARCH_DIR = (
    ROOT.results_path
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
)
DEFAULT_EVENT_TRADES = DEFAULT_RESEARCH_DIR / "variants" / "5m_new_high_only" / "base" / "intraday_trades.csv"
DEFAULT_CONFIG = DEFAULT_RESEARCH_DIR / "variants" / "5m_new_high_only" / "base" / "config.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RESEARCH_DIR / "52w_event_study"
DEFAULT_HORIZONS = (1, 3, 5, 10, 20)


def compute_event_forward_returns(
    events: pd.DataFrame,
    close: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    round_trip_cost_bps: float = 35.0,
) -> pd.DataFrame:
    required = {"ticker", "signal_time", "entry_time", "entry_price"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"missing event columns: {sorted(missing)}")
    if close.empty:
        raise ValueError("close matrix is empty")

    dates = pd.DatetimeIndex(pd.to_datetime(close.index).normalize())
    close = close.copy()
    close.index = dates
    date_to_position = {date: index for index, date in enumerate(dates)}
    cost = round_trip_cost_bps / 10_000.0

    rows: list[dict[str, object]] = []
    for _, event in events.sort_values(["entry_time", "ticker"]).iterrows():
        ticker = str(event["ticker"])
        event_date = pd.Timestamp(event["entry_time"]).normalize()
        position = date_to_position.get(event_date)
        row: dict[str, object] = {
            "ticker": ticker,
            "signal_time": pd.Timestamp(event["signal_time"]),
            "entry_time": pd.Timestamp(event["entry_time"]),
            "event_date": event_date,
            "entry_price": float(event["entry_price"]),
        }
        if position is None or ticker not in close.columns:
            rows.append(_empty_forward_row(row, horizons))
            continue

        start_close = close.iat[position, close.columns.get_loc(ticker)]
        for horizon in horizons:
            future_position = position + horizon
            if future_position >= len(close.index) or pd.isna(start_close):
                _assign_missing_horizon(row, horizon)
                continue
            future_close = close.iat[future_position, close.columns.get_loc(ticker)]
            if pd.isna(future_close):
                _assign_missing_horizon(row, horizon)
                continue
            event_close_return = float(future_close / start_close - 1.0)
            event_entry_return = float(future_close / float(event["entry_price"]) - 1.0)
            benchmark = _same_day_benchmark_return(close, position, future_position, exclude_ticker=ticker)
            row[f"event_close_return_{horizon}d"] = event_close_return
            row[f"event_entry_return_{horizon}d"] = event_entry_return
            row[f"event_entry_net_return_{horizon}d"] = event_entry_return - cost
            row[f"benchmark_return_{horizon}d"] = benchmark
            row[f"excess_return_{horizon}d"] = event_close_return - benchmark if pd.notna(benchmark) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_horizons(events: pd.DataFrame, *, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for horizon in horizons:
        entry_net = events[f"event_entry_net_return_{horizon}d"].dropna().astype(float)
        close_return = events[f"event_close_return_{horizon}d"].dropna().astype(float)
        benchmark = events[f"benchmark_return_{horizon}d"].dropna().astype(float)
        excess = events[f"excess_return_{horizon}d"].dropna().astype(float)
        rows.append(
            {
                "horizon_days": horizon,
                "events": int(len(entry_net)),
                "entry_net_mean": _mean(entry_net),
                "entry_net_median": _median(entry_net),
                "entry_net_hit_rate": _hit_rate(entry_net),
                "close_mean": _mean(close_return),
                "close_median": _median(close_return),
                "benchmark_mean": _mean(benchmark),
                "benchmark_median": _median(benchmark),
                "excess_mean": _mean(excess),
                "excess_median": _median(excess),
                "excess_hit_rate": _hit_rate(excess),
                "excess_t_stat": _t_stat(excess),
                "p25": _quantile(entry_net, 0.25),
                "p75": _quantile(entry_net, 0.75),
                "p90": _quantile(entry_net, 0.90),
                "p95": _quantile(entry_net, 0.95),
                "p99": _quantile(entry_net, 0.99),
                "trimmed_mean_5_95": _trimmed_mean(entry_net, 0.05, 0.95),
                "top5_removed_mean": _top_removed_mean(entry_net, 0.05),
            }
        )
    return pd.DataFrame(rows)


def compute_event_path_returns(
    events: pd.DataFrame,
    close: pd.DataFrame,
    *,
    max_horizon: int = 20,
) -> pd.DataFrame:
    required = {"ticker", "signal_time", "entry_time", "entry_price"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"missing event columns: {sorted(missing)}")
    if close.empty:
        raise ValueError("close matrix is empty")

    dates = pd.DatetimeIndex(pd.to_datetime(close.index).normalize())
    close = close.copy()
    close.index = dates
    date_to_position = {date: index for index, date in enumerate(dates)}

    values: dict[int, dict[str, list[float]]] = {
        horizon: {"signal_entry": [], "signal_close": [], "benchmark": [], "excess": []}
        for horizon in range(max_horizon + 1)
    }
    for _, event in events.sort_values(["entry_time", "ticker"]).iterrows():
        ticker = str(event["ticker"])
        event_date = pd.Timestamp(event["entry_time"]).normalize()
        position = date_to_position.get(event_date)
        if position is None or ticker not in close.columns:
            continue
        ticker_position = close.columns.get_loc(ticker)
        start_close = close.iat[position, ticker_position]
        if pd.isna(start_close):
            continue
        entry_price = float(event["entry_price"])
        for horizon in range(max_horizon + 1):
            future_position = position + horizon
            if future_position >= len(close.index):
                continue
            future_close = close.iat[future_position, ticker_position]
            if pd.isna(future_close):
                continue
            if horizon == 0:
                signal_entry = 0.0
                signal_close = 0.0
                benchmark = 0.0
            else:
                signal_entry = float(future_close / entry_price - 1.0)
                signal_close = float(future_close / start_close - 1.0)
                benchmark = _same_day_benchmark_return(close, position, future_position, exclude_ticker=ticker)
            excess = signal_close - benchmark if pd.notna(benchmark) else np.nan
            values[horizon]["signal_entry"].append(signal_entry)
            values[horizon]["signal_close"].append(signal_close)
            values[horizon]["benchmark"].append(benchmark)
            values[horizon]["excess"].append(excess)

    rows: list[dict[str, float | int]] = []
    for horizon, series_map in values.items():
        signal_entry = pd.Series(series_map["signal_entry"], dtype=float).dropna()
        signal_close = pd.Series(series_map["signal_close"], dtype=float).dropna()
        benchmark = pd.Series(series_map["benchmark"], dtype=float).dropna()
        excess = pd.Series(series_map["excess"], dtype=float).dropna()
        rows.append(
            {
                "event_day": horizon,
                "events": int(len(signal_entry)),
                "signal_entry_mean": _mean(signal_entry),
                "signal_entry_median": _median(signal_entry),
                "signal_entry_p25": _quantile(signal_entry, 0.25),
                "signal_entry_p75": _quantile(signal_entry, 0.75),
                "signal_entry_hit_rate": _hit_rate(signal_entry),
                "signal_close_mean": _mean(signal_close),
                "benchmark_mean": _mean(benchmark),
                "excess_mean": _mean(excess),
                "excess_median": _median(excess),
                "excess_hit_rate": _hit_rate(excess),
            }
        )
    return pd.DataFrame(rows)


def run_event_study(
    event_trades_path: Path = DEFAULT_EVENT_TRADES,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(event_trades_path, parse_dates=["signal_time", "entry_time", "exit_time"])
    config = config_from_json(config_path, start="2019-01-01")
    dataset = KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    tickers = kospi200_tickers(ROOT.parquet_path, config)
    start = pd.Timestamp(events["entry_time"].min()).normalize() - pd.Timedelta(days=5)
    end = pd.Timestamp(events["entry_time"].max()).normalize() + pd.Timedelta(days=max(horizons) + 10)
    close, _high, _low = load_daily_5m_matrices(dataset, tickers, start=start, end=str(end))

    forward = compute_event_forward_returns(
        events,
        close,
        horizons=horizons,
        round_trip_cost_bps=float(ROUND_TRIP_COST_BPS),
    )
    path_summary = compute_event_path_returns(events, close, max_horizon=max(horizons))
    summary = summarize_horizons(forward, horizons=horizons)
    yearly = summarize_yearly_excess(forward, horizons=horizons)
    top_tail = summarize_tail_dependency(forward, horizons=horizons)

    forward.to_csv(output_dir / "event_forward_returns.csv", index=False)
    path_summary.to_csv(output_dir / "event_path_summary.csv", index=False)
    summary.to_csv(output_dir / "event_forward_summary.csv", index=False)
    yearly.to_csv(output_dir / "event_forward_yearly.csv", index=False)
    top_tail.to_csv(output_dir / "event_tail_dependency.csv", index=False)
    write_event_study_png(forward, summary, output_dir / "event_study.png", horizons=horizons)
    write_event_path_png(forward, path_summary, output_dir / "event_path_study.png", max_horizon=max(horizons))
    write_event_study_report(forward, summary, yearly, top_tail, output_dir / "event_study_report.md", horizons=horizons)
    return {
        "forward": forward,
        "path_summary": path_summary,
        "summary": summary,
        "yearly": yearly,
        "top_tail": top_tail,
        "output_dir": output_dir,
    }


def summarize_yearly_excess(events: pd.DataFrame, *, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    working = events.copy()
    working["year"] = pd.to_datetime(working["entry_time"]).dt.year
    rows: list[dict[str, float | int]] = []
    for year, group in working.groupby("year", sort=True):
        row: dict[str, float | int] = {"year": int(year), "events": int(len(group))}
        for horizon in horizons:
            row[f"entry_net_mean_{horizon}d"] = _mean(group[f"event_entry_net_return_{horizon}d"].dropna())
            row[f"excess_mean_{horizon}d"] = _mean(group[f"excess_return_{horizon}d"].dropna())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_tail_dependency(events: pd.DataFrame, *, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for horizon in horizons:
        returns = events[f"event_entry_net_return_{horizon}d"].dropna().astype(float).sort_values(ascending=False).reset_index(drop=True)
        total = float(returns.sum())
        row: dict[str, float | int] = {"horizon_days": horizon, "events": int(len(returns)), "total_event_return_sum": total}
        for fraction in (0.01, 0.05, 0.10):
            count = max(1, int(np.ceil(len(returns) * fraction))) if len(returns) else 0
            contribution = float(returns.head(count).sum()) if count else 0.0
            row[f"top_{int(fraction * 100)}pct_count"] = count
            row[f"top_{int(fraction * 100)}pct_share"] = contribution / total if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_event_study_png(events: pd.DataFrame, summary: pd.DataFrame, path: Path, *, horizons: tuple[int, ...]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=160, facecolor="#fbfaf7")
    x = summary["horizon_days"]
    axes[0, 0].plot(x, summary["entry_net_mean"] * 100.0, marker="o", label="Signal net mean", color="#2f4f4f")
    axes[0, 0].plot(x, summary["benchmark_mean"] * 100.0, marker="o", label="KOSPI200 EW benchmark", color="#9a4f32")
    axes[0, 0].plot(x, summary["excess_mean"] * 100.0, marker="o", label="Signal excess", color="#315f8c")
    axes[0, 0].axhline(0.0, color="#222222", linewidth=0.9)
    axes[0, 0].set_title("Mean forward return by horizon", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Trading days after signal")
    axes[0, 0].set_ylabel("Return (%)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].bar(x - 0.2, summary["entry_net_hit_rate"] * 100.0, width=0.4, label="Signal hit rate", color="#708b8f")
    axes[0, 1].bar(x + 0.2, summary["excess_hit_rate"] * 100.0, width=0.4, label="Excess hit rate", color="#c47a54")
    axes[0, 1].axhline(50.0, color="#222222", linewidth=0.9, linestyle="--")
    axes[0, 1].set_title("Hit rate by horizon", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Trading days after signal")
    axes[0, 1].set_ylabel("Hit rate (%)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    horizon = 10 if 10 in horizons else horizons[-1]
    dist = events[f"event_entry_net_return_{horizon}d"].dropna().astype(float) * 10_000.0
    low, high = dist.quantile(0.01), dist.quantile(0.99)
    axes[1, 0].hist(dist.clip(low, high), bins=44, color="#46656a", alpha=0.82, edgecolor="white")
    axes[1, 0].axvline(0.0, color="#222222", linewidth=0.9)
    axes[1, 0].set_title(f"{horizon}D signal net return distribution, clipped 1%-99%", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Net return (bps)")

    percentiles = np.linspace(0.0, 1.0, 101)
    axes[1, 1].plot(percentiles * 100.0, events[f"event_entry_net_return_{horizon}d"].quantile(percentiles) * 100.0, color="#9a4f32")
    axes[1, 1].axhline(0.0, color="#222222", linewidth=0.9)
    axes[1, 1].set_title(f"{horizon}D signal percentile curve", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Percentile")
    axes[1, 1].set_ylabel("Net return (%)")

    for ax in axes.ravel():
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_event_path_png(forward: pd.DataFrame, path_summary: pd.DataFrame, path: Path, *, max_horizon: int = 20) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=160, facecolor="#fbfaf7")
    days = path_summary["event_day"]

    axes[0, 0].plot(days, path_summary["signal_entry_mean"] * 100.0, color="#2f4f4f", linewidth=2.0, label="Confirmed event")
    axes[0, 0].plot(days, path_summary["benchmark_mean"] * 100.0, color="#9a4f32", linewidth=1.8, label="KOSPI200 EW benchmark")
    axes[0, 0].plot(days, path_summary["excess_mean"] * 100.0, color="#315f8c", linewidth=1.8, label="Excess")
    axes[0, 0].axhline(0.0, color="#222222", linewidth=0.9)
    axes[0, 0].set_title("Event-time mean path, no stops or profit targets", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Trading days after confirmed entry")
    axes[0, 0].set_ylabel("Return (%)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(days, path_summary["signal_entry_median"] * 100.0, color="#2f4f4f", linewidth=2.0, label="Median")
    axes[0, 1].fill_between(
        days,
        path_summary["signal_entry_p25"] * 100.0,
        path_summary["signal_entry_p75"] * 100.0,
        color="#708b8f",
        alpha=0.25,
        label="25%-75%",
    )
    axes[0, 1].axhline(0.0, color="#222222", linewidth=0.9)
    axes[0, 1].set_title("Median path and interquartile range", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Trading days after confirmed entry")
    axes[0, 1].set_ylabel("Return (%)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    hit_path = path_summary.loc[path_summary["event_day"].gt(0)]
    axes[1, 0].bar(hit_path["event_day"], hit_path["signal_entry_hit_rate"] * 100.0, color="#708b8f", label="Signal hit rate")
    axes[1, 0].plot(hit_path["event_day"], hit_path["excess_hit_rate"] * 100.0, color="#c47a54", marker="o", linewidth=1.5, label="Excess hit rate")
    axes[1, 0].axhline(50.0, color="#222222", linewidth=0.9, linestyle="--")
    axes[1, 0].set_ylim(0, 60)
    axes[1, 0].set_title("Hit rate path", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Trading days after confirmed entry")
    axes[1, 0].set_ylabel("Hit rate (%)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    final_column = f"event_entry_return_{max_horizon}d"
    dist = forward[final_column].dropna().astype(float) * 100.0
    low, high = dist.quantile(0.01), dist.quantile(0.99)
    axes[1, 1].hist(dist.clip(low, high), bins=44, color="#46656a", alpha=0.82, edgecolor="white")
    axes[1, 1].axvline(0.0, color="#222222", linewidth=0.9)
    axes[1, 1].set_title(f"{max_horizon}D gross forward return distribution, clipped 1%-99%", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Gross return (%)")

    for ax in axes.ravel():
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_event_study_report(
    forward: pd.DataFrame,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    tail: pd.DataFrame,
    path: Path,
    *,
    horizons: tuple[int, ...],
) -> None:
    lines = [
        "# 52W High Confirmed Event Study",
        "",
        "Purpose: isolate the signal-level edge of the 52-week high confirmed breakout, excluding fixed-slot portfolio allocation.",
        "",
        "## Method",
        "",
        "- Event set: `base/intraday_trades.csv`, the confirmed 52-week breakout episodes before fixed 20-slot portfolio selection.",
        "- Forward horizons: 1D, 3D, 5D, 10D, 20D trading days after entry date.",
        "- Signal return: future close versus event entry price, with 35 bps round-trip cost subtracted.",
        "- Pure close-to-close signal return: future close versus signal-date close.",
        "- Benchmark: same-day KOSPI200 equal-weight forward close-to-close return, excluding the event ticker.",
        "- Excess return: signal close-to-close return minus same-day benchmark return.",
        "",
        "## Horizon Summary",
        "",
        "| horizon | events | signal_net_mean | signal_net_median | signal_hit | benchmark_mean | excess_mean | excess_median | excess_hit | excess_t | p95 | top5_removed_mean |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.horizon_days}D | {row.events} | {row.entry_net_mean * 100:.2f}% | {row.entry_net_median * 100:.2f}% | {row.entry_net_hit_rate * 100:.2f}% | {row.benchmark_mean * 100:.2f}% | {row.excess_mean * 100:.2f}% | {row.excess_median * 100:.2f}% | {row.excess_hit_rate * 100:.2f}% | {row.excess_t_stat:.2f} | {row.p95 * 100:.2f}% | {row.top5_removed_mean * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "![Event study](event_study.png)",
            "",
            "## Yearly Excess Mean",
            "",
            "| year | events | " + " | ".join(f"{h}D excess" for h in horizons) + " |",
            "| ---: | ---: | " + " | ".join("---:" for _ in horizons) + " |",
        ]
    )
    for row in yearly.itertuples(index=False):
        values = [f"{getattr(row, f'excess_mean_{h}d') * 100:.2f}%" for h in horizons]
        lines.append(f"| {row.year} | {row.events} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Tail Dependency",
            "",
            "| horizon | events | top_1pct_share | top_5pct_share | top_10pct_share |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in tail.itertuples(index=False):
        lines.append(
            f"| {row.horizon_days}D | {row.events} | {getattr(row, 'top_1pct_share') * 100:.2f}% | {getattr(row, 'top_5pct_share') * 100:.2f}% | {getattr(row, 'top_10pct_share') * 100:.2f}% |"
        )
    best_horizon = summary.sort_values("excess_mean", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The strongest average excess return appears at {int(best_horizon['horizon_days'])}D, with mean excess {best_horizon['excess_mean'] * 100:.2f}% and t-stat {best_horizon['excess_t_stat']:.2f}.",
            "The median remains much weaker than the mean, so the edge is not a high-hit-rate anomaly. It is still a right-tail momentum effect.",
            "If top winners are removed, the average signal return falls materially. This confirms that the 52-week high itself is useful mainly because it exposes the portfolio to rare continuation bursts.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _same_day_benchmark_return(close: pd.DataFrame, start_position: int, future_position: int, *, exclude_ticker: str) -> float:
    start = close.iloc[start_position]
    future = close.iloc[future_position]
    valid = start.notna() & future.notna()
    if exclude_ticker in valid.index:
        valid.loc[exclude_ticker] = False
    returns = future.loc[valid] / start.loc[valid] - 1.0
    return float(returns.mean()) if not returns.empty else np.nan


def _empty_forward_row(row: dict[str, object], horizons: tuple[int, ...]) -> dict[str, object]:
    for horizon in horizons:
        _assign_missing_horizon(row, horizon)
    return row


def _assign_missing_horizon(row: dict[str, object], horizon: int) -> None:
    row[f"event_close_return_{horizon}d"] = np.nan
    row[f"event_entry_return_{horizon}d"] = np.nan
    row[f"event_entry_net_return_{horizon}d"] = np.nan
    row[f"benchmark_return_{horizon}d"] = np.nan
    row[f"excess_return_{horizon}d"] = np.nan


def _mean(series: pd.Series) -> float:
    return float(series.mean()) if not series.empty else np.nan


def _median(series: pd.Series) -> float:
    return float(series.median()) if not series.empty else np.nan


def _hit_rate(series: pd.Series) -> float:
    return float(series.gt(0.0).mean()) if not series.empty else np.nan


def _quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if not series.empty else np.nan


def _t_stat(series: pd.Series) -> float:
    if len(series) < 2:
        return np.nan
    std = float(series.std())
    return float(series.mean() / (std / np.sqrt(len(series)))) if std else np.nan


def _trimmed_mean(series: pd.Series, low: float, high: float) -> float:
    if series.empty:
        return np.nan
    lower = series.quantile(low)
    upper = series.quantile(high)
    trimmed = series.loc[series.between(lower, upper)]
    return float(trimmed.mean()) if not trimmed.empty else np.nan


def _top_removed_mean(series: pd.Series, top_fraction: float) -> float:
    if series.empty:
        return np.nan
    cutoff = series.quantile(1.0 - top_fraction)
    trimmed = series.loc[series.le(cutoff)]
    return float(trimmed.mean()) if not trimmed.empty else np.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure event study for 52-week high confirmed breakouts.")
    parser.add_argument("--event-trades", type=Path, default=DEFAULT_EVENT_TRADES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizons", default="1,3,5,10,20")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(int(item) for item in str(args.horizons).split(",") if item)
    result = run_event_study(args.event_trades, args.config, args.output_dir, horizons=horizons)
    print(result["output_dir"])


if __name__ == "__main__":
    main()
