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

from backtesting.data.kr_stock_5m import KrStock5mDataset, read_tickers_bars
from root import ROOT
from scripts.run_flow_filtered_breakout_single import config_from_json
from scripts.run_tech_gamma_long_only import build_features
from scripts.verified_flow_backtest import fixed_notional_mtm_ledger, profit_factor


DEFAULT_RESEARCH_DIR = (
    ROOT.results_path
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
)
DEFAULT_SOURCE_DIR = DEFAULT_RESEARCH_DIR / "variants" / "5m_new_high_only"
DEFAULT_SELECTED_TRADES = DEFAULT_SOURCE_DIR / "fixed20" / "selected_trades.csv"
DEFAULT_CONFIG = DEFAULT_SOURCE_DIR / "base" / "config.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RESEARCH_DIR / "winner_holding_regime_comparison"


@dataclass(frozen=True, slots=True)
class WinnerRegimeConfig:
    atr_stop_multiplier: float = 1.0
    min_holding_days: int = 1
    round_trip_cost_bps: float = 35.0
    slots: int = 20


def compare_regimes(
    entries: pd.DataFrame,
    daily: pd.DataFrame,
    config: WinnerRegimeConfig = WinnerRegimeConfig(),
) -> dict[str, pd.DataFrame]:
    return {
        "baseline": simulate_exit_regime(entries, daily, config, regime="baseline"),
        "breakeven_1r": simulate_exit_regime(entries, daily, config, regime="breakeven_1r"),
    }


def simulate_exit_regime(
    entries: pd.DataFrame,
    daily: pd.DataFrame,
    config: WinnerRegimeConfig = WinnerRegimeConfig(),
    *,
    regime: str,
) -> pd.DataFrame:
    if regime not in {"baseline", "breakeven_1r"}:
        raise ValueError(f"unknown regime: {regime}")
    if entries.empty:
        return _empty_trades()

    daily_groups = {
        str(ticker): group.assign(date=pd.to_datetime(group["date"]).dt.normalize()).sort_values("date").reset_index(drop=True)
        for ticker, group in daily.groupby("ticker", sort=True)
    }
    rows: list[dict[str, object]] = []
    for ticker, ticker_entries in entries.groupby("ticker", sort=True):
        ticker_daily = daily_groups.get(str(ticker))
        if ticker_daily is None:
            continue
        for _, entry in ticker_entries.sort_values("signal_time").iterrows():
            trade = _simulate_trade(entry, ticker_daily, config, regime=regime)
            if trade is not None:
                rows.append(trade)
    if not rows:
        return _empty_trades()
    return pd.DataFrame(rows).sort_values(["entry_time", "ticker"]).reset_index(drop=True)


def run_experiment(
    selected_trades_path: Path = DEFAULT_SELECTED_TRADES,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(selected_trades_path, parse_dates=["signal_time", "entry_time", "exit_time"])
    tech_config = config_from_json(config_path, start="2019-01-01")
    dataset = KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    entries = attach_signal_atr(selected, dataset, tech_config)
    daily = load_daily_5m_ohlc(dataset, tuple(sorted(entries["ticker"].unique())), start=entries["date"].min() - pd.Timedelta(days=430), end=entries["exit_time"].max())
    daily_for_exits = daily.loc[daily["date"].ge(entries["date"].min())].reset_index(drop=True)
    close = daily.pivot(index="date", columns="ticker", values="close")

    trades_by_regime = compare_regimes(entries, daily_for_exits, WinnerRegimeConfig())
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, pd.DataFrame] = {}
    for name, trades in trades_by_regime.items():
        fixed, _missing = fixed_notional_mtm_ledger(trades, close, slots=20)
        trades.to_csv(output_dir / f"{name}_trades.csv", index=False)
        fixed.to_csv(output_dir / f"{name}_fixed_notional_ledger.csv", index_label="date")
        ledgers[name] = fixed
        rows.append(_metrics_row(name, trades, fixed))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "winner_holding_regime_metrics.csv", index=False)
    (output_dir / "winner_holding_regime_config.json").write_text(json.dumps(asdict(WinnerRegimeConfig()), indent=2), encoding="utf-8")
    write_dashboard(ledgers, trades_by_regime, metrics, output_dir / "winner_holding_regime_comparison.png")
    write_report(metrics, output_dir / "winner_holding_regime_report.md")
    return metrics


def attach_signal_atr(selected: pd.DataFrame, dataset: KrStock5mDataset, tech_config: Any) -> pd.DataFrame:
    working = selected.copy()
    working["signal_month"] = working["signal_time"].dt.to_period("M").astype(str)
    parts: list[pd.DataFrame] = []
    for month, month_trades in working.groupby("signal_month", sort=True):
        month_start = pd.Period(month, freq="M").to_timestamp()
        read_start = max(pd.Timestamp(tech_config.start) - pd.Timedelta(days=tech_config.high_lookback_days), month_start - pd.Timedelta(days=10))
        read_end = pd.Period(month, freq="M").end_time
        raw = read_tickers_bars(dataset, tuple(sorted(month_trades["ticker"].unique())), start=read_start, end=read_end)
        if raw.empty:
            continue
        features = build_features(raw.dropna(subset=["open", "high", "low", "close", "volume"]).copy(), tech_config)
        parts.append(features[["ticker", "ts", "date", "atr", "signal_score"]])
    if not parts:
        raise ValueError("could not reconstruct signal ATR features")
    features = pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "ts"])
    entries = working.merge(features, left_on=["ticker", "signal_time"], right_on=["ticker", "ts"], how="left", sort=False)
    missing = int(entries["atr"].isna().sum())
    if missing:
        raise ValueError(f"missing reconstructed ATR for {missing} selected trades")
    return pd.DataFrame(
        {
            "ticker": entries["ticker"].astype(str),
            "date": pd.to_datetime(entries["date"]).dt.normalize(),
            "signal_time": pd.to_datetime(entries["signal_time"]),
            "entry_time": pd.to_datetime(entries["entry_time"]),
            "exit_time": pd.to_datetime(entries["exit_time"]),
            "entry_price": entries["entry_price"].astype(float),
            "atr": entries["atr"].astype(float),
            "signal_score": entries["signal_score_y"].fillna(entries.get("signal_score_x")).astype(float)
            if "signal_score_y" in entries
            else entries["signal_score"].astype(float),
        }
    )


def load_daily_5m_ohlc(
    dataset: KrStock5mDataset,
    tickers: tuple[str, ...],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for month in pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"):
        close_path = dataset.field_path(str(month), "c")
        if not close_path.exists():
            continue
        available = set(pq.read_schema(close_path).names)
        selected = [ticker for ticker in tickers if ticker in available]
        if not selected:
            continue
        high = pd.read_parquet(dataset.field_path(str(month), "h"), columns=selected, engine="pyarrow")
        low = pd.read_parquet(dataset.field_path(str(month), "l"), columns=selected, engine="pyarrow")
        close = pd.read_parquet(close_path, columns=selected, engine="pyarrow")
        high.index = pd.to_datetime(high.index).normalize()
        low.index = pd.to_datetime(low.index).normalize()
        close.index = pd.to_datetime(close.index).normalize()
        daily = pd.concat(
            [
                high.groupby(level=0).max().stack(future_stack=True).rename("daily_high"),
                low.groupby(level=0).min().stack(future_stack=True).rename("daily_low"),
                close.groupby(level=0).last().stack(future_stack=True).rename("close"),
            ],
            axis=1,
        ).reset_index()
        daily.columns = ["date", "ticker", "daily_high", "daily_low", "close"]
        rows.append(daily)
    if not rows:
        raise ValueError("no daily 5m OHLC data loaded")
    frame = pd.concat(rows, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.loc[frame["date"].between(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())]
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    frame["prior_52w_close_high"] = frame.groupby("ticker", sort=False)["close"].transform(lambda item: item.shift(1).rolling(252, min_periods=1).max())
    return frame.dropna(subset=["close", "daily_high", "daily_low", "prior_52w_close_high"]).reset_index(drop=True)


def write_dashboard(
    ledgers: dict[str, pd.DataFrame],
    trades_by_regime: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=160, facecolor="#fbfaf7")
    colors = {"baseline": "#2f4f4f", "breakeven_1r": "#b45f3c"}
    for name, ledger in ledgers.items():
        axes[0, 0].plot(ledger.index, (ledger["equity"] - 1.0) * 100.0, label=_label(name, metrics), color=colors[name], linewidth=1.6)
        axes[1, 0].plot(ledger.index, ledger["drawdown"] * 100.0, color=colors[name], linewidth=1.0)
        axes[0, 1].plot(ledger.index, ledger["active_positions"], color=colors[name], linewidth=0.9, alpha=0.72)
    for name, trades in trades_by_regime.items():
        returns = trades["net_return"] * 10_000.0
        axes[1, 1].hist(returns.clip(returns.quantile(0.01), returns.quantile(0.99)), bins=36, alpha=0.45, label=name, color=colors[name])
    axes[0, 0].set_title("Fixed 20-slot cumulative return", loc="left", fontweight="bold")
    axes[1, 0].set_title("Drawdown", loc="left", fontweight="bold")
    axes[0, 1].set_title("Active positions", loc="left", fontweight="bold")
    axes[1, 1].set_title("Trade return distribution (1%-99% clipped)", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, path: Path) -> None:
    display = metrics.copy()
    for column in ["fixed_return", "mdd", "avg_trade_return", "median_trade_return", "hit_rate", "worst_trade", "best_trade"]:
        display[f"{column}_pct"] = display[column] * 100.0
    display["avg_trade_bps"] = display["avg_trade_return"] * 10_000.0
    lines = [
        "# Winner Holding Regime Comparison",
        "",
        "The entry set is fixed to the canonical `52W High 5M Breakout + ATR` selected trades. This isolates exit/holding behavior from entry filtering.",
        "",
        "- `baseline`: original ATR stop or 52-week close-high loss exit.",
        "- `breakeven_1r`: same as baseline, but after price reaches entry + initial risk, the active stop moves to entry price.",
        "- Same-day high/low ordering is treated conservatively: the breakeven stop can only matter after the strategy has survived the current daily stop check.",
        "",
        "| strategy | trades | fixed_return_pct | mdd_pct | avg_bps | median_pct | hit_rate_pct | profit_factor | atr_stop | breakeven_stop | new_high_lost | winner_1r |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.trades} | {row.fixed_return_pct:.2f} | {row.mdd_pct:.2f} | {row.avg_trade_bps:.2f} | {row.median_trade_return_pct:.2f} | {row.hit_rate_pct:.2f} | {row.profit_factor:.3f} | {row.atr_stop_count} | {row.breakeven_stop_count} | {row.new_high_lost_count} | {row.winner_1r_count} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _simulate_trade(entry: pd.Series, daily: pd.DataFrame, config: WinnerRegimeConfig, *, regime: str) -> dict[str, object] | None:
    entry_date = pd.Timestamp(entry["date"]).normalize()
    entry_price = float(entry["entry_price"])
    initial_risk = float(entry["atr"]) * config.atr_stop_multiplier
    if initial_risk <= 0.0:
        return None
    initial_stop = entry_price - initial_risk
    one_r_price = entry_price + initial_risk
    winner_active = False
    holding_days = (pd.to_datetime(daily["date"]) - entry_date).dt.days
    candidates = daily.loc[holding_days.ge(config.min_holding_days)]
    if candidates.empty:
        return None
    exit_row = candidates.iloc[-1]
    exit_reason = "end_of_data"
    exit_price = float(exit_row["close"])
    winner_reached = False
    for _, row in candidates.iterrows():
        row_close = float(row["close"])
        active_stop = entry_price if winner_active and regime == "breakeven_1r" else initial_stop
        if float(row["daily_low"]) <= active_stop:
            exit_row = row
            exit_reason = "breakeven_stop" if active_stop == entry_price else "atr_stop"
            exit_price = active_stop
            break
        if regime == "breakeven_1r" and float(row["daily_high"]) >= one_r_price:
            winner_active = True
            winner_reached = True
        if row_close <= float(row["prior_52w_close_high"]):
            exit_row = row
            exit_reason = "new_high_lost"
            exit_price = row_close
            break
    gross = exit_price / entry_price - 1.0
    return {
        "ticker": str(entry["ticker"]),
        "side": "long",
        "signal_time": pd.Timestamp(entry["signal_time"]),
        "entry_time": pd.Timestamp(entry["entry_time"]),
        "exit_time": pd.Timestamp(exit_row["date"]).normalize() + pd.Timedelta(hours=15, minutes=30),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "initial_stop": initial_stop,
        "one_r_price": one_r_price,
        "winner_1r_reached": bool(winner_reached),
        "signal_score": float(entry["signal_score"]),
        "gross_return": gross,
        "net_return": gross - config.round_trip_cost_bps / 10_000.0,
        "exit_reason": exit_reason,
    }


def _metrics_row(strategy: str, trades: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, Any]:
    returns = trades["net_return"] if not trades.empty else pd.Series(dtype=float)
    exits = trades["exit_reason"].value_counts()
    return {
        "strategy": strategy,
        "trades": int(len(trades)),
        "fixed_return": float(ledger["equity"].iloc[-1] - 1.0) if not ledger.empty else 0.0,
        "mdd": float(ledger["drawdown"].min()) if not ledger.empty else 0.0,
        "avg_trade_return": float(returns.mean()) if not returns.empty else 0.0,
        "median_trade_return": float(returns.median()) if not returns.empty else 0.0,
        "hit_rate": float(returns.gt(0.0).mean()) if not returns.empty else 0.0,
        "profit_factor": profit_factor(returns),
        "worst_trade": float(returns.min()) if not returns.empty else 0.0,
        "best_trade": float(returns.max()) if not returns.empty else 0.0,
        "atr_stop_count": int(exits.get("atr_stop", 0)),
        "breakeven_stop_count": int(exits.get("breakeven_stop", 0)),
        "new_high_lost_count": int(exits.get("new_high_lost", 0)),
        "end_of_data_count": int(exits.get("end_of_data", 0)),
        "winner_1r_count": int(trades["winner_1r_reached"].sum()) if "winner_1r_reached" in trades else 0,
    }


def _label(name: str, metrics: pd.DataFrame) -> str:
    row = metrics.loc[metrics["strategy"].eq(name)].iloc[0]
    return f"{name}: {row['fixed_return'] * 100.0:.1f}%, MDD {row['mdd'] * 100.0:.1f}%"


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "side",
            "signal_time",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "initial_stop",
            "one_r_price",
            "winner_1r_reached",
            "signal_score",
            "gross_return",
            "net_return",
            "exit_reason",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline exit with a 1R breakeven winner holding regime.")
    parser.add_argument("--selected-trades", type=Path, default=DEFAULT_SELECTED_TRADES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.selected_trades, args.config, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
