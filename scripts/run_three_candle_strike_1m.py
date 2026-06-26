from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from backtesting.strategies.intraday_three_candle_strike import (
    BacktestConfig,
    compute_indicators,
    run_backtest_from_signals,
)


DEFAULT_INPUT = Path("parquet/KOSPI200_1m.parquet")
DEFAULT_OUT_DIR = Path("results/three_candle_strike_1m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest 3 Candle Strike on 1-minute OHLC data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.0, 0.000005, 0.00001, 0.00002])
    parser.add_argument("--round-trip-cost-bps", type=float, default=2.0)
    parser.add_argument("--atr-buffer-fraction", type=float, default=0.05)
    parser.add_argument("--rr", type=float, default=3.0)
    return parser.parse_args()


def load_ohlc(path: Path) -> pd.DataFrame:
    cols = ["ts", "open", "high", "low", "close", "volume", "trade_date_kst", "hhmm_kst"]
    df = pd.read_parquet(path, columns=cols)
    df = df.dropna(subset=["open", "high", "low", "close", "ts", "trade_date_kst", "hhmm_kst"])
    df = df.sort_values(["trade_date_kst", "hhmm_kst", "ts"]).reset_index(drop=True)
    df["trade_date_kst"] = pd.to_datetime(df["trade_date_kst"]).dt.date
    return df


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else _format_float(float(value)))
        else:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))
    widths = {col: max(len(col), display[col].str.len().max()) for col in display.columns}
    header = "| " + " | ".join(col.ljust(widths[col]) for col in display.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def _format_float(value: float) -> str:
    if abs(value) < 0.0001 and value != 0.0:
        return f"{value:.8g}"
    return f"{value:.4f}"


def write_report(out_dir: Path, source: Path, summary: pd.DataFrame, best_trades: pd.DataFrame) -> None:
    best = summary.sort_values(["total_net_bps", "trades"], ascending=[False, False]).iloc[0]
    lines = [
        "# 3 Candle Strike 1-Minute Backtest",
        "",
        "## Setup",
        "",
        f"- Input: `{source.as_posix()}`",
        f"- Date range: `{best['start']}` to `{best['end']}`",
        "- Instrument: local KOSPI200 1-minute OHLC parquet.",
        "- Signal timestamp discipline: pattern is confirmed at candle close; entry uses next bar open.",
        "- Trend filter: SMMA(21) vs SMMA(50), with SMMA(21) direction.",
        "- Cross-reversal add-on: EMA(8) / SMMA(21) cross in the signal direction within 10 bars can also qualify direction.",
        "- Range filter: normalized rolling linear-regression slope must exceed threshold.",
        "- Stop: recent 4-bar extreme plus previous daily ATR buffer.",
        "- Fixed target: 3R. Trailing mode moves stop after 3R touch and exits on trailing stop or session end.",
        "",
        "## Grid Summary",
        "",
        markdown_table(
            summary[
                [
                    "trailing",
                    "slope_threshold",
                    "trades",
                    "long_trades",
                    "short_trades",
                    "hit_rate",
                    "avg_net_bps",
                    "median_net_bps",
                    "total_net_bps",
                    "avg_r",
                    "profit_factor",
                    "max_drawdown_bps",
                ]
            ]
        ),
        "",
        "## Best Run Trade Sample",
        "",
        markdown_table(
            best_trades[
                [
                    "trade_date",
                    "side",
                    "entry_ts",
                    "exit_ts",
                    "entry_price",
                    "exit_price",
                    "exit_reason",
                    "net_bps",
                    "r_multiple",
                ]
            ].head(20)
        ),
        "",
        "## Read",
        "",
        f"- Best in this small grid: trailing=`{best['trailing']}`, threshold `{best['slope_threshold']}`.",
        f"- Net result: `{best['total_net_bps']:.2f}` bps over `{int(best['trades'])}` trades.",
        "- Treat this as a mechanical hypothesis test, not a deployable strategy verdict. Fill ordering is conservative when stop and target are both touched in one bar.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_ohlc(args.input)
    signals = compute_indicators(df, slope_threshold=0.0)

    base = BacktestConfig(
        rr=args.rr,
        round_trip_cost_bps=args.round_trip_cost_bps,
        atr_buffer_fraction=args.atr_buffer_fraction,
    )
    summary_rows = []
    best_trades = pd.DataFrame()
    best_total = float("-inf")
    for trailing in (False, True):
        for threshold in args.thresholds:
            config = replace(base, slope_threshold=threshold, trailing=trailing)
            result = run_backtest_from_signals(signals, config)
            row = dict(result.summary)
            row["config"] = json.dumps(asdict(config), sort_keys=True)
            summary_rows.append(row)
            suffix = f"{'trailing' if trailing else 'fixed'}_slope_{threshold:g}".replace(".", "p")
            result.trades.to_csv(args.out_dir / f"trades_{suffix}.csv", index=False, encoding="utf-8-sig")
            if float(row["total_net_bps"]) > best_total:
                best_total = float(row["total_net_bps"])
                best_trades = result.trades.copy()

    summary = pd.DataFrame(summary_rows).sort_values(["total_net_bps", "trades"], ascending=[False, False])
    summary.to_csv(args.out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_report(args.out_dir, args.input, summary, best_trades)

    print(
        summary[
            ["trailing", "slope_threshold", "trades", "hit_rate", "avg_net_bps", "total_net_bps", "max_drawdown_bps"]
        ].to_string(index=False, float_format=_format_float)
    )
    print(f"out={args.out_dir}")


if __name__ == "__main__":
    main()
