"""Test filtered "liquidation absorption" entries after large down days.

Idea:
- A large down day T may trigger forced selling on T+1/T+2.
- Do not buy just because it is 09:15.
- Buy only if the 09:00-09:15 window actually shows a selloff and partial
  absorption/recovery.
- Exit quickly at 09:30/09:45/10:00, with an optional stop at the 09:00-09:15 low.

The test uses KOSPI200 1-minute futures/index bars. It does not directly observe
forced-liquidation orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path("parquet/KOSPI200_1m.parquet")
OUT_DIR = Path("results/forced_liquidation_absorption_test")


@dataclass(frozen=True)
class FilterSpec:
    name: str
    min_pressure: float
    min_recovery: float
    ofi_rule: str
    use_stop: bool


FILTERS = [
    FilterSpec("time_only_stop", 0.0, 0.0, "none", True),
    FilterSpec("time_only_no_stop", 0.0, 0.0, "none", False),
    FilterSpec("dip20_recover35_stop", 0.002, 0.35, "none", True),
    FilterSpec("dip20_recover35_no_stop", 0.002, 0.35, "none", False),
    FilterSpec("dip30_recover35_stop", 0.003, 0.35, "none", True),
    FilterSpec("dip30_recover35_no_stop", 0.003, 0.35, "none", False),
    FilterSpec("dip30_recover50_stop", 0.003, 0.50, "none", True),
    FilterSpec("dip40_recover50_stop", 0.004, 0.50, "none", True),
    FilterSpec("dip30_recover50_ofi_improve_stop", 0.003, 0.50, "improve", True),
    FilterSpec("dip30_recover50_ofi_positive_stop", 0.003, 0.50, "last5_positive", True),
    FilterSpec("dip30_recover50_no_stop", 0.003, 0.50, "none", False),
    FilterSpec("dip40_recover50_no_stop", 0.004, 0.50, "none", False),
]

EXITS = ["0920", "0925", "0930", "0935", "0940", "0945", "0950", "0955", "1000", "1015", "1030"]


def first_at_or_after(day: pd.DataFrame, hhmm: str) -> pd.Series | None:
    subset = day[day["hhmm_kst"] >= hhmm]
    return None if subset.empty else subset.iloc[0]


def last_at_or_before(day: pd.DataFrame, hhmm: str) -> pd.Series | None:
    subset = day[day["hhmm_kst"] <= hhmm]
    return None if subset.empty else subset.iloc[-1]


def load_data() -> pd.DataFrame:
    cols = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_vol",
        "sell_vol",
        "net_buy_vol",
        "ofi",
        "trade_date_kst",
        "hhmm_kst",
    ]
    df = pd.read_parquet(INPUT, columns=cols)
    df = df.sort_values(["trade_date_kst", "hhmm_kst", "ts"]).reset_index(drop=True)
    df["trade_date_kst"] = pd.to_datetime(df["trade_date_kst"]).dt.date
    return df


def daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("trade_date_kst", sort=True)
    first = grouped.head(1).set_index("trade_date_kst")
    last = grouped.tail(1).set_index("trade_date_kst")
    daily = pd.DataFrame({"open": first["open"], "close": last["close"], "bar_count": grouped.size()})
    daily["ret_cc"] = daily["close"].pct_change()
    daily["ret_oc"] = daily["close"] / daily["open"] - 1.0
    daily["date_index"] = np.arange(len(daily))
    return daily


def mark_events(daily: pd.DataFrame, cc_threshold: float = -0.025, oc_threshold: float = -0.020) -> pd.DataFrame:
    events = daily[(daily["ret_cc"] <= cc_threshold) | (daily["ret_oc"] <= oc_threshold)].copy()
    events["event_date"] = events.index
    events["event_idx"] = events["date_index"].astype(int)
    events["event_reason"] = np.where(
        (events["ret_cc"] <= cc_threshold) & (events["ret_oc"] <= oc_threshold),
        "cc_and_oc",
        np.where(events["ret_cc"] <= cc_threshold, "close_to_close", "open_to_close"),
    )
    return events.reset_index(drop=True)


def target_map(daily: pd.DataFrame, events: pd.DataFrame) -> dict[tuple[object, int], dict[str, object]]:
    dates = list(daily.index)
    out: dict[tuple[object, int], dict[str, object]] = {}
    for _, event in events.iterrows():
        event_idx = int(event["event_idx"])
        for lag in (1, 2):
            target_idx = event_idx + lag
            if target_idx >= len(dates):
                continue
            # If consecutive crash days map to the same target, keep the latest
            # event. In live trading, the latest crash is the relevant trigger.
            out[(dates[target_idx], lag)] = event.to_dict()
    return out


def window_features(day: pd.DataFrame) -> dict[str, object] | None:
    window = day[(day["hhmm_kst"] >= "0900") & (day["hhmm_kst"] <= "0915")]
    if len(window) < 8:
        return None
    start = first_at_or_after(day, "0900")
    entry = last_at_or_before(day, "0915")
    if start is None or entry is None:
        return None

    low_idx = window["low"].idxmin()
    low_row = day.loc[low_idx]
    high_px = float(window["high"].max())
    low_px = float(low_row["low"])
    start_px = float(start["close"])
    entry_px = float(entry["close"])
    rng = high_px - low_px
    recovery = (entry_px - low_px) / rng if rng > 0 else 0.0

    first10 = window[window["hhmm_kst"] <= "0910"]
    last5 = window[window["hhmm_kst"] >= "0911"]
    ofi_first = float(first10["ofi"].mean(skipna=True))
    ofi_last = float(last5["ofi"].mean(skipna=True))
    net_last = float(last5["net_buy_vol"].sum(skipna=True))
    buy_vol = float(window["buy_vol"].sum(skipna=True))
    sell_vol = float(window["sell_vol"].sum(skipna=True))
    total_aggr = buy_vol + sell_vol

    return {
        "start_px": start_px,
        "entry_px": entry_px,
        "low_px": low_px,
        "high_px": high_px,
        "low_hhmm": low_row["hhmm_kst"],
        "pressure": low_px / start_px - 1.0,
        "pressure_abs": 1.0 - low_px / start_px,
        "recovery": recovery,
        "ofi_first10": ofi_first,
        "ofi_last5": ofi_last,
        "ofi_improvement": ofi_last - ofi_first,
        "net_buy_last5": net_last,
        "sell_share": sell_vol / total_aggr if total_aggr else np.nan,
        "window_volume": float(window["volume"].sum()),
        "entry_hhmm": "0915",
    }


def passes_filter(features: dict[str, object], spec: FilterSpec) -> bool:
    if float(features["pressure_abs"]) < spec.min_pressure:
        return False
    if float(features["recovery"]) < spec.min_recovery:
        return False
    if spec.ofi_rule == "improve" and float(features["ofi_improvement"]) <= 0:
        return False
    if spec.ofi_rule == "last5_positive" and float(features["ofi_last5"]) <= 0:
        return False
    return True


def simulate_exit(day: pd.DataFrame, features: dict[str, object], exit_hhmm: str, use_stop: bool) -> dict[str, object] | None:
    entry_px = float(features["entry_px"])
    stop_px = float(features["low_px"])
    path = day[(day["hhmm_kst"] > "0915") & (day["hhmm_kst"] <= exit_hhmm)]
    exit_row = last_at_or_before(day, exit_hhmm)
    if path.empty or exit_row is None:
        return None

    stop_hit = False
    stop_hhmm = None
    if use_stop:
        stop_path = path[path["low"] <= stop_px]
        if not stop_path.empty:
            stop_hit = True
            stop_hhmm = stop_path.iloc[0]["hhmm_kst"]
            exit_px = stop_px
        else:
            exit_px = float(exit_row["close"])
    else:
        exit_px = float(exit_row["close"])

    return {
        "exit_hhmm": exit_hhmm,
        "exit_px": exit_px,
        "ret": exit_px / entry_px - 1.0,
        "stop_hit": stop_hit,
        "stop_hhmm": stop_hhmm,
    }


def build_trades(df: pd.DataFrame, daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    by_day = {date: frame.reset_index(drop=True) for date, frame in df.groupby("trade_date_kst", sort=True)}
    targets = target_map(daily, events)
    rows: list[dict[str, object]] = []

    for trade_date, day in by_day.items():
        sample_infos = [("baseline", 0, {})]
        for lag in (1, 2):
            event = targets.get((trade_date, lag))
            if event is not None:
                sample_infos.append((f"T+{lag}", lag, event))

        features = window_features(day)
        if features is None:
            continue

        for sample, lag, event in sample_infos:
            for spec in FILTERS:
                if not passes_filter(features, spec):
                    continue
                for exit_hhmm in EXITS:
                    result = simulate_exit(day, features, exit_hhmm, spec.use_stop)
                    if result is None:
                        continue
                    rows.append(
                        {
                            "trade_date": trade_date,
                            "sample": sample,
                            "lag": lag,
                            "event_date": event.get("event_date"),
                            "event_ret_cc": event.get("ret_cc"),
                            "event_ret_oc": event.get("ret_oc"),
                            "event_reason": event.get("event_reason"),
                            "filter": spec.name,
                            "min_pressure": spec.min_pressure,
                            "min_recovery": spec.min_recovery,
                            "ofi_rule": spec.ofi_rule,
                            "use_stop": spec.use_stop,
                            **features,
                            **result,
                        }
                    )

    return pd.DataFrame(rows)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min()) if len(dd) else np.nan


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["filter", "sample", "exit_hhmm"]
    for keys, group in trades.sort_values("trade_date").groupby(group_cols, sort=False):
        filt, sample, exit_hhmm = keys
        rets = group["ret"].astype(float)
        rows.append(
            {
                "filter": filt,
                "sample": sample,
                "exit_hhmm": exit_hhmm,
                "trades": int(len(group)),
                "mean_ret": float(rets.mean()),
                "median_ret": float(rets.median()),
                "total_return": float((1.0 + rets).prod() - 1.0),
                "win_rate": float((rets > 0).mean()),
                "stop_rate": float(group["stop_hit"].mean()),
                "max_drawdown": max_drawdown(rets),
                "avg_pressure": float(group["pressure"].mean()),
                "avg_recovery": float(group["recovery"].mean()),
                "avg_ofi_improvement": float(group["ofi_improvement"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_to_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["filter", "exit_hhmm"]
    baseline = summary[summary["sample"] == "baseline"].set_index(keys)
    for _, row in summary[summary["sample"].isin(["T+1", "T+2"])].iterrows():
        key = (row["filter"], row["exit_hhmm"])
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        rows.append(
            {
                **row.to_dict(),
                "baseline_trades": int(base["trades"]),
                "baseline_mean_ret": float(base["mean_ret"]),
                "mean_ret_vs_baseline": float(row["mean_ret"] - base["mean_ret"]),
                "baseline_win_rate": float(base["win_rate"]),
                "win_rate_vs_baseline": float(row["win_rate"] - base["win_rate"]),
            }
        )
    return pd.DataFrame(rows)


def plot_top_curves(trades: pd.DataFrame, top: pd.DataFrame) -> None:
    selected = top.head(6)
    if selected.empty:
        return
    fig, ax = plt.subplots(figsize=(15, 8), dpi=170)
    colors = ["#b83232", "#7a1f1f", "#cc6b1f", "#d79028", "#88451d", "#52616b"]
    for idx, (_, row) in enumerate(selected.iterrows()):
        subset = trades[
            (trades["filter"] == row["filter"])
            & (trades["sample"] == row["sample"])
            & (trades["exit_hhmm"].astype(str).str.zfill(4) == str(row["exit_hhmm"]).zfill(4))
        ].sort_values("trade_date")
        curve = (1.0 + subset["ret"].astype(float)).cumprod() - 1.0
        label = f"{row['sample']} {row['filter']} exit {str(row['exit_hhmm']).zfill(4)}"
        ax.plot(range(1, len(curve) + 1), curve * 100.0, label=label, linewidth=2.1, color=colors[idx % len(colors)])
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title("Filtered Liquidation-Absorption Candidates", fontsize=17, fontweight="bold")
    ax.set_xlabel("Trade number")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8)
    fig.text(
        0.01,
        0.01,
        "Entry: 09:15 only after a 09:00-09:15 dip/recovery filter. Stop: prior window low when enabled. Costs/slippage excluded.",
        fontsize=9,
        color="#4b5563",
    )
    fig.savefig(OUT_DIR / "filtered_absorption_top_curves.png", bbox_inches="tight")
    plt.close(fig)


def write_report(top: pd.DataFrame) -> None:
    def pct(x: float) -> str:
        return f"{x * 100:.3f}%"

    lines = ["# Filtered Liquidation Absorption Test", ""]
    lines.append("Entry is 09:15 after the 09:00-09:15 window satisfies a dip/recovery filter.")
    lines.append("Stops, when enabled, are placed at the 09:00-09:15 low. Exits tested: 09:30, 09:45, 10:00.")
    lines.append("")
    lines.append("## Top candidates by mean return vs baseline")
    lines.append("")
    lines.append("| sample | filter | exit | trades | mean | baseline | diff | win | stop | total | MDD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in top.head(20).iterrows():
        lines.append(
            f"| {row['sample']} | {row['filter']} | {str(row['exit_hhmm']).zfill(4)} | "
            f"{int(row['trades'])} | {pct(row['mean_ret'])} | {pct(row['baseline_mean_ret'])} | "
            f"{pct(row['mean_ret_vs_baseline'])} | {row['win_rate'] * 100:.1f}% | "
            f"{row['stop_rate'] * 100:.1f}% | {pct(row['total_return'])} | {pct(row['max_drawdown'])} |"
        )
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- A high mean with very few trades is fragile; prefer candidates with at least 25-30 trades.")
    lines.append("- If `baseline` also improves under the same filter, the edge may be generic open-reversal, not forced-liquidation-specific.")
    lines.append("- Costs and slippage are excluded. Small edges below roughly 0.05-0.10% per trade should be treated skeptically.")
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    daily = daily_frame(df)
    events = mark_events(daily)
    trades = build_trades(df, daily, events)
    summary = summarize(trades)
    compared = compare_to_baseline(summary)
    top = compared[
        (compared["trades"] >= 20)
        & (compared["mean_ret_vs_baseline"] > 0)
        & (compared["sample"].isin(["T+1", "T+2"]))
    ].sort_values(["mean_ret_vs_baseline", "mean_ret"], ascending=False)

    trades.to_csv(OUT_DIR / "trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "summary_grid.csv", index=False, encoding="utf-8-sig")
    compared.to_csv(OUT_DIR / "event_vs_baseline.csv", index=False, encoding="utf-8-sig")
    top.to_csv(OUT_DIR / "top_candidates.csv", index=False, encoding="utf-8-sig")
    plot_top_curves(trades, top)
    write_report(top)

    print(f"events={len(events)}")
    print(f"trades={len(trades)}")
    print(f"top_candidates={len(top)}")
    print(f"out={OUT_DIR}")


if __name__ == "__main__":
    main()
