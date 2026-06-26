"""Calendar-time equity curves for filtered liquidation-absorption strategies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("results/forced_liquidation_absorption_test")
TRADES = BASE / "trades.csv"
DAILY = Path("results/forced_liquidation_event_test/daily_returns.csv")
OUT_PNG = BASE / "filtered_absorption_calendar_equity.png"
OUT_CSV = BASE / "filtered_absorption_calendar_stats.csv"


STRATEGIES = [
    ("T+1 dip30/recover35 no-stop exit 10:00", "T+1", "dip30_recover35_no_stop", "1000"),
    ("T+1 time-only no-stop exit 10:00", "T+1", "time_only_no_stop", "1000"),
    ("T+1 time-only no-stop exit 10:30", "T+1", "time_only_no_stop", "1030"),
    ("T+1 dip20/recover35 no-stop exit 10:00", "T+1", "dip20_recover35_no_stop", "1000"),
]


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def main() -> None:
    trades = pd.read_csv(TRADES)
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    daily = pd.read_csv(DAILY)
    # daily_returns.csv stores the date in the first index column.
    date_col = daily.columns[0]
    dates = pd.to_datetime(daily[date_col])
    calendar = pd.DataFrame({"trade_date": dates}).drop_duplicates().sort_values("trade_date")

    curves: list[pd.DataFrame] = []
    stats: list[dict[str, object]] = []
    start = calendar["trade_date"].min()
    end = calendar["trade_date"].max()
    years = (end - start).days / 365.25

    for label, sample, filt, exit_hhmm in STRATEGIES:
        subset = trades[
            (trades["sample"] == sample)
            & (trades["filter"] == filt)
            & (trades["exit_hhmm"].astype(str).str.zfill(4) == exit_hhmm)
        ][["trade_date", "ret"]].copy()
        subset = subset.drop_duplicates("trade_date", keep="last")
        curve = calendar.merge(subset, on="trade_date", how="left")
        curve["ret"] = curve["ret"].fillna(0.0)
        curve["equity"] = (1.0 + curve["ret"]).cumprod()
        curve["cum_return"] = curve["equity"] - 1.0
        curve["label"] = label
        curves.append(curve)

        traded = subset["ret"].dropna()
        total_return = float(curve["equity"].iloc[-1] - 1.0)
        stats.append(
            {
                "strategy": label,
                "trades": int(len(traded)),
                "trades_per_year": float(len(traded) / years),
                "total_return": total_return,
                "cagr_cash_idle": float(curve["equity"].iloc[-1] ** (1.0 / years) - 1.0),
                "avg_trade_return": float(traded.mean()) if len(traded) else 0.0,
                "win_rate": float((traded > 0).mean()) if len(traded) else 0.0,
                "max_drawdown": max_drawdown(curve["equity"]),
                "first_trade": subset["trade_date"].min(),
                "last_trade": subset["trade_date"].max(),
            }
        )

    curves_df = pd.concat(curves, ignore_index=True)
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    curves_df.to_csv(BASE / "filtered_absorption_calendar_equity.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(15, 8), dpi=170)
    colors = ["#b83232", "#d79028", "#7a1f1f", "#88451d"]
    for color, (label, *_rest) in zip(colors, STRATEGIES):
        data = curves_df[curves_df["label"] == label]
        ax.plot(data["trade_date"], data["cum_return"] * 100.0, label=label, linewidth=2.1, color=color)
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title("Calendar-Time Equity: Forced-Liquidation Absorption Candidates", fontsize=17, fontweight="bold")
    ax.set_ylabel("Cumulative return with idle cash (%)")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=9, loc="upper left")
    fig.text(
        0.01,
        0.01,
        "Calendar equity assumes 1x notional exposure only during signal windows and 0% return while idle. Costs/slippage excluded.",
        fontsize=9,
        color="#4b5563",
    )
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
