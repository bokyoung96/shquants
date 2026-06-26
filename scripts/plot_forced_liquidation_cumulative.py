"""Plot cumulative returns for forced-liquidation event-test candidates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("results/forced_liquidation_event_test")
SAMPLES = BASE / "event_window_samples.csv"
OUT_PNG = BASE / "forced_liquidation_cumulative_returns.png"
OUT_EVENT_PNG = BASE / "forced_liquidation_cumulative_returns_by_trade.png"
OUT_CSV = BASE / "forced_liquidation_cumulative_returns.csv"


CANDIDATES = [
    {
        "label": "T+1 09:15->10:00",
        "window": "open_1st_0900",
        "sample": "T+1",
        "metric": "end_to_exit",
        "color": "#b83232",
        "linewidth": 2.8,
    },
    {
        "label": "T+1 09:15->Close",
        "window": "open_1st_0900",
        "sample": "T+1",
        "metric": "end_to_close",
        "color": "#7a1f1f",
        "linewidth": 2.2,
    },
    {
        "label": "T+2 10:30->11:00",
        "window": "cfd_1000",
        "sample": "T+2",
        "metric": "end_to_exit",
        "color": "#cc6b1f",
        "linewidth": 2.0,
    },
    {
        "label": "T+1 10:30->11:00",
        "window": "cfd_1000",
        "sample": "T+1",
        "metric": "end_to_exit",
        "color": "#d79028",
        "linewidth": 1.9,
    },
    {
        "label": "T+1 14:30->15:00",
        "window": "stockloan_1400",
        "sample": "T+1",
        "metric": "end_to_exit",
        "color": "#88451d",
        "linewidth": 1.9,
    },
]


def configure_korean_font() -> None:
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def build_curves(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves: list[pd.DataFrame] = []
    stats: list[dict[str, object]] = []

    for spec in CANDIDATES:
        subset = samples[
            (samples["window"] == spec["window"])
            & (samples["sample"] == spec["sample"])
        ].copy()
        subset["trade_date"] = pd.to_datetime(subset["trade_date"])
        subset = subset.sort_values(["trade_date", "event_date"])
        returns = subset[["trade_date", "event_date", spec["metric"]]].dropna()
        returns = returns.rename(columns={spec["metric"]: "ret"})
        # If overlapping crash events point to the same T+1/T+2 date, keep the
        # latest event annotation but trade only once per date.
        returns = returns.drop_duplicates(subset=["trade_date"], keep="last")
        returns["label"] = spec["label"]
        returns["event_number"] = range(1, len(returns) + 1)
        returns["equity"] = (1.0 + returns["ret"]).cumprod()
        returns["cum_return"] = returns["equity"] - 1.0
        curves.append(returns)
        stats.append(
            {
                "label": spec["label"],
                "trades": len(returns),
                "total_return": float(returns["equity"].iloc[-1] - 1.0) if len(returns) else 0.0,
                "mean_return": float(returns["ret"].mean()) if len(returns) else 0.0,
                "win_rate": float((returns["ret"] > 0).mean()) if len(returns) else 0.0,
                "max_drawdown": max_drawdown(returns["equity"]) if len(returns) else 0.0,
            }
        )

    curves_df = pd.concat(curves, ignore_index=True)
    stats_df = pd.DataFrame(stats)
    return curves_df, stats_df


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def plot(curves: pd.DataFrame, stats: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 10), dpi=180)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.3, 1.2], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for spec in CANDIDATES:
        label = spec["label"]
        data = curves[curves["label"] == label].sort_values("trade_date")
        ax.plot(
            data["trade_date"],
            data["cum_return"] * 100.0,
            label=label,
            color=spec["color"],
            linewidth=spec["linewidth"],
        )

    ax.axhline(0, color="#6b7280", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title(
        "Cumulative Returns: Buying Liquidation-Time Dips After Large Down Days",
        fontsize=18,
        fontweight="bold",
        pad=16,
    )
    ax.set_ylabel("Cumulative return (%)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", alpha=0.12)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    summary = stats.copy()
    summary["Total"] = summary["total_return"].map(pct)
    summary["Avg"] = summary["mean_return"].map(lambda x: f"{x * 100:.3f}%")
    summary["Win rate"] = summary["win_rate"].map(pct)
    summary["MDD"] = summary["max_drawdown"].map(pct)
    table_data = summary[["label", "trades", "Total", "Avg", "Win rate", "MDD"]].values.tolist()

    ax2.axis("off")
    table = ax2.table(
        cellText=table_data,
        colLabels=["Strategy", "Trades", "Total", "Avg/trade", "Win rate", "Max DD"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.36, 0.10, 0.12, 0.14, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6dee8")
        if row == 0:
            cell.set_facecolor("#12314a")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f8")
        else:
            cell.set_facecolor("#ffffff")

    fig.text(
        0.015,
        0.012,
        "Note: KOSPI200 1-minute futures/index bars. Forced-liquidation orders are not directly labeled. Costs and slippage are not included.",
        fontsize=9,
        color="#4b5563",
    )
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    fig2, ax_event = plt.subplots(figsize=(16, 8), dpi=180)
    for spec in CANDIDATES:
        label = spec["label"]
        data = curves[curves["label"] == label].sort_values("event_number")
        ax_event.plot(
            data["event_number"],
            data["cum_return"] * 100.0,
            label=label,
            color=spec["color"],
            linewidth=spec["linewidth"],
        )
    ax_event.axhline(0, color="#6b7280", linewidth=1, linestyle="--", alpha=0.7)
    ax_event.set_title(
        "Cumulative Returns by Event Trade Number",
        fontsize=18,
        fontweight="bold",
        pad=16,
    )
    ax_event.set_xlabel("Event trade number", fontsize=12)
    ax_event.set_ylabel("Cumulative return (%)", fontsize=12)
    ax_event.grid(True, axis="y", alpha=0.25)
    ax_event.grid(True, axis="x", alpha=0.12)
    ax_event.legend(loc="upper left", frameon=True, fontsize=10)
    fig2.text(
        0.015,
        0.012,
        "Each step is one post-selloff T+1/T+2 trade; overlapping signals on the same date are traded once per strategy.",
        fontsize=9,
        color="#4b5563",
    )
    fig2.savefig(OUT_EVENT_PNG, bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    configure_korean_font()
    samples = pd.read_csv(SAMPLES)
    curves, stats = build_curves(samples)
    curves.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    stats.to_csv(BASE / "forced_liquidation_cumulative_stats.csv", index=False, encoding="utf-8-sig")
    plot(curves, stats)
    print(OUT_PNG)
    print(OUT_EVENT_PNG)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
