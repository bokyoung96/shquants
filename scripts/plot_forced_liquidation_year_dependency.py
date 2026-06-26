"""Plot yearly return contribution for forced-liquidation absorption strategies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("results/forced_liquidation_absorption_test")
ANNUAL = BASE / "annual_strategy_returns.csv"
OUT = BASE / "year_dependency_returns.png"


def main() -> None:
    df = pd.read_csv(ANNUAL)
    pivot = df.pivot(index="year", columns="strategy", values="annual_return").fillna(0.0)
    selected = [
        "T+1 time-only no-stop exit 10:30",
        "T+1 time-only no-stop exit 10:00",
        "T+1 dip30/recover35 no-stop exit 10:00",
        "T+1 dip20/recover35 no-stop exit 10:00",
    ]
    pivot = pivot[[c for c in selected if c in pivot.columns]]

    ax = (pivot * 100.0).plot(kind="bar", figsize=(16, 8), width=0.82)
    ax.set_title("Annual Return Contribution: Is the Edge Mostly 2026?", fontsize=17, fontweight="bold")
    ax.set_ylabel("Annual contribution (%)")
    ax.set_xlabel("Year")
    ax.axhline(0, color="#6b7280", linewidth=1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT, dpi=170)
    plt.close()
    print(OUT)


if __name__ == "__main__":
    main()
