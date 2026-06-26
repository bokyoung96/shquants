from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from etc.sell_sidecar_economics import (
    build_rule_trades,
    default_rules,
    pair_sidecar_events,
    recommend_rules,
    summarize_rule_trades,
)


DEFAULT_PARQUET_DIR = Path("etc/data/sidecar/parquet")
DEFAULT_OUT_DIR = Path("results/sell_sidecar_economics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sell-sidecar economic strategy report.")
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def write_markdown(
    out_dir: Path,
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
    recommended: pd.DataFrame,
) -> None:
    best = recommended.iloc[0] if not recommended.empty else summary.iloc[0]
    default = summary[summary["rule"] == "trigger_plus3_release_plus3"].iloc[0]

    lines = [
        "# Sell-Sidecar Strategy Economics",
        "",
        "## Economic Read",
        "",
        "- A sell-sidecar is not just a price pattern. It is an exchange-level confirmation that futures-led downside pressure has become disorderly enough to halt program trading.",
        "- The useful behavior is not immediate panic chasing. The first minutes after activation are noisy; the better tradable signal is confirmed sell-side imbalance after the initial dislocation.",
        "- The core economic edge is `shock_continuation`: after a sell-sidecar activation, de-risking, hedge adjustment, and delayed ETF/index-futures alignment can continue during and shortly after the halt window.",
        "- `residual_pressure` also exists, but entries must use the next complete minute after release. Same-minute release entries can overstate tradability.",
        "- The sample is small, so the recommendation favors economically coherent, executable rules over the highest optimized return.",
        "",
        "## Data",
        "",
        f"- Sell-sidecar activations paired with releases: `{len(pairs)}`",
        f"- First event: `{pairs['activation_dt'].min()}`",
        f"- Last event: `{pairs['activation_dt'].max()}`",
        "- Instrument for sell-sidecar trades: `KODEX inverse` 1-minute close.",
        "- Returns are gross; transaction costs and slippage are not deducted.",
        "",
        "## Recommended Rule",
        "",
        f"- Rule: `{best['rule']}`",
        f"- Economic role: `{best['economic_role']}`",
        f"- Takeaway: {best['economic_takeaway']}",
        f"- Trades: `{int(best['n'])}`",
        f"- Win rate: `{best['win_rate'] * 100:.1f}%`",
        f"- Mean return: `{pct(best['mean_ret'])}`",
        f"- Compound return: `{pct(best['compound_ret'])}`",
        f"- Max drawdown: `{pct(best['max_drawdown'])}`",
        "",
        "## Conservative Core Rule",
        "",
        "- Rule: `trigger_plus3_release_plus3`",
        "- Interpretation: buy inverse after the activation has had a few minutes to confirm, then exit shortly after release.",
        f"- Win rate: `{default['win_rate'] * 100:.1f}%`",
        f"- Mean return: `{pct(default['mean_ret'])}`",
        f"- Compound return: `{pct(default['compound_ret'])}`",
        "- This is less optimized than the top rule and better as the baseline trading recipe.",
        "",
        "## Rule Table",
        "",
        "| rule | role | n | win | mean | compound | max DD | thesis |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    table = summary.sort_values("compound_ret", ascending=False)
    for _, row in table.iterrows():
        lines.append(
            f"| {row['rule']} | {row['economic_role']} | {int(row['n'])} | "
            f"{row['win_rate'] * 100:.1f}% | {pct(row['mean_ret'])} | "
            f"{pct(row['compound_ret'])} | {pct(row['max_drawdown'])} | {row['thesis']} |"
        )
    lines.extend(
        [
            "",
            "## Implementation Implication",
            "",
            "- Use `trigger_plus3_release_plus3` as the default executable rule.",
            "- Treat `trigger_plus5_release_plus10` and `release_nextbar_60m` as research variants, not default live rules, until more events accumulate.",
            "- Do not enter at activation minute close. Wait at least 3 minutes after activation or one full minute after release.",
        ]
    )
    (out_dir / "sell_sidecar_economics.md").write_text("\n".join(lines), encoding="utf-8")


def plot_summary(out_dir: Path, summary: pd.DataFrame) -> None:
    plot = summary.sort_values("compound_ret", ascending=False).reset_index(drop=True)
    labels = [_short_label(rule) for rule in plot["rule"]]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), dpi=160, gridspec_kw={"height_ratios": [1.35, 1.0]})
    fig.subplots_adjust(hspace=0.55)
    colors = ["#2f6f73" if value >= 0 else "#b84a4a" for value in plot["compound_ret"]]
    axes[0].bar(range(len(plot)), plot["compound_ret"] * 100.0, color=colors, edgecolor="#29323a", linewidth=0.6)
    axes[0].axhline(0, color="#1f2933", linewidth=0.9)
    axes[0].set_title("Sell-Sidecar Rule Economics", loc="left", fontsize=15, fontweight="bold")
    axes[0].set_ylabel("compound return (%)")
    axes[0].set_xticks(range(len(plot)))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].grid(axis="y", color="#d9dee3", alpha=0.9)
    for idx, value in enumerate(plot["compound_ret"] * 100.0):
        axes[0].text(idx, value / 2, f"{value:.1f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    axes[1].plot(range(len(plot)), plot["win_rate"] * 100.0, marker="o", color="#566b84", linewidth=2)
    axes[1].set_title("Win Rate by Rule", loc="left", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("win rate (%)")
    axes[1].set_xticks(range(len(plot)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].grid(color="#d9dee3", alpha=0.9)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f7f8fa")
    fig.patch.set_facecolor("#f7f8fa")
    fig.savefig(out_dir / "sell_sidecar_economics.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _short_label(rule: str) -> str:
    labels = {
        "trigger_plus5_close": "T+5\nclose",
        "trigger_plus5_release_plus10": "T+5\nR+10",
        "release_nextbar_60m": "R next\n+60m",
        "release_nextbar_close": "R next\nclose",
        "trigger_plus3_release_plus3": "T+3\nR+3",
    }
    return labels.get(rule, rule.replace("_", "\n"))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(args.parquet_dir / "sidecar_events.parquet")
    inverse = pd.read_parquet(args.parquet_dir / "kodex_inverse_1m.parquet")
    pairs = pair_sidecar_events(events, direction="sell")
    trades = build_rule_trades(pairs, inverse, default_rules())
    summary = summarize_rule_trades(trades)
    recommended = recommend_rules(summary, min_trades=10)

    pairs.to_csv(args.out_dir / "sell_sidecar_pairs.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(args.out_dir / "sell_sidecar_rule_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_dir / "sell_sidecar_rule_summary.csv", index=False, encoding="utf-8-sig")
    recommended.to_csv(args.out_dir / "sell_sidecar_recommended.csv", index=False, encoding="utf-8-sig")
    write_markdown(args.out_dir, pairs, summary, recommended)
    plot_summary(args.out_dir, summary)

    print(summary.sort_values("compound_ret", ascending=False).to_string(index=False))
    print(f"out={args.out_dir}")


if __name__ == "__main__":
    main()
