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

from backtesting.data.kr_stock_5m import KrStock5mDataset, read_tickers_bars
from root import ROOT
from scripts.run_flow_filtered_breakout_single import config_from_json, load_daily_5m_matrices
from scripts.tech_gamma_universe import kospi200_tickers


DEFAULT_RESEARCH_DIR = (
    ROOT.results_path
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
)
DEFAULT_EVENT_FORWARD = DEFAULT_RESEARCH_DIR / "52w_event_study" / "event_forward_returns.csv"
DEFAULT_CONFIG = DEFAULT_RESEARCH_DIR / "variants" / "5m_new_high_only" / "base" / "config.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RESEARCH_DIR / "52w_winner_feature_audit"
TARGET_COLUMN = "event_entry_return_20d"


NUMERIC_FEATURES = [
    "signal_minute",
    "breakout_52w_bps",
    "confirmation_bps",
    "entry_open_bps",
    "confirmation_return_bps",
    "signal_body_to_range",
    "signal_close_location",
    "signal_range_pct",
    "upper_wick_to_range",
    "lower_wick_to_range",
    "volume_spike",
    "vwap_bps",
    "ret_15m_bps",
    "ret_30m_bps",
    "same_day_event_count",
    "benchmark_return_20d",
    "event_entry_return_1d",
    "event_entry_return_5d",
]


def label_winner_buckets(
    frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    top_fraction: float = 0.05,
    bottom_fraction: float = 0.50,
) -> pd.DataFrame:
    if target_column not in frame.columns:
        raise ValueError(f"missing target column: {target_column}")
    working = frame.copy()
    valid = working[target_column].dropna().sort_values(ascending=False)
    top_count = max(1, int(np.ceil(len(valid) * top_fraction))) if len(valid) else 0
    bottom_count = max(1, int(np.ceil(len(valid) * bottom_fraction))) if len(valid) else 0
    top_index = set(valid.head(top_count).index)
    bottom_index = set(valid.tail(bottom_count).index)
    working["is_top_winner"] = working.index.to_series().isin(top_index)
    working["is_bottom_group"] = working.index.to_series().isin(bottom_index)
    return working


def summarize_numeric_features(
    frame: pd.DataFrame,
    features: list[str],
    *,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for feature in features:
        if feature not in frame.columns:
            continue
        values = frame[[feature, target_column, "is_top_winner", "is_bottom_group"]].dropna(subset=[feature, target_column])
        if values.empty:
            continue
        top = values.loc[values["is_top_winner"], feature]
        rest = values.loc[~values["is_top_winner"], feature]
        bottom = values.loc[values["is_bottom_group"], feature]
        all_std = float(values[feature].std())
        rows.append(
            {
                "feature": feature,
                "observations": int(len(values)),
                "all_mean": float(values[feature].mean()),
                "all_median": float(values[feature].median()),
                "top_winner_mean": float(top.mean()) if not top.empty else np.nan,
                "top_winner_median": float(top.median()) if not top.empty else np.nan,
                "rest_mean": float(rest.mean()) if not rest.empty else np.nan,
                "bottom_group_mean": float(bottom.mean()) if not bottom.empty else np.nan,
                "top_minus_rest": float(top.mean() - rest.mean()) if not top.empty and not rest.empty else np.nan,
                "top_minus_rest_z": float((top.mean() - rest.mean()) / all_std) if all_std else np.nan,
                "spearman_corr": float(values[[feature, target_column]].corr(method="spearman").iloc[0, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("top_minus_rest_z", key=lambda item: item.abs(), ascending=False).reset_index(drop=True)


def run_audit(
    event_forward_path: Path = DEFAULT_EVENT_FORWARD,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(event_forward_path, parse_dates=["signal_time", "entry_time", "event_date"])
    events = label_winner_buckets(events, target_column=TARGET_COLUMN)
    config = config_from_json(config_path, start="2019-01-01")
    dataset = KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    tickers = kospi200_tickers(ROOT.parquet_path, config)
    start = pd.Timestamp(events["entry_time"].min()).normalize() - pd.Timedelta(days=5)
    end = pd.Timestamp(events["entry_time"].max()).normalize() + pd.Timedelta(days=30)
    close, _high, _low = load_daily_5m_matrices(dataset, tickers, start=start, end=str(end))
    prior_high = close.shift(1).rolling(252, min_periods=1).max()

    signal_features = reconstruct_signal_features(events, dataset=dataset, prior_high=prior_high)
    audit = events.merge(signal_features, on=["ticker", "signal_time", "entry_time"], how="left", sort=False)
    audit["same_day_event_count"] = audit.groupby(pd.to_datetime(audit["entry_time"]).dt.normalize())["ticker"].transform("count")
    audit["t1_accept_close_above_entry"] = audit["event_entry_return_1d"].gt(0.0)
    audit["t5_accept_close_above_entry"] = audit["event_entry_return_5d"].gt(0.0)

    feature_summary = summarize_numeric_features(audit, NUMERIC_FEATURES, target_column=TARGET_COLUMN)
    deciles = summarize_feature_deciles(audit, feature_summary["feature"].head(8).tolist(), target_column=TARGET_COLUMN)
    bucket_summary = summarize_bucket_returns(audit, target_column=TARGET_COLUMN)

    audit.to_csv(output_dir / "event_feature_audit.csv", index=False)
    feature_summary.to_csv(output_dir / "winner_feature_summary.csv", index=False)
    deciles.to_csv(output_dir / "feature_decile_returns.csv", index=False)
    bucket_summary.to_csv(output_dir / "winner_bucket_summary.csv", index=False)
    write_audit_png(feature_summary, deciles, bucket_summary, output_dir / "winner_feature_audit.png")
    write_audit_report(feature_summary, deciles, bucket_summary, output_dir / "winner_feature_audit_report.md")
    return {
        "audit": audit,
        "feature_summary": feature_summary,
        "deciles": deciles,
        "bucket_summary": bucket_summary,
        "output_dir": output_dir,
    }


def reconstruct_signal_features(
    events: pd.DataFrame,
    *,
    dataset: KrStock5mDataset,
    prior_high: pd.DataFrame,
) -> pd.DataFrame:
    working = events[["ticker", "signal_time", "entry_time", "entry_price"]].copy()
    working["signal_month"] = working["signal_time"].dt.to_period("M").astype(str)
    parts: list[pd.DataFrame] = []
    for month, month_events in working.groupby("signal_month", sort=True):
        month_start = pd.Period(month, freq="M").to_timestamp()
        read_start = month_start - pd.Timedelta(days=7)
        read_end = pd.Period(month, freq="M").end_time
        raw = read_tickers_bars(
            dataset,
            tuple(sorted(month_events["ticker"].unique())),
            start=read_start,
            end=read_end,
        )
        if raw.empty:
            continue
        features = _intraday_signal_frame(raw)
        selected = features.merge(
            month_events[["ticker", "signal_time", "entry_time", "entry_price"]],
            left_on=["ticker", "ts"],
            right_on=["ticker", "signal_time"],
            how="inner",
            sort=False,
        )
        parts.append(selected)
    if not parts:
        return pd.DataFrame(columns=["ticker", "signal_time", "entry_time"])
    signals = pd.concat(parts, ignore_index=True)
    signals["date"] = pd.to_datetime(signals["signal_time"]).dt.normalize()
    prior_long = prior_high.stack(future_stack=True).rename("prior_52w_close_high").reset_index()
    prior_long.columns = ["date", "ticker", "prior_52w_close_high"]
    signals = signals.merge(prior_long, on=["date", "ticker"], how="left", sort=False)
    price_range = (signals["high"] - signals["low"]).replace(0.0, np.nan)
    body = signals["close"] - signals["open"]
    upper_wick = signals["high"] - signals[["open", "close"]].max(axis=1)
    lower_wick = signals[["open", "close"]].min(axis=1) - signals["low"]
    signal_minute = signals["signal_time"].dt.hour * 60 + signals["signal_time"].dt.minute

    out = pd.DataFrame(
        {
            "ticker": signals["ticker"].astype(str),
            "signal_time": pd.to_datetime(signals["signal_time"]),
            "entry_time": pd.to_datetime(signals["entry_time"]),
            "signal_minute": signal_minute.astype(float),
            "signal_open": signals["open"].astype(float),
            "signal_high": signals["high"].astype(float),
            "signal_low": signals["low"].astype(float),
            "signal_close": signals["close"].astype(float),
            "prior_52w_close_high": signals["prior_52w_close_high"].astype(float),
            "breakout_52w_bps": (signals["close"] / signals["prior_52w_close_high"] - 1.0) * 10_000.0,
            "confirmation_bps": (signals["next_close"] / signals["prior_52w_close_high"] - 1.0) * 10_000.0,
            "entry_open_bps": (signals["entry_price"] / signals["prior_52w_close_high"] - 1.0) * 10_000.0,
            "confirmation_return_bps": (signals["next_close"] / signals["close"] - 1.0) * 10_000.0,
            "signal_body_to_range": body.divide(price_range),
            "signal_close_location": (signals["close"] - signals["low"]).divide(price_range),
            "signal_range_pct": price_range.divide(signals["close"]),
            "upper_wick_to_range": upper_wick.divide(price_range),
            "lower_wick_to_range": lower_wick.divide(price_range),
            "volume_spike": signals["volume_spike"].astype(float),
            "vwap_bps": (signals["close"] / signals["vwap"] - 1.0) * 10_000.0,
            "ret_15m_bps": signals["ret_15m"].fillna(0.0).astype(float) * 10_000.0,
            "ret_30m_bps": signals["ret_30m"].fillna(0.0).astype(float) * 10_000.0,
        }
    )
    return out.drop_duplicates(["ticker", "signal_time", "entry_time"]).reset_index(drop=True)


def summarize_feature_deciles(audit: pd.DataFrame, features: list[str], *, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for feature in features:
        if feature not in audit.columns:
            continue
        working = audit[[feature, target_column]].dropna()
        if working[feature].nunique() < 3:
            continue
        try:
            working["decile"] = pd.qcut(working[feature], q=10, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        grouped = working.groupby("decile", sort=True)
        for decile, group in grouped:
            rows.append(
                {
                    "feature": feature,
                    "decile": int(decile),
                    "events": int(len(group)),
                    "feature_mean": float(group[feature].mean()),
                    "return_20d_mean": float(group[target_column].mean()),
                    "return_20d_median": float(group[target_column].median()),
                    "hit_rate_20d": float(group[target_column].gt(0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_bucket_returns(audit: pd.DataFrame, *, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    groups = {
        "top_5pct_winners": audit["is_top_winner"],
        "rest": ~audit["is_top_winner"],
        "bottom_50pct": audit["is_bottom_group"],
        "all": pd.Series(True, index=audit.index),
    }
    rows: list[dict[str, float | int | str]] = []
    for name, mask in groups.items():
        returns = audit.loc[mask, target_column].dropna()
        rows.append(
            {
                "bucket": name,
                "events": int(len(returns)),
                "mean_20d": float(returns.mean()) if not returns.empty else np.nan,
                "median_20d": float(returns.median()) if not returns.empty else np.nan,
                "hit_rate_20d": float(returns.gt(0.0).mean()) if not returns.empty else np.nan,
                "p95_20d": float(returns.quantile(0.95)) if not returns.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_audit_png(summary: pd.DataFrame, deciles: pd.DataFrame, buckets: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=160, facecolor="#fbfaf7")
    top = summary.head(12).sort_values("top_minus_rest_z")
    axes[0, 0].barh(top["feature"], top["top_minus_rest_z"], color=["#9a4f32" if item < 0 else "#2f4f4f" for item in top["top_minus_rest_z"]])
    axes[0, 0].axvline(0.0, color="#222222", linewidth=0.9)
    axes[0, 0].set_title("Top 5% winners vs rest: standardized feature gap", loc="left", fontweight="bold")

    corr = summary.reindex(summary["spearman_corr"].abs().sort_values(ascending=False).index).head(12).sort_values("spearman_corr")
    axes[0, 1].barh(corr["feature"], corr["spearman_corr"], color=["#9a4f32" if item < 0 else "#315f8c" for item in corr["spearman_corr"]])
    axes[0, 1].axvline(0.0, color="#222222", linewidth=0.9)
    axes[0, 1].set_title("Spearman correlation with 20D forward return", loc="left", fontweight="bold")

    for feature, group in deciles.groupby("feature", sort=False):
        axes[1, 0].plot(group["decile"], group["return_20d_mean"] * 100.0, marker="o", linewidth=1.3, label=feature)
    axes[1, 0].axhline(0.0, color="#222222", linewidth=0.9)
    axes[1, 0].set_title("20D mean return by feature decile", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Feature decile")
    axes[1, 0].set_ylabel("20D return (%)")
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    axes[1, 1].bar(buckets["bucket"], buckets["mean_20d"] * 100.0, color="#708b8f")
    axes[1, 1].axhline(0.0, color="#222222", linewidth=0.9)
    axes[1, 1].set_title("20D return by outcome bucket", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("Mean 20D return (%)")
    axes[1, 1].tick_params(axis="x", rotation=20)

    for ax in axes.ravel():
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_audit_report(summary: pd.DataFrame, deciles: pd.DataFrame, buckets: pd.DataFrame, path: Path) -> None:
    lines = [
        "# 52W Winner Feature Audit",
        "",
        "Purpose: diagnose common pre-event features of strong 20D winners without defining a new strategy rule.",
        "",
        "## Outcome Buckets",
        "",
        "| bucket | events | mean_20d | median_20d | hit_rate_20d | p95_20d |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in buckets.itertuples(index=False):
        lines.append(f"| {row.bucket} | {row.events} | {row.mean_20d * 100:.2f}% | {row.median_20d * 100:.2f}% | {row.hit_rate_20d * 100:.2f}% | {row.p95_20d * 100:.2f}% |")
    lines.extend(
        [
            "",
            "## Feature Differences",
            "",
            "| feature | observations | top_mean | rest_mean | bottom_mean | top_minus_rest_z | spearman_corr |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.head(18).itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.observations} | {row.top_winner_mean:.4f} | {row.rest_mean:.4f} | {row.bottom_group_mean:.4f} | {row.top_minus_rest_z:.3f} | {row.spearman_corr:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "This is an audit, not a strategy rule. Any candidate idea must be re-tested as a pre-declared rule on out-of-sample or walk-forward splits.",
            "Large standardized gaps are useful as diagnostics only when the feature has a plausible market microstructure interpretation.",
            "",
            "![Winner feature audit](winner_feature_audit.png)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _intraday_signal_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    frame["ts"] = pd.to_datetime(frame["ts"])
    frame = frame.sort_values(["ticker", "ts"]).reset_index(drop=True)
    frame["date"] = frame["ts"].dt.normalize()
    grouped = frame.groupby(["ticker", "date"], sort=False)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    traded_value = typical * frame["volume"]
    volume_sum = frame["volume"].groupby([frame["ticker"], frame["date"]]).cumsum()
    frame["vwap"] = traded_value.groupby([frame["ticker"], frame["date"]]).cumsum().divide(volume_sum.replace(0.0, np.nan))
    frame["volume_base"] = grouped["volume"].transform(lambda item: item.shift(1).rolling(6, min_periods=2).mean())
    frame["volume_spike"] = frame["volume"].divide(frame["volume_base"].replace(0.0, np.nan)).fillna(0.0)
    frame["ret_15m"] = grouped["close"].pct_change(3)
    frame["ret_30m"] = grouped["close"].pct_change(6)
    frame["next_close"] = grouped["close"].shift(-1)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit features of strong 52-week high confirmed breakout winners.")
    parser.add_argument("--event-forward", type=Path, default=DEFAULT_EVENT_FORWARD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(args.event_forward, args.config, args.output_dir)
    print(result["output_dir"])


if __name__ == "__main__":
    main()
