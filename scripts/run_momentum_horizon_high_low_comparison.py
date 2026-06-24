from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from root import ROOT

from backtesting.analytics import summarize_perf
from backtesting.data import ParquetStore
from backtesting.data.benchmarks import benchmark_price_series
from backtesting.strategies.positivity import positivity_score, return_momentum_score


OUTPUT_DIR = ROOT.results_path / "pos_research" / "momentum_horizon_high_low_2020"
START = pd.Timestamp("2020-01-01")
TRADING_DAYS_PER_MONTH = 21
HORIZONS = (
    {"label": "1M", "months": 1, "lookback_days": 21, "skip_days": 0, "sort_order": 1},
    {"label": "3M", "months": 3, "lookback_days": 63, "skip_days": 0, "sort_order": 2},
    {"label": "6M", "months": 6, "lookback_days": 126, "skip_days": 0, "sort_order": 3},
    {"label": "12M", "months": 12, "lookback_days": 252, "skip_days": 0, "sort_order": 4},
    {"label": "3-1M", "months": 3, "lookback_days": 63, "skip_days": 21, "sort_order": 5},
    {"label": "6-1M", "months": 6, "lookback_days": 126, "skip_days": 21, "sort_order": 6},
    {"label": "12-1M", "months": 12, "lookback_days": 252, "skip_days": 21, "sort_order": 7},
)
FACTOR_ORDER = ("positivity", "return_momentum", "high_sharpe", "high_low")


def main() -> None:
    result = run_comparison()
    written = write_outputs(result)
    print(json.dumps({"written": written, "top_q5": result["top_q5"]}, ensure_ascii=False, indent=2))


def run_comparison() -> dict[str, object]:
    store = ParquetStore(ROOT.parquet_path)
    close = store.read("qw_adj_c").astype(float)
    membership = store.read("qw_k200_yn").reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    benchmark = store.read("qw_BM")

    end = close.index.max()
    stock_returns = close.pct_change(fill_method=None)
    next_returns = stock_returns.shift(-1).where(membership)

    returns_by_name: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, object]] = []

    for horizon in HORIZONS:
        label = str(horizon["label"])
        lookback = int(horizon["lookback_days"])
        skip = int(horizon["skip_days"])
        window = lookback - skip
        signals = _signals_for_horizon(
            close=close,
            stock_returns=stock_returns,
            membership=membership,
            lookback=lookback,
            skip=skip,
            window=window,
        )
        for factor_name in FACTOR_ORDER:
            factor_returns = _rank_bucket_returns(signals[factor_name], next_returns)
            factor_returns = factor_returns.loc[factor_returns.index >= START].dropna(how="all")
            for bucket in ("q1", "q5"):
                series_name = f"{factor_name}_{label}_{bucket}"
                returns_by_name[series_name] = factor_returns[bucket]
                summary_rows.append(_summary_row(series_name, factor_returns[bucket], horizon, factor_name, bucket))
            spread = factor_returns["q5"].sub(factor_returns["q1"], fill_value=0.0)
            spread_name = f"{factor_name}_{label}_q5_minus_q1"
            returns_by_name[spread_name] = spread
            summary_rows.append(_summary_row(spread_name, spread, horizon, factor_name, "q5_minus_q1"))

    returns = pd.DataFrame(returns_by_name).sort_index()
    returns = returns.loc[returns.index >= START]
    benchmark_returns = _next_day_benchmark_returns(benchmark=benchmark, index=close.index).reindex(returns.index)
    returns["KOSPI200"] = benchmark_returns
    returns = returns.dropna(how="all")

    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0)
    summary = pd.DataFrame(summary_rows)
    top_q5 = (
        summary.loc[summary["leg"].eq("q5")]
        .sort_values(["sharpe", "cagr"], ascending=False)
        .head(10)
        .to_dict(orient="records")
    )
    return {
        "returns": returns,
        "equity": equity,
        "drawdown": drawdown,
        "summary": summary,
        "top_q5": top_q5,
        "config": {
            "analysis": "KOSPI200 positivity, return momentum, high Sharpe, high-low channel horizon comparison",
            "start": START.date().isoformat(),
            "warmup": "signals use all available data before start; returns are sliced to 2020-01-01 onward",
            "end": pd.Timestamp(end).date().isoformat(),
            "trading_days_per_month": TRADING_DAYS_PER_MONTH,
            "quintile_method": "row-wise percentile rank with deterministic tie break, q5 rank > 0.8, q1 rank <= 0.2",
            "return_alignment": "signal date t holds close-to-close return from t to next trading day",
            "high_low_signal": "(close at t-skip - rolling low over window) / (rolling high over window - rolling low over window)",
            "horizons": [
                {
                    **horizon,
                    "window_days": int(horizon["lookback_days"]) - int(horizon["skip_days"]),
                }
                for horizon in HORIZONS
            ],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def _signals_for_horizon(
    *,
    close: pd.DataFrame,
    stock_returns: pd.DataFrame,
    membership: pd.DataFrame,
    lookback: int,
    skip: int,
    window: int,
) -> dict[str, pd.DataFrame]:
    shifted_close = close.shift(skip) if skip else close
    shifted_returns = stock_returns.shift(skip) if skip else stock_returns
    signals = {
        "positivity": positivity_score(shifted_returns, lookback=window, min_periods=window),
        "return_momentum": return_momentum_score(shifted_close, lookback=window),
        "high_sharpe": _high_sharpe_score(shifted_returns, window=window),
        "high_low": _high_low_score(shifted_close, window=window),
    }
    return {name: signal.where(membership) for name, signal in signals.items()}


def _rank_bucket_returns(signal: pd.DataFrame, next_returns: pd.DataFrame) -> pd.DataFrame:
    common_index = signal.index.intersection(next_returns.index)
    common_columns = signal.columns.intersection(next_returns.columns)
    ranked = signal.loc[common_index, common_columns].rank(axis=1, method="first", pct=True)
    fwd = next_returns.loc[common_index, common_columns]
    q1 = fwd.where(ranked.le(0.2)).mean(axis=1)
    q5 = fwd.where(ranked.gt(0.8)).mean(axis=1)
    return pd.DataFrame({"q1": q1, "q5": q5}, index=common_index)


def _high_sharpe_score(returns: pd.DataFrame, *, window: int) -> pd.DataFrame:
    mean = returns.rolling(window, min_periods=window).mean()
    std = returns.rolling(window, min_periods=window).std(ddof=0)
    return mean.divide(std.where(std.gt(0.0)))


def _high_low_score(close: pd.DataFrame, *, window: int) -> pd.DataFrame:
    rolling_high = close.rolling(window, min_periods=window).max()
    rolling_low = close.rolling(window, min_periods=window).min()
    width = rolling_high.sub(rolling_low)
    return close.sub(rolling_low).divide(width.where(width.gt(0.0)))


def _summary_row(
    name: str,
    series: pd.Series,
    horizon: dict[str, object],
    factor_name: str,
    leg: str,
) -> dict[str, object]:
    clean = series.dropna()
    perf = summarize_perf(clean)
    equity = (1.0 + clean.fillna(0.0)).cumprod()
    return {
        "horizon": horizon["label"],
        "months": horizon["months"],
        "lookback_days": horizon["lookback_days"],
        "skip_days": horizon["skip_days"],
        "window_days": int(horizon["lookback_days"]) - int(horizon["skip_days"]),
        "sort_order": horizon["sort_order"],
        "factor": factor_name,
        "leg": leg,
        "portfolio": name,
        "observations": int(clean.count()),
        "total_return": float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan"),
        "cagr": perf["cagr"],
        "mdd": perf["mdd"],
        "sharpe": perf["sharpe"],
        "daily_win_rate": float(clean.gt(0.0).mean()) if not clean.empty else float("nan"),
        "avg_daily_return": float(clean.mean()) if not clean.empty else float("nan"),
        "daily_vol": float(clean.std(ddof=0)) if len(clean) > 1 else float("nan"),
    }


def _next_day_benchmark_returns(*, benchmark: pd.DataFrame, index: pd.Index) -> pd.Series:
    price = benchmark_price_series(benchmark, "IKS200").reindex(index).ffill().astype(float)
    return price.pct_change(fill_method=None).shift(-1).rename("KOSPI200")


def write_outputs(result: dict[str, object]) -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = result["returns"]
    equity = result["equity"]
    drawdown = result["drawdown"]
    summary = result["summary"]

    assert isinstance(returns, pd.DataFrame)
    assert isinstance(equity, pd.DataFrame)
    assert isinstance(drawdown, pd.DataFrame)
    assert isinstance(summary, pd.DataFrame)

    paths = {
        "daily_returns": OUTPUT_DIR / "daily_returns.csv",
        "equity": OUTPUT_DIR / "equity.csv",
        "drawdown": OUTPUT_DIR / "drawdown.csv",
        "summary": OUTPUT_DIR / "summary.csv",
        "config": OUTPUT_DIR / "config.json",
        "q5_equity_subplots": OUTPUT_DIR / "q5_equity_subplots.png",
        "q5_drawdown_subplots": OUTPUT_DIR / "q5_drawdown_subplots.png",
        "spread_equity_subplots": OUTPUT_DIR / "spread_equity_subplots.png",
        "spread_drawdown_subplots": OUTPUT_DIR / "spread_drawdown_subplots.png",
        "q5_metric_heatmaps": OUTPUT_DIR / "q5_metric_heatmaps.png",
        "spread_metric_heatmaps": OUTPUT_DIR / "spread_metric_heatmaps.png",
    }
    returns.to_csv(paths["daily_returns"], index_label="date")
    equity.to_csv(paths["equity"], index_label="date")
    drawdown.to_csv(paths["drawdown"], index_label="date")
    summary.to_csv(paths["summary"], index=False)
    paths["config"].write_text(json.dumps(result["config"], ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_factor_horizon_subplots(equity, paths["q5_equity_subplots"], leg="q5", title="Q5 cumulative return")
    _plot_factor_horizon_subplots(drawdown, paths["q5_drawdown_subplots"], leg="q5", title="Q5 drawdown")
    _plot_factor_horizon_subplots(equity, paths["spread_equity_subplots"], leg="q5_minus_q1", title="Q5-Q1 cumulative return")
    _plot_factor_horizon_subplots(drawdown, paths["spread_drawdown_subplots"], leg="q5_minus_q1", title="Q5-Q1 drawdown")
    _plot_metric_heatmaps(summary, paths["q5_metric_heatmaps"], leg="q5", title="Q5 metrics")
    _plot_metric_heatmaps(summary, paths["spread_metric_heatmaps"], leg="q5_minus_q1", title="Q5-Q1 metrics")
    return {key: str(path) for key, path in paths.items()}


def _plot_factor_horizon_subplots(frame: pd.DataFrame, path: Path, *, leg: str, title: str) -> None:
    colors = {
        "positivity": "#2563eb",
        "return_momentum": "#dc2626",
        "high_sharpe": "#16a34a",
        "high_low": "#9333ea",
    }
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    flat_axes = list(axes.flat)
    for ax, horizon in zip(flat_axes, HORIZONS, strict=False):
        for factor_name in FACTOR_ORDER:
            column = f"{factor_name}_{horizon['label']}_{leg}"
            if column in frame.columns:
                frame[column].plot(ax=ax, lw=1.35, color=colors[factor_name], label=factor_name)
        if leg == "q5" and "KOSPI200" in frame.columns:
            frame["KOSPI200"].plot(ax=ax, lw=1.0, color="#6b7280", alpha=0.7, label="KOSPI200")
        ax.axhline(1.0 if "cumulative" in title else 0.0, color="#333333", lw=0.5, alpha=0.35)
        ax.set_title(str(horizon["label"]), fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    for ax in flat_axes[len(HORIZONS) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=5 if leg == "q5" else 4, frameon=False)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_heatmaps(summary: pd.DataFrame, path: Path, *, leg: str, title: str) -> None:
    metrics = ("cagr", "mdd", "sharpe", "daily_win_rate")
    frame = summary.loc[summary["leg"].eq(leg)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, metric in zip(axes.flat, metrics, strict=True):
        pivot = frame.pivot(index="factor", columns="horizon", values=metric)
        pivot = pivot.reindex(index=list(FACTOR_ORDER), columns=[str(h["label"]) for h in HORIZONS])
        values = pivot.to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto", cmap="RdYlGn" if metric != "mdd" else "RdYlGn_r")
        ax.set_title(metric)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if pd.isna(value):
                    text = ""
                elif metric in {"cagr", "mdd", "daily_win_rate"}:
                    text = f"{value * 100:.1f}%"
                else:
                    text = f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="#111111")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
