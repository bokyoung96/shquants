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
from build_positivity_momentum_word_report import DocxBuilder, _num, _pct


OUTPUT_DIR = ROOT.results_path / "pos_research" / "momentum_reports"
START = pd.Timestamp("2020-01-01")
PERFORMANCE_END = pd.Timestamp("2026-05-29")
PORTFOLIO_DATE = pd.Timestamp("2026-05-29")
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
FACTOR_ORDER = ("positivity", "return_momentum", "high_sharpe", "near_high")
FACTOR_LABELS = {
    "positivity": "Positivity",
    "return_momentum": "Return momentum",
    "high_sharpe": "High Sharpe",
    "near_high": "Near high (P/High)",
}
COLORS = {
    "positivity": "#2563eb",
    "return_momentum": "#dc2626",
    "high_sharpe": "#16a34a",
    "near_high": "#9333ea",
}
QUINTILE_COLORS = {
    "q1": "#991b1b",
    "q2": "#f97316",
    "q3": "#6b7280",
    "q4": "#16a34a",
    "q5": "#2563eb",
}
LEG_LABELS = {"q5": "Q5 long-only", "q5_minus_q1": "Q5-Q1 spread"}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_comparison()
    written = write_outputs(result)
    report_path = write_word_report(result, written)
    written["word_report"] = str(report_path)
    print(json.dumps({"written": written, "top_q5": result["top_q5"]}, ensure_ascii=False, indent=2))


def run_comparison() -> dict[str, object]:
    store = ParquetStore(ROOT.parquet_path)
    close = store.read("qw_adj_c").astype(float)
    membership = store.read("qw_k200_yn").reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    benchmark = store.read("qw_BM")

    stock_returns = close.pct_change(fill_method=None)
    next_returns = stock_returns.shift(-1).where(membership)
    return_end_by_signal = _return_end_by_signal(close.index)
    eligible_signal_dates = return_end_by_signal.index[
        (return_end_by_signal.index >= START) & (return_end_by_signal <= PERFORMANCE_END)
    ]

    returns_by_name: dict[str, pd.Series] = {}
    positivity_quintiles_by_name: dict[str, pd.Series] = {}
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
            factor_returns = factor_returns.reindex(eligible_signal_dates).dropna(how="all")
            if factor_name == "positivity":
                quintile_returns = _rank_quintile_returns(signals[factor_name], next_returns)
                quintile_returns = quintile_returns.reindex(eligible_signal_dates).dropna(how="all")
                for quintile in ("q1", "q2", "q3", "q4", "q5"):
                    positivity_quintiles_by_name[f"positivity_{label}_{quintile}"] = quintile_returns[quintile]
            for bucket in ("q1", "q5"):
                series_name = f"{factor_name}_{label}_{bucket}"
                returns_by_name[series_name] = factor_returns[bucket]
                summary_rows.append(_summary_row(series_name, factor_returns[bucket], horizon, factor_name, bucket))
            spread = factor_returns["q5"].sub(factor_returns["q1"], fill_value=0.0)
            spread_name = f"{factor_name}_{label}_q5_minus_q1"
            returns_by_name[spread_name] = spread
            summary_rows.append(_summary_row(spread_name, spread, horizon, factor_name, "q5_minus_q1"))

    returns = pd.DataFrame(returns_by_name).sort_index()
    benchmark_returns = _next_day_benchmark_returns(benchmark=benchmark, index=close.index).reindex(returns.index)
    returns["KOSPI200"] = benchmark_returns
    returns = returns.dropna(how="all")
    positivity_quintile_returns = pd.DataFrame(positivity_quintiles_by_name).sort_index().dropna(how="all")

    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0)
    positivity_quintile_equity = (1.0 + positivity_quintile_returns.fillna(0.0)).cumprod()
    positivity_quintile_drawdown = positivity_quintile_equity.div(positivity_quintile_equity.cummax()).sub(1.0)
    summary = pd.DataFrame(summary_rows)
    latest_portfolio = _positivity_12m_portfolio(close=close, membership=membership, portfolio_date=PORTFOLIO_DATE)
    sector_summary = _sector_summary(latest_portfolio)
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
        "positivity_quintile_returns": positivity_quintile_returns,
        "positivity_quintile_equity": positivity_quintile_equity,
        "positivity_quintile_drawdown": positivity_quintile_drawdown,
        "summary": summary,
        "portfolio": latest_portfolio,
        "sector_summary": sector_summary,
        "top_q5": top_q5,
        "config": {
            "analysis": "KOSPI200 momentum horizon comparison with near_high replacing high_low",
            "start": START.date().isoformat(),
            "performance_end": PERFORMANCE_END.date().isoformat(),
            "portfolio_date": PORTFOLIO_DATE.date().isoformat(),
            "warmup": "signals use all available data before start; realized returns are limited so return end date is no later than performance_end",
            "trading_days_per_month": TRADING_DAYS_PER_MONTH,
            "quintile_method": "row-wise percentile rank with deterministic tie break, q5 rank > 0.8, q1 rank <= 0.2",
            "return_alignment": "signal date t holds close-to-close return from t to next trading day; last included signal has next close <= performance_end",
            "near_high_signal": "price at t-skip divided by rolling high over the lookback window excluding skip days when applicable",
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


def _return_end_by_signal(index: pd.Index) -> pd.Series:
    dates = pd.Index(index)
    return pd.Series(dates.to_series().shift(-1).to_numpy(), index=dates)


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
        "near_high": _near_high_score(shifted_close, window=window),
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


def _rank_quintile_returns(signal: pd.DataFrame, next_returns: pd.DataFrame) -> pd.DataFrame:
    common_index = signal.index.intersection(next_returns.index)
    common_columns = signal.columns.intersection(next_returns.columns)
    ranked = signal.loc[common_index, common_columns].rank(axis=1, method="first", pct=True)
    fwd = next_returns.loc[common_index, common_columns]
    buckets = {
        "q1": ranked.le(0.2),
        "q2": ranked.gt(0.2) & ranked.le(0.4),
        "q3": ranked.gt(0.4) & ranked.le(0.6),
        "q4": ranked.gt(0.6) & ranked.le(0.8),
        "q5": ranked.gt(0.8),
    }
    return pd.DataFrame({name: fwd.where(mask).mean(axis=1) for name, mask in buckets.items()}, index=common_index)


def _high_sharpe_score(returns: pd.DataFrame, *, window: int) -> pd.DataFrame:
    mean = returns.rolling(window, min_periods=window).mean()
    std = returns.rolling(window, min_periods=window).std(ddof=0)
    return mean.divide(std.where(std.gt(0.0)))


def _near_high_score(close: pd.DataFrame, *, window: int) -> pd.DataFrame:
    rolling_high = close.rolling(window, min_periods=window).max()
    return close.divide(rolling_high.where(rolling_high.gt(0.0)))


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


def _positivity_12m_portfolio(
    *,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    portfolio_date: pd.Timestamp,
) -> pd.DataFrame:
    names = pd.read_parquet(ROOT.parquet_path / "map__ticker_name_gics_sector_map.parquet")
    meta = names.set_index("TICKER")[["NAME", "GICS_SECTOR_NAME"]]
    returns = close.pct_change(fill_method=None)
    score = positivity_score(returns, lookback=252, min_periods=252).where(membership)
    ranks = score.rank(axis=1, method="first", pct=True)
    if portfolio_date not in score.index:
        raise ValueError(f"portfolio_date is not in close data: {portfolio_date.date().isoformat()}")
    q5_mask = ranks.loc[portfolio_date].gt(0.8) & score.loc[portfolio_date].notna() & membership.loc[portfolio_date]
    tickers = q5_mask.loc[q5_mask].index.tolist()
    weight = 1.0 / len(tickers)
    frame = pd.DataFrame(
        {
            "date": portfolio_date.date().isoformat(),
            "ticker": tickers,
            "code": [ticker.replace("A", "", 1) for ticker in tickers],
            "name": [str(meta.loc[ticker, "NAME"]) if ticker in meta.index else ticker for ticker in tickers],
            "sector": [
                str(meta.loc[ticker, "GICS_SECTOR_NAME"]) if ticker in meta.index else "Unknown"
                for ticker in tickers
            ],
            "weight": weight,
            "positivity_12m": [float(score.loc[portfolio_date, ticker]) for ticker in tickers],
            "rank_pct": [float(ranks.loc[portfolio_date, ticker]) for ticker in tickers],
        }
    )
    return frame.sort_values(["sector", "name"]).reset_index(drop=True)


def _sector_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    return (
        portfolio.groupby("sector", as_index=False)
        .agg(count=("ticker", "count"), weight=("weight", "sum"))
        .sort_values(["weight", "count"], ascending=False)
        .reset_index(drop=True)
    )


def write_outputs(result: dict[str, object]) -> dict[str, str]:
    returns = result["returns"]
    equity = result["equity"]
    drawdown = result["drawdown"]
    positivity_quintile_returns = result["positivity_quintile_returns"]
    positivity_quintile_equity = result["positivity_quintile_equity"]
    positivity_quintile_drawdown = result["positivity_quintile_drawdown"]
    summary = result["summary"]
    portfolio = result["portfolio"]
    sector_summary = result["sector_summary"]
    assert isinstance(returns, pd.DataFrame)
    assert isinstance(equity, pd.DataFrame)
    assert isinstance(drawdown, pd.DataFrame)
    assert isinstance(positivity_quintile_returns, pd.DataFrame)
    assert isinstance(positivity_quintile_equity, pd.DataFrame)
    assert isinstance(positivity_quintile_drawdown, pd.DataFrame)
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(portfolio, pd.DataFrame)
    assert isinstance(sector_summary, pd.DataFrame)

    paths = {
        "daily_returns": OUTPUT_DIR / "daily_returns.csv",
        "equity": OUTPUT_DIR / "equity.csv",
        "drawdown": OUTPUT_DIR / "drawdown.csv",
        "positivity_quintile_returns": OUTPUT_DIR / "positivity_quintile_returns.csv",
        "positivity_quintile_equity": OUTPUT_DIR / "positivity_quintile_equity.csv",
        "positivity_quintile_drawdown": OUTPUT_DIR / "positivity_quintile_drawdown.csv",
        "summary": OUTPUT_DIR / "summary.csv",
        "config": OUTPUT_DIR / "config.json",
        "portfolio": OUTPUT_DIR / "portfolio_2026_05_29_12m_positivity_q5.csv",
        "sector_summary": OUTPUT_DIR / "portfolio_2026_05_29_sector_summary.csv",
        "q5_equity_subplots": OUTPUT_DIR / "q5_equity_subplots.png",
        "q5_drawdown_subplots": OUTPUT_DIR / "q5_drawdown_subplots.png",
        "spread_equity_subplots": OUTPUT_DIR / "spread_equity_subplots.png",
        "spread_drawdown_subplots": OUTPUT_DIR / "spread_drawdown_subplots.png",
        "q5_metric_heatmaps": OUTPUT_DIR / "q5_metric_heatmaps.png",
        "spread_metric_heatmaps": OUTPUT_DIR / "spread_metric_heatmaps.png",
        "positivity_quintile_equity_subplots": OUTPUT_DIR / "positivity_quintile_equity_subplots.png",
        "positivity_quintile_drawdown_subplots": OUTPUT_DIR / "positivity_quintile_drawdown_subplots.png",
        "q5_return_histograms": OUTPUT_DIR / "q5_return_histograms.png",
        "spread_return_histograms": OUTPUT_DIR / "spread_return_histograms.png",
    }
    returns.to_csv(paths["daily_returns"], index_label="date")
    equity.to_csv(paths["equity"], index_label="date")
    drawdown.to_csv(paths["drawdown"], index_label="date")
    positivity_quintile_returns.to_csv(paths["positivity_quintile_returns"], index_label="date")
    positivity_quintile_equity.to_csv(paths["positivity_quintile_equity"], index_label="date")
    positivity_quintile_drawdown.to_csv(paths["positivity_quintile_drawdown"], index_label="date")
    summary.to_csv(paths["summary"], index=False)
    portfolio.to_csv(paths["portfolio"], index=False, encoding="utf-8-sig")
    sector_summary.to_csv(paths["sector_summary"], index=False, encoding="utf-8-sig")
    paths["config"].write_text(json.dumps(result["config"], ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_factor_horizon_subplots(equity, paths["q5_equity_subplots"], leg="q5", title="Q5 cumulative return")
    _plot_factor_horizon_subplots(drawdown, paths["q5_drawdown_subplots"], leg="q5", title="Q5 drawdown")
    _plot_factor_horizon_subplots(equity, paths["spread_equity_subplots"], leg="q5_minus_q1", title="Q5-Q1 cumulative return")
    _plot_factor_horizon_subplots(drawdown, paths["spread_drawdown_subplots"], leg="q5_minus_q1", title="Q5-Q1 drawdown")
    _plot_metric_heatmaps(summary, paths["q5_metric_heatmaps"], leg="q5", title="Q5 metrics")
    _plot_metric_heatmaps(summary, paths["spread_metric_heatmaps"], leg="q5_minus_q1", title="Q5-Q1 metrics")
    _plot_positivity_quintile_subplots(
        positivity_quintile_equity,
        paths["positivity_quintile_equity_subplots"],
        title="Positivity Q1-Q5 cumulative return",
        y_ref=1.0,
    )
    _plot_positivity_quintile_subplots(
        positivity_quintile_drawdown,
        paths["positivity_quintile_drawdown_subplots"],
        title="Positivity Q1-Q5 drawdown",
        y_ref=0.0,
    )
    _plot_histogram_subplots(returns, paths["q5_return_histograms"], leg="q5", title="Q5 daily return histograms")
    _plot_histogram_subplots(
        returns,
        paths["spread_return_histograms"],
        leg="q5_minus_q1",
        title="Q5-Q1 daily return histograms",
    )
    return {key: str(path) for key, path in paths.items()}


def _plot_factor_horizon_subplots(frame: pd.DataFrame, path: Path, *, leg: str, title: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    flat_axes = list(axes.flat)
    for ax, horizon in zip(flat_axes, HORIZONS, strict=False):
        for factor_name in FACTOR_ORDER:
            column = f"{factor_name}_{horizon['label']}_{leg}"
            if column in frame.columns:
                frame[column].plot(ax=ax, lw=1.35, color=COLORS[factor_name], label=FACTOR_LABELS[factor_name])
        if leg == "q5" and "KOSPI200" in frame.columns:
            frame["KOSPI200"].plot(ax=ax, lw=1.0, color="#6b7280", alpha=0.7, label="KOSPI200")
        ax.axhline(1.0 if "cumulative" in title else 0.0, color="#333333", lw=0.5, alpha=0.35)
        ax.set_title(str(horizon["label"]), fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    for ax in flat_axes[len(HORIZONS) :]:
        ax.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=5 if leg == "q5" else 4, frameon=False)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_heatmaps(summary: pd.DataFrame, path: Path, *, leg: str, title: str) -> None:
    metrics = [
        ("cagr", "CAGR"),
        ("mdd", "MDD"),
        ("sharpe", "Sharpe"),
        ("daily_win_rate", "Win rate"),
    ]
    frame = summary.loc[summary["leg"].eq(leg)].copy()
    frame["factor"] = pd.Categorical(frame["factor"], FACTOR_ORDER, ordered=True)
    frame["horizon"] = pd.Categorical(
        frame["horizon"],
        [str(horizon["label"]) for horizon in HORIZONS],
        ordered=True,
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (metric, label) in zip(axes.flat, metrics, strict=True):
        pivot = frame.pivot(index="factor", columns="horizon", values=metric).reindex(index=FACTOR_ORDER)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        ax.set_title(label)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), [FACTOR_LABELS[str(item)] for item in pivot.index])
        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = pivot.iloc[row_idx, col_idx]
                if pd.isna(value):
                    text = ""
                elif metric == "sharpe":
                    text = f"{value:.2f}"
                else:
                    text = f"{value * 100:.1f}%"
                ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_histogram_subplots(daily_returns: pd.DataFrame, path: Path, *, leg: str, title: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes_list = list(axes.flat)
    for ax, horizon in zip(axes_list, HORIZONS, strict=False):
        for factor in FACTOR_ORDER:
            column = f"{factor}_{horizon['label']}_{leg}"
            if column not in daily_returns.columns:
                continue
            values = daily_returns[column].dropna() * 100.0
            ax.hist(
                values,
                bins=45,
                alpha=0.28,
                density=True,
                color=COLORS[factor],
                label=FACTOR_LABELS[factor],
            )
        ax.axvline(0.0, color="#111111", lw=0.8, alpha=0.55)
        ax.set_title(str(horizon["label"]))
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes_list[len(HORIZONS) :]:
        ax.axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, frameon=False)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_positivity_quintile_subplots(frame: pd.DataFrame, path: Path, *, title: str, y_ref: float) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    flat_axes = list(axes.flat)
    for ax, horizon in zip(flat_axes, HORIZONS, strict=False):
        for quintile in ("q1", "q2", "q3", "q4", "q5"):
            column = f"positivity_{horizon['label']}_{quintile}"
            if column in frame.columns:
                frame[column].plot(ax=ax, lw=1.25, color=QUINTILE_COLORS[quintile], label=quintile.upper())
        ax.axhline(y_ref, color="#333333", lw=0.5, alpha=0.35)
        ax.set_title(str(horizon["label"]), fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    for ax in flat_axes[len(HORIZONS) :]:
        ax.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=5, frameon=False)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_word_report(result: dict[str, object], written: dict[str, str]) -> Path:
    summary = result["summary"]
    portfolio = result["portfolio"]
    sector_summary = result["sector_summary"]
    config = result["config"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(portfolio, pd.DataFrame)
    assert isinstance(sector_summary, pd.DataFrame)
    assert isinstance(config, dict)

    report_path = OUTPUT_DIR / "momentum_near_high_word_report.docx"
    doc = DocxBuilder(report_path)
    doc.heading("Positivity Momentum 보고서: Near-high 비교", level=0)
    doc.paragraph(f"작성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.paragraph(
        "이번 버전은 기존 high-low channel을 현재가/신고가 방식의 near_high = price / rolling_high로 교체했다. "
        "성과 측정은 2026년 5월 말까지만 반영되도록 수익률 종료일을 2026-05-29 이하로 제한했고, 포트폴리오는 2026-05-29 신호 snapshot 기준이다."
    )
    doc.bullets(
        [
            "near_high는 고저가 범위 내 위치가 아니라 신고가 대비 현재 가격의 근접도를 직접 측정하므로 momentum 해석이 더 깔끔하다.",
            "spread는 Q5 일간수익률에서 Q1 일간수익률을 뺀 뒤 누적한 long-short factor return이다.",
            "비용, 세금, 슬리피지는 반영하지 않은 gross 기준이며, close-to-close 다음 거래일 수익률로 측정했다.",
        ]
    )

    doc.heading("1. 핵심 성과 요약", level=1)
    doc.paragraph("아래 표는 각 horizon에서 Sharpe 1위 팩터와 positivity 성과를 비교한 것이다.")
    doc.table(_winner_table(summary), widths=[900, 1200, 1900, 1000, 1000, 1000, 900, 1000, 1000, 1000])
    doc.heading("Q5 Long-only 성과", level=2)
    doc.table(_metrics_table(summary, "q5"), widths=[900, 1500, 1100, 1000, 1000, 950, 950])
    doc.heading("Q5-Q1 Spread 성과", level=2)
    doc.table(_metrics_table(summary, "q5_minus_q1"), widths=[900, 1500, 1100, 1000, 1000, 950, 950])

    doc.heading("2. Spread Return", level=1)
    doc.paragraph(
        "Spread return은 동일한 ranking 방식에서 상위 quintile을 long, 하위 quintile을 short한 상대성과다. "
        "시장 방향성보다 팩터의 종목 선별력을 보려면 이 그래프가 더 중요하다."
    )
    doc.image(Path(written["spread_equity_subplots"]), "그림 1. Horizon별 Q5-Q1 누적 spread return")
    doc.image(Path(written["spread_drawdown_subplots"]), "그림 2. Horizon별 Q5-Q1 spread drawdown")
    doc.image(Path(written["spread_metric_heatmaps"]), "그림 3. Q5-Q1 spread 성과 heatmap")

    doc.heading("3. Positivity Quintile Ladder", level=1)
    doc.paragraph(
        "아래 그림은 positivity 점수 기준 Q1부터 Q5까지의 누적수익률과 drawdown이다. "
        "Q1에서 Q5로 갈수록 성과가 순서대로 정렬되는지 확인하면, 단순한 top bucket 우연이 아니라 팩터의 단조성이 있는지를 볼 수 있다."
    )
    doc.image(Path(written["positivity_quintile_equity_subplots"]), "그림 4. Positivity Q1-Q5 누적수익률")
    doc.image(Path(written["positivity_quintile_drawdown_subplots"]), "그림 5. Positivity Q1-Q5 drawdown")

    doc.heading("4. Q5 Long-only 및 Return Distribution", level=1)
    doc.paragraph("기존 리포트와 같은 양식으로 Q5 long-only 성과, drawdown, metric heatmap, 일간수익률 분포를 정리했다.")
    doc.image(Path(written["q5_equity_subplots"]), "그림 6. Horizon별 Q5 누적수익률")
    doc.image(Path(written["q5_drawdown_subplots"]), "그림 7. Horizon별 Q5 drawdown")
    doc.image(Path(written["q5_metric_heatmaps"]), "그림 8. Q5 성과지표 heatmap")
    doc.image(Path(written["q5_return_histograms"]), "그림 9. Q5 daily return distribution")
    doc.image(Path(written["spread_return_histograms"]), "그림 10. Q5-Q1 daily spread return distribution")

    doc.heading("5. Near-high와 Positivity의 경제적 해석", level=1)
    doc.paragraph(
        "near_high는 가격이 자신의 lookback 신고가에 얼마나 가까운지를 본다. 기존 high-low channel은 최저가도 분모에 포함하기 때문에 "
        "range oscillator 성격이 섞였지만, near_high는 신고가 근접도라는 momentum 신호에 더 직접적이다."
    )
    doc.paragraph(
        "그럼에도 positivity는 다른 차원의 정보를 본다. near_high가 '고점에서 얼마나 덜 빠졌는가'라면, positivity는 '얼마나 많은 날에 하락하지 않았는가'다. "
        "즉 near_high는 위치 신호이고, positivity는 경로 신호다."
    )
    doc.paragraph(
        "경제적으로 positivity가 강한 경우는 큰 이벤트 한 번으로 만들어진 momentum보다 여러 거래일에 걸친 점진적 재평가를 반영할 가능성이 높다. "
        "이런 종목은 단기 투기 수요나 급등 후 반락보다 펀더멘털 개선, 낮은 관심도, 정보 확산 지연과 더 잘 연결된다."
    )
    doc.paragraph(
        "따라서 near_high는 신고가 momentum의 더 순수한 대체 팩터로 유용하고, positivity는 그중에서도 상승 경로의 안정성과 정보 확산의 지속성을 포착하는 보완 신호로 해석하는 편이 타당하다."
    )

    doc.heading("6. 2026-05-29 기준 12M Positivity Q5 구성종목", level=1)
    doc.paragraph("아래 구성은 2026-05-29 기준 12M positivity Q5 포트폴리오이며, 종목별 섹터를 함께 표기했다.")
    doc.heading("섹터 비중", level=2)
    doc.table(_sector_table(sector_summary), widths=[2600, 1200, 1400])
    doc.heading("구성종목", level=2)
    doc.table(_portfolio_table(portfolio), widths=[900, 1700, 2400, 900, 1200, 1100])

    doc.heading("7. 재현 설정과 참고문헌", level=1)
    doc.table(
        [
            ["항목", "값"],
            ["테스트 시작", str(config["start"])],
            ["성과 종료일", str(config["performance_end"])],
            ["포트폴리오 기준일", str(config["portfolio_date"])],
            ["near_high 정의", str(config["near_high_signal"])],
            ["수익률 정렬", str(config["return_alignment"])],
        ],
        widths=[1800, 7200],
    )
    doc.table(_sources_table(), widths=[900, 2500, 5200])
    doc.save()
    return report_path


def _winner_table(summary: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = [[
        "Leg",
        "Horizon",
        "Sharpe 1위",
        "Best CAGR",
        "Best MDD",
        "Best Sharpe",
        "Pos Rank",
        "Pos CAGR",
        "Pos MDD",
        "Pos Sharpe",
    ]]
    for leg in ["q5", "q5_minus_q1"]:
        for horizon in [str(item["label"]) for item in HORIZONS]:
            sub = summary.loc[(summary["leg"].eq(leg)) & (summary["horizon"].eq(horizon))].copy()
            sub = sub.sort_values(["sharpe", "cagr"], ascending=False).reset_index(drop=True)
            best = sub.iloc[0]
            pos_idx = int(sub.index[sub["factor"].eq("positivity")][0])
            pos = sub.loc[pos_idx]
            rows.append(
                [
                    LEG_LABELS[leg],
                    horizon,
                    FACTOR_LABELS[str(best["factor"])],
                    _pct(best["cagr"]),
                    _pct(best["mdd"]),
                    _num(best["sharpe"]),
                    pos_idx + 1,
                    _pct(pos["cagr"]),
                    _pct(pos["mdd"]),
                    _num(pos["sharpe"]),
                ]
            )
    return rows


def _metrics_table(summary: pd.DataFrame, leg: str) -> list[list[object]]:
    rows: list[list[object]] = [["Horizon", "Factor", "CAGR", "MDD", "Sharpe", "Win rate", "Total return"]]
    frame = summary.loc[summary["leg"].eq(leg)].copy()
    frame["horizon"] = pd.Categorical(frame["horizon"], [str(item["label"]) for item in HORIZONS], ordered=True)
    frame["factor"] = pd.Categorical(frame["factor"], FACTOR_ORDER, ordered=True)
    frame = frame.sort_values(["horizon", "factor"])
    for _, row in frame.iterrows():
        rows.append(
            [
                row["horizon"],
                FACTOR_LABELS[str(row["factor"])],
                _pct(row["cagr"]),
                _pct(row["mdd"]),
                _num(row["sharpe"]),
                _pct(row["daily_win_rate"]),
                _pct(row["total_return"]),
            ]
        )
    return rows


def _sector_table(sector_summary: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = [["Sector", "종목 수", "비중"]]
    for _, row in sector_summary.iterrows():
        rows.append([row["sector"], int(row["count"]), _pct(row["weight"])])
    return rows


def _portfolio_table(portfolio: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = [["Code", "Name", "Sector", "Weight", "Positivity", "Rank pct"]]
    for _, row in portfolio.iterrows():
        rows.append(
            [
                row["code"],
                row["name"],
                row["sector"],
                _pct(row["weight"]),
                _pct(row["positivity_12m"]),
                _pct(row["rank_pct"]),
            ]
        )
    return rows


def _sources_table() -> list[list[object]]:
    return [
        ["ID", "Source", "보고서 활용"],
        [
            "S1",
            "Chen, Jiang, Liu, Zhu (2026), Positivity and long-lasting momentum, Journal of Empirical Finance. https://ideas.repec.org/a/eee/empfin/v87y2026ics0927539826000095.html",
            "Positivity 정의, 장기 예측력, high positivity winner의 value/펀더멘털 해석.",
        ],
        [
            "S2",
            "Da, Gurun, Warachka (2014), Frog in the Pan: Continuous Information and Momentum, Review of Financial Studies. https://ideas.repec.org/a/oup/rfinst/v27y2014i7p2171-2218..html",
            "점진적 정보 유입과 투자자 부주의가 더 지속적인 momentum을 만든다는 행동재무 논거.",
        ],
        [
            "S3",
            "Papailias, Liu, Thomakos (2021), Return Signal Momentum, Journal of Banking & Finance. https://ideas.repec.org/a/eee/jbfina/v124y2021ics0378426621000212.html",
            "수익률 부호 기반 momentum이 기존 time-series momentum 대비 Sharpe와 drawdown에서 강할 수 있다는 배경.",
        ],
        [
            "S4",
            "Jegadeesh and Titman (1993), Returns to Buying Winners and Selling Losers. https://ideas.repec.org/a/bla/jfinan/v48y1993i1p65-91.html",
            "전통적 3-12개월 momentum의 기준선.",
        ],
        [
            "S5",
            "George and Hwang (2004), The 52-week high and momentum investing. https://doi.org/10.1111/j.1540-6261.2004.00695.x",
            "신고가 근접도 기반 momentum 해석의 선행 연구.",
        ],
    ]


if __name__ == "__main__":
    main()
