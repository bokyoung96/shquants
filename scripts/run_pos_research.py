from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from root import ROOT

from backtesting.analytics import quantile_returns, summarize_perf
from backtesting.data import ParquetStore
from backtesting.data.benchmarks import benchmark_price_series
from backtesting.strategies.positivity import (
    build_band_holding_weights,
    build_positivity_stable_sleeve_strategy,
    build_positivity_pullback_reclaim_strategy,
    build_positivity_new_high_long_only_strategy,
    build_positivity_buckets,
    build_pure_signal_tilt_strategy,
    build_reacceleration_entry_weights,
    build_sector_neutral_positivity_long_short_weights,
    build_sector_positivity_event_core_strategy,
    build_sector_positivity_breakout_strategy,
    build_signal_band_strategy,
    build_positivity_quintile_returns,
    build_positivity_quintile_weights,
    build_sponsorship_group_weights,
    positivity_score,
    return_momentum_score,
)


DEFAULT_START = "2020-01-01"
DEFAULT_LOOKBACK = 252
RESULT_DIR = ROOT.results_path / "pos_research"


@dataclass(frozen=True, slots=True)
class PosResearchResult:
    returns: pd.DataFrame
    equity: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    metadata: dict[str, object]
    sponsorship_returns: pd.DataFrame | None = None
    sponsorship_summary: pd.DataFrame | None = None
    reacceleration_weights: pd.DataFrame | None = None
    band_holding_returns: pd.DataFrame | None = None
    band_holding_summary: pd.DataFrame | None = None
    band_holding_weights: pd.DataFrame | None = None
    signal_band_returns: pd.DataFrame | None = None
    signal_band_summary: pd.DataFrame | None = None
    signal_band_weights: pd.DataFrame | None = None
    signal_band_trades: pd.DataFrame | None = None
    pure_tilt_returns: pd.DataFrame | None = None
    pure_tilt_summary: pd.DataFrame | None = None
    pure_tilt_weights: pd.DataFrame | None = None
    pure_tilt_trades: pd.DataFrame | None = None
    sector_breakout_returns: pd.DataFrame | None = None
    sector_breakout_summary: pd.DataFrame | None = None
    sector_breakout_weights: pd.DataFrame | None = None
    sector_breakout_trades: pd.DataFrame | None = None
    sector_breakout_sector_state: pd.DataFrame | None = None
    sector_breakout_market_state: pd.DataFrame | None = None
    sector_breakout_entry_candidates: pd.DataFrame | None = None
    sector_event_returns: pd.DataFrame | None = None
    sector_event_summary: pd.DataFrame | None = None
    sector_event_weights: pd.DataFrame | None = None
    sector_event_trades: pd.DataFrame | None = None
    sector_event_sector_state: pd.DataFrame | None = None
    sector_event_market_state: pd.DataFrame | None = None
    sector_event_entry_candidates: pd.DataFrame | None = None
    sector_neutral_returns: pd.DataFrame | None = None
    sector_neutral_summary: pd.DataFrame | None = None
    sector_neutral_weights: pd.DataFrame | None = None
    new_high_returns: pd.DataFrame | None = None
    new_high_summary: pd.DataFrame | None = None
    new_high_weights: pd.DataFrame | None = None
    new_high_trades: pd.DataFrame | None = None
    pullback_returns: pd.DataFrame | None = None
    pullback_summary: pd.DataFrame | None = None
    pullback_weights: pd.DataFrame | None = None
    pullback_trades: pd.DataFrame | None = None
    stable_sleeve_returns: pd.DataFrame | None = None
    stable_sleeve_summary: pd.DataFrame | None = None
    stable_sleeve_weights: pd.DataFrame | None = None
    stable_sleeve_trades: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class MomentumComparisonResult:
    returns: pd.DataFrame
    equity: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, object]


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output)
    result = run_pos_research(
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        min_periods=args.min_periods,
    )
    written = write_outputs(output_dir, result)
    comparison = run_momentum_comparison(
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        min_periods=args.min_periods,
    )
    comparison_written = write_momentum_comparison_outputs(output_dir / "momentum_comparison", comparison)
    print(result.summary.to_string(index=False))
    print(comparison.summary.to_string(index=False))
    print(json.dumps({**written, "momentum_comparison": comparison_written}, ensure_ascii=False, indent=2))


def run_pos_research(
    *,
    start: str = DEFAULT_START,
    end: str | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    min_periods: int | None = None,
) -> PosResearchResult:
    store = ParquetStore(ROOT.parquet_path)
    close = store.read("qw_adj_c").astype(float)
    membership = store.read("qw_k200_yn").reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    benchmark = store.read("qw_BM")
    benchmark_weight = store.read("qw_bm_weights")
    sector = store.read("qw_wi_sec_26_big")
    foreign_flow = store.read("qw_foreign")
    institution_flow = store.read("qw_institution")
    retail_flow = store.read("qw_retail")
    eps = store.read("qw_eps_nfq1")
    op = store.read("qw_op_nfq1")

    end_ts = pd.Timestamp(end) if end is not None else close.index.max()
    close = close.loc[pd.Timestamp(start) : end_ts]
    membership = membership.loc[close.index]

    returns = build_positivity_quintile_returns(
        close=close,
        membership=membership,
        lookback=lookback,
        q=5,
        min_periods=min_periods,
    )
    benchmark_returns = _next_day_benchmark_returns(benchmark=benchmark, index=close.index).reindex(returns.index)
    if not benchmark_returns.empty:
        returns = returns.assign(KOSPI200=benchmark_returns)

    stock_returns = close.pct_change(fill_method=None)
    score = positivity_score(stock_returns, lookback=lookback, min_periods=min_periods).where(membership)
    weights = build_positivity_quintile_weights(score=score.reindex(returns.index), membership=membership.reindex(returns.index), q=5)
    buckets = build_positivity_buckets(score=score, membership=membership, q=5)
    (
        sponsorship_returns,
        sponsorship_summary,
        reacceleration_weights,
        band_holding_weights,
        signal_band_weights,
        signal_band_trades,
        pure_tilt_weights,
        pure_tilt_trades,
    ) = _build_sponsorship_research(
        buckets=buckets,
        close=close,
        score=score,
        stock_returns=stock_returns,
        foreign_flow=foreign_flow,
        institution_flow=institution_flow,
        retail_flow=retail_flow,
        eps=eps,
        op=op,
    )
    band_holding_returns = pd.DataFrame(
        {"band_holding": _weighted_next_day_returns(weights=band_holding_weights, stock_returns=stock_returns)}
    )
    band_holding_summary = summarize_quintile_returns(band_holding_returns)
    signal_band_returns = pd.DataFrame(
        {"signal_band_v1": _weighted_next_day_returns(weights=signal_band_weights, stock_returns=stock_returns)}
    )
    signal_band_summary = summarize_quintile_returns(signal_band_returns)
    pure_tilt_returns = pd.DataFrame(
        {"pure_tilt_v1": _weighted_next_day_returns(weights=pure_tilt_weights, stock_returns=stock_returns)}
    )
    pure_tilt_summary = summarize_quintile_returns(pure_tilt_returns)
    sector_breakout = _build_sector_breakout_research(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        stock_returns=stock_returns,
        foreign_flow=foreign_flow,
        institution_flow=institution_flow,
        retail_flow=retail_flow,
        eps=eps,
        op=op,
    )
    sector_event = _build_sector_event_research(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        stock_returns=stock_returns,
        foreign_flow=foreign_flow,
        institution_flow=institution_flow,
        retail_flow=retail_flow,
        eps=eps,
        op=op,
    )
    sector_neutral_weights = build_sector_neutral_positivity_long_short_weights(
        score=score,
        membership=membership,
        sector=sector,
        max_sectors=5,
        pairs_per_sector=1,
    )
    sector_neutral_returns = pd.DataFrame(
        {
            "sector_neutral_pos_ls": _weighted_next_day_returns(
                weights=sector_neutral_weights,
                stock_returns=stock_returns,
            )
        }
    )
    sector_neutral_summary = summarize_quintile_returns(sector_neutral_returns)
    new_high = build_positivity_new_high_long_only_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=5,
        max_positions_per_sector=1,
        positivity_lookback=60,
        min_periods=60,
        breakout_lookback=252,
        stop_lookback=20,
        relative_signal_groups=3,
        breakout_basis="absolute",
    )
    new_high_returns = pd.DataFrame(
        {"positivity_new_high_long_only": _weighted_next_day_returns(weights=new_high.weights, stock_returns=stock_returns)}
    )
    new_high_summary = summarize_quintile_returns(new_high_returns)
    pullback = build_positivity_pullback_reclaim_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=5,
        max_positions_per_sector=1,
        positivity_lookback=60,
        min_periods=60,
        high_lookback=252,
        reclaim_lookback=20,
        pullback_low_lookback=20,
        relative_signal_groups=3,
    )
    pullback_returns = pd.DataFrame(
        {"positivity_pullback_reclaim": _weighted_next_day_returns(weights=pullback.weights, stock_returns=stock_returns)}
    )
    pullback_summary = summarize_quintile_returns(pullback_returns)
    stable_sleeve = build_positivity_stable_sleeve_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=10,
        max_positions_per_sector=2,
        short_lookback=60,
        mid_lookback=120,
        long_lookback=252,
        min_periods=60,
        entry_group_count=3,
        hold_group_count=2,
    )
    stable_sleeve_returns = pd.DataFrame(
        {"positivity_stable_sleeve": _weighted_next_day_returns(weights=stable_sleeve.weights, stock_returns=stock_returns)}
    )
    stable_sleeve_summary = summarize_quintile_returns(stable_sleeve_returns)
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    summary = summarize_quintile_returns(returns)
    metadata: dict[str, object] = {
        "analysis": "positivity quintile momentum",
        "universe": "KOSPI200",
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end_ts).date()),
        "lookback": lookback,
        "min_periods": lookback if min_periods is None else min_periods,
        "signal": "percentage of trailing days with non-negative returns",
        "return_alignment": "signal date t uses close-to-close return from t to next trading day",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return PosResearchResult(
        returns=returns,
        equity=equity,
        weights=weights,
        summary=summary,
        metadata=metadata,
        sponsorship_returns=sponsorship_returns,
        sponsorship_summary=sponsorship_summary,
        reacceleration_weights=reacceleration_weights,
        band_holding_returns=band_holding_returns,
        band_holding_summary=band_holding_summary,
        band_holding_weights=band_holding_weights,
        signal_band_returns=signal_band_returns,
        signal_band_summary=signal_band_summary,
        signal_band_weights=signal_band_weights,
        signal_band_trades=signal_band_trades,
        pure_tilt_returns=pure_tilt_returns,
        pure_tilt_summary=pure_tilt_summary,
        pure_tilt_weights=pure_tilt_weights,
        pure_tilt_trades=pure_tilt_trades,
        sector_breakout_returns=sector_breakout["returns"],
        sector_breakout_summary=sector_breakout["summary"],
        sector_breakout_weights=sector_breakout["weights"],
        sector_breakout_trades=sector_breakout["trades"],
        sector_breakout_sector_state=sector_breakout["sector_state"],
        sector_breakout_market_state=sector_breakout["market_state"],
        sector_breakout_entry_candidates=sector_breakout["entry_candidates"],
        sector_event_returns=sector_event["returns"],
        sector_event_summary=sector_event["summary"],
        sector_event_weights=sector_event["weights"],
        sector_event_trades=sector_event["trades"],
        sector_event_sector_state=sector_event["sector_state"],
        sector_event_market_state=sector_event["market_state"],
        sector_event_entry_candidates=sector_event["entry_candidates"],
        sector_neutral_returns=sector_neutral_returns,
        sector_neutral_summary=sector_neutral_summary,
        sector_neutral_weights=sector_neutral_weights,
        new_high_returns=new_high_returns,
        new_high_summary=new_high_summary,
        new_high_weights=new_high.weights,
        new_high_trades=new_high.trades,
        pullback_returns=pullback_returns,
        pullback_summary=pullback_summary,
        pullback_weights=pullback.weights,
        pullback_trades=pullback.trades,
        stable_sleeve_returns=stable_sleeve_returns,
        stable_sleeve_summary=stable_sleeve_summary,
        stable_sleeve_weights=stable_sleeve.weights,
        stable_sleeve_trades=stable_sleeve.trades,
    )


def run_momentum_comparison(
    *,
    start: str = DEFAULT_START,
    end: str | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    min_periods: int | None = None,
) -> MomentumComparisonResult:
    store = ParquetStore(ROOT.parquet_path)
    close = store.read("qw_adj_c").astype(float)
    membership = store.read("qw_k200_yn").reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    benchmark = store.read("qw_BM")

    end_ts = pd.Timestamp(end) if end is not None else close.index.max()
    close = close.loc[pd.Timestamp(start) : end_ts]
    membership = membership.loc[close.index]
    stock_returns = close.pct_change(fill_method=None)

    positivity = positivity_score(stock_returns, lookback=lookback, min_periods=min_periods).where(membership)
    trailing_return = return_momentum_score(close, lookback=lookback).where(membership)
    next_returns = stock_returns.shift(-1).where(membership)

    positivity_returns = quantile_returns(positivity, next_returns, q=5).add_prefix("positivity_")
    return_momentum_returns = quantile_returns(trailing_return, next_returns, q=5).add_prefix("return_momentum_")
    returns = pd.concat([positivity_returns, return_momentum_returns], axis=1).dropna(how="all")
    benchmark_returns = _next_day_benchmark_returns(benchmark=benchmark, index=close.index).reindex(returns.index)
    if not benchmark_returns.empty:
        returns = returns.assign(KOSPI200=benchmark_returns)
    for prefix in ("positivity", "return_momentum"):
        q1 = f"{prefix}_q1"
        q5 = f"{prefix}_q5"
        if q1 in returns.columns and q5 in returns.columns:
            returns[f"{prefix}_q5_minus_q1"] = returns[q5].sub(returns[q1], fill_value=0.0)

    equity = (1.0 + returns.fillna(0.0)).cumprod()
    summary = summarize_quintile_returns(returns)
    metadata = {
        "analysis": "positivity momentum vs trailing return momentum",
        "universe": "KOSPI200",
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end_ts).date()),
        "lookback": lookback,
        "min_periods": lookback if min_periods is None else min_periods,
        "positivity_signal": "percentage of trailing days with non-negative returns",
        "return_momentum_signal": "trailing total return over lookback",
        "return_alignment": "signal date t uses close-to-close return from t to next trading day",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return MomentumComparisonResult(returns=returns, equity=equity, summary=summary, metadata=metadata)


def summarize_quintile_returns(returns: pd.DataFrame) -> pd.DataFrame:
    series_by_name = {column: returns[column].dropna() for column in returns.columns}
    if "q1" in returns.columns and "q5" in returns.columns:
        series_by_name["q5_minus_q1"] = returns["q5"].sub(returns["q1"], fill_value=0.0).dropna()

    rows: list[dict[str, object]] = []
    for name, series in series_by_name.items():
        perf = summarize_perf(series)
        equity = (1.0 + series.fillna(0.0)).cumprod()
        rows.append(
            {
                "portfolio": name,
                "observations": int(series.count()),
                "total_return": float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan"),
                "cagr": perf["cagr"],
                "mdd": perf["mdd"],
                "sharpe": perf["sharpe"],
                "daily_win_rate": float(series.gt(0.0).mean()) if not series.empty else float("nan"),
                "avg_daily_return": float(series.mean()) if not series.empty else float("nan"),
                "daily_vol": float(series.std(ddof=0)) if len(series) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(output_dir: Path, result: PosResearchResult) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    positions_dir = output_dir / "positions"
    plots_dir = output_dir / "plots"
    sponsorship_dir = output_dir / "sponsorship"
    band_holding_dir = output_dir / "band_holding"
    signal_band_dir = output_dir / "signal_band_strategy"
    pure_tilt_dir = output_dir / "pure_tilt_strategy"
    sector_breakout_dir = output_dir / "sector_positivity_breakout"
    sector_event_dir = output_dir / "sector_event_core"
    sector_neutral_dir = output_dir / "sector_neutral_long_short"
    new_high_dir = output_dir / "positivity_new_high_long_only"
    pullback_dir = output_dir / "positivity_pullback_reclaim"
    stable_sleeve_dir = output_dir / "positivity_stable_sleeve"
    positions_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    sponsorship_dir.mkdir(parents=True, exist_ok=True)
    band_holding_dir.mkdir(parents=True, exist_ok=True)
    signal_band_dir.mkdir(parents=True, exist_ok=True)
    pure_tilt_dir.mkdir(parents=True, exist_ok=True)
    sector_breakout_dir.mkdir(parents=True, exist_ok=True)
    sector_event_dir.mkdir(parents=True, exist_ok=True)
    sector_neutral_dir.mkdir(parents=True, exist_ok=True)
    new_high_dir.mkdir(parents=True, exist_ok=True)
    pullback_dir.mkdir(parents=True, exist_ok=True)
    stable_sleeve_dir.mkdir(parents=True, exist_ok=True)

    returns_path = output_dir / "daily_returns.csv"
    equity_path = output_dir / "equity.csv"
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    config_path = output_dir / "config.json"
    cumulative_plot_path = plots_dir / "cumulative_performance.png"
    summary_plot_path = plots_dir / "summary_metrics.png"
    sponsorship_returns_path = sponsorship_dir / "daily_returns.csv"
    sponsorship_summary_csv_path = sponsorship_dir / "summary.csv"
    sponsorship_summary_json_path = sponsorship_dir / "summary.json"
    sponsorship_plot_path = plots_dir / "sponsorship_groups.png"
    band_holding_returns_path = band_holding_dir / "daily_returns.csv"
    band_holding_summary_csv_path = band_holding_dir / "summary.csv"
    band_holding_summary_json_path = band_holding_dir / "summary.json"
    band_holding_plot_path = plots_dir / "band_holding.png"
    signal_band_returns_path = signal_band_dir / "daily_returns.csv"
    signal_band_summary_csv_path = signal_band_dir / "summary.csv"
    signal_band_summary_json_path = signal_band_dir / "summary.json"
    signal_band_trades_path = signal_band_dir / "trades.csv"
    signal_band_plot_path = plots_dir / "signal_band_v1.png"
    pure_tilt_returns_path = pure_tilt_dir / "daily_returns.csv"
    pure_tilt_summary_csv_path = pure_tilt_dir / "summary.csv"
    pure_tilt_summary_json_path = pure_tilt_dir / "summary.json"
    pure_tilt_trades_path = pure_tilt_dir / "trades.csv"
    pure_tilt_plot_path = plots_dir / "pure_tilt_v1.png"
    sector_breakout_returns_path = sector_breakout_dir / "daily_returns.csv"
    sector_breakout_summary_csv_path = sector_breakout_dir / "summary.csv"
    sector_breakout_summary_json_path = sector_breakout_dir / "summary.json"
    sector_breakout_trades_path = sector_breakout_dir / "trades.csv"
    sector_breakout_sector_state_path = sector_breakout_dir / "sector_state.csv"
    sector_breakout_market_state_path = sector_breakout_dir / "market_state.csv"
    sector_breakout_entry_candidates_path = sector_breakout_dir / "entry_candidates.csv"
    sector_breakout_plot_path = plots_dir / "sector_breakout_v1.png"
    sector_event_returns_path = sector_event_dir / "daily_returns.csv"
    sector_event_summary_csv_path = sector_event_dir / "summary.csv"
    sector_event_summary_json_path = sector_event_dir / "summary.json"
    sector_event_trades_path = sector_event_dir / "trades.csv"
    sector_event_sector_state_path = sector_event_dir / "sector_state.csv"
    sector_event_market_state_path = sector_event_dir / "market_state.csv"
    sector_event_entry_candidates_path = sector_event_dir / "entry_candidates.csv"
    sector_event_plot_path = plots_dir / "sector_event_core_v2.png"
    sector_neutral_returns_path = sector_neutral_dir / "daily_returns.csv"
    sector_neutral_summary_csv_path = sector_neutral_dir / "summary.csv"
    sector_neutral_summary_json_path = sector_neutral_dir / "summary.json"
    sector_neutral_plot_path = plots_dir / "sector_neutral_pos_ls.png"
    new_high_returns_path = new_high_dir / "daily_returns.csv"
    new_high_summary_csv_path = new_high_dir / "summary.csv"
    new_high_summary_json_path = new_high_dir / "summary.json"
    new_high_trades_path = new_high_dir / "trades.csv"
    new_high_plot_path = plots_dir / "positivity_new_high_long_only.png"
    pullback_returns_path = pullback_dir / "daily_returns.csv"
    pullback_summary_csv_path = pullback_dir / "summary.csv"
    pullback_summary_json_path = pullback_dir / "summary.json"
    pullback_trades_path = pullback_dir / "trades.csv"
    pullback_plot_path = plots_dir / "positivity_pullback_reclaim.png"
    stable_sleeve_returns_path = stable_sleeve_dir / "daily_returns.csv"
    stable_sleeve_summary_csv_path = stable_sleeve_dir / "summary.csv"
    stable_sleeve_summary_json_path = stable_sleeve_dir / "summary.json"
    stable_sleeve_trades_path = stable_sleeve_dir / "trades.csv"
    stable_sleeve_plot_path = plots_dir / "positivity_stable_sleeve.png"

    result.returns.to_csv(returns_path, index_label="date")
    result.equity.to_csv(equity_path, index_label="date")
    result.summary.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(result.summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_cumulative_performance(result.equity, cumulative_plot_path)
    _plot_summary_metrics(result.summary, summary_plot_path)
    if result.sponsorship_returns is not None and result.sponsorship_summary is not None:
        result.sponsorship_returns.to_csv(sponsorship_returns_path, index_label="date")
        result.sponsorship_summary.to_csv(sponsorship_summary_csv_path, index=False)
        sponsorship_summary_json_path.write_text(
            result.sponsorship_summary.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        _plot_cumulative_performance((1.0 + result.sponsorship_returns.fillna(0.0)).cumprod(), sponsorship_plot_path)
    if result.band_holding_returns is not None and result.band_holding_summary is not None:
        result.band_holding_returns.to_csv(band_holding_returns_path, index_label="date")
        result.band_holding_summary.to_csv(band_holding_summary_csv_path, index=False)
        band_holding_summary_json_path.write_text(
            result.band_holding_summary.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        _plot_cumulative_performance((1.0 + result.band_holding_returns.fillna(0.0)).cumprod(), band_holding_plot_path)
    if result.signal_band_returns is not None and result.signal_band_summary is not None:
        result.signal_band_returns.to_csv(signal_band_returns_path, index_label="date")
        result.signal_band_summary.to_csv(signal_band_summary_csv_path, index=False)
        signal_band_summary_json_path.write_text(
            result.signal_band_summary.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.signal_band_trades is not None:
            result.signal_band_trades.to_csv(signal_band_trades_path, index=False)
        _plot_cumulative_performance((1.0 + result.signal_band_returns.fillna(0.0)).cumprod(), signal_band_plot_path)
    if result.pure_tilt_returns is not None and result.pure_tilt_summary is not None:
        result.pure_tilt_returns.to_csv(pure_tilt_returns_path, index_label="date")
        result.pure_tilt_summary.to_csv(pure_tilt_summary_csv_path, index=False)
        pure_tilt_summary_json_path.write_text(
            result.pure_tilt_summary.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.pure_tilt_trades is not None:
            result.pure_tilt_trades.to_csv(pure_tilt_trades_path, index=False)
        _plot_cumulative_performance((1.0 + result.pure_tilt_returns.fillna(0.0)).cumprod(), pure_tilt_plot_path)
    if result.sector_breakout_returns is not None and result.sector_breakout_summary is not None:
        result.sector_breakout_returns.to_csv(sector_breakout_returns_path, index_label="date")
        result.sector_breakout_summary.to_csv(sector_breakout_summary_csv_path, index=False)
        sector_breakout_summary_json_path.write_text(
            result.sector_breakout_summary.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.sector_breakout_trades is not None:
            result.sector_breakout_trades.to_csv(sector_breakout_trades_path, index=False)
        if result.sector_breakout_sector_state is not None:
            result.sector_breakout_sector_state.to_csv(sector_breakout_sector_state_path)
        if result.sector_breakout_market_state is not None:
            result.sector_breakout_market_state.to_csv(sector_breakout_market_state_path, index_label="date")
        if result.sector_breakout_entry_candidates is not None:
            result.sector_breakout_entry_candidates.to_csv(sector_breakout_entry_candidates_path, index=False)
        _plot_cumulative_performance(
            (1.0 + result.sector_breakout_returns.fillna(0.0)).cumprod(),
            sector_breakout_plot_path,
            title="Sector Positivity Breakout",
        )
    if result.sector_event_returns is not None and result.sector_event_summary is not None:
        result.sector_event_returns.to_csv(sector_event_returns_path, index_label="date")
        result.sector_event_summary.to_csv(sector_event_summary_csv_path, index=False)
        sector_event_summary_json_path.write_text(
            result.sector_event_summary.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.sector_event_trades is not None:
            result.sector_event_trades.to_csv(sector_event_trades_path, index=False)
        if result.sector_event_sector_state is not None:
            result.sector_event_sector_state.to_csv(sector_event_sector_state_path)
        if result.sector_event_market_state is not None:
            result.sector_event_market_state.to_csv(sector_event_market_state_path, index_label="date")
        if result.sector_event_entry_candidates is not None:
            result.sector_event_entry_candidates.to_csv(sector_event_entry_candidates_path, index=False)
        _plot_cumulative_performance(
            (1.0 + result.sector_event_returns.fillna(0.0)).cumprod(),
            sector_event_plot_path,
            title="Sector Positivity Event Core",
        )
    if result.sector_neutral_returns is not None and result.sector_neutral_summary is not None:
        result.sector_neutral_returns.to_csv(sector_neutral_returns_path, index_label="date")
        result.sector_neutral_summary.to_csv(sector_neutral_summary_csv_path, index=False)
        sector_neutral_summary_json_path.write_text(
            result.sector_neutral_summary.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        _plot_cumulative_performance(
            (1.0 + result.sector_neutral_returns.fillna(0.0)).cumprod(),
            sector_neutral_plot_path,
            title="Sector Neutral Positivity Long-Short",
        )
    if result.new_high_returns is not None and result.new_high_summary is not None:
        result.new_high_returns.to_csv(new_high_returns_path, index_label="date")
        result.new_high_summary.to_csv(new_high_summary_csv_path, index=False)
        new_high_summary_json_path.write_text(
            result.new_high_summary.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.new_high_trades is not None:
            result.new_high_trades.to_csv(new_high_trades_path, index=False)
        _plot_cumulative_performance(
            (1.0 + result.new_high_returns.fillna(0.0)).cumprod(),
            new_high_plot_path,
            title="Positivity New High Long Only",
        )
    if result.pullback_returns is not None and result.pullback_summary is not None:
        result.pullback_returns.to_csv(pullback_returns_path, index_label="date")
        result.pullback_summary.to_csv(pullback_summary_csv_path, index=False)
        pullback_summary_json_path.write_text(
            result.pullback_summary.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.pullback_trades is not None:
            result.pullback_trades.to_csv(pullback_trades_path, index=False)
        _plot_cumulative_performance(
            (1.0 + result.pullback_returns.fillna(0.0)).cumprod(),
            pullback_plot_path,
            title="Positivity Pullback Reclaim",
        )
    if result.stable_sleeve_returns is not None and result.stable_sleeve_summary is not None:
        result.stable_sleeve_returns.to_csv(stable_sleeve_returns_path, index_label="date")
        result.stable_sleeve_summary.to_csv(stable_sleeve_summary_csv_path, index=False)
        stable_sleeve_summary_json_path.write_text(
            result.stable_sleeve_summary.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.stable_sleeve_trades is not None:
            result.stable_sleeve_trades.to_csv(stable_sleeve_trades_path, index=False)
        _plot_cumulative_performance(
            (1.0 + result.stable_sleeve_returns.fillna(0.0)).cumprod(),
            stable_sleeve_plot_path,
            title="Positivity Stable Sleeve",
        )

    for name, weights in result.weights.items():
        weights_path = positions_dir / f"weights_{name}.parquet"
        latest_path = positions_dir / f"latest_{name}.csv"
        weights.to_parquet(weights_path, engine="pyarrow")
        latest = _latest_weights(weights)
        latest.to_csv(latest_path, index=False)
    if result.reacceleration_weights is not None:
        reacceleration_weights_path = positions_dir / "weights_reacceleration.parquet"
        reacceleration_latest_path = positions_dir / "latest_reacceleration.csv"
        result.reacceleration_weights.to_parquet(reacceleration_weights_path, engine="pyarrow")
        _latest_weights(result.reacceleration_weights).to_csv(reacceleration_latest_path, index=False)
    if result.band_holding_weights is not None:
        band_holding_weights_path = positions_dir / "weights_band_holding.parquet"
        band_holding_latest_path = positions_dir / "latest_band_holding.csv"
        result.band_holding_weights.to_parquet(band_holding_weights_path, engine="pyarrow")
        _latest_weights(result.band_holding_weights).to_csv(band_holding_latest_path, index=False)
    if result.signal_band_weights is not None:
        signal_band_weights_path = positions_dir / "weights_signal_band_v1.parquet"
        signal_band_latest_path = positions_dir / "latest_signal_band_v1.csv"
        result.signal_band_weights.to_parquet(signal_band_weights_path, engine="pyarrow")
        _latest_weights(result.signal_band_weights).to_csv(signal_band_latest_path, index=False)
    if result.pure_tilt_weights is not None:
        pure_tilt_weights_path = positions_dir / "weights_pure_tilt_v1.parquet"
        pure_tilt_latest_path = positions_dir / "latest_pure_tilt_v1.csv"
        result.pure_tilt_weights.to_parquet(pure_tilt_weights_path, engine="pyarrow")
        _latest_weights(result.pure_tilt_weights).to_csv(pure_tilt_latest_path, index=False)
    if result.sector_breakout_weights is not None:
        sector_breakout_weights_path = positions_dir / "weights_sector_breakout_v1.parquet"
        sector_breakout_latest_path = positions_dir / "latest_sector_breakout_v1.csv"
        result.sector_breakout_weights.to_parquet(sector_breakout_weights_path, engine="pyarrow")
        _latest_weights(result.sector_breakout_weights).to_csv(sector_breakout_latest_path, index=False)
    if result.sector_event_weights is not None:
        sector_event_weights_path = positions_dir / "weights_sector_event_core_v2.parquet"
        sector_event_latest_path = positions_dir / "latest_sector_event_core_v2.csv"
        result.sector_event_weights.to_parquet(sector_event_weights_path, engine="pyarrow")
        _latest_weights(result.sector_event_weights).to_csv(sector_event_latest_path, index=False)
    if result.sector_neutral_weights is not None:
        sector_neutral_weights_path = positions_dir / "weights_sector_neutral_pos_ls.parquet"
        sector_neutral_latest_path = positions_dir / "latest_sector_neutral_pos_ls.csv"
        result.sector_neutral_weights.to_parquet(sector_neutral_weights_path, engine="pyarrow")
        _latest_weights(result.sector_neutral_weights).to_csv(sector_neutral_latest_path, index=False)
    if result.new_high_weights is not None:
        new_high_weights_path = positions_dir / "weights_positivity_new_high_long_only.parquet"
        new_high_latest_path = positions_dir / "latest_positivity_new_high_long_only.csv"
        result.new_high_weights.to_parquet(new_high_weights_path, engine="pyarrow")
        _latest_weights(result.new_high_weights).to_csv(new_high_latest_path, index=False)
    if result.pullback_weights is not None:
        pullback_weights_path = positions_dir / "weights_positivity_pullback_reclaim.parquet"
        pullback_latest_path = positions_dir / "latest_positivity_pullback_reclaim.csv"
        result.pullback_weights.to_parquet(pullback_weights_path, engine="pyarrow")
        _latest_weights(result.pullback_weights).to_csv(pullback_latest_path, index=False)
    if result.stable_sleeve_weights is not None:
        stable_sleeve_weights_path = positions_dir / "weights_positivity_stable_sleeve.parquet"
        stable_sleeve_latest_path = positions_dir / "latest_positivity_stable_sleeve.csv"
        result.stable_sleeve_weights.to_parquet(stable_sleeve_weights_path, engine="pyarrow")
        _latest_weights(result.stable_sleeve_weights).to_csv(stable_sleeve_latest_path, index=False)

    written = {
        "output_dir": str(output_dir),
        "returns": str(returns_path),
        "equity": str(equity_path),
        "summary_csv": str(summary_csv_path),
        "summary_json": str(summary_json_path),
        "config": str(config_path),
        "cumulative_plot": str(cumulative_plot_path),
        "summary_plot": str(summary_plot_path),
    }
    if result.sponsorship_returns is not None and result.sponsorship_summary is not None:
        written.update(
            {
                "sponsorship_returns": str(sponsorship_returns_path),
                "sponsorship_summary": str(sponsorship_summary_csv_path),
                "sponsorship_plot": str(sponsorship_plot_path),
            }
        )
    if result.band_holding_returns is not None and result.band_holding_summary is not None:
        written.update(
            {
                "band_holding_returns": str(band_holding_returns_path),
                "band_holding_summary": str(band_holding_summary_csv_path),
                "band_holding_plot": str(band_holding_plot_path),
            }
        )
    if result.signal_band_returns is not None and result.signal_band_summary is not None:
        written.update(
            {
                "signal_band_returns": str(signal_band_returns_path),
                "signal_band_summary": str(signal_band_summary_csv_path),
                "signal_band_trades": str(signal_band_trades_path),
                "signal_band_plot": str(signal_band_plot_path),
            }
        )
    if result.pure_tilt_returns is not None and result.pure_tilt_summary is not None:
        written.update(
            {
                "pure_tilt_returns": str(pure_tilt_returns_path),
                "pure_tilt_summary": str(pure_tilt_summary_csv_path),
                "pure_tilt_trades": str(pure_tilt_trades_path),
                "pure_tilt_plot": str(pure_tilt_plot_path),
            }
        )
    if result.sector_breakout_returns is not None and result.sector_breakout_summary is not None:
        written.update(
            {
                "sector_breakout_returns": str(sector_breakout_returns_path),
                "sector_breakout_summary": str(sector_breakout_summary_csv_path),
                "sector_breakout_trades": str(sector_breakout_trades_path),
                "sector_breakout_sector_state": str(sector_breakout_sector_state_path),
                "sector_breakout_market_state": str(sector_breakout_market_state_path),
                "sector_breakout_entry_candidates": str(sector_breakout_entry_candidates_path),
                "sector_breakout_plot": str(sector_breakout_plot_path),
            }
        )
    if result.sector_event_returns is not None and result.sector_event_summary is not None:
        written.update(
            {
                "sector_event_returns": str(sector_event_returns_path),
                "sector_event_summary": str(sector_event_summary_csv_path),
                "sector_event_trades": str(sector_event_trades_path),
                "sector_event_sector_state": str(sector_event_sector_state_path),
                "sector_event_market_state": str(sector_event_market_state_path),
                "sector_event_entry_candidates": str(sector_event_entry_candidates_path),
                "sector_event_plot": str(sector_event_plot_path),
            }
        )
    if result.sector_neutral_returns is not None and result.sector_neutral_summary is not None:
        written.update(
            {
                "sector_neutral_returns": str(sector_neutral_returns_path),
                "sector_neutral_summary": str(sector_neutral_summary_csv_path),
                "sector_neutral_plot": str(sector_neutral_plot_path),
            }
        )
    if result.new_high_returns is not None and result.new_high_summary is not None:
        written.update(
            {
                "new_high_returns": str(new_high_returns_path),
                "new_high_summary": str(new_high_summary_csv_path),
                "new_high_trades": str(new_high_trades_path),
                "new_high_plot": str(new_high_plot_path),
            }
        )
    if result.pullback_returns is not None and result.pullback_summary is not None:
        written.update(
            {
                "pullback_returns": str(pullback_returns_path),
                "pullback_summary": str(pullback_summary_csv_path),
                "pullback_trades": str(pullback_trades_path),
                "pullback_plot": str(pullback_plot_path),
            }
        )
    if result.stable_sleeve_returns is not None and result.stable_sleeve_summary is not None:
        written.update(
            {
                "stable_sleeve_returns": str(stable_sleeve_returns_path),
                "stable_sleeve_summary": str(stable_sleeve_summary_csv_path),
                "stable_sleeve_trades": str(stable_sleeve_trades_path),
                "stable_sleeve_plot": str(stable_sleeve_plot_path),
            }
        )
    return written


def write_momentum_comparison_outputs(output_dir: Path, result: MomentumComparisonResult) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    returns_path = output_dir / "daily_returns.csv"
    equity_path = output_dir / "equity.csv"
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    config_path = output_dir / "config.json"
    comparison_plot_path = output_dir / "comparison.png"

    result.returns.to_csv(returns_path, index_label="date")
    result.equity.to_csv(equity_path, index_label="date")
    result.summary.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(result.summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_columns = [
        column
        for column in (
            "positivity_q5",
            "return_momentum_q5",
            "positivity_q5_minus_q1",
            "return_momentum_q5_minus_q1",
            "KOSPI200",
        )
        if column in result.equity.columns
    ]
    _plot_cumulative_performance(
        result.equity.loc[:, plot_columns],
        comparison_plot_path,
        title="Positivity Momentum vs Return Momentum",
    )
    return {
        "output_dir": str(output_dir),
        "returns": str(returns_path),
        "equity": str(equity_path),
        "summary_csv": str(summary_csv_path),
        "summary_json": str(summary_json_path),
        "config": str(config_path),
        "comparison_plot": str(comparison_plot_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KOSPI200 positivity q1-q5 research portfolios.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--min-periods", type=int, default=None)
    parser.add_argument("--output", default=str(RESULT_DIR))
    return parser.parse_args()


def _next_day_benchmark_returns(*, benchmark: pd.DataFrame, index: pd.Index) -> pd.Series:
    price = benchmark_price_series(benchmark, "IKS200").reindex(index).ffill().astype(float)
    return price.pct_change(fill_method=None).shift(-1).rename("KOSPI200")


def _latest_weights(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=["symbol", "weight"])
    active = weights.astype(float).sum(axis=1).gt(0.0)
    if not bool(active.any()):
        return pd.DataFrame(columns=["symbol", "weight"])
    latest = weights.loc[active].iloc[-1].astype(float)
    latest = latest.loc[latest.gt(0.0)].sort_values(ascending=False)
    return pd.DataFrame({"symbol": latest.index.astype(str), "weight": latest.to_numpy(dtype=float)})


def _build_sector_breakout_research(
    *,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    benchmark_weight: pd.DataFrame,
    sector: pd.DataFrame,
    stock_returns: pd.DataFrame,
    foreign_flow: pd.DataFrame,
    institution_flow: pd.DataFrame,
    retail_flow: pd.DataFrame,
    eps: pd.DataFrame,
    op: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    common_index = (
        close.index.intersection(benchmark_weight.index)
        .intersection(sector.index)
        .intersection(foreign_flow.index)
        .intersection(institution_flow.index)
        .intersection(retail_flow.index)
        .intersection(eps.index)
        .intersection(op.index)
    )
    common_columns = (
        close.columns.intersection(benchmark_weight.columns)
        .intersection(sector.columns)
        .intersection(foreign_flow.columns)
        .intersection(institution_flow.columns)
        .intersection(retail_flow.columns)
        .intersection(eps.columns)
        .intersection(op.columns)
    )
    close = close.loc[common_index, common_columns]
    membership = membership.reindex(index=common_index, columns=common_columns).fillna(False).astype(bool)
    benchmark_weight = benchmark_weight.reindex(index=common_index, columns=common_columns).fillna(0.0)
    sector = sector.reindex(index=common_index, columns=common_columns)
    stock_returns = stock_returns.reindex(index=common_index, columns=common_columns)
    consensus_ok = _consensus_veto_mask(
        eps=eps.reindex(index=common_index, columns=common_columns),
        op=op.reindex(index=common_index, columns=common_columns),
        lookback=20,
    )
    result = build_sector_positivity_breakout_strategy(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        foreign_flow=foreign_flow.reindex(index=common_index, columns=common_columns),
        institution_flow=institution_flow.reindex(index=common_index, columns=common_columns),
        retail_flow=retail_flow.reindex(index=common_index, columns=common_columns),
        consensus_ok=consensus_ok,
        max_positions=10,
        positivity_lookback=60,
        min_periods=60,
        sector_slope_lookback=20,
        breakout_lookback=60,
        stop_lookback=60,
        flow_lookback=60,
        flow_long_lookback=120,
    )
    returns = pd.DataFrame(
        {"sector_breakout_v1": _weighted_next_day_returns(weights=result.weights, stock_returns=stock_returns)}
    )
    return {
        "returns": returns,
        "summary": summarize_quintile_returns(returns),
        "weights": result.weights.reindex(returns.index).fillna(0.0),
        "trades": result.trades,
        "sector_state": result.sector_state,
        "market_state": result.market_state,
        "entry_candidates": result.entry_candidates,
    }


def _build_sector_event_research(
    *,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    benchmark_weight: pd.DataFrame,
    sector: pd.DataFrame,
    stock_returns: pd.DataFrame,
    foreign_flow: pd.DataFrame,
    institution_flow: pd.DataFrame,
    retail_flow: pd.DataFrame,
    eps: pd.DataFrame,
    op: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    common_index = (
        close.index.intersection(benchmark_weight.index)
        .intersection(sector.index)
        .intersection(foreign_flow.index)
        .intersection(institution_flow.index)
        .intersection(retail_flow.index)
        .intersection(eps.index)
        .intersection(op.index)
    )
    common_columns = (
        close.columns.intersection(benchmark_weight.columns)
        .intersection(sector.columns)
        .intersection(foreign_flow.columns)
        .intersection(institution_flow.columns)
        .intersection(retail_flow.columns)
        .intersection(eps.columns)
        .intersection(op.columns)
    )
    close = close.loc[common_index, common_columns]
    membership = membership.reindex(index=common_index, columns=common_columns).fillna(False).astype(bool)
    benchmark_weight = benchmark_weight.reindex(index=common_index, columns=common_columns).fillna(0.0)
    sector = sector.reindex(index=common_index, columns=common_columns)
    stock_returns = stock_returns.reindex(index=common_index, columns=common_columns)
    consensus_ok = _consensus_veto_mask(
        eps=eps.reindex(index=common_index, columns=common_columns),
        op=op.reindex(index=common_index, columns=common_columns),
        lookback=20,
    )
    result = build_sector_positivity_event_core_strategy(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        foreign_flow=foreign_flow.reindex(index=common_index, columns=common_columns),
        institution_flow=institution_flow.reindex(index=common_index, columns=common_columns),
        retail_flow=retail_flow.reindex(index=common_index, columns=common_columns),
        consensus_ok=consensus_ok,
        max_positions=6,
        positivity_lookback=60,
        min_periods=60,
        sector_slope_lookback=20,
        breakout_lookback=60,
        stop_lookback=60,
        flow_lookback=60,
        flow_long_lookback=120,
        min_holding_days=40,
        trail_stop=False,
        market_entry_floor=None,
        leadership_market_floor=None,
        require_market_positive_slope=True,
        market_median_lookback=756,
        relative_signal_groups=3,
        sector_rank_groups=2,
        max_positions_per_sector=1,
    )
    returns = pd.DataFrame(
        {"sector_event_core_v2": _weighted_next_day_returns(weights=result.weights, stock_returns=stock_returns)}
    )
    return {
        "returns": returns,
        "summary": summarize_quintile_returns(returns),
        "weights": result.weights.reindex(returns.index).fillna(0.0),
        "trades": result.trades,
        "sector_state": result.sector_state,
        "market_state": result.market_state,
        "entry_candidates": result.entry_candidates,
    }


def _build_sponsorship_research(
    *,
    buckets: pd.DataFrame,
    close: pd.DataFrame,
    score: pd.DataFrame,
    stock_returns: pd.DataFrame,
    foreign_flow: pd.DataFrame,
    institution_flow: pd.DataFrame,
    retail_flow: pd.DataFrame,
    eps: pd.DataFrame,
    op: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_index = (
        buckets.index.intersection(foreign_flow.index)
        .intersection(institution_flow.index)
        .intersection(retail_flow.index)
        .intersection(eps.index)
        .intersection(op.index)
    )
    common_columns = (
        buckets.columns.intersection(foreign_flow.columns)
        .intersection(institution_flow.columns)
        .intersection(retail_flow.columns)
        .intersection(eps.columns)
        .intersection(op.columns)
    )
    buckets = buckets.loc[common_index, common_columns]
    close = close.reindex(index=common_index, columns=common_columns)
    score = score.reindex(index=common_index, columns=common_columns)
    stock_returns = stock_returns.reindex(index=common_index, columns=common_columns)
    q5_mask = buckets.eq(5)
    group_weights = build_sponsorship_group_weights(
        q5_mask=q5_mask,
        foreign_flow=foreign_flow.reindex(index=common_index, columns=common_columns),
        institution_flow=institution_flow.reindex(index=common_index, columns=common_columns),
        retail_flow=retail_flow.reindex(index=common_index, columns=common_columns),
    )
    sponsored = group_weights["foreign_persistent"].gt(0.0) | group_weights["institution_persistent"].gt(0.0)
    reacceleration_weights = build_reacceleration_entry_weights(
        buckets=buckets,
        sponsorship=sponsored,
        prior_lookback=63,
    )
    band_holding_weights = build_band_holding_weights(
        buckets=buckets,
        no_sponsor=group_weights["no_persistent_sponsorship"].gt(0.0),
        retail_supply=group_weights["retail_supply_absorption"].gt(0.0),
        dual_sponsorship=group_weights["dual_sponsorship"].gt(0.0),
        prior_lookback=63,
    )
    consensus_ok = _consensus_veto_mask(
        eps=eps.reindex(index=common_index, columns=common_columns),
        op=op.reindex(index=common_index, columns=common_columns),
        lookback=20,
    )
    signal_band = build_signal_band_strategy(
        buckets=buckets,
        close=close,
        no_sponsor=group_weights["no_persistent_sponsorship"].gt(0.0),
        retail_supply=group_weights["retail_supply_absorption"].gt(0.0),
        dual_sponsorship=group_weights["dual_sponsorship"].gt(0.0),
        consensus_ok=consensus_ok,
        prior_lookback=63,
        stop_lookback=20,
    )
    pure_tilt = build_pure_signal_tilt_strategy(
        buckets=buckets,
        close=close,
        signal_score=score,
        no_sponsor=group_weights["no_persistent_sponsorship"].gt(0.0),
        retail_supply=group_weights["retail_supply_absorption"].gt(0.0),
        dual_sponsorship=group_weights["dual_sponsorship"].gt(0.0),
        consensus_ok=consensus_ok,
        max_positions=5,
        prior_lookback=63,
        stop_lookback=20,
        breakout_lookback=20,
    )

    returns_by_group = {
        name: _weighted_next_day_returns(weights=weights, stock_returns=stock_returns)
        for name, weights in group_weights.items()
    }
    returns_by_group["reacceleration"] = _weighted_next_day_returns(
        weights=reacceleration_weights,
        stock_returns=stock_returns,
    )
    sponsorship_returns = pd.DataFrame(returns_by_group).dropna(how="all")
    sponsorship_summary = summarize_quintile_returns(sponsorship_returns)
    return (
        sponsorship_returns,
        sponsorship_summary,
        reacceleration_weights.reindex(sponsorship_returns.index).fillna(0.0),
        band_holding_weights.reindex(sponsorship_returns.index).fillna(0.0),
        signal_band.weights.reindex(sponsorship_returns.index).fillna(0.0),
        signal_band.trades,
        pure_tilt.weights.reindex(sponsorship_returns.index).fillna(0.0),
        pure_tilt.trades,
    )


def _consensus_veto_mask(*, eps: pd.DataFrame, op: pd.DataFrame, lookback: int) -> pd.DataFrame:
    eps_delta = eps.astype(float).sub(eps.astype(float).shift(lookback))
    op_delta = op.astype(float).sub(op.astype(float).shift(lookback))
    both_negative = eps_delta.lt(0.0) & op_delta.lt(0.0)
    return ~both_negative.fillna(False)


def _weighted_next_day_returns(*, weights: pd.DataFrame, stock_returns: pd.DataFrame) -> pd.Series:
    next_returns = stock_returns.shift(-1).reindex(index=weights.index, columns=weights.columns)
    active = weights.sum(axis=1).gt(0.0)
    valid_next_return = next_returns.notna().any(axis=1)
    out = weights.mul(next_returns).sum(axis=1, min_count=1).where(active, 0.0).where(valid_next_return)
    return out.dropna()


def _plot_cumulative_performance(
    equity: pd.DataFrame,
    path: Path,
    *,
    title: str = "Positivity Quintile Cumulative Performance",
) -> None:
    frame = equity.copy()
    ordered = _ordered_plot_columns(frame)
    if ordered:
        frame = frame.loc[:, ordered]

    fig, ax = plt.subplots(figsize=(12, 7))
    frame.plot(ax=ax, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1.0")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ordered_plot_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "positivity_q5",
        "return_momentum_q5",
        "positivity_q5_minus_q1",
        "return_momentum_q5_minus_q1",
        "band_holding",
        "signal_band_v1",
        "pure_tilt_v1",
        "sector_breakout_v1",
        "sector_event_core_v2",
        "KOSPI200",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    ordered.extend(column for column in frame.columns if column not in ordered)
    return ordered


def _plot_summary_metrics(summary: pd.DataFrame, path: Path) -> None:
    frame = summary.set_index("portfolio")
    metrics = [metric for metric in ("cagr", "sharpe", "mdd") if metric in frame.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics, strict=True):
        values = frame[metric].astype(float)
        values.plot(kind="bar", ax=ax, color="#3b6ea8" if metric != "mdd" else "#9b4d4d")
        ax.set_title(metric.upper())
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Positivity Portfolio Summary", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
