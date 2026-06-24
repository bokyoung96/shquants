from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from root import ROOT

from backtesting.data import ParquetStore
from backtesting.strategies.positivity import (
    positivity_score,
)
from scripts.run_pos_research import (
    DEFAULT_START,
    _next_day_benchmark_returns,
    _weighted_next_day_returns,
    build_positivity_quintile_returns,
    summarize_quintile_returns,
)


RESULT_DIR = ROOT.results_path / "pos_research" / "strategy_grid"


@dataclass(frozen=True, slots=True)
class StrategySpec:
    name: str
    family: str
    params: dict[str, Any]


def _overlay_params(data_overlay: str) -> dict[str, Any]:
    return {
        "data_overlay": data_overlay,
        "overlay_lookback": 20,
        "overlay_group_count": 1 if data_overlay == "price_only" else 2,
    }


def build_strategy_specs() -> list[StrategySpec]:
    specs: list[StrategySpec] = []

    for max_positions in (3, 5, 8, 10, 15):
        for sector_cap in (1, 2):
            for data_overlay, suffix in (
                ("price_only", "px"),
                ("sponsorship", "flow"),
                ("retail_contrarian", "retail"),
            ):
                specs.append(
                    StrategySpec(
                        name=f"new_high_{suffix}_n{max_positions}_sec{sector_cap}",
                        family="new_high",
                        params={
                            "max_positions": max_positions,
                            "max_positions_per_sector": sector_cap,
                            "positivity_lookback": 60,
                            "min_periods": 60,
                            "breakout_lookback": 252,
                            "stop_lookback": 20,
                            "relative_signal_groups": 3,
                            "breakout_basis": "absolute",
                            **_overlay_params(data_overlay),
                        },
                    )
                )

    for max_positions in (5, 10, 15):
        for sector_cap in (1, 2):
            for data_overlay, suffix in (("sponsorship", "flow"), ("op_revision", "oprev")):
                specs.append(
                    StrategySpec(
                        name=f"stable_{suffix}_n{max_positions}_sec{sector_cap}",
                        family="stable_sleeve",
                        params={
                            "max_positions": max_positions,
                            "max_positions_per_sector": sector_cap,
                            "short_lookback": 60,
                            "mid_lookback": 120,
                            "long_lookback": 120,
                            "min_periods": 60,
                            "entry_group_count": 3,
                            "hold_group_count": 2,
                            **_overlay_params(data_overlay),
                        },
                    )
                )

    for max_positions in (3, 5, 8, 10):
        for sector_cap in (1, 2):
            specs.append(
                StrategySpec(
                    name=f"pullback_epsrev_n{max_positions}_sec{sector_cap}",
                    family="pullback_reclaim",
                    params={
                        "max_positions": max_positions,
                        "max_positions_per_sector": sector_cap,
                        "positivity_lookback": 60,
                        "min_periods": 60,
                        "high_lookback": 252,
                        "reclaim_lookback": 20,
                        "pullback_low_lookback": 20,
                        "relative_signal_groups": 3,
                        **_overlay_params("eps_revision"),
                    },
                )
            )

    return specs


def rank_strategy_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["event_viable"] = ranked["trade_count"].gt(0).astype(int)
    if "validation_pass" not in ranked:
        ranked["validation_pass"] = 0
    late_cagr = ranked["late_cagr"] if "late_cagr" in ranked else ranked["cagr"]
    late_mdd = ranked["late_mdd"] if "late_mdd" in ranked else ranked["mdd"]
    ranked["robust_return"] = pd.concat([ranked["cagr"], late_cagr], axis=1).min(axis=1).fillna(0.0)
    ranked["worst_mdd"] = pd.concat([ranked["mdd"], late_mdd], axis=1).min(axis=1).fillna(0.0)
    drawdown_risk = ranked["worst_mdd"].abs().where(ranked["worst_mdd"].abs().gt(0.0))
    ranked["robust_score"] = ranked["robust_return"].div(drawdown_risk).replace([float("inf"), -float("inf")], pd.NA).fillna(0.0)
    ranked["selection_score"] = ranked["robust_score"]
    return ranked.sort_values(
        ["event_viable", "validation_pass", "robust_score", "robust_return", "worst_mdd", "sharpe"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def _sector_percentile_rank(
    *,
    values: pd.DataFrame,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    ranks = pd.DataFrame(float("nan"), index=values.index, columns=values.columns)
    sectors = sector.reindex(index=values.index, columns=values.columns)
    members = membership.reindex(index=values.index, columns=values.columns).fillna(False).astype(bool)
    for sector_name in pd.unique(sectors.to_numpy().ravel()):
        if pd.isna(sector_name):
            continue
        sector_members = sectors.eq(sector_name) & members
        sector_values = values.where(sector_members)
        ranks = ranks.where(~sector_members, sector_values.rank(axis=1, pct=True))
    return ranks


def _build_data_overlay_ranks(
    *,
    data: dict[str, pd.DataFrame],
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    lookback: int,
) -> dict[str, pd.DataFrame]:
    def align(stem: str) -> pd.DataFrame:
        return data[stem].reindex(index=membership.index, columns=membership.columns).ffill()

    foreign = align("foreign").fillna(0.0)
    institution = align("institution").fillna(0.0)
    retail = align("retail").fillna(0.0)
    op = align("op")
    eps = align("eps")
    signals = {
        "price_only": pd.DataFrame(0.0, index=membership.index, columns=membership.columns),
        "sponsorship": foreign.add(institution, fill_value=0.0).rolling(window=lookback, min_periods=lookback).sum(),
        "retail_contrarian": retail.mul(-1.0).rolling(window=lookback, min_periods=lookback).sum(),
        "op_revision": op.diff(lookback),
        "eps_revision": eps.diff(lookback),
    }
    ranks = {
        name: _sector_percentile_rank(values=signal.where(membership), sector=sector, membership=membership)
        for name, signal in signals.items()
    }
    ranks["price_only"] = signals["price_only"]
    return ranks


def _overlay_entry_mask(overlay_rank: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    if params["data_overlay"] == "price_only":
        return pd.DataFrame(True, index=overlay_rank.index, columns=overlay_rank.columns)
    entry_cut = 1.0 - 1.0 / int(params["overlay_group_count"])
    return overlay_rank.gt(entry_cut)


def _overlay_score(overlay_rank: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    if params["data_overlay"] == "price_only":
        return pd.DataFrame(0.0, index=overlay_rank.index, columns=overlay_rank.columns)
    return overlay_rank.fillna(0.0)


def _equal_weight_from_mask(mask: pd.DataFrame) -> pd.DataFrame:
    counts = mask.sum(axis=1).astype(float)
    denominator = counts.where(counts.ne(0.0))
    return mask.astype(float).div(denominator, axis=0).fillna(0.0)


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _period_metrics(returns: pd.Series, prefix: str) -> dict[str, float | int]:
    clean = returns.dropna()
    if clean.empty:
        return {
            f"{prefix}_observations": 0,
            f"{prefix}_cagr": float("nan"),
            f"{prefix}_mdd": float("nan"),
            f"{prefix}_sharpe": float("nan"),
        }
    summary = summarize_quintile_returns(pd.DataFrame({"strategy": clean})).iloc[0]
    return {
        f"{prefix}_observations": int(summary["observations"]),
        f"{prefix}_cagr": float(summary["cagr"]),
        f"{prefix}_mdd": float(summary["mdd"]),
        f"{prefix}_sharpe": float(summary["sharpe"]),
    }


def _split_validation_metrics(returns: pd.Series) -> dict[str, float | int]:
    clean = returns.dropna()
    midpoint = len(clean) // 2
    early = clean.iloc[:midpoint]
    late = clean.iloc[midpoint:]
    metrics = {
        **_period_metrics(early, "early"),
        **_period_metrics(late, "late"),
    }
    late_cagr = metrics["late_cagr"]
    late_sharpe = metrics["late_sharpe"]
    metrics["validation_pass"] = int(pd.notna(late_cagr) and pd.notna(late_sharpe) and late_cagr > 0.0 and late_sharpe > 0.0)
    return metrics


def _pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    candidates = summary.loc[summary["event_viable"].eq(1) & summary["validation_pass"].eq(1)].copy()
    if candidates.empty:
        candidates = summary.copy()
    if "robust_score" not in candidates:
        drawdown_risk = candidates["worst_mdd"].abs().where(candidates["worst_mdd"].abs().gt(0.0))
        candidates["robust_score"] = candidates["robust_return"].div(drawdown_risk).fillna(0.0)
    keep: list[int] = []
    for idx, row in candidates.iterrows():
        dominated = (
            candidates["robust_return"].ge(row["robust_return"])
            & candidates["worst_mdd"].ge(row["worst_mdd"])
            & (
                candidates["robust_return"].gt(row["robust_return"])
                | candidates["worst_mdd"].gt(row["worst_mdd"])
            )
        ).any()
        if not dominated:
            keep.append(idx)
    return candidates.loc[keep].sort_values(
        ["robust_score", "robust_return", "worst_mdd"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _write_grid_outputs(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    start: pd.Timestamp,
    end_ts: pd.Timestamp,
    specs: list[StrategySpec],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "strategy_grid_summary.csv", index=False)
    summary.head(10).to_csv(output_dir / "top10_strategy_grid_summary.csv", index=False)
    pareto = _pareto_frontier(summary)
    pareto.to_csv(output_dir / "pareto_frontier.csv", index=False)
    selected = {key: _json_safe(value) for key, value in summary.iloc[0].to_dict().items()}
    non_price = summary.loc[summary["data_overlay"].ne("price_only")]
    selected_non_price = {key: _json_safe(value) for key, value in non_price.iloc[0].to_dict().items()}
    low_mdd_candidates = summary.loc[summary["event_viable"].eq(1) & summary["validation_pass"].eq(1)]
    if low_mdd_candidates.empty:
        low_mdd_candidates = summary.loc[summary["event_viable"].eq(1)]
    selected_low_mdd_row = low_mdd_candidates.sort_values(
        ["worst_mdd", "robust_return", "robust_score"],
        ascending=[False, False, False],
    ).iloc[0]
    selected_low_mdd = {key: _json_safe(value) for key, value in selected_low_mdd_row.to_dict().items()}
    final_manager_candidates = pareto.loc[pareto["data_overlay"].ne("price_only")]
    if final_manager_candidates.empty:
        final_manager_candidates = non_price
    if "family" in final_manager_candidates:
        event_driven_candidates = final_manager_candidates.loc[
            final_manager_candidates["family"].isin(["new_high", "pullback_reclaim"])
        ]
        if not event_driven_candidates.empty:
            final_manager_candidates = event_driven_candidates
    selected_final_manager = {
        key: _json_safe(value)
        for key, value in final_manager_candidates.sort_values(
            ["robust_score", "robust_return", "worst_mdd"],
            ascending=[False, False, False],
        ).iloc[0].to_dict().items()
    }
    (output_dir / "selected_strategy.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    (output_dir / "selected_non_price_strategy.json").write_text(
        json.dumps(selected_non_price, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    (output_dir / "selected_low_mdd_strategy.json").write_text(
        json.dumps(selected_low_mdd, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    (output_dir / "selected_final_manager_strategy.json").write_text(
        json.dumps(selected_final_manager, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    (output_dir / "strategy_grid_config.json").write_text(
        json.dumps(
            {
                "analysis": "positivity 50 strategy structural grid",
                "start": str(pd.Timestamp(start).date()),
                "end": str(pd.Timestamp(end_ts).date()),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "spec_count": len(specs),
                "specs": [asdict(spec) for spec in specs],
                "overfit_guardrail": "No raw signal floors/cutoffs; grid varies structural lookbacks, cadence, breadth, and sector caps.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _select_symbols(
    *,
    ranked: pd.Series,
    held: pd.Series,
    sector_row: pd.Series,
    max_positions: int,
    sector_cap: int | None,
) -> list[object]:
    open_slots = max_positions - int(held.sum())
    if open_slots <= 0:
        return []
    selected: list[object] = []
    sector_counts = sector_row.loc[held.index[held]].value_counts(dropna=False).to_dict()
    for symbol in ranked.dropna().sort_values(ascending=False).index:
        sector_name = sector_row.loc[symbol]
        if sector_cap is not None:
            current_count = int(sector_counts.get(sector_name, 0))
            if current_count >= sector_cap:
                continue
            sector_counts[sector_name] = current_count + 1
        selected.append(symbol)
        if len(selected) >= open_slots:
            break
    return selected


def _simulate_stable_sleeve_fast(
    *,
    close: pd.DataFrame,
    sector: pd.DataFrame,
    rank_by_lookback: dict[int, pd.DataFrame],
    overlay_rank: pd.DataFrame,
    spec: StrategySpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = spec.params
    short_rank = rank_by_lookback[int(params["short_lookback"])]
    mid_rank = rank_by_lookback[int(params["mid_lookback"])]
    long_rank = rank_by_lookback[int(params["long_lookback"])]
    entry_cut = 1.0 - 1.0 / int(params["entry_group_count"])
    hold_cut = 1.0 - 1.0 / int(params["hold_group_count"])
    overlay_ok = _overlay_entry_mask(overlay_rank, params)
    entry_ok = short_rank.gt(entry_cut) & mid_rank.gt(entry_cut) & long_rank.gt(hold_cut) & overlay_ok
    hold_ok = short_rank.gt(hold_cut) & mid_rank.gt(hold_cut) & overlay_ok
    composite = (
        short_rank.add(mid_rank, fill_value=0.0)
        .add(long_rank, fill_value=0.0)
        .add(_overlay_score(overlay_rank, params), fill_value=0.0)
    )
    rebalance_dates = set(close.groupby(close.index.to_period("M")).tail(1).index)

    held = pd.Series(False, index=close.columns)
    entry_price = pd.Series(float("nan"), index=close.columns, dtype=float)
    rows: list[pd.Series] = []
    trades: list[dict[str, Any]] = []
    for ts in close.index:
        if ts in rebalance_dates:
            current_price = close.loc[ts]
            exit_mask = held & ~hold_ok.loc[ts]
            for symbol in exit_mask.index[exit_mask]:
                trades.append(
                    {
                        "symbol": str(symbol),
                        "return": float(current_price.loc[symbol] / entry_price.loc[symbol] - 1.0),
                        "exit_reason": "rank_exit",
                    }
                )
            held = held & ~exit_mask
            eligible_score = composite.loc[ts].where(entry_ok.loc[ts] & ~held & current_price.notna())
            selected = _select_symbols(
                ranked=eligible_score,
                held=held,
                sector_row=sector.loc[ts],
                max_positions=int(params["max_positions"]),
                sector_cap=params["max_positions_per_sector"],
            )
            for symbol in selected:
                held.loc[symbol] = True
                entry_price.loc[symbol] = float(current_price.loc[symbol])
        rows.append(held.copy())
    return _equal_weight_from_mask(pd.DataFrame(rows, index=close.index, columns=close.columns)), pd.DataFrame(trades)


def _simulate_new_high_fast(
    *,
    close: pd.DataFrame,
    sector: pd.DataFrame,
    rank60: pd.DataFrame,
    overlay_rank: pd.DataFrame,
    spec: StrategySpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = spec.params
    entry_cut = 1.0 - 1.0 / int(params["relative_signal_groups"])
    prior_high = close.shift(1).rolling(
        window=int(params["breakout_lookback"]),
        min_periods=int(params["breakout_lookback"]),
    ).max()
    stop_line = close.shift(1).rolling(window=int(params["stop_lookback"]), min_periods=int(params["stop_lookback"])).min()
    overlay_score = _overlay_score(overlay_rank, params)
    entry_signal = rank60.gt(entry_cut) & close.gt(prior_high) & _overlay_entry_mask(overlay_rank, params)
    review_dates = set(close.groupby(close.index.to_period("W")).tail(1).index)

    held = pd.Series(False, index=close.columns)
    entry_price = pd.Series(float("nan"), index=close.columns, dtype=float)
    rows: list[pd.Series] = []
    trades: list[dict[str, Any]] = []
    for ts in close.index:
        current_price = close.loc[ts]
        stop_mask = held & current_price.lt(stop_line.loc[ts])
        for symbol in stop_mask.index[stop_mask]:
            trades.append(
                {
                    "symbol": str(symbol),
                    "return": float(current_price.loc[symbol] / entry_price.loc[symbol] - 1.0),
                    "exit_reason": "stop",
                }
            )
        held = held & ~stop_mask
        if ts in review_dates:
            eligible_score = (
                rank60.loc[ts]
                .add(close.loc[ts].div(prior_high.loc[ts]).rank(pct=True), fill_value=0.0)
                .add(overlay_score.loc[ts], fill_value=0.0)
            )
            eligible_score = eligible_score.where(entry_signal.loc[ts] & ~held & current_price.notna())
            selected = _select_symbols(
                ranked=eligible_score,
                held=held,
                sector_row=sector.loc[ts],
                max_positions=int(params["max_positions"]),
                sector_cap=params["max_positions_per_sector"],
            )
            for symbol in selected:
                held.loc[symbol] = True
                entry_price.loc[symbol] = float(current_price.loc[symbol])
        rows.append(held.copy())
    return _equal_weight_from_mask(pd.DataFrame(rows, index=close.index, columns=close.columns)), pd.DataFrame(trades)


def _simulate_pullback_fast(
    *,
    close: pd.DataFrame,
    sector: pd.DataFrame,
    rank60: pd.DataFrame,
    overlay_rank: pd.DataFrame,
    spec: StrategySpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = spec.params
    entry_cut = 1.0 - 1.0 / int(params["relative_signal_groups"])
    prior_high = close.shift(1).rolling(window=int(params["high_lookback"]), min_periods=int(params["high_lookback"])).max()
    reclaim_high = close.shift(1).rolling(
        window=int(params["reclaim_lookback"]),
        min_periods=int(params["reclaim_lookback"]),
    ).max()
    pullback_low = close.shift(1).rolling(
        window=int(params["pullback_low_lookback"]),
        min_periods=int(params["pullback_low_lookback"]),
    ).min()
    overlay_score = _overlay_score(overlay_rank, params)
    entry_signal = (
        rank60.gt(entry_cut)
        & close.shift(1).lt(prior_high)
        & close.gt(reclaim_high)
        & close.lt(prior_high)
        & _overlay_entry_mask(overlay_rank, params)
    )

    held = pd.Series(False, index=close.columns)
    entry_price = pd.Series(float("nan"), index=close.columns, dtype=float)
    rows: list[pd.Series] = []
    trades: list[dict[str, Any]] = []
    for ts in close.index:
        current_price = close.loc[ts]
        stop_mask = held & current_price.lt(pullback_low.loc[ts])
        for symbol in stop_mask.index[stop_mask]:
            trades.append(
                {
                    "symbol": str(symbol),
                    "return": float(current_price.loc[symbol] / entry_price.loc[symbol] - 1.0),
                    "exit_reason": "pullback_low",
                }
            )
        held = held & ~stop_mask
        eligible_score = (
            rank60.loc[ts]
            .add(close.loc[ts].div(reclaim_high.loc[ts]).rank(pct=True), fill_value=0.0)
            .add(overlay_score.loc[ts], fill_value=0.0)
        )
        eligible_score = eligible_score.where(entry_signal.loc[ts] & ~held & current_price.notna())
        selected = _select_symbols(
            ranked=eligible_score,
            held=held,
            sector_row=sector.loc[ts],
            max_positions=int(params["max_positions"]),
            sector_cap=params["max_positions_per_sector"],
        )
        for symbol in selected:
            held.loc[symbol] = True
            entry_price.loc[symbol] = float(current_price.loc[symbol])
        rows.append(held.copy())
    return _equal_weight_from_mask(pd.DataFrame(rows, index=close.index, columns=close.columns)), pd.DataFrame(trades)


def run_strategy_grid(
    *,
    start: str = DEFAULT_START,
    end: str | None = None,
    output_dir: Path = RESULT_DIR,
) -> pd.DataFrame:
    store = ParquetStore(ROOT.parquet_path)
    close = store.read("qw_adj_c").astype(float)
    membership = store.read("qw_k200_yn").reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    benchmark = store.read("qw_BM")
    sector = store.read("qw_wi_sec_26_big")
    overlay_data = {
        "foreign": store.read("qw_foreign"),
        "institution": store.read("qw_institution"),
        "retail": store.read("qw_retail"),
        "op": store.read("qw_op_nfq1"),
        "eps": store.read("qw_eps_nfq1"),
    }

    end_ts = pd.Timestamp(end) if end is not None else close.index.max()
    close = close.loc[pd.Timestamp(start) : end_ts]
    membership = membership.loc[close.index]
    common_index = close.index.intersection(sector.index)
    common_columns = close.columns.intersection(sector.columns)
    close = close.loc[common_index, common_columns]
    membership = membership.reindex(index=common_index, columns=common_columns).fillna(False).astype(bool)
    sector = sector.reindex(index=common_index, columns=common_columns)
    stock_returns = close.pct_change(fill_method=None)
    score_by_lookback = {
        lookback: positivity_score(stock_returns, lookback=lookback, min_periods=60).where(membership)
        for lookback in (60, 120, 252)
    }
    rank_by_lookback = {
        lookback: _sector_percentile_rank(values=score, sector=sector, membership=membership)
        for lookback, score in score_by_lookback.items()
    }
    overlay_ranks = _build_data_overlay_ranks(
        data=overlay_data,
        sector=sector,
        membership=membership,
        lookback=20,
    )

    base_returns = build_positivity_quintile_returns(close=close, membership=membership, lookback=252, q=5)
    benchmark_returns = _next_day_benchmark_returns(benchmark=benchmark, index=close.index).reindex(base_returns.index)
    benchmark_frame = base_returns.assign(KOSPI200=benchmark_returns)

    specs = build_strategy_specs()
    rows: list[dict[str, Any]] = []
    for spec in specs:
        overlay_rank = overlay_ranks[str(spec.params["data_overlay"])]
        if spec.family == "stable_sleeve":
            weights, trades = _simulate_stable_sleeve_fast(
                close=close,
                sector=sector,
                rank_by_lookback=rank_by_lookback,
                overlay_rank=overlay_rank,
                spec=spec,
            )
        elif spec.family == "new_high":
            weights, trades = _simulate_new_high_fast(
                close=close,
                sector=sector,
                rank60=rank_by_lookback[60],
                overlay_rank=overlay_rank,
                spec=spec,
            )
        elif spec.family == "pullback_reclaim":
            weights, trades = _simulate_pullback_fast(
                close=close,
                sector=sector,
                rank60=rank_by_lookback[60],
                overlay_rank=overlay_rank,
                spec=spec,
            )
        else:
            raise ValueError(f"unknown strategy family: {spec.family}")

        strategy_returns = pd.DataFrame(
            {spec.name: _weighted_next_day_returns(weights=weights, stock_returns=stock_returns)}
        )
        aligned = benchmark_frame.join(strategy_returns, how="inner")
        perf = summarize_quintile_returns(aligned[[spec.name]]).iloc[0].to_dict()
        validation = _split_validation_metrics(aligned[spec.name])
        names = weights.reindex(aligned.index).fillna(0.0).gt(0.0).sum(axis=1)
        row = {
            "strategy": spec.name,
            "family": spec.family,
            "trade_count": int(len(trades)),
            "trade_win_rate": float(trades["return"].gt(0.0).mean()) if not trades.empty else 0.0,
            "avg_trade_return": float(trades["return"].mean()) if not trades.empty else 0.0,
            "median_trade_return": float(trades["return"].median()) if not trades.empty else 0.0,
            "avg_names": float(names.mean()) if not names.empty else 0.0,
            "max_names": int(names.max()) if not names.empty else 0,
            "active_day_ratio": float(names.gt(0).mean()) if not names.empty else 0.0,
            **spec.params,
            **validation,
        }
        for key, value in perf.items():
            if key != "portfolio":
                row[key] = value
        rows.append(row)

    summary = rank_strategy_summary(pd.DataFrame(rows))
    _write_grid_outputs(
        output_dir=output_dir,
        summary=summary,
        start=start,
        end_ts=end_ts,
        specs=specs,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run positivity structural strategy grid.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=str(RESULT_DIR))
    args = parser.parse_args()

    summary = run_strategy_grid(start=args.start, end=args.end, output_dir=Path(args.output))
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
