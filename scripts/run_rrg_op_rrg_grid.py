from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from root import ROOT

from backtesting.analytics import summarize_perf
from backtesting.catalog import DataCatalog, DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import DataLoader, LoadRequest, MarketData, ParquetStore
from backtesting.data.benchmarks import benchmark_price_series
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.execution import CostModel, WeeklySchedule
from backtesting.policy.base import PositionPlan
from backtesting.policy.pass_through import PassThroughPolicy
from backtesting.reporting import RunWriter
from backtesting.run import RunConfig, RunReport
from backtesting.signals.base import SignalBundle
from backtesting.strategies.rrg_sector_rotation import (
    _apply_rrg_transition_hysteresis,
    _build_rrg_context,
    _classify_rrg_states,
    _estimate_family_delta,
    _exclude_op_by_benchmark_weight,
    _map_sector_state_to_symbols,
    _map_sector_values_to_symbols,
    _sector_weighted_returns,
)


START = "2020-01-01"
END = "2026-05-11"
LOAD_START = "2019-01-01"
CAPITAL = 100_000_000.0
FEE = 0.0002
SELL_TAX = 0.0015
SLIPPAGE = 0.0005
GROSS_LONG = 1.0
GROSS_SHORT = 0.5
RESULT_DIR = ROOT.results_path / "rrg_research" / "op_rrg_grid"


@dataclass(frozen=True, slots=True)
class Variant:
    id: str
    op_rrg_mode: str
    stock_score: str
    compression: str
    confirm: str
    long_per_sector: int
    short_per_sector: int


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    market = _load_market()
    context = _build_context(market)
    variants = _variant_grid()
    writer = RunWriter(ROOT.results_path / "backtests", write_report_assets=False)
    engine = BacktestEngine(cost=CostModel(fee=FEE, sell_tax=SELL_TAX, slippage=SLIPPAGE))
    rows: list[dict[str, object]] = []

    for idx, variant in enumerate(variants, start=1):
        strategy_id = f"rrg_op_rrg_grid_{variant.id}"
        existing = _load_existing_metric(strategy_id=strategy_id, variant=variant, benchmark=context["benchmark_returns"])
        if existing is not None:
            print(f"[{idx}/{len(variants)}] skipping {strategy_id}")
            rows.append(existing)
            continue
        print(f"[{idx}/{len(variants)}] running {strategy_id}")
        plan = _build_plan(context=context, variant=variant)
        result = engine.run(
            close=context["close"],
            open=context["open"],
            weights=plan.target_weights,
            capital=CAPITAL,
            tradable=context["tradable"],
            exit_tradable=context["close"].notna(),
            schedule=WeeklySchedule(),
            fill_mode="next_open",
            allow_fractional=True,
        )
        result = _trim_result(result)
        plan = _trim_plan(plan)
        summary = summarize_perf(result.returns)
        summary["final_equity"] = float(result.equity.iloc[-1])
        summary["avg_turnover"] = float(result.turnover.mean())
        config = RunConfig(
            start=START,
            end=END,
            capital=CAPITAL,
            strategy=strategy_id,
            strategy_params={
                "op_rrg_mode": variant.op_rrg_mode,
                "stock_score": variant.stock_score,
                "compression": variant.compression,
                "confirm": variant.confirm,
                "long_per_sector": variant.long_per_sector,
                "short_per_sector": variant.short_per_sector,
                "gross_long": GROSS_LONG,
                "gross_short": GROSS_SHORT,
                "schedule": "weekly",
                "fill_mode": "next_open",
            },
            name=f"RRG OP RRG grid {variant.id}",
            schedule="weekly",
            fill_mode="next_open",
            fee=FEE,
            sell_tax=SELL_TAX,
            slippage=SLIPPAGE,
            borrow_fee_annual=0.0,
            short_cash_collateral_ratio=1.0,
            use_k200=True,
            warmup_days=365,
        )
        output_dir = writer.write(RunReport(config=config, summary=summary, result=result, position_plan=plan))
        rows.append(_metric_row(strategy_id=strategy_id, variant=variant, result=result, benchmark=context["benchmark_returns"], output_dir=output_dir))

    summary_frame = pd.DataFrame(rows).sort_values(["sharpe", "mdd", "cagr"], ascending=[False, False, False])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULT_DIR / f"grid_summary_{stamp}.csv"
    json_path = RESULT_DIR / f"grid_summary_{stamp}.json"
    summary_frame.to_csv(summary_path, index=False)
    json_path.write_text(summary_frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(summary_frame.head(15).to_string(index=False))
    print(json.dumps({"summary": str(summary_path), "json": str(json_path), "count": len(summary_frame)}, ensure_ascii=False, indent=2))


def _load_existing_metric(*, strategy_id: str, variant: Variant, benchmark: pd.Series) -> dict[str, object] | None:
    candidates: list[Path] = []
    for run_dir in ROOT.results_path.joinpath("backtests").glob("rrg-op-rrg-grid-*"):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if config.get("strategy") == strategy_id:
            candidates.append(run_dir)
    if not candidates:
        return None
    run_dir = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    returns = pd.read_csv(run_dir / "series" / "returns.csv", index_col="date", parse_dates=True)["returns"].astype(float)
    equity = pd.read_csv(run_dir / "series" / "equity.csv", index_col="date", parse_dates=True)["equity"].astype(float)
    turnover = pd.read_csv(run_dir / "series" / "turnover.csv", index_col="date", parse_dates=True)["turnover"].astype(float)
    weights = pd.read_parquet(run_dir / "positions" / "weights.parquet").fillna(0.0).astype(float)
    result = BacktestResult(
        equity=equity.loc[START:END],
        returns=returns.loc[START:END],
        weights=weights.loc[START:END],
        qty=pd.DataFrame(index=weights.loc[START:END].index),
        turnover=turnover.loc[START:END],
    )
    return _metric_row(strategy_id=strategy_id, variant=variant, result=result, benchmark=benchmark, output_dir=run_dir)


def _load_market() -> MarketData:
    loader = DataLoader(DataCatalog.default(), ParquetStore(ROOT.parquet_path))
    return loader.load(
        LoadRequest(
            datasets=[
                DatasetId.QW_ADJ_C,
                DatasetId.QW_ADJ_O,
                DatasetId.QW_BM,
                DatasetId.QW_BM_WEIGHTS,
                DatasetId.QW_K200_YN,
                DatasetId.QW_WI_SEC_26_BIG,
                DatasetId.QW_MKTCAP,
                DatasetId.QW_MKTCAP_FLT,
                DatasetId.QW_OP_NFQ1,
                DatasetId.QW_OP_NFQ2,
                DatasetId.QW_OP_NFY1,
                DatasetId.QW_OP_FWD_12M,
                DatasetId.QW_FOREIGN,
                DatasetId.QW_INSTITUTION,
                DatasetId.QW_RETAIL,
            ],
            start=LOAD_START,
            end=END,
        )
    )


def _build_context(market: MarketData) -> dict[str, object]:
    close = market.frames["close"].astype(float)
    k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
    active_columns = k200.columns[k200.any(axis=0)]
    close = close.loc[:, active_columns]
    open_ = market.frames["open"].reindex(index=close.index, columns=close.columns).astype(float)
    k200 = k200.loc[:, active_columns]
    sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
    market_cap = market.frames.get("float_market_cap", market.frames["market_cap"]).reindex(index=close.index, columns=close.columns).ffill().astype(float)
    benchmark = benchmark_price_series(market.frames["benchmark"], "IKS200").reindex(close.index).ffill().astype(float)
    benchmark_returns = benchmark.pct_change(fill_method=None).fillna(0.0).loc[START:END].rename("KOSPI200")

    price_state, _long, _short = _build_rrg_context(
        close=close,
        benchmark=benchmark,
        sector=sector,
        membership=k200,
        market_cap=market_cap,
        medium_lookback=126,
        momentum_lookback=21,
        short_lookback=42,
        transition_threshold=0.005,
    )
    op12 = market.frames["op_fwd_12m"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
    bm_weights = market.frames["bm_weights"].reindex(index=close.index, columns=close.columns).fillna(0.0).astype(float)
    op_states = {
        "base": _op_rrg_state_from_op(op=op12, sector=sector, membership=k200),
        "ex10": _op_rrg_state_from_op(
            op=_exclude_op_by_benchmark_weight(op=op12, benchmark_weights=bm_weights, threshold=0.10),
            sector=sector,
            membership=k200,
        ),
        "self": _op_rrg_state_from_sector_op(_sector_sum(values=op12, sector=sector, membership=k200).where(lambda frame: frame.gt(0.0))),
    }
    qavg, _count, _positive = _estimate_family_delta(
        frames=market.frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=close.index,
        columns=close.columns,
        sector=sector,
        lookback=20,
    )
    op12_delta, _op12_count, _op12_positive = _estimate_family_delta(
        frames=market.frames,
        keys=("op_fwd_12m",),
        index=close.index,
        columns=close.columns,
        sector=sector,
        lookback=20,
    )
    smart_flow = _smart_flow(market=market, like=close, market_cap=market_cap, lookback=20)
    stock_vs_sector_momentum = _stock_vs_sector_momentum(close=close, sector=sector, membership=k200, weights=market_cap, lookback=21)
    return {
        "close": close,
        "open": open_,
        "tradable": close.notna() & k200,
        "k200": k200,
        "sector": sector,
        "price_state": price_state,
        "op_states": op_states,
        "stock_scores": {"qavg": qavg, "op12": op12_delta},
        "smart_flow": smart_flow,
        "stock_vs_sector_momentum": stock_vs_sector_momentum,
        "benchmark_returns": benchmark_returns,
    }


def _op_rrg_state_from_op(*, op: pd.DataFrame, sector: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    sector_op = _sector_sum(values=op, sector=sector, membership=membership)
    market_op = op.where(membership).sum(axis=1, min_count=1)
    op_share = sector_op.where(sector_op.gt(0.0)).divide(market_op.where(market_op.gt(0.0)), axis=0)
    return _op_rrg_state_from_sector_op(op_share)


def _op_rrg_state_from_sector_op(frame: pd.DataFrame) -> pd.DataFrame:
    medium_mean = frame.rolling(126, min_periods=42).mean()
    short_mean = frame.rolling(42, min_periods=14).mean()
    relative_strength = frame.divide(medium_mean.replace(0.0, np.nan)) - 1.0
    short_relative = frame.divide(short_mean.replace(0.0, np.nan)) - 1.0
    momentum = short_relative - short_relative.shift(21)
    state, _long, _short = _classify_rrg_states(relative_strength=relative_strength, momentum=momentum)
    state = _apply_rrg_transition_hysteresis(state=state, relative_strength=relative_strength, momentum=momentum, threshold=0.005)
    return state.where(frame.notna(), "Unclassified")


def _sector_sum(*, values: pd.DataFrame, sector: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    rows: dict[pd.Timestamp, dict[object, float]] = {}
    for ts in values.index:
        valid = membership.loc[ts].astype(bool) & values.loc[ts].notna() & sector.loc[ts].notna()
        row: dict[object, float] = {}
        for sector_name in pd.unique(sector.loc[ts, valid]):
            names = values.columns[valid & sector.loc[ts].eq(sector_name)]
            row[sector_name] = float(values.loc[ts, names].sum())
        rows[ts] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=values.index)


def _smart_flow(*, market: MarketData, like: pd.DataFrame, market_cap: pd.DataFrame, lookback: int) -> pd.DataFrame:
    foreign = market.frames["foreign_flow"].reindex(index=like.index, columns=like.columns).fillna(0.0).astype(float)
    inst = market.frames["inst_flow"].reindex(index=like.index, columns=like.columns).fillna(0.0).astype(float)
    retail = market.frames["retail_flow"].reindex(index=like.index, columns=like.columns).fillna(0.0).astype(float)
    smart = foreign.add(inst, fill_value=0.0).sub(retail, fill_value=0.0)
    return smart.rolling(lookback, min_periods=max(5, lookback // 2)).sum().divide(market_cap.where(market_cap.gt(0.0)))


def _stock_vs_sector_momentum(*, close: pd.DataFrame, sector: pd.DataFrame, membership: pd.DataFrame, weights: pd.DataFrame, lookback: int) -> pd.DataFrame:
    stock_momentum = close.divide(close.shift(lookback)) - 1.0
    sector_returns = _sector_weighted_returns(returns=close.pct_change(fill_method=None), sector=sector, membership=membership, weights=weights)
    sector_index = (1.0 + sector_returns.fillna(0.0)).cumprod()
    sector_momentum = sector_index.divide(sector_index.shift(lookback)) - 1.0
    return stock_momentum.sub(_map_sector_values_to_symbols(sector=sector, sector_values=sector_momentum))


def _variant_grid() -> list[Variant]:
    variants: list[Variant] = []
    for op_rrg_mode in ("base", "ex10", "self"):
        for stock_score in ("qavg", "op12"):
            for compression, long_per_sector in (("k2", 2), ("k1", 1)):
                for confirm in ("none", "flow", "momo", "flow_momo"):
                    variants.append(
                        Variant(
                            id=f"{op_rrg_mode}_{stock_score}_{compression}_{confirm}",
                            op_rrg_mode=op_rrg_mode,
                            stock_score=stock_score,
                            compression=compression,
                            confirm=confirm,
                            long_per_sector=long_per_sector,
                            short_per_sector=1,
                        )
                    )
    return variants


def _build_plan(*, context: dict[str, object], variant: Variant) -> PositionPlan:
    bundle = _build_bundle(context=context, variant=variant)
    raw = _raw_rank_weights(bundle)
    sector = context["sector"].reindex(index=raw.index, columns=raw.columns)
    weights = pd.DataFrame(0.0, index=raw.index, columns=raw.columns, dtype=float)
    for ts in raw.index:
        row = raw.loc[ts]
        sector_row = sector.loc[ts]
        _compress_side(output=weights, ts=ts, names=row[row.gt(0.0)].index, per_sector=variant.long_per_sector, raw=row, sector_row=sector_row, score_row=bundle.alpha.loc[ts])
        _compress_side(output=weights, ts=ts, names=row[row.lt(0.0)].index, per_sector=variant.short_per_sector, raw=row, sector_row=sector_row, score_row=bundle.context["short_alpha"].loc[ts])
    construction = ConstructionResult(base_target_weights=weights, selection_mask=weights.ne(0.0), group_long_budget=None, group_short_budget=None, meta={})
    return PassThroughPolicy().apply(construction=construction, market=MarketData(frames={}, universe=None, benchmark=None), bundle=bundle)


def _build_bundle(*, context: dict[str, object], variant: Variant) -> SignalBundle:
    sector = context["sector"]
    k200 = context["k200"]
    price_state = context["price_state"]
    op_state = context["op_states"][variant.op_rrg_mode]
    stock_score = context["stock_scores"][variant.stock_score]
    price_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=price_state)
    op_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=op_state)
    long_ok = (
        price_by_symbol.isin(("Leading", "Improving", "Weakening"))
        & op_by_symbol.isin(("Leading", "Improving"))
        & stock_score.gt(0.0)
        & k200
    )
    short_ok = (
        price_by_symbol.eq("Lagging")
        & op_by_symbol.isin(("Lagging", "Weakening"))
        & stock_score.lt(0.0)
        & k200
    )
    if variant.confirm in {"flow", "flow_momo"}:
        smart_flow = context["smart_flow"]
        long_ok &= smart_flow.gt(0.0)
        short_ok &= smart_flow.lt(0.0)
    if variant.confirm in {"momo", "flow_momo"}:
        momentum = context["stock_vs_sector_momentum"]
        long_ok &= momentum.gt(0.0)
        short_ok &= momentum.lt(0.0)
    alpha = stock_score.where(long_ok)
    short_alpha = stock_score.mul(-1.0).where(short_ok)
    return SignalBundle(
        alpha=alpha,
        context={
            "tradable": k200,
            "entry_mask": long_ok.fillna(False).astype(bool),
            "hold_mask": long_ok.fillna(False).astype(bool),
            "short_entry_mask": short_ok.fillna(False).astype(bool),
            "short_hold_mask": short_ok.fillna(False).astype(bool),
            "short_alpha": short_alpha,
            "sector": sector,
        },
        meta={"price_rrg_state": price_state, "op_rrg_state": op_state, "stock_op_revision": stock_score},
    )


def _raw_rank_weights(bundle: SignalBundle) -> pd.DataFrame:
    alpha = bundle.alpha.astype(float)
    short_alpha = bundle.context["short_alpha"].reindex(index=alpha.index, columns=alpha.columns).astype(float)
    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns, dtype=float)
    for ts in alpha.index:
        long_scores = alpha.loc[ts].dropna()
        long_scores = long_scores[long_scores.gt(0.0)]
        if not long_scores.empty:
            ranks = long_scores.rank(method="first", ascending=True)
            weights.loc[ts, ranks.index] = ranks / float(ranks.sum()) * GROSS_LONG
        short_scores = short_alpha.loc[ts].dropna()
        short_scores = short_scores[short_scores.gt(0.0)]
        short_scores = short_scores.loc[short_scores.index.difference(long_scores.index)]
        if not short_scores.empty:
            ranks = short_scores.rank(method="first", ascending=True)
            weights.loc[ts, ranks.index] = -(ranks / float(ranks.sum()) * GROSS_SHORT)
    return weights


def _compress_side(
    *,
    output: pd.DataFrame,
    ts: pd.Timestamp,
    names: pd.Index,
    per_sector: int,
    raw: pd.Series,
    sector_row: pd.Series,
    score_row: pd.Series,
) -> None:
    if len(names) == 0:
        return
    for sector_name in pd.unique(sector_row.loc[names].dropna()):
        sector_names = names[sector_row.loc[names].eq(sector_name)]
        exposure = float(raw.loc[sector_names].sum())
        scores = score_row.reindex(sector_names).dropna()
        scores = scores[scores.gt(0.0)].sort_values(ascending=False, kind="stable")
        chosen = scores.head(per_sector).index
        if len(chosen) == 0:
            continue
        basis = raw.loc[chosen].abs()
        if float(basis.sum()) <= 0.0:
            basis = pd.Series(1.0, index=chosen, dtype=float)
        output.loc[ts, chosen] = basis / float(basis.sum()) * exposure


def _metric_row(*, strategy_id: str, variant: Variant, result: BacktestResult, benchmark: pd.Series, output_dir: Path) -> dict[str, object]:
    returns = result.returns.reindex(benchmark.index).fillna(0.0)
    equity = result.equity.reindex(benchmark.index).ffill()
    weights = result.weights.reindex(index=benchmark.index).fillna(0.0)
    turnover = result.turnover.reindex(benchmark.index).fillna(0.0)
    summary = summarize_perf(returns)
    monthly = (1.0 + returns).resample("ME").prod().sub(1.0)
    bm_monthly = (1.0 + benchmark).resample("ME").prod().sub(1.0)
    counts = weights.ne(0.0).sum(axis=1)
    active = weights.abs().sum(axis=1).gt(1e-12)
    return {
        "strategy_id": strategy_id,
        "op_rrg_mode": variant.op_rrg_mode,
        "stock_score": variant.stock_score,
        "compression": variant.compression,
        "confirm": variant.confirm,
        "cagr": summary["cagr"],
        "mdd": summary["mdd"],
        "sharpe": summary["sharpe"],
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "final_equity": float(equity.iloc[-1]),
        "monthly_win_rate": float(monthly.gt(0.0).mean()),
        "monthly_bm_win_rate": float(monthly.sub(bm_monthly, fill_value=0.0).gt(0.0).mean()),
        "avg_turnover": float(turnover.mean()),
        "avg_total_count": float(counts.loc[active].mean()) if bool(active.any()) else 0.0,
        "median_total_count": float(counts.loc[active].median()) if bool(active.any()) else 0.0,
        "p90_total_count": float(counts.loc[active].quantile(0.90)) if bool(active.any()) else 0.0,
        "max_total_count": int(counts.loc[active].max()) if bool(active.any()) else 0,
        "output_dir": str(output_dir),
    }


def _trim_result(result: BacktestResult) -> BacktestResult:
    return BacktestResult(
        equity=result.equity.loc[START:END].copy(),
        returns=result.returns.loc[START:END].copy(),
        weights=result.weights.loc[START:END].copy(),
        qty=result.qty.loc[START:END].copy(),
        turnover=result.turnover.loc[START:END].copy(),
    )


def _trim_plan(plan: PositionPlan) -> PositionPlan:
    ledger = plan.bucket_ledger
    if not ledger.empty and "date" in ledger.columns:
        dates = pd.to_datetime(ledger["date"])
        ledger = ledger.loc[dates.between(pd.Timestamp(START), pd.Timestamp(END))].copy()
    return PositionPlan(target_weights=plan.target_weights.loc[START:END].copy(), bucket_ledger=ledger, bucket_meta=plan.bucket_meta, validation=plan.validation)


if __name__ == "__main__":
    main()
