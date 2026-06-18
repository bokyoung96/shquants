from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

from root import ROOT

from backtesting.analytics import summarize_perf
from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data import DataLoader, LoadRequest, MarketData, ParquetStore
from backtesting.data.benchmarks import benchmark_price_series
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.execution import CostModel, WeeklySchedule
from backtesting.policy.base import BUCKET_LEDGER_COLUMNS, PositionPlan
from backtesting.signals.base import SignalBundle
from backtesting.strategies.rrg_sector_rotation import (
    _apply_rrg_transition_hysteresis,
    _build_rrg_context,
    _classify_rrg_states,
    _estimate_family_delta,
    _map_sector_state_to_symbols,
    _map_sector_values_to_symbols,
)


START = "2020-01-01"
END = "2026-05-11"
LOAD_START = "2019-01-01"
CAPITAL = 100_000_000.0
FEE = 0.0002
SELL_TAX = 0.0015
SLIPPAGE = 0.0005
RESULT_DIR = ROOT.results_path / "rrg_research" / "improvement_grid"

SCORE_MODES = ("qavg", "op12", "blend", "accel", "eps_op")
EVENT_MODES = (
    "none",
    "accel",
    "cross_up",
    "sector_turn",
    "price_turn",
    "new_high",
    "reclaim",
    "vol_break",
    "drawdown_repair",
    "op_lead",
)
FLOW_GATES = (
    "none",
    "smart",
    "foreign",
    "inst",
    "retail_contra",
    "smart_accel",
    "foreign_accel",
    "inst_accel",
    "flow_breadth",
    "anti_retail_breadth",
)
CONSTRUCTION_MODES = (
    "qavg_x115",
    "qavg_x120",
    "qavg_x125",
    "qavg_x128",
    "qavg_x130",
    "qavg_x132",
    "op12_x100",
    "op12_x105",
    "op12_x110",
    "qavg_x135",
)


@dataclass(frozen=True, slots=True)
class Variant:
    id: str
    score_mode: str
    event_mode: str
    flow_gate: str
    construction_mode: str


@dataclass(frozen=True, slots=True)
class ConstructionSpec:
    op_rrg_mode: str
    long_per_sector: int
    short_per_sector: int
    gross_long: float
    gross_short: float
    risk_overlay: str = "none"
    baseline_source: str | None = None
    baseline_scale: float = 1.0


def variant_grid() -> list[Variant]:
    variants: list[Variant] = []
    for score_mode in SCORE_MODES:
        for event_mode in EVENT_MODES:
            for flow_gate in FLOW_GATES:
                for construction_mode in CONSTRUCTION_MODES:
                    variants.append(
                        Variant(
                            id=f"rrgi_{score_mode}_{event_mode}_{flow_gate}_{construction_mode}",
                            score_mode=score_mode,
                            event_mode=event_mode,
                            flow_gate=flow_gate,
                            construction_mode=construction_mode,
                        )
                    )
    return variants


def rrg_hurdle() -> dict[str, float]:
    return {
        "min_cagr": 0.956171,
        "min_mdd": -0.166760,
        "min_sharpe": 2.302752,
        "min_monthly_bm_win_rate": 0.636364,
    }


def select_outperformers(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    hurdle = rrg_hurdle()
    selected = [
        row
        for row in rows
        if float(row["cagr"]) >= hurdle["min_cagr"]
        and float(row["mdd"]) >= hurdle["min_mdd"]
        and float(row["sharpe"]) >= hurdle["min_sharpe"]
        and float(row.get("monthly_bm_win_rate", hurdle["min_monthly_bm_win_rate"])) >= hurdle["min_monthly_bm_win_rate"]
    ]
    return sorted(selected, key=lambda row: (float(row["sharpe"]), float(row["mdd"]), float(row["cagr"])), reverse=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run 5,000 fixed RRG-improvement backtests.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N variants for smoke checks.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based variant offset.")
    parser.add_argument("--end", default=END)
    args = parser.parse_args(argv)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    market = _load_market(end=args.end)
    context = _build_context(market, end=args.end)
    variants = variant_grid()
    if args.start_index:
        variants = variants[args.start_index :]
    if args.limit is not None:
        variants = variants[: args.limit]

    engine = BacktestEngine(cost=CostModel(fee=FEE, sell_tax=SELL_TAX, slippage=SLIPPAGE))
    rows: list[dict[str, object]] = []
    for idx, variant in enumerate(variants, start=1 + args.start_index):
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
        result = _trim_result(result, end=args.end)
        rows.append(_metric_row(strategy_id=variant.id, variant=variant, result=result, benchmark=context["benchmark_returns"]))
        if idx == 1 or idx % 50 == 0 or idx == len(variants) + args.start_index:
            best = max(rows, key=lambda row: (float(row["sharpe"]), float(row["mdd"]), float(row["cagr"])))
            print(
                f"[{idx}/{len(variants) + args.start_index}] best={best['strategy_id']} "
                f"cagr={float(best['cagr']):.4f} mdd={float(best['mdd']):.4f} sharpe={float(best['sharpe']):.4f}",
                flush=True,
            )

    frame = pd.DataFrame(rows).sort_values(["sharpe", "mdd", "cagr"], ascending=[False, False, False])
    selected = select_outperformers(frame.to_dict(orient="records"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "smoke" if args.limit is not None else "full"
    summary_path = RESULT_DIR / f"rrg_improvement_{suffix}_{stamp}.csv"
    json_path = RESULT_DIR / f"rrg_improvement_{suffix}_{stamp}.json"
    selected_path = RESULT_DIR / f"rrg_improvement_{suffix}_selected_{stamp}.json"
    frame.to_csv(summary_path, index=False)
    json_path.write_text(frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.head(15).to_string(index=False))
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "json": str(json_path),
                "selected": str(selected_path),
                "count": len(frame),
                "outperformer_count": len(selected),
                "hurdle": rrg_hurdle(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_market(*, end: str) -> MarketData:
    loader = DataLoader(DataCatalog.default(), ParquetStore(ROOT.parquet_path))
    return loader.load(
        LoadRequest(
            datasets=[
                DatasetId.QW_ADJ_C,
                DatasetId.QW_ADJ_O,
                DatasetId.QW_BM,
                DatasetId.QW_K200_YN,
                DatasetId.QW_WICS_SEC_BIG,
                DatasetId.QW_MKTCAP,
                DatasetId.QW_MKTCAP_FLT,
                DatasetId.QW_OP_NFQ1,
                DatasetId.QW_OP_NFQ2,
                DatasetId.QW_OP_NFY1,
                DatasetId.QW_EPS_NFY1,
                DatasetId.QW_FOREIGN,
                DatasetId.QW_INSTITUTION,
                DatasetId.QW_RETAIL,
            ],
            start=LOAD_START,
            end=end,
        )
    )


def _build_context(market: MarketData, *, end: str) -> dict[str, object]:
    close = market.frames["close"].astype(float)
    k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
    active_columns = k200.columns[k200.any(axis=0)]
    close = close.loc[:, active_columns]
    open_ = market.frames["open"].reindex(index=close.index, columns=close.columns).astype(float)
    k200 = k200.loc[:, active_columns]
    sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
    market_cap = market.frames.get("float_market_cap", market.frames["market_cap"]).reindex(index=close.index, columns=close.columns).ffill().astype(float)
    benchmark = benchmark_price_series(market.frames["benchmark"], "IKS200").reindex(close.index).ffill().astype(float)
    benchmark_returns = benchmark.pct_change(fill_method=None).fillna(0.0).loc[START:end].rename("KOSPI200")

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
    op_for_rrg = market.frames["op_fwd"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
    op_states = {
        "base": _op_rrg_state_from_op(op=op_for_rrg, sector=sector, membership=k200),
        "self": _op_rrg_state_from_sector_op(_sector_sum(values=op_for_rrg, sector=sector, membership=k200).where(lambda frame: frame.gt(0.0))),
    }
    scores = _score_frames(frames=market.frames, close=close, sector=sector)
    events = {mode: _event_mask(mode=mode, close=close, score=scores["qavg"], price_state=price_state, op_state=op_states["base"], sector=sector) for mode in EVENT_MODES}
    flows = {gate: _flow_masks(gate=gate, market=market, like=close, market_cap=market_cap, sector=sector, membership=k200) for gate in FLOW_GATES}
    benchmark_trend = benchmark.gt(benchmark.rolling(120, min_periods=40).mean()).reindex(close.index).fillna(False)
    benchmark_vol = benchmark.pct_change(fill_method=None).rolling(63, min_periods=21).std().reindex(close.index)
    baseline_weights = _load_baseline_weights(index=close.index, columns=close.columns)
    return {
        "close": close,
        "open": open_,
        "tradable": close.notna() & k200,
        "k200": k200,
        "sector": sector,
        "price_state": price_state,
        "op_states": op_states,
        "scores": scores,
        "events": events,
        "flows": flows,
        "benchmark_trend": benchmark_trend,
        "benchmark_vol": benchmark_vol,
        "baseline_weights": baseline_weights,
        "benchmark_returns": benchmark_returns,
    }


def _score_frames(*, frames: dict[str, pd.DataFrame], close: pd.DataFrame, sector: pd.DataFrame) -> dict[str, pd.DataFrame]:
    qavg, _count, _positive = _estimate_family_delta(
        frames=frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=close.index,
        columns=close.columns,
        sector=sector,
        lookback=20,
    )
    op12, _op12_count, _op12_positive = _estimate_family_delta(
        frames=frames,
        keys=("op_fwd",),
        index=close.index,
        columns=close.columns,
        sector=sector,
        lookback=20,
    )
    eps, _eps_count, _eps_positive = _estimate_family_delta(
        frames=frames,
        keys=("eps_fwd",),
        index=close.index,
        columns=close.columns,
        sector=sector,
        lookback=20,
    )
    blend = qavg.add(op12, fill_value=0.0).div(2.0)
    accel = qavg.sub(qavg.shift(20))
    eps_op = qavg.add(eps, fill_value=0.0).div(2.0)
    return {"qavg": qavg, "op12": op12, "blend": blend, "accel": accel, "eps_op": eps_op}


def _build_plan(*, context: dict[str, object], variant: Variant) -> PositionPlan:
    spec = _construction_spec(variant.construction_mode)
    if spec.baseline_source is not None and spec.baseline_source in context["baseline_weights"]:
        weights = _baseline_overlay_weights(context=context, variant=variant, spec=spec)
        return _position_plan(weights)
    bundle = _build_bundle(context=context, variant=variant)
    raw = _raw_rank_weights(bundle=bundle, gross_long=spec.gross_long, gross_short=spec.gross_short)
    weights = _compress_weights(raw=raw, bundle=bundle, spec=spec)
    if spec.risk_overlay == "vol85":
        weights = _apply_vol_overlay(weights=weights, trend=context["benchmark_trend"], vol=context["benchmark_vol"])
    return _position_plan(weights)


def _position_plan(weights: pd.DataFrame) -> PositionPlan:
    return PositionPlan(
        target_weights=weights.fillna(0.0).astype(float),
        bucket_ledger=pd.DataFrame(columns=BUCKET_LEDGER_COLUMNS),
        bucket_meta={},
        validation={},
    )


def _build_bundle(*, context: dict[str, object], variant: Variant) -> SignalBundle:
    spec = _construction_spec(variant.construction_mode)
    sector = context["sector"]
    k200 = context["k200"]
    price_state = context["price_state"]
    op_state = context["op_states"][spec.op_rrg_mode]
    score = context["scores"][variant.score_mode]
    price_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=price_state)
    op_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=op_state)
    event_ok, short_event_ok = context["events"][variant.event_mode]
    flow_ok, short_flow_ok = context["flows"][variant.flow_gate]
    long_ok = (
        price_by_symbol.isin(("Leading", "Improving", "Weakening"))
        & op_by_symbol.isin(("Leading", "Improving"))
        & score.gt(0.0)
        & event_ok
        & flow_ok
        & k200
    )
    short_ok = (
        price_by_symbol.eq("Lagging")
        & op_by_symbol.isin(("Lagging", "Weakening"))
        & score.lt(0.0)
        & short_event_ok
        & short_flow_ok
        & k200
    )
    short_alpha = score.mul(-1.0).where(short_ok)
    return SignalBundle(
        alpha=score.where(long_ok),
        context={
            "tradable": k200,
            "entry_mask": long_ok.fillna(False).astype(bool),
            "hold_mask": long_ok.fillna(False).astype(bool),
            "short_entry_mask": short_ok.fillna(False).astype(bool),
            "short_hold_mask": short_ok.fillna(False).astype(bool),
            "short_alpha": short_alpha,
            "sector": sector,
        },
        meta={"price_rrg_state": price_state, "op_rrg_state": op_state, "stock_score": score},
    )


def _construction_spec(mode: str) -> ConstructionSpec:
    specs = {
        "qavg_x115": ConstructionSpec("base", 2, 1, 1.0, 0.5, baseline_source="qavg", baseline_scale=1.15),
        "qavg_x120": ConstructionSpec("base", 2, 1, 1.0, 0.5, baseline_source="qavg", baseline_scale=1.20),
        "qavg_x125": ConstructionSpec("base", 2, 1, 1.0, 0.5, baseline_source="qavg", baseline_scale=1.25),
        "qavg_x128": ConstructionSpec("base", 2, 1, 1.0, 0.5, baseline_source="qavg", baseline_scale=1.28),
        "qavg_x130": ConstructionSpec("base", 2, 1, 1.0, 0.5, baseline_source="qavg", baseline_scale=1.30),
        "qavg_x132": ConstructionSpec("base", 2, 1, 1.0, 0.5, baseline_source="qavg", baseline_scale=1.32),
        "op12_x100": ConstructionSpec("base", 2, 1, 1.0, 0.5, baseline_source="op12", baseline_scale=1.00),
        "op12_x105": ConstructionSpec("base", 2, 1, 1.0, 0.5, baseline_source="op12", baseline_scale=1.05),
        "op12_x110": ConstructionSpec("base", 2, 1, 1.0, 0.5, baseline_source="op12", baseline_scale=1.10),
        "qavg_x135": ConstructionSpec("base", 2, 1, 1.0, 0.5, baseline_source="qavg", baseline_scale=1.35),
    }
    try:
        return specs[mode]
    except KeyError as exc:
        raise ValueError(f"unknown construction mode: {mode}") from exc


def _load_baseline_weights(*, index: pd.Index, columns: pd.Index) -> dict[str, pd.DataFrame]:
    paths = {
        "qavg": ROOT.results_path / "backtests" / "rrg-op-rrg-grid-base_qavg_k2_none_20260618_173937" / "positions" / "weights.parquet",
        "op12": ROOT.results_path / "backtests" / "rrg-op-rrg-grid-base_op12_k2_none_20260618_174230" / "positions" / "weights.parquet",
    }
    weights: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if path.exists():
            weights[name] = pd.read_parquet(path).reindex(index=index, columns=columns).fillna(0.0).astype(float)
    return weights


def _baseline_overlay_weights(*, context: dict[str, object], variant: Variant, spec: ConstructionSpec) -> pd.DataFrame:
    weights = context["baseline_weights"][spec.baseline_source].mul(spec.baseline_scale)
    event_ok, short_event_ok = context["events"][variant.event_mode]
    flow_ok, short_flow_ok = context["flows"][variant.flow_gate]
    if variant.event_mode == "none" and variant.flow_gate == "none":
        return weights
    long_ok = event_ok & flow_ok
    short_ok = short_event_ok & short_flow_ok
    return weights.where((weights.gt(0.0) & long_ok) | (weights.lt(0.0) & short_ok), 0.0)


def _raw_rank_weights(*, bundle: SignalBundle, gross_long: float, gross_short: float) -> pd.DataFrame:
    alpha = bundle.alpha.astype(float)
    short_alpha = bundle.context["short_alpha"].reindex(index=alpha.index, columns=alpha.columns).astype(float)
    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns, dtype=float)
    for ts in alpha.index:
        long_scores = alpha.loc[ts].dropna()
        long_scores = long_scores[long_scores.gt(0.0)]
        if not long_scores.empty:
            ranks = long_scores.rank(method="first", ascending=True)
            weights.loc[ts, ranks.index] = ranks / float(ranks.sum()) * gross_long
        short_scores = short_alpha.loc[ts].dropna()
        short_scores = short_scores[short_scores.gt(0.0)]
        short_scores = short_scores.loc[short_scores.index.difference(long_scores.index)]
        if not short_scores.empty and gross_short > 0.0:
            ranks = short_scores.rank(method="first", ascending=True)
            weights.loc[ts, ranks.index] = -(ranks / float(ranks.sum()) * gross_short)
    return weights


def _compress_weights(*, raw: pd.DataFrame, bundle: SignalBundle, spec: ConstructionSpec) -> pd.DataFrame:
    sector = bundle.context["sector"].reindex(index=raw.index, columns=raw.columns)
    weights = pd.DataFrame(0.0, index=raw.index, columns=raw.columns, dtype=float)
    for ts in raw.index:
        row = raw.loc[ts]
        sector_row = sector.loc[ts]
        _compress_side(output=weights, ts=ts, names=row[row.gt(0.0)].index, per_sector=spec.long_per_sector, raw=row, sector_row=sector_row, score_row=bundle.alpha.loc[ts])
        _compress_side(output=weights, ts=ts, names=row[row.lt(0.0)].index, per_sector=spec.short_per_sector, raw=row, sector_row=sector_row, score_row=bundle.context["short_alpha"].loc[ts])
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


def _apply_vol_overlay(*, weights: pd.DataFrame, trend: pd.Series, vol: pd.Series) -> pd.DataFrame:
    median_vol = vol.rolling(252, min_periods=63).median()
    calm = vol.le(median_vol).reindex(weights.index).fillna(False)
    trend = trend.reindex(weights.index).fillna(False)
    scale = pd.Series(0.85, index=weights.index, dtype=float)
    scale.loc[trend & calm] = 1.0
    scale.loc[~trend & ~calm] = 0.65
    return weights.mul(scale, axis=0)


def _event_mask(
    *,
    mode: str,
    close: pd.DataFrame,
    score: pd.DataFrame,
    price_state: pd.DataFrame,
    op_state: pd.DataFrame,
    sector: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame(True, index=close.index, columns=close.columns)
    if mode == "none":
        return base, base
    if mode == "accel":
        return score.gt(score.shift(20)), score.lt(score.shift(20))
    if mode == "cross_up":
        return score.gt(0.0) & score.shift(20).le(0.0), score.lt(0.0) & score.shift(20).ge(0.0)
    price_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=price_state)
    op_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=op_state)
    if mode == "sector_turn":
        long_mask = op_by_symbol.isin(("Leading", "Improving")) & ~op_by_symbol.shift(20).isin(("Leading", "Improving"))
        short_mask = op_by_symbol.isin(("Lagging", "Weakening")) & ~op_by_symbol.shift(20).isin(("Lagging", "Weakening"))
        return long_mask, short_mask
    if mode == "price_turn":
        long_mask = price_by_symbol.isin(("Leading", "Improving")) & ~price_by_symbol.shift(20).isin(("Leading", "Improving"))
        short_mask = price_by_symbol.eq("Lagging") & ~price_by_symbol.shift(20).eq("Lagging")
        return long_mask, short_mask
    if mode == "new_high":
        high = close.rolling(63, min_periods=21).max()
        low = close.rolling(63, min_periods=21).min()
        return close.ge(high.mul(0.99)), close.le(low.mul(1.01))
    if mode == "reclaim":
        ma = close.rolling(60, min_periods=20).mean()
        return close.gt(ma) & close.shift(5).le(ma.shift(5)), close.lt(ma) & close.shift(5).ge(ma.shift(5))
    if mode == "vol_break":
        vol = close.pct_change(fill_method=None).rolling(21, min_periods=10).std()
        vol_base = vol.rolling(126, min_periods=42).median()
        return vol.lt(vol_base) & score.gt(0.0), vol.gt(vol_base) & score.lt(0.0)
    if mode == "drawdown_repair":
        peak = close.rolling(252, min_periods=63).max()
        trough = close.rolling(252, min_periods=63).min()
        return close.divide(peak).gt(0.85) & close.pct_change(21, fill_method=None).gt(0.0), close.divide(trough).lt(1.15) & close.pct_change(21, fill_method=None).lt(0.0)
    if mode == "op_lead":
        return op_by_symbol.eq("Leading"), op_by_symbol.eq("Lagging")
    raise ValueError(f"unknown event mode: {mode}")


def _flow_masks(
    *,
    gate: str,
    market: MarketData,
    like: pd.DataFrame,
    market_cap: pd.DataFrame,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame(True, index=like.index, columns=like.columns)
    if gate == "none":
        return base, base
    foreign = market.frames["foreign_flow"].reindex(index=like.index, columns=like.columns).fillna(0.0).astype(float)
    inst = market.frames["inst_flow"].reindex(index=like.index, columns=like.columns).fillna(0.0).astype(float)
    retail = market.frames["retail_flow"].reindex(index=like.index, columns=like.columns).fillna(0.0).astype(float)
    scale = market_cap.where(market_cap.gt(0.0))
    smart = foreign.add(inst, fill_value=0.0).sub(retail, fill_value=0.0)
    rolling = {
        "smart": smart.rolling(20, min_periods=10).sum().divide(scale),
        "foreign": foreign.rolling(20, min_periods=10).sum().divide(scale),
        "inst": inst.rolling(20, min_periods=10).sum().divide(scale),
        "retail_contra": retail.rolling(20, min_periods=10).sum().divide(scale).mul(-1.0),
    }
    if gate in rolling:
        signal = rolling[gate]
        return signal.gt(0.0), signal.lt(0.0)
    if gate == "smart_accel":
        signal = rolling["smart"].sub(rolling["smart"].shift(20))
        return signal.gt(0.0), signal.lt(0.0)
    if gate == "foreign_accel":
        signal = rolling["foreign"].sub(rolling["foreign"].shift(20))
        return signal.gt(0.0), signal.lt(0.0)
    if gate == "inst_accel":
        signal = rolling["inst"].sub(rolling["inst"].shift(20))
        return signal.gt(0.0), signal.lt(0.0)
    if gate in {"flow_breadth", "anti_retail_breadth"}:
        signal = rolling["smart"] if gate == "flow_breadth" else rolling["retail_contra"]
        sector_signal = _sector_average(values=signal, sector=sector, membership=membership)
        mapped = _map_sector_values_to_symbols(sector=sector, sector_values=sector_signal)
        return mapped.gt(0.0), mapped.lt(0.0)
    raise ValueError(f"unknown flow gate: {gate}")


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


def _sector_average(*, values: pd.DataFrame, sector: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    rows: dict[pd.Timestamp, dict[object, float]] = {}
    for ts in values.index:
        valid = membership.loc[ts].astype(bool) & values.loc[ts].notna() & sector.loc[ts].notna()
        row: dict[object, float] = {}
        for sector_name in pd.unique(sector.loc[ts, valid]):
            names = values.columns[valid & sector.loc[ts].eq(sector_name)]
            row[sector_name] = float(values.loc[ts, names].mean())
        rows[ts] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=values.index)


def _metric_row(*, strategy_id: str, variant: Variant, result: BacktestResult, benchmark: pd.Series) -> dict[str, object]:
    returns = result.returns.reindex(benchmark.index).fillna(0.0)
    equity = result.equity.reindex(benchmark.index).ffill()
    weights = result.weights.reindex(index=benchmark.index).fillna(0.0)
    turnover = result.turnover.reindex(benchmark.index).fillna(0.0)
    summary = summarize_perf(returns)
    monthly = (1.0 + returns).resample("ME").prod().sub(1.0)
    bm_monthly = (1.0 + benchmark).resample("ME").prod().sub(1.0)
    counts = weights.ne(0.0).sum(axis=1)
    active = weights.abs().sum(axis=1).gt(1e-12)
    hurdle = rrg_hurdle()
    return {
        "strategy_id": strategy_id,
        "score_mode": variant.score_mode,
        "event_mode": variant.event_mode,
        "flow_gate": variant.flow_gate,
        "construction_mode": variant.construction_mode,
        "cagr": summary["cagr"],
        "mdd": summary["mdd"],
        "sharpe": summary["sharpe"],
        "beats_rrg": summary["cagr"] >= hurdle["min_cagr"] and summary["mdd"] >= hurdle["min_mdd"] and summary["sharpe"] >= hurdle["min_sharpe"],
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "final_equity": float(equity.iloc[-1]),
        "monthly_win_rate": float(monthly.gt(0.0).mean()),
        "monthly_bm_win_rate": float(monthly.sub(bm_monthly, fill_value=0.0).gt(0.0).mean()),
        "avg_turnover": float(turnover.mean()),
        "avg_total_count": float(counts.loc[active].mean()) if bool(active.any()) else 0.0,
        "median_total_count": float(counts.loc[active].median()) if bool(active.any()) else 0.0,
        "p90_total_count": float(counts.loc[active].quantile(0.90)) if bool(active.any()) else 0.0,
        "max_total_count": int(counts.loc[active].max()) if bool(active.any()) else 0,
    }


def _trim_result(result: BacktestResult, *, end: str) -> BacktestResult:
    return BacktestResult(
        equity=result.equity.loc[START:end].copy(),
        returns=result.returns.loc[START:end].copy(),
        weights=result.weights.loc[START:end].copy(),
        qty=result.qty.loc[START:end].copy(),
        turnover=result.turnover.loc[START:end].copy(),
    )


if __name__ == "__main__":
    main()
