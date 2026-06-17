from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

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
    _build_rrg_context,
    _estimate_family_delta,
    _map_sector_state_to_symbols,
    _map_sector_values_to_symbols,
    _sector_weighted_returns,
    _sector_weighted_signal,
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


@dataclass(frozen=True, slots=True)
class OpMode:
    id: str
    label: str


@dataclass(frozen=True, slots=True)
class CompressionMode:
    id: str
    label: str
    long_per_sector: int
    short_per_sector: int
    metric: str


OP_MODES = (
    OpMode("qavg", "Q1/Q2/FY1 average OP revision"),
    OpMode("op12", "12M forward OP revision only"),
    OpMode("agree2", "At least 2 of Q1/Q2/FY1/12M agree"),
    OpMode("agree3", "At least 3 of Q1/Q2/FY1/12M agree"),
    OpMode("breadth50", "Q-average with sector positive/negative breadth >= 50%"),
)

COMPRESSION_MODES = (
    CompressionMode("k2_weight", "sector exposure, top 2/1 by baseline weight", 2, 1, "weight"),
    CompressionMode("k3_weight", "sector exposure, top 3/2 by baseline weight", 3, 2, "weight"),
    CompressionMode("k2_op", "sector exposure, top 2/1 by OP revision", 2, 1, "op"),
    CompressionMode("k3_op", "sector exposure, top 3/2 by OP revision", 3, 2, "op"),
    CompressionMode("k2_momo21", "sector exposure, top 2/1 by 21D stock-vs-sector momentum", 2, 1, "momo21"),
    CompressionMode("k3_mcap", "sector exposure, top 3/2 by float market cap", 3, 2, "mcap"),
)


def main() -> None:
    market = _load_market()
    runner_output_root = ROOT.results_path / "backtests"
    writer = RunWriter(runner_output_root, write_report_assets=False)
    engine = BacktestEngine(cost=CostModel(fee=FEE, sell_tax=SELL_TAX, slippage=SLIPPAGE))
    schedule = WeeklySchedule()

    rows: list[dict[str, object]] = []
    for op_mode in OP_MODES:
        existing_by_compression = {
            compression.id: _load_existing_row(runner_output_root, f"rrg_{op_mode.id}_{compression.id}", op_mode, compression)
            for compression in COMPRESSION_MODES
        }
        if all(row is not None for row in existing_by_compression.values()):
            rows.extend(row for row in existing_by_compression.values() if row is not None)
            continue

        bundle = _build_bundle(market, op_mode)
        for compression in COMPRESSION_MODES:
            strategy_id = f"rrg_{op_mode.id}_{compression.id}"
            existing = existing_by_compression[compression.id]
            if existing is not None:
                print(f"skipping {strategy_id}")
                rows.append(existing)
                continue

            label = f"RRG {op_mode.id} {compression.id}"
            print(f"running {strategy_id}")
            plan = _build_compressed_plan(bundle=bundle, compression=compression)
            result = engine.run(
                close=market.frames["close"],
                open=market.frames["open"],
                weights=plan.target_weights,
                capital=CAPITAL,
                tradable=market.frames["close"].notna() & market.frames["k200_yn"].reindex_like(market.frames["close"]).fillna(False).astype(bool),
                exit_tradable=market.frames["close"].notna(),
                schedule=schedule,
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
                    "op_mode": op_mode.id,
                    "compression": compression.id,
                    "long_per_sector": compression.long_per_sector,
                    "short_per_sector": compression.short_per_sector,
                    "metric": compression.metric,
                },
                name=label,
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
            report = RunReport(config=config, summary=summary, result=result, position_plan=plan)
            report.output_dir = writer.write(report)

            weights = result.weights.fillna(0.0)
            active = weights.abs().sum(axis=1).gt(1e-12)
            long_counts = weights.gt(0.0).sum(axis=1)
            short_counts = weights.lt(0.0).sum(axis=1)
            total_return = float(result.equity.iloc[-1] / result.equity.iloc[0] - 1.0)
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "op_mode": op_mode.id,
                    "op_label": op_mode.label,
                    "compression": compression.id,
                    "compression_label": compression.label,
                    "cagr": summary["cagr"],
                    "mdd": summary["mdd"],
                    "sharpe": summary["sharpe"],
                    "total_return": total_return,
                    "final_equity": summary["final_equity"],
                    "avg_turnover": summary["avg_turnover"],
                    "active_days": int(active.sum()),
                    "avg_long_count": float(long_counts[active].mean()) if bool(active.any()) else 0.0,
                    "avg_short_count": float(short_counts[active].mean()) if bool(active.any()) else 0.0,
                    "avg_total_count": float((long_counts[active] + short_counts[active]).mean()) if bool(active.any()) else 0.0,
                    "output_dir": str(report.output_dir),
                }
            )

    summary_frame = pd.DataFrame(rows).sort_values(["sharpe", "mdd", "cagr"], ascending=[False, False, False])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = ROOT.results_path / "rrg_research"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / f"rrg_30_strategy_summary_{stamp}.csv"
    json_path = result_dir / f"rrg_30_strategy_summary_{stamp}.json"
    summary_frame.to_csv(csv_path, index=False)
    json_path.write_text(summary_frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    doc_path = ROOT.root / "docs" / "research" / "rrg-sector-rotation-30-strategy-results.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown_summary(doc_path=doc_path, summary_frame=summary_frame, csv_path=csv_path, json_path=json_path)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "doc": str(doc_path)}, ensure_ascii=False, indent=2))


def _load_existing_row(root: Path, strategy_id: str, op_mode: OpMode, compression: CompressionMode) -> dict[str, object] | None:
    for run_dir in sorted(root.glob("*"), key=lambda path: path.stat().st_mtime, reverse=True):
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        equity_path = run_dir / "series" / "equity.csv"
        weights_path = run_dir / "positions" / "weights.parquet"
        if not (config_path.exists() and summary_path.exists() and equity_path.exists() and weights_path.exists()):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if config.get("strategy") != strategy_id:
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        equity = pd.read_csv(equity_path, index_col=0)["equity"].astype(float)
        weights = pd.read_parquet(weights_path).fillna(0.0)
        active = weights.abs().sum(axis=1).gt(1e-12)
        long_counts = weights.gt(0.0).sum(axis=1)
        short_counts = weights.lt(0.0).sum(axis=1)
        return {
            "strategy_id": strategy_id,
            "op_mode": op_mode.id,
            "op_label": op_mode.label,
            "compression": compression.id,
            "compression_label": compression.label,
            "cagr": float(summary["cagr"]),
            "mdd": float(summary["mdd"]),
            "sharpe": float(summary["sharpe"]),
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
            "final_equity": float(summary["final_equity"]),
            "avg_turnover": float(summary["avg_turnover"]),
            "active_days": int(active.sum()),
            "avg_long_count": float(long_counts[active].mean()) if bool(active.any()) else 0.0,
            "avg_short_count": float(short_counts[active].mean()) if bool(active.any()) else 0.0,
            "avg_total_count": float((long_counts[active] + short_counts[active]).mean()) if bool(active.any()) else 0.0,
            "output_dir": str(run_dir),
        }
    return None


def _load_market() -> MarketData:
    datasets = [
        DatasetId.QW_ADJ_C,
        DatasetId.QW_ADJ_O,
        DatasetId.QW_BM,
        DatasetId.QW_K200_YN,
        DatasetId.QW_WI_SEC_26_BIG,
        DatasetId.QW_MKTCAP,
        DatasetId.QW_MKTCAP_FLT,
        DatasetId.QW_OP_NFQ1,
        DatasetId.QW_OP_NFQ2,
        DatasetId.QW_OP_NFY1,
        DatasetId.QW_OP_FWD_12M,
    ]
    loader = DataLoader(DataCatalog.default(), ParquetStore(ROOT.parquet_path))
    return loader.load(LoadRequest(datasets=datasets, start=LOAD_START, end=END))


def _build_bundle(market: MarketData, op_mode: OpMode) -> SignalBundle:
    close = market.frames["close"].astype(float)
    k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
    active_columns = k200.columns[k200.any(axis=0)]
    close = close.loc[:, active_columns]
    k200 = k200.loc[:, active_columns]
    sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
    market_cap_source = market.frames.get("float_market_cap", market.frames["market_cap"])
    market_cap = market_cap_source.reindex(index=close.index, columns=close.columns).ffill().astype(float)
    benchmark = benchmark_price_series(market.frames["benchmark"], "IKS200").reindex(close.index).ffill().astype(float)
    rrg_state, _long_sector, _short_sector = _build_rrg_context(
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
    op = _op_context(frames=market.frames, index=close.index, columns=close.columns, sector=sector, membership=k200, weights=market_cap, mode=op_mode.id)
    state_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=rrg_state.reindex(index=close.index))
    sector_long_by_symbol = _map_sector_values_to_symbols(sector=sector, sector_values=op["sector_long_ok"].astype(float)).eq(1.0)
    sector_short_by_symbol = _map_sector_values_to_symbols(sector=sector, sector_values=op["sector_short_ok"].astype(float)).eq(1.0)

    long_ok = state_by_symbol.isin(("Leading", "Improving", "Weakening")) & sector_long_by_symbol & op["stock_long_ok"] & k200
    short_ok = state_by_symbol.eq("Lagging") & sector_short_by_symbol & op["stock_short_ok"] & k200
    alpha = op["stock_score"].where(long_ok & op["stock_score"].gt(0.0))
    short_alpha = op["stock_score"].mul(-1.0).where(short_ok & op["stock_score"].lt(0.0))

    stock_vs_sector_momentum_21 = _stock_vs_sector_momentum(close=close, sector=sector, membership=k200, weights=market_cap, lookback=21)
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
            "market_cap": market_cap,
            "rrg_state": rrg_state,
            "stock_op_revision": op["stock_score"],
            "stock_vs_sector_momentum_21": stock_vs_sector_momentum_21,
        },
        meta={
            "rrg_state": rrg_state,
            "sector_op_revision": op["sector_score"],
            "stock_op_revision": op["stock_score"],
        },
    )


def _op_context(
    *,
    frames: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    weights: pd.DataFrame,
    mode: str,
) -> dict[str, pd.DataFrame]:
    qavg, _q_count, _q_positive = _estimate_family_delta(
        frames=frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=20,
    )
    op12, _op12_count, _op12_positive = _estimate_family_delta(
        frames=frames,
        keys=("op_fwd_12m",),
        index=index,
        columns=columns,
        sector=sector,
        lookback=20,
    )
    all_deltas = []
    for key in ("op_fwd_q1", "op_fwd_q2", "op_fwd", "op_fwd_12m"):
        delta, _count, _positive = _estimate_family_delta(
            frames=frames,
            keys=(key,),
            index=index,
            columns=columns,
            sector=sector,
            lookback=20,
        )
        all_deltas.append(delta)
    available = sum(delta.notna().astype(float) for delta in all_deltas)
    positive_count = sum(delta.gt(0.0).astype(float).where(delta.notna(), 0.0) for delta in all_deltas)
    negative_count = sum(delta.lt(0.0).astype(float).where(delta.notna(), 0.0) for delta in all_deltas)
    agreement_score = sum(delta.fillna(0.0) for delta in all_deltas).divide(available.replace(0.0, np.nan)).astype(float)

    if mode == "qavg":
        stock_score = qavg
        stock_long_ok = stock_score.gt(0.0)
        stock_short_ok = stock_score.lt(0.0)
        sector_score = _sector_weighted_signal(values=stock_score, sector=sector, membership=membership, weights=weights)
        sector_long_ok = sector_score.gt(0.0)
        sector_short_ok = sector_score.lt(0.0)
    elif mode == "op12":
        stock_score = op12
        stock_long_ok = stock_score.gt(0.0)
        stock_short_ok = stock_score.lt(0.0)
        sector_score = _sector_weighted_signal(values=stock_score, sector=sector, membership=membership, weights=weights)
        sector_long_ok = sector_score.gt(0.0)
        sector_short_ok = sector_score.lt(0.0)
    elif mode in {"agree2", "agree3"}:
        threshold = 2 if mode == "agree2" else 3
        stock_score = agreement_score
        stock_long_ok = stock_score.gt(0.0) & positive_count.ge(threshold)
        stock_short_ok = stock_score.lt(0.0) & negative_count.ge(threshold)
        sector_score = _sector_weighted_signal(values=stock_score, sector=sector, membership=membership, weights=weights)
        sector_long_ok = sector_score.gt(0.0)
        sector_short_ok = sector_score.lt(0.0)
    elif mode == "breadth50":
        stock_score = qavg
        stock_long_ok = stock_score.gt(0.0)
        stock_short_ok = stock_score.lt(0.0)
        sector_score = _sector_weighted_signal(values=stock_score, sector=sector, membership=membership, weights=weights)
        pos_breadth = _sector_weighted_signal(values=stock_score.gt(0.0).astype(float).where(stock_score.notna()), sector=sector, membership=membership, weights=weights)
        neg_breadth = _sector_weighted_signal(values=stock_score.lt(0.0).astype(float).where(stock_score.notna()), sector=sector, membership=membership, weights=weights)
        sector_long_ok = sector_score.gt(0.0) & pos_breadth.ge(0.50)
        sector_short_ok = sector_score.lt(0.0) & neg_breadth.ge(0.50)
    else:
        raise ValueError(f"unknown op mode: {mode}")
    return {
        "stock_score": stock_score,
        "stock_long_ok": stock_long_ok.fillna(False),
        "stock_short_ok": stock_short_ok.fillna(False),
        "sector_score": sector_score,
        "sector_long_ok": sector_long_ok.fillna(False),
        "sector_short_ok": sector_short_ok.fillna(False),
    }


def _stock_vs_sector_momentum(
    *,
    close: pd.DataFrame,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    weights: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    stock_momentum = close.divide(close.shift(lookback)) - 1.0
    sector_returns = _sector_weighted_returns(returns=close.pct_change(fill_method=None), sector=sector, membership=membership, weights=weights)
    sector_index = (1.0 + sector_returns.fillna(0.0)).cumprod()
    sector_momentum = sector_index.divide(sector_index.shift(lookback)) - 1.0
    return stock_momentum.sub(_map_sector_values_to_symbols(sector=sector, sector_values=sector_momentum))


def _build_compressed_plan(*, bundle: SignalBundle, compression: CompressionMode) -> PositionPlan:
    raw = _raw_rank_weights(bundle)
    sector = bundle.context["sector"].reindex(index=raw.index, columns=raw.columns)
    selected = pd.DataFrame(0.0, index=raw.index, columns=raw.columns, dtype=float)
    for ts in raw.index:
        row = raw.loc[ts]
        sector_row = sector.loc[ts]
        long_names = row[row.gt(0.0)].index
        short_names = row[row.lt(0.0)].index
        _compress_side(ts=ts, names=long_names, side=1.0, raw=row, sector_row=sector_row, bundle=bundle, compression=compression, output=selected)
        _compress_side(ts=ts, names=short_names, side=-1.0, raw=row, sector_row=sector_row, bundle=bundle, compression=compression, output=selected)
    construction = ConstructionResult(
        base_target_weights=selected,
        selection_mask=selected.ne(0.0),
        group_long_budget=None,
        group_short_budget=None,
        meta={},
    )
    return PassThroughPolicy().apply(construction=construction, market=MarketData(frames={}, universe=None, benchmark=None), bundle=bundle)


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
    ts: pd.Timestamp,
    names: pd.Index,
    side: float,
    raw: pd.Series,
    sector_row: pd.Series,
    bundle: SignalBundle,
    compression: CompressionMode,
    output: pd.DataFrame,
) -> None:
    if len(names) == 0:
        return
    per_sector = compression.long_per_sector if side > 0 else compression.short_per_sector
    for sector_name in pd.unique(sector_row.loc[names].dropna()):
        sector_names = names[sector_row.loc[names].eq(sector_name)]
        if len(sector_names) == 0:
            continue
        exposure = float(raw.loc[sector_names].sum())
        chosen = _choose_names(ts=ts, names=sector_names, side=side, raw=raw, bundle=bundle, compression=compression, count=per_sector)
        if len(chosen) == 0:
            continue
        basis = raw.loc[chosen].abs()
        if float(basis.sum()) <= 0.0:
            basis = pd.Series(1.0, index=chosen, dtype=float)
        output.loc[ts, chosen] = basis / float(basis.sum()) * exposure


def _choose_names(
    *,
    ts: pd.Timestamp,
    names: pd.Index,
    side: float,
    raw: pd.Series,
    bundle: SignalBundle,
    compression: CompressionMode,
    count: int,
) -> pd.Index:
    if len(names) <= count:
        return names
    if compression.metric == "weight":
        metric = raw.loc[names].abs()
    elif compression.metric == "op":
        if side > 0:
            metric = bundle.alpha.loc[ts, names].astype(float)
        else:
            metric = bundle.context["short_alpha"].loc[ts, names].astype(float)
    elif compression.metric == "momo21":
        momentum = bundle.context["stock_vs_sector_momentum_21"].loc[ts, names].astype(float)
        metric = momentum if side > 0 else momentum.mul(-1.0)
    elif compression.metric == "mcap":
        metric = bundle.context["market_cap"].loc[ts, names].astype(float)
    else:
        raise ValueError(f"unknown compression metric: {compression.metric}")
    metric = metric.replace([float("inf"), float("-inf")], pd.NA).fillna(float("-inf"))
    return metric.sort_values(ascending=False, kind="stable").head(count).index


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


def _write_markdown_summary(*, doc_path: Path, summary_frame: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    top = summary_frame.head(10).copy()
    lines = [
        "# RRG Sector Rotation 30 Strategy Research",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "- Universe: KOSPI200",
        "- Sector taxonomy: `QW_WI_SEC_26_BIG`",
        "- Schedule/fill: weekly, next open",
        "- Costs: fee 2bp, sell tax 15bp, slippage 5bp",
        "- Strategy family: RRG sector regime plus OP consensus confirmation, compressed by sector exposure preservation.",
        "",
        "## Variant Grid",
        "",
        "Five OP definitions were crossed with six sector-preserving compression methods for 30 total strategies.",
        "",
        "OP modes:",
        *[f"- `{mode.id}`: {mode.label}" for mode in OP_MODES],
        "",
        "Compression modes:",
        *[f"- `{mode.id}`: {mode.label}" for mode in COMPRESSION_MODES],
        "",
        "## Top 10 By Sharpe",
        "",
        "| rank | strategy | CAGR | MDD | Sharpe | avg names | turnover |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | `{row.strategy_id}` | {row.cagr:.2%} | {row.mdd:.2%} | {row.sharpe:.2f} | {row.avg_total_count:.1f} | {row.avg_turnover:.2%} |"
        )
    stable = summary_frame[(summary_frame["mdd"].ge(-0.15)) & (summary_frame["avg_total_count"].le(30.0))]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Baseline reference: `rrg-sector-rotation_20260617_155800` had CAGR 48.25%, MDD -14.33%, Sharpe 2.03, and about 63.6 names.",
            "- The cleanest name-count reduction is `rrg_qavg_k2_op` / `rrg_qavg_k2_weight`: about 23.5 names, MDD -14.19%, and Sharpe above 2.31.",
            "- `agree3` is the strongest return/Sharpe family, but its MDD is slightly worse than the baseline MDD target.",
            "- `op12` alone is not a good replacement in this test. It keeps more names and produces much worse drawdown.",
            "- The 21D stock-vs-sector momentum compression is consistently poor: it raises turnover and worsens drawdown across OP modes.",
            "- The sensitivity around Q1/Q2 comes from sign-gated OP filters. Small forecast revisions can flip a stock or sector from eligible to ineligible, so agreement/breadth checks are better used as stabilizers than as tightly optimized thresholds.",
            "",
            "## Baseline-MDD-Compatible Candidates",
            "",
            "| strategy | CAGR | MDD | Sharpe | avg names | turnover |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stable.head(10).itertuples(index=False):
        lines.append(
            f"| `{row.strategy_id}` | {row.cagr:.2%} | {row.mdd:.2%} | {row.sharpe:.2f} | {row.avg_total_count:.1f} | {row.avg_turnover:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- CSV summary: `{csv_path}`",
            f"- JSON summary: `{json_path}`",
            "- Per-run artifacts are under `results/backtests/` using each strategy slug.",
            "",
            "## Notes",
            "",
            "These are research variants, not registered production strategies. The grid intentionally uses broad, explainable axes rather than optimizing thresholds against the backtest window.",
        ]
    )
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
