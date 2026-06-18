from __future__ import annotations

import json
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from root import ROOT

from backtesting.analytics import summarize_perf
from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data import DataLoader, LoadRequest, MarketData, ParquetStore
from backtesting.data.benchmarks import benchmark_price_series
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.execution import CostModel, WeeklySchedule
from backtesting.policy.base import PositionPlan
from backtesting.reporting import RunWriter
from backtesting.run import RunConfig, RunReport
from backtesting.strategies import build_strategy
from backtesting.strategies.signal_event_rotation import (
    CONSTRUCTION_MODES,
    EVENT_MODES,
    FLOW_GATES,
    RISK_MODES,
    SCORE_MODES,
    _SignalEventSectorCompressedWeight,
    _event_mask,
    _event_participation,
    _flow_masks,
    _gross_short_for_risk_mode,
    _score_frame,
)
from backtesting.strategies.rrg_sector_rotation import (
    _build_op_rrg_state,
    _build_rrg_context,
    _map_sector_state_to_symbols,
)
from backtesting.signals.base import SignalBundle


START = "2020-01-01"
END = "2026-05-11"
LOAD_START = "2019-01-01"
CAPITAL = 100_000_000.0
FEE = 0.0002
SELL_TAX = 0.0015
SLIPPAGE = 0.0005
RESULT_DIR = ROOT.results_path / "signal_event_research"
SELECTED_DOC = ROOT.root / "docs" / "research" / "signal-event-rotation-grid-results.md"


@dataclass(frozen=True, slots=True)
class Variant:
    id: str
    score_mode: str
    event_mode: str
    flow_gate: str
    construction_mode: str
    risk_mode: str


def main() -> None:
    args = _parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    market = _load_market()
    context = _build_context(market)
    benchmark = context["benchmark_returns"]
    writer = RunWriter(ROOT.results_path / "backtests", write_report_assets=False)
    engine = BacktestEngine(cost=CostModel(fee=FEE, sell_tax=SELL_TAX, slippage=SLIPPAGE))
    rows: list[dict[str, object]] = []
    variants = variant_grid()
    if args.limit is not None:
        variants = variants[: args.limit]
    for idx, variant in enumerate(variants, start=1):
        strategy_id = f"signal_event_rotation_{variant.id}"
        existing = _load_existing_metric(strategy_id=strategy_id, variant=variant, benchmark=benchmark)
        if existing is not None:
            print(f"[{idx}/{len(variants)}] skipping {strategy_id}")
            rows.append(existing)
            continue
        print(f"[{idx}/{len(variants)}] running {strategy_id}")
        config = _config_for_variant(strategy_id=strategy_id, variant=variant)
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
        output_dir = writer.write(RunReport(config=config, summary=summary, result=result, position_plan=plan))
        rows.append(_metric_row(strategy_id=strategy_id, variant=variant, result=result, benchmark=benchmark, output_dir=output_dir))

    summary = pd.DataFrame(rows)
    summary["robust_score"] = _robust_score(summary)
    summary = summary.sort_values(["robust_score", "sharpe", "mdd", "cagr"], ascending=[False, False, False, False])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULT_DIR / f"signal_event_grid_summary_{stamp}.csv"
    json_path = RESULT_DIR / f"signal_event_grid_summary_{stamp}.json"
    summary.to_csv(csv_path, index=False)
    json_path.write_text(summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    _write_selected_doc(summary=summary, csv_path=csv_path, json_path=json_path)
    print(summary.head(15).to_string(index=False))
    print(json.dumps({"summary": str(csv_path), "json": str(json_path), "count": len(summary)}, ensure_ascii=False, indent=2))


def variant_grid() -> list[Variant]:
    variants: list[Variant] = []
    for score_idx, score_mode in enumerate(SCORE_MODES):
        for event_idx, event_mode in enumerate(EVENT_MODES):
            for flow_idx, flow_gate in enumerate(FLOW_GATES):
                for construction_idx, construction_mode in enumerate(CONSTRUCTION_MODES):
                    risk_mode = RISK_MODES[(score_idx + event_idx + flow_idx + construction_idx) % len(RISK_MODES)]
                    variants.append(
                        Variant(
                            id=f"sev_{score_mode}_{event_mode}_{flow_gate}_{construction_mode}_{risk_mode}",
                            score_mode=score_mode,
                            event_mode=event_mode,
                            flow_gate=flow_gate,
                            construction_mode=construction_mode,
                            risk_mode=risk_mode,
                        )
                    )
    return variants


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed 500-candidate KOSPI200 signal-event rotation grid.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N variants for smoke testing.")
    return parser.parse_args()


def _config_for_variant(*, strategy_id: str, variant: Variant) -> RunConfig:
    return RunConfig(
        start=START,
        end=END,
        capital=CAPITAL,
        strategy="signal_event_rotation",
        strategy_params={
            "score_mode": variant.score_mode,
            "event_mode": variant.event_mode,
            "flow_gate": variant.flow_gate,
            "construction_mode": variant.construction_mode,
            "risk_mode": variant.risk_mode,
            "lookback": 20,
            "flow_lookback": 20,
            "high_lookback": 252,
            "participation_steps": 3,
        },
        name=strategy_id,
        schedule="weekly",
        fill_mode="next_open",
        fee=FEE,
        sell_tax=SELL_TAX,
        slippage=SLIPPAGE,
        borrow_fee_annual=0.0,
        short_cash_collateral_ratio=1.0,
        use_k200=True,
        allow_fractional=True,
        warmup_days=365,
    )


def _load_existing_metric(*, strategy_id: str, variant: Variant, benchmark: pd.Series) -> dict[str, object] | None:
    candidates: list[Path] = []
    for run_dir in ROOT.results_path.joinpath("backtests").glob("signal_event_rotation_*"):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if config.get("name") == strategy_id:
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
    strategy = build_strategy("signal_event_rotation")
    datasets = list(dict.fromkeys((*strategy.datasets, DatasetId.QW_ADJ_O)))
    loader = DataLoader(DataCatalog.default(), ParquetStore(ROOT.parquet_path))
    return loader.load(LoadRequest(datasets=datasets, start=LOAD_START, end=END))


def _benchmark_returns(market: MarketData) -> pd.Series:
    benchmark = benchmark_price_series(market.frames["benchmark"], "IKS200").astype(float)
    return benchmark.pct_change(fill_method=None).fillna(0.0).loc[START:END].rename("KOSPI200")


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
    op = market.frames["op_fwd"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
    op_state = _build_op_rrg_state(
        op=op,
        sector=sector,
        membership=k200,
        medium_lookback=126,
        momentum_lookback=21,
        short_lookback=42,
        transition_threshold=0.005,
    )
    price_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=price_state)
    op_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=op_state)
    scores = {
        mode: _score_frame(
            frames=market.frames,
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=20,
            mode=mode,
        )
        for mode in SCORE_MODES
    }
    flows = {
        gate: _flow_masks(
            market=market,
            like=close,
            market_cap=market_cap,
            lookback=20,
            gate=gate,
        )
        for gate in FLOW_GATES
    }
    events = {
        (score_mode, event_mode): _event_mask(
            mode=event_mode,
            score=score,
            close=close,
            price_state=price_by_symbol,
            op_state=op_by_symbol,
            lookback=20,
            high_lookback=252,
        )
        for score_mode, score in scores.items()
        for event_mode in EVENT_MODES
    }
    return {
        "close": close,
        "open": open_,
        "tradable": close.notna() & k200,
        "k200": k200,
        "sector": sector,
        "price_by_symbol": price_by_symbol,
        "op_by_symbol": op_by_symbol,
        "scores": scores,
        "flows": flows,
        "events": events,
        "benchmark_returns": benchmark.pct_change(fill_method=None).fillna(0.0).loc[START:END].rename("KOSPI200"),
    }


def _build_plan(*, context: dict[str, object], variant: Variant) -> PositionPlan:
    score = context["scores"][variant.score_mode]
    k200 = context["k200"]
    price_by_symbol = context["price_by_symbol"]
    op_by_symbol = context["op_by_symbol"]
    flow_ok, short_flow_ok = context["flows"][variant.flow_gate]
    long_regime = price_by_symbol.isin(("Leading", "Improving", "Weakening")) & op_by_symbol.isin(("Leading", "Improving"))
    short_regime = price_by_symbol.eq("Lagging") & op_by_symbol.isin(("Lagging", "Weakening"))
    long_hold = long_regime & score.gt(0.0) & flow_ok & k200
    long_event = context["events"][(variant.score_mode, variant.event_mode)] & long_hold
    participation = _event_participation(
        event=long_event.fillna(False).astype(bool),
        hold=long_hold.fillna(False).astype(bool),
        steps=3,
    )
    short_hold = short_regime & score.lt(0.0) & short_flow_ok & k200
    bundle = SignalBundle(
        alpha=score.where(long_hold).mul(participation),
        context={
            "tradable": k200,
            "entry_mask": long_event.fillna(False).astype(bool),
            "hold_mask": long_hold.fillna(False).astype(bool),
            "short_entry_mask": short_hold.fillna(False).astype(bool),
            "short_hold_mask": short_hold.fillna(False).astype(bool),
            "short_alpha": score.mul(-1.0).where(short_hold),
            "sector": context["sector"],
            "participation": participation,
        },
        meta={},
    )
    construction = _SignalEventSectorCompressedWeight(
        gross_long=1.0,
        gross_short=_gross_short_for_risk_mode(variant.risk_mode),
        construction_mode=variant.construction_mode,
    ).build(bundle)
    from backtesting.policy.pass_through import PassThroughPolicy

    return PassThroughPolicy().apply(construction=construction, market=MarketData(frames={}, universe=None, benchmark=None), bundle=bundle)


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
        **asdict(variant),
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


def _robust_score(frame: pd.DataFrame) -> pd.Series:
    mdd_penalty = frame["mdd"].abs().clip(lower=0.05)
    turnover_penalty = frame["avg_turnover"].clip(lower=0.01)
    return (
        frame["sharpe"].fillna(0.0)
        + frame["cagr"].fillna(0.0)
        + frame["monthly_bm_win_rate"].fillna(0.0)
        - mdd_penalty
        - turnover_penalty.mul(0.25)
    )


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
    return PositionPlan(
        target_weights=plan.target_weights.loc[START:END].copy(),
        bucket_ledger=ledger,
        bucket_meta=plan.bucket_meta,
        validation=plan.validation,
    )


def _write_selected_doc(*, summary: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    top = summary.head(10).copy()
    lines = [
        "# Signal Event Rotation 500-Candidate Results",
        "",
        "## Setup",
        "",
        "- Universe: KOSPI200.",
        "- Execution: weekly rebalance, next-open fill, 2bp fee, 15bp sell tax, 5bp slippage.",
        "- Candidate count: 500 fixed combinations. No optimized thresholds.",
        "- Economic rationale: OP consensus events, sector price/OP cycle confirmation, and investor-flow confirmation.",
        "",
        "## Selected Candidates",
        "",
        "| rank | strategy | CAGR | MDD | Sharpe | BM monthly win | turnover | avg names |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | `{row.strategy_id}` | {row.cagr:.2%} | {row.mdd:.2%} | {row.sharpe:.2f} | "
            f"{row.monthly_bm_win_rate:.2%} | {row.avg_turnover:.2%} | {row.avg_total_count:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- CSV: `{_repo_relative(csv_path)}`",
            f"- JSON: `{_repo_relative(json_path)}`",
            "- Per-run artifacts: `results/backtests/signal_event_rotation_*`",
        ]
    )
    SELECTED_DOC.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.root.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
