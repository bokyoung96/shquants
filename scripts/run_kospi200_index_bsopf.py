from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtesting.analytics import summarize_perf
from backtesting.catalog import DataCatalog
from backtesting.data import DataLoader, LoadRequest, ParquetStore
from backtesting.engine import BacktestEngine
from backtesting.execution import CostModel, WeeklySchedule, MonthlySchedule, DailySchedule
from backtesting.reporting import RunWriter
from backtesting.strategies import build_strategy
from backtesting.universe import UniverseRegistry
from root import ROOT


def schedule_obj(name: str):
    if name == 'daily':
        return DailySchedule()
    if name == 'weekly':
        return WeeklySchedule()
    if name == 'monthly':
        return MonthlySchedule()
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--strategy', required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--schedule', choices=('daily', 'weekly', 'monthly'), default='weekly')
    p.add_argument('--fill-mode', choices=('close',), default='close')
    p.add_argument('--capital', type=float, default=100_000_000.0)
    p.add_argument('--fee', type=float, default=0.0)
    p.add_argument('--sell-tax', type=float, default=0.0)
    p.add_argument('--slippage', type=float, default=0.0)
    p.add_argument('--name')
    p.add_argument('--out-root')
    args = p.parse_args()

    catalog = DataCatalog.default()
    store = ParquetStore(ROOT.parquet_path)
    loader = DataLoader(catalog, store)
    universe_registry = UniverseRegistry.default()
    strategy = build_strategy(args.strategy)
    universe_spec = universe_registry.get('legacy_k200')
    datasets = list(dict.fromkeys([*strategy.datasets]))
    market = loader.load(LoadRequest(datasets=datasets, start=args.start, end=args.end, universe_id=None))
    market.universe = market.frames['k200_yn'].fillna(0).astype(bool)

    bundle = strategy.signal_producer.build(market)
    plan = strategy.build_plan(market)

    benchmark_close = bundle.context['benchmark_close'].astype(float)
    tradable = bundle.context['tradable'].astype(bool)
    engine = BacktestEngine(cost=CostModel(fee=args.fee, sell_tax=args.sell_tax, slippage=args.slippage))
    result = engine.run(
        close=benchmark_close,
        weights=plan.target_weights,
        capital=args.capital,
        tradable=tradable,
        schedule=schedule_obj(args.schedule),
        fill_mode='close',
        allow_fractional=True,
    )

    summary = summarize_perf(result.returns)
    summary['final_equity'] = float(result.equity.iloc[-1])
    summary['avg_turnover'] = float(result.turnover.mean())

    from backtesting.run import RunConfig, RunReport
    config = RunConfig(
        start=args.start,
        end=args.end,
        capital=args.capital,
        strategy=args.strategy,
        name=args.name,
        schedule=args.schedule,
        fill_mode='close',
        fee=args.fee,
        sell_tax=args.sell_tax,
        slippage=args.slippage,
        benchmark_code='IKS200',
        benchmark_name='KOSPI200',
        benchmark_dataset='qw_BM',
    )
    report = RunReport(config=config, summary=summary, result=result, position_plan=plan)
    writer = RunWriter(Path(args.out_root) if args.out_root else (ROOT.results_path / 'backtests'))
    report.output_dir = writer.write(report)
    print(json.dumps({'config': asdict(config), 'summary': summary, 'output_dir': str(report.output_dir)}, ensure_ascii=False, indent=2))
    print(result.equity.tail())


if __name__ == '__main__':
    main()
