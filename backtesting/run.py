from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from root import ROOT

from .analytics import summarize_perf
from .catalog import DataCatalog, DatasetId
from .data import DataLoader, LoadRequest, ParquetStore
from .engine import BacktestEngine, BacktestResult
from .execution import CostModel, CustomSchedule, DailySchedule, MonthlySchedule, WeeklySchedule
from .ingest import IngestJob
from .policy.base import PositionPlan
from .reporting import RunWriter
from .specs import ExecutionSpec, ScheduleSpec, get_hook, get_preset, load_execution_spec, resolve_execution_spec
from .strategies import build_strategy, list_strategies
from .universe import UniverseRegistry, UniverseSpec
from .validation import validate_position_plan


@dataclass(frozen=True, slots=True)
class RunConfig:
    start: str
    end: str
    capital: float = 100_000_000.0
    strategy: str = "momentum"
    name: str | None = None
    top_n: int = 20
    lookback: int = 20
    flow_lookback: int = 20
    momentum_lookback: int = 60
    liquidity_lookback: int = 20
    momentum_weight: float = 0.5
    schedule: str = "monthly"
    fill_mode: str = "next_open"
    fee: float = 0.0
    sell_tax: float = 0.0
    slippage: float = 0.0
    use_k200: bool = True
    allow_fractional: bool = True
    universe_id: str | None = None
    benchmark_code: str | None = None
    benchmark_name: str | None = None
    benchmark_dataset: str | None = None
    warmup_days: int = 0


@dataclass(slots=True)
class RunReport:
    config: RunConfig
    summary: dict[str, float]
    result: BacktestResult
    position_plan: PositionPlan | None = None
    output_dir: Path | None = None
    resolved_spec: object | None = None
    execution_resolution: dict[str, object] | None = None


class BacktestRunner:
    def __init__(
        self,
        *,
        catalog: DataCatalog | None = None,
        raw_dir: Path | None = None,
        parquet_dir: Path | None = None,
        result_dir: Path | None = None,
        universe_registry: UniverseRegistry | None = None,
    ) -> None:
        self.catalog = catalog or DataCatalog.default()
        self.universe_registry = universe_registry or UniverseRegistry.default()
        self.raw_dir = raw_dir or ROOT.raw_path
        self.parquet_dir = parquet_dir or ROOT.parquet_path
        self.result_dir = result_dir or (ROOT.results_path / "backtests")
        self.store = ParquetStore(self.parquet_dir)
        self.loader = DataLoader(self.catalog, self.store)
        self.ingest = IngestJob(self.catalog, self.raw_dir, self.parquet_dir)
        self.writer = RunWriter(self.result_dir)

    def run(self, config: RunConfig) -> RunReport:
        return self.run_spec(self.resolve_spec_from_config(config))

    def run_resolved_cli(
        self,
        *,
        preset_id: str | None = None,
        spec_path: str | None = None,
        config: RunConfig | None = None,
    ) -> RunReport:
        selected = sum(bool(item) for item in (preset_id, spec_path, config))
        if selected != 1:
            raise ValueError("choose exactly one of preset_id, spec_path, or config")
        if preset_id is not None:
            spec = get_preset(preset_id)
        elif spec_path is not None:
            spec = load_execution_spec(spec_path)
        else:
            assert config is not None
            spec = self._execution_spec_from_config(config)
        return self.run_spec(self.resolve_spec(spec))

    def resolve_spec_from_config(self, config: RunConfig):
        return self.resolve_spec(self._execution_spec_from_config(config))

    def resolve_spec(self, spec: ExecutionSpec):
        universe_spec = self._resolve_universe_spec_from_spec(spec)
        return resolve_execution_spec(
            spec,
            catalog=self.catalog,
            parquet_dir=self.parquet_dir,
            raw_dir=self.raw_dir,
            universe_spec=universe_spec,
        )

    def run_spec(self, resolved_spec) -> RunReport:
        spec = resolved_spec.execution
        universe_spec = self._resolve_universe_spec_from_spec(spec)
        effective_config = self._resolve_effective_config_from_spec(spec, universe_spec)

        self._ensure_parquet(list(resolved_spec.dataset_ids))
        market = self.loader.load(
            LoadRequest(
                datasets=list(resolved_spec.dataset_ids),
                start=self._resolve_load_start(effective_config.start, effective_config.warmup_days),
                end=effective_config.end,
                universe_id=effective_config.universe_id,
            )
        )
        market.universe = self._universe(market, universe_spec)

        if spec.weight_source.kind == "hook":
            hook = get_hook(resolved_spec.hook_id or "")
            hook_plan = hook.build_plan(market=market, resolved_spec=resolved_spec, universe_spec=universe_spec)
            plan = hook_plan.position_plan
            schedule_input = hook_plan.schedule
            extra_tradable = hook_plan.tradable
            resolution_meta = hook_plan.metadata
            if hook_plan.schedule is not None:
                rebalance_dates = tuple(ts.date().isoformat() for ts in hook_plan.schedule[hook_plan.schedule].index)
                resolved_spec = replace(
                    resolved_spec,
                    schedule=ScheduleSpec(kind="custom_dates", dates=rebalance_dates),
                )
        else:
            strategy = build_strategy(
                spec.strategy,
                top_n=spec.top_n,
                lookback=spec.lookback,
                flow_lookback=spec.flow_lookback,
                momentum_lookback=spec.momentum_lookback,
                liquidity_lookback=spec.liquidity_lookback,
                momentum_weight=spec.momentum_weight,
            )
            plan = strategy.build_plan(market)
            schedule_input = self._schedule_from_spec(resolved_spec)
            extra_tradable = None
            resolution_meta = {}

        validate_position_plan(plan)
        weights = plan.target_weights
        close = market.frames["close"]
        tradable = close.notna()
        if market.universe is not None:
            tradable = tradable & market.universe.reindex_like(close).fillna(False).astype(bool)
        if extra_tradable is not None:
            tradable = tradable & extra_tradable.reindex_like(close).fillna(False).astype(bool)

        engine = BacktestEngine(
            cost=CostModel(
                fee=effective_config.fee,
                sell_tax=effective_config.sell_tax,
                slippage=effective_config.slippage,
            )
        )
        result = engine.run(
            close=close,
            open=market.frames.get("open"),
            weights=weights,
            capital=effective_config.capital,
            tradable=tradable,
            schedule=schedule_input,
            fill_mode=effective_config.fill_mode,
            allow_fractional=effective_config.allow_fractional,
        )
        result = self._trim_result_to_display_range(result, start=effective_config.start, end=effective_config.end)
        plan = self._trim_plan_to_display_range(plan, start=effective_config.start, end=effective_config.end)
        if result.equity.empty:
            raise ValueError(
                f"no backtest rows remain after trimming to display range {effective_config.start}..{effective_config.end}"
            )

        summary = summarize_perf(result.returns)
        summary["final_equity"] = float(result.equity.iloc[-1])
        summary["avg_turnover"] = float(result.turnover.mean())
        report = RunReport(config=effective_config, summary=summary, result=result, position_plan=plan)
        report.resolved_spec = resolved_spec
        report.execution_resolution = {
            "spec_source": spec.spec_source,
            "preset_id": spec.preset_id,
            "hook_id": resolved_spec.hook_id,
            "resolution_notes": list(resolved_spec.resolution_notes),
            "fallbacks_applied": list(spec.data_policy.fallbacks_applied),
            **resolution_meta,
        }
        report.output_dir = self.writer.write(report)
        return report

    def _execution_spec_from_config(self, config: RunConfig) -> ExecutionSpec:
        return ExecutionSpec(
            start=config.start,
            end=config.end,
            capital=config.capital,
            strategy=config.strategy,
            name=config.name,
            top_n=config.top_n,
            lookback=config.lookback,
            flow_lookback=config.flow_lookback,
            momentum_lookback=config.momentum_lookback,
            liquidity_lookback=config.liquidity_lookback,
            momentum_weight=config.momentum_weight,
            schedule=ScheduleSpec(kind="named", name=config.schedule),
            fill_mode=config.fill_mode,
            fee=config.fee,
            sell_tax=config.sell_tax,
            slippage=config.slippage,
            use_k200=config.use_k200,
            allow_fractional=config.allow_fractional,
            universe_id=config.universe_id,
            benchmark_code=config.benchmark_code,
            benchmark_name=config.benchmark_name,
            benchmark_dataset=config.benchmark_dataset,
            warmup_days=config.warmup_days,
            spec_source="cli",
        )

    def _ensure_parquet(self, dataset_ids: list[DatasetId]) -> None:
        for dataset_id in dataset_ids:
            stem = self.catalog.get(dataset_id).stem
            path = self.parquet_dir / f"{stem}.parquet"
            if not path.exists():
                self.ingest.run(dataset_id)

    def _resolve_universe_spec_from_spec(self, spec: ExecutionSpec) -> UniverseSpec | None:
        if spec.universe_id is not None:
            return self.universe_registry.get(spec.universe_id)
        if spec.use_k200:
            return self.universe_registry.get("legacy_k200")
        return None

    @staticmethod
    def _resolve_effective_config_from_spec(spec: ExecutionSpec, universe_spec: UniverseSpec | None) -> RunConfig:
        config = RunConfig(
            start=spec.start,
            end=spec.end,
            capital=spec.capital,
            strategy=spec.strategy,
            name=spec.name,
            top_n=spec.top_n,
            lookback=spec.lookback,
            flow_lookback=spec.flow_lookback,
            momentum_lookback=spec.momentum_lookback,
            liquidity_lookback=spec.liquidity_lookback,
            momentum_weight=spec.momentum_weight,
            schedule=spec.schedule.name or "custom",
            fill_mode=spec.fill_mode,
            fee=spec.fee,
            sell_tax=spec.sell_tax,
            slippage=spec.slippage,
            use_k200=spec.use_k200,
            allow_fractional=spec.allow_fractional,
            universe_id=spec.universe_id,
            benchmark_code=spec.benchmark_code,
            benchmark_name=spec.benchmark_name,
            benchmark_dataset=spec.benchmark_dataset,
            warmup_days=spec.warmup_days,
        )
        if universe_spec is None:
            return replace(
                config,
                benchmark_code=config.benchmark_code or "IKS200",
                benchmark_name=config.benchmark_name or "KOSPI200",
                benchmark_dataset=config.benchmark_dataset or "qw_BM",
            )
        universe_id = config.universe_id
        if universe_id is None and universe_spec.id != "legacy_k200":
            universe_id = universe_spec.id
        return replace(
            config,
            use_k200=universe_spec.id == "legacy_k200" if config.universe_id is not None else config.use_k200,
            universe_id=universe_id,
            benchmark_code=config.benchmark_code or universe_spec.default_benchmark_code,
            benchmark_name=config.benchmark_name or universe_spec.default_benchmark_name,
            benchmark_dataset=config.benchmark_dataset or universe_spec.default_benchmark_dataset,
        )

    @staticmethod
    def _universe(market, universe_spec: UniverseSpec | None) -> pd.DataFrame | None:
        if universe_spec is None:
            return None
        membership_key = "k200_yn" if universe_spec.membership_dataset is DatasetId.QW_K200_YN else "universe_membership"
        membership = market.frames.get(membership_key)
        if membership is None:
            return None
        return membership.fillna(0).astype(bool)

    @staticmethod
    def _schedule_from_spec(resolved_spec):
        schedule = resolved_spec.schedule
        if schedule.kind == "named":
            if schedule.name == "daily":
                return DailySchedule()
            if schedule.name == "weekly":
                return WeeklySchedule()
            if schedule.name == "monthly":
                return MonthlySchedule()
            raise ValueError(f"unsupported schedule: {schedule.name}")
        if schedule.kind == "custom_dates":
            return CustomSchedule(pd.to_datetime(list(schedule.dates)))
        raise ValueError(f"unsupported schedule kind: {schedule.kind}")

    @staticmethod
    def _resolve_load_start(start: str, warmup_days: int) -> str:
        if warmup_days <= 0:
            return start
        return (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).date().isoformat()

    @staticmethod
    def _trim_result_to_display_range(result: BacktestResult, *, start: str, end: str) -> BacktestResult:
        return BacktestResult(
            equity=result.equity.loc[start:end].copy(),
            returns=result.returns.loc[start:end].copy(),
            weights=result.weights.loc[start:end].copy(),
            qty=result.qty.loc[start:end].copy(),
            turnover=result.turnover.loc[start:end].copy(),
        )

    @staticmethod
    def _trim_plan_to_display_range(plan: PositionPlan, *, start: str, end: str) -> PositionPlan:
        trimmed_weights = plan.target_weights.loc[start:end].copy()
        trimmed_ledger = plan.bucket_ledger
        ledger_changed = False
        if not trimmed_ledger.empty and "date" in trimmed_ledger.columns:
            date_values = pd.to_datetime(trimmed_ledger["date"])
            mask = (date_values >= pd.Timestamp(start)) & (date_values <= pd.Timestamp(end))
            ledger_changed = not bool(mask.all())
            trimmed_ledger = trimmed_ledger.loc[mask].reset_index(drop=True)
        if trimmed_weights.equals(plan.target_weights) and not ledger_changed:
            return plan
        return PositionPlan(
            target_weights=trimmed_weights,
            bucket_ledger=trimmed_ledger,
            bucket_meta=plan.bucket_meta,
            validation=plan.validation,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run registered backtests.")
    parser.add_argument("--strategy", choices=list_strategies(), default="momentum")
    parser.add_argument("--name")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--capital", type=float, default=100_000_000.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--schedule", choices=("daily", "weekly", "monthly"), default="monthly")
    parser.add_argument("--fill-mode", choices=("close", "next_open"), default="next_open")
    parser.add_argument("--fee", type=float, default=0.0)
    parser.add_argument("--sell-tax", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--flow-lookback", type=int, default=20)
    parser.add_argument("--momentum-lookback", type=int, default=60)
    parser.add_argument("--liquidity-lookback", type=int, default=20)
    parser.add_argument("--momentum-weight", type=float, default=0.5)
    parser.add_argument("--out-root")
    parser.add_argument("--universe", choices=("kosdaq150",), dest="universe_id")
    parser.add_argument("--no-k200", action="store_true")
    parser.add_argument("--no-fractional", action="store_true")
    parser.add_argument("--preset")
    parser.add_argument("--spec")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.preset and args.spec:
        raise SystemExit("choose exactly one advanced execution source: --preset or --spec")
    config = None
    if args.preset is None and args.spec is None:
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required unless --preset or --spec is provided")
        config = RunConfig(
            start=args.start,
            end=args.end,
            capital=args.capital,
            strategy=args.strategy,
            name=args.name,
            top_n=args.top_n,
            lookback=args.lookback,
            flow_lookback=args.flow_lookback,
            momentum_lookback=args.momentum_lookback,
            liquidity_lookback=args.liquidity_lookback,
            momentum_weight=args.momentum_weight,
            schedule=args.schedule,
            fill_mode=args.fill_mode,
            fee=args.fee,
            sell_tax=args.sell_tax,
            slippage=args.slippage,
            universe_id=args.universe_id,
            use_k200=not args.no_k200,
            allow_fractional=not args.no_fractional,
        )
    runner = BacktestRunner(result_dir=Path(args.out_root) if args.out_root else None)
    report = runner.run_resolved_cli(preset_id=args.preset, spec_path=args.spec, config=config)
    payload = {
        "config": asdict(report.config),
        "summary": report.summary,
        "output_dir": None if report.output_dir is None else str(report.output_dir),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(report.result.equity.tail())


if __name__ == "__main__":
    main()
