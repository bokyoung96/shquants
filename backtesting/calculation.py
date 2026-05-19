from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import AbstractContextManager
from typing import Any, Callable, Protocol

import pandas as pd

from .analytics import summarize_perf
from .catalog import DatasetId
from .data import DataLoader, LoadRequest, MarketData
from .engine import BacktestEngine, BacktestResult
from .execution import CostModel
from .policy.base import PositionPlan
from .specs import ExecutionSpec, ResolvedExecutionSpec, ScheduleSpec
from .universe import UniverseSpec


class EffectiveRunConfig(Protocol):
    start: str
    end: str
    capital: float
    fee: float
    sell_tax: float
    slippage: float
    borrow_fee_annual: float
    short_cash_collateral_ratio: float
    fill_mode: str
    allow_fractional: bool
    universe_id: str | None
    warmup_days: int


class StageTimer(Protocol):
    def stage(self, name: str) -> AbstractContextManager[None]: ...


@dataclass(slots=True)
class BacktestCalculationResult:
    config: EffectiveRunConfig
    summary: dict[str, float]
    result: BacktestResult
    position_plan: PositionPlan
    resolved_spec: ResolvedExecutionSpec
    execution_resolution: dict[str, object]


@dataclass(slots=True)
class BacktestCalculationContext:
    loader: DataLoader
    ensure_parquet: Callable[[list[DatasetId]], None]
    resolve_universe_spec: Callable[[ExecutionSpec], UniverseSpec | None]
    resolve_effective_config: Callable[[ExecutionSpec, UniverseSpec | None], EffectiveRunConfig]
    resolve_load_start: Callable[[str, int], str]
    resolve_universe: Callable[[MarketData, UniverseSpec | None], pd.DataFrame | None]
    schedule_from_spec: Callable[..., Any]
    trim_result_to_display_range: Callable[[BacktestResult], BacktestResult]
    trim_plan_to_display_range: Callable[[PositionPlan], PositionPlan]


@dataclass(slots=True)
class BacktestCalculationAdapters:
    build_strategy: Callable[..., Any]
    build_position_plan: Callable[[ExecutionSpec, MarketData], PositionPlan]
    build_target_weight_plan: Callable[[ExecutionSpec, MarketData], tuple[PositionPlan, dict[str, object]]]
    get_hook: Callable[[str], Any]
    engine_factory: Callable[..., BacktestEngine]
    validate_position_plan: Callable[[PositionPlan], None]


@dataclass(slots=True)
class BacktestCalculation:
    context: BacktestCalculationContext
    adapters: BacktestCalculationAdapters
    timer: StageTimer

    def run(self, resolved_spec: ResolvedExecutionSpec) -> BacktestCalculationResult:
        spec = resolved_spec.execution
        universe_spec = self.context.resolve_universe_spec(spec)
        effective_config = self.context.resolve_effective_config(spec, universe_spec)

        with self.timer.stage("data_load"):
            self.context.ensure_parquet(list(resolved_spec.dataset_ids))
            market = self.context.loader.load(
                LoadRequest(
                    datasets=list(resolved_spec.dataset_ids),
                    start=self.context.resolve_load_start(effective_config.start, effective_config.warmup_days),
                    end=effective_config.end,
                    universe_id=effective_config.universe_id,
                )
            )
            market.universe = self.context.resolve_universe(market, universe_spec)

        with self.timer.stage("plan_build"):
            plan, schedule_input, extra_tradable, resolution_meta, resolved_spec = self._build_plan(
                spec=spec,
                resolved_spec=resolved_spec,
                market=market,
                universe_spec=universe_spec,
            )
            _validate_shorting_enabled_for_plan(spec, plan)
            self.adapters.validate_position_plan(plan)
            weights = plan.target_weights
            if schedule_input is None:
                schedule_input = self.context.schedule_from_spec(resolved_spec, weights=weights)
            close = market.frames["close"]
            tradable = close.notna()
            if market.universe is not None:
                tradable = tradable & market.universe.reindex_like(close).fillna(False).astype(bool)
            if extra_tradable is not None:
                tradable = tradable & extra_tradable.reindex_like(close).fillna(False).astype(bool)

            engine = self.adapters.engine_factory(
                cost=CostModel(
                    fee=effective_config.fee,
                    sell_tax=effective_config.sell_tax,
                    slippage=effective_config.slippage,
                    borrow_fee_annual=effective_config.borrow_fee_annual,
                    short_cash_collateral_ratio=effective_config.short_cash_collateral_ratio,
                )
            )

        with self.timer.stage("engine_run"):
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
            result = self.context.trim_result_to_display_range(
                result,
                start=effective_config.start,
                end=effective_config.end,
            )
            plan = self.context.trim_plan_to_display_range(plan, start=effective_config.start, end=effective_config.end)
            if result.equity.empty:
                raise ValueError(
                    f"no backtest rows remain after trimming to display range {effective_config.start}..{effective_config.end}"
                )

            summary = summarize_perf(result.returns)
            summary["final_equity"] = float(result.equity.iloc[-1])
            summary["avg_turnover"] = float(result.turnover.mean())

        return BacktestCalculationResult(
            config=effective_config,
            summary=summary,
            result=result,
            position_plan=plan,
            resolved_spec=resolved_spec,
            execution_resolution={
                "spec_source": spec.spec_source,
                "preset_id": spec.preset_id,
                "hook_id": resolved_spec.hook_id,
                "resolution_notes": list(resolved_spec.resolution_notes),
                "fallbacks_applied": list(spec.data_policy.fallbacks_applied),
                **resolution_meta,
            },
        )

    def _build_plan(
        self,
        *,
        spec: ExecutionSpec,
        resolved_spec: ResolvedExecutionSpec,
        market: MarketData,
        universe_spec: UniverseSpec | None,
    ) -> tuple[PositionPlan, Any, pd.DataFrame | None, dict[str, object], ResolvedExecutionSpec]:
        if spec.target_weights is not None:
            plan, metadata = self.adapters.build_target_weight_plan(spec, market)
            return plan, None, None, metadata, resolved_spec

        if spec.weight_source.kind == "hook":
            hook = self.adapters.get_hook(resolved_spec.hook_id or "")
            hook_plan = hook.build_plan(market=market, resolved_spec=resolved_spec, universe_spec=universe_spec)
            if hook_plan.schedule is not None:
                rebalance_dates = tuple(ts.date().isoformat() for ts in hook_plan.schedule[hook_plan.schedule].index)
                resolved_spec = replace(
                    resolved_spec,
                    schedule=ScheduleSpec(kind="custom_dates", dates=rebalance_dates),
                )
            return (
                hook_plan.position_plan,
                hook_plan.schedule,
                hook_plan.tradable,
                hook_plan.metadata,
                resolved_spec,
            )
        if spec.uses_composable_plan:
            return (
                self.adapters.build_position_plan(spec, market),
                None,
                None,
                {"plan_source": "selection_weighting_position_policy"},
                resolved_spec,
            )

        strategy = self.adapters.build_strategy(
            spec.strategy,
            **_strategy_kwargs(spec),
        )
        return strategy.build_plan(market), None, None, {}, resolved_spec


def _validate_shorting_enabled_for_plan(spec: ExecutionSpec, plan: PositionPlan) -> None:
    if bool(plan.target_weights.lt(0.0).any().any()) and not spec.shorting.enabled:
        raise ValueError("negative target weights require shorting.enabled = true")


def _strategy_kwargs(spec: ExecutionSpec) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "top_n": spec.top_n,
        "lookback": spec.lookback,
        "flow_lookback": spec.flow_lookback,
        "momentum_lookback": spec.momentum_lookback,
        "liquidity_lookback": spec.liquidity_lookback,
        "momentum_weight": spec.momentum_weight,
    }
    kwargs.update(spec.strategy_params)
    return kwargs
