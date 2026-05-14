from __future__ import annotations

import pandas as pd

from backtesting.construction import LongShortTopBottom, SectorNeutralTopBottom
from backtesting.data import MarketData
from backtesting.execution.schedule import CustomSchedule, DailySchedule, MonthlySchedule, WeeklySchedule
from backtesting.features import build_features
from backtesting.policy import PositionPlan, build_position_plan_from_spec
from backtesting.selection import build_selection, selection_fields
from backtesting.signals.base import SignalBundle
from backtesting.specs.models import (
    ExecutionSpec,
    PortfolioShapeSpec,
    PositionPolicySpec,
    ScheduleEvaluationSpec,
    SelectionSpec,
    WeightingSpec,
)
from backtesting.weighting import build_weights, weighting_fields


def build_position_plan_from_execution_spec(spec: ExecutionSpec, market: MarketData) -> PositionPlan:
    if spec.selection is None:
        raise ValueError("spec-driven plan requires selection")

    weighting = spec.weighting or WeightingSpec(kind="equal_weight")
    position_policy = spec.position_policy or PositionPolicySpec(kind="pass_through")
    fields = tuple(
        dict.fromkeys(
            (
                *selection_fields(spec.selection),
                *weighting_fields(weighting),
                *_portfolio_shape_fields(spec.portfolio_shape),
                *_shorting_fields(spec),
            )
        )
    )
    features = build_features(market, fields)
    if spec.portfolio_shape is not None and spec.portfolio_shape.kind in {"long_short", "sector_neutral"}:
        construction = _build_portfolio_shape_construction(
            selection_spec=spec.selection,
            portfolio_shape=spec.portfolio_shape,
            features=features,
        )
        base_weights = construction.base_target_weights
        selection = construction.selection_mask
        if market.universe is not None:
            universe = market.universe.reindex(index=selection.index, columns=selection.columns)
            universe = universe.where(universe.notna(), False).astype(bool)
            base_weights = base_weights.where(universe, 0.0).astype(float)
            selection = selection.where(universe, False).astype(bool)
    else:
        selection = build_selection(spec.selection, features)

        if market.universe is not None:
            universe = market.universe.reindex(index=selection.index, columns=selection.columns)
            universe = universe.where(universe.notna(), False).astype(bool)
            selection = selection.where(selection.notna(), False).astype(bool) & universe

        base_weights = build_weights(weighting, selection, features)
    base_weights, selection = _apply_shorting(
        spec,
        base_weights=base_weights,
        selection=selection,
        features=features,
    )
    base_weights, selection = _apply_scheduled_evaluation(
        spec,
        base_weights=base_weights,
        selection=selection,
    )
    return build_position_plan_from_spec(
        position_policy,
        base_target_weights=base_weights,
        selection_mask=selection,
        market=market,
    )


def _portfolio_shape_fields(portfolio_shape: PortfolioShapeSpec | None) -> tuple[str, ...]:
    if portfolio_shape is not None and portfolio_shape.kind == "sector_neutral":
        return (portfolio_shape.group_field,)
    return ()


def _shorting_fields(spec: ExecutionSpec) -> tuple[str, ...]:
    if spec.shorting.shortable_field is not None:
        return (spec.shorting.shortable_field,)
    return ()


def _build_portfolio_shape_construction(
    *,
    selection_spec: SelectionSpec,
    portfolio_shape: PortfolioShapeSpec,
    features: dict[str, pd.DataFrame],
):
    if selection_spec.kind != "rank_top_bottom":
        raise ValueError(f"portfolio_shape kind '{portfolio_shape.kind}' requires selection kind 'rank_top_bottom'")
    if selection_spec.field is None:
        raise ValueError("selection kind 'rank_top_bottom' requires field")
    if selection_spec.top_n is None or selection_spec.bottom_n is None:
        raise ValueError("selection kind 'rank_top_bottom' requires top_n and bottom_n")

    alpha = features[selection_spec.field]
    bundle = SignalBundle(alpha=alpha, context={})
    if portfolio_shape.kind == "long_short":
        return LongShortTopBottom(
            top_n=selection_spec.top_n,
            bottom_n=selection_spec.bottom_n,
            gross_long=portfolio_shape.gross_long,
            gross_short=portfolio_shape.gross_short,
        ).build(bundle)
    if portfolio_shape.kind == "sector_neutral":
        bundle = SignalBundle(
            alpha=alpha,
            context={"sector": features[portfolio_shape.group_field]},
        )
        return SectorNeutralTopBottom(
            top_n=selection_spec.top_n,
            bottom_n=selection_spec.bottom_n,
            group_budget=portfolio_shape.group_budget,
        ).build(bundle)
    raise ValueError(f"unknown portfolio_shape kind: {portfolio_shape.kind}")


def _apply_shorting(
    spec: ExecutionSpec,
    *,
    base_weights: pd.DataFrame,
    selection: pd.DataFrame,
    features: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    has_short_targets = bool(base_weights.lt(0.0).any().any())
    if has_short_targets and not spec.shorting.enabled:
        raise ValueError("negative target weights require shorting.enabled = true")

    if not spec.shorting.enabled or spec.shorting.shortable_field is None:
        return base_weights, selection

    shortable = (
        features[spec.shorting.shortable_field]
        .reindex(index=base_weights.index, columns=base_weights.columns)
        .fillna(False)
        .astype(bool)
    )
    blocked_short = base_weights.lt(0.0) & ~shortable
    if not bool(blocked_short.any().any()):
        return base_weights, selection

    adjusted_weights = base_weights.mask(blocked_short, 0.0).astype(float)
    adjusted_selection = selection.where(~blocked_short, False).astype(bool)
    return adjusted_weights, adjusted_selection


def _apply_scheduled_evaluation(
    spec: ExecutionSpec,
    *,
    base_weights: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = spec.schedule
    if schedule.kind == "signal_dates":
        if schedule.evaluation is not None:
            return _freeze_to_flags(
                base_weights=base_weights,
                selection=selection,
                flags=_evaluation_flags(schedule.evaluation, base_weights.index),
            )
        return base_weights, selection
    if schedule.kind == "named" and schedule.name == "monthly" and not schedule.evaluate_on_schedule:
        return base_weights, selection

    flags = _schedule_flags(schedule, base_weights.index)
    if flags is None:
        return base_weights, selection

    return _freeze_to_flags(base_weights=base_weights, selection=selection, flags=flags)


def _freeze_to_flags(
    *,
    base_weights: pd.DataFrame,
    selection: pd.DataFrame,
    flags: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = flags.reindex(base_weights.index).fillna(False).astype(bool)
    scheduled_weights = base_weights.where(flags, float("nan")).ffill().fillna(0.0).astype(float)
    scheduled_selection = (
        selection.astype(float)
        .where(flags, float("nan"))
        .ffill()
        .fillna(0.0)
        .astype(bool)
    )
    return scheduled_weights, scheduled_selection


def _evaluation_flags(evaluation: ScheduleEvaluationSpec, index: pd.Index) -> pd.Series:
    flags = _schedule_flags(evaluation, index)
    if flags is None:
        raise ValueError(f"unsupported schedule evaluation: {evaluation.kind}")
    return flags


def _schedule_flags(schedule, index: pd.Index) -> pd.Series | None:
    datetime_index = pd.DatetimeIndex(index)
    if schedule.kind == "named":
        if schedule.name == "daily":
            return DailySchedule().flags(datetime_index)
        if schedule.name == "weekly":
            return WeeklySchedule().flags(datetime_index)
        if schedule.name == "monthly":
            return MonthlySchedule().flags(datetime_index)
        return None
    if schedule.kind == "custom_dates":
        return CustomSchedule(pd.to_datetime(list(schedule.dates))).flags(datetime_index)
    return None
