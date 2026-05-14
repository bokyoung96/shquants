from __future__ import annotations

from backtesting.data import MarketData
from backtesting.features import build_features
from backtesting.policy import PositionPlan, build_position_plan_from_spec
from backtesting.selection import build_selection, selection_fields
from backtesting.specs.models import ExecutionSpec, PositionPolicySpec, WeightingSpec
from backtesting.specs.portfolio_shapes import build_portfolio_shape_construction, portfolio_shape_fields
from backtesting.specs.schedule_evaluation import apply_scheduled_evaluation
from backtesting.specs.shorting import apply_shorting, shorting_fields
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
                *portfolio_shape_fields(spec.portfolio_shape),
                *shorting_fields(spec),
            )
        )
    )
    features = build_features(market, fields)
    if spec.portfolio_shape is not None and spec.portfolio_shape.kind in {"long_short", "sector_neutral"}:
        construction = build_portfolio_shape_construction(
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
    base_weights, selection = apply_shorting(
        spec,
        base_weights=base_weights,
        selection=selection,
        features=features,
    )
    base_weights, selection = apply_scheduled_evaluation(
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
