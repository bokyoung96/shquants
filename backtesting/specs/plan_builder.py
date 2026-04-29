from __future__ import annotations

from backtesting.data import MarketData
from backtesting.features import build_features
from backtesting.policy import PositionPlan, build_position_plan_from_spec
from backtesting.selection import build_selection, selection_fields
from backtesting.specs.models import ExecutionSpec, PositionPolicySpec, WeightingSpec
from backtesting.weighting import build_weights, weighting_fields


def build_position_plan_from_execution_spec(spec: ExecutionSpec, market: MarketData) -> PositionPlan:
    if spec.selection is None:
        raise ValueError("spec-driven plan requires selection")

    weighting = spec.weighting or WeightingSpec(kind="equal_weight")
    position_policy = spec.position_policy or PositionPolicySpec(kind="pass_through")
    fields = tuple(dict.fromkeys((*selection_fields(spec.selection), *weighting_fields(weighting))))
    features = build_features(market, fields)
    selection = build_selection(spec.selection, features)

    if market.universe is not None:
        universe = market.universe.reindex(index=selection.index, columns=selection.columns, fill_value=False).astype(bool)
        selection = selection & universe

    base_weights = build_weights(weighting, selection, features)
    return build_position_plan_from_spec(
        position_policy,
        base_target_weights=base_weights,
        selection_mask=selection,
        market=market,
    )
