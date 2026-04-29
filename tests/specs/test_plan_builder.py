from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backtesting.data import MarketData
from backtesting.specs import build_position_plan_from_execution_spec
from backtesting.specs.models import (
    ConditionSpec,
    ExecutionSpec,
    PositionBucketSpec,
    PositionPolicySpec,
    PositionRuleSpec,
    SelectionSpec,
    WeightingSpec,
)


def _market(
    *,
    close: pd.DataFrame | None = None,
    market_cap: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
) -> MarketData:
    anchor = close if close is not None else market_cap
    if anchor is None:
        raise ValueError("market helper requires at least one frame")

    frames: dict[str, pd.DataFrame] = {}
    frames["close"] = close if close is not None else pd.DataFrame(10.0, index=anchor.index, columns=anchor.columns)
    if market_cap is not None:
        frames["market_cap"] = market_cap
    return MarketData(frames=frames, universe=universe, benchmark=None)


def test_filter_selection_with_default_equal_weight_gives_equal_weights_to_passing_names() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    market_cap = pd.DataFrame(
        {
            "A": [100.0, 200.0],
            "B": [40.0, 80.0],
            "C": [120.0, 30.0],
        },
        index=index,
    )
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-03",
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="market_cap", op=">=", value=80.0),),
        ),
    )

    plan = build_position_plan_from_execution_spec(spec, _market(market_cap=market_cap))

    expected = pd.DataFrame(
        {
            "A": [0.5, 0.5],
            "B": [0.0, 0.5],
            "C": [0.5, 0.0],
        },
        index=index,
    )
    assert_frame_equal(plan.target_weights, expected.astype(float))


def test_market_universe_masks_out_names_before_weighting() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    market_cap = pd.DataFrame(
        {
            "A": [100.0, 100.0],
            "B": [100.0, 100.0],
            "C": [100.0, 100.0],
        },
        index=index,
    )
    universe = pd.DataFrame(
        {
            "A": [True, True],
            "B": [False, True],
            "C": [True, False],
        },
        index=index,
        dtype=bool,
    )
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-03",
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="market_cap", op=">=", value=1.0),),
        ),
    )

    plan = build_position_plan_from_execution_spec(spec, _market(market_cap=market_cap, universe=universe))

    expected = pd.DataFrame(
        {
            "A": [0.5, 0.5],
            "B": [0.0, 0.5],
            "C": [0.5, 0.0],
        },
        index=index,
    )
    assert_frame_equal(plan.target_weights, expected.astype(float))


def test_market_universe_treats_nullable_entries_as_false() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    market_cap = pd.DataFrame(
        {
            "A": [100.0, 100.0],
            "B": [100.0, 100.0],
            "C": [100.0, 100.0],
        },
        index=index,
    )
    universe = pd.DataFrame(
        {
            "A": [True, pd.NA],
            "B": [pd.NA, True],
            "C": [True, None],
        },
        index=index,
        dtype="object",
    )
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-03",
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="market_cap", op=">=", value=1.0),),
        ),
    )

    plan = build_position_plan_from_execution_spec(spec, _market(market_cap=market_cap, universe=universe))

    expected = pd.DataFrame(
        {
            "A": [0.5, 0.0],
            "B": [0.0, 1.0],
            "C": [0.5, 0.0],
        },
        index=index,
    )
    assert_frame_equal(plan.target_weights, expected.astype(float))


def test_staged_position_policy_is_applied_after_base_weights_are_built() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    market_cap = pd.DataFrame(
        {
            "A": [100.0, 100.0],
            "B": [300.0, 300.0],
        },
        index=index,
    )
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-03",
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="market_cap", op=">=", value=1.0),),
        ),
        weighting=WeightingSpec(kind="market_cap"),
        position_policy=PositionPolicySpec(
            kind="staged",
            buckets=(PositionBucketSpec("b0", 0.25), PositionBucketSpec("b1", 0.75)),
            adds=(PositionRuleSpec("still_passes_after_rebalances", count=1),),
        ),
    )

    plan = build_position_plan_from_execution_spec(spec, _market(market_cap=market_cap))

    expected = pd.DataFrame(
        {
            "A": [0.0625, 0.25],
            "B": [0.1875, 0.75],
        },
        index=index,
    )
    assert_frame_equal(plan.target_weights, expected.astype(float))



def test_missing_selection_raises_value_error() -> None:
    index = pd.to_datetime(["2024-01-02"])
    close = pd.DataFrame({"A": [10.0]}, index=index)
    spec = ExecutionSpec(start="2024-01-02", end="2024-01-02")

    with pytest.raises(ValueError, match="spec-driven plan requires selection"):
        build_position_plan_from_execution_spec(spec, _market(close=close))
