from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backtesting.data import MarketData
from backtesting.specs import build_position_plan_from_execution_spec
from backtesting.specs.models import (
    ConditionSpec,
    ExecutionSpec,
    PortfolioShapeSpec,
    PositionBucketSpec,
    PositionPolicySpec,
    PositionRuleSpec,
    ScheduleEvaluationSpec,
    ScheduleSpec,
    SelectionSpec,
    ShortingSpec,
    WeightingSpec,
)


def _market(
    *,
    close: pd.DataFrame | None = None,
    market_cap: pd.DataFrame | None = None,
    sector: pd.DataFrame | None = None,
    shortable: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
) -> MarketData:
    anchor = close if close is not None else market_cap
    if anchor is None:
        raise ValueError("market helper requires at least one frame")

    frames: dict[str, pd.DataFrame] = {}
    frames["close"] = close if close is not None else pd.DataFrame(10.0, index=anchor.index, columns=anchor.columns)
    if market_cap is not None:
        frames["market_cap"] = market_cap
    if sector is not None:
        frames["sector_big"] = sector
    if shortable is not None:
        frames["shortable"] = shortable
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


def test_named_weekly_schedule_evaluates_and_holds_only_scheduled_positions() -> None:
    index = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    close = pd.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0],
            "B": [10.0, 10.0, 9.0, 8.0, 7.0],
        },
        index=index,
    )
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-01-05",
        schedule=ScheduleSpec(kind="named", name="weekly"),
        selection=SelectionSpec(kind="rank_top_n", field="close", n=1),
        weighting=WeightingSpec(kind="equal_weight"),
    )

    plan = build_position_plan_from_execution_spec(spec, _market(close=close))

    expected = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0, 0.0, 1.0],
            "B": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    assert_frame_equal(plan.target_weights, expected)


def test_signal_dates_evaluation_schedule_evaluates_and_holds_only_scheduled_positions() -> None:
    index = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    close = pd.DataFrame(
        {
            "A": [9.0, 11.0, 12.0, 13.0, 14.0],
            "B": [10.0, 10.0, 10.0, 10.0, 10.0],
        },
        index=index,
    )
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-01-05",
        schedule=ScheduleSpec(
            kind="signal_dates",
            evaluation=ScheduleEvaluationSpec(kind="named", name="weekly"),
        ),
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="close", op=">", value=10.0),),
        ),
        weighting=WeightingSpec(kind="equal_weight"),
    )

    plan = build_position_plan_from_execution_spec(spec, _market(close=close))

    expected = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0, 0.0, 1.0],
            "B": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    assert_frame_equal(plan.target_weights, expected)


def test_rank_top_bottom_long_short_portfolio_shape_builds_dollar_neutral_plan() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    close = pd.DataFrame(
        {
            "A": [5.0, 1.0],
            "B": [4.0, 0.0],
            "C": [1.0, 3.0],
            "D": [0.0, 2.0],
        },
        index=index,
    )
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-03",
        schedule=ScheduleSpec(kind="named", name="daily"),
        selection=SelectionSpec(kind="rank_top_bottom", field="close", top_n=2, bottom_n=1),
        portfolio_shape=PortfolioShapeSpec(kind="long_short"),
        shorting=ShortingSpec(enabled=True),
    )

    plan = build_position_plan_from_execution_spec(spec, _market(close=close))

    expected = pd.DataFrame(
        {
            "A": [0.5, 0.0],
            "B": [0.5, -1.0],
            "C": [0.0, 0.5],
            "D": [-1.0, 0.5],
        },
        index=index,
    )
    assert_frame_equal(plan.target_weights, expected)


def test_rank_top_bottom_sector_neutral_portfolio_shape_balances_each_sector() -> None:
    index = pd.to_datetime(["2024-01-02"])
    close = pd.DataFrame(
        {"A": [9.0], "B": [1.0], "C": [8.0], "D": [0.0]},
        index=index,
    )
    sector = pd.DataFrame(
        {"A": ["Tech"], "B": ["Tech"], "C": ["Energy"], "D": ["Energy"]},
        index=index,
    )
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-02",
        schedule=ScheduleSpec(kind="named", name="daily"),
        selection=SelectionSpec(kind="rank_top_bottom", field="close", top_n=1, bottom_n=1),
        portfolio_shape=PortfolioShapeSpec(kind="sector_neutral", group_field="sector"),
        shorting=ShortingSpec(enabled=True),
    )

    plan = build_position_plan_from_execution_spec(
        spec,
        _market(close=close, sector=sector),
    )

    expected = pd.DataFrame(
        {"A": [0.5], "B": [-0.5], "C": [0.5], "D": [-0.5]},
        index=index,
    )
    assert_frame_equal(plan.target_weights, expected)


def test_rank_top_bottom_sector_neutral_portfolio_shape_uses_proportional_selected_group_budget() -> None:
    index = pd.to_datetime(["2024-01-02"])
    close = pd.DataFrame(
        {"A": [10.0], "B": [9.0], "C": [1.0], "D": [0.0], "E": [8.0], "F": [2.0]},
        index=index,
    )
    sector = pd.DataFrame(
        {
            "A": ["Tech"],
            "B": ["Tech"],
            "C": ["Tech"],
            "D": ["Tech"],
            "E": ["Energy"],
            "F": ["Energy"],
        },
        index=index,
    )
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-02",
        schedule=ScheduleSpec(kind="named", name="daily"),
        selection=SelectionSpec(kind="rank_top_bottom", field="close", top_n=2, bottom_n=2),
        portfolio_shape=PortfolioShapeSpec(
            kind="sector_neutral",
            group_field="sector",
            group_budget="proportional_selected",
        ),
        shorting=ShortingSpec(enabled=True),
    )

    plan = build_position_plan_from_execution_spec(spec, _market(close=close, sector=sector))

    expected = pd.DataFrame(
        {
            "A": [1.0 / 3.0],
            "B": [1.0 / 3.0],
            "C": [-1.0 / 3.0],
            "D": [-1.0 / 3.0],
            "E": [1.0 / 3.0],
            "F": [-1.0 / 3.0],
        },
        index=index,
    )
    assert_frame_equal(plan.target_weights, expected)


def test_shorting_must_be_enabled_for_negative_portfolio_shape_weights() -> None:
    index = pd.to_datetime(["2024-01-02"])
    close = pd.DataFrame({"A": [5.0], "B": [1.0]}, index=index)
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-02",
        schedule=ScheduleSpec(kind="named", name="daily"),
        selection=SelectionSpec(kind="rank_top_bottom", field="close", top_n=1, bottom_n=1),
        portfolio_shape=PortfolioShapeSpec(kind="long_short"),
    )

    with pytest.raises(ValueError, match="shorting.enabled"):
        build_position_plan_from_execution_spec(spec, _market(close=close))


def test_shortable_field_zeroes_short_targets_for_unshortable_names() -> None:
    index = pd.to_datetime(["2024-01-02"])
    close = pd.DataFrame({"A": [5.0], "B": [1.0], "C": [0.0]}, index=index)
    shortable = pd.DataFrame({"A": [True], "B": [True], "C": [False]}, index=index)
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-02",
        schedule=ScheduleSpec(kind="named", name="daily"),
        selection=SelectionSpec(kind="rank_top_bottom", field="close", top_n=1, bottom_n=1),
        portfolio_shape=PortfolioShapeSpec(kind="long_short"),
        shorting=ShortingSpec(enabled=True, shortable_field="shortable"),
    )

    plan = build_position_plan_from_execution_spec(spec, _market(close=close, shortable=shortable))

    expected = pd.DataFrame({"A": [1.0], "B": [0.0], "C": [0.0]}, index=index)
    assert_frame_equal(plan.target_weights, expected)



def test_missing_selection_raises_value_error() -> None:
    index = pd.to_datetime(["2024-01-02"])
    close = pd.DataFrame({"A": [10.0]}, index=index)
    spec = ExecutionSpec(start="2024-01-02", end="2024-01-02")

    with pytest.raises(ValueError, match="spec-driven plan requires selection"):
        build_position_plan_from_execution_spec(spec, _market(close=close))
