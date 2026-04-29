from __future__ import annotations

import pandas as pd
import pytest

from backtesting.data import MarketData
from backtesting.policy import build_position_plan_from_spec
from backtesting.specs.models import PositionBucketSpec, PositionPolicySpec, PositionRuleSpec


def _market(base: pd.DataFrame) -> MarketData:
    close = pd.DataFrame(10.0, index=base.index, columns=base.columns)
    return MarketData(frames={"close": close}, universe=None, benchmark=None)


def test_build_position_plan_from_spec_pass_through_preserves_base_target_weights() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    base = pd.DataFrame({"A": [0.4, 0.2], "B": [0.6, 0.8]}, index=index)
    selection_mask = pd.DataFrame({"A": [True, False], "B": [True, True]}, index=index)

    plan = build_position_plan_from_spec(
        PositionPolicySpec(kind="pass_through"),
        base_target_weights=base,
        selection_mask=selection_mask,
        market=_market(base),
    )

    pd.testing.assert_frame_equal(plan.target_weights, base)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (PositionPolicySpec(kind="pass_through", hook_id="demo"), "pass_through position policy does not support hook_id"),
        (PositionPolicySpec(kind="pass_through", params={"x": 1}), "pass_through position policy does not support params"),
        (
            PositionPolicySpec(
                kind="staged",
                buckets=(PositionBucketSpec("b0", 1.0),),
                params={"x": 1},
            ),
            "staged position policy does not support params",
        ),
        (
            PositionPolicySpec(
                kind="staged",
                buckets=(PositionBucketSpec("b0", 1.0),),
                hook_id="demo",
            ),
            "staged position policy does not support hook_id",
        ),
    ],
)
def test_build_position_plan_from_spec_rejects_unsupported_hook_id_or_params(
    spec: PositionPolicySpec,
    match: str,
) -> None:
    index = pd.to_datetime(["2024-01-02"])
    base = pd.DataFrame({"A": [1.0]}, index=index)
    selection_mask = pd.DataFrame({"A": [True]}, index=index)

    with pytest.raises(ValueError, match=match):
        build_position_plan_from_spec(
            spec,
            base_target_weights=base,
            selection_mask=selection_mask,
            market=_market(base),
        )


def test_build_position_plan_from_spec_staged_activates_first_bucket_on_selection_pass() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    base = pd.DataFrame({"A": [1.0, 1.0]}, index=index)
    selection_mask = pd.DataFrame({"A": [True, True]}, index=index)

    plan = build_position_plan_from_spec(
        PositionPolicySpec(
            kind="staged",
            buckets=(PositionBucketSpec("b0", 0.25), PositionBucketSpec("b1", 0.75)),
            adds=(PositionRuleSpec("still_passes_after_rebalances", count=1),),
        ),
        base_target_weights=base,
        selection_mask=selection_mask,
        market=_market(base),
    )

    assert plan.target_weights["A"].tolist() == [0.25, 1.0]


def test_build_position_plan_from_spec_staged_add_bucket_requires_rebalance_hold_count() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    base = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=index)
    selection_mask = pd.DataFrame({"A": [True, False, True]}, index=index)

    plan = build_position_plan_from_spec(
        PositionPolicySpec(
            kind="staged",
            buckets=(PositionBucketSpec("b0", 0.5), PositionBucketSpec("b1", 0.5)),
            adds=(PositionRuleSpec("still_passes_after_rebalances", count=1),),
        ),
        base_target_weights=base,
        selection_mask=selection_mask,
        market=_market(base),
    )

    assert plan.target_weights["A"].tolist() == [0.5, 0.0, 0.5]


def test_build_position_plan_from_spec_staged_clears_exposure_on_selection_failure() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    base = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=index)
    selection_mask = pd.DataFrame({"A": [True, False, False]}, index=index)

    plan = build_position_plan_from_spec(
        PositionPolicySpec(
            kind="staged",
            buckets=(PositionBucketSpec("b0", 0.5), PositionBucketSpec("b1", 0.5)),
            adds=(PositionRuleSpec("still_passes_after_rebalances", count=1),),
        ),
        base_target_weights=base,
        selection_mask=selection_mask,
        market=_market(base),
    )

    assert plan.target_weights["A"].tolist() == [0.5, 0.0, 0.0]


@pytest.mark.parametrize("rule_kind", ["selection_passes", "selection_fails"])
def test_build_position_plan_from_spec_rejects_non_zero_count_for_pass_fail_rules(rule_kind: str) -> None:
    index = pd.to_datetime(["2024-01-02"])
    base = pd.DataFrame({"A": [1.0]}, index=index)
    selection_mask = pd.DataFrame({"A": [True]}, index=index)

    with pytest.raises(ValueError, match=f"{rule_kind} rule count must be 0"):
        build_position_plan_from_spec(
            PositionPolicySpec(
                kind="staged",
                buckets=(PositionBucketSpec("b0", 1.0),),
                entry=PositionRuleSpec(rule_kind, count=2),
            ),
            base_target_weights=base,
            selection_mask=selection_mask,
            market=_market(base),
        )


def test_build_position_plan_from_spec_rejects_unsupported_position_rule_kind() -> None:
    index = pd.to_datetime(["2024-01-02"])
    base = pd.DataFrame({"A": [1.0]}, index=index)
    selection_mask = pd.DataFrame({"A": [True]}, index=index)

    with pytest.raises(ValueError, match="unsupported position rule kind"):
        build_position_plan_from_spec(
            PositionPolicySpec(
                kind="staged",
                buckets=(PositionBucketSpec("b0", 1.0),),
                entry=PositionRuleSpec("mystery_rule"),
            ),
            base_target_weights=base,
            selection_mask=selection_mask,
            market=_market(base),
        )
