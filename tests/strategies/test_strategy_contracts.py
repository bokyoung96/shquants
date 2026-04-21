import pandas as pd
from pandas.testing import assert_frame_equal

from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.policy.base import BUCKET_LEDGER_COLUMNS
from backtesting.policy.pass_through import PassThroughPolicy
from backtesting.signals.base import SignalBundle


def test_pass_through_policy_emits_base_weights_as_single_bucket() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    weights = pd.DataFrame({"A": [0.5, 0.5], "B": [-0.5, -0.5]}, index=index)
    bundle = SignalBundle(alpha=weights.abs(), context={"tradable": weights.notna()})
    construction = ConstructionResult(
        base_target_weights=weights,
        selection_mask=weights.ne(0.0),
        group_long_budget=None,
        group_short_budget=None,
        meta={},
    )

    plan = PassThroughPolicy().apply(
        construction=construction,
        market=MarketData(frames={"close": weights.abs().add(10.0)}, universe=None, benchmark=None),
        bundle=bundle,
    )

    assert_frame_equal(plan.target_weights, weights)
    assert set(plan.bucket_ledger["bucket_id"]) == {"base"}
    assert plan.bucket_ledger["target_weight"].sum() == 0.0
    assert plan.bucket_meta["policy_name"].iloc[0] == "pass_through"
    assert plan.validation == {}


def test_pass_through_policy_preserves_empty_ledger_schema() -> None:
    index = pd.to_datetime(["2024-01-02"])
    weights = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=index)
    bundle = SignalBundle(alpha=weights.abs(), context={"tradable": weights.notna()})
    construction = ConstructionResult(
        base_target_weights=weights,
        selection_mask=weights.ne(0.0),
        group_long_budget=None,
        group_short_budget=None,
        meta={},
    )

    plan = PassThroughPolicy().apply(
        construction=construction,
        market=MarketData(frames={"close": weights.abs().add(10.0)}, universe=None, benchmark=None),
        bundle=bundle,
    )

    assert_frame_equal(plan.target_weights, weights)
    assert plan.bucket_ledger.empty
    assert tuple(plan.bucket_ledger.columns) == BUCKET_LEDGER_COLUMNS
    assert plan.bucket_meta["policy_name"].iloc[0] == "pass_through"
    assert plan.validation == {}
