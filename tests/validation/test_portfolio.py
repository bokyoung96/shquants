import pandas as pd
import pytest

from backtesting.policy.base import BUCKET_LEDGER_COLUMNS, PositionPlan
from backtesting.validation import validate_position_plan
from backtesting.validation.portfolio import _target_weight_values


def test_target_weight_values_omits_zero_positions_but_keeps_material_weights() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    weights = pd.DataFrame(
        {
            "A": [0.0, 0.5],
            "B": [0.0, 0.0],
            "C": [1e-10, -0.25],
        },
        index=index,
    )

    values = _target_weight_values(weights, tolerance=1e-8)

    assert values.to_dict() == {
        (pd.Timestamp("2024-01-03"), "A"): 0.5,
        (pd.Timestamp("2024-01-03"), "C"): -0.25,
    }


def test_validate_position_plan_rejects_bucket_sum_mismatch() -> None:
    index = pd.to_datetime(["2024-01-02"])
    plan = PositionPlan(
        target_weights=pd.DataFrame({"A": [0.60], "B": [0.40]}, index=index),
        bucket_ledger=pd.DataFrame.from_records(
            [
                {
                    "date": index[0],
                    "symbol": "A",
                    "side": "long",
                    "bucket_id": "base",
                    "stage_index": 0,
                    "target_weight": 0.90,
                    "actual_weight": 0.90,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "pass_through",
                    "construction_group": None,
                    "budget_id": "base",
                }
            ],
            columns=BUCKET_LEDGER_COLUMNS,
        ),
    )

    with pytest.raises(ValueError, match="bucket target_weight values do not match plan target_weights"):
        validate_position_plan(plan)


def test_validate_position_plan_rejects_symbol_swapped_bucket_weights() -> None:
    index = pd.to_datetime(["2024-01-02"])
    plan = PositionPlan(
        target_weights=pd.DataFrame({"A": [0.60], "B": [0.40]}, index=index),
        bucket_ledger=pd.DataFrame.from_records(
            [
                {
                    "date": index[0],
                    "symbol": "A",
                    "side": "long",
                    "bucket_id": "base",
                    "stage_index": 0,
                    "target_weight": 0.40,
                    "actual_weight": 0.40,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "pass_through",
                    "construction_group": None,
                    "budget_id": "base",
                },
                {
                    "date": index[0],
                    "symbol": "B",
                    "side": "long",
                    "bucket_id": "base",
                    "stage_index": 0,
                    "target_weight": 0.60,
                    "actual_weight": 0.60,
                    "target_qty": 0.0,
                    "actual_qty": 0.0,
                    "entry_price": None,
                    "mark_price": None,
                    "bucket_return": 0.0,
                    "state": "active",
                    "event": "pass_through",
                    "construction_group": None,
                    "budget_id": "base",
                },
            ],
            columns=BUCKET_LEDGER_COLUMNS,
        ),
    )

    with pytest.raises(ValueError, match="bucket target_weight values do not match plan target_weights"):
        validate_position_plan(plan)
