import json
from pathlib import Path

import pytest

from backtesting.specs import (
    ConditionSpec,
    PortfolioShapeSpec,
    PositionBucketSpec,
    PositionPolicySpec,
    PositionRuleSpec,
    ScheduleEvaluationSpec,
    SelectionSpec,
    ShortingSpec,
    TargetWeightsSpec,
    WeightingSpec,
    load_execution_spec,
)


def _write_spec(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "run_spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_execution_spec_from_json(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "strategy": "trend_rank",
                "schedule": {"kind": "named", "name": "monthly"},
                "weight_source": {"kind": "strategy"},
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.start == "2024-01-01"
    assert spec.schedule.name == "monthly"
    assert spec.weight_source.kind == "strategy"


def test_load_execution_spec_parses_signal_dates_schedule(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "schedule": {"kind": "signal_dates", "weight_change_tolerance": 1e-6},
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.schedule.kind == "signal_dates"
    assert spec.schedule.weight_change_tolerance == 1e-6


def test_load_execution_spec_parses_signal_dates_evaluation_schedule(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "schedule": {
                    "kind": "signal_dates",
                    "weight_change_tolerance": 1e-6,
                    "evaluation": {"kind": "named", "name": "weekly"},
                },
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.schedule.kind == "signal_dates"
    assert spec.schedule.evaluation == ScheduleEvaluationSpec(kind="named", name="weekly")


def test_load_execution_spec_parses_rank_top_bottom_and_portfolio_shape(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {
                    "kind": "rank_top_bottom",
                    "field": "momentum_60d",
                    "top_n": 20,
                    "bottom_n": 10,
                },
                "portfolio_shape": {
                    "kind": "long_short",
                    "gross_long": 1.5,
                    "gross_short": 0.5,
                    "group_budget": "proportional_selected",
                },
                "shorting": {
                    "enabled": True,
                    "borrow_fee_annual": 0.03,
                    "shortable_field": "shortable",
                    "cash_collateral_ratio": 1.25,
                },
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.selection == SelectionSpec(
        kind="rank_top_bottom",
        field="momentum_60d",
        top_n=20,
        bottom_n=10,
    )
    assert spec.portfolio_shape == PortfolioShapeSpec(
        kind="long_short",
        gross_long=1.5,
        gross_short=0.5,
        group_budget="proportional_selected",
    )
    assert spec.shorting == ShortingSpec(
        enabled=True,
        borrow_fee_annual=0.03,
        shortable_field="shortable",
        cash_collateral_ratio=1.25,
    )


def test_load_execution_spec_parses_target_weights_file(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "target_weights": {
                    "kind": "file",
                    "path": "weights.csv",
                    "missing_policy": "zero",
                    "untradable_policy": "fail",
                    "unshortable_policy": "fail",
                },
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.target_weights == TargetWeightsSpec(
        kind="file",
        path="weights.csv",
        missing_policy="zero",
        untradable_policy="fail",
        unshortable_policy="fail",
    )
    assert spec.uses_composable_plan is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "hook", "target_weights.kind must be one of"),
        ("missing_policy", "fail", "target_weights.missing_policy must be one of"),
        ("untradable_policy", "zero", "target_weights.untradable_policy must be one of"),
        ("unshortable_policy", "zero", "target_weights.unshortable_policy must be one of"),
    ],
)
def test_load_execution_spec_rejects_invalid_target_weights_policy(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    target_weights = {"kind": "file", "path": "weights.csv"}
    target_weights[field] = value
    path = _write_spec(
        tmp_path,
        {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "target_weights": target_weights,
        },
    )

    with pytest.raises(ValueError, match=message):
        load_execution_spec(path)


def test_load_execution_spec_rejects_yaml_without_parser_support(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.yaml"
    path.write_text("start: 2024-01-01\nend: 2024-12-31\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML spec loading is not available without an approved YAML dependency"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_non_boolean_flags(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "use_k200": "false",
                "allow_fractional": "false",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="use_k200 must be a boolean"):
        load_execution_spec(path)


def test_load_execution_spec_parses_selection_weighting_and_staged_policy(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {
                    "kind": "filter",
                    "conditions": [
                        {"field": "momentum_60d", "op": ">", "value": 0},
                        {"field": "market_cap", "op": ">=", "value": 100_000_000_000},
                    ],
                },
                "weighting": {"kind": "equal_weight"},
                "position_policy": {
                    "kind": "staged",
                    "buckets": [
                        {"id": "entry", "fraction": 0.5},
                        {"id": "add_1", "fraction": 0.5},
                    ],
                    "rules": {
                        "entry": {"kind": "selection_passes"},
                        "adds": [{"kind": "still_passes_after_rebalances", "count": 1}],
                        "exit": {"kind": "selection_fails"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.selection is not None
    assert spec.selection.kind == "filter"
    assert spec.selection.conditions == (
        ConditionSpec(field="momentum_60d", op=">", value=0),
        ConditionSpec(field="market_cap", op=">=", value=100_000_000_000),
    )
    assert spec.weighting == WeightingSpec(kind="equal_weight")
    assert spec.position_policy == PositionPolicySpec(
        kind="staged",
        buckets=(
            PositionBucketSpec(id="entry", fraction=0.5),
            PositionBucketSpec(id="add_1", fraction=0.5),
        ),
        entry=PositionRuleSpec("selection_passes"),
        adds=(PositionRuleSpec("still_passes_after_rebalances", count=1),),
        exit=PositionRuleSpec("selection_fails"),
    )
    assert spec.uses_composable_plan is True


def test_load_execution_spec_keeps_legacy_strategy_specs_on_legacy_path(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "strategy": "trend_rank",
                "top_n": 3,
                "lookback": 1,
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.strategy == "trend_rank"
    assert spec.top_n == 3
    assert spec.lookback == 1
    assert spec.selection is None
    assert spec.weighting is None
    assert spec.position_policy is None
    assert spec.uses_composable_plan is False


def test_load_execution_spec_defaults_weighting_when_selection_exists(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {
                    "kind": "rank_top_n",
                    "field": "momentum_20d",
                    "n": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.selection == SelectionSpec(kind="rank_top_n", field="momentum_20d", n=2)
    assert spec.weighting == WeightingSpec(kind="equal_weight")
    assert spec.position_policy == PositionPolicySpec(kind="pass_through")


def test_load_execution_spec_rejects_non_object_position_policy(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {"kind": "rank_top_n", "field": "momentum_20d", "n": 2},
                "position_policy": "staged",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="position_policy must be an object"):
        load_execution_spec(path)

def test_load_execution_spec_rejects_non_boolean_selection_ascending(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {
                    "kind": "rank_top_n",
                    "field": "momentum_20d",
                    "n": 2,
                    "ascending": "false",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selection.ascending must be a boolean"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_null_weighting_kind(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {"kind": "rank_top_n", "field": "momentum_20d", "n": 2},
                "weighting": {"kind": None},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="weighting.kind must be a string"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_non_object_selection_params(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {
                    "kind": "rank_top_n",
                    "field": "momentum_20d",
                    "n": 2,
                    "params": [["window", 20]],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selection.params must be an object"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_null_position_policy_kind(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {"kind": "rank_top_n", "field": "momentum_20d", "n": 2},
                "position_policy": {"kind": None},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="position_policy.kind must be a string"):
        load_execution_spec(path)



def test_load_execution_spec_rejects_non_object_position_policy_params(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {"kind": "rank_top_n", "field": "momentum_20d", "n": 2},
                "position_policy": {"params": [["x", 1]]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="position_policy.params must be an object"):
        load_execution_spec(path)



def test_load_execution_spec_rejects_null_position_policy_entry_kind(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {"kind": "rank_top_n", "field": "momentum_20d", "n": 2},
                "position_policy": {"rules": {"entry": {"kind": None}}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="position_policy.rules.entry.kind must be a string"):
        load_execution_spec(path)



def test_load_execution_spec_rejects_invalid_weighting_field_label(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {"kind": "rank_top_n", "field": "momentum_20d", "n": 2},
                "weighting": {"field": 123},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="weighting.field must be a string"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_invalid_selection_field_label(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "selection": {"kind": "rank_top_n", "field": 123, "n": 2},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selection.field must be a string"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_non_object_root(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, ["not", "an", "object"])

    with pytest.raises(ValueError, match="execution spec must be an object"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_missing_or_malformed_dates(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, {"start": "2024/01/01", "end": "2024-12-31"})

    with pytest.raises(ValueError, match="start must be a YYYY-MM-DD date string"):
        load_execution_spec(path)

    path = _write_spec(tmp_path, {"start": "2024-12-31", "end": "2024-01-01"})

    with pytest.raises(ValueError, match="start must be on or before end"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_bool_and_negative_integer_fields(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, {"start": "2024-01-01", "end": "2024-12-31", "top_n": True})

    with pytest.raises(ValueError, match="top_n must be an integer"):
        load_execution_spec(path)

    path = _write_spec(tmp_path, {"start": "2024-01-01", "end": "2024-12-31", "warmup_days": -1})

    with pytest.raises(ValueError, match="warmup_days must be >= 0"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_missing_condition_fields(tmp_path: Path) -> None:
    path = _write_spec(
        tmp_path,
        {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "selection": {"kind": "filter", "conditions": [{"op": ">", "value": 0}]},
        },
    )

    with pytest.raises(ValueError, match="selection.conditions.field must be a string"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_invalid_portfolio_shape_values(tmp_path: Path) -> None:
    path = _write_spec(
        tmp_path,
        {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "portfolio_shape": {"kind": "market_neutral"},
        },
    )

    with pytest.raises(ValueError, match="portfolio_shape.kind must be one of"):
        load_execution_spec(path)

    path = _write_spec(
        tmp_path,
        {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "portfolio_shape": {"kind": "sector_neutral", "group_budget": "market_cap"},
        },
    )

    with pytest.raises(ValueError, match="portfolio_shape.group_budget must be one of"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_invalid_shorting_and_bucket_values(tmp_path: Path) -> None:
    path = _write_spec(
        tmp_path,
        {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "shorting": {"borrow_fee_annual": -0.01},
        },
    )

    with pytest.raises(ValueError, match="shorting.borrow_fee_annual must be >= 0"):
        load_execution_spec(path)

    path = _write_spec(
        tmp_path,
        {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "selection": {"kind": "rank_top_n", "field": "momentum_20d", "n": 2},
            "position_policy": {"kind": "staged", "buckets": [{"id": "entry", "fraction": 1.5}]},
        },
    )

    with pytest.raises(ValueError, match="position_policy.buckets.fraction must be <= 1"):
        load_execution_spec(path)


def test_example_specs_parse() -> None:
    examples = [
        Path("docs/superpowers/specs/examples/filter-equal-weight.json"),
        Path("docs/superpowers/specs/examples/filter-staged.json"),
        Path("docs/openclaw/examples/signal-dates-filter.json"),
        Path("docs/openclaw/examples/signal-dates-weekly-evaluation.json"),
        Path("docs/openclaw/examples/rank-top-bottom-long-short.json"),
        Path("docs/openclaw/examples/rank-top-bottom-sector-neutral.json"),
    ]
    for path in examples:
        spec = load_execution_spec(path)
        assert spec.uses_composable_plan
