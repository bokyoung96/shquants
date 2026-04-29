import json
from pathlib import Path

import pytest

from backtesting.specs import (
    ConditionSpec,
    PositionBucketSpec,
    PositionPolicySpec,
    PositionRuleSpec,
    SelectionSpec,
    WeightingSpec,
    load_execution_spec,
)


def test_load_execution_spec_from_json(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "strategy": "momentum",
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
                "strategy": "momentum",
                "top_n": 3,
                "lookback": 1,
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.strategy == "momentum"
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

