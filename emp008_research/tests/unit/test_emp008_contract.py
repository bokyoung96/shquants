from pathlib import Path

import pandas as pd

from emp008.data import Emp008Config
from emp008.factor_registry import (
    FACTOR_SET_DEFINITIONS,
    FactorSetId,
    get_factor_set_definition,
    parse_factor_set,
)
from emp008.strategy import Emp008Result


def test_default_config_preserves_production_core_contract() -> None:
    config = Emp008Config()

    assert config.factor_set is FactorSetId.PRODUCTION_CORE
    assert config.risk_window == 36
    assert not hasattr(config, "risk_model")
    assert not hasattr(config, "expected_alpha_estimator")


def test_factor_set_names_describe_role_and_legacy_mfbt_is_compatibility_alias():
    assert parse_factor_set("mfbt") is FactorSetId.PRODUCTION_CORE
    definition = get_factor_set_definition(FactorSetId.PRODUCTION_CORE)
    assert definition.category == "production"
    assert definition.label == "Core price / earnings / dividend / retail / value / size"


def test_factor_set_categories_cover_all_canonical_sets():
    assert set(definition.category for definition in FACTOR_SET_DEFINITIONS.values()) == {
        "production", "research", "reference", "diagnostic"
    }


def test_result_writes_target_and_active_weight_contract(tmp_path: Path) -> None:
    index = pd.DatetimeIndex(["2020-01-31"])
    target = pd.DataFrame([[0.6, 0.4]], index=index, columns=["A", "B"])
    active = pd.DataFrame([[0.1, -0.1]], index=index, columns=["A", "B"])
    result = Emp008Result(target_weights=target, active_weights=active, diagnostics=pd.DataFrame())

    result.write_outputs(tmp_path)

    assert (tmp_path / "target_weights.parquet").exists()
    assert (tmp_path / "active_weights.parquet").exists()
    assert list(result.weights_for_export().index) == ["A", "B"]
