from __future__ import annotations

from enum import Enum
from types import MappingProxyType

import pytest

from backtesting.catalog import DatasetId
from backtesting.strategies.emp008.data import Emp008Config
from backtesting.strategies.emp008.factor_registry import (
    FACTOR_DEFINITIONS,
    FACTOR_SET_DEFINITIONS,
    FactorDefinition,
    FactorDirection,
    FactorId,
    FactorSetDefinition,
    FactorSetId,
    _validate_registry,
    factor_definitions_for_set,
    factor_set_values,
    get_factor_definition,
    get_factor_set_definition,
    parse_factor_set,
)


def test_factor_and_factor_set_enums_define_expected_exact_values() -> None:
    assert issubclass(FactorId, Enum)
    assert issubclass(FactorSetId, Enum)
    assert [member.value for member in FactorId] == [
        "price_momentum",
        "positivity_momentum",
        "earnings_momentum",
        "dividend_yield",
        "retail_flow",
        "value",
        "ln_market_cap",
        "LnMktcap",
        "Momentum_12M",
        "DY",
    ]
    assert [member.value for member in FactorSetId] == [
        "mfbt",
        "mfbt_pos",
        "mfbt_origin_smallcap",
        "origin",
        "origin_new_dividend",
    ]
    assert [member.value for member in FactorDirection] == ["high", "low"]


def test_registry_exposes_read_only_complete_mappings() -> None:
    assert isinstance(FACTOR_DEFINITIONS, MappingProxyType)
    assert isinstance(FACTOR_SET_DEFINITIONS, MappingProxyType)
    assert tuple(FACTOR_DEFINITIONS) == tuple(FactorId)
    assert tuple(FACTOR_SET_DEFINITIONS) == tuple(FactorSetId)

    with pytest.raises(TypeError):
        FACTOR_DEFINITIONS[FactorId.PRICE_MOMENTUM] = FACTOR_DEFINITIONS[FactorId.PRICE_MOMENTUM]  # type: ignore[index]
    with pytest.raises(TypeError):
        FACTOR_SET_DEFINITIONS[FactorSetId.MFBT] = FACTOR_SET_DEFINITIONS[FactorSetId.MFBT]  # type: ignore[index]


def test_factor_definitions_capture_expected_datasets_directions_and_config_hooks() -> None:
    price = get_factor_definition(FactorId.PRICE_MOMENTUM)
    positivity = get_factor_definition(FactorId.POSITIVITY_MOMENTUM)
    earnings = get_factor_definition(FactorId.EARNINGS_MOMENTUM)
    dividend = get_factor_definition(FactorId.DIVIDEND_YIELD)
    retail = get_factor_definition(FactorId.RETAIL_FLOW)
    value = get_factor_definition(FactorId.VALUE)
    ln_market_cap = get_factor_definition(FactorId.LN_MARKET_CAP)
    origin_ln_market_cap = get_factor_definition(FactorId.ORIGIN_LN_MKTCAP)
    origin_momentum = get_factor_definition(FactorId.ORIGIN_MOMENTUM_12M)
    origin_dividend = get_factor_definition(FactorId.ORIGIN_DY)

    assert price == FactorDefinition(id=FactorId.PRICE_MOMENTUM, builder=price.builder, datasets=())
    assert positivity == FactorDefinition(id=FactorId.POSITIVITY_MOMENTUM, builder=positivity.builder, datasets=())
    assert earnings.datasets == (DatasetId.QW_OP_FWD_12M,)
    assert dividend.datasets == (DatasetId.QW_DPS_TTM,)
    assert retail.datasets == (DatasetId.QW_RETAIL,)
    assert retail.requires_construction_sector is True
    assert value.datasets == (
        DatasetId.QW_FCF,
        DatasetId.QW_INT_BEARING_LIAB_NFQ0,
        DatasetId.QW_QUICK_ASSETS_NFQ0,
    )
    assert value.winsor_config_attr == "value_raw_winsor_quantile"
    assert value.zscore_cap_config_attr == "value_zscore_cap"
    assert ln_market_cap.direction is FactorDirection.LOW
    assert ln_market_cap.rank_transform is True
    assert ln_market_cap.neutralize_large_benchmark_weight is True
    assert origin_ln_market_cap.direction is FactorDirection.LOW
    assert origin_ln_market_cap.rank_transform is True
    assert origin_ln_market_cap.neutralize_large_benchmark_weight is False
    assert origin_momentum.direction is FactorDirection.HIGH
    assert origin_dividend.datasets == (DatasetId.QW_DIVIDEND_YLD_FY0,)


def test_factor_set_membership_order_and_metadata_match_expected_contract() -> None:
    assert get_factor_set_definition(FactorSetId.MFBT) == FactorSetDefinition(
        id=FactorSetId.MFBT,
        factors=(
            FactorId.PRICE_MOMENTUM,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
    )
    assert get_factor_set_definition(FactorSetId.MFBT_POS) == FactorSetDefinition(
        id=FactorSetId.MFBT_POS,
        factors=(
            FactorId.POSITIVITY_MOMENTUM,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
    )
    assert get_factor_set_definition(FactorSetId.MFBT_ORIGIN_SMALLCAP) == FactorSetDefinition(
        id=FactorSetId.MFBT_ORIGIN_SMALLCAP,
        factors=(
            FactorId.PRICE_MOMENTUM,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
        constrain_expected_alpha_to_direction=True,
    )
    assert get_factor_set_definition(FactorSetId.ORIGIN) == FactorSetDefinition(
        id=FactorSetId.ORIGIN,
        factors=(FactorId.ORIGIN_LN_MKTCAP, FactorId.ORIGIN_MOMENTUM_12M, FactorId.ORIGIN_DY),
        constrain_expected_alpha_to_direction=True,
        snapshot_forward_days=7,
    )
    assert get_factor_set_definition(FactorSetId.ORIGIN_NEW_DIVIDEND) == FactorSetDefinition(
        id=FactorSetId.ORIGIN_NEW_DIVIDEND,
        factors=(FactorId.ORIGIN_LN_MKTCAP, FactorId.ORIGIN_MOMENTUM_12M, FactorId.DIVIDEND_YIELD),
        constrain_expected_alpha_to_direction=True,
    )


def test_factor_definitions_for_set_preserve_origin_output_names() -> None:
    assert [definition.id.value for definition in factor_definitions_for_set(FactorSetId.ORIGIN)] == [
        "LnMktcap",
        "Momentum_12M",
        "DY",
    ]
    assert [
        definition.id.value for definition in factor_definitions_for_set(FactorSetId.ORIGIN_NEW_DIVIDEND)
    ] == ["LnMktcap", "Momentum_12M", "dividend_yield"]


def test_parse_factor_set_normalizes_string_once_and_errors_with_supported_values() -> None:
    assert parse_factor_set("  origin_new_dividend  ") is FactorSetId.ORIGIN_NEW_DIVIDEND
    assert parse_factor_set(FactorSetId.MFBT) is FactorSetId.MFBT

    with pytest.raises(ValueError, match="mfbt, mfbt_pos, mfbt_origin_smallcap, origin, origin_new_dividend"):
        parse_factor_set("legacy")


def test_factor_set_values_returns_declared_order() -> None:
    assert factor_set_values() == (
        "mfbt",
        "mfbt_pos",
        "mfbt_origin_smallcap",
        "origin",
        "origin_new_dividend",
    )


def test_emp008_config_normalizes_factor_set_to_enum_member() -> None:
    config = Emp008Config(factor_set="origin")

    assert config.factor_set is FactorSetId.ORIGIN


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("rank_transform_factors", ("ln_market_cap",)),
        ("large_bm_neutral_factor_names", ("ln_market_cap",)),
        ("expected_alpha_policy", "mean"),
        ("monthly_snapshot_forward_days", 7),
    ],
)
def test_emp008_config_rejects_removed_metadata_constructor_kwargs(field_name: str, value: object) -> None:
    with pytest.raises(TypeError, match=field_name):
        Emp008Config(**{field_name: value})  # type: ignore[arg-type]


def test_factor_definition_builders_share_config_signature() -> None:
    config = Emp008Config()

    for definition in FACTOR_DEFINITIONS.values():
        assert callable(definition.builder)
        assert definition.id.value
        assert isinstance(definition.direction, FactorDirection)
        assert isinstance(definition.datasets, tuple)
        assert definition.builder.__name__
        assert definition.builder.__globals__
        assert config is not None


def test_validate_registry_rejects_duplicate_factor_ids_within_factor_set() -> None:
    duplicate_set_definitions = dict(FACTOR_SET_DEFINITIONS)
    duplicate_set_definitions[FactorSetId.MFBT] = FactorSetDefinition(
        id=FactorSetId.MFBT,
        factors=(
            FactorId.PRICE_MOMENTUM,
            FactorId.PRICE_MOMENTUM,
        ),
    )

    with pytest.raises(ValueError, match="duplicate factor ids.*mfbt.*price_momentum"):
        _validate_registry(
            factor_definitions=FACTOR_DEFINITIONS,
            factor_set_definitions=duplicate_set_definitions,
        )
