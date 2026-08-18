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
    strategy_factor_set_values,
)


def test_factor_and_factor_set_enums_define_expected_exact_values() -> None:
    assert issubclass(FactorId, Enum)
    assert issubclass(FactorSetId, Enum)
    assert [member.value for member in FactorId] == [
        "price_to_252d_high",
        "positivity_momentum",
        "momentum_12m",
        "momentum_12_1m",
        "earnings_momentum",
        "dividend_yield_ttm",
        "dividend_yield_fy0",
        "retail_flow",
        "value",
        "ln_market_cap",
    ]
    assert all("origin" not in member.name.lower() for member in FactorId)
    assert [member.value for member in FactorSetId] == [
        "mfbt",
        "adjust",
        "mfbt_pos",
        "mfbt_origin_smallcap",
        "origin",
        "origin_new_dividend",
        "origin_12_1m",
        "size_only",
        "size_momentum_12m",
        "size_momentum_12_1m",
        "size_momentum_high",
        "size_earnings_momentum",
        "size_retail_flow",
        "size_value_fcf_tev",
        "size_momentum_earnings_value",
        "size_value_dividend_fy0",
        "size_value_dividend_ttm",
        "all_factors",
    ]
    assert [member.value for member in FactorDirection] == ["high", "low"]


def test_strategy_sets_select_independent_factor_ids_and_own_neutralization_policy() -> (
    None
):
    mfbt = get_factor_set_definition(FactorSetId.MFBT)
    adjust = get_factor_set_definition(FactorSetId.ADJUST)
    origin = get_factor_set_definition(FactorSetId.ORIGIN)
    origin_new_dividend = get_factor_set_definition(FactorSetId.ORIGIN_NEW_DIVIDEND)
    origin_12_1m = get_factor_set_definition(FactorSetId.ORIGIN_12_1M)
    size_value_fcf_tev = get_factor_set_definition(FactorSetId.SIZE_VALUE_FCF_TEV)
    size_retail_flow = get_factor_set_definition(FactorSetId.SIZE_RETAIL_FLOW)
    size_value_dividend_fy0 = get_factor_set_definition(
        FactorSetId.SIZE_VALUE_DIVIDEND_FY0
    )
    size_value_dividend_ttm = get_factor_set_definition(
        FactorSetId.SIZE_VALUE_DIVIDEND_TTM
    )
    all_factors = get_factor_set_definition(FactorSetId.ALL_FACTORS)

    assert mfbt.factors == (
        FactorId.PRICE_TO_252D_HIGH,
        FactorId.EARNINGS_MOMENTUM,
        FactorId.DIVIDEND_YIELD_TTM,
        FactorId.RETAIL_FLOW,
        FactorId.VALUE,
        FactorId.LN_MARKET_CAP,
    )
    assert mfbt.neutralize_large_benchmark_weight_factors == (FactorId.LN_MARKET_CAP,)
    assert adjust.factors == (
        FactorId.MOMENTUM_12_1M,
        FactorId.EARNINGS_MOMENTUM,
        FactorId.DIVIDEND_YIELD_TTM,
        FactorId.VALUE,
        FactorId.LN_MARKET_CAP,
    )
    assert FactorId.PRICE_TO_252D_HIGH not in adjust.factors
    assert FactorId.RETAIL_FLOW not in adjust.factors
    assert adjust.neutralize_large_benchmark_weight_factors == (FactorId.LN_MARKET_CAP,)
    assert origin.factors == (
        FactorId.LN_MARKET_CAP,
        FactorId.MOMENTUM_12M,
        FactorId.DIVIDEND_YIELD_FY0,
    )
    assert origin.neutralize_large_benchmark_weight_factors == ()
    assert origin_new_dividend.factors == (
        FactorId.LN_MARKET_CAP,
        FactorId.MOMENTUM_12M,
        FactorId.DIVIDEND_YIELD_TTM,
    )
    assert origin_12_1m.factors == (
        FactorId.LN_MARKET_CAP,
        FactorId.MOMENTUM_12_1M,
        FactorId.DIVIDEND_YIELD_FY0,
    )
    assert origin_12_1m.neutralize_large_benchmark_weight_factors == ()
    assert size_value_fcf_tev.factors == (
        FactorId.LN_MARKET_CAP,
        FactorId.VALUE,
    )
    assert size_retail_flow.factors == (
        FactorId.LN_MARKET_CAP,
        FactorId.RETAIL_FLOW,
    )
    assert size_retail_flow.neutralize_large_benchmark_weight_factors == ()
    assert size_value_fcf_tev.neutralize_large_benchmark_weight_factors == ()
    assert size_value_dividend_fy0.factors == (
        FactorId.LN_MARKET_CAP,
        FactorId.DIVIDEND_YIELD_FY0,
    )
    assert size_value_dividend_ttm.factors == (
        FactorId.LN_MARKET_CAP,
        FactorId.DIVIDEND_YIELD_TTM,
    )
    assert all_factors.factors == tuple(FactorId)
    assert len(all_factors.factors) == len(set(all_factors.factors))
    assert all_factors.diagnostics_only is True


def test_registry_exposes_read_only_complete_mappings() -> None:
    assert isinstance(FACTOR_DEFINITIONS, MappingProxyType)
    assert isinstance(FACTOR_SET_DEFINITIONS, MappingProxyType)
    assert tuple(FACTOR_DEFINITIONS) == tuple(FactorId)
    assert tuple(FACTOR_SET_DEFINITIONS) == tuple(FactorSetId)

    with pytest.raises(TypeError):
        FACTOR_DEFINITIONS[FactorId.PRICE_TO_252D_HIGH] = FACTOR_DEFINITIONS[
            FactorId.PRICE_TO_252D_HIGH
        ]  # type: ignore[index]
    with pytest.raises(TypeError):
        FACTOR_SET_DEFINITIONS[FactorSetId.MFBT] = FACTOR_SET_DEFINITIONS[
            FactorSetId.MFBT
        ]  # type: ignore[index]


def test_factor_definitions_capture_expected_datasets_directions_and_config_hooks() -> (
    None
):
    price = get_factor_definition(FactorId.PRICE_TO_252D_HIGH)
    positivity = get_factor_definition(FactorId.POSITIVITY_MOMENTUM)
    momentum_12m = get_factor_definition(FactorId.MOMENTUM_12M)
    momentum_12_1m = get_factor_definition(FactorId.MOMENTUM_12_1M)
    earnings = get_factor_definition(FactorId.EARNINGS_MOMENTUM)
    dividend_ttm = get_factor_definition(FactorId.DIVIDEND_YIELD_TTM)
    dividend_fy0 = get_factor_definition(FactorId.DIVIDEND_YIELD_FY0)
    retail = get_factor_definition(FactorId.RETAIL_FLOW)
    value = get_factor_definition(FactorId.VALUE)
    ln_market_cap = get_factor_definition(FactorId.LN_MARKET_CAP)

    assert price == FactorDefinition(
        id=FactorId.PRICE_TO_252D_HIGH, builder=price.builder, datasets=()
    )
    assert positivity == FactorDefinition(
        id=FactorId.POSITIVITY_MOMENTUM, builder=positivity.builder, datasets=()
    )
    assert momentum_12m.datasets == ()
    assert momentum_12_1m.datasets == ()
    assert earnings.datasets == (DatasetId.QW_OP_FWD_12M,)
    assert dividend_ttm.datasets == (DatasetId.QW_DPS_TTM,)
    assert dividend_fy0.datasets == (DatasetId.QW_DIVIDEND_YLD_FY0,)
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
    assert not hasattr(ln_market_cap, "rank_transform")
    assert not hasattr(ln_market_cap, "neutralize_large_benchmark_weight")


def test_factor_set_membership_order_and_metadata_match_expected_contract() -> None:
    assert get_factor_set_definition(FactorSetId.MFBT) == FactorSetDefinition(
        id=FactorSetId.MFBT,
        factors=(
            FactorId.PRICE_TO_252D_HIGH,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD_TTM,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
        rank_transform_factors=(FactorId.LN_MARKET_CAP,),
        neutralize_large_benchmark_weight_factors=(FactorId.LN_MARKET_CAP,),
    )
    assert get_factor_set_definition(FactorSetId.MFBT_POS) == FactorSetDefinition(
        id=FactorSetId.MFBT_POS,
        factors=(
            FactorId.POSITIVITY_MOMENTUM,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD_TTM,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
        rank_transform_factors=(FactorId.LN_MARKET_CAP,),
        neutralize_large_benchmark_weight_factors=(FactorId.LN_MARKET_CAP,),
    )
    assert get_factor_set_definition(
        FactorSetId.MFBT_ORIGIN_SMALLCAP
    ) == FactorSetDefinition(
        id=FactorSetId.MFBT_ORIGIN_SMALLCAP,
        factors=(
            FactorId.PRICE_TO_252D_HIGH,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD_TTM,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
        rank_transform_factors=(FactorId.LN_MARKET_CAP,),
        neutralize_large_benchmark_weight_factors=(FactorId.LN_MARKET_CAP,),
        constrain_expected_alpha_to_direction=True,
    )
    assert get_factor_set_definition(FactorSetId.ORIGIN) == FactorSetDefinition(
        id=FactorSetId.ORIGIN,
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.MOMENTUM_12M,
            FactorId.DIVIDEND_YIELD_FY0,
        ),
        constrain_expected_alpha_to_direction=True,
        snapshot_forward_days=7,
    )
    assert get_factor_set_definition(
        FactorSetId.ORIGIN_NEW_DIVIDEND
    ) == FactorSetDefinition(
        id=FactorSetId.ORIGIN_NEW_DIVIDEND,
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.MOMENTUM_12M,
            FactorId.DIVIDEND_YIELD_TTM,
        ),
        constrain_expected_alpha_to_direction=True,
    )
    assert get_factor_set_definition(FactorSetId.ORIGIN_12_1M) == FactorSetDefinition(
        id=FactorSetId.ORIGIN_12_1M,
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.MOMENTUM_12_1M,
            FactorId.DIVIDEND_YIELD_FY0,
        ),
        constrain_expected_alpha_to_direction=True,
        snapshot_forward_days=7,
    )


@pytest.mark.parametrize(
    (
        "factor_set_id",
        "expected_factors",
        "expected_rank_transform_factors",
        "expected_neutralization_factors",
        "expected_snapshot_forward_days",
        "expected_extra_datasets",
    ),
    [
        pytest.param(
            FactorSetId.SIZE_ONLY,
            (FactorId.LN_MARKET_CAP,),
            (),
            (),
            0,
            (),
            id="size-only",
        ),
        pytest.param(
            FactorSetId.SIZE_MOMENTUM_12M,
            (FactorId.LN_MARKET_CAP, FactorId.MOMENTUM_12M),
            (),
            (),
            0,
            (),
            id="size-momentum-12m",
        ),
        pytest.param(
            FactorSetId.SIZE_MOMENTUM_12_1M,
            (FactorId.LN_MARKET_CAP, FactorId.MOMENTUM_12_1M),
            (),
            (),
            0,
            (),
            id="size-momentum-12-1m",
        ),
        pytest.param(
            FactorSetId.SIZE_MOMENTUM_HIGH,
            (FactorId.LN_MARKET_CAP, FactorId.PRICE_TO_252D_HIGH),
            (),
            (),
            0,
            (),
            id="size-momentum-high",
        ),
        pytest.param(
            FactorSetId.SIZE_EARNINGS_MOMENTUM,
            (FactorId.LN_MARKET_CAP, FactorId.EARNINGS_MOMENTUM),
            (),
            (),
            0,
            (DatasetId.QW_OP_FWD_12M,),
            id="size-earnings-momentum",
        ),
        pytest.param(
            FactorSetId.SIZE_RETAIL_FLOW,
            (FactorId.LN_MARKET_CAP, FactorId.RETAIL_FLOW),
            (),
            (),
            0,
            (DatasetId.QW_RETAIL,),
            id="size-retail-flow",
        ),
        pytest.param(
            FactorSetId.SIZE_VALUE_FCF_TEV,
            (FactorId.LN_MARKET_CAP, FactorId.VALUE),
            (),
            (),
            0,
            (
                DatasetId.QW_FCF,
                DatasetId.QW_INT_BEARING_LIAB_NFQ0,
                DatasetId.QW_QUICK_ASSETS_NFQ0,
            ),
            id="size-value-fcf-tev",
        ),
        pytest.param(
            FactorSetId.SIZE_MOMENTUM_EARNINGS_VALUE,
            (
                FactorId.LN_MARKET_CAP,
                FactorId.MOMENTUM_12M,
                FactorId.EARNINGS_MOMENTUM,
                FactorId.VALUE,
            ),
            (),
            (),
            0,
            (
                DatasetId.QW_OP_FWD_12M,
                DatasetId.QW_FCF,
                DatasetId.QW_INT_BEARING_LIAB_NFQ0,
                DatasetId.QW_QUICK_ASSETS_NFQ0,
            ),
            id="size-momentum-earnings-value",
        ),
        pytest.param(
            FactorSetId.SIZE_VALUE_DIVIDEND_FY0,
            (FactorId.LN_MARKET_CAP, FactorId.DIVIDEND_YIELD_FY0),
            (),
            (),
            7,
            (DatasetId.QW_DIVIDEND_YLD_FY0,),
            id="size-value-dividend-fy0",
        ),
        pytest.param(
            FactorSetId.SIZE_VALUE_DIVIDEND_TTM,
            (FactorId.LN_MARKET_CAP, FactorId.DIVIDEND_YIELD_TTM),
            (),
            (),
            0,
            (DatasetId.QW_DPS_TTM,),
            id="size-value-dividend-ttm",
        ),
    ],
)
def test_size_value_factor_set_contracts_are_exact(
    factor_set_id: FactorSetId,
    expected_factors: tuple[FactorId, ...],
    expected_rank_transform_factors: tuple[FactorId, ...],
    expected_neutralization_factors: tuple[FactorId, ...],
    expected_snapshot_forward_days: int,
    expected_extra_datasets: tuple[DatasetId, ...],
) -> None:
    definition = get_factor_set_definition(factor_set_id)
    factor_definitions = factor_definitions_for_set(factor_set_id)

    assert definition.factors == expected_factors
    assert definition.rank_transform_factors == expected_rank_transform_factors
    assert (
        definition.neutralize_large_benchmark_weight_factors
        == expected_neutralization_factors
    )
    assert definition.constrain_expected_alpha_to_direction is True
    assert definition.snapshot_forward_days == expected_snapshot_forward_days
    assert definition.diagnostics_only is False
    assert factor_definitions[0].id is FactorId.LN_MARKET_CAP
    assert factor_definitions[0].datasets == ()
    assert (
        tuple(
            dataset_id
            for factor_definition in factor_definitions[1:]
            for dataset_id in factor_definition.datasets
        )
        == expected_extra_datasets
    )


def test_all_factors_membership_order_and_metadata_match_expected_contract() -> None:
    assert get_factor_set_definition(FactorSetId.ALL_FACTORS) == FactorSetDefinition(
        id=FactorSetId.ALL_FACTORS,
        factors=tuple(FactorId),
        snapshot_forward_days=7,
        diagnostics_only=True,
    )


def test_factor_definitions_for_set_use_factor_names_independent_of_strategy() -> None:
    assert [
        definition.id.value
        for definition in factor_definitions_for_set(FactorSetId.ORIGIN)
    ] == [
        "ln_market_cap",
        "momentum_12m",
        "dividend_yield_fy0",
    ]
    assert [
        definition.id.value
        for definition in factor_definitions_for_set(FactorSetId.ORIGIN_NEW_DIVIDEND)
    ] == ["ln_market_cap", "momentum_12m", "dividend_yield_ttm"]


def test_parse_factor_set_normalizes_string_once_and_errors_with_supported_values() -> (
    None
):
    assert (
        parse_factor_set("  origin_new_dividend  ") is FactorSetId.ORIGIN_NEW_DIVIDEND
    )
    assert parse_factor_set(" adjust ") is FactorSetId.ADJUST
    assert parse_factor_set(FactorSetId.MFBT) is FactorSetId.MFBT

    with pytest.raises(
        ValueError,
        match=(
            "mfbt, adjust, mfbt_pos, mfbt_origin_smallcap, origin, origin_new_dividend, "
            "origin_12_1m, size_only, size_momentum_12m, size_momentum_12_1m, "
            "size_momentum_high, size_earnings_momentum, size_retail_flow, size_value_fcf_tev, "
            "size_momentum_earnings_value, "
            "size_value_dividend_fy0, "
            "size_value_dividend_ttm, all_factors"
        ),
    ):
        parse_factor_set("legacy")


def test_factor_set_values_returns_declared_order() -> None:
    assert factor_set_values() == (
        "mfbt",
        "adjust",
        "mfbt_pos",
        "mfbt_origin_smallcap",
        "origin",
        "origin_new_dividend",
        "origin_12_1m",
        "size_only",
        "size_momentum_12m",
        "size_momentum_12_1m",
        "size_momentum_high",
        "size_earnings_momentum",
        "size_retail_flow",
        "size_value_fcf_tev",
        "size_momentum_earnings_value",
        "size_value_dividend_fy0",
        "size_value_dividend_ttm",
        "all_factors",
    )


def test_strategy_factor_set_values_excludes_diagnostics_only_sets() -> None:
    assert strategy_factor_set_values() == (
        "mfbt",
        "adjust",
        "mfbt_pos",
        "mfbt_origin_smallcap",
        "origin",
        "origin_new_dividend",
        "origin_12_1m",
        "size_only",
        "size_momentum_12m",
        "size_momentum_12_1m",
        "size_momentum_high",
        "size_earnings_momentum",
        "size_retail_flow",
        "size_value_fcf_tev",
        "size_momentum_earnings_value",
        "size_value_dividend_fy0",
        "size_value_dividend_ttm",
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
def test_emp008_config_rejects_removed_metadata_constructor_kwargs(
    field_name: str, value: object
) -> None:
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
            FactorId.PRICE_TO_252D_HIGH,
            FactorId.PRICE_TO_252D_HIGH,
        ),
    )

    with pytest.raises(
        ValueError, match="duplicate factor ids.*mfbt.*price_to_252d_high"
    ):
        _validate_registry(
            factor_definitions=FACTOR_DEFINITIONS,
            factor_set_definitions=duplicate_set_definitions,
        )


def test_validate_registry_rejects_strategy_policy_for_unselected_factor() -> None:
    invalid_set_definitions = dict(FACTOR_SET_DEFINITIONS)
    invalid_set_definitions[FactorSetId.ORIGIN] = FactorSetDefinition(
        id=FactorSetId.ORIGIN,
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.MOMENTUM_12M,
            FactorId.DIVIDEND_YIELD_FY0,
        ),
        neutralize_large_benchmark_weight_factors=(FactorId.VALUE,),
    )

    with pytest.raises(ValueError, match="neutralization.*origin.*value"):
        _validate_registry(
            factor_definitions=FACTOR_DEFINITIONS,
            factor_set_definitions=invalid_set_definitions,
        )
