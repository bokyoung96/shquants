from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping

import pandas as pd

from data.catalog import DatasetId
from data.loader import MarketData

from . import factor_builders as builders

if TYPE_CHECKING:
    from .data import Emp008Config

FactorBuilder = Callable[[MarketData, "Emp008Config"], pd.DataFrame]


@unique
class FactorId(StrEnum):
    PRICE_TO_252D_HIGH = "price_to_252d_high"
    POSITIVITY_MOMENTUM = "positivity_momentum"
    MOMENTUM_12M = "momentum_12m"
    MOMENTUM_12_1M = "momentum_12_1m"
    EARNINGS_MOMENTUM = "earnings_momentum"
    DIVIDEND_YIELD_TTM = "dividend_yield_ttm"
    DIVIDEND_YIELD_FY0 = "dividend_yield_fy0"
    RETAIL_FLOW = "retail_flow"
    VALUE = "value"
    LN_MARKET_CAP = "ln_market_cap"


@unique
class FactorSetId(StrEnum):
    PRODUCTION_CORE = "production_core"
    RESEARCH_12_1M_MOMENTUM = "research_12_1m_momentum"
    RESEARCH_POSITIVITY_MOMENTUM = "research_positivity_momentum"
    RESEARCH_ORIGIN_SMALL_CAP_RULE = "research_origin_small_cap_rule"
    REFERENCE_ORIGIN = "reference_origin"
    REFERENCE_ORIGIN_TTM_DIVIDEND = "reference_origin_ttm_dividend"
    REFERENCE_ORIGIN_12_1M = "reference_origin_12_1m"
    RESEARCH_SIZE_ONLY = "research_size_only"
    RESEARCH_SIZE_MOMENTUM_12M = "research_size_momentum_12m"
    RESEARCH_SIZE_MOMENTUM_12_1M = "research_size_momentum_12_1m"
    RESEARCH_SIZE_MOMENTUM_HIGH = "research_size_momentum_high"
    RESEARCH_SIZE_EARNINGS_MOMENTUM = "research_size_earnings_momentum"
    RESEARCH_SIZE_RETAIL_FLOW = "research_size_retail_flow"
    RESEARCH_SIZE_VALUE_FCF_TEV = "research_size_value_fcf_tev"
    RESEARCH_SIZE_MOMENTUM_EARNINGS_VALUE = "research_size_momentum_earnings_value"
    RESEARCH_SIZE_VALUE_DIVIDEND_FY0 = "research_size_value_dividend_fy0"
    RESEARCH_SIZE_VALUE_DIVIDEND_TTM = "research_size_value_dividend_ttm"
    DIAGNOSTIC_ALL_FACTORS = "diagnostic_all_factors"


@unique
class FactorDirection(StrEnum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    id: FactorId
    builder: FactorBuilder
    datasets: tuple[DatasetId, ...]
    direction: FactorDirection = FactorDirection.HIGH
    winsor_config_attr: str | None = None
    zscore_cap_config_attr: str | None = None
    requires_construction_sector: bool = False


@dataclass(frozen=True, slots=True)
class FactorSetDefinition:
    id: FactorSetId
    factors: tuple[FactorId, ...]
    category: str = "research"
    label: str = ""
    rank_transform_factors: tuple[FactorId, ...] = ()
    neutralize_large_benchmark_weight_factors: tuple[FactorId, ...] = ()
    constrain_expected_alpha_to_direction: bool = False
    snapshot_forward_days: int = 0
    diagnostics_only: bool = False


_FACTOR_DEFINITIONS = {
    FactorId.PRICE_TO_252D_HIGH: FactorDefinition(
        id=FactorId.PRICE_TO_252D_HIGH,
        builder=builders.build_price_to_252d_high,
        datasets=(),
    ),
    FactorId.POSITIVITY_MOMENTUM: FactorDefinition(
        id=FactorId.POSITIVITY_MOMENTUM,
        builder=builders.build_positivity_momentum,
        datasets=(),
    ),
    FactorId.MOMENTUM_12M: FactorDefinition(
        id=FactorId.MOMENTUM_12M,
        builder=builders.build_momentum_12m,
        datasets=(),
    ),
    FactorId.MOMENTUM_12_1M: FactorDefinition(
        id=FactorId.MOMENTUM_12_1M,
        builder=builders.build_momentum_12_1m,
        datasets=(),
    ),
    FactorId.EARNINGS_MOMENTUM: FactorDefinition(
        id=FactorId.EARNINGS_MOMENTUM,
        builder=builders.build_earnings_momentum,
        datasets=(DatasetId.QW_OP_FWD_12M,),
    ),
    FactorId.DIVIDEND_YIELD_TTM: FactorDefinition(
        id=FactorId.DIVIDEND_YIELD_TTM,
        builder=builders.build_dividend_yield_ttm,
        datasets=(DatasetId.QW_DPS_TTM,),
    ),
    FactorId.DIVIDEND_YIELD_FY0: FactorDefinition(
        id=FactorId.DIVIDEND_YIELD_FY0,
        builder=builders.build_dividend_yield_fy0,
        datasets=(DatasetId.QW_DIVIDEND_YLD_FY0,),
    ),
    FactorId.RETAIL_FLOW: FactorDefinition(
        id=FactorId.RETAIL_FLOW,
        builder=builders.build_retail_flow,
        datasets=(DatasetId.QW_RETAIL,),
        requires_construction_sector=True,
    ),
    FactorId.VALUE: FactorDefinition(
        id=FactorId.VALUE,
        builder=builders.build_value,
        datasets=(
            DatasetId.QW_FCF,
            DatasetId.QW_INT_BEARING_LIAB_NFQ0,
            DatasetId.QW_QUICK_ASSETS_NFQ0,
        ),
        winsor_config_attr="value_raw_winsor_quantile",
        zscore_cap_config_attr="value_zscore_cap",
    ),
    FactorId.LN_MARKET_CAP: FactorDefinition(
        id=FactorId.LN_MARKET_CAP,
        builder=builders.build_ln_market_cap,
        datasets=(),
        direction=FactorDirection.LOW,
    ),
}

_FACTOR_SET_DEFINITIONS = {
    FactorSetId.PRODUCTION_CORE: FactorSetDefinition(
        id=FactorSetId.PRODUCTION_CORE,
        category="production",
        label="Core price / earnings / dividend / retail / value / size",
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
    ),
    FactorSetId.RESEARCH_12_1M_MOMENTUM: FactorSetDefinition(
        id=FactorSetId.RESEARCH_12_1M_MOMENTUM,
        factors=(
            FactorId.MOMENTUM_12_1M,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD_TTM,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
        rank_transform_factors=(FactorId.LN_MARKET_CAP,),
        neutralize_large_benchmark_weight_factors=(FactorId.LN_MARKET_CAP,),
    ),
    FactorSetId.RESEARCH_POSITIVITY_MOMENTUM: FactorSetDefinition(
        id=FactorSetId.RESEARCH_POSITIVITY_MOMENTUM,
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
    ),
    FactorSetId.RESEARCH_ORIGIN_SMALL_CAP_RULE: FactorSetDefinition(
        id=FactorSetId.RESEARCH_ORIGIN_SMALL_CAP_RULE,
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
    ),
    FactorSetId.REFERENCE_ORIGIN: FactorSetDefinition(
        id=FactorSetId.REFERENCE_ORIGIN,
        category="reference",
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.MOMENTUM_12M,
            FactorId.DIVIDEND_YIELD_FY0,
        ),
        constrain_expected_alpha_to_direction=True,
        snapshot_forward_days=7,
    ),
    FactorSetId.REFERENCE_ORIGIN_TTM_DIVIDEND: FactorSetDefinition(
        id=FactorSetId.REFERENCE_ORIGIN_TTM_DIVIDEND,
        category="reference",
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.MOMENTUM_12M,
            FactorId.DIVIDEND_YIELD_TTM,
        ),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.REFERENCE_ORIGIN_12_1M: FactorSetDefinition(
        id=FactorSetId.REFERENCE_ORIGIN_12_1M,
        category="reference",
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.MOMENTUM_12_1M,
            FactorId.DIVIDEND_YIELD_FY0,
        ),
        constrain_expected_alpha_to_direction=True,
        snapshot_forward_days=7,
    ),
    FactorSetId.RESEARCH_SIZE_ONLY: FactorSetDefinition(
        id=FactorSetId.RESEARCH_SIZE_ONLY,
        factors=(FactorId.LN_MARKET_CAP,),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.RESEARCH_SIZE_MOMENTUM_12M: FactorSetDefinition(
        id=FactorSetId.RESEARCH_SIZE_MOMENTUM_12M,
        factors=(FactorId.LN_MARKET_CAP, FactorId.MOMENTUM_12M),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.RESEARCH_SIZE_MOMENTUM_12_1M: FactorSetDefinition(
        id=FactorSetId.RESEARCH_SIZE_MOMENTUM_12_1M,
        factors=(FactorId.LN_MARKET_CAP, FactorId.MOMENTUM_12_1M),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.RESEARCH_SIZE_MOMENTUM_HIGH: FactorSetDefinition(
        id=FactorSetId.RESEARCH_SIZE_MOMENTUM_HIGH,
        factors=(FactorId.LN_MARKET_CAP, FactorId.PRICE_TO_252D_HIGH),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.RESEARCH_SIZE_EARNINGS_MOMENTUM: FactorSetDefinition(
        id=FactorSetId.RESEARCH_SIZE_EARNINGS_MOMENTUM,
        factors=(FactorId.LN_MARKET_CAP, FactorId.EARNINGS_MOMENTUM),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.RESEARCH_SIZE_RETAIL_FLOW: FactorSetDefinition(
        id=FactorSetId.RESEARCH_SIZE_RETAIL_FLOW,
        factors=(FactorId.LN_MARKET_CAP, FactorId.RETAIL_FLOW),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.RESEARCH_SIZE_VALUE_FCF_TEV: FactorSetDefinition(
        id=FactorSetId.RESEARCH_SIZE_VALUE_FCF_TEV,
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.VALUE,
        ),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.RESEARCH_SIZE_MOMENTUM_EARNINGS_VALUE: FactorSetDefinition(
        id=FactorSetId.RESEARCH_SIZE_MOMENTUM_EARNINGS_VALUE,
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.MOMENTUM_12M,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.VALUE,
        ),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.RESEARCH_SIZE_VALUE_DIVIDEND_FY0: FactorSetDefinition(
        id=FactorSetId.RESEARCH_SIZE_VALUE_DIVIDEND_FY0,
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.DIVIDEND_YIELD_FY0,
        ),
        constrain_expected_alpha_to_direction=True,
        snapshot_forward_days=7,
    ),
    FactorSetId.RESEARCH_SIZE_VALUE_DIVIDEND_TTM: FactorSetDefinition(
        id=FactorSetId.RESEARCH_SIZE_VALUE_DIVIDEND_TTM,
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.DIVIDEND_YIELD_TTM,
        ),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.DIAGNOSTIC_ALL_FACTORS: FactorSetDefinition(
        id=FactorSetId.DIAGNOSTIC_ALL_FACTORS,
        category="diagnostic",
        factors=tuple(FactorId),
        snapshot_forward_days=7,
        diagnostics_only=True,
    ),
}

FACTOR_DEFINITIONS = MappingProxyType(_FACTOR_DEFINITIONS)
FACTOR_SET_DEFINITIONS = MappingProxyType(_FACTOR_SET_DEFINITIONS)

LEGACY_FACTOR_SET_ALIASES = MappingProxyType({
    "mfbt": FactorSetId.PRODUCTION_CORE,
    "adjust": FactorSetId.RESEARCH_12_1M_MOMENTUM,
    "mfbt_pos": FactorSetId.RESEARCH_POSITIVITY_MOMENTUM,
    "mfbt_origin_smallcap": FactorSetId.RESEARCH_ORIGIN_SMALL_CAP_RULE,
    "origin": FactorSetId.REFERENCE_ORIGIN,
    "origin_new_dividend": FactorSetId.REFERENCE_ORIGIN_TTM_DIVIDEND,
    "origin_12_1m": FactorSetId.REFERENCE_ORIGIN_12_1M,
    "size_only": FactorSetId.RESEARCH_SIZE_ONLY,
    "size_momentum_12m": FactorSetId.RESEARCH_SIZE_MOMENTUM_12M,
    "size_momentum_12_1m": FactorSetId.RESEARCH_SIZE_MOMENTUM_12_1M,
    "size_momentum_high": FactorSetId.RESEARCH_SIZE_MOMENTUM_HIGH,
    "size_earnings_momentum": FactorSetId.RESEARCH_SIZE_EARNINGS_MOMENTUM,
    "size_retail_flow": FactorSetId.RESEARCH_SIZE_RETAIL_FLOW,
    "size_value_fcf_tev": FactorSetId.RESEARCH_SIZE_VALUE_FCF_TEV,
    "size_momentum_earnings_value": FactorSetId.RESEARCH_SIZE_MOMENTUM_EARNINGS_VALUE,
    "size_value_dividend_fy0": FactorSetId.RESEARCH_SIZE_VALUE_DIVIDEND_FY0,
    "size_value_dividend_ttm": FactorSetId.RESEARCH_SIZE_VALUE_DIVIDEND_TTM,
    "all_factors": FactorSetId.DIAGNOSTIC_ALL_FACTORS,
})


def get_factor_definition(factor_id: FactorId) -> FactorDefinition:
    return FACTOR_DEFINITIONS[factor_id]


def get_factor_set_definition(factor_set: FactorSetId | str) -> FactorSetDefinition:
    return FACTOR_SET_DEFINITIONS[parse_factor_set(factor_set)]


def factor_definitions_for_set(
    factor_set: FactorSetId | str,
) -> tuple[FactorDefinition, ...]:
    definition = get_factor_set_definition(factor_set)
    return tuple(get_factor_definition(factor_id) for factor_id in definition.factors)


def factor_set_values() -> tuple[str, ...]:
    return tuple(factor_set_id.value for factor_set_id in FactorSetId)


def strategy_factor_set_values() -> tuple[str, ...]:
    return tuple(
        factor_set_id.value
        for factor_set_id, definition in FACTOR_SET_DEFINITIONS.items()
        if not definition.diagnostics_only
    )


def parse_factor_set(value: FactorSetId | str) -> FactorSetId:
    if isinstance(value, FactorSetId):
        return value
    normalized = value.strip()
    try:
        return FactorSetId(normalized)
    except ValueError as exc:
        if normalized in LEGACY_FACTOR_SET_ALIASES:
            return LEGACY_FACTOR_SET_ALIASES[normalized]
        supported = ", ".join(factor_set_values())
        raise ValueError(
            f"unknown factor set '{value}'. Supported values: {supported}"
        ) from exc


def _validate_registry(
    *,
    factor_definitions: Mapping[FactorId, FactorDefinition] = FACTOR_DEFINITIONS,
    factor_set_definitions: Mapping[
        FactorSetId, FactorSetDefinition
    ] = FACTOR_SET_DEFINITIONS,
) -> None:
    if tuple(factor_definitions) != tuple(FactorId):
        raise ValueError("every FactorId must have exactly one definition")
    if tuple(factor_set_definitions) != tuple(FactorSetId):
        raise ValueError("every FactorSetId must have exactly one definition")
    for factor_set_id, definition in factor_set_definitions.items():
        if not definition.factors:
            raise ValueError(f"factor set '{factor_set_id.value}' must not be empty")
        seen: set[FactorId] = set()
        for factor_id in definition.factors:
            if factor_id in seen:
                raise ValueError(
                    f"duplicate factor ids in factor set '{factor_set_id.value}': {factor_id.value}"
                )
            seen.add(factor_id)
            if factor_id not in factor_definitions:
                raise ValueError(
                    f"factor set '{factor_set_id.value}' references undefined factor '{factor_id.value}'"
                )
        neutralization_ids = definition.neutralize_large_benchmark_weight_factors
        if len(neutralization_ids) != len(set(neutralization_ids)):
            raise ValueError(
                f"duplicate neutralization factor ids in factor set '{factor_set_id.value}'"
            )
        for factor_id in neutralization_ids:
            if factor_id not in definition.factors:
                raise ValueError(
                    f"neutralization policy in factor set '{factor_set_id.value}' references unselected factor "
                    f"'{factor_id.value}'"
                )


_validate_registry()


__all__ = [
    "FACTOR_DEFINITIONS",
    "FACTOR_SET_DEFINITIONS",
    "FactorDefinition",
    "FactorDirection",
    "FactorId",
    "FactorSetDefinition",
    "FactorSetId",
    "factor_definitions_for_set",
    "factor_set_values",
    "get_factor_definition",
    "get_factor_set_definition",
    "parse_factor_set",
    "strategy_factor_set_values",
]
