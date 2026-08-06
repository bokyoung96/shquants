from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from . import factor_builders as builders

if TYPE_CHECKING:
    from .data import Emp008Config

FactorBuilder = Callable[[MarketData, "Emp008Config"], pd.DataFrame]


@unique
class FactorId(StrEnum):
    PRICE_TO_252D_HIGH = "price_to_252d_high"
    POSITIVITY_MOMENTUM = "positivity_momentum"
    MOMENTUM_12M = "momentum_12m"
    EARNINGS_MOMENTUM = "earnings_momentum"
    DIVIDEND_YIELD_TTM = "dividend_yield_ttm"
    DIVIDEND_YIELD_FY0 = "dividend_yield_fy0"
    RETAIL_FLOW = "retail_flow"
    VALUE = "value"
    LN_MARKET_CAP = "ln_market_cap"


@unique
class FactorSetId(StrEnum):
    MFBT = "mfbt"
    MFBT_POS = "mfbt_pos"
    MFBT_ORIGIN_SMALLCAP = "mfbt_origin_smallcap"
    ORIGIN = "origin"
    ORIGIN_NEW_DIVIDEND = "origin_new_dividend"


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
    rank_transform: bool = False
    winsor_config_attr: str | None = None
    zscore_cap_config_attr: str | None = None
    requires_construction_sector: bool = False


@dataclass(frozen=True, slots=True)
class FactorSetDefinition:
    id: FactorSetId
    factors: tuple[FactorId, ...]
    neutralize_large_benchmark_weight_factors: tuple[FactorId, ...] = ()
    constrain_expected_alpha_to_direction: bool = False
    snapshot_forward_days: int = 0


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
        rank_transform=True,
    ),
}

_FACTOR_SET_DEFINITIONS = {
    FactorSetId.MFBT: FactorSetDefinition(
        id=FactorSetId.MFBT,
        factors=(
            FactorId.PRICE_TO_252D_HIGH,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD_TTM,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
        neutralize_large_benchmark_weight_factors=(FactorId.LN_MARKET_CAP,),
    ),
    FactorSetId.MFBT_POS: FactorSetDefinition(
        id=FactorSetId.MFBT_POS,
        factors=(
            FactorId.POSITIVITY_MOMENTUM,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD_TTM,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
        neutralize_large_benchmark_weight_factors=(FactorId.LN_MARKET_CAP,),
    ),
    FactorSetId.MFBT_ORIGIN_SMALLCAP: FactorSetDefinition(
        id=FactorSetId.MFBT_ORIGIN_SMALLCAP,
        factors=(
            FactorId.PRICE_TO_252D_HIGH,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD_TTM,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
        neutralize_large_benchmark_weight_factors=(FactorId.LN_MARKET_CAP,),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.ORIGIN: FactorSetDefinition(
        id=FactorSetId.ORIGIN,
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.MOMENTUM_12M,
            FactorId.DIVIDEND_YIELD_FY0,
        ),
        constrain_expected_alpha_to_direction=True,
        snapshot_forward_days=7,
    ),
    FactorSetId.ORIGIN_NEW_DIVIDEND: FactorSetDefinition(
        id=FactorSetId.ORIGIN_NEW_DIVIDEND,
        factors=(
            FactorId.LN_MARKET_CAP,
            FactorId.MOMENTUM_12M,
            FactorId.DIVIDEND_YIELD_TTM,
        ),
        constrain_expected_alpha_to_direction=True,
    ),
}

FACTOR_DEFINITIONS = MappingProxyType(_FACTOR_DEFINITIONS)
FACTOR_SET_DEFINITIONS = MappingProxyType(_FACTOR_SET_DEFINITIONS)


def get_factor_definition(factor_id: FactorId) -> FactorDefinition:
    return FACTOR_DEFINITIONS[factor_id]


def get_factor_set_definition(factor_set: FactorSetId | str) -> FactorSetDefinition:
    return FACTOR_SET_DEFINITIONS[parse_factor_set(factor_set)]


def factor_definitions_for_set(factor_set: FactorSetId | str) -> tuple[FactorDefinition, ...]:
    definition = get_factor_set_definition(factor_set)
    return tuple(get_factor_definition(factor_id) for factor_id in definition.factors)


def factor_set_values() -> tuple[str, ...]:
    return tuple(factor_set_id.value for factor_set_id in FactorSetId)


def parse_factor_set(value: FactorSetId | str) -> FactorSetId:
    if isinstance(value, FactorSetId):
        return value
    normalized = value.strip()
    try:
        return FactorSetId(normalized)
    except ValueError as exc:
        supported = ", ".join(factor_set_values())
        raise ValueError(f"unknown factor set '{value}'. Supported values: {supported}") from exc


def _validate_registry(
    *,
    factor_definitions: Mapping[FactorId, FactorDefinition] = FACTOR_DEFINITIONS,
    factor_set_definitions: Mapping[FactorSetId, FactorSetDefinition] = FACTOR_SET_DEFINITIONS,
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
                raise ValueError(f"factor set '{factor_set_id.value}' references undefined factor '{factor_id.value}'")
        neutralization_ids = definition.neutralize_large_benchmark_weight_factors
        if len(neutralization_ids) != len(set(neutralization_ids)):
            raise ValueError(f"duplicate neutralization factor ids in factor set '{factor_set_id.value}'")
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
]
