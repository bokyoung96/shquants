from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    field: str
    dataset_ids: tuple[DatasetId, ...]
    warmup_days: int
    build: Callable[[MarketData], pd.DataFrame]


_REGISTRY: dict[str, FeatureDefinition] = {}


def register_feature(definition: FeatureDefinition) -> None:
    if definition.field in _REGISTRY:
        raise ValueError(f"feature already registered: {definition.field}")
    _REGISTRY[definition.field] = definition


def get_feature(field: str) -> FeatureDefinition:
    try:
        return _REGISTRY[field]
    except KeyError as exc:
        raise KeyError(f"unknown feature field: {field}") from exc


def feature_dataset_ids(fields: list[str] | tuple[str, ...]) -> tuple[DatasetId, ...]:
    ordered: list[DatasetId] = []
    seen: set[DatasetId] = set()
    for field in fields:
        for dataset_id in get_feature(field).dataset_ids:
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            ordered.append(dataset_id)
    return tuple(ordered)


def feature_warmup_days(fields: list[str] | tuple[str, ...]) -> int:
    warmups = (get_feature(field).warmup_days for field in fields)
    return max(warmups, default=0)


def build_features(market: MarketData, fields: list[str] | tuple[str, ...]) -> dict[str, pd.DataFrame]:
    features: dict[str, pd.DataFrame] = {}
    for field in fields:
        if field in features:
            continue
        definition = get_feature(field)
        features[field] = definition.build(market)
    return features


def _frame_builder(frame_key: str) -> Callable[[MarketData], pd.DataFrame]:
    def _build(market: MarketData) -> pd.DataFrame:
        return _require_frame(market, frame_key)

    return _build


def _require_frame(market: MarketData, frame_key: str) -> pd.DataFrame:
    try:
        return market.frames[frame_key]
    except KeyError as exc:
        raise KeyError(f"missing market frame: {frame_key}") from exc


def _shortable_builder(market: MarketData) -> pd.DataFrame:
    if "shortable" in market.frames:
        return market.frames["shortable"]
    return ~_require_frame(market, "trade_ban").fillna(1).astype(bool)


register_feature(
    FeatureDefinition(
        field="close",
        dataset_ids=(DatasetId.QW_ADJ_C,),
        warmup_days=0,
        build=_frame_builder("close"),
    )
)
register_feature(
    FeatureDefinition(
        field="open",
        dataset_ids=(DatasetId.QW_ADJ_O,),
        warmup_days=0,
        build=_frame_builder("open"),
    )
)
register_feature(
    FeatureDefinition(
        field="momentum_20d",
        dataset_ids=(DatasetId.QW_ADJ_C,),
        warmup_days=20,
        build=lambda market: _require_frame(market, "close").pct_change(20, fill_method=None),
    )
)
register_feature(
    FeatureDefinition(
        field="momentum_60d",
        dataset_ids=(DatasetId.QW_ADJ_C,),
        warmup_days=60,
        build=lambda market: _require_frame(market, "close").pct_change(60, fill_method=None),
    )
)
register_feature(
    FeatureDefinition(
        field="market_cap",
        dataset_ids=(DatasetId.QW_MKTCAP,),
        warmup_days=0,
        build=_frame_builder("market_cap"),
    )
)
register_feature(
    FeatureDefinition(
        field="float_market_cap",
        dataset_ids=(DatasetId.QW_MKTCAP_FLT,),
        warmup_days=0,
        build=_frame_builder("float_market_cap"),
    )
)
register_feature(
    FeatureDefinition(
        field="avg_trading_value_20d",
        dataset_ids=(DatasetId.QW_ADJ_C, DatasetId.QW_V),
        warmup_days=20,
        build=lambda market: (_require_frame(market, "close") * _require_frame(market, "volume")).rolling(20, min_periods=20).mean(),
    )
)
register_feature(
    FeatureDefinition(
        field="foreign_ratio",
        dataset_ids=(DatasetId.QW_FOREIGN_RATIO,),
        warmup_days=0,
        build=_frame_builder("foreign_ratio"),
    )
)
register_feature(
    FeatureDefinition(
        field="institution_flow_20d",
        dataset_ids=(DatasetId.QW_INSTITUTION,),
        warmup_days=20,
        build=lambda market: _require_frame(market, "inst_flow").rolling(20, min_periods=20).sum(),
    )
)
register_feature(
    FeatureDefinition(
        field="retail_flow_20d",
        dataset_ids=(DatasetId.QW_RETAIL,),
        warmup_days=20,
        build=lambda market: _require_frame(market, "retail_flow").rolling(20, min_periods=20).sum(),
    )
)
register_feature(
    FeatureDefinition(
        field="sector",
        dataset_ids=(DatasetId.QW_WICS_SEC_BIG,),
        warmup_days=0,
        build=_frame_builder("sector_big"),
    )
)
register_feature(
    FeatureDefinition(
        field="shortable",
        dataset_ids=(DatasetId.QW_TRS_BAN,),
        warmup_days=0,
        build=_shortable_builder,
    )
)
