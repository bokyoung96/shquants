from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data import DataLoader, LoadRequest, MarketData, ParquetStore
from backtesting.strategies.emp008.mfbt_emp008_factor_registry import (
    FactorSetId,
    factor_definitions_for_set,
    get_factor_set_definition,
    parse_factor_set,
)

FORWARD_SNAPSHOT_FRAME_KEYS = frozenset({"dividend_yld_fy0"})


@dataclass(frozen=True, slots=True)
class MfbtEmp008Config:
    sector_dataset: DatasetId = DatasetId.QW_WI_SEC_26_BIG
    sector_neutral_dataset: DatasetId | None = None
    bm_weights_dataset: DatasetId = DatasetId.QW_BM_WEIGHTS
    universe_dataset: DatasetId = DatasetId.QW_K200_YN
    float_market_cap_dataset: DatasetId = DatasetId.QW_MKTCAP_FLT
    retail_flow_lookback_days: int = 252
    positivity_momentum_lookback_days: int = 252
    low_op_threshold: float = 100_000_000_000.0
    extreme_growth_threshold: float = 0.50
    large_bm_neutral_weight_threshold: float = 0.10
    risk_window: int = 36
    tracking_error: float = 0.007 / (12**0.5)
    risk_model: str = "factor_idio"
    factor_set: FactorSetId = FactorSetId.MFBT
    value_raw_winsor_quantile: float | None = None
    value_zscore_cap: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_set", parse_factor_set(self.factor_set))

    @property
    def rank_transform_factors(self) -> tuple[str, ...]:
        return tuple(
            definition.id.value
            for definition in factor_definitions_for_set(self.factor_set)
            if definition.rank_transform
        )

    @property
    def large_bm_neutral_factor_names(self) -> tuple[str, ...]:
        return tuple(
            definition.id.value
            for definition in factor_definitions_for_set(self.factor_set)
            if definition.neutralize_large_benchmark_weight
        )

    @property
    def expected_alpha_policy(self) -> str:
        factor_set_definition = get_factor_set_definition(self.factor_set)
        if not factor_set_definition.constrain_expected_alpha_to_direction:
            return "mean"
        if self.factor_set is FactorSetId.MFBT_ORIGIN_SMALLCAP:
            return "origin_small_cap"
        return "origin_sign"

    @property
    def monthly_snapshot_forward_days(self) -> int:
        return get_factor_set_definition(self.factor_set).snapshot_forward_days


def required_datasets(config: MfbtEmp008Config) -> tuple[DatasetId, ...]:
    factor_definitions = factor_definitions_for_set(config.factor_set)
    factor_datasets = [dataset_id for definition in factor_definitions for dataset_id in definition.datasets]
    ordered = [
        DatasetId.QW_ADJ_C,
        config.bm_weights_dataset,
        *factor_datasets,
    ]
    if any(definition.requires_construction_sector for definition in factor_definitions):
        ordered.append(config.sector_dataset)
    ordered.append(config.sector_neutral_dataset or config.sector_dataset)
    ordered.extend(
        [
            DatasetId.QW_MKTCAP,
            config.float_market_cap_dataset,
            config.universe_dataset,
        ]
    )
    return tuple(dict.fromkeys(ordered))


def load_mfbt_emp008_market(
    *,
    parquet_dir: Path,
    start: str,
    end: str,
    config: MfbtEmp008Config,
) -> MarketData:
    loader = DataLoader(DataCatalog.default(), ParquetStore(parquet_dir))
    load_start = padded_history_start(start, config)
    load_end = padded_snapshot_end(end, config)
    neutral_dataset = config.sector_neutral_dataset or config.sector_dataset
    datasets = list(required_datasets(config))
    if neutral_dataset != config.sector_dataset:
        base_datasets = [dataset for dataset in datasets if dataset != neutral_dataset]
        market = loader.load(LoadRequest(datasets=base_datasets, start=load_start, end=load_end))
        neutral_market = loader.load(LoadRequest(datasets=[neutral_dataset], start=load_start, end=load_end))
        market = MarketData(
            frames={**market.frames, "sector_neutral_big": neutral_market.frames["sector_big"]},
            universe=market.universe,
            benchmark=market.benchmark,
        )
    else:
        market = loader.load(LoadRequest(datasets=datasets, start=load_start, end=load_end))
        market = MarketData(
            frames={**market.frames, "sector_neutral_big": market.frames["sector_big"]},
            universe=market.universe,
            benchmark=market.benchmark,
        )
    return _trim_non_forward_snapshot_frames(market, end=end, config=config)


def padded_history_start(start: str, config: MfbtEmp008Config) -> str:
    buffer_days = config.retail_flow_lookback_days * 2 + config.risk_window * 31
    return (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def padded_snapshot_end(end: str, config: MfbtEmp008Config) -> str:
    forward_days = max(get_factor_set_definition(config.factor_set).snapshot_forward_days, 0)
    return (pd.Timestamp(end) + pd.Timedelta(days=forward_days)).strftime("%Y-%m-%d")


def _trim_non_forward_snapshot_frames(market: MarketData, *, end: str, config: MfbtEmp008Config) -> MarketData:
    if get_factor_set_definition(config.factor_set).snapshot_forward_days <= 0:
        return market

    requested_end = pd.Timestamp(end)
    frames = {
        key: frame if key in FORWARD_SNAPSHOT_FRAME_KEYS else frame.loc[:requested_end]
        for key, frame in market.frames.items()
    }
    return MarketData(frames=frames, universe=market.universe, benchmark=market.benchmark)
