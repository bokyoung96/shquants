from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data.catalog import DatasetId
from data.loader import MarketData, load_market_data
from .factor_registry import FactorSetId, factor_definitions_for_set, get_factor_set_definition, parse_factor_set
from .factor_timing import FactorTimingConfig


@dataclass(frozen=True, slots=True)
class Emp008Config:
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
    factor_set: object = "mfbt"
    expected_alpha_estimator: str = "mean"
    factor_timing: FactorTimingConfig | None = None
    value_raw_winsor_quantile: float | None = None
    value_zscore_cap: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_set", parse_factor_set(self.factor_set))
        if self.expected_alpha_estimator not in {"mean", "ewma36", "mean_1se"}:
            raise ValueError("expected_alpha_estimator must be 'mean', 'ewma36', or 'mean_1se'")

    @property
    def rank_transform_factors(self) -> tuple[str, ...]:
        definition = get_factor_set_definition(self.factor_set)
        return tuple(factor_id.value for factor_id in definition.rank_transform_factors)

    @property
    def large_bm_neutral_factor_names(self) -> tuple[str, ...]:
        definition = get_factor_set_definition(self.factor_set)
        return tuple(factor_id.value for factor_id in definition.neutralize_large_benchmark_weight_factors)

    @property
    def expected_alpha_policy(self) -> str:
        definition = get_factor_set_definition(self.factor_set)
        if not definition.constrain_expected_alpha_to_direction:
            return "mean"
        return "origin_small_cap" if self.factor_set is FactorSetId.MFBT_ORIGIN_SMALLCAP else "origin_sign"

    @property
    def monthly_snapshot_forward_days(self) -> int:
        return get_factor_set_definition(self.factor_set).snapshot_forward_days


def required_datasets(config: Emp008Config) -> tuple[DatasetId, ...]:
    factor_datasets = [dataset for definition in factor_definitions_for_set(config.factor_set) for dataset in definition.datasets]
    ordered = [DatasetId.QW_ADJ_C, config.bm_weights_dataset, *factor_datasets]
    if any(definition.requires_construction_sector for definition in factor_definitions_for_set(config.factor_set)):
        ordered.append(config.sector_dataset)
    ordered.append(config.sector_neutral_dataset or config.sector_dataset)
    ordered.extend([DatasetId.QW_MKTCAP, config.float_market_cap_dataset, config.universe_dataset])
    return tuple(dict.fromkeys(ordered))


def load_emp008_market(*, parquet_dir: Path, start: str, end: str, config: Emp008Config) -> MarketData:
    return load_market_data(parquet_dir, start=start, end=end, config=config)


def padded_history_start(start: str, config: Emp008Config) -> str:
    buffer_days = config.retail_flow_lookback_days * 2 + config.risk_window * 31
    import pandas as pd
    return (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def padded_snapshot_end(end: str, config: Emp008Config) -> str:
    import pandas as pd
    return (pd.Timestamp(end) + pd.Timedelta(days=max(config.monthly_snapshot_forward_days, 0))).strftime("%Y-%m-%d")
