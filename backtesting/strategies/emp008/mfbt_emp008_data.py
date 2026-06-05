from __future__ import annotations

from dataclasses import dataclass

from backtesting.catalog import DatasetId


@dataclass(frozen=True, slots=True)
class MfbtEmp008Config:
    sector_dataset: DatasetId = DatasetId.QW_WI_SEC_26_BIG
    bm_weights_dataset: DatasetId = DatasetId.QW_BM_WEIGHTS
    universe_dataset: DatasetId = DatasetId.QW_K200_YN
    float_market_cap_dataset: DatasetId = DatasetId.QW_MKTCAP_FLT
    retail_flow_lookback_days: int = 252
    low_op_threshold: float = 100_000_000_000.0
    extreme_growth_threshold: float = 0.50
    risk_window: int = 36
    tracking_error: float = 0.5 / (12**0.5)


def required_datasets(config: MfbtEmp008Config) -> tuple[DatasetId, ...]:
    ordered = [
        DatasetId.QW_ADJ_C,
        DatasetId.QW_C,
        config.bm_weights_dataset,
        DatasetId.QW_OP_FWD_12M,
        DatasetId.QW_DPS_TTM,
        DatasetId.QW_RETAIL,
        config.sector_dataset,
        DatasetId.QW_MKTCAP,
        config.float_market_cap_dataset,
        DatasetId.QW_FCF,
        DatasetId.QW_INT_BEARING_LIAB_NFQ0,
        DatasetId.QW_QUICK_ASSETS_NFQ0,
        config.universe_dataset,
    ]
    deduped: list[DatasetId] = []
    for dataset in ordered:
        if dataset not in deduped:
            deduped.append(dataset)
    return tuple(deduped)
