from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data import DataLoader, LoadRequest, MarketData, ParquetStore

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
    rank_transform_factors: tuple[str, ...] = ("ln_market_cap",)
    large_bm_neutral_factor_names: tuple[str, ...] = ("ln_market_cap",)
    large_bm_neutral_weight_threshold: float = 0.10
    risk_window: int = 36
    tracking_error: float = 0.007 / (12**0.5)
    risk_model: str = "factor_idio"
    factor_set: str = "mfbt"
    expected_alpha_policy: str = "mean"
    monthly_snapshot_forward_days: int = 0
    value_raw_winsor_quantile: float | None = None
    value_zscore_cap: float | None = None


def required_datasets(config: MfbtEmp008Config) -> tuple[DatasetId, ...]:
    ordered = [
        DatasetId.QW_ADJ_C,
        DatasetId.QW_C,
        config.bm_weights_dataset,
        DatasetId.QW_OP_FWD_12M,
        DatasetId.QW_DPS_TTM,
        DatasetId.QW_RETAIL,
        config.sector_dataset,
        config.sector_neutral_dataset or config.sector_dataset,
        DatasetId.QW_MKTCAP,
        config.float_market_cap_dataset,
        DatasetId.QW_FCF,
        DatasetId.QW_INT_BEARING_LIAB_NFQ0,
        DatasetId.QW_QUICK_ASSETS_NFQ0,
        config.universe_dataset,
    ]
    if config.factor_set == "origin":
        ordered.append(DatasetId.QW_DIVIDEND_YLD_FY0)
    deduped: list[DatasetId] = []
    for dataset in ordered:
        if dataset not in deduped:
            deduped.append(dataset)
    return tuple(deduped)


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


def load_mfbt_emp008_bm_weights(
    *,
    parquet_dir: Path,
    start: str,
    end: str,
    config: MfbtEmp008Config,
) -> pd.DataFrame:
    loader = DataLoader(DataCatalog.default(), ParquetStore(parquet_dir))
    market = loader.load(LoadRequest(datasets=[config.bm_weights_dataset], start=start, end=end))
    return market.frames["bm_weights"]


def padded_history_start(start: str, config: MfbtEmp008Config) -> str:
    buffer_days = config.retail_flow_lookback_days * 2 + config.risk_window * 31
    return (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def padded_snapshot_end(end: str, config: MfbtEmp008Config) -> str:
    return (pd.Timestamp(end) + pd.Timedelta(days=max(config.monthly_snapshot_forward_days, 0))).strftime("%Y-%m-%d")


def _trim_non_forward_snapshot_frames(market: MarketData, *, end: str, config: MfbtEmp008Config) -> MarketData:
    if config.monthly_snapshot_forward_days <= 0:
        return market

    requested_end = pd.Timestamp(end)
    frames = {
        key: frame if key in FORWARD_SNAPSHOT_FRAME_KEYS else frame.loc[:requested_end]
        for key, frame in market.frames.items()
    }
    return MarketData(frames=frames, universe=market.universe, benchmark=market.benchmark)
