from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from root import ROOT

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data import DataLoader, LoadRequest, ParquetStore


_REQUIRED_DATASETS = (
    DatasetId.QW_ADJ_C,
    DatasetId.QW_BM,
    DatasetId.QW_K200_YN,
    DatasetId.QW_WICS_SEC_BIG,
    DatasetId.QW_MKTCAP,
)


@dataclass(frozen=True, slots=True)
class SectorIndexResult:
    sector_returns: pd.DataFrame
    sector_index: pd.DataFrame
    confidence: pd.DataFrame
    sector_weight_sum: pd.DataFrame


@dataclass(frozen=True, slots=True)
class RrgInputData:
    sector_prices: pd.DataFrame
    benchmark: pd.Series
    confidence: pd.DataFrame
    sector_returns: pd.DataFrame
    sector_weight_sum: pd.DataFrame


def required_kospi200_wics_datasets() -> tuple[str, ...]:
    return tuple(dataset.value for dataset in _REQUIRED_DATASETS)


def load_kospi200_wics_sector_rrg_input(
    *,
    start: str,
    end: str,
    parquet_dir: Path | None = None,
    catalog: DataCatalog | None = None,
    benchmark_code: str = "IKS200",
) -> RrgInputData:
    data_catalog = catalog or DataCatalog.default()
    store = ParquetStore(parquet_dir or ROOT.parquet_path)
    loader = DataLoader(data_catalog, store)
    market = loader.load(LoadRequest(datasets=list(_REQUIRED_DATASETS), start=start, end=end))

    close = market.frames["close"].astype(float)
    membership = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
    sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
    market_cap = market.frames["market_cap"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
    benchmark_frame = market.frames["benchmark"]
    benchmark = benchmark_frame[benchmark_code] if benchmark_code in benchmark_frame.columns else benchmark_frame.iloc[:, 0]
    benchmark = benchmark.reindex(close.index).ffill().astype(float)

    sector_result = build_sector_return_index(
        close=close,
        membership=membership,
        sector=sector,
        market_cap=market_cap,
    )
    return RrgInputData(
        sector_prices=sector_result.sector_index,
        benchmark=benchmark,
        confidence=sector_result.confidence,
        sector_returns=sector_result.sector_returns,
        sector_weight_sum=sector_result.sector_weight_sum,
    )


def build_sector_return_index(
    *,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    sector: pd.DataFrame,
    market_cap: pd.DataFrame,
) -> SectorIndexResult:
    prices = close.astype(float).sort_index()
    returns = prices.pct_change(fill_method=None)
    aligned_membership = membership.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
    aligned_sector = sector.reindex(index=prices.index, columns=prices.columns)
    aligned_market_cap = market_cap.reindex(index=prices.index, columns=prices.columns).fillna(0.0).astype(float)

    return_rows: dict[pd.Timestamp, dict[object, float]] = {}
    count_rows: dict[pd.Timestamp, dict[object, float]] = {}
    weight_rows: dict[pd.Timestamp, dict[object, float]] = {}
    for ts in prices.index:
        valid = aligned_membership.loc[ts] & returns.loc[ts].notna() & aligned_sector.loc[ts].notna()
        return_row: dict[object, float] = {}
        count_row: dict[object, float] = {}
        weight_row: dict[object, float] = {}
        for sector_name in pd.unique(aligned_sector.loc[ts, valid]):
            names = returns.columns[valid & aligned_sector.loc[ts].eq(sector_name)]
            weights = aligned_market_cap.loc[ts, names].clip(lower=0.0)
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                weights = pd.Series(1.0, index=names, dtype=float)
                weight_sum = float(weights.sum())
            normalized = weights / weight_sum
            return_row[sector_name] = float((returns.loc[ts, names] * normalized).sum())
            count_row[sector_name] = float(len(names))
            weight_row[sector_name] = weight_sum
        return_rows[ts] = return_row
        count_rows[ts] = count_row
        weight_rows[ts] = weight_row

    sector_returns = pd.DataFrame.from_dict(return_rows, orient="index").reindex(index=prices.index)
    sector_index = (1.0 + sector_returns.fillna(0.0)).cumprod()
    confidence = pd.DataFrame.from_dict(count_rows, orient="index").reindex(index=prices.index).fillna(0.0)
    sector_weight_sum = pd.DataFrame.from_dict(weight_rows, orient="index").reindex(index=prices.index).fillna(0.0)
    return SectorIndexResult(
        sector_returns=sector_returns,
        sector_index=sector_index,
        confidence=confidence,
        sector_weight_sum=sector_weight_sum,
    )
