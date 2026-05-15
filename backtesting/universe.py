from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from backtesting.catalog import DatasetId


_DATASET_ALIAS_BY_ID: dict[DatasetId, str] = {
    DatasetId.QW_ADJ_C: "close",
    DatasetId.QW_ADJ_O: "open",
    DatasetId.QW_ADJ_H: "high",
    DatasetId.QW_ADJ_L: "low",
    DatasetId.QW_V: "volume",
    DatasetId.QW_MKTCAP: "market_cap",
    DatasetId.QW_MKTCAP_FLT: "float_market_cap",
    DatasetId.QW_WICS_SEC_BIG: "sector_big",
    DatasetId.QW_K200_YN: "membership",
}


@dataclass(frozen=True, slots=True)
class UniverseSpec:
    id: str
    display_name: str
    description: str
    membership_dataset: DatasetId | None
    default_benchmark_code: str
    default_benchmark_name: str
    default_benchmark_dataset: str
    dataset_aliases: Mapping[str, DatasetId]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_aliases", MappingProxyType(dict(self.dataset_aliases)))

    def resolve_dataset(self, dataset_id: DatasetId) -> DatasetId:
        alias = _DATASET_ALIAS_BY_ID.get(dataset_id)
        if alias is None:
            return dataset_id
        return self.dataset_aliases.get(alias, dataset_id)


@dataclass(frozen=True, slots=True)
class UniverseRegistry:
    specs: Mapping[str, UniverseSpec]

    @classmethod
    def default(cls) -> "UniverseRegistry":
        return cls(
            specs={
                "legacy_k200": UniverseSpec(
                    id="legacy_k200",
                    display_name="Legacy KOSPI200",
                    description="Existing K200 membership path for backward compatibility.",
                    membership_dataset=DatasetId.QW_K200_YN,
                    default_benchmark_code="IKS200",
                    default_benchmark_name="KOSPI200",
                    default_benchmark_dataset="qw_BM",
                    dataset_aliases={},
                ),
                "kosdaq150": UniverseSpec(
                    id="kosdaq150",
                    display_name="KOSDAQ150",
                    description="KOSDAQ150 price and membership dataset family.",
                    membership_dataset=DatasetId.QW_KSDQ150_YN,
                    default_benchmark_code="IKQ150",
                    default_benchmark_name="KOSDAQ150",
                    default_benchmark_dataset="qw_BM",
                    dataset_aliases={
                        "close": DatasetId.QW_KSDQ_ADJ_C,
                        "open": DatasetId.QW_KSDQ_ADJ_O,
                        "high": DatasetId.QW_KSDQ_ADJ_H,
                        "low": DatasetId.QW_KSDQ_ADJ_L,
                        "volume": DatasetId.QW_KSDQ_V,
                        "market_cap": DatasetId.QW_KSDQ_MKTCAP,
                        "float_market_cap": DatasetId.QW_KSDQ_MKTCAP_FLT,
                        "sector_big": DatasetId.QW_KSDQ_WICS_SEC_BIG,
                        "membership": DatasetId.QW_KSDQ150_YN,
                    },
                ),
                "etf": UniverseSpec(
                    id="etf",
                    display_name="ETF",
                    description="ETF-only price dataset family.",
                    membership_dataset=None,
                    default_benchmark_code="IKS200",
                    default_benchmark_name="KOSPI200",
                    default_benchmark_dataset="qw_BM",
                    dataset_aliases={
                        "close": DatasetId.QW_ETF_ADJ_C,
                        "open": DatasetId.QW_ETF_ADJ_O,
                        "high": DatasetId.QW_ETF_ADJ_H,
                        "low": DatasetId.QW_ETF_ADJ_L,
                        "volume": DatasetId.QW_ETF_ADJ_V,
                    },
                ),
            }
        )

    def get(self, universe_id: str) -> UniverseSpec:
        try:
            return self.specs[universe_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.specs))
            raise KeyError(f"unknown universe '{universe_id}'. Available: {available}") from exc
