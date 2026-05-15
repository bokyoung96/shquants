import pandas as pd
import pytest

import backtesting
from backtesting.catalog import DatasetId
from backtesting.universe import UniverseRegistry
from root import ROOT


def test_registry_returns_kosdaq150_defaults() -> None:
    registry = UniverseRegistry.default()
    spec = registry.get("kosdaq150")

    assert spec.id == "kosdaq150"
    assert spec.membership_dataset is DatasetId.QW_KSDQ150_YN
    assert spec.default_benchmark_dataset == "qw_BM"
    assert spec.default_benchmark_code == "IKQ150"
    assert spec.default_benchmark_name == "KOSDAQ150"
    assert spec.dataset_aliases["close"] is DatasetId.QW_KSDQ_ADJ_C
    assert spec.dataset_aliases["market_cap"] is DatasetId.QW_KSDQ_MKTCAP


def test_registry_returns_etf_defaults() -> None:
    registry = UniverseRegistry.default()
    spec = registry.get("etf")

    assert spec.id == "etf"
    assert spec.membership_dataset is None
    assert spec.default_benchmark_dataset == "qw_BM"
    assert spec.default_benchmark_code == "IKS200"
    assert spec.default_benchmark_name == "KOSPI200"
    assert spec.dataset_aliases["close"] is DatasetId.QW_ETF_ADJ_C
    assert spec.dataset_aliases["open"] is DatasetId.QW_ETF_ADJ_O
    assert spec.dataset_aliases["volume"] is DatasetId.QW_ETF_ADJ_V


def test_registry_returns_legacy_k200_defaults() -> None:
    registry = UniverseRegistry.default()
    spec = registry.get("legacy_k200")

    assert spec.id == "legacy_k200"
    assert spec.membership_dataset is DatasetId.QW_K200_YN
    assert spec.default_benchmark_dataset == "qw_BM"
    assert spec.default_benchmark_code == "IKS200"
    assert spec.default_benchmark_name == "KOSPI200"


def test_registry_default_benchmark_codes_exist_in_qw_bm_parquet() -> None:
    registry = UniverseRegistry.default()
    columns = set(pd.read_parquet(ROOT.parquet_path / "qw_BM.parquet").columns.astype(str))

    for universe_id in ("legacy_k200", "kosdaq150"):
        spec = registry.get(universe_id)
        assert spec.default_benchmark_code in columns


def test_registry_remaps_generic_dataset_ids_to_universe_specific_ids() -> None:
    registry = UniverseRegistry.default()
    spec = registry.get("kosdaq150")

    assert spec.resolve_dataset(DatasetId.QW_ADJ_C) is DatasetId.QW_KSDQ_ADJ_C
    assert spec.resolve_dataset(DatasetId.QW_MKTCAP) is DatasetId.QW_KSDQ_MKTCAP
    assert spec.resolve_dataset(DatasetId.QW_OP_NFY1) is DatasetId.QW_OP_NFY1


def test_registry_remaps_generic_dataset_ids_to_etf_specific_ids() -> None:
    registry = UniverseRegistry.default()
    spec = registry.get("etf")

    assert spec.resolve_dataset(DatasetId.QW_ADJ_C) is DatasetId.QW_ETF_ADJ_C
    assert spec.resolve_dataset(DatasetId.QW_ADJ_O) is DatasetId.QW_ETF_ADJ_O
    assert spec.resolve_dataset(DatasetId.QW_V) is DatasetId.QW_ETF_ADJ_V


def test_registry_rejects_unknown_universe() -> None:
    registry = UniverseRegistry.default()

    with pytest.raises(KeyError, match="unknown universe"):
        registry.get("not-real")


def test_top_level_package_exports_universe_types() -> None:
    assert backtesting.UniverseRegistry is UniverseRegistry
    assert backtesting.UniverseSpec.__name__ == "UniverseSpec"
