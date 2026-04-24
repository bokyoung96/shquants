from pathlib import Path

import pandas as pd
import pytest

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.specs import ExecutionSpec, ScheduleSpec, WeightSourceSpec, DataPolicySpec, resolve_execution_spec
from backtesting.universe import UniverseRegistry


def test_resolve_spec_records_market_cap_fallback_when_float_cap_dataset_missing(tmp_path: Path) -> None:
    market_path = tmp_path / f"{DataCatalog.default().get(DatasetId.QW_MKTCAP).stem}.parquet"
    pd.DataFrame({"A": [100.0]}, index=pd.to_datetime(["2024-01-02"])).to_parquet(market_path)

    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        strategy="momentum",
        schedule=ScheduleSpec(kind="named", name="monthly"),
        weight_source=WeightSourceSpec(kind="hook", hook_id="kospi200_semiannual_floatcap"),
        data_policy=DataPolicySpec(requested_weight_basis="float_market_cap", fallback_order=("market_cap",)),
    )

    resolved = resolve_execution_spec(
        spec,
        catalog=DataCatalog.default(),
        parquet_dir=tmp_path,
        raw_dir=None,
        universe_spec=UniverseRegistry.default().get("legacy_k200"),
    )

    assert resolved.execution.data_policy.requested_weight_basis == "float_market_cap"
    assert resolved.execution.data_policy.resolved_weight_basis == "market_cap"
    assert resolved.execution.data_policy.fallbacks_applied == (
        {"from": "float_market_cap", "to": "market_cap", "reason": "missing qw_mktcap_flt source"},
    )
    assert DatasetId.QW_MKTCAP in resolved.dataset_ids
    assert DatasetId.QW_MKTCAP_FLT not in resolved.dataset_ids


def test_resolve_spec_rejects_unknown_hook_id(tmp_path: Path) -> None:
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        strategy="momentum",
        schedule=ScheduleSpec(kind="named", name="monthly"),
        weight_source=WeightSourceSpec(kind="hook", hook_id="does-not-exist"),
    )

    with pytest.raises(KeyError, match="unknown hook_id"):
        resolve_execution_spec(
            spec,
            catalog=DataCatalog.default(),
            parquet_dir=tmp_path,
            raw_dir=None,
            universe_spec=UniverseRegistry.default().get("legacy_k200"),
        )


def test_resolve_spec_uses_float_market_cap_when_parquet_exists(tmp_path: Path) -> None:
    catalog = DataCatalog.default()
    float_path = tmp_path / f"{catalog.get(DatasetId.QW_MKTCAP_FLT).stem}.parquet"
    float_path.write_text("placeholder", encoding="utf-8")

    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        strategy="momentum",
        schedule=ScheduleSpec(kind="named", name="monthly"),
        weight_source=WeightSourceSpec(kind="hook", hook_id="kospi200_semiannual_floatcap"),
        data_policy=DataPolicySpec(requested_weight_basis="float_market_cap", fallback_order=("market_cap",)),
    )

    resolved = resolve_execution_spec(
        spec,
        catalog=catalog,
        parquet_dir=tmp_path,
        raw_dir=None,
        universe_spec=UniverseRegistry.default().get("legacy_k200"),
    )

    assert resolved.execution.data_policy.resolved_weight_basis == "float_market_cap"
    assert resolved.execution.data_policy.fallbacks_applied == ()
    assert DatasetId.QW_MKTCAP_FLT in resolved.dataset_ids


def test_resolve_spec_prefers_float_market_cap_when_raw_source_exists(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "qw_mktcap_flt.csv").write_text("date,A\n2024-01-02,100\n", encoding="utf-8")

    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        strategy="momentum",
        schedule=ScheduleSpec(kind="named", name="monthly"),
        weight_source=WeightSourceSpec(kind="hook", hook_id="kospi200_semiannual_floatcap"),
        data_policy=DataPolicySpec(requested_weight_basis="float_market_cap", fallback_order=("market_cap",)),
    )

    resolved = resolve_execution_spec(
        spec,
        catalog=DataCatalog.default(),
        parquet_dir=tmp_path,
        raw_dir=raw_dir,
        universe_spec=UniverseRegistry.default().get("legacy_k200"),
    )

    assert resolved.execution.data_policy.resolved_weight_basis == "float_market_cap"
    assert resolved.execution.data_policy.fallbacks_applied == ()
    assert DatasetId.QW_MKTCAP_FLT in resolved.dataset_ids
    assert DatasetId.QW_MKTCAP not in resolved.dataset_ids
