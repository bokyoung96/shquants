import uuid
from pathlib import Path

import pandas as pd
import pytest

import backtesting.specs.hooks as hooks_module
import backtesting.specs.resolve as resolve_module

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.specs import (
    ConditionSpec,
    DataPolicySpec,
    ExecutionSpec,
    HookRegistration,
    ScheduleSpec,
    SelectionSpec,
    WeightSourceSpec,
    WeightingSpec,
    register_hook,
    resolve_execution_spec,
)
from backtesting.universe import UniverseRegistry


def test_resolve_spec_records_market_cap_fallback_when_float_cap_dataset_missing(tmp_path: Path) -> None:
    market_path = tmp_path / f"{DataCatalog.default().get(DatasetId.QW_MKTCAP).stem}.parquet"
    pd.DataFrame({"A": [100.0]}, index=pd.to_datetime(["2024-01-02"])).to_parquet(market_path)

    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        strategy="trend_rank",
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
        strategy="trend_rank",
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
        strategy="trend_rank",
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
        strategy="trend_rank",
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


def test_resolve_spec_adds_feature_datasets_for_composable_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_strategy(*args: object, **kwargs: object) -> object:
        raise AssertionError("legacy strategy resolution should be skipped for composable specs")

    def _unexpected_hook(*args: object, **kwargs: object) -> object:
        raise AssertionError("legacy hook resolution should be skipped for composable specs")

    monkeypatch.setattr(resolve_module, "build_strategy", _unexpected_strategy)
    monkeypatch.setattr(resolve_module, "get_hook", _unexpected_hook)

    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        schedule=ScheduleSpec(kind="named", name="monthly"),
        selection=SelectionSpec(
            kind="filter",
            conditions=(
                ConditionSpec(field="momentum_60d", op=">", value=0.0),
                ConditionSpec(field="market_cap", op=">", value=0.0),
            ),
        ),
        weighting=WeightingSpec(kind="float_market_cap"),
    )

    resolved = resolve_execution_spec(
        spec,
        catalog=DataCatalog.default(),
        parquet_dir=tmp_path,
        raw_dir=None,
        universe_spec=None,
    )

    assert resolved.dataset_ids == (
        DatasetId.QW_ADJ_C,
        DatasetId.QW_ADJ_O,
        DatasetId.QW_MKTCAP,
        DatasetId.QW_MKTCAP_FLT,
    )
    assert len(resolved.dataset_ids) == 4


def test_resolve_spec_increases_warmup_for_feature_lookbacks(tmp_path: Path) -> None:
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        name="momentum filter",
        capital=123_456_789.0,
        schedule=ScheduleSpec(kind="named", name="monthly"),
        warmup_days=5,
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="momentum_60d", op=">", value=0.0),),
        ),
    )

    resolved = resolve_execution_spec(
        spec,
        catalog=DataCatalog.default(),
        parquet_dir=tmp_path,
        raw_dir=None,
        universe_spec=None,
    )

    assert spec.warmup_days == 5
    assert resolved.execution.warmup_days == 60
    assert resolved.execution.start == spec.start
    assert resolved.execution.end == spec.end
    assert resolved.execution.name == spec.name
    assert resolved.execution.capital == spec.capital
    assert resolved.execution.selection == spec.selection
    assert resolved.execution.schedule == spec.schedule
    assert "warmup_days increased to 60 for feature/weighting lookbacks" in resolved.resolution_notes


def test_resolve_spec_merges_hook_datasets_for_composable_plan(tmp_path: Path) -> None:
    hook_id = f"resolve_test_hook_{uuid.uuid4().hex}"
    register_hook(
        HookRegistration(
            hook_id=hook_id,
            required_datasets=(DatasetId.QW_MKTCAP,),
            build_plan=lambda **_: None,
        )
    )
    try:
        spec = ExecutionSpec(
            start="2024-01-01",
            end="2024-12-31",
            schedule=ScheduleSpec(kind="named", name="monthly"),
            weight_source=WeightSourceSpec(kind="hook", hook_id=hook_id),
            selection=SelectionSpec(
                kind="filter",
                conditions=(ConditionSpec(field="momentum_60d", op=">", value=0.0),),
            ),
            weighting=WeightingSpec(kind="float_market_cap"),
        )

        resolved = resolve_execution_spec(
            spec,
            catalog=DataCatalog.default(),
            parquet_dir=tmp_path,
            raw_dir=None,
            universe_spec=None,
        )
    finally:
        hooks_module._HOOKS.pop(hook_id, None)

    assert DatasetId.QW_MKTCAP in resolved.dataset_ids
    assert DatasetId.QW_MKTCAP_FLT in resolved.dataset_ids


def test_resolve_spec_increases_warmup_for_inverse_vol_weighting(tmp_path: Path) -> None:
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        schedule=ScheduleSpec(kind="named", name="monthly"),
        warmup_days=5,
        weighting=WeightingSpec(kind="inverse_vol"),
    )

    resolved = resolve_execution_spec(
        spec,
        catalog=DataCatalog.default(),
        parquet_dir=tmp_path,
        raw_dir=None,
        universe_spec=None,
    )

    assert spec.warmup_days == 5
    assert resolved.execution.warmup_days == 20
    assert "warmup_days increased to 20 for feature/weighting lookbacks" in resolved.resolution_notes


def test_resolve_spec_preserves_higher_existing_warmup_for_inverse_vol_weighting(tmp_path: Path) -> None:
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        schedule=ScheduleSpec(kind="named", name="monthly"),
        warmup_days=30,
        weighting=WeightingSpec(kind="inverse_vol"),
    )

    resolved = resolve_execution_spec(
        spec,
        catalog=DataCatalog.default(),
        parquet_dir=tmp_path,
        raw_dir=None,
        universe_spec=None,
    )

    assert resolved.execution.warmup_days == 30
    assert resolved.resolution_notes == ()


def test_resolve_spec_rejects_unknown_feature_fields(tmp_path: Path) -> None:
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        schedule=ScheduleSpec(kind="named", name="monthly"),
        selection=SelectionSpec(
            kind="filter",
            conditions=(ConditionSpec(field="unknown_feature", op=">", value=0.0),),
        ),
    )

    with pytest.raises(KeyError, match="unknown feature field"):
        resolve_execution_spec(
            spec,
            catalog=DataCatalog.default(),
            parquet_dir=tmp_path,
            raw_dir=None,
            universe_spec=None,
        )


def test_resolve_spec_maps_feature_datasets_through_universe(tmp_path: Path) -> None:
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        schedule=ScheduleSpec(kind="named", name="monthly"),
        selection=SelectionSpec(
            kind="filter",
            conditions=(
                ConditionSpec(field="momentum_60d", op=">", value=0.0),
                ConditionSpec(field="market_cap", op=">", value=0.0),
            ),
        ),
        weighting=WeightingSpec(kind="float_market_cap"),
    )

    resolved = resolve_execution_spec(
        spec,
        catalog=DataCatalog.default(),
        parquet_dir=tmp_path,
        raw_dir=None,
        universe_spec=UniverseRegistry.default().get("kosdaq150"),
    )

    assert resolved.dataset_ids == (
        DatasetId.QW_KSDQ_ADJ_C,
        DatasetId.QW_KSDQ_ADJ_O,
        DatasetId.QW_KSDQ150_YN,
        DatasetId.QW_KSDQ_MKTCAP,
        DatasetId.QW_KSDQ_MKTCAP_FLT,
    )
